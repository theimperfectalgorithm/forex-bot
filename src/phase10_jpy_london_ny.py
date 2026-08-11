"""
Forex Bot - Phase 10: JPY crosses in London/NY sessions (H1)
===============================================================
Roadmap item (PROJECT_REPORT.md section 8, "Immediate" list): extend the
project's one proven edge family -- JPY-cross session structure -- out of
its current Asian-hours-only territory (AMR mean reversion 00:00-07:00
server, ARB breakout 07:00-09:00 server) into London/NY, where the
project has never tested anything OTHER than the two mechanics already
proven dead there:
  - London Open Range Breakout (LORB) on GBPJPY/EURJPY/AUDJPY/USDJPY:
    catastrophic (IS PF 0.69-0.98, DD to 39%) -- phase 3.
  - NY-overlap ATR-threshold continuation on the same 4 pairs: also dead
    (IS PF 0.78-1.04) -- phase 3.
Re-running either mechanic on the untested crosses (CADJPY, NZDJPY,
CHFJPY) would just be re-plowing the same failed geometry on siblings of
pairs where it already failed -- low information value. This phase tests
two STRUCTURALLY DIFFERENT mechanics instead, both novel to this
project:

  1. sqz  Volatility-squeeze breakout (H1): Bollinger(20,2) band-width
     compressed into its own bottom percentile (a genuine "coiling"
     signal, not a fixed time-window range like LORB) followed by a
     close breaking outside the bands, during London/NY hours. ATR-
     scaled SL/TP. No time exit -- lets a real breakout run to target,
     same philosophy as the validated ARB family.
  2. xmo  Cross-asset JPY momentum: USDJPY's own London-session move (as
     a broad JPY-strength/weakness proxy) gates a same-direction entry
     on the OTHER JPY crosses at the London/NY overlap. A relative-value
     mechanic, never tried in this project. Session-scoped by design
     (time exit at NY close) since the thesis is that day's specific
     USD/JPY move, not a multi-day hold.

Pairs: GBPJPY, EURJPY, AUDJPY, CADJPY (already live in the Asian book,
tested here on a genuinely different mechanic/session) + NZDJPY, CHFJPY
(never tested in this project at all). USDJPY is the momentum proxy for
mechanic 2 (not itself a signal pair there); included as a signal pair
for mechanic 1.

Methodology matches every prior phase: 36 months H1 fetched live from
MT5, IS = first 24 months / OOS = last 12, selection criteria on IS only
(PF > 1.3, DD < 8%, >= 60% profitable months), OOS must be positive.
Spread paid per trade. SL-priority on same-bar SL+TP touch. Friday close.

Requirements: MetaTrader 5 OPEN and LOGGED IN.
Output: data/phase10_results.csv
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import data_loader
from strategy_matrix_backtest import (
    SPREAD_PIPS, MONTHS, IS_MONTHS, INITIAL_BALANCE, RISK_PCT_DEFAULT,
    CRIT_PF, CRIT_MAX_DD, CRIT_PROF_MONTHS, windowed_atr,
    run_sim, metrics, REPO_ROOT,
)

SPREAD_PIPS.setdefault('CADJPY', 2.0)
SPREAD_PIPS.setdefault('NZDJPY', 2.2)
SPREAD_PIPS.setdefault('CHFJPY', 2.5)

OUT_CSV    = REPO_ROOT / 'data' / 'phase10_results.csv'
TRADES_DIR = REPO_ROOT / 'data' / 'phase10_trades'

SIGNAL_PAIRS = ['GBPJPY', 'EURJPY', 'AUDJPY', 'CADJPY', 'NZDJPY', 'CHFJPY']
ALL_PAIRS    = SIGNAL_PAIRS + ['USDJPY']

# session windows (server hours, matching the project's existing
# convention -- see PROJECT_REPORT.md 2.3: London ~07-16, NY ~12-21)
LONDON_START, LONDON_END = 7, 16
NY_END                   = 21


def fetch_h1(pair: str) -> pd.DataFrame:
    date_to   = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=MONTHS * 30 + 60)
    return data_loader.get_bars(pair, 'H1', date_from, date_to)


# ── 1. Volatility-squeeze breakout ──────────────────────────────────────────

def signals_squeeze(h1: pd.DataFrame, pip: float, spread: float,
                    pctile: float, sl_atr: float, tp_atr: float,
                    bb_period: int = 20, width_lookback: int = 100
                    ) -> list:
    closes = h1['Close'].to_numpy()
    highs  = h1['High'].to_numpy()
    lows   = h1['Low'].to_numpy()
    hours  = h1.index.hour

    s = pd.Series(closes)
    mid = s.rolling(bb_period).mean()
    sd  = s.rolling(bb_period).std()
    upper = (mid + 2 * sd).to_numpy()
    lower = (mid - 2 * sd).to_numpy()
    width = ((upper - lower) / mid.to_numpy())
    width_s = pd.Series(width)
    # trailing percentile RANK of today's width vs the last `width_lookback`
    # bars (excluding itself) -- low rank = compressed / "squeezed"
    width_rank = width_s.rolling(width_lookback).apply(
        lambda w: (w[:-1] < w[-1]).mean(), raw=True).to_numpy()

    atr = windowed_atr(highs, lows, closes, 14, 66) / pip
    min_tp = max(3.0, 2.0 * spread)

    out = []
    for i in range(max(bb_period, width_lookback) + 1, len(h1)):
        hr = hours[i]
        if not (LONDON_START <= hr < NY_END):
            continue
        if np.isnan(width_rank[i - 1]) or np.isnan(atr[i]) or atr[i] <= 0:
            continue
        if width_rank[i - 1] > pctile:          # not squeezed on the PRIOR bar
            continue
        c = closes[i]
        sig = None
        if c > upper[i]:
            sig = 'BUY'
        elif c < lower[i]:
            sig = 'SELL'
        if sig:
            sl_pips = sl_atr * atr[i]
            tp_pips = tp_atr * atr[i]
            if tp_pips >= min_tp:
                out.append((i, sig, sl_pips, tp_pips))
    return out


# ── 2. Cross-asset JPY momentum (USDJPY proxy) ──────────────────────────────

def build_usdjpy_proxy(h1_usdjpy: pd.DataFrame, pip: float):
    """Per-bar: pips moved from that day's LONDON_START-hour open to the
    current close, and that day's ATR at London start (for thresholding)."""
    opens  = h1_usdjpy['Open'].to_numpy()
    closes = h1_usdjpy['Close'].to_numpy()
    highs  = h1_usdjpy['High'].to_numpy()
    lows   = h1_usdjpy['Low'].to_numpy()
    hours  = h1_usdjpy.index.hour
    dates  = h1_usdjpy.index.date
    atr = windowed_atr(highs, lows, closes, 14, 66) / pip

    n = len(h1_usdjpy)
    move_pips = np.full(n, np.nan)
    day_atr   = np.full(n, np.nan)
    cur_date, open_px, a0 = None, None, np.nan
    for i in range(n):
        if dates[i] != cur_date:
            cur_date, open_px, a0 = dates[i], None, np.nan
        if hours[i] == LONDON_START:
            open_px = opens[i]
            a0 = atr[i]
        if open_px is not None and not np.isnan(a0):
            move_pips[i] = (closes[i] - open_px) / pip
            day_atr[i] = a0
    return move_pips, day_atr


def signals_xmomentum(h1: pd.DataFrame, pip: float, spread: float,
                      usdjpy_move: np.ndarray, usdjpy_atr: np.ndarray,
                      check_hour: int, thr_atr: float,
                      sl_atr: float, tp_atr: float) -> list:
    closes = h1['Close'].to_numpy()
    highs  = h1['High'].to_numpy()
    lows   = h1['Low'].to_numpy()
    hours  = h1.index.hour
    atr = windowed_atr(highs, lows, closes, 14, 66) / pip
    min_tp = max(3.0, 2.0 * spread)

    out = []
    n = min(len(h1), len(usdjpy_move))
    for i in range(66, n):
        if hours[i] != check_hour:
            continue
        mv, a0 = usdjpy_move[i], usdjpy_atr[i]
        if np.isnan(mv) or np.isnan(a0) or a0 <= 0 or np.isnan(atr[i]) or atr[i] <= 0:
            continue
        if abs(mv) < thr_atr * a0:
            continue
        # USDJPY up = JPY weakening -> broad JPY weakness -> BUY the cross
        # USDJPY down = JPY strengthening -> SELL the cross
        sig = 'BUY' if mv > 0 else 'SELL'
        sl_pips = sl_atr * atr[i]
        tp_pips = tp_atr * atr[i]
        if tp_pips >= min_tp:
            out.append((i, sig, sl_pips, tp_pips))
    return out


# ── evaluation ───────────────────────────────────────────────────────────────

def evaluate(label, pair, h1, cands, time_exit_hour, is_start, oos_start,
            now, rows):
    pip = 0.01 if 'JPY' in pair else 0.0001
    tdf, eq = run_sim(h1, cands, pip, SPREAD_PIPS.get(pair, 2.0),
                      RISK_PCT_DEFAULT, time_exit_hour=time_exit_hour)
    mi = metrics(tdf, eq, is_start, oos_start)
    mo = metrics(tdf, eq, oos_start, now)
    is_pass = (mi.get('trades', 0) >= 30 and mi.get('pf', 0) > CRIT_PF
               and mi.get('max_dd_pct', 99) < CRIT_MAX_DD
               and mi.get('prof_months_pct', 0) >= CRIT_PROF_MONTHS)
    passed = (is_pass and mo.get('trades', 0) >= 10 and mo.get('pnl', 0) > 0
              and mo.get('max_dd_pct', 99) < CRIT_MAX_DD)
    row = {
        'run': label, 'pair': pair,
        'is_trades': mi.get('trades', 0), 'is_wr': mi.get('win_rate', 0),
        'is_pf': mi.get('pf', 0), 'is_pnl': mi.get('pnl', 0),
        'is_dd': mi.get('max_dd_pct', 0),
        'is_prof_months': mi.get('prof_months_pct', 0),
        'oos_trades': mo.get('trades', 0), 'oos_wr': mo.get('win_rate', 0),
        'oos_pf': mo.get('pf', 0), 'oos_pnl': mo.get('pnl', 0),
        'oos_dd': mo.get('max_dd_pct', 0),
        'IS_PASS': is_pass, 'PASS': passed,
    }
    rows.append(row)
    if not tdf.empty:
        TRADES_DIR.mkdir(parents=True, exist_ok=True)
        safe = label.replace('/', '_').replace(' ', '')
        tdf.to_csv(TRADES_DIR / f'{safe}__{pair}.csv', index=False)
    tag = 'PASS' if passed else ('is-ok' if is_pass else 'fail ')
    print(f"[{tag}] {label:<26} {pair:7}  "
          f"IS: n={row['is_trades']:>4} PF={row['is_pf']:<5} "
          f"DD={row['is_dd']:>5.2f}% pm={row['is_prof_months']:>5.1f}%  "
          f"OOS: n={row['oos_trades']:>4} PF={row['oos_pf']:<5} "
          f"P&L=${row['oos_pnl']:>+9,.2f} DD={row['oos_dd']:.2f}%")


def main():
    now = pd.Timestamp.now(tz='UTC')
    oos_start = now - pd.DateOffset(months=MONTHS - IS_MONTHS)
    is_start  = now - pd.DateOffset(months=MONTHS)

    print('================ PHASE 10: JPY CROSSES, LONDON/NY ================\n')
    data = {}
    for p in ALL_PAIRS:
        print(f'Fetching {p} H1 ...')
        data[p] = fetch_h1(p)
        print(f'  {len(data[p]):,} bars ({data[p].index[0].date()} -> '
              f'{data[p].index[-1].date()})')
    print()

    rows = []

    # ---- 1. Volatility-squeeze breakout ----
    print('=== 1. Volatility-squeeze breakout (H1, London/NY 07-21 server) ===')
    for pair in SIGNAL_PAIRS + ['USDJPY']:
        pip = 0.01
        h1 = data[pair]
        for pctile in (0.15, 0.25):
            for sl_atr, tp_atr in ((1.5, 2.5), (1.25, 3.0)):
                label = f'sqz(p{pctile},sl{sl_atr},tp{tp_atr})'
                cands = signals_squeeze(h1, pip, SPREAD_PIPS.get(pair, 2.0),
                                        pctile, sl_atr, tp_atr)
                evaluate(label, pair, h1, cands, None,
                         is_start, oos_start, now, rows)

    # ---- 2. Cross-asset JPY momentum vs USDJPY proxy ----
    print('\n=== 2. Cross-asset JPY momentum (USDJPY proxy, London/NY overlap) ===')
    usdjpy_move, usdjpy_atr = build_usdjpy_proxy(data['USDJPY'], 0.01)
    for pair in SIGNAL_PAIRS:
        pip = 0.01
        h1 = data[pair]
        for check_hour in (12, 14):
            for thr_atr in (0.75, 1.25):
                label = f'xmo(ch{check_hour},thr{thr_atr})'
                cands = signals_xmomentum(
                    h1, pip, SPREAD_PIPS.get(pair, 2.0),
                    usdjpy_move, usdjpy_atr, check_hour, thr_atr,
                    sl_atr=1.5, tp_atr=2.5)
                evaluate(label, pair, h1, cands, NY_END,
                         is_start, oos_start, now, rows)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f'\nSaved: {OUT_CSV}')
    print(f"Full passes (IS + OOS): {int(df['PASS'].sum())} / {len(df)}")
    print(f"IS-only passes:         {int(df['IS_PASS'].sum())} / {len(df)}")
    if df['PASS'].any():
        print('\n--- FULL PASSES ---')
        print(df[df['PASS']].to_string(index=False))
    near = df[(~df['PASS']) & (df['oos_pnl'] > 0) & (df['is_pf'] >= 1.15)]
    if len(near):
        print('\n--- NEAR-MISSES (IS PF >= 1.15, OOS positive) ---')
        print(near[['run', 'pair', 'is_trades', 'is_pf', 'is_dd',
                    'is_prof_months', 'oos_trades', 'oos_pf', 'oos_pnl']]
              .to_string(index=False))


if __name__ == '__main__':
    main()
