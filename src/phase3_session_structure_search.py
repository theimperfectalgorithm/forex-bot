"""
Forex Bot - Phase 3: Session-structure strategy search (M15 / H1 / H4)
=======================================================================
Phases 1-2 established that indicator-following strategies (EMA/MA cross,
RSI divergence, RSI mean reversion) on H1 have no edge net of spread on any
major -- the lone survivor (Asian range breakout GBPJPY) exploits SESSION
STRUCTURE, not indicator geometry. Phase 3 tests four structurally
different families built on that lesson, using timeframes beyond H1/H4:

  1. lorb  (M15) London Open Range Breakout -- the GBPJPY-validated thesis
           applied to London-session pairs at London open: pre-open (or
           first-hour) range, breakout entry, SL at opposite range edge,
           TP = R multiple, TIME EXIT at 16:00 UTC (flat before NY close;
           no overnight floating risk against the prop-firm daily limit).
  2. amr   (M15) Asian-hours mean reversion -- quiet-hours (00:00-06:00
           UTC) z-score reversion to the M15 SMA20; the classic
           low-participation inefficiency window; time exit 07:00 before
           London opens. Gives true INSIDE-Asian-session coverage (never
           tested before in this repo -- asian_breakout is a stub).
  3. nyc   (H1)  NY-overlap continuation -- if the London move (08:00 open
           to 13:00) exceeds a threshold in ATR units, trade continuation
           into the overlap with ATR-scaled SL/TP, time exit 20:00.
  4. don   (H4)  Donchian 20-bar channel breakout with ATR(14)-scaled
           SL/TP (2x / 4x) -- slow multi-day trend capture, volatility-
           adaptive stops (every phase-1 failure used fixed pips).

Methodology identical to phases 1-2: 36 months, IS = first 24m /
OOS = last 12m, criteria PF > 1.3, DD < 8%, >= 60% profitable months,
positive OOS. Spread-cost deducted per trade. Small per-family grids to
limit multiple-testing; robustness across neighbour variants is reported.

Requirements: MetaTrader 5 OPEN and LOGGED IN.
Output: data/phase3_results.csv
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
    MAJORS, SPREAD_PIPS, RISK_PCT_DEFAULT, MONTHS, IS_MONTHS,
    CRIT_PF, CRIT_MAX_DD, CRIT_PROF_MONTHS,
    windowed_atr, run_sim, metrics, REPO_ROOT,
)

OUT_CSV = REPO_ROOT / 'data' / 'phase3_results.csv'


def fetch_tf(pair: str, timeframe: str) -> pd.DataFrame:
    date_to   = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=MONTHS * 30 + 60)
    return data_loader.get_bars(pair, timeframe, date_from, date_to)


# ── 1. LORB -- London Open Range Breakout (M15) ─────────────────────────────

def signals_lorb(m15, pip, spread, range_def, r_mult):
    """range_def: 'pre' = 05:00-08:00 range, entries 08:00-12:00;
                  'fh'  = 08:00-09:00 range, entries 09:00-12:00."""
    rng_lo_h, rng_hi_h, ent_lo, ent_hi = \
        (5, 8, 8, 12) if range_def == 'pre' else (8, 9, 9, 12)
    closes = m15['Close'].to_numpy()
    highs  = m15['High'].to_numpy()
    lows   = m15['Low'].to_numpy()
    hours  = m15.index.hour
    dates  = m15.index.date
    min_range = max(5.0, 2.0 * spread)

    out = []
    cur, r_hi, r_lo, done = None, None, None, False
    for i in range(len(m15)):
        d, hr = dates[i], hours[i]
        if d != cur:
            cur, r_hi, r_lo, done = d, None, None, False
        if rng_lo_h <= hr < rng_hi_h:
            r_hi = highs[i] if r_hi is None else max(r_hi, highs[i])
            r_lo = lows[i]  if r_lo is None else min(r_lo, lows[i])
            continue
        if done or r_hi is None or not (ent_lo <= hr < ent_hi):
            continue
        rng_pips = (r_hi - r_lo) / pip
        if rng_pips < min_range:
            done = True          # degenerate range -- skip the whole day
            continue
        c = closes[i]
        if c > r_hi:
            sl_pips = (c - r_lo) / pip
            out.append((i, 'BUY', sl_pips, sl_pips * r_mult))
            done = True
        elif c < r_lo:
            sl_pips = (r_hi - c) / pip
            out.append((i, 'SELL', sl_pips, sl_pips * r_mult))
            done = True
    return out


# ── 2. AMR -- Asian-hours mean reversion (M15) ──────────────────────────────

def signals_amr(m15, pip, spread, z_thr, sl_mult=1.5):
    closes = m15['Close'].to_numpy()
    s = pd.Series(closes)
    sma20 = s.rolling(20).mean().to_numpy()
    std20 = s.rolling(20).std().to_numpy()
    hours = m15.index.hour
    min_tp = max(3.0, 2.5 * spread)

    out = []
    for i in range(20, len(m15)):
        if hours[i] >= 6:                      # entries 00:00-05:45 only
            continue
        if np.isnan(std20[i]) or std20[i] <= 0:
            continue
        z = (closes[i] - sma20[i]) / std20[i]
        if z <= -z_thr:                        # stretched down -> revert up
            tp_pips = (sma20[i] - closes[i]) / pip
            if tp_pips >= min_tp:
                out.append((i, 'BUY', tp_pips * sl_mult, tp_pips))
        elif z >= z_thr:                       # stretched up -> revert down
            tp_pips = (closes[i] - sma20[i]) / pip
            if tp_pips >= min_tp:
                out.append((i, 'SELL', tp_pips * sl_mult, tp_pips))
    return out


# ── 3. NYC -- NY-overlap continuation (H1) ──────────────────────────────────

def signals_nyc(h1, pip, thr_atr, sl_atr=1.5, tp_atr=2.25):
    closes = h1['Close'].to_numpy()
    opens  = h1['Open'].to_numpy()
    atr = windowed_atr(h1['High'].to_numpy(), h1['Low'].to_numpy(), closes,
                       14, 66) / pip
    hours = h1.index.hour
    out = []
    for i in range(66, len(h1)):
        if hours[i] != 12:                     # bar 12:00-13:00 = London morning close
            continue
        if np.isnan(atr[i]) or atr[i] <= 0 or hours[i - 4] != 8:
            continue                           # need contiguous 08:00 bar
        move_pips = (closes[i] - opens[i - 4]) / pip
        if abs(move_pips) < thr_atr * atr[i]:
            continue
        d = 'BUY' if move_pips > 0 else 'SELL'
        out.append((i, d, sl_atr * atr[i], tp_atr * atr[i]))
    return out


# ── 4. DON -- H4 Donchian breakout, ATR-scaled SL/TP ────────────────────────

def signals_don(h4, pip, n_channel=20, sl_atr=2.0, tp_atr=4.0):
    closes = h4['Close'].to_numpy()
    highs  = h4['High'].to_numpy()
    lows   = h4['Low'].to_numpy()
    atr = windowed_atr(highs, lows, closes, 14, 66) / pip
    out = []
    for i in range(max(66, n_channel), len(h4)):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        ch_hi = highs[i - n_channel:i].max()
        ch_lo = lows[i - n_channel:i].min()
        if closes[i] > ch_hi:
            out.append((i, 'BUY', sl_atr * atr[i], tp_atr * atr[i]))
        elif closes[i] < ch_lo:
            out.append((i, 'SELL', sl_atr * atr[i], tp_atr * atr[i]))
    return out


# ── evaluation ───────────────────────────────────────────────────────────────

def evaluate(label, pair, bars, cands, time_exit, is_start, oos_start, now,
             rows):
    pip = 0.01 if 'JPY' in pair else 0.0001
    tdf, eq = run_sim(bars, cands, pip, SPREAD_PIPS[pair], RISK_PCT_DEFAULT,
                      time_exit_hour=time_exit)
    m_is  = metrics(tdf, eq, is_start, oos_start)
    m_oos = metrics(tdf, eq, oos_start, now)
    is_pass = (
        m_is.get('trades', 0) >= 30
        and m_is.get('pf', 0) > CRIT_PF
        and m_is.get('max_dd_pct', 99) < CRIT_MAX_DD
        and m_is.get('prof_months_pct', 0) >= CRIT_PROF_MONTHS
    )
    passed = (
        is_pass
        and m_oos.get('trades', 0) >= 10
        and m_oos.get('pnl', 0) > 0
        and m_oos.get('max_dd_pct', 99) < CRIT_MAX_DD
    )
    rows.append({
        'run': label, 'pair': pair,
        'is_trades': m_is.get('trades', 0), 'is_wr': m_is.get('win_rate', 0),
        'is_pf': m_is.get('pf', 0), 'is_pnl': m_is.get('pnl', 0),
        'is_dd': m_is.get('max_dd_pct', 0),
        'is_prof_months': m_is.get('prof_months_pct', 0),
        'oos_trades': m_oos.get('trades', 0), 'oos_wr': m_oos.get('win_rate', 0),
        'oos_pf': m_oos.get('pf', 0), 'oos_pnl': m_oos.get('pnl', 0),
        'oos_dd': m_oos.get('max_dd_pct', 0),
        'IS_PASS': is_pass, 'PASS': passed,
    })
    r = rows[-1]
    tag = 'PASS' if passed else ('is-ok' if is_pass else 'fail ')
    print(f"[{tag}] {label:<22} {pair}  "
          f"IS: n={r['is_trades']:>4} PF={r['is_pf']:<5} DD={r['is_dd']:>5.2f}% "
          f"pm={r['is_prof_months']:>5.1f}%  "
          f"OOS: n={r['oos_trades']:>4} PF={r['oos_pf']:<5} "
          f"P&L=${r['oos_pnl']:>+9,.2f} DD={r['oos_dd']:.2f}%")


def main():
    now = pd.Timestamp.now(tz='UTC')
    oos_start = now - pd.DateOffset(months=MONTHS - IS_MONTHS)
    is_start  = now - pd.DateOffset(months=MONTHS)

    m15_data, h1_data, h4_data = {}, {}, {}
    for pair in MAJORS:
        print(f'Fetching {pair} M15/H1/H4 ...')
        m15_data[pair] = fetch_tf(pair, 'M15')
        h1_data[pair]  = fetch_tf(pair, 'H1')
        h4_data[pair]  = fetch_tf(pair, 'H4')
        m15_start = m15_data[pair].index[0]
        if m15_start > is_start:
            print(f'  NOTE: {pair} M15 history starts {m15_start.date()} -- '
                  f'in-sample window is shortened for M15 runs')
    print()

    rows = []

    print('=== 1. LORB -- London Open Range Breakout (M15) ===')
    for pair in MAJORS:
        pip = 0.01 if 'JPY' in pair else 0.0001
        for range_def in ('pre', 'fh'):
            for r_mult in (1.5, 2.0):
                label = f'lorb({range_def},R{r_mult})'
                cands = signals_lorb(m15_data[pair], pip, SPREAD_PIPS[pair],
                                     range_def, r_mult)
                evaluate(label, pair, m15_data[pair], cands, 16,
                         is_start, oos_start, now, rows)

    print('\n=== 2. AMR -- Asian-hours mean reversion (M15) ===')
    for pair in MAJORS:
        pip = 0.01 if 'JPY' in pair else 0.0001
        for z_thr in (2.0, 2.5):
            label = f'amr(z{z_thr})'
            cands = signals_amr(m15_data[pair], pip, SPREAD_PIPS[pair], z_thr)
            evaluate(label, pair, m15_data[pair], cands, 7,
                     is_start, oos_start, now, rows)

    print('\n=== 3. NYC -- NY-overlap continuation (H1) ===')
    for pair in MAJORS:
        pip = 0.01 if 'JPY' in pair else 0.0001
        for thr in (0.75, 1.25):
            label = f'nyc(thr{thr})'
            cands = signals_nyc(h1_data[pair], pip, thr)
            evaluate(label, pair, h1_data[pair], cands, 20,
                     is_start, oos_start, now, rows)

    print('\n=== 4. DON -- H4 Donchian breakout, ATR stops ===')
    for pair in MAJORS:
        pip = 0.01 if 'JPY' in pair else 0.0001
        cands = signals_don(h4_data[pair], pip)
        evaluate('don(20,2x/4x)', pair, h4_data[pair], cands, None,
                 is_start, oos_start, now, rows)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f'\nSaved: {OUT_CSV}')
    print(f"Full passes (IS + OOS): {int(df['PASS'].sum())} / {len(df)}")
    print(f"IS-only passes:         {int(df['IS_PASS'].sum())} / {len(df)}")
    if df['PASS'].any():
        print('\n--- FULL PASSES ---')
        print(df[df['PASS']].to_string(index=False))
    # near-misses worth a look: OOS-profitable with decent IS
    near = df[(~df['PASS']) & (df['oos_pnl'] > 0) & (df['is_pf'] >= 1.15)]
    if len(near):
        print('\n--- NEAR-MISSES (IS PF >= 1.15, OOS positive) ---')
        print(near.to_string(index=False))


if __name__ == '__main__':
    main()
