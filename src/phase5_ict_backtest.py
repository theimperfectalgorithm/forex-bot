"""
Forex Bot - Phase 5: Mechanical ICT "2022 model" backtest (M5)
===============================================================
Tests a fully mechanical implementation of the ICT (Inner Circle Trader)
day-trading model on EURUSD (+ GBPUSD as a robustness cross-check).

The model, made mechanical (long side; shorts mirrored):
  1. LIQUIDITY POOL: a reference level holding resting stops --
       'asian' = today's Asian session low (00:00-07:00 UTC), or
       'pd'    = previous calendar day's low (PDL).
  2. KILLZONE: the sweep must occur inside a killzone --
       'ldn' = 07:00-10:00 UTC, 'ny' = 12:00-15:00 UTC.
  3. SWEEP: a bar trades BELOW the pool (low < pool). The running low of
     the sweep is the protected extreme.
  4. DISPLACEMENT / MSS (market structure shift): within 24 M5 bars of
     the sweep, a bar CLOSES above the most recent confirmed M5 fractal
     swing high (2-bar fractals, confirmed 2 bars later, formed within
     the last 24 bars) with a candle body >= 1.0 x ATR(14) -- momentum,
     not drift.
  5. FVG ENTRY: the displacement leg must leave a bullish fair value gap
     (3-bar imbalance: low[j] > high[j-2], gap >= 0.5 pip) at or within
     3 bars after the MSS bar. Entry on the first bar (within 12 bars,
     before killzone end + 1h) that retraces INTO the gap (low <= gap
     top) without closing through its bottom. Entry at that bar's CLOSE
     -- a conservative approximation of ICT's limit-at-FVG fill (real
     fills would be at equal-or-better prices).
  6. SL at the sweep extreme. TP: '2R' = 2x the SL distance, or 'pool' =
     the opposite liquidity pool. Time exit 16:00 (ldn) / 20:00 (ny).
     One setup per day (engine also enforces one trade/day).

Honest context: phases 3-4 already killed the nearest relative of this
setup (sweep + re-close fade, "turtle soup") on EURUSD at M15. The NEW
elements under test are the displacement/MSS confirmation, the FVG
retracement entry, and M5 granularity. Everything else (criteria,
36-month IS/OOS split, spread costs) is identical to phases 1-4.

Requirements: MetaTrader 5 OPEN and LOGGED IN.
Output: data/phase5_ict_results.csv
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import MetaTrader5 as mt5

from strategy_matrix_backtest import (
    SPREAD_PIPS, MONTHS, IS_MONTHS, windowed_atr, REPO_ROOT,
)
from phase3_session_structure_search import evaluate

OUT_CSV = REPO_ROOT / 'data' / 'phase5_ict_results.csv'
PAIRS = ['EURUSD', 'GBPUSD']
PIP = 0.0001

KILLZONES = {'ldn': (7, 10, 16), 'ny': (12, 15, 20)}   # start, end, time_exit

MSS_WINDOW    = 24    # M5 bars after sweep in which MSS must occur (2h)
FRACTAL_LOOK  = 24    # swing must have formed within this many bars
FVG_SCAN      = 3     # bars after MSS in which the FVG may complete
ENTRY_WINDOW  = 12    # bars after FVG formation for the retracement fill
MIN_GAP_PIPS  = 0.5
MIN_SL_PIPS   = 2.0
BODY_ATR_MIN  = 1.0   # displacement body >= 1 x ATR


def fetch_m5(pair: str) -> pd.DataFrame:
    if not mt5.initialize():
        raise RuntimeError(f'MT5 init failed: {mt5.last_error()}')
    date_to   = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=MONTHS * 30 + 30)
    rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M5, date_from, date_to)
    if rates is None or len(rates) == 0:
        raise ValueError(f'no M5 data for {pair}: {mt5.last_error()}')
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High',
                       'low': 'Low', 'close': 'Close'}, inplace=True)
    return df[['Open', 'High', 'Low', 'Close', 'tick_volume']]


def signals_ict(m5, pip, spread, pool_kind, kz):
    kz_lo, kz_hi, _ = KILLZONES[kz]
    opens  = m5['Open'].to_numpy()
    highs  = m5['High'].to_numpy()
    lows   = m5['Low'].to_numpy()
    closes = m5['Close'].to_numpy()
    hours  = m5.index.hour
    dates  = m5.index.date
    n = len(m5)
    atr = windowed_atr(highs, lows, closes, 14, 66)      # price units

    # 2-bar fractals, confirmed 2 bars later
    is_fhi = np.zeros(n, bool)
    is_flo = np.zeros(n, bool)
    for i in range(2, n - 2):
        if highs[i] == max(highs[i-2:i+3]):
            is_fhi[i] = True
        if lows[i] == min(lows[i-2:i+3]):
            is_flo[i] = True

    out = {}   # bar_idx -> (dir, sl_pips, tp_pips); first setup of day wins

    day_hi = day_lo = None
    pdh = pdl = None
    cur = None
    asian_hi = asian_lo = None
    # per-direction day state
    L = {}
    S = {}

    def reset_day():
        return {'sweep': False, 'ext': None, 'start': None, 'done': False,
                'fvg': None, 'fvg_i': None, 'mss_i': None}

    for i in range(n):
        d, hr = dates[i], hours[i]
        if d != cur:
            pdh, pdl = day_hi, day_lo
            day_hi = day_lo = None
            cur = d
            asian_hi = asian_lo = None
            L, S = reset_day(), reset_day()
        day_hi = highs[i] if day_hi is None else max(day_hi, highs[i])
        day_lo = lows[i]  if day_lo is None else min(day_lo, lows[i])
        if hr < 7:
            asian_hi = highs[i] if asian_hi is None else max(asian_hi, highs[i])
            asian_lo = lows[i]  if asian_lo is None else min(asian_lo, lows[i])

        if pool_kind == 'asian':
            pool_lo, pool_hi = asian_lo, asian_hi
            if pool_lo is not None and (pool_hi - pool_lo) / pip < 5:
                pool_lo = pool_hi = None
        else:
            pool_lo, pool_hi = pdl, pdh
        if pool_lo is None or np.isnan(atr[i]) or atr[i] <= 0:
            continue

        in_kz     = kz_lo <= hr < kz_hi
        in_entry  = kz_lo <= hr < kz_hi + 1

        # ================= LONG side (sweep of the low pool) ==============
        st = L
        if not st['done']:
            if in_kz and lows[i] < pool_lo:
                if not st['sweep']:
                    st['sweep'], st['start'], st['ext'] = True, i, lows[i]
                else:
                    st['ext'] = min(st['ext'], lows[i])
            if st['sweep'] and st['fvg'] is None:
                if i - st['start'] > MSS_WINDOW:
                    st['done'] = True
                else:
                    # most recent confirmed fractal high within lookback
                    fh_val = None
                    for j in range(i - 2, max(i - 2 - FRACTAL_LOOK, 1), -1):
                        if is_fhi[j]:
                            fh_val = highs[j]
                            break
                    body = closes[i] - opens[i]
                    if (fh_val is not None and closes[i] > fh_val
                            and body >= BODY_ATR_MIN * atr[i]):
                        st['mss_i'] = i
                        # find bullish FVG on/after the MSS bar
                        for j in range(i, min(i + FVG_SCAN + 1, n)):
                            if j >= 2 and lows[j] - highs[j - 2] >= MIN_GAP_PIPS * pip:
                                st['fvg'] = (highs[j - 2], lows[j])   # bottom, top
                                st['fvg_i'] = j
                                break
                        if st['fvg'] is None and i - st['start'] >= MSS_WINDOW:
                            st['done'] = True
            elif st['fvg'] is not None:
                gap_bot, gap_top = st['fvg']
                if i > st['fvg_i']:
                    if (i - st['fvg_i'] > ENTRY_WINDOW) or not in_entry:
                        st['done'] = True
                    elif closes[i] < gap_bot:
                        st['done'] = True                 # gap invalidated
                    elif lows[i] <= gap_top:
                        entry = closes[i]
                        sl_pips = (entry - st['ext']) / pip
                        if sl_pips >= MIN_SL_PIPS:
                            if pool_kind == 'asian':
                                target = asian_hi
                            else:
                                target = pdh
                            tp2r  = 2.0 * sl_pips
                            tpool = (target - entry) / pip if target else 0
                            out[i] = ('BUY', sl_pips, tp2r, tpool)
                        st['done'] = True

        # ================= SHORT side (sweep of the high pool) ============
        st = S
        if not st['done'] and pool_hi is not None:
            if in_kz and highs[i] > pool_hi:
                if not st['sweep']:
                    st['sweep'], st['start'], st['ext'] = True, i, highs[i]
                else:
                    st['ext'] = max(st['ext'], highs[i])
            if st['sweep'] and st['fvg'] is None:
                if i - st['start'] > MSS_WINDOW:
                    st['done'] = True
                else:
                    fl_val = None
                    for j in range(i - 2, max(i - 2 - FRACTAL_LOOK, 1), -1):
                        if is_flo[j]:
                            fl_val = lows[j]
                            break
                    body = opens[i] - closes[i]
                    if (fl_val is not None and closes[i] < fl_val
                            and body >= BODY_ATR_MIN * atr[i]):
                        st['mss_i'] = i
                        for j in range(i, min(i + FVG_SCAN + 1, n)):
                            if j >= 2 and lows[j - 2] - highs[j] >= MIN_GAP_PIPS * pip:
                                st['fvg'] = (lows[j - 2], highs[j])   # top, bottom
                                st['fvg_i'] = j
                                break
                        if st['fvg'] is None and i - st['start'] >= MSS_WINDOW:
                            st['done'] = True
            elif st['fvg'] is not None:
                gap_top, gap_bot = st['fvg']
                if i > st['fvg_i']:
                    if (i - st['fvg_i'] > ENTRY_WINDOW) or not in_entry:
                        st['done'] = True
                    elif closes[i] > gap_top:
                        st['done'] = True
                    elif highs[i] >= gap_bot:
                        entry = closes[i]
                        sl_pips = (st['ext'] - entry) / pip
                        if sl_pips >= MIN_SL_PIPS:
                            target = asian_lo if pool_kind == 'asian' else pdl
                            tp2r  = 2.0 * sl_pips
                            tpool = (entry - target) / pip if target else 0
                            out[i] = ('SELL', sl_pips, tp2r, tpool)
                        st['done'] = True

    return out


def build_cands(raw: dict, tp_mode: str, spread: float):
    min_tp = max(3.0, 2.0 * spread)
    cands = []
    for i, (d, sl_pips, tp2r, tpool) in raw.items():
        tp = tp2r if tp_mode == '2R' else tpool
        if tp >= min_tp:
            cands.append((i, d, sl_pips, tp))
    return sorted(cands)


def main():
    now = pd.Timestamp.now(tz='UTC')
    oos_start = now - pd.DateOffset(months=MONTHS - IS_MONTHS)
    is_start  = now - pd.DateOffset(months=MONTHS)

    rows = []
    for pair in PAIRS:
        print(f'Fetching {pair} M5 ...')
        m5 = fetch_m5(pair)
        print(f'  {len(m5):,} bars ({m5.index[0].date()} -> {m5.index[-1].date()})')
        for pool in ('asian', 'pd'):
            for kz, (_, _, t_exit) in KILLZONES.items():
                raw = signals_ict(m5, PIP, SPREAD_PIPS[pair], pool, kz)
                for tp_mode in ('2R', 'pool'):
                    cands = build_cands(raw, tp_mode, SPREAD_PIPS[pair])
                    label = f'ict({pool},{kz},{tp_mode})'
                    evaluate(label, pair, m5, cands, t_exit,
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
