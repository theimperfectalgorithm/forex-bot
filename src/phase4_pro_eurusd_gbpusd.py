"""
Forex Bot - Phase 4: Professional-style EURUSD/GBPUSD strategies
=================================================================
How institutional algo desks actually trade these two pairs, mapped to
what a retail MT5 stack can implement (no order-flow, no HFT co-location,
and NO news-reaction plays -- the 5ers forbids entries/exits within
+/-5 min of high-impact news):

  1. lfb  (M15) London false-breakout FADE -- liquidity-sweep reversal.
          Phase 3 measured chasing these breakouts at PF 0.75-0.85 with
          spread already paid: a consistent ANTI-edge. Pros trade the
          other side: a break of the Asian range that closes back inside
          within k bars is a failed sweep; enter the reversal, SL beyond
          the sweep extreme, TP mid-range or opposite edge, flat by 16:00.
  2. fix  (M15) WMR 16:00 London fix flows -- the heaviest predictable
          institutional flow in EURUSD/GBPUSD.
          fixmom: pre-fix momentum -- banks pre-hedge fix orders; enter at
                  15:15 in the direction of the 13:00->15:15 move, exit at
                  the fix (16:00 open).
          fixrev: post-fix reversion -- once fix flow completes, the
                  flow-driven push partially retraces; if the 14:00->16:00
                  move exceeds a threshold in ATR units, fade it at 16:00
                  targeting a 50% retrace, time exit 18:00.
  3. rv   (M15) Relative value EURUSD vs GBPUSD -- classic stat-arb,
          expressed directly through EURGBP (the ratio of the two).
          Z-score of close vs rolling mean; fade |z| >= threshold during
          London hours; TP = the mean, SL = 1.5x. This is the textbook
          way quant desks trade "EURUSD against GBPUSD" without paying
          two spreads.
  4. bbr  (H1)  Volatility-regime-conditioned mean reversion -- pros
          never run MR unconditionally; they gate it on the vol regime.
          Bollinger(20,2) fade taken ONLY when ATR(14) sits in the low
          percentile of its trailing 500-bar distribution.

Same methodology as phases 1-3: 36 months, IS = first 24m / OOS = last
12m, PF > 1.3, DD < 8%, >= 60% profitable months, positive OOS, spread
deducted per trade. Small grids; robustness matters more than any cell.

Requirements: MetaTrader 5 OPEN and LOGGED IN.
Output: data/phase4_results.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy_matrix_backtest import (
    SPREAD_PIPS, MONTHS, IS_MONTHS, windowed_atr, REPO_ROOT,
)
from phase3_session_structure_search import fetch_tf, evaluate

SPREAD_PIPS['EURGBP'] = 1.5          # cross not in the phase-1 table

OUT_CSV = REPO_ROOT / 'data' / 'phase4_results.csv'
PAIRS = ['EURUSD', 'GBPUSD']
PIP = 0.0001


# ── 1. LFB -- London false-breakout fade (M15) ──────────────────────────────

def signals_lfb(m15, pip, spread, k_bars, tp_target):
    """Asian range (00:00-07:45). First M15 close beyond an edge during
    08:00-12:00 arms that side; if a close back INSIDE the range follows
    within k_bars, fade the failed break. SL = sweep extreme, TP = 'mid'
    (range midpoint) or 'opp' (opposite edge). One armed attempt per side
    per day."""
    closes = m15['Close'].to_numpy()
    highs  = m15['High'].to_numpy()
    lows   = m15['Low'].to_numpy()
    hours  = m15.index.hour
    dates  = m15.index.date
    min_range = max(8.0, 2.0 * spread)
    min_tp    = max(3.0, 2.0 * spread)

    out = []
    cur = None
    for i in range(len(m15)):
        d, hr = dates[i], hours[i]
        if d != cur:
            cur = d
            r_hi = r_lo = None
            up_state = dn_state = None       # None | (break_idx, extreme) | 'done'
        if hr < 8:
            r_hi = highs[i] if r_hi is None else max(r_hi, highs[i])
            r_lo = lows[i]  if r_lo is None else min(r_lo, lows[i])
            continue
        if r_hi is None or (r_hi - r_lo) / pip < min_range or not (8 <= hr < 12):
            continue
        mid = (r_hi + r_lo) / 2.0
        c = closes[i]

        # ---- upside sweep
        if up_state is None:
            if c > r_hi:
                up_state = (i, highs[i])
        elif up_state != 'done':
            bidx, ext = up_state
            ext = max(ext, highs[i])
            if c < r_hi:                      # closed back inside -> failed break
                sl_pips = (ext - c) / pip
                tp_pips = ((c - mid) if tp_target == 'mid' else (c - r_lo)) / pip
                if sl_pips > 0 and tp_pips >= min_tp:
                    out.append((i, 'SELL', sl_pips, tp_pips))
                up_state = 'done'
            elif i - bidx >= k_bars:
                up_state = 'done'             # break held -> no fade this side
            else:
                up_state = (bidx, ext)

        # ---- downside sweep
        if dn_state is None:
            if c < r_lo:
                dn_state = (i, lows[i])
        elif dn_state != 'done':
            bidx, ext = dn_state
            ext = min(ext, lows[i])
            if c > r_lo:
                sl_pips = (c - ext) / pip
                tp_pips = ((mid - c) if tp_target == 'mid' else (r_hi - c)) / pip
                if sl_pips > 0 and tp_pips >= min_tp:
                    out.append((i, 'BUY', sl_pips, tp_pips))
                dn_state = 'done'
            elif i - bidx >= k_bars:
                dn_state = 'done'
            else:
                dn_state = (bidx, ext)
    return out


# ── 2. FIX -- WMR 16:00 fix flows (M15) ─────────────────────────────────────

def signals_fixmom(m15, pip):
    """Enter at the close of the 15:00 bar (=15:15) in the direction of the
    13:00->15:15 move; wide ATR stops so the 16:00 time exit does the work."""
    closes = m15['Close'].to_numpy()
    opens  = m15['Open'].to_numpy()
    atr = windowed_atr(m15['High'].to_numpy(), m15['Low'].to_numpy(), closes,
                       14, 66) / pip
    hours, mins = m15.index.hour, m15.index.minute
    dates = m15.index.date
    out = []
    cur, o13 = None, None
    for i in range(66, len(m15)):
        if dates[i] != cur:
            cur, o13 = dates[i], None
        if hours[i] == 13 and mins[i] == 0:
            o13 = opens[i]
        if hours[i] == 15 and mins[i] == 0 and o13 is not None \
                and not np.isnan(atr[i]) and atr[i] > 0:
            move_pips = (closes[i] - o13) / pip
            if abs(move_pips) >= 2.0 * atr[i]:     # need a real pre-fix trend
                d = 'BUY' if move_pips > 0 else 'SELL'
                out.append((i, d, 10.0 * atr[i], 10.0 * atr[i]))
    return out


def signals_fixrev(m15, pip, thr_atr):
    """At 16:00 (close of the 15:45 bar), fade the 14:00->16:00 move if it
    exceeds thr_atr x ATR(M15); TP = 50% retrace, SL = 6x ATR, exit 18:00."""
    closes = m15['Close'].to_numpy()
    opens  = m15['Open'].to_numpy()
    atr = windowed_atr(m15['High'].to_numpy(), m15['Low'].to_numpy(), closes,
                       14, 66) / pip
    hours, mins = m15.index.hour, m15.index.minute
    dates = m15.index.date
    out = []
    cur, o14 = None, None
    for i in range(66, len(m15)):
        if dates[i] != cur:
            cur, o14 = dates[i], None
        if hours[i] == 14 and mins[i] == 0:
            o14 = opens[i]
        if hours[i] == 15 and mins[i] == 45 and o14 is not None \
                and not np.isnan(atr[i]) and atr[i] > 0:
            move_pips = (closes[i] - o14) / pip
            if abs(move_pips) >= thr_atr * atr[i]:
                d = 'SELL' if move_pips > 0 else 'BUY'
                out.append((i, d, 6.0 * atr[i], abs(move_pips) / 2.0))
    return out


# ── 3. RV -- EURUSD/GBPUSD relative value via EURGBP (M15) ──────────────────

def signals_rv(m15, pip, spread, window, z_thr, sl_mult=1.5):
    closes = m15['Close'].to_numpy()
    s = pd.Series(closes)
    mean = s.rolling(window).mean().to_numpy()
    std  = s.rolling(window).std().to_numpy()
    hours = m15.index.hour
    min_tp = max(3.0, 2.5 * spread)
    out = []
    for i in range(window, len(m15)):
        if not (7 <= hours[i] < 16):
            continue
        if np.isnan(std[i]) or std[i] <= 0:
            continue
        z = (closes[i] - mean[i]) / std[i]
        if z <= -z_thr:
            tp_pips = (mean[i] - closes[i]) / pip
            if tp_pips >= min_tp:
                out.append((i, 'BUY', tp_pips * sl_mult, tp_pips))
        elif z >= z_thr:
            tp_pips = (closes[i] - mean[i]) / pip
            if tp_pips >= min_tp:
                out.append((i, 'SELL', tp_pips * sl_mult, tp_pips))
    return out


# ── 4. BBR -- vol-regime-conditioned Bollinger fade (H1) ────────────────────

def signals_bbr(h1, pip, spread, atr_pct_max, sl_mult=1.5):
    closes = h1['Close'].to_numpy()
    s = pd.Series(closes)
    mid = s.rolling(20).mean().to_numpy()
    sd  = s.rolling(20).std().to_numpy()
    atr = windowed_atr(h1['High'].to_numpy(), h1['Low'].to_numpy(), closes,
                       14, 66) / pip
    hours = h1.index.hour
    min_tp = max(3.0, 2.0 * spread)
    LOOK = 500
    out = []
    for i in range(max(66, LOOK), len(h1)):
        if not (7 <= hours[i] < 16):
            continue
        if np.isnan(sd[i]) or sd[i] <= 0 or np.isnan(atr[i]):
            continue
        w = atr[i - LOOK:i]
        w = w[~np.isnan(w)]
        if len(w) < 100:
            continue
        if (w < atr[i]).mean() > atr_pct_max:   # not a low-vol regime
            continue
        upper, lower = mid[i] + 2 * sd[i], mid[i] - 2 * sd[i]
        c = closes[i]
        if c < lower:
            tp_pips = (mid[i] - c) / pip
            if tp_pips >= min_tp:
                out.append((i, 'BUY', tp_pips * sl_mult, tp_pips))
        elif c > upper:
            tp_pips = (c - mid[i]) / pip
            if tp_pips >= min_tp:
                out.append((i, 'SELL', tp_pips * sl_mult, tp_pips))
    return out


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    now = pd.Timestamp.now(tz='UTC')
    oos_start = now - pd.DateOffset(months=MONTHS - IS_MONTHS)
    is_start  = now - pd.DateOffset(months=MONTHS)

    m15, h1 = {}, {}
    for pair in PAIRS + ['EURGBP']:
        print(f'Fetching {pair} M15/H1 ...')
        m15[pair] = fetch_tf(pair, 'M15')
        if pair != 'EURGBP':
            h1[pair] = fetch_tf(pair, 'H1')
    print()

    rows = []

    print('=== 1. LFB -- London false-breakout fade (M15) ===')
    for pair in PAIRS:
        for k in (2, 3):
            for tgt in ('mid', 'opp'):
                cands = signals_lfb(m15[pair], PIP, SPREAD_PIPS[pair], k, tgt)
                evaluate(f'lfb(k{k},{tgt})', pair, m15[pair], cands, 16,
                         is_start, oos_start, now, rows)

    print('\n=== 2. FIX -- WMR 16:00 fix flows (M15) ===')
    for pair in PAIRS:
        cands = signals_fixmom(m15[pair], PIP)
        evaluate('fixmom(2.0atr)', pair, m15[pair], cands, 16,
                 is_start, oos_start, now, rows)
        for thr in (2.5, 4.0):
            cands = signals_fixrev(m15[pair], PIP, thr)
            evaluate(f'fixrev(thr{thr})', pair, m15[pair], cands, 18,
                     is_start, oos_start, now, rows)

    print('\n=== 3. RV -- EURUSD vs GBPUSD relative value (EURGBP M15) ===')
    for window in (96, 480):
        for z in (2.0, 2.5):
            cands = signals_rv(m15['EURGBP'], PIP, SPREAD_PIPS['EURGBP'],
                               window, z)
            evaluate(f'rv(w{window},z{z})', 'EURGBP', m15['EURGBP'], cands,
                     None, is_start, oos_start, now, rows)

    print('\n=== 4. BBR -- vol-regime Bollinger fade (H1) ===')
    for pair in PAIRS:
        for pct in (0.4, 0.6):
            cands = signals_bbr(h1[pair], PIP, SPREAD_PIPS[pair], pct)
            evaluate(f'bbr(pct{pct})', pair, h1[pair], cands, 20,
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
