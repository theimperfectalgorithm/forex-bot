"""
Forex Bot - Phase 2: MeanReversion sweep + AsianRangeBreakout grid search
==========================================================================
Follow-up to strategy_matrix_backtest.py (which produced 0/33 passes).

1. MeanReversion (implemented, never validated) across all 9 majors --
   the documented next candidate after an empty Saturday GBPUSD search.
2. Bounded grid search on AsianRangeBreakout (the most structurally
   consistent family from phase 1: positive OOS on 4/6 pairs), varying
   TP multiplier and the H4 trend filter. Selection uses IN-SAMPLE data
   only; the out-of-sample window is touched exactly once per variant
   for confirmation -- no reselection on OOS results.
3. Fragility probe on momentum_divergence_session GBPUSD (the lone
   in-sample pass in phase 1): neighbouring parameter values. If the IS
   edge disappears with small parameter shifts, the phase-1 IS pass was
   noise, corroborating its OOS failure.

Same engine, criteria, spread and sizing assumptions as phase 1.
Requirements: MetaTrader 5 OPEN and LOGGED IN.

Output: data/phase2_results.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy_matrix_backtest import (
    MAJORS, SPREAD_PIPS, RISK_PCT_DEFAULT, MONTHS, IS_MONTHS,
    CRIT_PF, CRIT_MAX_DD, CRIT_PROF_MONTHS,
    windowed_ema, windowed_atr, windowed_rsi, h4_regime_series,
    fetch, run_sim, metrics, _self_check, REPO_ROOT,
)

OUT_CSV = REPO_ROOT / 'data' / 'phase2_results.csv'


# ── MEAN REVERSION (mirrors strategies/mean_reversion.py) ────────────────────

def h4_range_series(h1, h4, pip):
    """Per-H1-bar |H4 SMA50 - SMA200| in pips, using closed H4 bars only."""
    c = h4['Close'].to_numpy()
    sma50  = pd.Series(c).rolling(50).mean().to_numpy()
    sma200 = pd.Series(c).rolling(200).mean().to_numpy()
    gap = np.abs(sma50 - sma200) / pip
    h4_close_time = h4.index + pd.Timedelta(hours=4)
    h1_close_time = h1.index + pd.Timedelta(hours=1)
    idx = np.searchsorted(h4_close_time.values, h1_close_time.values,
                          side='right') - 1
    out = np.full(len(h1), np.nan)
    ok = idx >= 0
    out[ok] = gap[idx[ok]]
    return out


def signals_mrev(h1, h4, pip, range_thresh_pips,
                 rsi_lo=30, rsi_hi=70, sl_mult=1.5):
    """RSI(14) reversion to H1 SMA20 during a ranging H4, London 08-16 UTC.
    TP = SMA20 at signal; SL = sl_mult x entry-to-TP distance, opposite side.
    Returns (bar_idx, dir, sl_pips, tp_pips) with per-trade dynamic SL/TP."""
    W = 21                                       # H1_BARS_NEEDED
    closes = h1['Close'].to_numpy()
    rsi   = windowed_rsi(closes, 14, W)
    sma20 = pd.Series(closes).rolling(20).mean().to_numpy()
    h4gap = h4_range_series(h1, h4, pip)
    hours = h1.index.hour
    out = []
    for i in range(W, len(h1)):
        if not (8 <= hours[i] < 16):
            continue
        if np.isnan(rsi[i]) or np.isnan(sma20[i]) or np.isnan(h4gap[i]):
            continue
        if h4gap[i] >= range_thresh_pips:        # H4 trending -> no trade
            continue
        c = closes[i]
        if rsi[i] < rsi_lo and sma20[i] > c:     # BUY, revert up to mean
            tp_pips = (sma20[i] - c) / pip
            out.append((i, 'BUY', tp_pips * sl_mult, tp_pips))
        elif rsi[i] > rsi_hi and sma20[i] < c:   # SELL, revert down to mean
            tp_pips = (c - sma20[i]) / pip
            out.append((i, 'SELL', tp_pips * sl_mult, tp_pips))
    return out


# ── ARB with parameters (grid version of phase-1 signals_arb) ───────────────

def signals_arb_p(h1, h4, pip, tp_mult, use_h4, min_range=10):
    closes = h1['Close'].to_numpy()
    highs  = h1['High'].to_numpy()
    lows   = h1['Low'].to_numpy()
    regime = h4_regime_series(h1, h4) if use_h4 else None
    hours = h1.index.hour
    dates = h1.index.date
    out = []
    cur_date, a_hi, a_lo = None, None, None
    for i in range(len(h1)):
        d, hr = dates[i], hours[i]
        if d != cur_date:
            cur_date, a_hi, a_lo = d, None, None
        if hr < 7:
            a_hi = highs[i] if a_hi is None else max(a_hi, highs[i])
            a_lo = lows[i]  if a_lo is None else min(a_lo, lows[i])
            continue
        if hr in (7, 8) and a_hi is not None:
            rng_pips = (a_hi - a_lo) / pip
            if rng_pips < min_range:
                continue
            c = closes[i]
            if c > a_hi and (not use_h4 or regime[i] == 1):
                out.append((i, 'BUY', (c - a_lo) / pip, rng_pips * tp_mult))
            elif c < a_lo and (not use_h4 or regime[i] == -1):
                out.append((i, 'SELL', (a_hi - c) / pip, rng_pips * tp_mult))
    return out


# ── MDS fragility probe (GBPUSD only) ────────────────────────────────────────

def signals_mds_p(h1, pip, sl_pips, tp_pips, lookback, rsi_lo, rsi_hi):
    GAP = 3
    W = lookback + 14 + 5
    closes = h1['Close'].to_numpy()
    highs  = h1['High'].to_numpy()
    lows   = h1['Low'].to_numpy()
    rsi = windowed_rsi(closes, 14, W)
    hours = h1.index.hour
    out = []
    for i in range(W + lookback, len(h1)):
        if not (12 <= hours[i] < 17):
            continue
        h = highs[i - lookback + 1:i + 1]
        l = lows[i - lookback + 1:i + 1]
        r = rsi[i - lookback + 1:i + 1]
        sh = [j for j in range(1, lookback - 1)
              if h[j] > h[j - 1] and h[j] > h[j + 1]]
        sl_ = [j for j in range(1, lookback - 1)
               if l[j] < l[j - 1] and l[j] < l[j + 1]]
        sig = None
        if len(sh) >= 2:
            i1, i2 = sh[-2], sh[-1]
            if (i2 - i1 >= GAP and h[i2] > h[i1]
                    and not np.isnan(r[i1]) and not np.isnan(r[i2])
                    and r[i1] > rsi_hi and r[i2] < r[i1]):
                sig = 'SELL'
        if sig is None and len(sl_) >= 2:
            i1, i2 = sl_[-2], sl_[-1]
            if (i2 - i1 >= GAP and l[i2] < l[i1]
                    and not np.isnan(r[i1]) and not np.isnan(r[i2])
                    and r[i1] < rsi_lo and r[i2] > r[i1]):
                sig = 'BUY'
        if sig:
            out.append((i, sig, sl_pips, tp_pips))
    return out


# ── RUN ──────────────────────────────────────────────────────────────────────

def evaluate(label, pair, h1, cands, is_start, oos_start, now, rows):
    pip = 0.01 if 'JPY' in pair else 0.0001
    tdf, eq = run_sim(h1, cands, pip, SPREAD_PIPS[pair], RISK_PCT_DEFAULT)
    m_is  = metrics(tdf, eq, is_start, oos_start)
    m_oos = metrics(tdf, eq, oos_start, now)
    passed = (
        m_is.get('trades', 0) >= 30
        and m_is.get('pf', 0) > CRIT_PF
        and m_is.get('max_dd_pct', 99) < CRIT_MAX_DD
        and m_is.get('prof_months_pct', 0) >= CRIT_PROF_MONTHS
        and m_oos.get('trades', 0) >= 10
        and m_oos.get('pnl', 0) > 0
        and m_oos.get('max_dd_pct', 99) < CRIT_MAX_DD
    )
    is_pass_only = (
        m_is.get('trades', 0) >= 30
        and m_is.get('pf', 0) > CRIT_PF
        and m_is.get('max_dd_pct', 99) < CRIT_MAX_DD
        and m_is.get('prof_months_pct', 0) >= CRIT_PROF_MONTHS
    )
    row = {
        'run': label, 'pair': pair,
        'is_trades': m_is.get('trades', 0), 'is_wr': m_is.get('win_rate', 0),
        'is_pf': m_is.get('pf', 0), 'is_pnl': m_is.get('pnl', 0),
        'is_dd': m_is.get('max_dd_pct', 0),
        'is_prof_months': m_is.get('prof_months_pct', 0),
        'oos_trades': m_oos.get('trades', 0), 'oos_pf': m_oos.get('pf', 0),
        'oos_pnl': m_oos.get('pnl', 0), 'oos_dd': m_oos.get('max_dd_pct', 0),
        'IS_PASS': is_pass_only, 'PASS': passed,
    }
    rows.append(row)
    tag = 'PASS' if passed else ('is-ok' if is_pass_only else 'fail ')
    print(f"[{tag}] {label:<34} {pair}  "
          f"IS: n={row['is_trades']:>3} PF={row['is_pf']:<5} "
          f"DD={row['is_dd']:>5.2f}% pm={row['is_prof_months']:>5.1f}%  "
          f"OOS: n={row['oos_trades']:>3} PF={row['oos_pf']:<5} "
          f"P&L=${row['oos_pnl']:>+9,.2f} DD={row['oos_dd']:.2f}%")
    return row


def main():
    _self_check()
    now = pd.Timestamp.now(tz='UTC')
    oos_start = now - pd.DateOffset(months=MONTHS - IS_MONTHS)
    is_start  = now - pd.DateOffset(months=MONTHS)

    pair_data, pair_atr = {}, {}
    for pair in MAJORS:
        print(f'Fetching {pair} ...')
        h1, h4 = fetch(pair)
        pip = 0.01 if 'JPY' in pair else 0.0001
        atr = windowed_atr(h1['High'].to_numpy(), h1['Low'].to_numpy(),
                           h1['Close'].to_numpy(), 14, 66) / pip
        pair_data[pair] = (h1, h4)
        pair_atr[pair] = float(np.nanmedian(atr))
    gbp_atr = pair_atr['GBPUSD']
    print()

    rows = []

    # ---- 1. MeanReversion across majors (threshold ATR-scaled; 20p @ GBPUSD)
    print('=== 1. MeanReversion sweep ===')
    for pair in MAJORS:
        h1, h4 = pair_data[pair]
        pip = 0.01 if 'JPY' in pair else 0.0001
        thresh = 20.0 * pair_atr[pair] / gbp_atr
        cands = signals_mrev(h1, h4, pip, thresh)
        evaluate('mean_reversion(t=%.0fp)' % thresh, pair, h1, cands,
                 is_start, oos_start, now, rows)

    # ---- 2. ARB grid: tp_mult x h4_filter (selection on IS only)
    print('\n=== 2. AsianRangeBreakout grid ===')
    ARB_PAIRS = ['USDJPY', 'AUDJPY', 'GBPJPY', 'EURJPY', 'AUDUSD', 'NZDUSD']
    for pair in ARB_PAIRS:
        h1, h4 = pair_data[pair]
        pip = 0.01 if 'JPY' in pair else 0.0001
        for tp_mult in (1.0, 1.5, 2.0, 2.5):
            for use_h4 in (True, False):
                label = f'arb(tp={tp_mult},h4={"Y" if use_h4 else "N"})'
                cands = signals_arb_p(h1, h4, pip, tp_mult, use_h4)
                evaluate(label, pair, h1, cands, is_start, oos_start, now, rows)

    # ---- 3. MDS GBPUSD fragility probe
    print('\n=== 3. MDS GBPUSD fragility probe (phase-1 IS pass) ===')
    h1, h4 = pair_data['GBPUSD']
    pip = 0.0001
    for lookback, rlo, rhi in [(30, 35, 65),   # phase-1 params (reference)
                               (25, 35, 65), (35, 35, 65),
                               (30, 30, 70), (30, 40, 60)]:
        label = f'mds(lb={lookback},rsi={rlo}/{rhi})'
        cands = signals_mds_p(h1, pip, 25, 50, lookback, rlo, rhi)
        evaluate(label, 'GBPUSD', h1, cands, is_start, oos_start, now, rows)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f'\nSaved: {OUT_CSV}')
    print(f"\nFull passes (IS + OOS): {int(df['PASS'].sum())} / {len(df)}")
    print(f"IS-only passes:         {int(df['IS_PASS'].sum())} / {len(df)}")
    if df['PASS'].any():
        print(df[df['PASS']].to_string(index=False))


if __name__ == '__main__':
    main()
