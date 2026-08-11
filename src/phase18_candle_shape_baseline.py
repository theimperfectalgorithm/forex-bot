"""
Forex Bot - Phase 18: T0 Candle-Shape Baseline (controlled trading experiment)
==================================================================================
ONE controlled baseline experiment testing whether the phase-17 T0-available
candle-shape signal (smaller body, more wick, higher close location on
reversal-outcome down-candles) improves the economics of the phase-15
failed baseline. This is NOT optimization -- no parameter is searched.

Three baselines, identical trade mechanics, differing only in which events
qualify and which direction is traded:
  A. UNFILTERED    -- all >=1.0 ATR down-moves, fade (BUY), the phase-15 control
  B. REVERSAL-SHAPE -- only events with shape_score >= pooled median, fade (BUY)
  C. CONTINUATION-SHAPE CONTROL -- only events with shape_score < pooled
     median, trade continuation (SELL)

Shape definitions (frozen BEFORE backtesting, Part 1):
  body_ratio        = |close-open| / (high-low)
  close_location     = (close-low) / (high-low)
  lower_wick_ratio    = (min(open,close)-low) / (high-low)
  upper_wick_ratio    = (high-max(open,close)) / (high-low)
  shape_score = close_location - body_ratio   (frozen composite -- see report
    Part 1 for the reasoning: body_ratio+upper_wick_ratio+lower_wick_ratio=1
    identically, so these four are not independent; phase17 found
    close_location and body_ratio to be the two cleanest, least-redundant
    Tier-1 effects (d=+0.101 and d=-0.107), so the composite is built from
    those two rather than all four to avoid double-counting the wick split.)
Threshold: the POOLED HISTORICAL MEDIAN of shape_score across all >=1.0 ATR
down-move events in the research sample -- NOT searched for backtest
performance, computed once from the full population before any trade
simulation, same methodology this project has already used for volatility-
regime terciles in phases 3b/16/17.

No existing strategy (ARB/AMR/Monday-drift/XAUUSD ARB) is read, imported,
or modified. No parameter is searched at any point in this script.

Requirements: MetaTrader 5 OPEN and LOGGED IN (or CSV cache).
Output: reports/phase18_baseline_log.txt, data/phase18_*.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy_matrix_backtest import REPO_ROOT
from phase15_downmove_reversion_baseline import (
    PAIRS, ASIAN, LONDON, OVERLAP, NY, fetch_m15, SPREAD_PIPS_NORMAL,
)
from phase16_downmove_mechanism import PairData, THR, HORIZON

SL_ATR = 1.0
TP_R = 1.0

LOG: list[str] = []


def say(msg=''):
    print(msg)
    LOG.append(str(msg))


# ═══════════════════════════════════════════════════════════════════════
# PART 1 -- FROZEN SHAPE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════

def compute_shape(highs, lows, opens, closes):
    rng = np.maximum(highs - lows, 1e-9)
    body_ratio = np.abs(closes - opens) / rng
    close_location = (closes - lows) / rng
    lower_wick_ratio = (np.minimum(opens, closes) - lows) / rng
    upper_wick_ratio = (highs - np.maximum(opens, closes)) / rng
    shape_score = close_location - body_ratio
    return body_ratio, close_location, lower_wick_ratio, upper_wick_ratio, shape_score


# ═══════════════════════════════════════════════════════════════════════
# BUILD EVENT TABLE (down-moves, frozen from phase15/16/17, + shape fields)
# ═══════════════════════════════════════════════════════════════════════

def build_events(pdz: PairData) -> pd.DataFrame:
    n = len(pdz.closes)
    closes, highs, lows, opens, atr, pip = (pdz.closes, pdz.highs, pdz.lows,
                                             pdz.df['Open'].to_numpy(), pdz.atr, pdz.pip)
    fwd_close = np.roll(closes, -HORIZON); fwd_close[-HORIZON:] = np.nan
    fwd_atr = (fwd_close - closes) / pip / np.where(atr > 0, atr, np.nan)
    down = (pdz.move_atr <= -THR) & ~np.isnan(fwd_atr)
    idxs = np.where(down)[0]
    idxs = idxs[(idxs >= 40) & (idxs < n - HORIZON - 2)]
    if len(idxs) == 0:
        return pd.DataFrame()

    body_ratio, close_location, lower_wick_ratio, upper_wick_ratio, shape_score = compute_shape(
        highs[idxs], lows[idxs], opens[idxs], closes[idxs])

    return pd.DataFrame(dict(
        pair=pdz.pair, i=idxs, time=pdz.idx[idxs], year=pdz.year[idxs],
        dow=pdz.dow[idxs], session=pdz.session[idxs],
        atr_at_event=atr[idxs],
        fwd_atr=fwd_atr[idxs],
        outcome=np.where(fwd_atr[idxs] >= 0, 'REVERSAL', 'CONTINUATION'),
        body_ratio=body_ratio, close_location=close_location,
        lower_wick_ratio=lower_wick_ratio, upper_wick_ratio=upper_wick_ratio,
        shape_score=shape_score,
    ))


# ═══════════════════════════════════════════════════════════════════════
# TRADE SIMULATOR (generalized direction, mirrors phase15's run_baseline_fade)
# ═══════════════════════════════════════════════════════════════════════

def simulate(pdz: PairData, event_idxs: np.ndarray, direction: str,
             spread_pips=None, entry_delay_bars=0) -> pd.DataFrame:
    spread = spread_pips if spread_pips is not None else SPREAD_PIPS_NORMAL.get(pdz.pair, 1.5)
    closes, highs, lows, opens, atr, pip = (pdz.closes, pdz.highs, pdz.lows,
                                             pdz.df['Open'].to_numpy(), pdz.atr, pdz.pip)
    n = len(closes)
    trades = []
    event_set = sorted(event_idxs)
    used_until = -1
    for i in event_set:
        if i <= used_until:
            continue  # no overlapping trades, mirrors phase15
        entry_i = i + 1 + entry_delay_bars
        if entry_i >= n - HORIZON - 1 or np.isnan(atr[i]) or atr[i] <= 0:
            continue
        sl_dist = SL_ATR * atr[i] * pip
        if direction == 'BUY':
            entry_px = opens[entry_i] + spread * pip
            sl_px = entry_px - sl_dist
            tp_px = entry_px + TP_R * sl_dist
        else:  # SELL
            entry_px = opens[entry_i] - spread * pip
            sl_px = entry_px + sl_dist
            tp_px = entry_px - TP_R * sl_dist
        exit_px, exit_reason = None, None
        mfe, mae = 0.0, 0.0
        for b in range(entry_i, min(entry_i + HORIZON, n)):
            lo, hi = lows[b], highs[b]
            if direction == 'BUY':
                mfe = max(mfe, (hi - entry_px) / pip)
                mae = min(mae, (lo - entry_px) / pip)
                if lo <= sl_px:
                    exit_px, exit_reason = sl_px, 'SL'; break
                if hi >= tp_px:
                    exit_px, exit_reason = tp_px, 'TP'; break
            else:
                mfe = max(mfe, (entry_px - lo) / pip)
                mae = min(mae, (entry_px - hi) / pip)
                if hi >= sl_px:
                    exit_px, exit_reason = sl_px, 'SL'; break
                if lo <= tp_px:
                    exit_px, exit_reason = tp_px, 'TP'; break
        if exit_px is None:
            last_b = min(entry_i + HORIZON - 1, n - 1)
            exit_px, exit_reason = closes[last_b], 'TIME'
        pnl_pips = (exit_px - entry_px) / pip if direction == 'BUY' else (entry_px - exit_px) / pip
        r_multiple = pnl_pips / (sl_dist / pip)
        trades.append(dict(pair=pdz.pair, entry_i=entry_i, year=pdz.year[i], session=pdz.session[i],
                            pnl_pips=pnl_pips, r_multiple=r_multiple, reason=exit_reason,
                            mfe_pips=mfe, mae_pips=mae))
        used_until = entry_i + HORIZON - 1
    return pd.DataFrame(trades)


def summarize(tdf: pd.DataFrame) -> dict:
    if tdf.empty:
        return dict(n=0, win_rate=np.nan, pf=np.nan, expectancy_pips=np.nan, mean_r=np.nan,
                    median_r=np.nan, mean_mfe=np.nan, mean_mae=np.nan)
    wins = tdf[tdf.pnl_pips > 0]['pnl_pips'].sum()
    losses = -tdf[tdf.pnl_pips < 0]['pnl_pips'].sum()
    pf = wins / losses if losses > 0 else np.nan
    return dict(n=len(tdf), win_rate=float((tdf.pnl_pips > 0).mean()), pf=float(pf),
                expectancy_pips=float(tdf.pnl_pips.mean()), mean_r=float(tdf.r_multiple.mean()),
                median_r=float(tdf.r_multiple.median()), mean_mfe=float(tdf.mfe_pips.mean()),
                mean_mae=float(tdf.mae_pips.mean()))


def bootstrap_ci_diff(a: pd.Series, b: pd.Series, n_boot=2000, seed=41):
    """Bootstrap CI on mean(a) - mean(b), resampling each independently."""
    rng = np.random.default_rng(seed)
    a_arr, b_arr = a.to_numpy(), b.to_numpy()
    diffs = np.array([
        rng.choice(a_arr, size=len(a_arr), replace=True).mean() -
        rng.choice(b_arr, size=len(b_arr), replace=True).mean()
        for _ in range(n_boot)
    ])
    return dict(mean_diff=float(diffs.mean()), ci_low=float(np.percentile(diffs, 2.5)),
                ci_high=float(np.percentile(diffs, 97.5)),
                pct_above_zero=float((diffs > 0).mean()))


def main():
    say('=' * 90)
    say('PHASE 18 -- T0 CANDLE-SHAPE BASELINE (controlled trading experiment, NOT optimization)')
    say('=' * 90)
    say('PART 1 -- FROZEN SHAPE DEFINITIONS')
    say('  body_ratio        = |close-open| / (high-low)')
    say('  close_location    = (close-low) / (high-low)')
    say('  lower_wick_ratio  = (min(open,close)-low) / (high-low)')
    say('  upper_wick_ratio  = (high-max(open,close)) / (high-low)')
    say('  shape_score = close_location - body_ratio  (composite, frozen BEFORE this backtest --')
    say('    body_ratio+upper_wick_ratio+lower_wick_ratio=1 identically, so these 4 quantities are')
    say('    not independent; phase17 found close_location (d=+0.101) and body_ratio (d=-0.107) to')
    say('    be the two cleanest, least-redundant Tier-1 effects, so the composite uses only those two.')
    say('  Threshold = the POOLED HISTORICAL MEDIAN of shape_score across the full research sample,')
    say('    computed ONCE before any trade simulation -- not searched for backtest performance.')
    say('  Mechanics (identical across A/B/C, unchanged from phase15): entry = next candle open,')
    say(f'  stop = {SL_ATR}xATR, target = {TP_R}R, max hold = {HORIZON} bars (60min), SL priority on same-bar touch.')

    all_events = []
    pd_map = {}
    for pair in PAIRS:
        try:
            m15 = fetch_m15(pair)
        except Exception as e:
            say(f'{pair}: SKIP ({e})'); continue
        if len(m15) < 3000:
            say(f'{pair}: SKIP (insufficient data)'); continue
        pdz = PairData(pair, m15)
        pd_map[pair] = pdz
        ev = build_events(pdz)
        all_events.append(ev)
    events = pd.concat(all_events, ignore_index=True)
    median_shape = events['shape_score'].median()
    say(f'\nPooled events: {len(events)}. Frozen shape_score median threshold = {median_shape:.4f}.')

    events['is_reversal_shape'] = events['shape_score'] >= median_shape

    say('\nInformation-timing check (Part 4): shape_score uses ONLY the event bar (bar i) OHLC --')
    say('open/high/low/close of bar i are all known at the close of bar i (=T0) by construction.')
    say('No future candle, volatility, session, or range information is used in the shape rule.')

    # ---- assign each event to a baseline population + direction ----
    idx_A = events.index                                    # all events, BUY
    idx_B = events.index[events.is_reversal_shape]           # reversal-shape only, BUY
    idx_C = events.index[~events.is_reversal_shape]          # continuation-shape only, SELL

    say(f'\nBaseline A (unfiltered): {len(idx_A)} events, direction=BUY (fade)')
    say(f'Baseline B (reversal-shape >= median): {len(idx_B)} events, direction=BUY (fade)')
    say(f'Baseline C (continuation-shape < median): {len(idx_C)} events, direction=SELL (continuation)')

    def run_all(idx_subset, direction, spread_mult=1.0, delay=0):
        all_t = []
        for pair, g in events.loc[idx_subset].groupby('pair'):
            pdz = pd_map[pair]
            spread = SPREAD_PIPS_NORMAL.get(pair, 1.5) * spread_mult
            t = simulate(pdz, g['i'].to_numpy(), direction, spread_pips=spread, entry_delay_bars=delay)
            all_t.append(t)
        return pd.concat(all_t, ignore_index=True) if all_t else pd.DataFrame()

    say('\n' + '=' * 90)
    say('BASELINES A / B / C -- NORMAL SPREAD, NO DELAY')
    say('=' * 90)
    trades_A = run_all(idx_A, 'BUY')
    trades_B = run_all(idx_B, 'BUY')
    trades_C = run_all(idx_C, 'SELL')
    for label, t in [('A_UNFILTERED', trades_A), ('B_REVERSAL_SHAPE', trades_B), ('C_CONTINUATION_SHAPE', trades_C)]:
        s = summarize(t)
        say(f'  {label}: {s}')

    say('\n' + '=' * 90)
    say('PART 3 -- POPULATION-LEVEL COMPARISON (reversal/continuation probability by shape group)')
    say('=' * 90)
    say(events.groupby('is_reversal_shape')['outcome'].value_counts(normalize=True).to_string())

    say('\n' + '=' * 90)
    say('PART 5 -- PAIR-LEVEL RESULTS')
    say('=' * 90)
    for label, idx_subset, direction in [('A', idx_A, 'BUY'), ('B', idx_B, 'BUY'), ('C', idx_C, 'SELL')]:
        say(f'-- Baseline {label} --')
        rows = []
        for pair in PAIRS:
            g = events.loc[idx_subset]
            g = g[g.pair == pair]
            if g.empty:
                continue
            pdz = pd_map[pair]
            t = simulate(pdz, g['i'].to_numpy(), direction)
            s = summarize(t)
            rows.append(dict(pair=pair, **s))
        say(pd.DataFrame(rows).to_string(index=False))

    say('\n' + '=' * 90)
    say('PART 6 -- YEAR-LEVEL RESULTS')
    say('=' * 90)
    for label, idx_subset, direction, tdf in [('A', idx_A, 'BUY', trades_A), ('B', idx_B, 'BUY', trades_B),
                                               ('C', idx_C, 'SELL', trades_C)]:
        say(f'-- Baseline {label} --')
        rows = []
        for yr in [2023, 2024, 2025, 2026]:
            sub = tdf[tdf.year == yr]
            if sub.empty:
                continue
            rows.append(dict(year=yr, **summarize(sub)))
        say(pd.DataFrame(rows).to_string(index=False))

    say('\n' + '=' * 90)
    say('PART 7 -- SESSION-LEVEL RESULTS')
    say('=' * 90)
    for label, tdf in [('A', trades_A), ('B', trades_B), ('C', trades_C)]:
        say(f'-- Baseline {label} --')
        rows = []
        for sess in [ASIAN, LONDON, OVERLAP, NY]:
            sub = tdf[tdf.session == sess]
            if sub.empty:
                continue
            rows.append(dict(session=sess, **summarize(sub)))
        say(pd.DataFrame(rows).to_string(index=False))

    say('\n' + '=' * 90)
    say('PART 8 -- COST STRESS (pooled)')
    say('=' * 90)
    for label, idx_subset, direction in [('A', idx_A, 'BUY'), ('B', idx_B, 'BUY'), ('C', idx_C, 'SELL')]:
        say(f'-- Baseline {label} --')
        rows = []
        for scen, mult, delay in [('normal', 1.0, 0), ('1.5x_spread', 1.5, 0),
                                   ('2x_spread', 2.0, 0), ('1bar_delay', 1.0, 1)]:
            t = run_all(idx_subset, direction, spread_mult=mult, delay=delay)
            rows.append(dict(scenario=scen, **summarize(t)))
        say(pd.DataFrame(rows).to_string(index=False))

    say('\n' + '=' * 90)
    say('PART 9 -- STATISTICAL COMPARISON: B vs A (bootstrap CI on expectancy and mean-R difference)')
    say('=' * 90)
    exp_ci = bootstrap_ci_diff(trades_B['pnl_pips'], trades_A['pnl_pips'])
    r_ci = bootstrap_ci_diff(trades_B['r_multiple'], trades_A['r_multiple'])
    say(f'Expectancy (pips) B-A: mean_diff={exp_ci["mean_diff"]:+.4f}  '
        f'95% CI=[{exp_ci["ci_low"]:+.4f}, {exp_ci["ci_high"]:+.4f}]  '
        f'P(B>A)={exp_ci["pct_above_zero"]:.3f}')
    say(f'Mean R B-A: mean_diff={r_ci["mean_diff"]:+.4f}  '
        f'95% CI=[{r_ci["ci_low"]:+.4f}, {r_ci["ci_high"]:+.4f}]  '
        f'P(B>A)={r_ci["pct_above_zero"]:.3f}')
    rev_prob_A = (events.loc[idx_A, 'outcome'] == 'REVERSAL').mean()
    rev_prob_B = (events.loc[idx_B, 'outcome'] == 'REVERSAL').mean()
    say(f'Reversal probability: A={rev_prob_A:.4f}  B={rev_prob_B:.4f}  diff={rev_prob_B - rev_prob_A:+.4f}')
    say('Multiple-testing note: this is a single pre-registered filter-selection test (one composite')
    say('score, one frozen median threshold, one comparison A-vs-B-vs-C) -- not a search across many')
    say('candidate filters, so no additional multiple-testing correction beyond what phases 15-17')
    say('already applied when selecting body_ratio/close_location as the composite\'s inputs.')

    out_dir = REPO_ROOT / 'data'
    events.to_csv(out_dir / 'phase18_events.csv', index=False)
    trades_A.to_csv(out_dir / 'phase18_trades_A.csv', index=False)
    trades_B.to_csv(out_dir / 'phase18_trades_B.csv', index=False)
    trades_C.to_csv(out_dir / 'phase18_trades_C.csv', index=False)

    report_path = REPO_ROOT / 'reports' / 'phase18_baseline_log.txt'
    report_path.write_text('\n'.join(LOG), encoding='utf-8')
    say(f'\nFull log written to {report_path}')


if __name__ == '__main__':
    main()
