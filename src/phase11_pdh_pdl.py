"""
Forex Bot - Phase 11: Previous Day High/Low reactions (London/NY)
====================================================================
The one genuinely untested hypothesis from the master research prompt
(2026-08-11) after mapping its 5 strategy families against this
project's existing record: families 1-4 (Asian-range breakout, London
trend-pullback, London->NY continuation, Asian sweep-reversal) are all
duplicates of hypotheses already tested to failure on these exact
pairs (phases 1, 3, 4, 5 -- see PROJECT_REPORT.md section 4). Family 5
-- reactions around the PREVIOUS COMPLETE DAY's high/low -- has not
been tested as its own hypothesis in this project.

Universe: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF, NZDUSD (the
prompt's named 7 pairs; USDCHF is new to this project).

SESSION WINDOWS: this script uses the project's OWN established
server-time session convention (see PROJECT_REPORT.md section 2.3:
London ~07-16 server, NY ~12-21 server) rather than the master
prompt's suggested UTC windows. This is a deliberate choice, not an
oversight: this project already paid a real cost (a live 3-week bug,
see the "server-time fix" in this project's history) for using a
session-window convention that didn't match MT5's actual bar
timestamps. Re-introducing a second, different convention here would
risk exactly that class of error again.

METHODOLOGY (per the master prompt's own process):
  STEP 1-2: descriptive analysis of the PDH/PDL phenomenon FIRST, no
    trading -- frequency of session touches/breaks, penetration depth,
    close-back-inside rate, and (measured, not traded) forward
    continuation vs reversal probability.
  STEP 3: baseline strategies, objective and unoptimized -- breakout
    and rejection/reversal variants, ATR-scaled SL/TP, one trade/day,
    London and NY tested separately. 7 pairs x 2 sessions x 2 variants
    = 28 baseline experiments, each logged individually to the
    experiment ledger (experiments/experiments.csv) -- this is the
    project's standard first-pass market sweep, not a violation of
    "change one variable at a time" (that rule governs ITERATIVE
    refinement of a chosen candidate afterward).
  Any baseline that clears in-sample criteria proceeds to walk-forward
  + Monte Carlo (src/walk_forward.py, src/monte_carlo.py) before any
  "candidate" label is applied.

Criteria (same as every phase in this project): PF > 1.3, DD < 8%,
>= 60% profitable months in-sample; positive out-of-sample.

Requirements: MetaTrader 5 OPEN and LOGGED IN.
Output: data/phase11_descriptive.csv, data/phase11_results.csv,
        experiments/experiments.csv (appended)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy_matrix_backtest import (
    MONTHS, IS_MONTHS, CRIT_PF, CRIT_MAX_DD, CRIT_PROF_MONTHS,
    windowed_atr, run_sim, metrics, REPO_ROOT,
)
from phase3_session_structure_search import fetch_tf
from walk_forward import run_walk_forward
from monte_carlo import run_monte_carlo
from research_ledger import log_experiment, next_experiment_id

PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD']
SPREAD_PIPS = {'EURUSD': 1.0, 'GBPUSD': 1.2, 'USDJPY': 1.0, 'AUDUSD': 1.0,
               'USDCAD': 1.5, 'USDCHF': 1.5, 'NZDUSD': 1.5}

# server-hour session windows (project convention, see docstring)
SESSIONS = {
    'london': (7, 16),
    'ny':     (12, 21),
}

OUT_DESC    = REPO_ROOT / 'data' / 'phase11_descriptive.csv'
OUT_RESULTS = REPO_ROOT / 'data' / 'phase11_results.csv'
TRADES_DIR  = REPO_ROOT / 'data' / 'phase11_trades'


def pip_size(pair: str) -> float:
    return 0.01 if 'JPY' in pair else 0.0001


def prev_day_levels(h1: pd.DataFrame) -> pd.DataFrame:
    """Per-H1-bar previous COMPLETE server-day High/Low/Close/Range,
    using only fully-closed prior days (never the current day's own
    developing range -- no lookahead)."""
    d = pd.DataFrame({
        'High': h1['High'].resample('1D').max(),
        'Low':  h1['Low'].resample('1D').min(),
        'Close': h1['Close'].resample('1D').last(),
    }).dropna()
    pdh = d['High'].shift(1)
    pdl = d['Low'].shift(1)
    pdc = d['Close'].shift(1)
    prange = (pdh - pdl)
    daily = pd.DataFrame({'PDH': pdh, 'PDL': pdl, 'PDC': pdc,
                          'PRange': prange}).dropna()
    # broadcast onto every H1 bar of that calendar day
    out = daily.reindex(h1.index.normalize())
    out.index = h1.index
    return out


# ── STEP 1-2: descriptive analysis (no trading) ─────────────────────────────

def descriptive_analysis(pair: str, h1: pd.DataFrame) -> list:
    pip = pip_size(pair)
    lv = prev_day_levels(h1)
    closes = h1['Close'].to_numpy()
    highs  = h1['High'].to_numpy()
    lows   = h1['Low'].to_numpy()
    hours  = h1.index.hour
    pdh, pdl, prange = lv['PDH'].to_numpy(), lv['PDL'].to_numpy(), lv['PRange'].to_numpy()

    rows = []
    for sess_name, (s_start, s_end) in SESSIONS.items():
        touches_high = touches_low = 0
        closes_beyond_high = closes_beyond_low = 0
        reclaims_high = reclaims_low = 0     # touched beyond then closed back inside same bar
        penetration_high, penetration_low = [], []
        fwd_ret_after_break_high, fwd_ret_after_break_low = [], []
        fwd_ret_after_reclaim_high, fwd_ret_after_reclaim_low = [], []
        n_valid = 0
        FWD = 8   # bars ahead to measure forward drift (H1 -> 8h)

        for i in range(len(h1) - FWD):
            if not (s_start <= hours[i] < s_end) or np.isnan(pdh[i]) or prange[i] <= 0:
                continue
            n_valid += 1
            c = closes[i]
            if highs[i] > pdh[i]:
                touches_high += 1
                pen = (highs[i] - pdh[i]) / pip
                penetration_high.append(pen)
                if c > pdh[i]:
                    closes_beyond_high += 1
                    fwd_ret_after_break_high.append((closes[i + FWD] - c) / pip)
                else:
                    reclaims_high += 1
                    fwd_ret_after_reclaim_high.append((c - closes[i + FWD]) / pip)
            if lows[i] < pdl[i]:
                touches_low += 1
                pen = (pdl[i] - lows[i]) / pip
                penetration_low.append(pen)
                if c < pdl[i]:
                    closes_beyond_low += 1
                    fwd_ret_after_break_low.append((c - closes[i + FWD]) / pip)
                else:
                    reclaims_low += 1
                    fwd_ret_after_reclaim_low.append((closes[i + FWD] - c) / pip)

        def summarize(label, side, touches, beyond, reclaims, pen, fwd_break, fwd_reclaim):
            rows.append({
                'pair': pair, 'session': sess_name, 'side': side,
                'valid_bars': n_valid, 'touches': touches,
                'touch_rate_pct': round(touches / n_valid * 100, 2) if n_valid else 0,
                'closes_beyond': beyond,
                'reclaims_same_bar': reclaims,
                'reclaim_rate_of_touches_pct': round(reclaims / touches * 100, 1) if touches else 0,
                'median_penetration_pips': round(float(np.median(pen)), 1) if pen else 0,
                'fwd8h_median_pips_after_break': round(float(np.median(fwd_break)), 2) if fwd_break else 0,
                'fwd8h_positive_pct_after_break': round(float(np.mean(np.array(fwd_break) > 0) * 100), 1) if fwd_break else 0,
                'fwd8h_median_pips_after_reclaim': round(float(np.median(fwd_reclaim)), 2) if fwd_reclaim else 0,
                'fwd8h_positive_pct_after_reclaim': round(float(np.mean(np.array(fwd_reclaim) > 0) * 100), 1) if fwd_reclaim else 0,
            })

        summarize('PDH', 'high', touches_high, closes_beyond_high, reclaims_high,
                  penetration_high, fwd_ret_after_break_high, fwd_ret_after_reclaim_high)
        summarize('PDL', 'low', touches_low, closes_beyond_low, reclaims_low,
                  penetration_low, fwd_ret_after_break_low, fwd_ret_after_reclaim_low)
    return rows


# ── STEP 3: baseline strategies ─────────────────────────────────────────────

def signals_breakout(h1, pip, spread, session_hours, sl_atr=1.5, tp_atr=2.5):
    s_start, s_end = session_hours
    lv = prev_day_levels(h1)
    pdh, pdl = lv['PDH'].to_numpy(), lv['PDL'].to_numpy()
    closes = h1['Close'].to_numpy()
    highs  = h1['High'].to_numpy()
    lows   = h1['Low'].to_numpy()
    hours  = h1.index.hour
    dates  = h1.index.date
    atr = windowed_atr(highs, lows, closes, 14, 66) / pip
    min_tp = max(3.0, 2.0 * spread)

    out = []
    cur_date, done = None, False
    for i in range(66, len(h1)):
        if dates[i] != cur_date:
            cur_date, done = dates[i], False
        if done or not (s_start <= hours[i] < s_end):
            continue
        if np.isnan(pdh[i]) or np.isnan(atr[i]) or atr[i] <= 0:
            continue
        c = closes[i]
        sig = None
        if c > pdh[i]:
            sig = 'BUY'
        elif c < pdl[i]:
            sig = 'SELL'
        if sig:
            sl_pips, tp_pips = sl_atr * atr[i], tp_atr * atr[i]
            if tp_pips >= min_tp:
                out.append((i, sig, sl_pips, tp_pips))
                done = True
    return out


def signals_rejection(h1, pip, spread, session_hours, sl_buffer_atr=0.3,
                      tp_atr=2.0, k_bars=3):
    """Price closes beyond PDH/PDL, then closes back inside within
    k_bars -> fade the failed break. SL beyond the extreme reached
    while outside the level (+ a small ATR buffer), TP by ATR
    multiple. Mirrors the sweep-and-reclaim logic already used
    elsewhere in this project (phase 4's LFB), but referenced to the
    previous FULL DAY's level rather than the Asian range."""
    s_start, s_end = session_hours
    lv = prev_day_levels(h1)
    pdh, pdl = lv['PDH'].to_numpy(), lv['PDL'].to_numpy()
    closes = h1['Close'].to_numpy()
    highs  = h1['High'].to_numpy()
    lows   = h1['Low'].to_numpy()
    hours  = h1.index.hour
    dates  = h1.index.date
    atr = windowed_atr(highs, lows, closes, 14, 66) / pip
    min_tp = max(3.0, 2.0 * spread)

    out = []
    cur_date = None
    up_state = dn_state = None    # None | (break_idx, extreme) | 'done'
    for i in range(66, len(h1)):
        if dates[i] != cur_date:
            cur_date, up_state, dn_state = dates[i], None, None
        if np.isnan(pdh[i]) or np.isnan(atr[i]) or atr[i] <= 0:
            continue
        in_sess = s_start <= hours[i] < s_end
        c = closes[i]

        if up_state is None:
            if in_sess and c > pdh[i]:
                up_state = (i, highs[i])
        elif up_state != 'done':
            bidx, ext = up_state
            ext = max(ext, highs[i])
            if c < pdh[i]:
                sl_pips = (ext - c) / pip + sl_buffer_atr * atr[i]
                tp_pips = tp_atr * atr[i]
                if tp_pips >= min_tp:
                    out.append((i, 'SELL', sl_pips, tp_pips))
                up_state = 'done'
            elif i - bidx >= k_bars or not in_sess:
                up_state = 'done'
            else:
                up_state = (bidx, ext)

        if dn_state is None:
            if in_sess and c < pdl[i]:
                dn_state = (i, lows[i])
        elif dn_state != 'done':
            bidx, ext = dn_state
            ext = min(ext, lows[i])
            if c > pdl[i]:
                sl_pips = (c - ext) / pip + sl_buffer_atr * atr[i]
                tp_pips = tp_atr * atr[i]
                if tp_pips >= min_tp:
                    out.append((i, 'BUY', sl_pips, tp_pips))
                dn_state = 'done'
            elif i - bidx >= k_bars or not in_sess:
                dn_state = 'done'
            else:
                dn_state = (bidx, ext)
    return sorted(out)


def evaluate_and_log(family, hypothesis, label, pair, session, h1,
                     cands, is_start, oos_start, now, rows):
    pip = pip_size(pair)
    spread = SPREAD_PIPS[pair]
    tdf, eq = run_sim(h1, cands, pip, spread, 0.005)
    mi = metrics(tdf, eq, is_start, oos_start)
    mo = metrics(tdf, eq, oos_start, now)
    is_pass = (mi.get('trades', 0) >= 30 and mi.get('pf', 0) > CRIT_PF
               and mi.get('max_dd_pct', 99) < CRIT_MAX_DD
               and mi.get('prof_months_pct', 0) >= CRIT_PROF_MONTHS)
    passed = (is_pass and mo.get('trades', 0) >= 10 and mo.get('pnl', 0) > 0
              and mo.get('max_dd_pct', 99) < CRIT_MAX_DD)

    row = {'run': label, 'pair': pair, 'session': session,
          'is_trades': mi.get('trades', 0), 'is_wr': mi.get('win_rate', 0),
          'is_pf': mi.get('pf', 0), 'is_pnl': mi.get('pnl', 0),
          'is_dd': mi.get('max_dd_pct', 0),
          'is_prof_months': mi.get('prof_months_pct', 0),
          'oos_trades': mo.get('trades', 0), 'oos_pf': mo.get('pf', 0),
          'oos_pnl': mo.get('pnl', 0), 'oos_dd': mo.get('max_dd_pct', 0),
          'IS_PASS': is_pass, 'PASS': passed}
    rows.append(row)

    eid = next_experiment_id()
    log_experiment(eid, family=family, hypothesis=hypothesis, pair=pair,
                   session=session, timeframe='H1',
                   dataset_start=h1.index[0].date(), dataset_end=h1.index[-1].date(),
                   is_metrics=mi, oos_metrics=mo,
                   decision=('KEEP' if passed else 'INVESTIGATE' if is_pass else 'REJECT'),
                   status=('PROMISING' if passed else 'OVERFIT_RISK' if is_pass else 'FAILED'),
                   notes=label)

    if not tdf.empty:
        TRADES_DIR.mkdir(parents=True, exist_ok=True)
        safe = f'{label}_{pair}_{session}'.replace('/', '_')
        tdf.to_csv(TRADES_DIR / f'{safe}.csv', index=False)

    tag = 'PASS' if passed else ('is-ok' if is_pass else 'fail ')
    print(f"[{tag}] {eid} {label:<12} {pair:7} {session:6}  "
          f"IS: n={row['is_trades']:>3} PF={row['is_pf']:<5} "
          f"DD={row['is_dd']:>5.2f}% pm={row['is_prof_months']:>5.1f}%  "
          f"OOS: n={row['oos_trades']:>3} PF={row['oos_pf']:<5} "
          f"P&L=${row['oos_pnl']:>+8,.0f} DD={row['oos_dd']:.2f}%")
    return tdf, eq


def main():
    now = pd.Timestamp.now(tz='UTC')
    oos_start = now - pd.DateOffset(months=MONTHS - IS_MONTHS)
    is_start  = now - pd.DateOffset(months=MONTHS)

    print('========= PHASE 11: PREVIOUS DAY HIGH/LOW (London/NY) =========\n')
    data = {}
    for p in PAIRS:
        print(f'Fetching {p} H1 ...')
        data[p] = fetch_tf(p, 'H1')

    # ---- STEP 1-2: descriptive analysis (no trading) ----
    print('\n=== STEP 1-2: descriptive analysis (measuring, not trading) ===')
    desc_rows = []
    for p in PAIRS:
        desc_rows.extend(descriptive_analysis(p, data[p]))
    desc_df = pd.DataFrame(desc_rows)
    desc_df.to_csv(OUT_DESC, index=False)
    pd.set_option('display.width', 200)
    print(desc_df.to_string(index=False))
    print(f'\nSaved: {OUT_DESC}')

    # ---- STEP 3: baseline strategies ----
    print('\n=== STEP 3: baseline backtests (breakout + rejection) ===')
    result_rows = []
    for p in PAIRS:
        h1 = data[p]
        for sess_name, hrs in SESSIONS.items():
            cands_bo = signals_breakout(h1, pip_size(p), SPREAD_PIPS[p], hrs)
            evaluate_and_log('pdh_pdl_breakout',
                             'PDH/PDL breakout: H1 close beyond the previous '
                             'complete day\'s high/low during session hours '
                             'continues in that direction',
                             'pdh_breakout', p, sess_name, h1, cands_bo,
                             is_start, oos_start, now, result_rows)

            cands_rj = signals_rejection(h1, pip_size(p), SPREAD_PIPS[p], hrs)
            evaluate_and_log('pdh_pdl_rejection',
                             'PDH/PDL rejection: a break of the previous '
                             'day\'s high/low that closes back inside within '
                             '3 bars reverses',
                             'pdh_rejection', p, sess_name, h1, cands_rj,
                             is_start, oos_start, now, result_rows)

    df = pd.DataFrame(result_rows)
    df.to_csv(OUT_RESULTS, index=False)
    print(f'\nSaved: {OUT_RESULTS}')
    print(f"Full passes: {int(df['PASS'].sum())} / {len(df)}")
    print(f"IS-only passes: {int(df['IS_PASS'].sum())} / {len(df)}")

    # ---- deep validation for anything that passed IS ----
    candidates = df[df['IS_PASS']]
    if candidates.empty:
        print('\nNo candidate cleared in-sample criteria -- no walk-forward/'
             'Monte Carlo run. This is a valid research outcome (see report).')
        return

    print(f'\n=== DEEP VALIDATION: {len(candidates)} IS-passing candidate(s) ===')
    for _, row in candidates.iterrows():
        p, sess, label = row['pair'], row['session'], row['run']
        h1 = data[p]
        hrs = SESSIONS[sess]
        fn = signals_breakout if label == 'pdh_breakout' else signals_rejection
        cands = fn(h1, pip_size(p), SPREAD_PIPS[p], hrs)
        wf = run_walk_forward(h1, cands, pip_size(p), SPREAD_PIPS[p], 0.005,
                              train_months=12, test_months=6)
        tdf, eq = run_sim(h1, cands, pip_size(p), SPREAD_PIPS[p], 0.005)
        mc = run_monte_carlo(tdf, initial_balance=100_000)

        print(f'\n--- {label} {p} {sess} ---')
        print(wf.to_string(index=False))
        print(f"walk-forward: {wf.attrs.get('n_folds')} folds, "
              f"{wf.attrs.get('consistency_pct')}% profitable, "
              f"{wf.attrs.get('data_constraint_note')}")
        print(f"Monte Carlo: risk_of_ruin={mc.get('risk_of_ruin_pct')}%  "
              f"median_max_dd={mc.get('mc_max_dd_p50_pct')}%  "
              f"worst5%_max_dd={mc.get('mc_max_dd_worst_pct')}%  "
              f"losing_streak p50/p95={mc.get('worst_losing_streak_p50')}/"
              f"{mc.get('worst_losing_streak_p95')}")

        eid = next_experiment_id()
        log_experiment(eid, family=f'pdh_pdl_{label.split("_")[1]}_deepval',
                       hypothesis=f'walk-forward + Monte Carlo validation of {label} {p} {sess}',
                       pair=p, session=sess,
                       dataset_start=h1.index[0].date(), dataset_end=h1.index[-1].date(),
                       wf_folds=wf.attrs.get('n_folds'),
                       wf_consistency_pct=wf.attrs.get('consistency_pct'),
                       mc_risk_of_ruin_pct=mc.get('risk_of_ruin_pct'),
                       decision='INVESTIGATE', status='VALIDATION',
                       notes=f'follow-up to baseline experiment for {label} {p} {sess}')


if __name__ == '__main__':
    main()
