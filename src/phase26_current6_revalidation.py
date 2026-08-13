"""
Forex Bot - Phase 26: Current 6-Strategy Live Portfolio Revalidation
========================================================================
VALIDATION ONLY. No strategy created, optimized, tuned, deployed, or
modified. No live config changed. Frozen parameters throughout --
neighborhood sensitivity tests perturb inputs for DIAGNOSIS only, never
select a "best" value, and the live specification is never altered
based on any result in this script.

Covers the exact 6 strategies currently active on the 5ers account
(GBPJPY ARB and XAUUSD ARB excluded -- demoted 2026-07-31, per
reports/live_portfolio_validation_audit.md):
  CADJPY ARB (0.50%), GBPJPY/EURJPY/AUDJPY/CADJPY AMR (0.25% each),
  GBPUSD Monday Drift (0.25%).

Frozen live parameters (Step 0, verified directly against pairs/*.yaml
and strategies/*.py -- not from memory):
  CADJPY ARB:   signals_arb_p(tp_multiplier=2.0, use_h4=False), spread 2.0p, risk 0.50%
  GBPJPY AMR:   signals_amr_v(z_thr=2.5, sl_mult=1.25, end_hour=4), spread 2.0p, risk 0.25%
  EURJPY AMR:   signals_amr_v(z_thr=2.0, sl_mult=1.5,  end_hour=6), spread 2.0p, risk 0.25%
  AUDJPY AMR:   signals_amr_v(z_thr=2.0, sl_mult=1.5,  end_hour=4), spread 2.0p, risk 0.25%
  CADJPY AMR:   signals_amr_v(z_thr=2.0, sl_mult=1.5,  end_hour=4), spread 2.0p, risk 0.25%
  GBPUSD MONDAY: signals_monday(sl_mult=1.25, tp_mult=1.0), spread 1.2p, risk 0.25%, time_exit=21:00

Requirements: MetaTrader 5 OPEN and LOGGED IN (or CSV cache).
Output: reports/phase26_revalidation_log.txt, data/phase26_*.csv
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
from strategy_matrix_backtest import run_sim, windowed_atr, REPO_ROOT
from phase2_meanrev_arb_search import signals_arb_p
from phase3b_amr_jpy_refine import signals_amr_v
from phase8_monday_validation import signals_monday
from phase15_downmove_reversion_baseline import ASIAN, LONDON, OVERLAP, NY, session_of_hour

MONTHS = 36
MIN_SAMPLE = 20
N_MC_RUNS = 10_000
SEED = 97
STARTING_CAPITAL = 100_000.0

# ---- Step 0: frozen specification (verified against source, 2026-08-13) ----
SPECS = {
    'CADJPY_ARB': dict(family='ARB', pair='CADJPY', tf='H1', spread=2.0, risk=0.0050,
                        params=dict(tp_mult=2.0, use_h4=False, min_range=10)),
    'GBPJPY_AMR': dict(family='AMR', pair='GBPJPY', tf='M15', spread=2.0, risk=0.0025,
                        params=dict(z_thr=2.5, sl_mult=1.25, end_hour=4)),
    'EURJPY_AMR': dict(family='AMR', pair='EURJPY', tf='M15', spread=2.0, risk=0.0025,
                        params=dict(z_thr=2.0, sl_mult=1.5, end_hour=6)),
    'AUDJPY_AMR': dict(family='AMR', pair='AUDJPY', tf='M15', spread=2.0, risk=0.0025,
                        params=dict(z_thr=2.0, sl_mult=1.5, end_hour=4)),
    'CADJPY_AMR': dict(family='AMR', pair='CADJPY', tf='M15', spread=2.0, risk=0.0025,
                        params=dict(z_thr=2.0, sl_mult=1.5, end_hour=4)),
    'GBPUSD_MONDAY': dict(family='MONDAY', pair='GBPUSD', tf='H1', spread=1.2, risk=0.0025,
                           params=dict(sl_mult=1.25, tp_mult=1.0)),
}

LOG: list[str] = []


def say(msg=''):
    print(msg)
    LOG.append(str(msg))


def fetch(pair, tf):
    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=MONTHS * 30 + 30)
    return data_loader.get_bars(pair, tf, date_from, date_to)


def pip_of(pair):
    return 0.01 if pair.endswith('JPY') else 0.0001


def build_candidates(name, df, spec, override=None):
    p = dict(spec['params'])
    if override:
        p.update(override)
    pip = pip_of(spec['pair'])
    if spec['family'] == 'ARB':
        h4 = fetch(spec['pair'], 'H4')
        return signals_arb_p(df, h4, pip, p['tp_mult'], p['use_h4'], min_range=p.get('min_range', 10))
    if spec['family'] == 'AMR':
        return signals_amr_v(df, pip, spec['spread'], p['z_thr'], p['sl_mult'], p['end_hour'])
    if spec['family'] == 'MONDAY':
        return signals_monday(df, p['sl_mult'], p['tp_mult'])
    raise ValueError(name)


def run_variant(name, df, spec, cands, spread_mult=1.0, delay_bars=0):
    pip = pip_of(spec['pair'])
    kwargs = dict(spread_pips=spec['spread'] * spread_mult, risk_pct=spec['risk'])
    if spec['family'] == 'MONDAY':
        kwargs['time_exit_hour'] = 21
    if delay_bars:
        n = len(df)
        cands = [(min(i + delay_bars, n - 1), d, sl, tp) for i, d, sl, tp in cands]
    tdf, _ = run_sim(df, cands, pip, **kwargs)
    return tdf


def enrich(tdf, df, spec):
    if tdf.empty:
        return tdf
    tdf = tdf.copy()
    pip = pip_of(spec['pair'])
    highs, lows, closes = df['High'].to_numpy(), df['Low'].to_numpy(), df['Close'].to_numpy()
    atr = windowed_atr(highs, lows, closes, 14, 66) / pip
    atr_pctile = pd.Series(atr, index=df.index).rank(pct=True)
    er_net = np.abs(closes - np.roll(closes, 20)); er_net[:20] = np.nan
    diffs = np.abs(np.diff(closes, prepend=closes[0]))
    er_sum = pd.Series(diffs).rolling(20).sum().to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        er = er_net / er_sum
    er_series = pd.Series(er, index=df.index)

    idx_map = {t: i for i, t in enumerate(df.index)}
    entry_idx = tdf['entry_time'].map(idx_map)
    tdf['atr_pctile'] = entry_idx.map(pd.Series(atr_pctile.to_numpy()))
    tdf['efficiency_ratio'] = entry_idx.map(pd.Series(er_series.to_numpy()))
    tdf['year'] = tdf['entry_time'].dt.year
    tdf['session'] = tdf['entry_time'].apply(lambda t: session_of_hour(t.hour))
    tdf['dow'] = tdf['entry_time'].dt.day_name()
    tdf['hold_hours'] = (tdf['exit_time'] - tdf['entry_time']).dt.total_seconds() / 3600
    with np.errstate(divide='ignore', invalid='ignore'):
        tdf['r_multiple'] = np.where(tdf['sl_pips'] > 0, tdf['pips'] / tdf['sl_pips'], np.nan)
    tdf['vol_tercile'] = pd.cut(tdf['atr_pctile'], [0, 1/3, 2/3, 1.0001], labels=['LOW', 'NORMAL', 'HIGH'])
    er_valid = tdf['efficiency_ratio'].dropna()
    if len(er_valid) >= 30:
        terc = er_valid.quantile([1/3, 2/3]).to_numpy()
        tdf['trend_tercile'] = pd.cut(tdf['efficiency_ratio'], [-np.inf, terc[0], terc[1], np.inf],
                                       labels=['LOW_TREND', 'NORMAL_TREND', 'HIGH_TREND'])
    else:
        tdf['trend_tercile'] = np.nan
    return tdf


def summarize(tdf: pd.DataFrame) -> dict:
    if tdf.empty:
        return dict(n=0)
    wins = tdf[tdf.pnl > 0]['pnl'].sum()
    losses = -tdf[tdf.pnl < 0]['pnl'].sum()
    pf = wins / losses if losses > 0 else np.nan
    cum_r = tdf['r_multiple'].cumsum()
    dd_r = (cum_r - cum_r.cummax()).min()
    is_loss = (tdf['pnl'] < 0).to_numpy()
    is_win = (tdf['pnl'] > 0).to_numpy()
    max_ls, cur = 0, 0
    for x in is_loss:
        cur = cur + 1 if x else 0
        max_ls = max(max_ls, cur)
    max_ws, cur = 0, 0
    for x in is_win:
        cur = cur + 1 if x else 0
        max_ws = max(max_ws, cur)
    winners = tdf[tdf.pnl > 0]['pnl']
    losers = tdf[tdf.pnl < 0]['pnl']
    return dict(
        n=len(tdf), win_rate=float((tdf.pnl > 0).mean()), loss_rate=float((tdf.pnl < 0).mean()),
        pf=float(pf), expectancy_r=float(tdf['r_multiple'].mean()), avg_r=float(tdf['r_multiple'].mean()),
        median_r=float(tdf['r_multiple'].median()), total_r=float(tdf['r_multiple'].sum()),
        avg_winner=float(winners.mean()) if len(winners) else np.nan,
        avg_loser=float(losers.mean()) if len(losers) else np.nan,
        payoff_ratio=float(winners.mean() / -losers.mean()) if len(winners) and len(losers) else np.nan,
        max_dd_r=float(dd_r), max_losing_streak=int(max_ls), max_winning_streak=int(max_ws),
        avg_hold_hours=float(tdf['hold_hours'].mean()) if 'hold_hours' in tdf else np.nan,
        median_hold_hours=float(tdf['hold_hours'].median()) if 'hold_hours' in tdf else np.nan,
        total_pnl=float(tdf['pnl'].sum()),
    )


def bootstrap_ci(vals: pd.Series, n_boot=2000, seed=101):
    rng = np.random.default_rng(seed)
    arr = vals.dropna().to_numpy()
    if len(arr) < 10:
        return dict(mean=np.nan, ci_low=np.nan, ci_high=np.nan)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)])
    return dict(mean=float(arr.mean()), ci_low=float(np.percentile(means, 2.5)),
                ci_high=float(np.percentile(means, 97.5)))


def monte_carlo(tdf: pd.DataFrame, n_runs=N_MC_RUNS, seed=SEED):
    if len(tdf) < 30:
        return {}
    r = tdf['r_multiple'].to_numpy()
    pnl = tdf['pnl'].to_numpy()
    loss_mask = pnl < 0
    rng = np.random.default_rng(seed)
    n = len(pnl)
    mc_dd = np.empty(n_runs)
    mc_streak = np.empty(n_runs, dtype=int)
    for i in range(n_runs):
        perm = rng.permutation(n)
        cum = np.cumsum(r[perm])
        dd = cum - np.maximum.accumulate(cum)
        mc_dd[i] = -dd.min()
        streak = cur = 0
        for x in loss_mask[perm]:
            cur = cur + 1 if x else 0
            streak = max(streak, cur)
        mc_streak[i] = streak
    return dict(
        dd_p50=np.percentile(mc_dd, 50), dd_p75=np.percentile(mc_dd, 75),
        dd_p90=np.percentile(mc_dd, 90), dd_p95=np.percentile(mc_dd, 95), dd_p99=np.percentile(mc_dd, 99),
        streak_p50=np.percentile(mc_streak, 50), streak_p90=np.percentile(mc_streak, 90),
        streak_p95=np.percentile(mc_streak, 95), streak_p99=np.percentile(mc_streak, 99),
        actual_dd_r=-((tdf['r_multiple'].cumsum() - tdf['r_multiple'].cumsum().cummax()).min()),
    )


def main():
    say('=' * 90)
    say('PHASE 26 -- CURRENT 6-STRATEGY LIVE PORTFOLIO REVALIDATION (frozen params, no tuning)')
    say('=' * 90)
    say('STEP 0 -- FROZEN SPECIFICATION (verified directly against pairs/*.yaml, 2026-08-13):')
    for name, spec in SPECS.items():
        say(f'  {name}: {spec}')

    all_data = {}
    all_trades = {}
    for name, spec in SPECS.items():
        say(f'\nReconstructing {name} ...')
        df = fetch(spec['pair'], spec['tf'])
        cands = build_candidates(name, df, spec)
        tdf = run_variant(name, df, spec, cands)
        tdf = enrich(tdf, df, spec)
        all_data[name] = df
        all_trades[name] = tdf
        say(f'  {len(tdf)} trades, {tdf["entry_time"].min()} to {tdf["entry_time"].max()}' if not tdf.empty else '  NO TRADES')

    combined = pd.concat([t.assign(strategy=n) for n, t in all_trades.items() if not t.empty], ignore_index=True)
    combined.to_csv(REPO_ROOT / 'data' / 'phase26_all_trades.csv', index=False)

    # ---- STEP 3: historical baseline ----
    say('\n' + '=' * 90); say('STEP 3 -- HISTORICAL BASELINE (full reconstruction)'); say('=' * 90)
    baseline_stats = {}
    for name, tdf in all_trades.items():
        s = summarize(tdf)
        baseline_stats[name] = s
        say(f'\n-- {name} --')
        say(str(s))
        say('  Year-by-year:')
        for yr, g in tdf.groupby('year'):
            if len(g) >= 5:
                sy = summarize(g)
                say(f'    {yr}: n={sy["n"]} win_rate={sy["win_rate"]:.3f} pf={sy["pf"]:.3f} expectancy_r={sy["expectancy_r"]:+.4f} total_r={sy["total_r"]:+.2f}')

    # ---- STEP 4: OOS ----
    say('\n' + '=' * 90); say('STEP 4 -- TRUE OUT-OF-SAMPLE (chronological, frozen params, no re-fit)'); say('=' * 90)
    oos_stats = {}
    for name, tdf in all_trades.items():
        t0, t1 = tdf['entry_time'].min(), tdf['entry_time'].max()
        oos_start = t1 - pd.DateOffset(months=12)
        oos = tdf[tdf['entry_time'] >= oos_start]
        s = summarize(oos)
        oos_stats[name] = dict(oos_window_start=str(oos_start.date()), oos_window_end=str(t1.date()), **s)
        say(f'{name}: OOS window (trailing 12mo, {oos_start.date()} to {t1.date()}): {s}')
    say('\nNOTE: this is a RECOMPUTED trailing-12-month OOS window using the exact frozen live')
    say('parameters, run fresh in this script -- NOT the original discovery-time OOS figures')
    say('(which were never persisted at trade level). Labeled explicitly, not blended.')

    # ---- STEP 5: walk-forward ----
    say('\n' + '=' * 90); say('STEP 5 -- WALK-FORWARD (6-month rolling windows, frozen params)'); say('=' * 90)
    wf_stats = {}
    for name, tdf in all_trades.items():
        t0, t1 = tdf['entry_time'].min(), tdf['entry_time'].max()
        windows = pd.date_range(t0, t1 - pd.Timedelta(days=180), freq='90D')
        rows = []
        for ws in windows:
            we = ws + pd.Timedelta(days=180)
            sub = tdf[(tdf['entry_time'] >= ws) & (tdf['entry_time'] < we)]
            s = summarize(sub)
            rows.append(dict(window_start=str(ws.date()), window_end=str(we.date()), **s))
        wfdf = pd.DataFrame(rows)
        wf_stats[name] = wfdf
        pf_valid = wfdf[wfdf.n >= 10]['pf'].dropna()
        say(f'\n-- {name} -- {len(wfdf)} windows')
        say(wfdf[['window_start', 'window_end', 'n', 'win_rate', 'pf', 'expectancy_r', 'total_r']].to_string(index=False))
        if len(pf_valid):
            say(f'  % profitable windows (n>=10): {100*(pf_valid > 1.0).mean():.1f}%  median PF: {pf_valid.median():.3f}  '
                f'worst PF: {pf_valid.min():.3f}  best PF: {pf_valid.max():.3f}')

    # ---- STEP 6: cost stress ----
    say('\n' + '=' * 90); say('STEP 6 -- COST STRESS'); say('=' * 90)
    cost_stats = {}
    for name, spec in SPECS.items():
        df = all_data[name]
        cands = build_candidates(name, df, spec)
        rows = []
        for label, mult, delay in [('normal', 1.0, 0), ('1.5x_spread', 1.5, 0), ('2x_spread', 2.0, 0), ('1bar_delay', 1.0, 1)]:
            t = run_variant(name, df, spec, cands, spread_mult=mult, delay_bars=delay)
            t = enrich(t, df, spec)
            rows.append(dict(scenario=label, **summarize(t)))
        cdf = pd.DataFrame(rows)
        cost_stats[name] = cdf
        say(f'\n-- {name} --')
        say(cdf.to_string(index=False))
        pf_normal = cdf.iloc[0]['pf']
        pf_2x = cdf[cdf.scenario == '2x_spread']['pf'].iloc[0]
        if pf_2x >= 1.0:
            classification = 'ROBUST'
        elif pf_2x >= 0.9:
            classification = 'COST-SENSITIVE'
        else:
            classification = 'COST-FRAGILE'
        say(f'  Classification: {classification} (PF normal={pf_normal:.3f}, PF@2x={pf_2x:.3f})')
        cost_stats[name + '_class'] = classification

    # ---- STEP 7: parameter sensitivity ----
    say('\n' + '=' * 90); say('STEP 7 -- PARAMETER SENSITIVITY (small neighborhood, NOT optimization)'); say('=' * 90)
    param_stats = {}
    for name, spec in SPECS.items():
        df = all_data[name]
        p = spec['params']
        neighborhoods = {}
        if spec['family'] == 'ARB':
            neighborhoods['tp_mult'] = [round(p['tp_mult'] * f, 3) for f in [0.9, 0.95, 1.0, 1.05, 1.1]]
        elif spec['family'] == 'AMR':
            neighborhoods['z_thr'] = [round(p['z_thr'] * f, 3) for f in [0.9, 0.95, 1.0, 1.05, 1.1]]
            neighborhoods['sl_mult'] = [round(p['sl_mult'] * f, 3) for f in [0.9, 0.95, 1.0, 1.05, 1.1]]
        elif spec['family'] == 'MONDAY':
            neighborhoods['sl_mult'] = [round(p['sl_mult'] * f, 3) for f in [0.9, 0.95, 1.0, 1.05, 1.1]]
            neighborhoods['tp_mult'] = [round(p['tp_mult'] * f, 3) for f in [0.9, 0.95, 1.0, 1.05, 1.1]]
        say(f'\n-- {name} --')
        for pname, vals in neighborhoods.items():
            rows = []
            for v in vals:
                override = {pname: v}
                cands = build_candidates(name, df, spec, override=override)
                t = run_variant(name, df, spec, cands)
                t = enrich(t, df, spec)
                s = summarize(t)
                rows.append(dict(value=v, **s))
            pdf = pd.DataFrame(rows)
            param_stats[f'{name}_{pname}'] = pdf
            say(f'  {pname}: ' + ' | '.join(f'{r.value}=PF{r.pf:.3f}' for r in pdf.itertuples()))
            pf_vals = pdf['pf'].dropna()
            if len(pf_vals) == 5:
                frozen_pf = pf_vals.iloc[2]
                is_isolated = frozen_pf == pf_vals.max() and (pf_vals.max() - pf_vals.drop(2).max()) > 0.3
                say(f'    frozen={frozen_pf:.3f}  worst_neighbor={pf_vals.min():.3f}  best_neighbor={pf_vals.max():.3f}  '
                    f'isolated_peak={is_isolated}')

    # ---- STEP 8: regime analysis ----
    say('\n' + '=' * 90); say('STEP 8 -- REGIME ANALYSIS (diagnostic only)'); say('=' * 90)
    for name, tdf in all_trades.items():
        say(f'\n-- {name} --')
        say('  By volatility tercile:')
        for v in ['LOW', 'NORMAL', 'HIGH']:
            sub = tdf[tdf.vol_tercile == v]
            if len(sub) >= MIN_SAMPLE:
                s = summarize(sub)
                say(f'    {v}: n={s["n"]} pf={s["pf"]:.3f} expectancy_r={s["expectancy_r"]:+.4f}')
        if tdf['trend_tercile'].notna().sum() > 30:
            say('  By trend tercile:')
            for t in ['LOW_TREND', 'NORMAL_TREND', 'HIGH_TREND']:
                sub = tdf[tdf.trend_tercile == t]
                if len(sub) >= MIN_SAMPLE:
                    s = summarize(sub)
                    say(f'    {t}: n={s["n"]} pf={s["pf"]:.3f} expectancy_r={s["expectancy_r"]:+.4f}')
        say('  By session:')
        for sess, g in tdf.groupby('session'):
            if len(g) >= MIN_SAMPLE:
                s = summarize(g)
                say(f'    {sess}: n={s["n"]} pf={s["pf"]:.3f} expectancy_r={s["expectancy_r"]:+.4f}')
        say('  By day of week:')
        for d, g in tdf.groupby('dow'):
            if len(g) >= 10:
                s = summarize(g)
                say(f'    {d}: n={s["n"]} pf={s["pf"]:.3f} expectancy_r={s["expectancy_r"]:+.4f}')

    # ---- STEP 9: directional ----
    say('\n' + '=' * 90); say('STEP 9 -- DIRECTIONAL ANALYSIS (diagnostic only)'); say('=' * 90)
    for name, tdf in all_trades.items():
        say(f'\n-- {name} --')
        for d, g in tdf.groupby('dir'):
            s = summarize(g)
            say(f'  {d}: n={s["n"]} win_rate={s["win_rate"]:.3f} pf={s["pf"]:.3f} expectancy_r={s["expectancy_r"]:+.4f} total_r={s["total_r"]:+.2f} max_dd_r={s["max_dd_r"]:.2f}')

    # ---- STEP 10: Monte Carlo ----
    say('\n' + '=' * 90); say(f'STEP 10 -- MONTE CARLO ({N_MC_RUNS:,} runs per strategy)'); say('=' * 90)
    mc_stats = {}
    for name, tdf in all_trades.items():
        mc = monte_carlo(tdf)
        mc_stats[name] = mc
        say(f'{name}: {mc}')

    # ---- STEP 11: bootstrap ----
    say('\n' + '=' * 90); say('STEP 11 -- BOOTSTRAP CONFIDENCE INTERVALS'); say('=' * 90)
    boot_stats = {}
    for name, tdf in all_trades.items():
        ci_r = bootstrap_ci(tdf['r_multiple'])
        boot_stats[name] = ci_r
        crosses_zero = ci_r['ci_low'] <= 0 <= ci_r['ci_high']
        say(f'{name}: expectancy_r mean={ci_r["mean"]:+.4f}  95% CI=[{ci_r["ci_low"]:+.4f}, {ci_r["ci_high"]:+.4f}]  '
            f'crosses_zero={crosses_zero}')

    # ---- STEP 13: exit reason breakdown ----
    say('\n' + '=' * 90); say('STEP 13 -- EXIT REASON BREAKDOWN'); say('=' * 90)
    for name, tdf in all_trades.items():
        say(f'{name}: ' + str(tdf['reason'].value_counts().to_dict()))

    # ---- STEP 14: current 6-strategy portfolio reconstruction ----
    say('\n' + '=' * 90); say('STEP 14 -- CURRENT 6-STRATEGY PORTFOLIO RECONSTRUCTION'); say('=' * 90)
    port = combined.sort_values('entry_time').reset_index(drop=True)
    port['portfolio_equity'] = STARTING_CAPITAL + port['pnl'].cumsum()
    port['date'] = port['entry_time'].dt.date

    n_trades = len(port)
    wins = port[port.pnl > 0]['pnl'].sum(); losses = -port[port.pnl < 0]['pnl'].sum()
    port_pf = wins / losses if losses > 0 else np.nan
    port_exp_r = port['r_multiple'].mean()
    port_total_r = port['r_multiple'].sum()
    eq = port['portfolio_equity']
    dd_pct = 100 * (eq - eq.cummax()) / eq.cummax()
    port_max_dd_pct = dd_pct.min()
    is_loss = (port['pnl'] < 0).to_numpy()
    port_max_streak, cur = 0, 0
    for x in is_loss:
        cur = cur + 1 if x else 0
        port_max_streak = max(port_max_streak, cur)

    say(f'Total trades: {n_trades}  PF: {port_pf:.3f}  Expectancy: {port_exp_r:+.4f}R  Total R: {port_total_r:+.2f}')
    say(f'Max DD: {port_max_dd_pct:.2f}%   Max losing streak: {port_max_streak}')

    port_mc = monte_carlo(port.rename(columns={'r_multiple': 'r_multiple', 'pnl': 'pnl'}))
    say(f'Portfolio Monte Carlo: {port_mc}')

    # JPY exposure (all 6 current strategies are JPY-exposed except GBPUSD Monday)
    jpy_strats = ['CADJPY_ARB', 'GBPJPY_AMR', 'EURJPY_AMR', 'AUDJPY_AMR', 'CADJPY_AMR']
    port['is_jpy'] = port['strategy'].isin(jpy_strats)
    pct_trades_jpy = 100 * port['is_jpy'].mean()
    weights = {n: s['risk'] for n, s in SPECS.items()}
    port['weight'] = port['strategy'].map(weights)
    pct_risk_jpy = 100 * port.loc[port.is_jpy, 'weight'].sum() / port['weight'].sum()
    say(f'\n% trades JPY-exposed: {pct_trades_jpy:.1f}%   % risk-weight JPY-exposed: {pct_risk_jpy:.1f}%')

    jpy_daily = port[port.is_jpy].pivot_table(index='date', columns='strategy', values='pnl', aggfunc='sum').fillna(0)
    corr = jpy_daily.corr()
    say('\nJPY daily P&L correlation matrix:')
    say(corr.round(3).to_string())
    n_losing = (jpy_daily < 0).sum(axis=1)
    say(f'\nDays with 2+ JPY strategies losing: {(n_losing >= 2).sum()} ({100*(n_losing>=2).mean():.1f}%)')
    say(f'Days with 3+ JPY strategies losing: {(n_losing >= 3).sum()} ({100*(n_losing>=3).mean():.1f}%)')
    jpy_day_total = jpy_daily.sum(axis=1)
    worst_day = jpy_day_total.idxmin()
    say(f'Worst clustered JPY loss day: {worst_day}, ${jpy_day_total.min():+,.2f}')

    port.to_csv(REPO_ROOT / 'data' / 'phase26_portfolio.csv', index=False)

    report_path = REPO_ROOT / 'reports' / 'phase26_revalidation_log.txt'
    report_path.write_text('\n'.join(LOG), encoding='utf-8')
    say(f'\nFull log written to {report_path}')

    return dict(baseline=baseline_stats, oos=oos_stats, mc=mc_stats, boot=boot_stats,
                port_pf=port_pf, port_exp_r=port_exp_r, port_total_r=port_total_r,
                port_max_dd_pct=port_max_dd_pct, port_max_streak=port_max_streak, port_mc=port_mc)


if __name__ == '__main__':
    main()
