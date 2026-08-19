"""
Phase 46 -- current six-strategy robustness audit, applying the same
gates used for Phase33+ research candidates to the six LIVE strategies,
wherever the existing trade-level ledger supports it. Parameter
perturbation and cost-stress re-simulation are explicitly out of scope
(see reports/phase46_preregistration.md section 4) -- NOT fabricated.
No strategy modified, no repair, no live change.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).parent.parent
OUT = REPO / 'reports'
RNG = np.random.default_rng(20261101)

STRATS = ['AUDJPY_AMR', 'CADJPY_AMR', 'EURJPY_AMR', 'GBPJPY_AMR', 'CADJPY_ARB', 'GBPUSD_MONDAY']
TRAIN_START = pd.Timestamp('2023-08-01', tz='UTC')
TRAIN_END = pd.Timestamp('2024-08-31', tz='UTC')
VAL_START = pd.Timestamp('2024-09-01', tz='UTC')
VAL_END = pd.Timestamp('2025-04-30', tz='UTC')
OOS_START = pd.Timestamp('2025-05-01', tz='UTC')
OOS_END = pd.Timestamp('2026-08-13', tz='UTC')


def load_control():
    df = pd.read_csv(REPO / 'data' / 'phase26_all_trades.csv')
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df['trade_date'] = df['entry_time'].dt.date
    return df


def edge_metrics(sub):
    if len(sub) == 0:
        return {'trades': 0, 'win_rate_pct': None, 'pf': None, 'expectancy_R': None, 'total_R': None}
    r = sub['r_multiple']
    wins, losses = r[r > 0], r[r < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else np.nan
    return {'trades': len(sub), 'win_rate_pct': round((r > 0).mean() * 100, 1),
            'pf': round(pf, 3) if pf == pf else None, 'expectancy_R': round(r.mean(), 4),
            'total_R': round(r.sum(), 2)}


def max_streak(r):
    s = ms = 0
    for v in r:
        if v < 0: s += 1; ms = max(ms, s)
        else: s = 0
    return ms


def dd_of(r):
    cum = np.cumsum(r)
    return float((cum - np.maximum.accumulate(cum)).min()) if len(r) else None


def main():
    df = load_control()
    print(f"[control] {len(df)} trades, reconciled: {len(df) == 2712}")

    # ============ Gate 1: OOS results ============
    oos_rows = []
    for s in STRATS:
        sub = df[df.strategy == s]
        oos = sub[(sub.entry_time >= OOS_START) & (sub.entry_time <= OOS_END)]
        m = edge_metrics(oos)
        gate1 = 'PASS' if (m['pf'] is not None and m['pf'] > 1.0) else ('INSUFFICIENT SAMPLE' if m['trades'] < 30 else 'FAIL')
        oos_rows.append({'strategy': s, **m, 'max_dd_R': round(dd_of(oos['r_multiple'].values), 2) if len(oos) else None,
                          'max_losing_streak': max_streak(oos['r_multiple'].tolist()), 'gate1_classification': gate1})
    oos_df = pd.DataFrame(oos_rows)
    oos_df.to_csv(OUT / 'phase46_oos_results.csv', index=False)
    print("\n[OOS results]"); print(oos_df.to_string())

    # ============ OOS sub-half ============
    subhalf_rows = []
    for s in STRATS:
        sub = df[df.strategy == s]
        oos = sub[(sub.entry_time >= OOS_START) & (sub.entry_time <= OOS_END)]
        if len(oos) < 4:
            subhalf_rows.append({'strategy': s, 'verdict': 'INSUFFICIENT SAMPLE'})
            continue
        mid = oos['entry_time'].median()
        h1, h2 = oos[oos.entry_time < mid], oos[oos.entry_time >= mid]
        m1, m2 = edge_metrics(h1), edge_metrics(h2)
        consistent = (m1['expectancy_R'] or 0) * (m2['expectancy_R'] or 0) > 0
        verdict = 'PASS' if consistent else ('WARNING (n<40)' if len(oos) < 40 else 'FAIL')
        subhalf_rows.append({'strategy': s, 'h1_trades': m1['trades'], 'h1_expectancy_R': m1['expectancy_R'], 'h1_pf': m1['pf'],
                              'h2_trades': m2['trades'], 'h2_expectancy_R': m2['expectancy_R'], 'h2_pf': m2['pf'],
                              'sign_consistent': consistent, 'total_oos_trades': len(oos), 'verdict': verdict})
    subhalf_df = pd.DataFrame(subhalf_rows)
    subhalf_df.to_csv(OUT / 'phase46_oos_subhalf.csv', index=False)
    print("\n[OOS sub-half]"); print(subhalf_df.to_string())

    # ============ Parameter perturbation / stability -- INSUFFICIENT DATA (disclosed) ============
    pert_df = pd.DataFrame([{'strategy': s, 'status': 'INSUFFICIENT DATA / REQUIRES NEW RE-EXECUTION INFRASTRUCTURE',
                              'reason': 'The historical ledger stores only already-executed outcomes, not a re-runnable backtest engine bound to live price history -- see phase46_preregistration.md section 4. Not fabricated.'} for s in STRATS])
    pert_df.to_csv(OUT / 'phase46_parameter_perturbation.csv', index=False)
    stab_df = pd.DataFrame([{'strategy': s, 'plateau_classification': 'E. INSUFFICIENT DATA',
                              'informal_prior_evidence': 'See phase46_strategy_definitions.csv -- AMR pairs cite a 36-variant informal grid (Phase3/3b) reported parameter-insensitive; GBPUSD Monday cites an informal SL/TP grid (OOS PF 2.9-3.1); these predate and are methodologically DIFFERENT from the frozen +/-20% single-perturbation standard and are not treated as equivalent evidence'} for s in STRATS])
    stab_df.to_csv(OUT / 'phase46_parameter_stability.csv', index=False)
    print("\n[parameter perturbation/stability] INSUFFICIENT DATA -- disclosed scope limitation, see preregistration section 4")

    # ============ Cost stress -- NOT COMPUTABLE (disclosed) ============
    cost_df = pd.DataFrame([{'strategy': s, 'status': 'NOT COMPUTABLE FROM THIS DATASET',
                              'reason': 'r_multiple/pnl do not separately expose each trades cost component; same disclosed limitation as Phase44 (reports/phase44_cost_sensitivity.csv)'} for s in STRATS])
    cost_df.to_csv(OUT / 'phase46_cost_stress.csv', index=False)

    # ============ Regime robustness ============
    periods = {
        'A_2019_2020': None, 'B_2021_2022': None,
        'C_2023_2024': (pd.Timestamp('2023-08-01', tz='UTC'), pd.Timestamp('2024-12-31', tz='UTC')),
        'D_2025': (pd.Timestamp('2025-01-01', tz='UTC'), pd.Timestamp('2025-12-31', tz='UTC')),
        'E_2026_YTD': (pd.Timestamp('2026-01-01', tz='UTC'), pd.Timestamp('2026-08-13', tz='UTC')),
    }
    regime_rows = []
    for s in STRATS:
        sub = df[df.strategy == s]
        for pname, rng in periods.items():
            if rng is None:
                regime_rows.append({'strategy': s, 'period': pname, 'note': 'UNKNOWN BY DATA ABSENCE'})
                continue
            psub = sub[(sub.entry_time >= rng[0]) & (sub.entry_time <= rng[1])]
            m = edge_metrics(psub)
            regime_rows.append({'strategy': s, 'period': pname, **m})
    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(OUT / 'phase46_regime_robustness.csv', index=False)
    print("\n[regime robustness] (sample)"); print(regime_df[regime_df.period == 'E_2026_YTD'].to_string())

    # ============ Volatility behavior (reuse vol_tercile) ============
    df_v = df.dropna(subset=['vol_tercile'])
    vol_rows = []
    for s in STRATS:
        sub = df_v[df_v.strategy == s]
        for state in ['LOW', 'NORMAL', 'HIGH']:
            ssub = sub[sub.vol_tercile == state]
            m = edge_metrics(ssub)
            vol_rows.append({'strategy': s, 'vol_state': state, **m})
    vol_df = pd.DataFrame(vol_rows)
    vol_df.to_csv(OUT / 'phase46_volatility_behavior.csv', index=False)
    print("\n[volatility behavior] (HIGH state)"); print(vol_df[vol_df.vol_state == 'HIGH'].to_string())

    # ============ Drawdown correlation (reuse Phase41's OOS-window-matched methodology) ============
    hist = df.copy()
    hist['trade_date'] = hist['entry_time'].dt.date
    daily_control = hist.groupby('trade_date')['r_multiple'].sum().rename('control_R')
    oos_start_date = OOS_START.date()
    daily_control_oos = daily_control[daily_control.index >= oos_start_date]
    cum = daily_control_oos.cumsum()
    dd = cum - cum.cummax()
    dd_thresh = dd.quantile(0.10)
    dd_days = set(dd[dd <= dd_thresh].index)

    ddcorr_rows = []
    for s in STRATS:
        sub = df[(df.strategy == s) & (df.entry_time >= OOS_START)]
        sub = sub.copy(); sub['trade_date'] = sub['entry_time'].dt.date
        daily_s = sub.groupby('trade_date')['r_multiple'].sum().rename('strategy_R')
        # exclude the strategy's own contribution from control to avoid trivial self-correlation
        control_excl = (daily_control_oos - daily_s.reindex(daily_control_oos.index).fillna(0)).rename('control_R')
        merged = pd.concat([control_excl, daily_s], axis=1).dropna()
        merged['is_dd'] = merged.index.isin(dd_days)
        normal_corr = merged.loc[~merged.is_dd, ['control_R', 'strategy_R']].corr().iloc[0, 1] if (~merged.is_dd).sum() > 5 else None
        n_dd = int(merged.is_dd.sum())
        dd_corr = merged.loc[merged.is_dd, ['control_R', 'strategy_R']].corr().iloc[0, 1] if n_dd >= 8 else None
        cls = ('UNKNOWN (n<8 overlap)' if dd_corr is None else
               'STRONG DIVERSIFIER' if normal_corr is not None and dd_corr <= normal_corr else
               'NEUTRAL' if normal_corr is not None and dd_corr <= normal_corr + 0.15 else 'CORRELATED')
        ddcorr_rows.append({'strategy': s, 'overlapping_days': len(merged), 'normal_day_corr': round(normal_corr, 3) if normal_corr is not None else None,
                             'n_dd_days_overlap': n_dd, 'dd_day_corr': round(dd_corr, 3) if dd_corr is not None else None, 'classification': cls})
    ddcorr_df = pd.DataFrame(ddcorr_rows)
    ddcorr_df.to_csv(OUT / 'phase46_drawdown_correlation.csv', index=False)
    print("\n[drawdown correlation]"); print(ddcorr_df.to_string())

    # ============ Portfolio integration (leave-one-out) ============
    daily_full = daily_control
    def metrics_of(series):
        c = series.cumsum()
        ddser = c - c.cummax()
        return {'total_R': round(series.sum(), 2), 'max_dd': round(ddser.min(), 2)}
    full_m = metrics_of(daily_full)
    loo_rows = [{'configuration': 'FULL_SIX_STRATEGY_CONTROL', **full_m}]
    for s in STRATS:
        s_daily = df[df.strategy == s].groupby(df[df.strategy == s].entry_time.dt.date)['r_multiple'].sum()
        without = (daily_full - s_daily.reindex(daily_full.index).fillna(0))
        m = metrics_of(without)
        loo_rows.append({'configuration': f'WITHOUT_{s}', **m})
    loo_df = pd.DataFrame(loo_rows)
    loo_df.to_csv(OUT / 'phase46_portfolio_integration.csv', index=False)
    print("\n[portfolio integration -- leave-one-out]"); print(loo_df.to_string())

    # ============ Strategy contribution ============
    contrib_rows = []
    total_R = df['r_multiple'].sum()
    for s in STRATS:
        sub = df[df.strategy == s]
        oos_sub = sub[sub.entry_time >= OOS_START]
        contrib_rows.append({
            'strategy': s, 'total_R': round(sub['r_multiple'].sum(), 2), 'pct_of_portfolio_R': round(sub['r_multiple'].sum() / total_R * 100, 1),
            'trade_count': len(sub), 'oos_R': round(oos_sub['r_multiple'].sum(), 2),
        })
    contrib_df = pd.DataFrame(contrib_rows).sort_values('pct_of_portfolio_R', ascending=False)
    contrib_df.to_csv(OUT / 'phase46_strategy_contribution.csv', index=False)
    print("\n[strategy contribution]"); print(contrib_df.to_string())

    # ============ Monte Carlo ============
    mc_rows = []
    for s in STRATS:
        sub = df[df.strategy == s]
        oos = sub[(sub.entry_time >= OOS_START)]
        r_arr = oos['r_multiple'].values
        if len(r_arr) < 10:
            mc_rows.append({'strategy': s, 'n_sims': 0, 'note': 'insufficient trades'})
            continue
        mc_dds = []
        for _ in range(10000):
            shuf = RNG.permutation(r_arr)
            cum = np.cumsum(shuf)
            mc_dds.append((cum - np.maximum.accumulate(cum)).min())
        mc_dds = np.array(mc_dds)
        actual_dd = dd_of(r_arr)
        mc_rows.append({'strategy': s, 'n_sims': 10000, 'n_oos_trades': len(r_arr), 'data_type': 'SIMULATED',
                         'actual_max_dd_R': round(actual_dd, 2), 'mc_dd_median': round(np.median(mc_dds), 2),
                         'mc_dd_p95': round(np.percentile(mc_dds, 95), 2),
                         'actual_dd_percentile_in_mc': round(float((mc_dds < actual_dd).mean() * 100), 1)})
    mc_df = pd.DataFrame(mc_rows)
    mc_df.to_csv(OUT / 'phase46_monte_carlo.csv', index=False)
    print("\n[Monte Carlo]"); print(mc_df.to_string())

    # ============ Live comparison + sufficiency (reuse Phase45 methodology exactly) ============
    live = pd.read_csv(REPO / 'reports' / '5ers_trade_export.csv')
    live['entry_time'] = pd.to_datetime(live['entry_time'], errors='coerce')
    live['R'] = pd.to_numeric(live['R'], errors='coerce')
    live_closed = live[live['status'] == 'CLOSED'].copy()
    live_closed['strategy_norm'] = live_closed['strategy'].apply(lambda s: 'GBPUSD_MONDAY' if s == 'GBPUSD_MON' else s)
    post_demo = live_closed[live_closed['entry_time'] >= pd.Timestamp('2026-07-31', tz='UTC')]

    livecmp_rows, suff_rows = [], []
    for s in STRATS:
        hsub = df[df.strategy == s]
        lsub = post_demo[post_demo.strategy_norm == s]
        n_l = len(lsub)
        livecmp_rows.append({'strategy': s, 'historical_expectancy_R': round(hsub['r_multiple'].mean(), 4),
                              'live_n_trades': n_l, 'live_total_R': round(lsub['R'].sum(), 3) if n_l else 0,
                              'live_expectancy_R': round(lsub['R'].mean(), 4) if n_l else None,
                              'live_win_rate_pct': round((lsub['R'] > 0).mean() * 100, 1) if n_l else None})
        hist_r = hsub.sort_values('entry_time')['r_multiple'].values
        if n_l > 0 and n_l < len(hist_r):
            boot = []
            for _ in range(10000):
                start = RNG.integers(0, len(hist_r) - n_l)
                boot.append(hist_r[start:start + n_l].sum())
            boot = np.array(boot)
            live_total = lsub['R'].sum()
            pctile = float((boot < live_total).mean() * 100)
            interp = ('CONSISTENT' if 25 <= pctile <= 75 else 'UNUSUAL BUT NOT DECISIVE' if 5 <= pctile < 25 or 75 < pctile <= 95 else
                      'POSSIBLE DETERIORATION (extreme tail)' if pctile < 5 else 'POSSIBLE OUTPERFORMANCE (extreme tail)')
            suff_rows.append({'strategy': s, 'live_n_trades': n_l, 'live_total_R': round(live_total, 3),
                               'bootstrap_median_R': round(np.median(boot), 3), 'live_percentile_in_bootstrap': round(pctile, 1),
                               'classification': interp})
        else:
            suff_rows.append({'strategy': s, 'live_n_trades': n_l, 'classification': 'INSUFFICIENT SAMPLE'})
    livecmp_df = pd.DataFrame(livecmp_rows)
    livecmp_df.to_csv(OUT / 'phase46_live_comparison.csv', index=False)
    suff_df = pd.DataFrame(suff_rows)
    suff_df.to_csv(OUT / 'phase46_live_sample_sufficiency.csv', index=False)
    print("\n[live comparison]"); print(livecmp_df.to_string())
    print("\n[live sample sufficiency]"); print(suff_df.to_string())

    # ============ Candidate comparison vs Phase33-40 ============
    ledger = pd.read_csv(OUT / 'phase45_research_master_ledger.csv')
    confirm = ledger[~ledger['notes'].astype(str).str.lower().str.contains('screen')]
    candidate_pf_values = pd.to_numeric(confirm['oos_pf'], errors='coerce').dropna()
    candidate_gate1_pass_rate = (candidate_pf_values > 1.0).mean() * 100
    cand_rows = []
    for _, row in oos_df.iterrows():
        s = row['strategy']
        would_reject = row['gate1_classification'] != 'PASS'
        cand_rows.append({
            'strategy': s, 'oos_pf': row['pf'], 'gate1_vs_phase33_40_candidates': row['gate1_classification'],
            'phase33_40_candidate_gate1_pass_rate_pct': round(candidate_gate1_pass_rate, 1),
            'would_be_rejected_if_new_candidate_today': would_reject,
            'rejecting_gate_if_any': ('Gate1 (OOS edge)' if would_reject else 'None on the computable gates -- parameter/cost robustness gates could not be evaluated (see preregistration section 4)'),
        })
    cand_df = pd.DataFrame(cand_rows)
    cand_df.to_csv(OUT / 'phase46_candidate_comparison.csv', index=False)
    print("\n[candidate comparison]"); print(cand_df.to_string())

    # ============ Survivorship audit ============
    surv_df = pd.DataFrame([{
        'question': 'Were these 6 strategies selected via the current Phase33+ competitive gate?',
        'answer': 'NO -- each was individually validated via an earlier, informal process (Phase3/3b for AMR pairs, Phase6 for CADJPY_ARB/AMR, Phase8 for GBPUSD Monday) documented in their own source docstrings/YAML comments, before the Phase33+ preregistration/perturbation/cost-stress framework existed',
        'bias_implication': 'The historical PF figures on record (both original-validation and this phases OOS re-measurement) describe strategies that were already known to look good BEFORE being placed into live rotation -- this is a real, disclosed survivorship consideration; the historical portfolio result should not be treated as independent confirmation of a de novo discovery process the way a Phase33+ candidate is',
        'resolution_status': 'ACKNOWLEDGED, NOT RESOLVED -- this phase does not attempt to correct for survivorship bias quantitatively, consistent with the preregistered scope',
    }])
    surv_df.to_csv(OUT / 'phase46_survivorship_audit.csv', index=False)
    print("\n[survivorship audit]"); print(surv_df.to_string())

    summary = {'n_strategies': len(STRATS), 'oos_gate1_pass_count': int((oos_df.gate1_classification == 'PASS').sum())}
    with open(OUT / '_phase46_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n" + json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
