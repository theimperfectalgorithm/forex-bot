"""
Phase 49 -- portfolio stress mechanism & contribution audit. Diagnostic
only: no filter, control, or intervention is implemented. Reuses the
frozen control (data/phase26_all_trades.csv) as the primary population.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from phase49_stress_dataset import load_control, build_daily_dataset  # noqa: E402
from phase49_joint_state import add_binary_flags, run_joint_state_analysis  # noqa: E402
from phase49_temporal_validation import run_temporal_validation  # noqa: E402
from research_data_validator import ValidationReport, validate_column_count_consistency  # noqa: E402

REPO = Path(__file__).parent.parent
OUT = REPO / 'reports'
MIN_CELL = 10
MIN_STRESS = 8


def eff(vals, n_thresh=MIN_CELL):
    n = len(vals)
    if n < n_thresh:
        return None, n, 'INSUFFICIENT SAMPLE'
    return round(float(np.mean(vals)), 4), n, None


def main():
    hist_path = REPO / 'data' / 'phase26_all_trades.csv'
    rep = ValidationReport(path=str(hist_path))
    validate_column_count_consistency(hist_path, rep)
    print(f"[validate] {rep.summary()}")

    df = load_control()
    ledger = build_daily_dataset(df)
    ledger.to_csv(OUT / 'phase49_daily_portfolio_dataset.csv', index=False)
    print(f"[daily dataset] {len(ledger)} days")

    q = {p: np.percentile(ledger.total_R, p) for p in [1, 5, 10, 20]}
    buckets = {
        'worst_1pct': ledger[ledger.total_R <= q[1]], 'worst_5pct': ledger[ledger.total_R <= q[5]],
        'worst_10pct': ledger[ledger.total_R <= q[10]], 'normal': ledger[ledger.total_R > q[20]],
    }
    stress_def_df = pd.DataFrame([
        {'bucket': k, 'n_days': len(v), 'threshold_R': q.get(k.split('_')[1].replace('pct', '')) if 'worst' in k else None}
        for k, v in buckets.items()
    ])
    stress_def_df.to_csv(OUT / 'phase49_stress_definition.csv', index=False)
    print("\n[stress definition]"); print(stress_def_df.to_string())

    # --- Part 8: marginal stress factors ---
    factor_cols = {
        'vol_pctile': 'volatility percentile', 'max_concurrent': 'concurrency',
        'jpy_share_pct': 'JPY exposure', 'amr_share_pct': 'AMR exposure', 'arb_share_pct': 'ARB exposure',
        'long_share_pct': 'directional bias (long share)', 'n_strategies_active': 'strategy count',
        'n_simultaneous_jpy': 'simultaneous JPY positions', 'n_simultaneous_amr': 'simultaneous AMR positions',
    }
    marg_rows = []
    for col, label in factor_cols.items():
        normal_mean, normal_n, _ = eff(buckets['normal'][col].dropna())
        for bname in ['worst_20pct' if False else 'worst_10pct', 'worst_5pct', 'worst_1pct']:
            pass
        for bname, bdf in [('worst_10pct', buckets['worst_10pct']), ('worst_5pct', buckets['worst_5pct']), ('worst_1pct', buckets['worst_1pct'])]:
            stress_mean, stress_n, flag = eff(bdf[col].dropna())
            effect = round(stress_mean - normal_mean, 3) if (stress_mean is not None and normal_mean is not None) else None
            marg_rows.append({'factor': label, 'column': col, 'bucket': bname, 'normal_mean': normal_mean, 'normal_n': normal_n,
                               'stress_mean': stress_mean, 'stress_n': stress_n, 'effect': effect,
                               'evidence': ('INSUFFICIENT SAMPLE' if flag else
                                            'STRONG' if effect is not None and abs(effect) > (abs(normal_mean) * 0.5 if normal_mean else 1) else
                                            'MODERATE' if effect is not None and abs(effect) > (abs(normal_mean) * 0.2 if normal_mean else 0.3) else
                                            'PLAUSIBLE' if effect is not None else 'NO EVIDENCE')})
    marg_df = pd.DataFrame(marg_rows)
    marg_df.to_csv(OUT / 'phase49_marginal_stress_factors.csv', index=False)
    print("\n[marginal stress factors] (worst_5pct)"); print(marg_df[marg_df.bucket == 'worst_5pct'].to_string())

    # --- Part 9: joint-state analysis (exactly the 12 preregistered combos, via phase49_joint_state.py) ---
    ledger = add_binary_flags(ledger)
    joint_df = run_joint_state_analysis(ledger)
    joint_df.to_csv(OUT / 'phase49_joint_state_analysis.csv', index=False)
    print(f"\n[joint-state analysis] {len(joint_df)} combination-states tested ({(joint_df.evidence == 'ADEQUATE SAMPLE').sum()} adequately sampled)")

    # --- Part 10: stress clusters ---
    worst10 = buckets['worst_10pct'].sort_values('date').reset_index(drop=True)
    worst10['date_ts'] = pd.to_datetime(worst10['date'])
    gaps = worst10['date_ts'].diff().dt.days
    clustered = (gaps <= 5).sum()
    cluster_pct = clustered / max(len(worst10) - 1, 1) * 100
    cluster_class = ('B. CLUSTERED' if cluster_pct > 40 else 'A. ISOLATED' if cluster_pct < 15 else 'D. MIXED')
    cluster_df = pd.DataFrame([{'n_worst10pct_days': len(worst10), 'pct_within_5days_of_another_stress_day': round(cluster_pct, 1),
                                 'classification': cluster_class}])
    cluster_df.to_csv(OUT / 'phase49_stress_clusters.csv', index=False)
    print("\n[stress clusters]"); print(cluster_df.to_string())

    # --- Part 11: pre-stress exposure (T-1; intraday disclosed as N/A) ---
    ledger_idx = ledger.set_index('date')
    pre_rows = []
    for d in buckets['worst_5pct']['date']:
        prior_dates = [x for x in ledger['date'] if x < d]
        if not prior_dates:
            continue
        t1 = prior_dates[-1]
        row = ledger_idx.loc[t1]
        pre_rows.append({'stress_date': d, 't1_date': t1, 't1_max_concurrent': row['max_concurrent'],
                          't1_vol_state': row['vol_state'], 't1_jpy_share': row['jpy_share_pct'], 't1_amr_share': row['amr_share_pct'],
                          't1_long_share': row['long_share_pct']})
    pre_df = pd.DataFrame(pre_rows)
    pre_df['t60min_exposure'] = 'UNKNOWN BY DATA LIMITATION -- no continuous intraday position-snapshot series exists at this granularity'
    pre_df['t30min_exposure'] = 'UNKNOWN BY DATA LIMITATION'
    pre_df['t15min_exposure'] = 'UNKNOWN BY DATA LIMITATION'
    pre_df.to_csv(OUT / 'phase49_pre_stress_exposure.csv', index=False)
    print(f"\n[pre-stress exposure] {len(pre_df)} worst-5% days characterized at T-1; intraday T-60/30/15min UNKNOWN BY DATA LIMITATION")

    # --- Part 12: transition analysis (re-test Phase42, control for concurrency where possible) ---
    trans_rows = []
    for trans in ['NORMAL_to_HIGH', 'HIGH_to_HIGH', 'HIGH_to_NORMAL', 'NORMAL_to_NORMAL', 'LOW_to_HIGH']:
        sub = ledger[ledger.vol_transition == trans]
        if len(sub) < MIN_CELL:
            trans_rows.append({'transition': trans, 'n_days': len(sub), 'evidence': 'INSUFFICIENT SAMPLE'})
            continue
        hi_conc = sub[sub.conc_4plus]
        lo_conc = sub[~sub.conc_4plus]
        trans_rows.append({'transition': trans, 'n_days': len(sub), 'mean_R': round(sub.total_R.mean(), 4),
                            'mean_R_high_concurrency_subset': round(hi_conc.total_R.mean(), 4) if len(hi_conc) >= MIN_CELL else None,
                            'mean_R_low_concurrency_subset': round(lo_conc.total_R.mean(), 4) if len(lo_conc) >= MIN_CELL else None,
                            'evidence': 'ADEQUATE SAMPLE'})
    trans_df = pd.DataFrame(trans_rows)
    trans_df.to_csv(OUT / 'phase49_transition_analysis.csv', index=False)
    print("\n[transition analysis]"); print(trans_df.to_string())

    # --- Part 13: concurrency analysis ---
    conc_rows = []
    for thresh in [1, 2, 3, 4, 5, 6]:
        sub = ledger[ledger.max_concurrent >= thresh]
        if len(sub) < MIN_CELL:
            conc_rows.append({'threshold': f'{thresh}+', 'n_days': len(sub), 'evidence': 'INSUFFICIENT SAMPLE'}); continue
        hv = sub[sub.vol_high]
        conc_rows.append({'threshold': f'{thresh}+', 'n_days': len(sub), 'mean_R': round(sub.total_R.mean(), 4),
                           'worst_R': round(sub.total_R.min(), 4), 'high_vol_pct': round(sub.vol_high.mean() * 100, 1),
                           'high_vol_mean_R': round(hv.total_R.mean(), 4) if len(hv) >= MIN_CELL else None,
                           'evidence': 'ADEQUATE SAMPLE'})
    conc_df = pd.DataFrame(conc_rows)
    conc_df.to_csv(OUT / 'phase49_concurrency_analysis.csv', index=False)
    print("\n[concurrency analysis]"); print(conc_df.to_string())

    # --- Part 14: strategy-combination analysis ---
    combo_rows = []
    for combo, sub in ledger.groupby('active_strategies'):
        if len(sub) < MIN_CELL:
            continue
        combo_rows.append({'strategy_combination': combo, 'n_days': len(sub), 'mean_R': round(sub.total_R.mean(), 4),
                            'worst_R': round(sub.total_R.min(), 4), 'stress_day_pct': round((sub.date.isin(buckets['worst_10pct'].date)).mean() * 100, 1)})
    combo_df = pd.DataFrame(combo_rows).sort_values('n_days', ascending=False)
    combo_df.to_csv(OUT / 'phase49_strategy_combinations.csv', index=False)
    print(f"\n[strategy combinations] {len(combo_df)} combos with >=10 days"); print(combo_df.head(10).to_string())

    # --- Part 15: GBPJPY_AMR contribution analysis ---
    gj = df[df.strategy == 'GBPJPY_AMR']
    gj_daily = gj.groupby('trade_date')['r_multiple'].sum()
    ledger_gj = ledger.copy()
    ledger_gj['gbpjpy_R'] = ledger_gj['date'].map(gj_daily).fillna(0)
    active_days = ledger_gj[ledger_gj.gbpjpy_amr_active > 0]
    inactive_days = ledger_gj[ledger_gj.gbpjpy_amr_active == 0]
    gj_rows = [
        {'question': 'GBPJPY alone (portfolio-day R on days GBPJPY_AMR trades vs days it does not)',
         'active_mean_R': round(active_days.total_R.mean(), 4), 'active_n': len(active_days),
         'inactive_mean_R': round(inactive_days.total_R.mean(), 4), 'inactive_n': len(inactive_days)},
        {'question': 'GBPJPY + HIGH volatility', 'active_mean_R': round(active_days[active_days.vol_high].total_R.mean(), 4) if len(active_days[active_days.vol_high]) >= MIN_CELL else None,
         'active_n': len(active_days[active_days.vol_high]), 'inactive_mean_R': None, 'inactive_n': None},
        {'question': 'GBPJPY + 4+ concurrency', 'active_mean_R': round(active_days[active_days.conc_4plus].total_R.mean(), 4) if len(active_days[active_days.conc_4plus]) >= MIN_CELL else None,
         'active_n': len(active_days[active_days.conc_4plus]), 'inactive_mean_R': None, 'inactive_n': None},
        {'question': 'GBPJPY own-R correlation with total portfolio R on active days',
         'active_mean_R': round(active_days['gbpjpy_R'].corr(active_days['total_R']), 3), 'active_n': len(active_days), 'inactive_mean_R': None, 'inactive_n': None},
    ]
    gj_df = pd.DataFrame(gj_rows)
    gj_df.to_csv(OUT / 'phase49_gbpjpy_analysis.csv', index=False)
    print("\n[GBPJPY_AMR analysis]"); print(gj_df.to_string())

    # --- Part 16: JPY analysis controlling for concurrency/volatility ---
    jpy_rows = []
    for vs in ['LOW', 'NORMAL', 'HIGH']:
        for cf in [True, False]:
            sub = ledger[(ledger.vol_state == vs) & (ledger.conc_4plus == cf)]
            jpy_hi = sub[sub.jpy_high]
            jpy_lo = sub[~sub.jpy_high]
            jpy_rows.append({'vol_state': vs, 'concurrency_4plus': cf, 'jpy_high_n': len(jpy_hi), 'jpy_high_mean_R': round(jpy_hi.total_R.mean(), 4) if len(jpy_hi) >= MIN_CELL else None,
                              'jpy_low_n': len(jpy_lo), 'jpy_low_mean_R': round(jpy_lo.total_R.mean(), 4) if len(jpy_lo) >= MIN_CELL else None})
    jpy_df = pd.DataFrame(jpy_rows)
    jpy_df.to_csv(OUT / 'phase49_jpy_analysis.csv', index=False)
    print("\n[JPY analysis, controlling for vol/concurrency]"); print(jpy_df.to_string())

    # --- Part 17: AMR analysis ---
    amr_rows = []
    for vs in ['LOW', 'NORMAL', 'HIGH']:
        sub = ledger[ledger.vol_state == vs]
        amr_hi = sub[sub.amr_high]; amr_lo = sub[~sub.amr_high]
        amr_rows.append({'vol_state': vs, 'amr_high_n': len(amr_hi), 'amr_high_mean_R': round(amr_hi.total_R.mean(), 4) if len(amr_hi) >= MIN_CELL else None,
                          'amr_low_n': len(amr_lo), 'amr_low_mean_R': round(amr_lo.total_R.mean(), 4) if len(amr_lo) >= MIN_CELL else None})
    amr_df = pd.DataFrame(amr_rows)
    amr_df.to_csv(OUT / 'phase49_amr_analysis.csv', index=False)

    # --- Part 18: directional asymmetry ---
    dir_rows = []
    for cond_name, cond in [('normal_vol', ledger.vol_state != 'HIGH'), ('high_vol', ledger.vol_state == 'HIGH'),
                             ('high_concurrency', ledger.conc_4plus), ('stress_day', ledger.date.isin(buckets['worst_10pct'].date)),
                             ('jpy_heavy', ledger.jpy_high), ('amr_heavy', ledger.amr_high)]:
        sub = ledger[cond]
        lg = sub[sub.long_heavy]; sh = sub[~sub.long_heavy]
        dir_rows.append({'condition': cond_name, 'long_heavy_n': len(lg), 'long_heavy_mean_R': round(lg.total_R.mean(), 4) if len(lg) >= MIN_CELL else None,
                          'short_heavy_n': len(sh), 'short_heavy_mean_R': round(sh.total_R.mean(), 4) if len(sh) >= MIN_CELL else None})
    dir_df = pd.DataFrame(dir_rows)
    dir_df.to_csv(OUT / 'phase49_directional_analysis.csv', index=False)
    print("\n[directional asymmetry]"); print(dir_df.to_string())

    # --- Part 19: session analysis ---
    ledger['session_profile'] = np.select(
        [ledger.asian_share_pct >= 90, ledger.london_share_pct >= 90,
         (ledger.asian_share_pct > 0) & (ledger.london_share_pct > 0), ledger.ny_share_pct > 0],
        ['ASIAN_ONLY', 'LONDON_ONLY', 'ASIAN_AND_LONDON', 'HAS_NY_EXPOSURE'], default='OTHER')
    sess_rows = []
    for prof, sub in ledger.groupby('session_profile'):
        sess_rows.append({'session_profile': prof, 'n_days': len(sub), 'mean_R': round(sub.total_R.mean(), 4) if len(sub) >= MIN_CELL else None,
                           'stress_day_pct': round((sub.date.isin(buckets['worst_10pct'].date)).mean() * 100, 1)})
    sess_df = pd.DataFrame(sess_rows)
    sess_df.to_csv(OUT / 'phase49_session_analysis.csv', index=False)
    print("\n[session analysis]"); print(sess_df.to_string())
    print(f"NY exposure confirmed: {(ledger.ny_share_pct > 0).sum()} of {len(ledger)} days have any NY-session trade")

    # --- Part 20: multi-factor OLS model ---
    model_df = ledger.dropna(subset=['vol_pctile']).copy()
    X_cols = ['vol_pctile', 'max_concurrent', 'jpy_share_pct', 'amr_share_pct', 'arb_share_pct', 'long_share_pct', 'n_strategies_active']
    X_raw = model_df[X_cols].fillna(0).values
    X_z = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)
    X = np.column_stack([np.ones(len(X_z)), X_z])
    y = model_df['total_R'].values
    beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    n, k = X.shape
    sigma2 = (resid @ resid) / (n - k)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    ss_res = (resid ** 2).sum(); ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    model_rows = [{'predictor': 'intercept', 'coefficient': round(beta[0], 4), 'std_error': round(se[0], 4)}]
    for i, c in enumerate(X_cols):
        model_rows.append({'predictor': c, 'coefficient': round(beta[i + 1], 4), 'std_error': round(se[i + 1], 4),
                            't_stat': round(beta[i + 1] / se[i + 1], 2) if se[i + 1] > 0 else None})
    model_df_out = pd.DataFrame(model_rows)
    model_df_out.attrs['r2'] = r2
    model_df_out.to_csv(OUT / 'phase49_multifactor_model.csv', index=False)
    with open(OUT / '_phase49_model_r2.txt', 'w') as f:
        f.write(f"R2={r2:.4f} n={n}\n")
    print(f"\n[multi-factor OLS model] n={n} R2={r2:.4f}"); print(model_df_out.to_string())

    # --- Part 21: temporal validation (via phase49_temporal_validation.py) ---
    temp_df = run_temporal_validation(ledger)
    surv_df = temp_df.attrs['survival']
    temp_df.to_csv(OUT / 'phase49_temporal_validation.csv', index=False)
    print("\n[temporal validation]"); print(temp_df.to_string())
    print("\n[temporal survival summary]"); print(surv_df.to_string())

    # --- Part 22: multiple testing log ---
    mt_rows = [
        {'item': '12 preregistered joint-state combinations (Part9)', 'type': 'PRIMARY PREREGISTERED', 'n_sub_tests': len(joint_df)},
        {'item': '9 marginal stress factors (Part8)', 'type': 'PRIMARY PREREGISTERED', 'n_sub_tests': len(marg_df)},
        {'item': 'Transition analysis (Part12)', 'type': 'PRIMARY PREREGISTERED (reuses Phase42 definition)', 'n_sub_tests': len(trans_df)},
        {'item': 'Concurrency thresholds 1-6+ (Part13)', 'type': 'PRIMARY PREREGISTERED', 'n_sub_tests': len(conc_df)},
        {'item': 'Strategy-combination analysis (Part14)', 'type': 'EXPLORATORY (data-driven grouping, not preregistered a priori)', 'n_sub_tests': len(combo_df)},
        {'item': 'GBPJPY_AMR-specific analysis (Part15)', 'type': 'PRIMARY PREREGISTERED (directly motivated by Phase48 finding)', 'n_sub_tests': len(gj_df)},
        {'item': 'JPY/AMR conditional analysis (Part16-17)', 'type': 'PRIMARY PREREGISTERED', 'n_sub_tests': len(jpy_df) + len(amr_df)},
        {'item': 'Directional asymmetry (Part18)', 'type': 'PRIMARY PREREGISTERED (reuses Phase42 finding)', 'n_sub_tests': len(dir_df)},
        {'item': 'Session analysis (Part19)', 'type': 'PRIMARY PREREGISTERED', 'n_sub_tests': len(sess_df)},
        {'item': 'Multi-factor OLS model (Part20)', 'type': 'PRIMARY PREREGISTERED, EXPLANATORY ONLY', 'n_sub_tests': len(X_cols)},
        {'item': 'Temporal validation (Part21)', 'type': 'PRIMARY PREREGISTERED', 'n_sub_tests': len(temp_df)},
    ]
    pd.DataFrame(mt_rows).to_csv(OUT / 'phase49_multiple_testing.csv', index=False)

    # --- Part 23: worst-day contribution decomposition ---
    decomp_rows = []
    for bname in ['worst_1pct', 'worst_5pct', 'worst_10pct']:
        bdates = set(buckets[bname]['date'])
        bucket_trades = df[df.trade_date.isin(bdates)]
        total_loss = bucket_trades['r_multiple'].sum()
        for strat, sub in bucket_trades.groupby('strategy'):
            decomp_rows.append({'bucket': bname, 'strategy': strat, 'strategy_R': round(sub['r_multiple'].sum(), 3),
                                 'pct_of_bucket_total': round(sub['r_multiple'].sum() / total_loss * 100, 1) if total_loss != 0 else None,
                                 'n_trades': len(sub)})
    decomp_df = pd.DataFrame(decomp_rows)
    decomp_df.to_csv(OUT / 'phase49_worst_day_decomposition.csv', index=False)
    # concentration classification
    conc_class_rows = []
    for bname in decomp_df.bucket.unique():
        sub = decomp_df[decomp_df.bucket == bname]
        max_share = sub['pct_of_bucket_total'].abs().max()
        cls = 'CONCENTRATED' if max_share > 50 else 'PROPORTIONAL' if max_share < 30 else 'MIXED'
        conc_class_rows.append({'bucket': bname, 'max_single_strategy_share_pct': max_share, 'classification': cls})
    print("\n[worst-day decomposition -- concentration]"); print(pd.DataFrame(conc_class_rows).to_string())

    # --- Part 24: loss sequence analysis ---
    seq_rows = []
    for bname in ['worst_1pct', 'worst_5pct', 'worst_10pct']:
        bdates = set(buckets[bname]['date'])
        bucket_trades = df[df.trade_date.isin(bdates)]
        by_day = bucket_trades.groupby('trade_date')
        n_sim_losers = []
        for d, day in by_day:
            by_strat = day.groupby('strategy')['r_multiple'].sum()
            n_sim_losers.append((by_strat < 0).sum())
        seq_rows.append({'bucket': bname, 'n_days': len(bdates), 'avg_simultaneous_losing_strategies': round(np.mean(n_sim_losers), 2),
                          'pct_days_with_2plus_losers': round(np.mean([n >= 2 for n in n_sim_losers]) * 100, 1)})
    seq_df = pd.DataFrame(seq_rows)
    seq_df.to_csv(OUT / 'phase49_loss_sequence.csv', index=False)
    print("\n[loss sequence]"); print(seq_df.to_string())

    # --- Part 25: descriptive counterfactuals ---
    cf_rows = []
    for bname in ['worst_10pct']:
        bdates = set(buckets[bname]['date'])
        actual_total = buckets[bname].total_R.sum()
        no_gj = df[~((df.trade_date.isin(bdates)) & (df.strategy == 'GBPJPY_AMR'))]
        no_gj_total = no_gj[no_gj.trade_date.isin(bdates)]['r_multiple'].sum()
        no_highconc_days = buckets[bname][buckets[bname].max_concurrent < 4]
        cf_rows.append({'counterfactual': 'Remove GBPJPY_AMR trades from worst-10% days', 'actual_total_R': round(actual_total, 3),
                         'counterfactual_total_R': round(no_gj_total, 3), 'label': 'DESCRIPTIVE COUNTERFACTUAL -- not a validated control'})
        cf_rows.append({'counterfactual': 'Worst-10% days with concurrency<4 only (excludes 4+ days entirely)',
                         'actual_total_R': round(actual_total, 3), 'counterfactual_total_R': round(no_highconc_days.total_R.sum(), 3),
                         'label': 'DESCRIPTIVE COUNTERFACTUAL -- not a validated control'})
    cf_df = pd.DataFrame(cf_rows)
    cf_df.to_csv(OUT / 'phase49_descriptive_counterfactuals.csv', index=False)
    print("\n[descriptive counterfactuals]"); print(cf_df.to_string())

    # --- Part 26: live comparison (reuse Phase45/46/48 already-validated live data) ---
    live = pd.read_csv(REPO / 'reports' / '5ers_trade_export.csv')
    live['entry_time'] = pd.to_datetime(live['entry_time'], errors='coerce')
    live['R'] = pd.to_numeric(live['R'], errors='coerce')
    live_closed = live[live['status'] == 'CLOSED'].copy()
    live_closed['strategy_norm'] = live_closed['strategy'].apply(lambda s: 'GBPUSD_MONDAY' if s == 'GBPUSD_MON' else s)
    post_demo = live_closed[live_closed['entry_time'] >= pd.Timestamp('2026-07-31', tz='UTC')]
    live_cmp = pd.DataFrame([{
        'population': 'POST-DEMOTION LIVE (current-6 only)', 'n_trades': len(post_demo),
        'total_R': round(post_demo['R'].sum(), 3), 'jpy_share_pct': round(post_demo['strategy_norm'].str.contains('JPY').mean() * 100, 1) if len(post_demo) else None,
        'amr_share_pct': round(post_demo['strategy_norm'].str.contains('AMR').mean() * 100, 1) if len(post_demo) else None,
        'resemblance_to_historical_stress_pattern': 'DESCRIPTIVE ONLY -- sample (n=%d) too small for a confident comparison to the historical joint-state findings above' % len(post_demo),
    }])
    live_cmp.to_csv(OUT / 'phase49_live_comparison.csv', index=False)
    print("\n[live comparison]"); print(live_cmp.to_string())

    summary = {'n_days': len(ledger), 'model_r2': round(r2, 4), 'n_worst10_days': len(buckets['worst_10pct']),
               'cluster_classification': cluster_class}
    with open(OUT / '_phase49_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n" + json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
