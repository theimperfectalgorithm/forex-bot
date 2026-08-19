"""
Phase 45 -- portfolio viability & evidence sufficiency audit. Reuses
already-validated infrastructure from Phases 41/44 (correlation matrix,
daily ledger, baseline metrics, Monte Carlo methodology) rather than
recomputing from scratch. No new strategy, no backtest, no intervention.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).parent.parent
OUT = REPO / 'reports'
RNG = np.random.default_rng(20261015)
MECH_RE = re.compile(r'_(AMR|ARB|MONDAY)$')


def load_control():
    df = pd.read_csv(REPO / 'data' / 'phase26_all_trades.csv')
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df['instrument'] = df['strategy'].apply(lambda s: s.replace('_AMR', '').replace('_ARB', '').replace('_MONDAY', ''))
    df['mechanism'] = df['strategy'].apply(lambda s: MECH_RE.search(s).group(1) if MECH_RE.search(s) else 'UNKNOWN')
    df['is_jpy'] = df['instrument'].str.contains('JPY')
    return df


def main():
    df = load_control()
    print(f"[control] {len(df)} trades, reconciled: {len(df) == 2712}")
    strategies = sorted(df['strategy'].unique())

    # ============ Part 7: research-family audit (from the 71-row master ledger) ============
    ledger = pd.read_csv(OUT / 'phase45_research_master_ledger.csv')
    screen = ledger[ledger['notes'].astype(str).str.lower().str.contains('screen')]
    confirm = ledger[~ledger.index.isin(screen.index)]
    fam_rows = []
    for fam, sub in confirm.groupby('strategy_family'):
        gate1_pass = sub['final_classification'].astype(str).str.contains(
            r'^(?!.*NO CREDIBLE|.*NO EDGE).*', regex=True).sum()  # crude: count NOT explicitly "no edge/no credible"
        no_edge = sub['final_classification'].astype(str).str.contains('NO CREDIBLE|NO EDGE', case=False).sum()
        qualified = sub['final_classification'].astype(str).str.contains(r'\bJ\.|PORTFOLIO QUALIFIED', case=False).sum()
        fam_rows.append({
            'strategy_family': fam, 'n_hypotheses': len(sub),
            'n_reaching_gate1_edge': len(sub) - no_edge,
            'n_rejected_no_edge': no_edge,
            'n_portfolio_qualified': qualified,
            'classifications': '; '.join(sorted(set(sub['final_classification'].astype(str).str[:50]))),
        })
    fam_df = pd.DataFrame(fam_rows).sort_values('n_hypotheses', ascending=False)
    fam_df.to_csv(OUT / 'phase45_research_family_audit.csv', index=False)
    print(f"\n[research family audit] {len(confirm)} confirmatory hypotheses, {len(screen)} screen cells excluded")
    print(fam_df.to_string())

    # ============ Part 8: strategy independence (reuse Phase41 correlation matrix) ============
    corr = pd.read_csv(REPO / 'reports' / 'phase41_conditional_correlation.csv')
    avg_full_corr = corr['full_period_corr'].mean()
    avg_stress_corr = corr['stress_worst20pct_corr'].mean()
    # effective N (Phase31 methodology): N / (1 + (N-1)*avg_corr)
    n_strat = len(strategies)
    eff_n_full = n_strat / (1 + (n_strat - 1) * avg_full_corr)
    eff_n_stress = n_strat / (1 + (n_strat - 1) * avg_stress_corr) if pd.notna(avg_stress_corr) else None
    indep_rows = []
    for s in strategies:
        instr = s.replace('_AMR', '').replace('_ARB', '').replace('_MONDAY', '')
        mech = MECH_RE.search(s).group(1) if MECH_RE.search(s) else 'UNKNOWN'
        sub = df[df.strategy == s]
        indep_rows.append({
            'strategy': s, 'instrument': instr, 'is_jpy': bool(sub['is_jpy'].iloc[0]) if len(sub) else None,
            'mechanism': mech, 'session_mode': sub['session'].mode().iloc[0] if len(sub) else None,
            'n_trades': len(sub), 'pct_of_portfolio_trades': round(len(sub) / len(df) * 100, 1),
        })
    indep_df = pd.DataFrame(indep_rows)
    indep_df.loc['summary'] = {
        'strategy': 'PORTFOLIO SUMMARY', 'instrument': f'{n_strat} nominal strategies',
        'is_jpy': f"{int(df.is_jpy.mean()*100)}% of trades JPY-linked", 'mechanism': 'see phase41_mechanism_factor.csv',
        'session_mode': 'ASIAN/LONDON only (zero NY exposure, confirmed Phase31/41/42)',
        'n_trades': len(df), 'pct_of_portfolio_trades': 100.0,
    }
    indep_df.to_csv(OUT / '_scratch_phase45_independence.csv', index=False)
    print(f"\n[strategy independence] nominal N={n_strat}, effective N (full-period avg corr {avg_full_corr:.3f}) = {eff_n_full:.2f}, "
          f"effective N (stress avg corr {avg_stress_corr:.3f}) = {eff_n_stress:.2f}" if eff_n_stress else "")

    # ============ Part 9: historical portfolio edge (reuse Phase41/44 baseline) ============
    baseline = pd.read_csv(REPO / 'reports' / 'phase44_baseline.csv').iloc[0]
    base_dist = pd.read_csv(REPO / 'reports' / 'phase41_baseline_distribution.csv').iloc[0]
    hist_port = pd.DataFrame([{
        'total_trades': baseline['trade_count'], 'total_R': baseline['total_R'], 'pf': baseline['pf'],
        'expectancy_R': round(baseline['total_R'] / baseline['trade_count'], 4),
        'win_rate_pct': round((df['r_multiple'] > 0).mean() * 100, 1),
        'avg_win_R': round(df.loc[df.r_multiple > 0, 'r_multiple'].mean(), 3),
        'avg_loss_R': round(df.loc[df.r_multiple < 0, 'r_multiple'].mean(), 3),
        'max_dd_R': baseline['max_dd_R'], 'longest_dd_days': baseline['dd_duration_days'],
        'recovery_days': baseline['recovery_duration_days'],
        'daily_std_R': base_dist['std_daily_R'], 'daily_skew': base_dist['skew'], 'daily_kurtosis': base_dist['kurtosis'],
        'worst_day_R': baseline['worst_day_R'], 'worst_5day_R': baseline['worst_5day_R'], 'worst_10day_R': baseline['worst_10day_R'],
    }])
    hist_port.to_csv(OUT / 'phase45_historical_portfolio.csv', index=False)
    print("\n[historical portfolio edge]"); print(hist_port.to_string())

    # ============ Part 10: strategy contribution ============
    contrib_rows = []
    for s in strategies:
        sub = df[df.strategy == s]
        contrib_rows.append({
            'strategy': s, 'total_R': round(sub['r_multiple'].sum(), 2),
            'pct_of_portfolio_R': round(sub['r_multiple'].sum() / df['r_multiple'].sum() * 100, 1),
            'trade_count': len(sub), 'pct_of_portfolio_trades': round(len(sub) / len(df) * 100, 1),
            'win_contribution_R': round(sub.loc[sub.r_multiple > 0, 'r_multiple'].sum(), 2),
            'loss_contribution_R': round(sub.loc[sub.r_multiple < 0, 'r_multiple'].sum(), 2),
            'pf': round(sub.loc[sub.r_multiple > 0, 'r_multiple'].sum() / abs(sub.loc[sub.r_multiple < 0, 'r_multiple'].sum()), 3)
                  if sub.loc[sub.r_multiple < 0, 'r_multiple'].sum() != 0 else None,
        })
    contrib_df = pd.DataFrame(contrib_rows).sort_values('pct_of_portfolio_R', ascending=False)
    contrib_df.to_csv(OUT / 'phase45_strategy_contribution.csv', index=False)
    print("\n[strategy contribution]"); print(contrib_df.to_string())

    # ============ Part 11: strategy stability audit ============
    stab_rows = []
    for s in strategies:
        sub = df[df.strategy == s]
        pf = sub.loc[sub.r_multiple > 0, 'r_multiple'].sum() / abs(sub.loc[sub.r_multiple < 0, 'r_multiple'].sum()) if sub.loc[sub.r_multiple < 0, 'r_multiple'].sum() != 0 else None
        # worst losing streak
        streak = ms = 0
        for v in sub.sort_values('entry_time')['r_multiple']:
            if v < 0: streak += 1; ms = max(ms, streak)
            else: streak = 0
        n = len(sub)
        evidence = 'STRONG' if n >= 400 else ('MODERATE' if n >= 150 else ('WEAK' if n >= 50 else 'INSUFFICIENT'))
        stab_rows.append({
            'strategy': s, 'n_trades': n, 'historical_pf': round(pf, 3) if pf else None,
            'expectancy_R': round(sub['r_multiple'].mean(), 4), 'worst_losing_streak': ms,
            'parameter_robustness': 'NOT SEPARATELY TESTED IN THIS LEDGER (predates the Phase33+ preregistration discipline)',
            'cost_robustness': 'NOT SEPARATELY TESTED IN THIS LEDGER (predates the Phase33+ preregistration discipline)',
            'sample_size_evidence': evidence,
            'note': 'Historical PF/expectancy is IN-SAMPLE/HISTORICAL evidence (evidence-hierarchy tier 6), not independently OOS-validated the way Phase33+ candidates were',
        })
    stab_df = pd.DataFrame(stab_rows)
    stab_df.to_csv(OUT / 'phase45_strategy_stability.csv', index=False)
    print("\n[strategy stability]"); print(stab_df.to_string())

    # ============ Part 12: live validation (freshest local export) ============
    live = pd.read_csv(REPO / 'reports' / '5ers_trade_export.csv')
    live['entry_time'] = pd.to_datetime(live['entry_time'], errors='coerce')
    live['R'] = pd.to_numeric(live['R'], errors='coerce')
    live['profit'] = pd.to_numeric(live['profit'], errors='coerce')
    live_closed = live[live['status'] == 'CLOSED'].copy()
    control_strats = set(strategies)
    # live strategy names may have different suffix formatting (e.g. GBPUSD_MON vs GBPUSD_MONDAY)
    def norm_strat(s):
        s = str(s)
        if s == 'GBPUSD_MON':
            return 'GBPUSD_MONDAY'
        return s
    live_closed['strategy_norm'] = live_closed['strategy'].apply(norm_strat)
    live_current6 = live_closed[live_closed['strategy_norm'].isin(control_strats)]
    live_other = live_closed[~live_closed['strategy_norm'].isin(control_strats)]
    post_demotion_current6 = live_current6[live_current6['entry_time'] >= pd.Timestamp('2026-07-31', tz='UTC')]

    live_val_rows = [{
        'population': 'ALL CLOSED (current-6 strategies only)', 'n_trades': len(live_current6),
        'total_R': round(live_current6['R'].sum(), 3) if 'R' in live_current6 else None,
        'total_profit': round(live_current6['profit'].sum(), 2) if 'profit' in live_current6 else None,
        'win_rate_pct': round((live_current6['R'] > 0).mean() * 100, 1) if len(live_current6) else None,
        'pf': round(live_current6.loc[live_current6.R > 0, 'R'].sum() / abs(live_current6.loc[live_current6.R < 0, 'R'].sum()), 3)
              if len(live_current6) and live_current6.loc[live_current6.R < 0, 'R'].sum() != 0 else None,
    }, {
        'population': 'POST-DEMOTION (>=2026-07-31), current-6 only', 'n_trades': len(post_demotion_current6),
        'total_R': round(post_demotion_current6['R'].sum(), 3) if len(post_demotion_current6) else None,
        'total_profit': round(post_demotion_current6['profit'].sum(), 2) if len(post_demotion_current6) else None,
        'win_rate_pct': round((post_demotion_current6['R'] > 0).mean() * 100, 1) if len(post_demotion_current6) else None,
        'pf': None,
    }, {
        'population': 'GBPJPY_ARB (7th strategy, NOT part of frozen current-6 control -- reported separately)', 'n_trades': len(live_other),
        'total_R': round(live_other['R'].sum(), 3) if len(live_other) else None,
        'total_profit': round(live_other['profit'].sum(), 2) if len(live_other) else None,
        'win_rate_pct': round((live_other['R'] > 0).mean() * 100, 1) if len(live_other) else None,
        'pf': None,
    }]
    live_val_df = pd.DataFrame(live_val_rows)
    live_val_df.to_csv(OUT / 'phase45_live_validation.csv', index=False)
    print("\n[live validation]"); print(live_val_df.to_string())
    print(f"OPEN trades (not counted in closed-trade R): {(live['status']=='OPEN').sum()}")

    # per-strategy live breakdown
    live_strat_rows = []
    for s in strategies:
        sub = live_current6[live_current6.strategy_norm == s]
        live_strat_rows.append({
            'strategy': s, 'n_closed_trades': len(sub), 'total_R': round(sub['R'].sum(), 3) if len(sub) else 0,
            'win_rate_pct': round((sub['R'] > 0).mean() * 100, 1) if len(sub) else None,
        })
    live_strat_df = pd.DataFrame(live_strat_rows)
    print("\n[live per-strategy breakdown]"); print(live_strat_df.to_string())

    # ============ Part 13: live sample sufficiency (block bootstrap) ============
    n_live = len(post_demotion_current6)
    hist_r = df.sort_values('entry_time')['r_multiple'].values
    if n_live > 0 and n_live < len(hist_r):
        boot_totals = []
        for _ in range(10000):
            start = RNG.integers(0, len(hist_r) - n_live)
            block = hist_r[start:start + n_live]
            boot_totals.append(block.sum())
        boot_totals = np.array(boot_totals)
        live_total = post_demotion_current6['R'].sum()
        pctile = float((boot_totals < live_total).mean() * 100)
        suff_df = pd.DataFrame([{
            'live_n_trades': n_live, 'live_total_R': round(live_total, 3),
            'n_bootstrap_draws': 10000, 'bootstrap_method': 'SIMULATED (contiguous block bootstrap of historical trade-order R, block size = live sample size)',
            'bootstrap_median_R': round(np.median(boot_totals), 3), 'bootstrap_p5_R': round(np.percentile(boot_totals, 5), 3),
            'bootstrap_p95_R': round(np.percentile(boot_totals, 95), 3),
            'live_result_percentile_in_bootstrap': round(pctile, 1),
            'interpretation': ('WITHIN EXPECTED VARIATION' if 10 <= pctile <= 90 else
                                'UNUSUAL (outside 10th-90th percentile of same-sized historical samples)' if 2 <= pctile < 10 or 90 < pctile <= 98 else
                                'STATISTICALLY NOTABLE (outside 2nd-98th percentile) -- still not proof of deterioration given n=%d' % n_live),
        }])
    else:
        suff_df = pd.DataFrame([{'live_n_trades': n_live, 'note': 'INSUFFICIENT LIVE SAMPLE for bootstrap (n=0 or exceeds historical length)'}])
    suff_df.to_csv(OUT / 'phase45_live_sample_sufficiency.csv', index=False)
    print("\n[live sample sufficiency]"); print(suff_df.to_string())

    # ============ Part 14: live vs historical comparison ============
    lvh_rows = []
    hist_expectancy = df['r_multiple'].mean()
    for s in strategies:
        hsub = df[df.strategy == s]
        lsub = post_demotion_current6[post_demotion_current6.strategy_norm == s]
        n_l = len(lsub)
        if n_l == 0:
            cls = 'INSUFFICIENT SAMPLE (0 post-demotion closed trades)'
        elif n_l < 5:
            cls = f'INSUFFICIENT SAMPLE (n={n_l})'
        else:
            live_exp = lsub['R'].mean()
            hist_exp = hsub['r_multiple'].mean()
            cls = 'CONSISTENT' if abs(live_exp - hist_exp) < hsub['r_multiple'].std() else 'WITHIN EXPECTED VARIATION' if abs(live_exp - hist_exp) < 2 * hsub['r_multiple'].std() else 'UNUSUAL'
        lvh_rows.append({
            'strategy': s, 'historical_expectancy_R': round(hsub['r_multiple'].mean(), 4),
            'live_n_trades': n_l, 'live_expectancy_R': round(lsub['R'].mean(), 4) if n_l else None,
            'classification': cls,
        })
    lvh_df = pd.DataFrame(lvh_rows)
    lvh_df.to_csv(OUT / 'phase45_live_vs_historical.csv', index=False)
    print("\n[live vs historical]"); print(lvh_df.to_string())

    # ============ Part 15/16: deterioration / viability frameworks ============
    det_rows = [
        {'dimension': 'Sustained negative expectancy', 'threshold': f'Live expectancy below the historical worst {"regime-period" if True else ""} expectancy (Phase36/41/42 characterized periods) for >=100 portfolio-level closed trades', 'justification': 'Reuses Phase36/41 regime-period expectancy figures as the historical floor; 100-trade minimum matches Phase37/38 OOS statistical-informativeness convention (n>=30-40 minimum, scaled up for portfolio-level noise)'},
        {'dimension': 'PF deterioration', 'threshold': 'Post-demotion PF sustained below 0.8 (worse than the worst individual rejected Phase35 candidate) for >=100 trades', 'justification': 'Reuses the Gate1 PF>1.0 bar and Phase35s worst rejected PF (0.540-0.890) as a reference floor'},
        {'dimension': 'Regime failure', 'threshold': 'Live performance inconsistent with ALL 3 available historical regime periods (2023-2024/2025/2026YTD) simultaneously, per Phase42s regime-robustness convention', 'justification': 'Reuses Phase42/43s 3-period regime-robustness check exactly'},
        {'dimension': 'Parameter instability', 'threshold': 'NOT YET JUSTIFIABLE -- the current-6 strategies predate this projects preregistration discipline and were never subjected to the +/-20% perturbation framework used since Phase33', 'justification': 'Honest gap, not invented'},
        {'dimension': 'Execution deterioration', 'threshold': 'NOT YET JUSTIFIABLE -- no slippage/fill-quality baseline has been established in this project to compare against', 'justification': 'Honest gap, not invented'},
        {'dimension': 'Repeated failure outside known historical weakness', 'threshold': 'Losses concentrated in a regime/session/volatility-state NOT already characterized as a weak spot in Phase31/41/42 (e.g. a NEW failure mode, not the already-known HIGH-vol/AMR-ARB weakness)', 'justification': 'Reuses Phase41/42s own factor-attribution findings as the reference map of ALREADY-KNOWN weak spots'},
        {'dimension': 'Statistically unusual loss sequence', 'threshold': 'Live max drawdown or losing streak falls outside the 95th percentile of the Phase37-40/44 Monte Carlo reshuffle distribution for a same-sized sample', 'justification': 'Reuses the established Monte Carlo methodology (see phase45_live_sample_sufficiency.csv)'},
    ]
    det_df = pd.DataFrame(det_rows)
    det_df.to_csv(OUT / 'phase45_deterioration_framework.csv', index=False)

    via_rows = [
        {'dimension': 'Live performance within historical distribution', 'evidence_needed': 'Bootstrap percentile of live result within the 10th-90th percentile band (see phase45_live_sample_sufficiency.csv methodology)'},
        {'dimension': 'No new failure mode', 'evidence_needed': 'Live losses attributable to already-characterized factors (JPY/AMR saturation, HIGH-vol, Phase41/42 findings), not a novel pattern'},
        {'dimension': 'Stable execution', 'evidence_needed': 'NOT YET JUSTIFIABLE -- no baseline established (same gap as deterioration framework)'},
        {'dimension': 'No unexplained regime break', 'evidence_needed': 'Live volatility/session/currency characteristics consistent with the historical regime periods already characterized (Phase36/41/42)'},
        {'dimension': 'Strategy-level evidence', 'evidence_needed': 'Each strategy individually not classified UNUSUAL or worse in phase45_live_vs_historical.csv'},
        {'dimension': 'Portfolio-level evidence', 'evidence_needed': 'Portfolio-level live result within expected variation per the block-bootstrap (phase45_live_sample_sufficiency.csv)'},
        {'dimension': 'Sufficient sample size', 'evidence_needed': 'See phase45_forward_validation requirement -- current post-demotion sample (n reported in phase45_live_validation.csv) is explicitly assessed against this bar, not assumed sufficient'},
    ]
    via_df = pd.DataFrame(via_rows)
    via_df.to_csv(OUT / 'phase45_viability_framework.csv', index=False)
    print("\n[deterioration + viability frameworks written]")

    # ============ Part 17/18: scorecards ============
    scorecard_rows = []
    for _, row in stab_df.iterrows():
        s = row['strategy']
        lvh_row = lvh_df[lvh_df.strategy == s].iloc[0]
        hist_ev = 'STRONG' if row['n_trades'] >= 400 else ('MODERATE' if row['n_trades'] >= 150 else 'WEAK')
        robustness = 'WEAK (not separately parameter/cost-stress tested, predates Phase33+ discipline)'
        live_ev = lvh_row['classification']
        div = corr[(corr.strategy_1 == s) | (corr.strategy_2 == s)]['full_period_corr'].mean() if len(corr) else None
        diversification = 'LOW (avg corr %.2f with other strategies)' % div if div is not None and div == div else 'UNKNOWN'
        if 'INSUFFICIENT' in str(live_ev):
            status = 'AMBER -- WATCH (live sample insufficient to confirm or refute)'
        elif 'UNUSUAL' in str(live_ev):
            status = 'AMBER -- WATCH'
        else:
            status = 'GREEN -- CONTINUE VALIDATION'
        scorecard_rows.append({
            'strategy': s, 'historical_evidence': hist_ev, 'robustness': robustness,
            'live_evidence': live_ev, 'diversification': diversification,
            'current_confidence': 'MODERATE' if hist_ev in ('STRONG', 'MODERATE') else 'WEAK',
            'status': status,
        })
    scorecard_df = pd.DataFrame(scorecard_rows)
    scorecard_df.to_csv(OUT / 'phase45_strategy_evidence.csv', index=False)
    print("\n[strategy scorecard]"); print(scorecard_df.to_string())

    port_score_rows = [
        {'dimension': 'Historical edge', 'rating': 'MODERATE', 'explanation': f"PF {baseline['pf']}, total R {baseline['total_R']} over {baseline['trade_count']} trades -- real but not exceptional (see phase45_historical_portfolio.csv)"},
        {'dimension': 'Robustness', 'rating': 'WEAK', 'explanation': 'Current-6 strategies predate this projects preregistration/parameter-perturbation discipline -- no formal +/-20% or cost-stress test exists for any of them, unlike every Phase33+ candidate'},
        {'dimension': 'Regime diversity', 'rating': 'MODERATE', 'explanation': 'Reconstruction spans 2023-08 to 2026-08 (3 of 5 project-standard regime periods); pre-2023 UNKNOWN BY DATA ABSENCE'},
        {'dimension': 'Strategy independence', 'rating': f'WEAK (effective N={eff_n_full:.1f} of {n_strat} nominal)', 'explanation': f'Average full-period pairwise correlation {avg_full_corr:.3f}; JPY/AMR concentration is structural (Phase41/42), not a stress-specific artifact'},
        {'dimension': 'Live validation', 'rating': 'INSUFFICIENT' if n_live < 30 else 'WEAK', 'explanation': f'{n_live} post-demotion closed trades -- see phase45_live_sample_sufficiency.csv for the quantified bootstrap assessment'},
        {'dimension': 'Execution integrity', 'rating': 'UNKNOWN', 'explanation': 'No slippage/fill-quality monitoring baseline established in this project'},
        {'dimension': 'Risk integrity', 'rating': 'NOT SEPARATELY AUDITED THIS PHASE', 'explanation': 'Position-sizing/risk-limit configuration audit is out of this phases scope (would require live config inspection, not historical data analysis)'},
        {'dimension': 'Stress behaviour', 'rating': 'MODERATE', 'explanation': 'Phase41/42/43 found no single dominant stress factor; HIGH-volatility is the strongest (MODERATE) association; Phase44 found no portfolio control improves this without cost'},
        {'dimension': 'Research breadth', 'rating': 'MODERATE', 'explanation': f'{len(confirm)} confirmatory hypotheses across {fam_df.shape[0]} distinct families tested (Phase30-40); Phase39 found FX-technical ceiling reached for undifferentiated search'},
        {'dimension': 'Evidence sufficiency (overall)', 'rating': 'WEAK-TO-MODERATE', 'explanation': 'Strong historical reconstruction evidence, weak formal robustness evidence, and an explicitly insufficient live sample -- see final classification in the master report'},
    ]
    port_score_df = pd.DataFrame(port_score_rows)
    port_score_df.to_csv(OUT / 'phase45_portfolio_scorecard.csv', index=False)
    print("\n[portfolio-level scorecard]"); print(port_score_df.to_string())

    # ============ Part 25: information gaps ============
    gaps_rows = [
        {'gap': 'Live post-demotion sample size', 'category': 'REQUIRES TIME', 'detail': f'Only {n_live} closed current-6 trades since demotion -- see phase45_live_sample_sufficiency.csv for the quantified forward-validation requirement'},
        {'gap': 'Formal parameter/cost-stress robustness testing of the current-6 strategies themselves', 'category': 'CAN FIX NOW', 'detail': 'These strategies predate the Phase33+ preregistration discipline and have never been individually subjected to it -- technically feasible today with existing infrastructure'},
        {'gap': 'Historical economic-calendar / point-in-time macro data', 'category': 'REQUIRES NEW DATA SOURCE', 'detail': 'Confirmed blocked in Phase39 -- unchanged'},
        {'gap': 'Execution-quality / slippage baseline', 'category': 'REQUIRES TIME + NEW DATA SOURCE', 'detail': 'No historical fill-quality monitoring has been established; would need to be built and then observed over a live period'},
        {'gap': 'Pre-2023 historical data for the current-6 strategies', 'category': 'REQUIRES NEW DATA SOURCE (or may not exist)', 'detail': 'The control reconstruction starts 2023-08-01; whether older broker/strategy history exists was not investigated in this phase'},
        {'gap': 'Independent (non-in-sample) validation of Phase44s counterfactual controls', 'category': 'REQUIRES TIME', 'detail': 'Phase44 explicitly labeled all findings IN-SAMPLE COUNTERFACTUAL EVIDENCE'},
    ]
    gaps_df = pd.DataFrame(gaps_rows)
    gaps_df.to_csv(OUT / 'phase45_information_gaps.csv', index=False)
    print("\n[information gaps]"); print(gaps_df.to_string())

    # ============ Part 24: research priority ============
    priority_rows = [
        {'activity': 'Continued live validation (accumulate post-demotion sample)', 'info_gain': 'HIGH', 'cost': 'LOW (already running)', 'overfitting_risk': 'NONE (observational)', 'recommendation': 'CONTINUE'},
        {'activity': 'Formal parameter/cost-stress robustness testing of the current-6 live strategies', 'info_gain': 'MODERATE-HIGH (closes a real, disclosed gap)', 'cost': 'LOW (existing infrastructure)', 'overfitting_risk': 'LOW if no re-parameterization follows', 'recommendation': 'CONTINUE -- high-value, low-cost'},
        {'activity': 'Portfolio-control OOS validation (if a future control is designed)', 'info_gain': 'UNKNOWN (no control has passed Phase44s bar yet)', 'cost': 'MODERATE', 'overfitting_risk': 'LOW if genuinely OOS', 'recommendation': 'NOT YET APPLICABLE -- no control qualifies (Phase44)'},
        {'activity': 'Volatility-conditioned research (self-calculated, non-directional framing)', 'info_gain': 'MODERATE (Phase39 found this the most immediately researchable alternative)', 'cost': 'LOW', 'overfitting_risk': 'LOW-MEDIUM', 'recommendation': 'CONTINUE -- per Phase39, not restarted here'},
        {'activity': 'Event/macro data infrastructure investment', 'info_gain': 'POTENTIALLY HIGH (targets Gap2 directly)', 'cost': 'HIGH', 'overfitting_risk': 'HIGH once built (Phase39s own finding)', 'recommendation': 'CONTINUE TO DEFER -- unchanged from Phase39/41-44'},
        {'activity': 'Index-based research infrastructure', 'info_gain': 'MODERATE', 'cost': 'MODERATE', 'overfitting_risk': 'LOW-MEDIUM', 'recommendation': 'CONTINUE TO CONSIDER -- unchanged from Phase39'},
        {'activity': 'Another undifferentiated FX-technical strategy search', 'info_gain': 'LOW (Phase39 ceiling finding, reconfirmed by Phase40s rejection)', 'cost': 'LOW', 'overfitting_risk': 'MODERATE (multiple-testing accumulation)', 'recommendation': 'STOP'},
        {'activity': 'Rescuing/re-parameterizing any rejected Phase33-44 candidate', 'info_gain': 'LOW (explicitly the p-hacking pattern this projects discipline exists to prevent)', 'cost': 'LOW', 'overfitting_risk': 'HIGH', 'recommendation': 'STOP'},
        {'activity': 'Optimizing a portfolio control against the same historical sample used in Phase41-44', 'info_gain': 'LOW (in-sample, already disclosed as such)', 'cost': 'LOW', 'overfitting_risk': 'HIGH', 'recommendation': 'STOP'},
    ]
    priority_df = pd.DataFrame(priority_rows)
    priority_df.to_csv(OUT / 'phase45_research_priority.csv', index=False)
    print("\n[research priority]"); print(priority_df.to_string())

    # ============ Part 27: future requirements (minimum evidence + forward validation) ============
    future_rows = [
        {'decision': 'A. Changing a strategy', 'minimum_evidence': 'Individual strategy classified UNUSUAL or worse in phase45_live_vs_historical.csv for >=100 live trades, corroborated by a specific, characterized new failure mode (not already-known JPY/AMR/HIGH-vol weakness)'},
        {'decision': 'B. Pausing a strategy', 'minimum_evidence': 'Sustained negative expectancy (phase45_deterioration_framework.csv row 1) for that specific strategy over >=100 trades, OR a statistically unusual loss sequence (Monte Carlo <5th percentile) with no corroborating benign explanation'},
        {'decision': 'C. Changing portfolio risk', 'minimum_evidence': 'NO EVIDENCE-BASED THRESHOLD ESTABLISHED -- this project has never formally studied risk-sizing sensitivity'},
        {'decision': 'D. Adding a new strategy', 'minimum_evidence': 'A candidate reaching Phase33-40s J.PORTFOLIO QUALIFIED classification -- none has, to date (see phase45_research_master_ledger.csv)'},
        {'decision': 'E. Removing a strategy', 'minimum_evidence': 'Same bar as pausing (B), sustained over a longer window (this project has never established a distinct, higher bar for permanent removal vs. temporary pause)'},
        {'decision': 'F. Deploying a portfolio control', 'minimum_evidence': 'A control reaching Phase44s A.HISTORICALLY PROMISING classification, followed by independent OOS/paper-trading validation -- none has, to date (see phase44_evidence_matrix.csv)'},
        {'decision': 'G. Increasing capital', 'minimum_evidence': 'NO EVIDENCE-BASED THRESHOLD ESTABLISHED -- this project has not studied capital-scaling effects'},
        {'decision': 'H. Decreasing capital', 'minimum_evidence': 'NO EVIDENCE-BASED THRESHOLD ESTABLISHED -- same gap as G'},
    ]
    future_df = pd.DataFrame(future_rows)
    future_df.to_csv(OUT / 'phase45_future_requirements.csv', index=False)
    print("\n[minimum evidence requirements]"); print(future_df.to_string())

    summary = {
        'n_confirmatory_hypotheses': len(confirm), 'n_families': fam_df.shape[0],
        'effective_N_full': round(eff_n_full, 2), 'n_live_post_demotion': n_live,
        'live_total_R': round(post_demotion_current6['R'].sum(), 3) if n_live else None,
        'baseline_total_R': baseline['total_R'], 'baseline_pf': baseline['pf'],
    }
    with open(OUT / '_phase45_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n" + json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
