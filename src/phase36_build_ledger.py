"""Phase 36 -- build the consolidated research ledger from committed artifacts."""
import pandas as pd

REPO = 'c:/Users/bandh/forex-bot'
rows = []

p30 = pd.read_csv(f'{REPO}/reports/non_jpy_candidate_registry.csv')
for _, r in p30.iterrows():
    rows.append({
        'experiment_id': r['experiment_id'], 'phase': 'Phase30', 'hypothesis_id': r['experiment_id'],
        'strategy_family': r['strategy_family'], 'instrument': r['instrument'], 'mechanism': r['strategy_family'],
        'session': r['session'], 'timeframe': r['timeframe'],
        'train_period': r['train_period'], 'validation_period': r['validation_period'], 'oos_period': r['oos_period'],
        'oos_trades': r['trade_count'], 'oos_pf': r['oos_pf'], 'oos_expectancy': r['oos_expectancy'],
        'oos_total_R': 'NA', 'oos_max_dd': 'NA', 'oos_first_half_R': 'NA', 'oos_second_half_R': 'NA',
        'parameter_robustness': 'NOT TESTED (screen stage)', 'cost_robustness': r['cost_stress_status'],
        'high_vol_result': 'NA', 'drawdown_correlation': r['drawdown_correlation'],
        'portfolio_fit': r['diversification_status'], 'final_classification': r['final_classification'],
        'rejection_reason': r['notes'],
        'preregistered': 'PARTIAL (screening bar pre-registered; individual cells not pre-selected before the sweep)',
        'post_result_modification': 'NO',
        'notes': 'Exploratory calendar-drift screen cell, not a standalone confirmatory candidate',
    })

p33_reg = pd.read_csv(f'{REPO}/reports/phase33_candidate_registry.csv')
p33_res = pd.read_csv(f'{REPO}/reports/phase33_candidate_results.csv')
p33_rob = pd.read_csv(f'{REPO}/reports/phase33_robustness_results.csv')
p33_rank = pd.read_csv(f'{REPO}/reports/phase33_final_rankings.csv')
for i in range(len(p33_reg)):
    r = p33_reg.iloc[i]
    res = p33_res.iloc[i]
    rob = p33_rob.iloc[i]
    rank = p33_rank.iloc[i]
    rows.append({
        'experiment_id': r['experiment_id'], 'phase': 'Phase33', 'hypothesis_id': res['candidate_id'],
        'strategy_family': r['strategy_family'], 'instrument': r['instrument'], 'mechanism': r['strategy_family'],
        'session': r['session'], 'timeframe': r['timeframe'],
        'train_period': '2023-01-01 to 2024-08-31', 'validation_period': '2024-09-01 to 2025-04-30',
        'oos_period': '2025-05-01 to 2026-08-14',
        'oos_trades': res['oos_trades'], 'oos_pf': res['oos_pf'], 'oos_expectancy': res['oos_expectancy_R'],
        'oos_total_R': res['oos_total_R'], 'oos_max_dd': 'NA',
        'oos_first_half_R': rob['oos_h1_expectancy_R'], 'oos_second_half_R': rob['oos_h2_expectancy_R'],
        'parameter_robustness': 'FAIL (sign-inconsistent)' if not rob['param_sensitivity_sign_consistent'] else 'PASS',
        'cost_robustness': rank['cost_stress_gate'],
        'high_vol_result': rank['high_vol_gate'], 'drawdown_correlation': rank['drawdown_diversification_gate'],
        'portfolio_fit': rank['portfolio_fit_note'], 'final_classification': rank['final_classification'],
        'rejection_reason': rank['primary_rejection_reason'],
        'preregistered': 'YES (frozen commit 8bcd30e before backtesting)',
        'post_result_modification': 'NO', 'notes': 'Confirmatory pre-registered candidate',
    })

p35_reg = pd.read_csv(f'{REPO}/reports/phase35_candidate_registry.csv')
p35_res = pd.read_csv(f'{REPO}/reports/phase35_candidate_results.csv')
p35_rank = pd.read_csv(f'{REPO}/reports/phase35_final_rankings.csv')
p35_oos = pd.read_csv(f'{REPO}/reports/phase35_oos_consistency.csv')
for i in range(len(p35_reg)):
    r = p35_reg.iloc[i]
    res = p35_res.iloc[i]
    rank = p35_rank.iloc[i]
    oos_c = p35_oos.iloc[i]
    rows.append({
        'experiment_id': r['hypothesis_id'], 'phase': 'Phase35', 'hypothesis_id': res['candidate_id'],
        'strategy_family': r['strategy_family'], 'instrument': r['instrument'], 'mechanism': r['strategy_family'],
        'session': r['session'], 'timeframe': r['timeframe'],
        'train_period': '2023-01-01 to 2024-08-31', 'validation_period': '2024-09-01 to 2025-04-30',
        'oos_period': '2025-05-01 to 2026-08-14',
        'oos_trades': res['oos_trades'], 'oos_pf': res['oos_pf'], 'oos_expectancy': res['oos_expectancy_R'],
        'oos_total_R': res['oos_total_R'], 'oos_max_dd': 'NA',
        'oos_first_half_R': oos_c.get('oos_h1_expectancy_R', 'NA'), 'oos_second_half_R': oos_c.get('oos_h2_expectancy_R', 'NA'),
        'parameter_robustness': 'FAIL (negative across all perturbations, no positive edge to be robust)',
        'cost_robustness': 'NOT TESTED (Gate 1 failed)', 'high_vol_result': 'see phase35_regime_analysis.csv',
        'drawdown_correlation': 'NOT TESTED (Gate 1 failed)', 'portfolio_fit': 'NOT TESTED (Gate 1 failed)',
        'final_classification': rank['final_classification'], 'rejection_reason': rank['primary_rejection_reason'],
        'preregistered': 'YES (frozen commit 7821cd7 before backtesting)', 'post_result_modification': 'NO',
        'notes': 'Confirmatory pre-registered candidate',
    })

rows.append({
    'experiment_id': 'EXP-121 (Phase30 origin)', 'phase': 'Phase30/32/34/35 (cross-phase standing candidate)',
    'hypothesis_id': 'AUDUSD_MONDAY_LONG',
    'strategy_family': 'calendar_drift', 'instrument': 'AUDUSD', 'mechanism': 'calendar_drift',
    'session': 'Monday full session', 'timeframe': 'D1', 'train_period': '2023-01-01 to 2025-01-01',
    'validation_period': 'N/A -- single IS/OOS split', 'oos_period': '2025-01-01 to 2026-08-14',
    'oos_trades': 84, 'oos_pf': 3.070, 'oos_expectancy': 'NA', 'oos_total_R': 'NA', 'oos_max_dd': 'NA',
    'oos_first_half_R': 'NA', 'oos_second_half_R': 'NA',
    'parameter_robustness': 'NOT FORMALLY TESTED (no +/-20% perturbation run in Phase30)',
    'cost_robustness': 'PASS (OOS PF 2.647 at 2x cost stress)',
    'high_vol_result': 'STRONG (best of 3 vol terciles, mean R +0.248/trade in HIGH bucket, Phase32)',
    'drawdown_correlation': 'CORRELATED-LEANING (0.29 to control, above the 0.192 control-internal average)',
    'portfolio_fit': 'Not formally integration-tested at the CONTROL+CANDIDATE level',
    'final_classification': 'E. PROMISING -- requires more validation (unchanged since Phase30)',
    'rejection_reason': 'N/A -- not rejected, but not promoted: IS t=1.65 did not clear the pre-registered IS+OOS t>=2.0 bar despite OOS t=4.15',
    'preregistered': 'YES (Phase30 screening bar)', 'post_result_modification': 'NO',
    'notes': 'The projects one standing PROMISING candidate across all phases; not re-tested or modified in Phase36',
})

ledger = pd.DataFrame(rows)
ledger.to_csv(f'{REPO}/reports/phase36_research_ledger.csv', index=False)
print(f"Total ledger rows: {len(ledger)}")
print(ledger['phase'].value_counts())
