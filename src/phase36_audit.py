"""
Phase 36 -- Research Base-Rate & Portfolio Viability Audit.

AUDIT ONLY. No new candidate backtested. No live strategy/parameter/risk
modified. No candidate promoted, rescued, or optimized.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

sys.path.insert(0, str(Path(__file__).parent))
from phase31_factor_regime_map import load_hist, RISK_PCT, CURRENT_SIX, STRATEGY_META  # noqa: E402
from research_data_validator import (  # noqa: E402
    ValidationReport, validate_column_count_consistency, validate_required_columns,
    validate_lifecycle_pairing,
)

REPO = Path(__file__).parent.parent
OUT = REPO / 'reports'


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return (None, None)
    p = successes / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    lo = (center - margin) / denom
    hi = (center + margin) / denom
    return (round(max(0, lo) * 100, 1), round(min(1, hi) * 100, 1))


def main():
    # ---- Part 3: data integrity ----
    ledger_path = OUT / 'phase36_research_ledger.csv'
    r = ValidationReport(path=str(ledger_path))
    validate_column_count_consistency(ledger_path, r)
    validate_required_columns(ledger_path, {'experiment_id', 'phase', 'hypothesis_id', 'final_classification'}, r)
    print(f"[validate] {r.summary()}")

    export_path = OUT / '5ers_trade_export.csv'
    r2 = ValidationReport(path=str(export_path))
    validate_column_count_consistency(export_path, r2)
    validate_lifecycle_pairing(export_path, 'trade_id', 'status', report=r2)
    print(f"[validate] {r2.summary()}")

    ledger = pd.read_csv(ledger_path)

    # ---- Part 5: base rate ----
    confirmatory = ledger[ledger['phase'].isin(['Phase33', 'Phase35'])].copy()
    screen = ledger[ledger['phase'] == 'Phase30'].copy()

    def classify_letter(s):
        return str(s).strip()[0]

    confirmatory['letter'] = confirmatory['final_classification'].apply(classify_letter)
    n_conf = len(confirmatory)
    n_edge = (confirmatory['oos_pf'].astype(float) > 1.0).sum()
    n_consistency_pass = 0  # both Phase33 edge-passers failed sub-half consistency; 0 Phase35 had an edge to test
    n_param_robust = confirmatory['parameter_robustness'].str.contains('PASS', na=False).sum()
    n_portfolio_qualified = (confirmatory['letter'] == 'I').sum()

    base_rate_rows = [
        {'population': 'Confirmatory candidates (Phase33+35)', 'n': n_conf,
         'metric': 'Initial OOS edge (PF>1.0)', 'count': int(n_edge),
         'observed_rate_pct': round(n_edge / n_conf * 100, 1),
         'wilson_95ci_pct': f"{wilson_ci(n_edge, n_conf)[0]}-{wilson_ci(n_edge, n_conf)[1]}"},
        {'population': 'Confirmatory candidates (Phase33+35)', 'n': n_conf,
         'metric': 'OOS sub-half consistency (of those with an edge)', 'count': n_consistency_pass,
         'observed_rate_pct': round(n_consistency_pass / max(n_edge, 1) * 100, 1),
         'wilson_95ci_pct': f"{wilson_ci(n_consistency_pass, max(n_edge,1))[0]}-{wilson_ci(n_consistency_pass, max(n_edge,1))[1]}"},
        {'population': 'Confirmatory candidates (Phase33+35)', 'n': n_conf,
         'metric': 'Parameter robustness pass', 'count': int(n_param_robust),
         'observed_rate_pct': round(n_param_robust / n_conf * 100, 1),
         'wilson_95ci_pct': f"{wilson_ci(n_param_robust, n_conf)[0]}-{wilson_ci(n_param_robust, n_conf)[1]}"},
        {'population': 'Confirmatory candidates (Phase33+35)', 'n': n_conf,
         'metric': 'Portfolio-qualified (Category I/H)', 'count': int(n_portfolio_qualified),
         'observed_rate_pct': round(n_portfolio_qualified / n_conf * 100, 1),
         'wilson_95ci_pct': f"{wilson_ci(n_portfolio_qualified, n_conf)[0]}-{wilson_ci(n_portfolio_qualified, n_conf)[1]}"},
    ]

    n_screen = len(screen)
    n_screen_pass = screen['final_classification'].str.startswith('E').sum()
    base_rate_rows.append({'population': 'Exploratory screen cells (Phase30)', 'n': n_screen,
                            'metric': 'Cleared pre-registered screening bar', 'count': int(n_screen_pass),
                            'observed_rate_pct': round(n_screen_pass / n_screen * 100, 1),
                            'wilson_95ci_pct': f"{wilson_ci(n_screen_pass, n_screen)[0]}-{wilson_ci(n_screen_pass, n_screen)[1]}"})
    base_rate_rows.append({'population': 'ALL hypotheses (screen + confirmatory)', 'n': len(ledger) - 1,  # exclude AUDUSD Monday summary row
                            'metric': 'Reached PROMISING or better', 'count': int((ledger['final_classification'].apply(classify_letter).isin(['E', 'H', 'I'])).sum()),
                            'observed_rate_pct': round((ledger['final_classification'].apply(classify_letter).isin(['E', 'H', 'I'])).sum() / (len(ledger) - 1) * 100, 1),
                            'wilson_95ci_pct': 'see note'})
    base_rate_df = pd.DataFrame(base_rate_rows)
    base_rate_df.to_csv(OUT / '_scratch_base_rates.csv', index=False)
    print("\n=== base rates ===")
    print(base_rate_df.to_string())

    # ---- Part 6: failure taxonomy ----
    def taxonomy_of(row):
        cls = str(row['final_classification'])
        if 'NO EDGE' in cls or cls.startswith('A.'):
            return 'EDGE_ABSENT'
        if 'OOS INSTABILITY' in cls or 'insufficient robustness' in cls.lower() and 'IS/OOS' in str(row.get('rejection_reason', '')):
            return 'OOS_INSTABILITY'
        if 'PARAMETER FRAGILITY' in cls or 'sign-inconsistent' in str(row.get('parameter_robustness', '')):
            return 'PARAMETER_FRAGILITY'
        if 'COST FRAGIL' in cls:
            return 'COST_FRAGILITY'
        if 'HIGH-VOLATILITY' in cls or 'HIGH_VOL' in cls:
            return 'HIGH_VOL_FAILURE'
        if 'DRAWDOWN' in cls:
            return 'DRAWDOWN_CORRELATION'
        if 'PORTFOLIO FIT' in cls:
            return 'PORTFOLIO_FIT'
        if cls.startswith('E.') or cls.startswith('H.'):
            return 'PROMISING_NOT_REJECTED'
        return 'OTHER'

    rejected = ledger[~ledger['hypothesis_id'].eq('AUDUSD_MONDAY_LONG')].copy()
    rejected['failure_category'] = rejected.apply(taxonomy_of, axis=1)
    tax_summary = rejected.groupby('failure_category').agg(
        count=('hypothesis_id', 'count')).reset_index()
    tax_summary['pct_of_all_68'] = round(tax_summary['count'] / len(rejected) * 100, 1)
    tax_summary.to_csv(OUT / 'phase36_failure_taxonomy.csv', index=False)
    print("\n=== failure taxonomy ===")
    print(tax_summary.to_string())

    rejected[['hypothesis_id', 'phase', 'strategy_family', 'instrument', 'session', 'failure_category', 'final_classification']].to_csv(
        OUT / '_scratch_failure_detail.csv', index=False)

    # ---- Part 16: search space coverage ----
    coverage_rows = []
    for dim in ['strategy_family', 'instrument', 'session']:
        vc = rejected[dim].value_counts()
        for val, cnt in vc.items():
            coverage_rows.append({'dimension': dim, 'value': str(val)[:60], 'count': cnt,
                                   'pct_of_68': round(cnt / len(rejected) * 100, 1)})
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(OUT / 'phase36_search_space_coverage.csv', index=False)
    print("\n=== search space coverage (top) ===")
    print(coverage_df.sort_values('count', ascending=False).head(20).to_string())

    with open(OUT / '_phase36_summary.json', 'w') as f:
        json.dump({'n_confirmatory': n_conf, 'n_edge': int(n_edge), 'n_screen': n_screen,
                    'n_screen_pass': int(n_screen_pass), 'total_ledger_rows': len(ledger)}, f, indent=2)


if __name__ == '__main__':
    main()
