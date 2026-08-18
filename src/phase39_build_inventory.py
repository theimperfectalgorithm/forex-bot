"""
Phase 39 -- build reports/phase39_fx_research_inventory.csv by extending
the Phase36 68-row consolidated ledger (reports/phase36_research_ledger.csv)
with Phase37's AUDUSD full-validation update (same hypothesis, richer
result, not a new row) and Phase38's H1/H2 (2 new rows). No new backtest
performed here -- this is a reconciliation/audit script only.
"""
import csv
from pathlib import Path

REPO = Path(__file__).parent.parent
SRC = REPO / 'reports' / 'phase36_research_ledger.csv'
OUT = REPO / 'reports' / 'phase39_fx_research_inventory.csv'

with open(SRC, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 68, f"expected 68 rows from Phase36 ledger, got {len(rows)}"

# --- update the AUDUSD Monday LONG row with Phase37's full-validation result ---
updated = False
for r in rows:
    if r['hypothesis_id'] == 'AUDUSD_MONDAY_LONG':
        r['phase'] = 'Phase30/32/34/35/37 (cross-phase standing candidate, fully validated Phase37)'
        r['oos_expectancy'] = '0.2548'
        r['oos_total_R'] = '21.4'
        r['oos_max_dd'] = '-2.87'
        r['oos_first_half_R'] = '+0.2323R (42 trades, PF 2.678)'
        r['oos_second_half_R'] = '+0.2773R (42 trades, PF 3.572)'
        r['parameter_robustness'] = 'PASS (ATR-window +/-20%: PF 3.051/3.070/3.152, no sign reversal -- Phase37, disclosed limited informativeness)'
        r['cost_robustness'] = 'PASS (OOS PF 2.647 at 2x cost -- Phase37, matches Phase32 figure)'
        r['high_vol_result'] = 'STRONG (Phase37: 20 trades, PF 6.25, expectancy +0.3537R, best of 3 terciles)'
        r['drawdown_correlation'] = 'CORRELATED (Phase37 formal test: normal-day corr 0.228, drawdown-day corr 0.742, 9-day overlap)'
        r['portfolio_fit'] = 'Formally integration-tested (Phase37): control max_dd -14.53 -> -15.24 at 1.0x weight (worse)'
        r['final_classification'] = 'F. REJECTED -- POOR DRAWDOWN DIVERSIFICATION (Phase37, supersedes Phase30-35 E.PROMISING status)'
        r['rejection_reason'] = 'Formally validated Phase37: passes every gate (edge, OOS consistency, param robustness, cost stress, 5/5 historical regimes, HIGH-vol) except drawdown correlation to the 6-strategy control (0.742 vs 0.228 normal, exceeding the 0.15 threshold)'
        r['post_result_modification'] = 'NO (Phase37 was a formal, preregistered full-validation extension of an already-frozen candidate definition, not a post-hoc edit)'
        r['notes'] = 'Superseded by Phase37 full validation -- see reports/phase37_master_report.md. No longer PROMISING; formally REJECTED.'
        updated = True
assert updated, "AUDUSD_MONDAY_LONG row not found -- STOP"

# --- append Phase38 H1/H2 ---
new_rows = [
    {
        'experiment_id': 'EXP-135', 'phase': 'Phase38', 'hypothesis_id': 'H1_CROSS_SECTIONAL_FX',
        'strategy_family': 'cross_sectional_relative_momentum', 'instrument': 'SYNTHETIC (7 USD-pair legs: EURUSD/GBPUSD/AUDUSD/NZDUSD/USDJPY/USDCAD/USDCHF)',
        'mechanism': 'cross_sectional_relative_momentum', 'session': 'session-independent (weekly rebalance)', 'timeframe': 'D1/weekly',
        'train_period': '2023-01-01 to 2025-01-01', 'validation_period': 'N/A -- two-way split (matches Phase37 AUDUSD convention)',
        'oos_period': '2025-01-01 to 2026-08-14', 'oos_trades': '84', 'oos_pf': '0.649', 'oos_expectancy': '-0.1648',
        'oos_total_R': '-13.84', 'oos_max_dd': '-14.61', 'oos_first_half_R': '-0.1976R (42 trades)', 'oos_second_half_R': '-0.132R (42 trades)',
        'parameter_robustness': 'PASS/no reversal (lookback 16/20/24d: PF 0.582/0.649/0.517, negative throughout)',
        'cost_robustness': 'FAIL (PF 0.608 at 2x, already below 1.0 before stress)',
        'high_vol_result': 'WEAK (15 trades, expectancy -0.6319R, worst regime)',
        'drawdown_correlation': 'CORRELATED (0.611 drawdown-day vs 0.136 normal, 8-day overlap at the preregistered floor)',
        'portfolio_fit': 'Worsens control at every weight (total_R and max_dd both deteriorate)',
        'final_classification': 'B. REJECTED -- NO CREDIBLE OOS EDGE',
        'rejection_reason': 'OOS PF 0.649, negative in IS and OOS, negative across all lookback perturbations and cost stress',
        'preregistered': 'YES (frozen commit af03e04 before backtesting)', 'post_result_modification': 'NO',
        'notes': 'Structurally A. GENUINELY DISTINCT (first cross-sectional-ranking hypothesis in the ledger); see reports/phase38_master_report.md',
    },
    {
        'experiment_id': 'EXP-136', 'phase': 'Phase38', 'hypothesis_id': 'H2_SESSION_STRUCTURE_ASIAN_BREAKOUT',
        'strategy_family': 'session_transition_breakout_continuation', 'instrument': 'EURUSD/GBPUSD/AUDUSD',
        'mechanism': 'asian_range_breakout_continuation_london_to_ny', 'session': 'Asian range -> London open trigger -> NY close exit (multi-session)', 'timeframe': 'H1',
        'train_period': '2023-01-01 to 2025-01-01', 'validation_period': 'N/A -- two-way split (matches Phase37 AUDUSD convention)',
        'oos_period': '2025-01-01 to 2026-08-14', 'oos_trades': '458', 'oos_pf': '0.798', 'oos_expectancy': '-0.119',
        'oos_total_R': '-54.5', 'oos_max_dd': '-57.62', 'oos_first_half_R': '-0.1157R (229 trades)', 'oos_second_half_R': '-0.1223R (229 trades)',
        'parameter_robustness': 'FAIL (Asian-window +/-20%: PF 1.050/0.798/degenerate-0-trades, sign reversal)',
        'cost_robustness': 'FAIL (PF 0.697 at 2x, already below 1.0 before stress)',
        'high_vol_result': 'nominally STRONG (198 trades, expectancy +0.0025R -- economically negligible)',
        'drawdown_correlation': 'CORRELATED (0.269 drawdown-day vs -0.085 normal, 28-day overlap, well-sampled)',
        'portfolio_fit': 'Materially worsens control (max_dd -14.53 -> -22.78 at 1.0x weight -- largest degradation of any Phase37/38 candidate)',
        'final_classification': 'B. REJECTED -- NO CREDIBLE OOS EDGE',
        'rejection_reason': 'OOS PF 0.798 on the largest sample of the phase (458 trades), negative in IS and OOS, sign-reversed under parameter perturbation',
        'preregistered': 'YES (frozen commit af03e04, one disclosed pre-results amendment 111e09d before backtesting)', 'post_result_modification': 'NO (amendment was pre-results per Phase38 preregistration)',
        'notes': 'Structurally B. RELATED BUT MEANINGFULLY DIFFERENT from AMR/Phase35 NY hypotheses; see reports/phase38_master_report.md',
    },
]

all_rows = rows + new_rows
fieldnames = list(rows[0].keys())
with open(OUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(all_rows)

print(f"Wrote {len(all_rows)} rows to {OUT} (68 Phase36 rows, AUDUSD row updated in place, +2 Phase38 rows)")
