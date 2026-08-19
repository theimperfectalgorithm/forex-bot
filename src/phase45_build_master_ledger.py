"""
Phase 45 -- extend Phase39's 70-row research inventory with Phase40's
HIGH-volatility-conditioned candidate to form the 71-row master ledger.
No new backtest performed here -- pure reconciliation.
"""
import csv
from pathlib import Path

REPO = Path(__file__).parent.parent
SRC = REPO / 'reports' / 'phase39_fx_research_inventory.csv'
OUT = REPO / 'reports' / 'phase45_research_master_ledger.csv'

with open(SRC, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 70, f"expected 70 rows from Phase39 inventory, got {len(rows)}"

new_row = {
    'experiment_id': 'EXP-138', 'phase': 'Phase40', 'hypothesis_id': 'H_VOLATILITY_CONDITIONED_TREND_CONTINUATION',
    'strategy_family': 'volatility_conditioned_trend_continuation', 'instrument': 'EURUSD/GBPUSD/AUDUSD/USDCAD',
    'mechanism': 'volatility_conditioned_trend_continuation', 'session': 'New York (13:00-21:00 UTC-server-hour)', 'timeframe': 'H1',
    'train_period': '2023-01-01 to 2024-08-31', 'validation_period': '2024-09-01 to 2025-04-30',
    'oos_period': '2025-05-01 to 2026-08-14', 'oos_trades': '2228', 'oos_pf': '0.668', 'oos_expectancy': '-0.1767',
    'oos_total_R': '-393.71', 'oos_max_dd': '-395.05', 'oos_first_half_R': '-0.1384R (1113 trades)', 'oos_second_half_R': '-0.2149R (1115 trades)',
    'parameter_robustness': 'PASS/no reversal (ATR11/14/17: PF 0.718/0.668/0.691, negative throughout)',
    'cost_robustness': 'FAIL (PF 0.539 at 2x, already below 1.0 before stress)',
    'high_vol_result': 'C. MATERIALLY DETERIORATES IN HIGH VOLATILITY (this candidate trades ONLY the HIGH state by construction)',
    'drawdown_correlation': 'CORRELATED (0.251 drawdown-day vs 0.090 normal-day, 26-day overlap)',
    'portfolio_fit': 'Most severe portfolio-integration failure of any Phase37-40 candidate (control total_R 126.72 -> combined -266.99 at 1.0x weight)',
    'final_classification': 'B. REJECTED -- NO CREDIBLE OOS EDGE',
    'rejection_reason': 'OOS PF 0.668 on 2228 trades, the largest and most decisive OOS sample tested in this project to date; negative in both sub-halves, all historical regimes, and worsening toward the present',
    'preregistered': 'YES (frozen commit bea0a31 before backtesting)', 'post_result_modification': 'NO',
    'notes': 'First volatility-activation-gated hypothesis in the ledger (structurally B.RELATED BUT MEANINGFULLY DIFFERENT); see reports/phase40_master_report.md',
}

all_rows = rows + [new_row]
fieldnames = list(rows[0].keys())
with open(OUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(all_rows)

print(f"Wrote {len(all_rows)} rows to {OUT} (70 Phase39 rows + 1 Phase40 row)")
