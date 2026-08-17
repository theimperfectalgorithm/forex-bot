"""Phase 37 Track B -- build the return-stream classification CSVs via
csv.DictWriter (guaranteed proper quoting -- avoids the exact incident class
this project's research_data_validator exists to catch)."""
import csv
from pathlib import Path

REPO = Path(__file__).parent.parent
OUT = REPO / 'reports'

FIELDNAMES = ['class_name', 'return_driver', 'underlying_market', 'mechanism',
              'expected_correlation_to_current_book', 'expected_drawdown_correlation',
              'expected_high_vol_behaviour', 'session_dependency', 'data_quality',
              'historical_data_availability', 'implementation_complexity', 'overfitting_risk',
              'execution_complexity', 'transaction_cost_sensitivity', 'research_cost',
              'potential_independence', 'reason_to_test', 'reason_not_to_test']

ROWS = [
    {'class_name': 'Cross-asset relationships',
     'return_driver': 'Lead-lag between a commodity/rate proxy and a currency',
     'underlying_market': 'FX + a second asset class', 'mechanism': 'Statistical/predictive relationship',
     'expected_correlation_to_current_book': 'LOW', 'expected_drawdown_correlation': 'UNKNOWN',
     'expected_high_vol_behaviour': 'UNKNOWN', 'session_dependency': 'MEDIUM', 'data_quality': 'MEDIUM',
     'historical_data_availability': 'UNKNOWN (second data series not yet sourced in this project)',
     'implementation_complexity': 'MEDIUM', 'overfitting_risk': 'MEDIUM', 'execution_complexity': 'MEDIUM',
     'transaction_cost_sensitivity': 'MEDIUM', 'research_cost': 'MEDIUM', 'potential_independence': 'HIGH',
     'reason_to_test': 'Outside the price-action/technical-indicator space used exclusively so far',
     'reason_not_to_test': "Second data series not yet confirmed available in this project's MT5 toolchain"},
    {'class_name': 'Commodity-based return streams',
     'return_driver': 'Supply/inventory/macro-driven commodity moves (beyond the already-tested XAUUSD)',
     'underlying_market': 'Commodities (e.g. crude oil)', 'mechanism': 'Directional/breakout on a distinct fundamental driver',
     'expected_correlation_to_current_book': 'LOW', 'expected_drawdown_correlation': 'UNKNOWN',
     'expected_high_vol_behaviour': 'MEDIUM (event-driven)', 'session_dependency': 'MEDIUM', 'data_quality': 'MEDIUM',
     'historical_data_availability': 'UNKNOWN (not yet checked for this specific instrument)',
     'implementation_complexity': 'LOW', 'overfitting_risk': 'MEDIUM', 'execution_complexity': 'LOW',
     'transaction_cost_sensitivity': 'MEDIUM', 'research_cost': 'LOW', 'potential_independence': 'MEDIUM',
     'reason_to_test': 'Low-cost extension of already-available MT5 infrastructure',
     'reason_not_to_test': "May replicate XAUUSD's own diagnosed macro/hedge correlation-to-portfolio risk unless specifically checked"},
    {'class_name': 'Index-based return streams', 'return_driver': 'Equity-risk-sentiment-driven moves',
     'underlying_market': 'Equity indices (CFD)', 'mechanism': 'Session-timed directional/momentum',
     'expected_correlation_to_current_book': 'LOW', 'expected_drawdown_correlation': 'UNKNOWN',
     'expected_high_vol_behaviour': 'MEDIUM-HIGH (event windows)', 'session_dependency': 'HIGH (NY open, data releases)',
     'data_quality': 'UNKNOWN', 'historical_data_availability': 'UNKNOWN (index CFD access not yet confirmed in this project)',
     'implementation_complexity': 'LOW-MEDIUM', 'overfitting_risk': 'MEDIUM', 'execution_complexity': 'MEDIUM',
     'transaction_cost_sensitivity': 'MEDIUM', 'research_cost': 'MEDIUM', 'potential_independence': 'HIGH',
     'reason_to_test': 'A fully different asset class never explored in this project',
     'reason_not_to_test': 'Data access unconfirmed'},
    {'class_name': 'Cross-sectional FX', 'return_driver': 'Relative strength/momentum ranking across a currency basket',
     'underlying_market': 'Multiple FX pairs (basket)', 'mechanism': 'Rank-based relative-strength continuation',
     'expected_correlation_to_current_book': 'LOW-MEDIUM', 'expected_drawdown_correlation': 'UNKNOWN',
     'expected_high_vol_behaviour': 'MEDIUM', 'session_dependency': 'LOW',
     'data_quality': 'HIGH (already-available MT5 multi-pair feed)',
     'historical_data_availability': 'HIGH (same feed already used since Phase 30)',
     'implementation_complexity': 'MEDIUM', 'overfitting_risk': 'MEDIUM', 'execution_complexity': 'MEDIUM',
     'transaction_cost_sensitivity': 'LOW-MEDIUM', 'research_cost': 'MEDIUM', 'potential_independence': 'MEDIUM-HIGH',
     'reason_to_test': "Directly extends this project's own already-validated CADJPY cross-sectional momentum finding (PROJECT_REPORT.md phase 6) -- the most evidence-backed alternative",
     'reason_not_to_test': 'Basket construction and rebalancing rules need their own dedicated pre-registration'},
    {'class_name': 'Relative-value / spread structures',
     'return_driver': 'Convergence/divergence between two correlated instruments',
     'underlying_market': 'Two correlated FX pairs', 'mechanism': 'Spread mean-reversion or breakout',
     'expected_correlation_to_current_book': 'LOW-MEDIUM', 'expected_drawdown_correlation': 'UNKNOWN',
     'expected_high_vol_behaviour': 'UNKNOWN', 'session_dependency': 'LOW',
     'data_quality': 'HIGH (already-available data)', 'historical_data_availability': 'HIGH',
     'implementation_complexity': 'MEDIUM', 'overfitting_risk': 'MEDIUM-HIGH', 'execution_complexity': 'MEDIUM',
     'transaction_cost_sensitivity': 'MEDIUM', 'research_cost': 'MEDIUM', 'potential_independence': 'MEDIUM-HIGH',
     'reason_to_test': 'Structurally distinct return driver (relationship, not outright direction) with potentially strong independence if properly currency-neutral',
     'reason_not_to_test': 'Spread relationships can be unstable and easy to curve-fit in-sample; needs new stability-testing infrastructure'},
    {'class_name': 'Volatility-conditioned systems',
     'return_driver': 'Positioning based on a volatility-regime signal as the PRIMARY trigger (not a secondary classification)',
     'underlying_market': 'A dedicated volatility index or realized-vol term structure', 'mechanism': 'Regime-transition timing',
     'expected_correlation_to_current_book': 'LOW', 'expected_drawdown_correlation': 'UNKNOWN',
     'expected_high_vol_behaviour': 'HIGH (by construction -- directly targets Phase32 Priority 1)',
     'session_dependency': 'LOW', 'data_quality': 'LOW',
     'historical_data_availability': "NO -- confirmed data gap (a genuine volatility index/options-derived series is not currently available via this project's MT5 feed)",
     'implementation_complexity': 'HIGH (data sourcing is the primary barrier)', 'overfitting_risk': 'MEDIUM',
     'execution_complexity': 'MEDIUM', 'transaction_cost_sensitivity': 'MEDIUM', 'research_cost': 'HIGH',
     'potential_independence': 'MEDIUM-HIGH',
     'reason_to_test': "The single most direct match to Phase32's own top-ranked factor (HIGH-vol compatibility)",
     'reason_not_to_test': "Required data does not appear to already be available in this project's toolchain -- a real, disclosed gap"},
    {'class_name': 'Multi-asset momentum', 'return_driver': 'Momentum signal combined across FX + commodities + (if available) indices',
     'underlying_market': 'Multiple asset classes', 'mechanism': 'Diversified momentum sleeve',
     'expected_correlation_to_current_book': 'LOW', 'expected_drawdown_correlation': 'UNKNOWN',
     'expected_high_vol_behaviour': 'MEDIUM-HIGH', 'session_dependency': 'LOW',
     'data_quality': 'UNKNOWN (depends on classes 1-3)', 'historical_data_availability': 'UNKNOWN (depends on classes 1-3)',
     'implementation_complexity': 'HIGH', 'overfitting_risk': 'MEDIUM-HIGH', 'execution_complexity': 'HIGH',
     'transaction_cost_sensitivity': 'MEDIUM', 'research_cost': 'HIGH',
     'potential_independence': 'HIGH (highest theoretical diversification ceiling on this list)',
     'reason_to_test': 'Highest theoretical diversification ceiling by combining independent signal sources',
     'reason_not_to_test': "Highest implementation cost; depends on solving classes 1-3's data-access questions first"},
    {'class_name': 'Event/macro-conditioned systems',
     'return_driver': 'Entries conditioned on a macro regime classifier (rate-cycle, risk-on/risk-off)',
     'underlying_market': 'FX (any pair) + a macro/sentiment classifier', 'mechanism': 'Regime-filtered directional entry',
     'expected_correlation_to_current_book': 'LOW', 'expected_drawdown_correlation': 'UNKNOWN',
     'expected_high_vol_behaviour': 'HIGH (by construction -- directly targets Phase32 Priority 2)',
     'session_dependency': 'LOW', 'data_quality': 'LOW',
     'historical_data_availability': 'NO -- confirmed data gap (Phase31 already found risk-off regime could not be reconstructed from available data)',
     'implementation_complexity': 'HIGH', 'overfitting_risk': 'MEDIUM (if the classifier itself is kept simple and pre-registered)',
     'execution_complexity': 'MEDIUM', 'transaction_cost_sensitivity': 'MEDIUM', 'research_cost': 'HIGH',
     'potential_independence': 'MEDIUM-HIGH',
     'reason_to_test': "Would directly resolve a standing, repeatedly-flagged gap (Phase31's own NOT AVAILABLE finding for the risk-off stress scenario) and the single most direct match to Phase32 Priority 2",
     'reason_not_to_test': 'The classifier itself must be built and validated first -- a nontrivial prerequisite project'},
    {'class_name': 'Session-specific structures',
     'return_driver': 'A specific recurring calendar/event structure (not a generic session window)',
     'underlying_market': 'FX (any pair)', 'mechanism': 'Event-conditioned entry',
     'expected_correlation_to_current_book': 'LOW-MEDIUM', 'expected_drawdown_correlation': 'UNKNOWN',
     'expected_high_vol_behaviour': 'MEDIUM', 'session_dependency': 'HIGH',
     'data_quality': 'MEDIUM-HIGH (core/news_calendar.py already exists for the live news-blackout gate)',
     'historical_data_availability': 'HIGH', 'implementation_complexity': 'LOW-MEDIUM', 'overfitting_risk': 'MEDIUM',
     'execution_complexity': 'LOW-MEDIUM', 'transaction_cost_sensitivity': 'LOW-MEDIUM', 'research_cost': 'LOW-MEDIUM',
     'potential_independence': 'MEDIUM',
     'reason_to_test': 'Builds directly on existing project infrastructure (core/news_calendar.py) -- lowest implementation cost of the data-available options',
     'reason_not_to_test': 'Calendar effects can be numerous and easy to data-mine without the same pre-registration discipline already established'},
    {'class_name': 'Other structurally distinct mechanisms',
     'return_driver': 'Options-derived positioning; order-flow/microstructure; extended seasonality',
     'underlying_market': 'Varies', 'mechanism': 'Varies', 'expected_correlation_to_current_book': 'UNKNOWN',
     'expected_drawdown_correlation': 'UNKNOWN', 'expected_high_vol_behaviour': 'UNKNOWN', 'session_dependency': 'UNKNOWN',
     'data_quality': 'LOW-UNKNOWN', 'historical_data_availability': 'UNKNOWN (tick-level/DOM data likely beyond current MT5 access for order-flow specifically)',
     'implementation_complexity': 'UNKNOWN', 'overfitting_risk': 'HIGH (for microstructure specifically)',
     'execution_complexity': 'UNKNOWN', 'transaction_cost_sensitivity': 'UNKNOWN', 'research_cost': 'UNKNOWN',
     'potential_independence': 'UNKNOWN', 'reason_to_test': 'Catch-all for directions not yet scoped in detail',
     'reason_not_to_test': 'Each would need its own dedicated scoping project before any ranking is possible'},
]


def main():
    with open(OUT / 'phase37_return_stream_classes.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in ROWS:
            w.writerow(r)
    print(f"written {len(ROWS)} rows")


if __name__ == '__main__':
    main()
