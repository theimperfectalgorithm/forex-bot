"""Phase 37 Track B -- rebuild phase37_data_availability.csv via
csv.DictWriter (guaranteed proper quoting)."""
import csv
from pathlib import Path

REPO = Path(__file__).parent.parent
OUT = REPO / 'reports'

FIELDNAMES = ['class_name', 'historical_data_source', 'minimum_usable_history', 'timeframe_availability',
              'bid_ask_availability', 'spread_availability', 'corporate_action_concerns', 'rollover_concerns',
              'contract_change_concerns', 'survivorship_concerns', 'timestamp_quality',
              'data_normalization_requirements', 'overall_data_readiness']

ROWS = [
    {'class_name': 'Cross-asset relationships', 'historical_data_source': 'UNKNOWN -- second data series not yet sourced',
     'minimum_usable_history': 'UNKNOWN', 'timeframe_availability': 'UNKNOWN', 'bid_ask_availability': 'UNKNOWN',
     'spread_availability': 'UNKNOWN', 'corporate_action_concerns': 'LOW (FX/commodity proxies typically N/A)',
     'rollover_concerns': 'MEDIUM (if commodity futures-based)', 'contract_change_concerns': 'UNKNOWN',
     'survivorship_concerns': 'LOW', 'timestamp_quality': 'UNKNOWN (depends on source)',
     'data_normalization_requirements': "MEDIUM (aligning two series' timestamps/lags)",
     'overall_data_readiness': 'NOT READY -- requires new data sourcing'},
    {'class_name': 'Commodity-based return streams',
     'historical_data_source': 'Likely same MT5 feed already used for XAUUSD -- specific instrument not yet checked',
     'minimum_usable_history': 'UNKNOWN', 'timeframe_availability': 'Likely good (same feed pattern as existing instruments)',
     'bid_ask_availability': 'Likely yes (same feed pattern)', 'spread_availability': 'Likely yes',
     'corporate_action_concerns': 'LOW', 'rollover_concerns': 'MEDIUM (futures-linked CFDs can have rollover effects)',
     'contract_change_concerns': 'MEDIUM (if futures-based)', 'survivorship_concerns': 'LOW',
     'timestamp_quality': 'Likely good (same feed already validated for XAUUSD)', 'data_normalization_requirements': 'LOW',
     'overall_data_readiness': 'PARTIALLY READY -- verification needed'},
    {'class_name': 'Index-based return streams', 'historical_data_source': 'UNKNOWN -- index CFD access not yet confirmed in this project',
     'minimum_usable_history': 'UNKNOWN', 'timeframe_availability': 'UNKNOWN', 'bid_ask_availability': 'UNKNOWN',
     'spread_availability': 'UNKNOWN', 'corporate_action_concerns': 'MEDIUM (dividend adjustments on some index CFDs)',
     'rollover_concerns': 'MEDIUM', 'contract_change_concerns': 'LOW', 'survivorship_concerns': 'LOW',
     'timestamp_quality': 'UNKNOWN', 'data_normalization_requirements': 'MEDIUM',
     'overall_data_readiness': 'NOT READY -- requires confirmation of MT5 index CFD access'},
    {'class_name': 'Cross-sectional FX', 'historical_data_source': 'Same MT5 feed already used since Phase 30 (validated)',
     'minimum_usable_history': '3+ years already confirmed (2023-2026, extendable to 2019+ per Phase36/37 own AUDUSD pull)',
     'timeframe_availability': 'Confirmed good (D1/H1/H4 all already used successfully)',
     'bid_ask_availability': 'Not directly available (this project has used mid-price OHLC with a flat spread assumption throughout, not live bid/ask history)',
     'spread_availability': 'Assumed via the same flat-cost convention used project-wide',
     'corporate_action_concerns': 'None (spot FX)', 'rollover_concerns': 'None (spot FX)', 'contract_change_concerns': 'None',
     'survivorship_concerns': 'LOW', 'timestamp_quality': 'Confirmed good (validated via research_data_validator across every prior phase)',
     'data_normalization_requirements': 'LOW (basket alignment across already-used pairs)', 'overall_data_readiness': 'READY'},
    {'class_name': 'Relative-value / spread structures', 'historical_data_source': 'Same MT5 feed',
     'minimum_usable_history': 'Same as cross-sectional FX', 'timeframe_availability': 'Confirmed good',
     'bid_ask_availability': 'Same limitation as cross-sectional FX', 'spread_availability': 'Same flat-cost convention',
     'corporate_action_concerns': 'None', 'rollover_concerns': 'None', 'contract_change_concerns': 'None',
     'survivorship_concerns': 'LOW', 'timestamp_quality': 'Confirmed good',
     'data_normalization_requirements': 'MEDIUM (spread/cointegration construction)',
     'overall_data_readiness': 'READY for data; MEDIUM readiness on methodology'},
    {'class_name': 'Volatility-conditioned systems',
     'historical_data_source': "NO CONFIRMED SOURCE -- a genuine volatility index or options-derived series is not available via this project's current MT5 feed",
     'minimum_usable_history': 'UNKNOWN', 'timeframe_availability': 'UNKNOWN', 'bid_ask_availability': 'UNKNOWN',
     'spread_availability': 'UNKNOWN', 'corporate_action_concerns': 'LOW', 'rollover_concerns': 'LOW',
     'contract_change_concerns': 'UNKNOWN', 'survivorship_concerns': 'LOW', 'timestamp_quality': 'UNKNOWN',
     'data_normalization_requirements': 'HIGH (would need a new proxy construction, e.g. realized-vol term structure from existing OHLC, if a true vol index is unavailable)',
     'overall_data_readiness': 'NOT READY -- confirmed data gap'},
    {'class_name': 'Multi-asset momentum', 'historical_data_source': 'Depends on classes 1-3 being solved first',
     'minimum_usable_history': 'UNKNOWN', 'timeframe_availability': 'UNKNOWN', 'bid_ask_availability': 'UNKNOWN',
     'spread_availability': 'UNKNOWN', 'corporate_action_concerns': 'Combines the concerns of classes 1-3',
     'rollover_concerns': 'Combines the concerns of classes 1-3', 'contract_change_concerns': 'Combines the concerns of classes 1-3',
     'survivorship_concerns': 'LOW', 'timestamp_quality': 'UNKNOWN', 'data_normalization_requirements': 'HIGH',
     'overall_data_readiness': 'NOT READY -- compound dependency on unresolved data gaps'},
    {'class_name': 'Event/macro-conditioned systems',
     'historical_data_source': 'NO CONFIRMED SOURCE for a risk-on/risk-off classifier -- Phase31 already found this could not be reconstructed from available data',
     'minimum_usable_history': 'UNKNOWN', 'timeframe_availability': 'UNKNOWN', 'bid_ask_availability': 'UNKNOWN',
     'spread_availability': 'UNKNOWN', 'corporate_action_concerns': 'LOW', 'rollover_concerns': 'LOW',
     'contract_change_concerns': 'UNKNOWN', 'survivorship_concerns': 'LOW', 'timestamp_quality': 'UNKNOWN',
     'data_normalization_requirements': 'HIGH (classifier construction itself is a prerequisite project)',
     'overall_data_readiness': 'NOT READY -- confirmed data gap (already independently flagged in Phase31)'},
    {'class_name': 'Session-specific structures',
     'historical_data_source': 'Same MT5 feed for price data + core/news_calendar.py already exists for calendar data',
     'minimum_usable_history': 'Same as cross-sectional FX for price; calendar-feed history not yet audited for depth',
     'timeframe_availability': 'Confirmed good for price', 'bid_ask_availability': 'Same limitation as above',
     'spread_availability': 'Same flat-cost convention', 'corporate_action_concerns': 'None', 'rollover_concerns': 'None',
     'contract_change_concerns': 'None', 'survivorship_concerns': 'LOW',
     'timestamp_quality': 'Confirmed good for price; calendar feed quality not yet audited',
     'data_normalization_requirements': 'MEDIUM (event-to-price-window alignment)',
     'overall_data_readiness': 'MOSTLY READY -- calendar feed depth needs a quick audit'},
    {'class_name': 'Other structurally distinct mechanisms',
     'historical_data_source': 'UNKNOWN (varies by sub-idea; order-flow/microstructure likely needs tick/DOM data beyond current MT5 access)',
     'minimum_usable_history': 'UNKNOWN', 'timeframe_availability': 'UNKNOWN', 'bid_ask_availability': 'UNKNOWN',
     'spread_availability': 'UNKNOWN', 'corporate_action_concerns': 'UNKNOWN', 'rollover_concerns': 'UNKNOWN',
     'contract_change_concerns': 'UNKNOWN', 'survivorship_concerns': 'UNKNOWN', 'timestamp_quality': 'UNKNOWN',
     'data_normalization_requirements': 'UNKNOWN', 'overall_data_readiness': 'NOT READY -- undefined, needs dedicated scoping first'},
]


def main():
    with open(OUT / 'phase37_data_availability.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in ROWS:
            w.writerow(r)
    print(f"written {len(ROWS)} rows")


if __name__ == '__main__':
    main()
