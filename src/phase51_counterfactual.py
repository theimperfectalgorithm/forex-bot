"""
Phase 51 -- counterfactual P&L for genuine timing deviations only, per the
frozen, disclosed methodology in phase51_preregistration.md Part 6: no
independent tick/bar price source is available on this machine for the
live window, so counterfactual fields are NOT_AVAILABLE unless a genuine
deviation exists AND a reference price is already present in the export.
"""
from src.phase51_trade_audit import NA

DEVIATION_CLASSES = {'B. EXECUTED_BUT_MISCLASSIFIED', 'C. EXIT_SIGNAL_MISSING',
                     'D. EXIT_REQUEST_MISSING', 'E. EXECUTION_REJECTED',
                     'F. EXECUTION_FAILED_UNKNOWN_REASON', 'G. EXIT_DELAYED',
                     'I. CONFIGURATION_MISMATCH'}


def build_counterfactual(audited):
    rows = []
    for a in audited:
        if a.get('london_exit_classification') != 'LONDON_EXIT_EXPECTED':
            continue
        if a['status'] != 'CLOSED':
            continue
        is_deviation = a['event_classification'] in DEVIATION_CLASSES
        rows.append({
            'trade_id': a.get('trade_id'),
            'strategy': a.get('strategy'),
            'event_classification': a['event_classification'],
            'is_deviation': is_deviation,
            'expected_exit_utc': a.get('expected_exit_utc', NA),
            'actual_exit_time': a.get('exit_time', NA),
            'actual_exit_price': a.get('exit_price', NA),
            'actual_exit_pnl': a.get('profit', NA),
            'intended_exit_price': NA,
            'intended_exit_pnl': NA,
            'pnl_difference': NA,
            'counterfactual_note': (
                'no independent tick/bar price source available on this machine for the live '
                'window (see preregistration Part 6) -- INTENDED values not estimated, per the '
                'no-hindsight-price rule'
                if is_deviation else
                'not a deviation -- no counterfactual needed'
            ),
        })
    return rows
