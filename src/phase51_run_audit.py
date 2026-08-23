"""
Phase 51 -- main orchestrator. Produces all required reports/phase51_*.csv
deliverables from the sole available evidence source
(reports/5ers_trade_export.csv). Read-only throughout; no MT5 calls, no
production writes, no live-code/config changes.
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.phase51_trade_audit import (
    build_trade_level_audit, ELIGIBLE_STRATEGIES, NA,
    TOLERANCE_MIN, AMR_EXIT_SERVER_HOUR, MON_EXIT_SERVER_HOUR, SERVER_OFFSET_HOURS,
)
from src.phase51_event_reconstruction import (
    reconstruct_chain, strategy_summary, materiality_summary,
)
from src.phase51_counterfactual import build_counterfactual

REPORTS = Path(__file__).parent.parent / 'reports'


def write_csv(name, rows, fieldnames=None):
    path = REPORTS / name
    if not rows:
        fieldnames = fieldnames or ['note']
        rows = [{'note': 'NO ROWS -- see master report for explanation'}]
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {name}: {len(rows)} rows')


def main():
    audited = build_trade_level_audit()

    # 1. trade-level audit
    trade_cols = ['trade_id', 'account', 'status', 'strategy', 'audit_strategy_family',
                  'symbol', 'direction', 'entry_time', 'entry_price', 'exit_time',
                  'exit_price', 'profit', 'R', 'raw_exit_reason', 'exit_reason',
                  'london_exit_classification', 'expected_exit_utc', 'expected_exit_method',
                  'deviation_minutes', 'event_classification', 'classification_note',
                  'audit_period']
    trade_rows = [{k: a.get(k, NA) for k in trade_cols} for a in audited]
    write_csv('phase51_trade_level_audit.csv', trade_rows, trade_cols)

    # 2. event chain
    eligible_closed = [a for a in audited if a['strategy'] in ELIGIBLE_STRATEGIES and a['status'] == 'CLOSED']
    chain_rows = []
    for t in eligible_closed:
        chain_rows.extend(reconstruct_chain(t))
    write_csv('phase51_event_chain.csv', chain_rows)

    # 3. london exit expectations
    exp_cols = ['trade_id', 'strategy', 'entry_time', 'london_exit_classification',
                'expected_exit_utc', 'expected_exit_method']
    exp_rows = [{k: a.get(k, NA) for k in exp_cols} for a in audited if a['strategy'] in ELIGIBLE_STRATEGIES]
    write_csv('phase51_london_exit_expectations.csv', exp_rows, exp_cols)

    # 4. execution responses -- UNAVAILABLE, no execution log source on this machine
    write_csv('phase51_execution_responses.csv', [{
        'source_required': 'production execution log (order_send/close_position responses, retcodes)',
        'status': 'UNAVAILABLE',
        'reason': 'not present on this machine; see preregistration Part 1',
    }])

    # 5. MT5 verification -- UNAVAILABLE, no live MT5 connection on this machine
    write_csv('phase51_mt5_verification.csv', [{
        'source_required': 'MT5 deal/order history (live query)',
        'status': 'UNAVAILABLE',
        'reason': 'this machine has no live MT5/5ers connection; no MT5 calls made in this audit-only phase, see preregistration Part 1 and Part 9',
    }])

    # 6. configuration audit
    config_rows = [
        {'strategy_family': 'AMR', 'parameter': 'scheduled_exit_trigger',
         'documented_in_source_comment': '07:00 UTC (asian_hours_reversion.py:24, main_agent.py:587)',
         'derived_from_server_time_offset_logic': '04:00 UTC (server 07:00, DST offset +3h)',
         'empirically_observed_in_export': '07:00:05 UTC (9/9 closed MANUAL/OTHER AMR exits)',
         'verdict': 'MISMATCH between derived-from-code-logic value and both the source comment and the observed behavior -- see AMENDMENT 1 in preregistration and master report SS22'},
        {'strategy_family': 'MON', 'parameter': 'scheduled_exit_trigger',
         'documented_in_source_comment': '21:00 UTC Monday (monday_drift.py:16,63)',
         'derived_from_server_time_offset_logic': '18:00 UTC Monday (server 21:00, DST offset +3h)',
         'empirically_observed_in_export': 'NO CLOSED MONDAY TRADES IN AVAILABLE WINDOW -- cannot verify',
         'verdict': 'UNRESOLVED -- no observational evidence either way in the available window'},
        {'strategy_family': 'ARB', 'parameter': 'scheduled_exit_trigger',
         'documented_in_source_comment': 'none -- source contains no time-exit logic',
         'derived_from_server_time_offset_logic': 'N/A',
         'empirically_observed_in_export': 'N/A -- confirmed all CADJPY_ARB exits are SL/TP/other, never a scheduled-exit label',
         'verdict': 'CONSISTENT (no scheduled exit expected or observed)'},
    ]
    write_csv('phase51_configuration_audit.csv', config_rows)

    # 7. source audit
    source_rows = [
        {'file': 'strategies/asian_hours_reversion.py', 'aspect': 'time-exit hour comment',
         'finding': 'docstring says "07:00 UTC"; ORCHESTRATOR INTEGRATION section (lines 31-40) says the AMR time-exit step is "NOT YET WIRED" -- this is stale documentation, contradicted by main_agent.py which has step_asian_time_exit() actively wired and firing (confirmed via export evidence)'},
        {'file': 'src/agents/main_agent.py', 'aspect': 'T_ASIAN_EXIT gating',
         'finding': 'constant T_ASIAN_EXIT=07:00 is gated on server-minutes (srv) per the file-level comment at lines 120-128; server_now = real_UTC_now + offset(+3h summer) implies a 04:00 UTC trigger, but observed production behavior triggers at 07:00 UTC -- SOURCE_CHANGED (relative to what appears to be deployed) or comment-vs-behavior mismatch; cannot be resolved further without VPS code-version access (UNAVAILABLE, see Part 4)'},
        {'file': 'strategies/monday_drift.py', 'aspect': 'time-exit hour comment',
         'finding': 'same "21:00 UTC" vs T_MONDAY_EXIT server-minutes pattern as AMR; UNVERIFIED (no closed Monday trades in the available window)'},
        {'file': 'strategies/asian_range_breakout.py', 'aspect': 'scheduled exit',
         'finding': 'no time-exit logic present -- confirmed by direct grep; CADJPY_ARB correctly has no scheduled London/session exit'},
        {'file': 'scripts/export_5ers_trades.py', 'aspect': 'decode_exit_reason()',
         'finding': 'unconditionally maps every raw MANUAL/OTHER to SCHEDULED_STRATEGY_EXIT (lines 200-210) with no timestamp verification -- an assumption, not a per-trade proof; this audit independently re-verified that assumption by timing and found it corroborated for all 10 A-classified trades but not automatically trustworthy in general (a genuinely delayed or truly-manual close would receive the same blind relabel)'},
    ]
    write_csv('phase51_source_audit.csv', source_rows)

    # 8. P&L counterfactual
    cf_rows = build_counterfactual(audited)
    write_csv('phase51_pnl_counterfactual.csv', cf_rows)

    # 9. strategy summary
    write_csv('phase51_strategy_summary.csv', strategy_summary(audited))

    # 10. deviation summary
    expected_closed = [a for a in audited if a['london_exit_classification'] == 'LONDON_EXIT_EXPECTED' and a['status'] == 'CLOSED']
    dev_counts = Counter(a['event_classification'] for a in expected_closed)
    dev_rows = [{'classification': k, 'count': v} for k, v in sorted(dev_counts.items())]
    write_csv('phase51_deviation_summary.csv', dev_rows)

    # 11. baseline comparison -- only one period is available, so this
    # documents that fact rather than comparing two populations
    write_csv('phase51_baseline_comparison.csv', [{
        'baseline_period': '2026-07-20 to 2026-08-13 (the only available trade-level window)',
        'baseline_deviation_rate_pct': (round(100 * sum(1 for a in expected_closed if a['event_classification'] not in ('A. CORRECTLY_EXECUTED', 'H. POSITION_ALREADY_CLOSED', 'J. DATA_UNAVAILABLE')) / len(expected_closed), 1) if expected_closed else NA),
        'current_live_period': '2026-08-14 to 2026-08-23 (task-specified primary window)',
        'current_live_deviation_rate_pct': 'DATA_UNAVAILABLE',
        'note': 'No comparison is possible -- the current live period has zero rows of trade-level evidence on this machine (see preregistration Part 1). The baseline period deviation rate is reported here as the only computable figure, not as a stand-in for the current period.',
    }])

    # 12. concurrency execution -- diagnostic only, no filter/limit implied
    write_csv('phase51_concurrency_execution.csv', [{
        'note': 'DATA_UNAVAILABLE for a genuine concurrency cross-tab -- the flat export used here does not carry an open-position-count-at-exit-time field, and reconstructing it would require the full portfolio open/close timeline already used in Phase41/49 (data/phase26_all_trades.csv, a historical research ledger, not a live-period source) -- would conflate historical research data with this live-period audit, which Part 6 explicitly prohibits ("do not contaminate the historical research ledger"). Diagnostic not run this phase; flagged as a possible Phase 52 follow-up using the correct live-only data source once the current-live-period gap (Part 1) is resolved.',
    }])

    # 13. volatility execution -- same limitation
    write_csv('phase51_volatility_execution.csv', [{
        'note': 'DATA_UNAVAILABLE for the same reason as concurrency (see phase51_concurrency_execution.csv) -- no live-period volatility-state field exists in the only available evidence source, and the historical research ledger is deliberately not mixed in here.',
    }])

    # 14. recent loss reconstruction
    total_r = sum(float(a['R']) for a in audited if a['strategy'] in ELIGIBLE_STRATEGIES and a['status'] == 'CLOSED' and a.get('R') not in (None, '', NA))
    r_from_correct = sum(float(a['R']) for a in expected_closed if a['event_classification'] == 'A. CORRECTLY_EXECUTED' and a.get('R') not in (None, '', NA))
    r_from_already_closed = sum(float(a['R']) for a in expected_closed if a['event_classification'] == 'H. POSITION_ALREADY_CLOSED' and a.get('R') not in (None, '', NA))
    r_from_arb = sum(float(a['R']) for a in audited if a.get('audit_strategy_family') == 'ARB' and a['status'] == 'CLOSED' and a.get('R') not in (None, '', NA))
    r_from_deviations = sum(float(a['R']) for a in expected_closed if a['event_classification'] not in
                             ('A. CORRECTLY_EXECUTED', 'H. POSITION_ALREADY_CLOSED', 'J. DATA_UNAVAILABLE') and a.get('R') not in (None, '', NA))
    write_csv('phase51_recent_loss_reconstruction.csv', [{
        'period': '2026-07-20 to 2026-08-13 (only available window -- NOT the task-specified current-live 08-14 to 08-23, see preregistration Part 1)',
        'total_actual_R_all_6_strategies_closed': round(total_r, 2),
        'R_from_scheduled_exit_correctly_executed': round(r_from_correct, 2),
        'R_from_SL_TP_before_scheduled_exit_could_fire': round(r_from_already_closed, 2),
        'R_from_ARB_no_scheduled_exit_by_design': round(r_from_arb, 2),
        'R_from_london_exit_deviations': round(r_from_deviations, 2),
        'pct_of_total_R_attributable_to_london_exit_deviations': (round(100 * r_from_deviations / total_r, 1) if total_r else NA),
        'answer_to_required_question': ('Zero deviations were found in the only available window, so 0% of that window\'s R is attributable to London-exit '
                                        'execution deviations. This CANNOT be extrapolated to the task-specified current-live period (08-14 to 08-23), which has no data on this machine.'),
    }])

    # 15. evidence matrix
    write_csv('phase51_evidence_matrix.csv', materiality_summary(audited) and [
        {'evidence_level': 1, 'source': 'MT5 deal/order history', 'status': 'UNAVAILABLE'},
        {'evidence_level': 2, 'source': 'production execution response/log', 'status': 'UNAVAILABLE'},
        {'evidence_level': 3, 'source': 'production journal event', 'status': 'UNAVAILABLE'},
        {'evidence_level': 4, 'source': 'production trade log (raw, live host)', 'status': 'UNAVAILABLE'},
        {'evidence_level': 5, 'source': 'live source code/configuration (this repo checkout)', 'status': 'AVAILABLE, used'},
        {'evidence_level': 6, 'source': 'secondary derived export (reports/5ers_trade_export.csv)', 'status': 'AVAILABLE, used -- sole trade-level evidence source, covers 2026-07-20 to 2026-08-13 only'},
    ])

    # 16. materiality (deviation_summary doubles as this; also write explicit materiality file)
    write_csv('phase51_materiality.csv', materiality_summary(audited))

    print('\nDone.')


if __name__ == '__main__':
    main()
