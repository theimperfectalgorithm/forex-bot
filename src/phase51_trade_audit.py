"""
Phase 51 -- trade-level London/session-exit expectation and classification.
Reads ONLY reports/5ers_trade_export.csv (the sole available trade-level
evidence source on this machine, see phase51_preregistration.md Part 1).
Read-only. No MT5 calls, no production writes.
"""
import csv
from datetime import datetime, timezone, timedelta
from pathlib import Path

EXPORT_PATH = Path(__file__).parent.parent / 'reports' / '5ers_trade_export.csv'

ELIGIBLE_STRATEGIES = {
    'AUDJPY_AMR', 'CADJPY_AMR', 'EURJPY_AMR', 'GBPJPY_AMR',
    'CADJPY_ARB', 'GBPUSD_MONDAY',
}
AMR_STRATEGIES = {'AUDJPY_AMR', 'CADJPY_AMR', 'EURJPY_AMR', 'GBPJPY_AMR'}
MON_STRATEGIES = {'GBPUSD_MONDAY'}
ARB_STRATEGIES = {'CADJPY_ARB'}

SERVER_OFFSET_HOURS = 3  # US DST in effect for entire 2026-07-20..2026-08-13 window
AMR_EXIT_SERVER_HOUR = 7
MON_EXIT_SERVER_HOUR = 21
TOLERANCE_MIN = 30

PRIMARY_WINDOW_START = datetime(2026, 8, 14, tzinfo=timezone.utc)
PRIMARY_WINDOW_END = datetime(2026, 8, 23, 23, 59, 59, tzinfo=timezone.utc)
BASELINE_WINDOW_START = datetime(2026, 7, 20, tzinfo=timezone.utc)
BASELINE_WINDOW_END = datetime(2026, 8, 13, 23, 59, 59, tzinfo=timezone.utc)

NA = 'NOT_AVAILABLE'


def _parse_ts(s):
    if not s or s == NA:
        return None
    try:
        return datetime.fromisoformat(s.replace('Z', '+00:00'))
    except ValueError:
        return None


def load_export():
    with open(EXPORT_PATH, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def expected_exit_utc(strategy, entry_dt):
    """Returns (expected_exit_utc, method) or (None, reason)."""
    if strategy in AMR_STRATEGIES:
        # AMENDMENT 1 (phase51_preregistration.md): the server-time-converted
        # value (04:00 UTC) was empirically falsified by 9/9 observed AMR
        # scheduled-exit timestamps clustering at 07:00:05 UTC. Using the
        # empirically observed operative time instead, per the disclosed
        # amendment -- this also matches the strategy source's own
        # docstring language ("TIME EXIT at 07:00 UTC").
        exit_utc_hour = AMR_EXIT_SERVER_HOUR  # amended: literal UTC, not server-converted
        exit_dt = entry_dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(hours=exit_utc_hour)
        if exit_dt < entry_dt:
            exit_dt += timedelta(days=1)
        return exit_dt, 'AMR: entry-day 07:00 UTC (amended, empirically observed)'
    if strategy in MON_STRATEGIES:
        days_to_subtract = entry_dt.weekday()  # Monday=0
        monday = (entry_dt - timedelta(days=days_to_subtract)).replace(hour=0, minute=0, second=0, microsecond=0)
        exit_utc_hour = MON_EXIT_SERVER_HOUR - SERVER_OFFSET_HOURS
        exit_dt = monday + timedelta(hours=exit_utc_hour)
        return exit_dt, 'MON: entry-week Monday server 21:00 -> UTC'
    return None, 'ARB: no scheduled exit by design'


def classify_trade(row):
    """Returns dict with all audit fields for one trade row."""
    strategy = row.get('strategy', '')
    status = row.get('status', '')
    out = dict(row)
    out['audit_strategy_family'] = (
        'AMR' if strategy in AMR_STRATEGIES else
        'MON' if strategy in MON_STRATEGIES else
        'ARB' if strategy in ARB_STRATEGIES else 'UNKNOWN'
    )

    if strategy not in ELIGIBLE_STRATEGIES:
        out['london_exit_classification'] = 'UNKNOWN'
        out['event_classification'] = 'J. DATA_UNAVAILABLE'
        out['classification_note'] = f'strategy "{strategy}" not one of the 6 eligible live strategies'
        return out

    entry_dt = _parse_ts(row.get('entry_time'))
    if entry_dt is None:
        out['london_exit_classification'] = 'UNKNOWN'
        out['event_classification'] = 'J. DATA_UNAVAILABLE'
        out['classification_note'] = 'entry_time unparseable/missing'
        return out

    exp_exit, method = expected_exit_utc(strategy, entry_dt)
    out['expected_exit_method'] = method

    if exp_exit is None:
        out['london_exit_classification'] = 'LONDON_EXIT_NOT_EXPECTED'
        out['expected_exit_utc'] = NA
        out['event_classification'] = 'N/A -- not applicable (ARB has no scheduled exit)'
        out['classification_note'] = 'ARB exits via SL/TP/Friday-close only, by design'
        return out

    out['london_exit_classification'] = 'LONDON_EXIT_EXPECTED'
    out['expected_exit_utc'] = exp_exit.isoformat()

    if status != 'CLOSED':
        out['event_classification'] = 'J. DATA_UNAVAILABLE'
        out['classification_note'] = f'trade still {status} in the export as of the export snapshot -- outcome not yet determined'
        return out

    actual_exit = _parse_ts(row.get('exit_time'))
    raw_reason = row.get('raw_exit_reason', NA)
    decoded_reason = row.get('exit_reason', NA)

    if actual_exit is None:
        out['event_classification'] = 'J. DATA_UNAVAILABLE'
        out['classification_note'] = 'exit_time unparseable/missing despite CLOSED status'
        return out

    delta_min = (actual_exit - exp_exit).total_seconds() / 60.0
    out['deviation_minutes'] = round(delta_min, 1)

    if raw_reason in ('SL', 'TP'):
        if actual_exit < exp_exit:
            out['event_classification'] = 'H. POSITION_ALREADY_CLOSED'
            out['classification_note'] = f'position hit {raw_reason} {abs(delta_min):.0f} min before the scheduled exit -- scheduled exit correctly never fired on an already-closed position'
        else:
            out['event_classification'] = 'F. EXECUTION_FAILED_UNKNOWN_REASON'
            out['classification_note'] = (f'position remained open past the scheduled {method.split(":")[0]} exit time and was '
                                          f'eventually closed by {raw_reason}, not by the scheduled exit -- scheduled exit did not fire; '
                                          f'root cause (signal/request/execution layer) cannot be isolated from export-level evidence alone (see Part 4)')
        return out

    if raw_reason == 'MANUAL/OTHER':
        if abs(delta_min) <= TOLERANCE_MIN:
            if decoded_reason == 'SCHEDULED_STRATEGY_EXIT':
                out['event_classification'] = 'A. CORRECTLY_EXECUTED'
                out['classification_note'] = (f'actual exit {abs(delta_min):.0f} min from scheduled time, within {TOLERANCE_MIN}-min tolerance; '
                                              f'export decode_exit_reason() label (SCHEDULED_STRATEGY_EXIT) independently corroborated by this audit\'s own timing check, not merely assumed')
            else:
                out['event_classification'] = 'B. EXECUTED_BUT_MISCLASSIFIED'
                out['classification_note'] = 'timing matches scheduled exit but export label was not decoded as SCHEDULED_STRATEGY_EXIT'
        elif delta_min > TOLERANCE_MIN:
            out['event_classification'] = 'G. EXIT_DELAYED'
            out['classification_note'] = f'closed {delta_min:.0f} min AFTER the scheduled exit time (beyond {TOLERANCE_MIN}-min tolerance); export\'s blanket MANUAL/OTHER->SCHEDULED_STRATEGY_EXIT relabel is NOT timing-corroborated for this trade'
        else:
            out['event_classification'] = 'H. POSITION_ALREADY_CLOSED'
            out['classification_note'] = f'closed {abs(delta_min):.0f} min BEFORE the scheduled exit time via a client-side (MANUAL/OTHER) close -- not explainable as the scheduled time-exit itself'
        return out

    if raw_reason in ('FRIDAY_CLOSE', 'EOD_CLOSE'):
        out['event_classification'] = 'H. POSITION_ALREADY_CLOSED'
        out['classification_note'] = f'closed by {raw_reason} (a different scheduled mechanism), not the AMR/MON time-exit'
        return out

    out['event_classification'] = 'J. DATA_UNAVAILABLE'
    out['classification_note'] = f'unrecognized raw_exit_reason value: {raw_reason!r}'
    return out


def build_trade_level_audit():
    rows = load_export()
    audited = [classify_trade(r) for r in rows]
    for a in audited:
        entry_dt = _parse_ts(a.get('entry_time'))
        if entry_dt is None:
            a['audit_period'] = 'UNKNOWN'
        elif BASELINE_WINDOW_START <= entry_dt <= BASELINE_WINDOW_END:
            a['audit_period'] = 'BASELINE'
        elif PRIMARY_WINDOW_START <= entry_dt <= PRIMARY_WINDOW_END:
            a['audit_period'] = 'CURRENT_LIVE'
        else:
            a['audit_period'] = 'OUT_OF_WINDOW'
    return audited


if __name__ == '__main__':
    audited = build_trade_level_audit()
    print(f'{len(audited)} total export rows audited')
    from collections import Counter
    print(Counter(a['audit_period'] for a in audited))
    print(Counter(a['event_classification'] for a in audited))
