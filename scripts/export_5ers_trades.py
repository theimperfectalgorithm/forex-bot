"""
Read-only trade export tool for production 5ers data.

Joins a trades_log.csv (OPEN + CLOSED rows) with a journal/events.jsonl
(signal/entry/exit events) into a single flat CSV, replicating the
dashboard's (mcp/server.py) exact R-calculation and exit-reason decoding.

READ-ONLY / SAFE TO RUN AGAINST PRODUCTION:
  - Every input file is opened with mode 'r' only. Nothing is ever written
    to --trades or --journal, or to any path under their parent directories.
  - No MT5 calls of any kind (no order_send, no positions_get, nothing) --
    this script never touches a running terminal or account.
  - Explicit --trades / --journal / --output arguments only: there are no
    defaults that could resolve to a production path by accident, and the
    output path is never inferred from the input paths.
  - Missing input files fail loudly (FileNotFoundError, non-zero exit),
    never silently return an empty/partial export.
  - No record is silently dropped: every row in trades_log.csv (OPEN and
    CLOSED) becomes exactly one output row, tagged with a `status` column.

Usage:
    python export_5ers_trades.py \
        --trades "C:\\forex-bot-5ers\\data\\trades_log.csv" \
        --journal "C:\\forex-bot-5ers\\data\\journal\\events.jsonl" \
        --output "C:\\5ers-research\\5ers_trade_export.csv" \
        --account 5ERS-<login>

    Add --dry-run to run the full join + validation summary without
    writing --output at all (useful for a first pass against production
    data before trusting the export).
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

NA = 'NOT_AVAILABLE'

PIP_VALUE_USD = {'default': 10.0, 'JPY': 6.7, 'XAUUSD': 10.0}

DEMOTION_DATE = datetime(2026, 7, 31, tzinfo=timezone.utc)
DEMOTED_STRATEGIES = {'GBPJPY_ARB', 'XAUUSD_ARB'}  # demoted from 5ers 2026-07-31

# Secondary join tolerance when a trade has no exact Ticket match in the
# journal (e.g. legacy rows predating journaling) -- symbol+direction must
# also match within this window of the trades_log entry Timestamp.
TIMESTAMP_MATCH_TOLERANCE = timedelta(minutes=5)

EXPORT_COLUMNS = [
    'trade_id', 'account', 'status', 'strategy', 'symbol', 'direction',
    'signal_time', 'entry_time', 'exit_time', 'entry_price', 'exit_price',
    'lots', 'risk_percent', 'initial_risk', 'profit', 'swap', 'commission',
    'R', 'stop_loss', 'take_profit', 'spread', 'ATR', 'holding_time',
    'exit_reason', 'raw_exit_reason', 'strategy_reason', 'strategy_version',
    'demotion_status', 'r_source', 'match_method', 'source_timestamp',
    'source_record_id',
]


def require_file(path: Path, label: str) -> Path:
    """Fail loudly (do not silently return empty data) if a required input
    is missing -- this is a hard requirement for running against production
    paths, where a typo'd path must never be mistaken for 'zero trades'."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")
    return path


def _pip_value_usd(pair: str) -> float:
    if not pair:
        return PIP_VALUE_USD['default']
    if pair.upper() == 'XAUUSD':
        return PIP_VALUE_USD['XAUUSD']
    if pair.upper().endswith('JPY'):
        return PIP_VALUE_USD['JPY']
    return PIP_VALUE_USD['default']


def _to_float(v):
    try:
        if v in (None, ''):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _parse_ts(s):
    if not s:
        return None
    try:
        t = datetime.fromisoformat(str(s).replace('Z', '+00:00'))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t
    except (ValueError, TypeError):
        return None


def load_trades(path: Path):
    """All rows (OPEN + CLOSED), read-only. Never filters -- filtering by
    status happens downstream, not here, so nothing is silently dropped
    at load time."""
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    return rows


def load_journal_events(path: Path):
    """Parses every line; malformed lines are counted, not silently
    skipped without a trace, so the validation summary can report them."""
    out = {'signal': [], 'entry': [], 'exit': [], 'rejection': []}
    malformed = 0
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            ev['_line_no'] = i
            kind = ev.get('kind')
            if kind in out:
                out[kind].append(ev)
    return out, malformed


def index_by_ticket(events):
    m = {}
    for ev in events:
        t = ev.get('ticket')
        if t is not None:
            m.setdefault(str(t), ev)  # first match wins; duplicates tracked separately
    return m


def find_timestamp_match(entry_events, symbol, direction, ts, used_line_nos):
    """Fallback join for trades with no exact Ticket match in the journal
    (e.g. legacy pre-journaling rows): symbol + direction match, entry
    event's ts_utc within TIMESTAMP_MATCH_TOLERANCE of the trade's
    Timestamp, and not already claimed by another trade. Returns the
    closest such event or None."""
    if ts is None:
        return None
    best = None
    best_delta = None
    for ev in entry_events:
        if ev['_line_no'] in used_line_nos:
            continue
        if ev.get('symbol') != symbol or ev.get('direction') != direction:
            continue
        ev_ts = _parse_ts(ev.get('ts_utc'))
        if ev_ts is None:
            continue
        delta = abs(ev_ts - ts)
        if delta <= TIMESTAMP_MATCH_TOLERANCE and (best_delta is None or delta < best_delta):
            best, best_delta = ev, delta
    return best


def strategy_from_key(key: str) -> str:
    """Journal 'key' fields look like 'GBPJPY@arb' / 'AUDJPY@amr' -- normalize
    to the PAIR_STRATEGY form used elsewhere in this project's reports."""
    if not key:
        return None
    if '@' in key:
        pair, strat = key.split('@', 1)
        return f"{pair.upper()}_{strat.upper()}"
    return key.upper()


def strategy_from_trades_log(row: dict) -> str:
    """Fallback when no journal match exists (by ticket or timestamp):
    Session+Pair. Documented as an approximation -- only unambiguous if no
    two active strategies share both the same pair and same session."""
    pair = (row.get('Pair') or '').upper()
    session = (row.get('Session') or '').upper()
    if not pair:
        return NA
    return f"{pair}_{session or 'UNKNOWN_SESSION'} (approx, no journal match)"


def classify_demotion(strategy: str, entry_time_str) -> str:
    strat_key = strategy.split(' ')[0] if strategy else ''  # strip '(approx...)' suffix
    if strat_key not in DEMOTED_STRATEGIES:
        return 'N/A (not a demoted strategy)'
    t = _parse_ts(entry_time_str)
    if t is None:
        return 'UNKNOWN (unparseable entry_time)'
    return 'PRE_DEMOTION' if t < DEMOTION_DATE else 'POST_DEMOTION'


def decode_exit_reason(raw: str) -> str:
    """Per explicit project convention: MANUAL/OTHER in this system does NOT
    mean manual discretionary intervention -- it is MT5's label for any
    client-side close it can't attribute to SL/TP, which in this bot is the
    scheduled London-open / session-based strategy exit (unless the Friday
    force-close or legacy EOD-close labels already disambiguate it)."""
    if raw in (None, ''):
        return NA
    if raw == 'MANUAL/OTHER':
        return 'SCHEDULED_STRATEGY_EXIT'
    return raw  # TP, SL, FRIDAY_CLOSE, EOD_CLOSE pass through unchanged


def build_export(trades_path: Path, journal_path: Path, account: str):
    trades = load_trades(trades_path)
    events, malformed_journal_lines = load_journal_events(journal_path)
    entry_by_ticket = index_by_ticket(events['entry'])
    exit_by_ticket = index_by_ticket(events['exit'])

    # Duplicate ticket detection (source integrity check, not silently ignored).
    # NOTE: trades_log.csv normally logs TWO rows per completed trade (an OPEN
    # row written at entry, a CLOSED row written at exit) sharing one Ticket --
    # that is expected schema behavior, not corruption. This count only
    # matters as a red flag if it exceeds 2 rows for the same ticket, or if a
    # ticket appears twice with the same Status.
    ticket_seen = {}
    duplicate_tickets = set()
    for row in trades:
        t = str(row.get('Ticket') or '')
        if not t:
            continue
        ticket_seen[t] = ticket_seen.get(t, 0) + 1
        if ticket_seen[t] > 1:
            duplicate_tickets.add(t)

    used_ts_match_lines = set()
    rows = []
    matched_count = 0
    unmatched_count = 0
    missing_strategy_count = 0

    for row in trades:
        ticket = str(row.get('Ticket') or '')
        status = row.get('Status', NA)
        entry_ts = _parse_ts(row.get('Timestamp'))

        entry_ev = entry_by_ticket.get(ticket)
        exit_ev = exit_by_ticket.get(ticket)
        match_method = 'ticket' if entry_ev else None

        if entry_ev is None:
            # Fallback: symbol + direction + timestamp proximity
            entry_ev = find_timestamp_match(
                events['entry'], row.get('Pair'), row.get('Direction'),
                entry_ts, used_ts_match_lines)
            if entry_ev is not None:
                used_ts_match_lines.add(entry_ev['_line_no'])
                match_method = 'timestamp_fallback'
                # exit event for a timestamp-matched entry: try its own ticket field
                ev_ticket = str(entry_ev.get('ticket') or '')
                if ev_ticket:
                    exit_ev = exit_by_ticket.get(ev_ticket, exit_ev)

        if entry_ev:
            matched_count += 1
            strategy = strategy_from_key(entry_ev.get('key')) or strategy_from_trades_log(row)
            if strategy_from_key(entry_ev.get('key')) is None:
                missing_strategy_count += 1
            strategy_reason = entry_ev.get('strategy_reason', NA)
            spread = entry_ev.get('spread_pips', NA)
            atr = entry_ev.get('atr14_h1_pips', NA)
            risk_pct = entry_ev.get('risk_pct_config', NA)
            risk_usd = _to_float(entry_ev.get('risk_usd_intended'))
            r_source = 'journal' if risk_usd else None
            signal_time = entry_ev.get('ts_utc', NA)
            entry_time = entry_ev.get('ts_utc', row.get('Timestamp', NA))
        else:
            unmatched_count += 1
            missing_strategy_count += 1
            strategy = strategy_from_trades_log(row)
            strategy_reason = NA
            spread = NA
            atr = NA
            risk_pct = NA
            risk_usd = None
            r_source = None
            signal_time = NA
            entry_time = row.get('Timestamp', NA)
            match_method = 'none'

        pnl = _to_float(row.get('PnL'))
        if risk_usd is None:
            sl_pips = _to_float(row.get('SLPips'))
            lots = _to_float(row.get('Lots'))
            if sl_pips and lots:
                risk_usd = abs(sl_pips) * lots * _pip_value_usd(row.get('Pair'))
                r_source = 'fallback'
        r = round(pnl / risk_usd, 2) if (pnl is not None and risk_usd) else NA

        holding_time = exit_ev.get('hold_hours') if exit_ev else NA
        raw_exit_reason = row.get('ExitReason') if status == 'CLOSED' else None

        rows.append({
            'trade_id': ticket,
            'account': account,
            'status': status,
            'strategy': strategy,
            'symbol': row.get('Pair', NA),
            'direction': row.get('Direction', NA),
            'signal_time': signal_time,
            'entry_time': entry_time,
            'exit_time': row.get('ExitTime') or NA,
            'entry_price': row.get('EntryPrice', NA),
            'exit_price': row.get('ExitPrice') or NA,
            'lots': row.get('Lots', NA),
            'risk_percent': risk_pct,
            'initial_risk': round(risk_usd, 2) if risk_usd else NA,
            'profit': row.get('PnL') or NA,
            'swap': NA,        # not persisted anywhere in this file-based system
            'commission': NA,  # not persisted anywhere in this file-based system
            'R': r,
            'stop_loss': row.get('SL', NA),
            'take_profit': row.get('TP', NA),
            'spread': spread,
            'ATR': atr,
            'holding_time': holding_time,
            'exit_reason': decode_exit_reason(raw_exit_reason) if status == 'CLOSED' else NA,
            'raw_exit_reason': raw_exit_reason if status == 'CLOSED' else NA,
            'strategy_reason': strategy_reason,
            'strategy_version': NA,  # not tracked for general trades
            'demotion_status': classify_demotion(strategy, entry_time),
            'r_source': r_source or 'none',
            'match_method': match_method,
            'source_timestamp': row.get('Timestamp', NA),
            'source_record_id': f"trades_log.csv:Ticket={ticket}"
                                 + (f";journal_entry_line={entry_ev['_line_no']}" if entry_ev else '')
                                 + (f";journal_exit_line={exit_ev['_line_no']}" if exit_ev else ''),
        })

    closed_rows = [r for r in trades if r.get('Status') == 'CLOSED']
    open_rows = [r for r in trades if r.get('Status') == 'OPEN']
    entry_timestamps = [t for t in (_parse_ts(r.get('Timestamp')) for r in trades) if t]

    summary = {
        'trades_source_path': str(trades_path),
        'journal_source_path': str(journal_path),
        'source_trade_rows': len(trades),
        'closed_trades': len(closed_rows),
        'open_trades': len(open_rows),
        'other_status_rows': len(trades) - len(closed_rows) - len(open_rows),
        'journal_signal_events': len(events['signal']),
        'journal_entry_events': len(events['entry']),
        'journal_exit_events': len(events['exit']),
        'journal_rejection_events': len(events['rejection']),
        'journal_malformed_lines': malformed_journal_lines,
        'matched_trades': matched_count,
        'unmatched_trades': unmatched_count,
        'matched_via_ticket': sum(1 for r in rows if r['match_method'] == 'ticket'),
        'matched_via_timestamp_fallback': sum(1 for r in rows if r['match_method'] == 'timestamp_fallback'),
        'duplicate_tickets': sorted(duplicate_tickets),
        'duplicate_ticket_count': len(duplicate_tickets),
        'missing_strategy_attribution': missing_strategy_count,
        'date_range_start': min(entry_timestamps).isoformat() if entry_timestamps else NA,
        'date_range_end': max(entry_timestamps).isoformat() if entry_timestamps else NA,
        'output_row_count': len(rows),
    }
    return rows, summary


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trades', required=True, help='Path to trades_log.csv (read-only)')
    ap.add_argument('--journal', required=True, help='Path to journal/events.jsonl (read-only)')
    ap.add_argument('--output', required=True, help='Path to write the export CSV (not under the source data dir)')
    ap.add_argument('--account', default=None,
                     help='Explicit account label. Defaults to the trades file\'s grandparent '
                          'directory name (e.g. "forex-bot-5ers") if omitted.')
    ap.add_argument('--dry-run', action='store_true',
                     help='Run the full join + validation summary but do not write --output')
    args = ap.parse_args()

    trades_path = require_file(Path(args.trades), '--trades')
    journal_path = require_file(Path(args.journal), '--journal')
    output_path = Path(args.output)

    # Refuse to write inside either source file's directory tree -- a hard
    # guard against ever touching production data, even by accident.
    for guarded in (trades_path.parent.resolve(), journal_path.parent.resolve()):
        try:
            output_path.resolve().relative_to(guarded)
            print(f"ERROR: --output resolves inside a production source directory ({guarded}) -- refusing to write.",
                  file=sys.stderr)
            sys.exit(2)
        except ValueError:
            pass  # output is NOT inside this guarded directory -- good

    account = args.account or trades_path.resolve().parent.parent.name

    rows, summary = build_export(trades_path, journal_path, account)

    print('=== Validation summary ===')
    print(json.dumps(summary, indent=2))

    if args.dry_run:
        print('\n--dry-run: no output file written.')
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=EXPORT_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\nWrote {len(rows)} rows to {output_path}")


if __name__ == '__main__':
    main()
