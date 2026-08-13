"""
Read-only trade export tool.

Joins trades_log.csv (CLOSED rows) with journal/events.jsonl (entry/exit/signal
events, matched by MT5 Ticket) into a single flat CSV using the schema
requested for the 5ers forensic re-audit.

READ-ONLY: this script only opens files with 'r' mode. It never writes to
data/trades_log.csv, data/journal/events.jsonl, or any other source file,
and it never touches MT5 order/position state (no order_send, no trading
calls of any kind).

Usage:
    python scripts/export_5ers_trades.py --data-dir data --account DEMO-5052472770 \
        --out reports/5ers_trade_export.csv

Point --data-dir at the data/ directory of whichever clone you want to
export (this clone's local data/, or a copy pulled from the VPS clone that
is actually bound to the 5ers terminal). This script does not know which
account a given data/ directory belongs to -- pass --account explicitly so
the output is never mislabeled.
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

NA = 'NOT_AVAILABLE'

PIP_VALUE_USD = {'default': 10.0, 'JPY': 6.7, 'XAUUSD': 10.0}

DEMOTION_DATE = datetime(2026, 7, 31, tzinfo=timezone.utc)
DEMOTED_STRATEGIES = {'GBPJPY_ARB', 'XAUUSD_ARB'}  # demoted from 5ers 2026-07-31

EXPORT_COLUMNS = [
    'trade_id', 'account', 'strategy', 'symbol', 'direction',
    'signal_time', 'entry_time', 'exit_time', 'entry_price', 'exit_price',
    'lots', 'risk_percent', 'initial_risk', 'profit', 'swap', 'commission',
    'R', 'stop_loss', 'take_profit', 'spread', 'ATR', 'holding_time',
    'exit_reason', 'raw_exit_reason', 'strategy_reason', 'strategy_version',
    'demotion_status', 'r_source', 'source_timestamp', 'source_record_id',
]


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


def load_closed_trades(data_dir: Path):
    path = data_dir / 'trades_log.csv'
    if not path.exists():
        return [], str(path)
    with open(path, newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r.get('Status') == 'CLOSED']
    return rows, str(path)


def load_journal_events(data_dir: Path):
    path = data_dir / 'journal' / 'events.jsonl'
    if not path.exists():
        return {'signal': [], 'entry': [], 'exit': [], 'rejection': []}, str(path)
    out = {'signal': [], 'entry': [], 'exit': [], 'rejection': []}
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev['_line_no'] = i
            kind = ev.get('kind')
            if kind in out:
                out[kind].append(ev)
    return out, str(path)


def index_by_ticket(events):
    m = {}
    for ev in events:
        t = ev.get('ticket')
        if t is not None:
            m[str(t)] = ev
    return m


def strategy_from_key(key: str) -> str:
    """Journal 'key' fields look like 'GBPJPY@arb' / 'AUDJPY@amr' -- normalize
    to the PAIR_STRATEGY form used elsewhere in this project's reports."""
    if not key:
        return NA
    if '@' in key:
        pair, strat = key.split('@', 1)
        return f"{pair.upper()}_{strat.upper()}"
    return key.upper()


def strategy_from_trades_log(row: dict) -> str:
    """Fallback when no journal match exists: Session+Pair. Documented as an
    approximation -- only unambiguous if no two active strategies share both
    the same pair and same session."""
    pair = (row.get('Pair') or '').upper()
    session = (row.get('Session') or '').upper()
    if not pair:
        return NA
    return f"{pair}_{session or 'UNKNOWN_SESSION'} (approx, no journal match)"


def classify_demotion(strategy: str, entry_time_str: str) -> str:
    strat_key = strategy.split(' ')[0]  # strip the '(approx...)' suffix if present
    if strat_key not in DEMOTED_STRATEGIES:
        return 'N/A (not a demoted strategy)'
    try:
        t = datetime.fromisoformat(entry_time_str.replace('Z', '+00:00'))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError, TypeError):
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


def build_export(data_dir: Path, account: str):
    closed, trades_log_path = load_closed_trades(data_dir)
    events, journal_path = load_journal_events(data_dir)
    entry_by_ticket = index_by_ticket(events['entry'])
    exit_by_ticket = index_by_ticket(events['exit'])

    rows = []
    for row in closed:
        ticket = str(row.get('Ticket') or '')
        entry_ev = entry_by_ticket.get(ticket)
        exit_ev = exit_by_ticket.get(ticket)

        if entry_ev:
            strategy = strategy_from_key(entry_ev.get('key'))
            strategy_reason = entry_ev.get('strategy_reason', NA)
            spread = entry_ev.get('spread_pips', NA)
            atr = entry_ev.get('atr14_h1_pips', NA)
            risk_pct = entry_ev.get('risk_pct_config', NA)
            risk_usd = _to_float(entry_ev.get('risk_usd_intended'))
            r_source = 'journal' if risk_usd else None
            signal_time = entry_ev.get('ts_utc', NA)
            entry_time = entry_ev.get('ts_utc', row.get('Timestamp', NA))
        else:
            strategy = strategy_from_trades_log(row)
            strategy_reason = NA
            spread = NA
            atr = NA
            risk_pct = NA
            risk_usd = None
            r_source = None
            signal_time = NA
            entry_time = row.get('Timestamp', NA)

        pnl = _to_float(row.get('PnL'))
        if risk_usd is None:
            sl_pips = _to_float(row.get('SLPips'))
            lots = _to_float(row.get('Lots'))
            if sl_pips and lots:
                risk_usd = abs(sl_pips) * lots * _pip_value_usd(row.get('Pair'))
                r_source = 'fallback'
        r = round(pnl / risk_usd, 2) if (pnl is not None and risk_usd) else NA

        holding_time = exit_ev.get('hold_hours') if exit_ev else NA
        raw_exit_reason = row.get('ExitReason', NA)

        rows.append({
            'trade_id': ticket,
            'account': account,
            'strategy': strategy,
            'symbol': row.get('Pair', NA),
            'direction': row.get('Direction', NA),
            'signal_time': signal_time,
            'entry_time': entry_time,
            'exit_time': row.get('ExitTime', NA),
            'entry_price': row.get('EntryPrice', NA),
            'exit_price': row.get('ExitPrice', NA),
            'lots': row.get('Lots', NA),
            'risk_percent': risk_pct,
            'initial_risk': round(risk_usd, 2) if risk_usd else NA,
            'profit': row.get('PnL', NA),
            'swap': NA,        # not persisted anywhere in this file-based system -- see audit report
            'commission': NA,  # not persisted anywhere in this file-based system -- see audit report
            'R': r,
            'stop_loss': row.get('SL', NA),
            'take_profit': row.get('TP', NA),
            'spread': spread,
            'ATR': atr,
            'holding_time': holding_time,
            'exit_reason': decode_exit_reason(raw_exit_reason),
            'raw_exit_reason': raw_exit_reason,
            'strategy_reason': strategy_reason,
            'strategy_version': NA,  # not tracked for general trades -- see audit report
            'demotion_status': classify_demotion(strategy, str(entry_time)),
            'r_source': r_source or 'none',
            'source_timestamp': row.get('Timestamp', NA),
            'source_record_id': f"trades_log.csv:Ticket={ticket}"
                                 + (f";journal_entry_line={entry_ev['_line_no']}" if entry_ev else '')
                                 + (f";journal_exit_line={exit_ev['_line_no']}" if exit_ev else ''),
        })

    meta = {
        'trades_log_path': trades_log_path,
        'journal_path': journal_path,
        'trades_log_exists': (data_dir / 'trades_log.csv').exists(),
        'journal_exists': (data_dir / 'journal' / 'events.jsonl').exists(),
        'closed_trade_count': len(closed),
        'journal_entry_events': len(events['entry']),
        'journal_exit_events': len(events['exit']),
        'journal_matched_entries': sum(1 for r in closed if str(r.get('Ticket') or '') in entry_by_ticket),
    }
    return rows, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', default='data', help='Path to a clone\'s data/ directory (read-only)')
    ap.add_argument('--account', required=True, help='Explicit account label, e.g. DEMO-5052472770 or 5ERS-<login>')
    ap.add_argument('--out', default='reports/5ers_trade_export.csv')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    rows, meta = build_export(data_dir, args.account)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=EXPORT_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to {out_path}")
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()
