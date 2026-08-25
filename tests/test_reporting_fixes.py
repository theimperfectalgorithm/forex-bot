from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import logging
import os
from types import SimpleNamespace
import time

from core.mt5_time import server_epoch_to_utc
from core.trade_cost_ledger import (LOCK_STALE_AFTER_SECONDS, _ExclusiveFileLock,
                                    aggregate_position_deals, append_cost_record,
                                    load_cost_ledger)
from core.trade_journal import compute_hold_hours


def _deal(ticket, position_id=7, profit=0, commission=0, swap=0, fee=0):
    return SimpleNamespace(ticket=ticket, position_id=position_id, profit=profit,
                           commission=commission, swap=swap, fee=fee)


def test_aggregate_gross_costs_and_net():
    result = aggregate_position_deals([_deal(1, profit=10, commission=-1, swap=-2, fee=-.5)], 7)
    assert result['gross_pnl'] == 10
    assert result['net_pnl'] == 6.5


def test_duplicate_deal_ids_count_once_and_entry_exit_costs_are_kept_once():
    entry = _deal(11, profit=0, commission=-1)
    exit_deal = _deal(12, profit=20, commission=-2, swap=-.25, fee=-.1)
    duplicate_exit = _deal(12, profit=20, commission=-2, swap=-.25, fee=-.1)
    result = aggregate_position_deals([entry, exit_deal, duplicate_exit], 7)
    assert result['deal_count'] == 2
    assert result['gross_pnl'] == 20
    assert result['net_pnl'] == 16.65


def test_missing_deal_identifier_is_not_assigned_invented_uniqueness():
    a = SimpleNamespace(position_id=7, profit=1, commission=0, swap=0, fee=0)
    b = SimpleNamespace(position_id=7, profit=1, commission=0, swap=0, fee=0)
    assert aggregate_position_deals([a, b], 7)['gross_pnl'] == 2


def test_ledger_repeated_ticket_is_idempotent(tmp_path):
    path = tmp_path / 'costs.jsonl'
    record = {'ticket': 7, 'gross_pnl': 10, 'commission': -1, 'swap': 0,
              'fee': 0, 'net_pnl': 9}
    assert append_cost_record(record, path)
    assert not append_cost_record(record, path)
    assert len(path.read_text().splitlines()) == 1


def test_concurrent_ledger_writes_do_not_duplicate_ticket(tmp_path):
    path = tmp_path / 'costs.jsonl'
    record = {'ticket': 8, 'gross_pnl': 10, 'commission': -1, 'swap': 0,
              'fee': 0, 'net_pnl': 9}
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: append_cost_record(record, path), range(8)))
    assert sum(results) == 1
    assert len(path.read_text().splitlines()) == 1


def test_later_complete_record_replaces_incomplete_record(tmp_path):
    path = tmp_path / 'costs.jsonl'
    path.write_text('{"ticket": 9, "gross_pnl": 10}\n')
    complete = {'ticket': 9, 'gross_pnl': 10, 'commission': -1, 'swap': 0,
                'fee': 0, 'net_pnl': 9}
    with path.open('a') as f:
        import json
        f.write(json.dumps(complete) + '\n')
    assert load_cost_ledger(path)['9']['net_pnl'] == 9


def test_malformed_and_missing_ledger_data_is_readable(tmp_path):
    path = tmp_path / 'costs.jsonl'
    path.write_text('not-json\n{"ticket": 1}\n')
    assert load_cost_ledger(path) == {}


def test_stale_lock_recovery(tmp_path):
    lock = tmp_path / 'costs.jsonl.lock'
    lock.write_text('999999999')
    old = time.time() - LOCK_STALE_AFTER_SECONDS - 1
    os.utime(lock, (old, old))
    with _ExclusiveFileLock(lock, timeout=0.2):
        assert lock.exists()
    assert not lock.exists()


def test_active_old_lock_is_not_removed(tmp_path):
    lock = tmp_path / 'costs.jsonl.lock'
    lock.write_text(str(os.getpid()))
    old = time.time() - LOCK_STALE_AFTER_SECONDS - 1
    os.utime(lock, (old, old))
    try:
        with _ExclusiveFileLock(lock, timeout=0.05):
            raise AssertionError('active lock should not be acquired')
    except TimeoutError:
        pass
    assert lock.exists()
    lock.unlink()


def test_lock_cleanup_after_exception(tmp_path):
    lock = tmp_path / 'costs.jsonl.lock'
    try:
        with _ExclusiveFileLock(lock, timeout=0.2):
            raise RuntimeError('simulated append failure')
    except RuntimeError:
        pass
    assert not lock.exists()


def test_invalid_ledger_write_logs_incomplete_coverage(caplog, tmp_path):
    caplog.set_level(logging.ERROR, logger='TRADE_COST_LEDGER')
    assert not append_cost_record({'ticket': 55, 'gross_pnl': float('nan')}, tmp_path / 'costs.jsonl')
    assert 'ticket=55' in caplog.text
    assert 'incomplete' in caplog.text


def test_lock_timeout_logs_failed_coverage(caplog, tmp_path, monkeypatch):
    import core.trade_cost_ledger as ledger
    caplog.set_level(logging.ERROR, logger='TRADE_COST_LEDGER')
    class FailingLock:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            raise TimeoutError('simulated stale lock timeout')
        def __exit__(self, *args):
            return False
    monkeypatch.setattr(ledger, '_ExclusiveFileLock', FailingLock)
    record = {'ticket': 56, 'gross_pnl': 1.0, 'commission': 0.0,
              'swap': 0.0, 'fee': 0.0, 'net_pnl': 1.0}
    assert not ledger.append_cost_record(record, tmp_path / 'costs.jsonl')
    assert 'ticket=56' in caplog.text
    assert 'incomplete' in caplog.text


def test_nonfinite_values_rejected_and_finite_values_accepted(tmp_path):
    for value in (float('nan'), float('inf'), float('-inf')):
        path = tmp_path / f'{str(value)}.jsonl'
        record = {'ticket': str(value), 'gross_pnl': value, 'commission': 0.0,
                  'swap': 0.0, 'fee': 0.0, 'net_pnl': value}
        assert not append_cost_record(record, path)
    path = tmp_path / 'finite.jsonl'
    for ticket, value in ((1, -2.5), (2, 0.0), (3, 4.5)):
        record = {'ticket': ticket, 'gross_pnl': value, 'commission': 0.0,
                  'swap': 0.0, 'fee': 0.0, 'net_pnl': value}
        assert append_cost_record(record, path)
    assert len(load_cost_ledger(path)) == 3


def test_utc2_and_utc3_and_midnight_boundary():
    server_epoch = datetime(2026, 1, 1, 1, 30, tzinfo=timezone.utc).timestamp()
    assert server_epoch_to_utc(server_epoch, 2).isoformat() == '2025-12-31T23:30:00+00:00'
    assert server_epoch_to_utc(server_epoch, 3).isoformat() == '2025-12-31T22:30:00+00:00'


def test_hold_hours_uses_normalized_utc_exit():
    assert compute_hold_hours('2025-12-31T22:30:00+00:00',
                              '2025-12-31T23:30:00+00:00') == 1.0


def test_net_reconciliation_uses_persisted_net_pnl():
    from src.agents.agent_reporting import _reconciliation_check, _log
    accounting = {'gross_pnl': 10.0, 'commission': -1.0, 'swap': -1.0,
                  'fee': 0.0, 'net_pnl': 8.0, 'cost_covered': 1,
                  'legacy_without_costs': 0}
    result = _reconciliation_check('2026-08-25', accounting, 1008, 1000, _log())
    assert result['matched'] is True
    assert result['net_pnl'] == 8.0


def test_export_keeps_normalized_utc_date_and_ledger_net(tmp_path):
    import csv
    import json
    from scripts.export_5ers_trades import build_export
    trades = tmp_path / 'trades.csv'
    journal = tmp_path / 'events.jsonl'
    ledger = tmp_path / 'costs.jsonl'
    fields = ['Ticket', 'Status', 'Timestamp', 'ExitTime', 'PnL', 'Pair', 'Direction',
              'Lots', 'SLPips', 'TPPips', 'EntryPrice', 'ExitPrice', 'ExitReason']
    with trades.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        w.writerow({'Ticket': '42', 'Status': 'CLOSED', 'Timestamp': '2025-12-31T22:30:00+00:00',
                    'ExitTime': '2025-12-31T23:30:00+00:00', 'PnL': '10', 'Pair': 'EURUSD',
                    'Direction': 'BUY', 'Lots': '1', 'SLPips': '10', 'TPPips': '20'})
    journal.write_text('')
    ledger.write_text(json.dumps({'ticket': 42, 'gross_pnl': 10, 'commission': -1,
                                  'swap': 0, 'fee': 0, 'net_pnl': 9}) + '\n')
    rows, summary = build_export(trades, journal, 'test', ledger)
    assert rows[0]['exit_time'].startswith('2025-12-31')
    assert rows[0]['net_pnl'] == 9
    assert summary['cost_covered_closed_trades'] == 1


def test_unknown_close_is_explicitly_incomplete_without_costs():
    import logging
    import src.agents.agent_execution as execution
    original = (execution._connect, execution.mt5.positions_get,
                execution._get_closed_deal, execution._write_trade_log)
    try:
        execution._connect = lambda log: True
        execution.mt5.positions_get = lambda ticket: []
        execution._get_closed_deal = lambda ticket, log: None
        execution._write_trade_log = lambda row: None
        _, closed = execution.monitor_positions(
            [{'ticket': 99, 'symbol': 'EURUSD', 'direction': 'BUY', 'session': 'London',
            'lots': 1.0, 'entry_price': 1.1, 'sl': 1.09, 'tp': 1.12}], logging.getLogger('test'))
        # The first call retries; force the retry threshold to exercise UNKNOWN.
        trade = {'ticket': 99, 'symbol': 'EURUSD', 'direction': 'BUY', 'session': 'London',
                 'lots': 1.0, 'entry_price': 1.1, 'sl': 1.09, 'tp': 1.12,
                 'close_retry': execution.MAX_CLOSE_RETRIES - 1}
        _, closed = execution.monitor_positions([trade], logging.getLogger('test'))
        assert closed[0]['accounting_coverage'] == 'incomplete'
        assert closed[0]['net_pnl'] is None
    finally:
        execution._connect, execution.mt5.positions_get, execution._get_closed_deal, execution._write_trade_log = original


def test_closed_path_with_mt5_accounting_is_complete():
    import logging
    import src.agents.agent_execution as execution
    original = (execution._connect, execution.mt5.positions_get,
                execution._get_closed_deal, execution._write_trade_log,
                execution.append_cost_record)
    try:
        execution._connect = lambda log: True
        execution.mt5.positions_get = lambda ticket: []
        execution._get_closed_deal = lambda ticket, log: {
            'exit_price': 1.2, 'exit_time': '2025-12-31T23:30:00+00:00',
            'exit_reason': 'TP', 'exit_pnl': 10, 'gross_pnl': 10,
            'commission': -1, 'swap': 0, 'fee': 0, 'net_pnl': 9,
            'deal_count': 2, 'server_offset_h': 2}
        execution._write_trade_log = lambda row: None
        execution.append_cost_record = lambda record: True
        _, closed = execution.monitor_positions(
            [{'ticket': 100, 'symbol': 'EURUSD', 'direction': 'BUY', 'session': 'London',
              'lots': 1.0, 'entry_price': 1.1, 'sl': 1.09, 'tp': 1.12}],
            logging.getLogger('test'))
        assert closed[0]['net_pnl'] == 9
    finally:
        (execution._connect, execution.mt5.positions_get, execution._get_closed_deal,
         execution._write_trade_log, execution.append_cost_record) = original
