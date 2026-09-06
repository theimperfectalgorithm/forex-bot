"""Task018B: deterministic calendar evidence and entry/management boundaries."""
from __future__ import annotations

import ast
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from core import news_calendar as news
from src.agents import agent_risk as risk
from src.agents import agent_execution as execution
from src.agents import main_agent as main
import test_task015_actual_risk_sizing as sizing
import test_task012_friday_close_retry as friday


NOW = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)
LOG = logging.getLogger('task018b')


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch, _isolated_news):
    monkeypatch.setattr(news, '_utc_now', lambda: NOW)


def event(when=NOW, currency='USD', impact='High'):
    return {'title': 'Test release', 'country': currency, 'impact': impact,
            'date': when.isoformat()}


def payload(*events):
    # Bracketing records are NOT completeness evidence (Task018D).
    return [event(NOW - timedelta(days=2), impact='Low'), *events,
            event(NOW + timedelta(days=2), impact='Low')]


def cache(raw=None, fetched=NOW):
    return {'schema_version': 1, 'fetched_at': fetched.isoformat(),
            'raw_events': payload() if raw is None else raw}


def install_feed(monkeypatch, raw):
    monkeypatch.setattr(news, '_fetch_feed', lambda: deepcopy(raw))


def evaluate(symbol='EURUSD', **kwargs):
    return news.evaluate_news(symbol, now=NOW, **kwargs)


def unavailable():
    raise OSError('calendar offline')


@pytest.mark.parametrize('seconds,blocked', [(-301, False), (-300, True),
                                          (0, True), (300, True), (301, False)])
def test_inclusive_blackout_boundaries(monkeypatch, seconds, blocked):
    install_feed(monkeypatch, payload(event(NOW + timedelta(seconds=seconds))))
    result = evaluate()
    assert result.status is (news.NewsStatus.BLACKOUT if blocked else news.NewsStatus.UNKNOWN)
    assert not result.entries_allowed
    assert bool(result.matching_events) is blocked


@pytest.mark.parametrize('symbol,currency,blocked', [
    ('EURUSD', 'USD', True), ('EURUSD', 'EUR', True), ('EURUSD', 'CAD', False),
    ('CADJPY', 'CAD', True), ('CADJPY', 'JPY', True), ('AUDJPY', 'AUD', True),
    ('XAUUSD', 'USD', True), ('XAUUSD', 'EUR', False),
])
def test_currency_relevance(monkeypatch, symbol, currency, blocked):
    install_feed(monkeypatch, payload(event(currency=currency)))
    assert evaluate(symbol).status is (news.NewsStatus.BLACKOUT if blocked else news.NewsStatus.UNKNOWN)


def test_validated_no_high_events_is_unknown(monkeypatch):
    install_feed(monkeypatch, payload())
    result = evaluate()
    assert result.status is news.NewsStatus.UNKNOWN and result.source == 'feed'
    assert not result.entries_allowed and not result.matching_events
    # New cache preserves compatibility keys without trusting high-only data.
    saved = json.loads(news.CACHE_FILE.read_text())
    assert saved['events'] == [] and saved['raw_events'] == payload()
    monkeypatch.setattr(news, '_fetch_feed', unavailable)
    assert evaluate().source == 'memory'


@pytest.mark.parametrize('raw', [None, [], {}, {'events': []}, [None], [42],
                                [event(impact='Unknown')]])
def test_ambiguous_or_malformed_feed_is_unknown(monkeypatch, raw):
    install_feed(monkeypatch, raw)
    result = evaluate()
    assert result.status is news.NewsStatus.UNKNOWN and not result.entries_allowed


@pytest.mark.parametrize('field,value', [
    ('date', None), ('date', '2026-09-02T12:00:00'), ('date', 'bad'),
    ('impact', None), ('country', ''), ('country', '???'), ('title', None),
])
def test_partial_record_failure_never_discards_high_event(monkeypatch, field, value):
    bad = event(currency='CAD')
    bad[field] = value
    install_feed(monkeypatch, payload(event(currency='EUR'), bad))
    assert evaluate().status is news.NewsStatus.UNKNOWN
    assert not news.CACHE_FILE.exists()


@pytest.mark.parametrize('symbol', ['UNKNOWN', 'EURUSD.a', 'USD', 'ABCUSD',
                                  'EURGBPextra', 'USDUSD', '', None])
def test_unmappable_symbol_is_unknown(monkeypatch, symbol):
    install_feed(monkeypatch, payload())
    result = evaluate(symbol)
    assert result.status is news.NewsStatus.UNKNOWN and not result.entries_allowed


def test_network_exception_is_unknown(monkeypatch):
    monkeypatch.setattr(news, '_fetch_feed', unavailable)
    result = evaluate()
    assert result.status is news.NewsStatus.UNKNOWN and not result.entries_allowed
    assert 'calendar offline' in result.reason


@pytest.mark.parametrize('stored', [None, [], {}, {'fetched_at': NOW.isoformat(), 'events': []},
                                   cache([]), cache([event(impact='unclassified')])])
def test_invalid_cache_cannot_be_fallback(monkeypatch, stored):
    news.CACHE_FILE.write_text(json.dumps(stored))
    monkeypatch.setattr(news, '_fetch_feed', unavailable)
    result = evaluate()
    assert result.status is news.NewsStatus.UNKNOWN and not result.entries_allowed


def test_partial_cache_write_is_unknown(monkeypatch):
    news.CACHE_FILE.write_text('{"schema_version":')
    monkeypatch.setattr(news, '_fetch_feed', unavailable)
    assert evaluate().status is news.NewsStatus.UNKNOWN


@pytest.mark.parametrize('age,allowed', [
    (timedelta(hours=5, minutes=59), True), (timedelta(hours=6), False),
    (timedelta(hours=71), False), (timedelta(minutes=-4), True),
    (timedelta(minutes=-6), False), (timedelta(days=-10000), False),
])
def test_cache_age_policy(monkeypatch, age, allowed):
    news.CACHE_FILE.write_text(json.dumps(cache(fetched=NOW-age)))
    monkeypatch.setattr(news, '_fetch_feed', unavailable)
    result = evaluate()
    assert not result.entries_allowed
    assert (result.snapshot is not None) is allowed


@pytest.mark.parametrize('stamp', ['bad', '2026-09-02T12:00:00', None])
def test_cache_timestamp_must_be_usable(monkeypatch, stamp):
    stored = cache()
    stored['fetched_at'] = stamp
    news.CACHE_FILE.write_text(json.dumps(stored))
    monkeypatch.setattr(news, '_fetch_feed', unavailable)
    assert evaluate().status is news.NewsStatus.UNKNOWN


@pytest.mark.parametrize('raw', [
    payload(event(NOW + timedelta(days=10))),
    [event(NOW - timedelta(days=7)), event(NOW - timedelta(days=3))],
    [event(NOW, impact='Low'), event(NOW + timedelta(hours=1), impact='Low')],
])
def test_unproven_period_and_window_coverage_is_unknown(monkeypatch, raw):
    install_feed(monkeypatch, raw)
    assert evaluate().status is news.NewsStatus.UNKNOWN


def test_week_rollover_cannot_relabel_old_payload_as_current(monkeypatch):
    old = [event(NOW - timedelta(days=7)), event(NOW - timedelta(days=3))]
    news.CACHE_FILE.write_text(json.dumps(cache(old)))
    install_feed(monkeypatch, old)
    assert evaluate().status is news.NewsStatus.UNKNOWN


def test_midnight_window_requires_both_sides(monkeypatch):
    midnight = NOW.replace(hour=0)
    install_feed(monkeypatch, [event(midnight), event(NOW)])
    assert evaluate(when=midnight).status is news.NewsStatus.BLACKOUT


def test_slow_fetch_crossing_blackout_uses_completion_time(monkeypatch):
    ticks = iter([NOW, NOW + timedelta(seconds=2)])
    monkeypatch.setattr(news, '_utc_now', lambda: next(ticks))
    install_feed(monkeypatch, payload(event(NOW + timedelta(seconds=301))))
    assert news.evaluate_news('EURUSD').status is news.NewsStatus.BLACKOUT


def test_slow_fetch_crossing_coverage_end_is_unknown(monkeypatch):
    ticks = iter([NOW, NOW + timedelta(seconds=2)])
    monkeypatch.setattr(news, '_utc_now', lambda: next(ticks))
    install_feed(monkeypatch, [event(NOW-timedelta(hours=1)),
                               event(NOW+timedelta(seconds=301), impact='Low')])
    assert news.evaluate_news('EURUSD').status is news.NewsStatus.UNKNOWN


def test_aware_offset_conversion_and_naive_query(monkeypatch):
    install_feed(monkeypatch, payload(event(NOW.astimezone(timezone(timedelta(hours=3))))))
    assert evaluate().status is news.NewsStatus.BLACKOUT
    assert evaluate(when=NOW.replace(tzinfo=None)).status is news.NewsStatus.UNKNOWN


def test_valid_feed_recovers_invalid_cache(monkeypatch):
    news.CACHE_FILE.write_text('broken')
    install_feed(monkeypatch, payload())
    assert evaluate().status is news.NewsStatus.UNKNOWN
    assert evaluate().snapshot is not None


def test_bad_refresh_does_not_overwrite_previous_cache(monkeypatch):
    original = json.dumps(cache(fetched=NOW-timedelta(hours=7)))
    news.CACHE_FILE.write_text(original)
    install_feed(monkeypatch, payload({'impact': 'High'}))
    assert evaluate().status is news.NewsStatus.UNKNOWN
    assert news.CACHE_FILE.read_text() == original


def test_cache_replace_failure_preserves_file_and_valid_live_result(monkeypatch):
    news.CACHE_FILE.write_text('old')
    install_feed(monkeypatch, payload())
    calls = []
    def fail_replace(source, target):
        calls.append((source, target))
        assert json.loads(Path(source).read_text())['raw_events'] == payload()
        raise OSError('disk failure')
    monkeypatch.setattr(news.os, 'replace', fail_replace)
    assert evaluate().status is news.NewsStatus.UNKNOWN
    assert evaluate().snapshot is not None
    assert news.CACHE_FILE.read_text() == 'old'
    assert len(calls) == 1 and not Path(calls[0][0]).exists()


@pytest.mark.parametrize('key,value', [('news_filter', '"false"'),
                                     ('news_fail_closed', '"false"'),
                                     ('news_window_min', '-5'),
                                     ('news_window_min', 'true'),
                                     ('news_window_min', '5.5')])
def test_invalid_config_cannot_weaken_protection(monkeypatch, key, value):
    local = news.CONFIG_FILE.parent / 'local_config.yaml'
    local.write_text(f'global:\n  {key}: {value}\n')
    install_feed(monkeypatch, payload())
    result = evaluate()
    assert result.status is news.NewsStatus.UNKNOWN and not result.entries_allowed
    assert result.source == 'config'


@pytest.mark.parametrize('text', ['global: [', 'global: []', 'null', '[]'])
def test_malformed_local_config_is_not_ignored(text):
    (news.CONFIG_FILE.parent / 'local_config.yaml').write_text(text)
    assert not evaluate().entries_allowed


def test_missing_global_config_blocks():
    news.CONFIG_FILE.unlink()
    assert not evaluate().entries_allowed


@pytest.mark.parametrize('text', ['{}', 'global: {}', 'global:\n  news_filter: true\n'])
def test_missing_effective_news_configuration_blocks(text):
    news.CONFIG_FILE.write_text(text)
    assert not evaluate().entries_allowed


def test_configured_window_is_preserved(monkeypatch):
    (news.CONFIG_FILE.parent / 'local_config.yaml').write_text('global:\n  news_window_min: 8\n')
    install_feed(monkeypatch, payload(event(NOW+timedelta(minutes=8))))
    assert evaluate().status is news.NewsStatus.BLACKOUT
    assert evaluate(when=NOW-timedelta(seconds=1)).status is news.NewsStatus.UNKNOWN


def test_local_override_and_failopen_semantics(monkeypatch):
    local = news.CONFIG_FILE.parent / 'local_config.yaml'
    local.write_text('global:\n  news_fail_closed: false\n')
    monkeypatch.setattr(news, '_fetch_feed', unavailable)
    result = evaluate()
    assert result.status is news.NewsStatus.UNKNOWN and result.entries_allowed
    assert 'UNKNOWN / FAIL-OPEN' in result.entry_message
    monkeypatch.setattr(news, '_retry_after', 0.0)  # a later refresh cycle
    install_feed(monkeypatch, payload(event()))
    assert not evaluate().entries_allowed  # fail-open does not bypass known news
    local.write_text('global:\n  news_fail_closed: true\n')
    news.CACHE_FILE.unlink()
    monkeypatch.setattr(news, '_memory_snapshot', None)
    monkeypatch.setattr(news, '_fetch_feed', unavailable)
    assert not evaluate().entries_allowed


def test_filter_disabled_is_explicit_unknown_permission():
    (news.CONFIG_FILE.parent / 'local_config.yaml').write_text('global:\n  news_filter: false\n')
    result = evaluate()
    assert result.status is news.NewsStatus.UNKNOWN and result.entries_allowed
    assert 'FILTER DISABLED' in result.entry_message


def risk_environment(monkeypatch):
    balance = max(risk.HARD_FLOOR * 2, 100000)
    monkeypatch.setattr(risk, 'mt5', NS(initialize=lambda: True,
                        account_info=lambda: NS(balance=balance, equity=balance)))
    monkeypatch.setattr(risk, '_open_risk_usd', lambda _log: (0, 0))
    monkeypatch.setattr(risk, '_same_currency_count', lambda _s: 0)
    monkeypatch.setattr(risk, '_spread_pips', lambda *_a: 0)
    monkeypatch.setattr(risk, 'evaluate_prop_risk', lambda *_a, **_k:
                        NS(allowed=True, reason='test'))


def run_risk():
    return risk.run('EURUSD', 'BUY', 50, {}, risk_pct=.001)


@pytest.mark.parametrize('status,allowed', [('CLEAR', True), ('BLACKOUT', False), ('UNKNOWN', False)])
def test_risk_and_direct_execution_enforce_structured_result(monkeypatch, status, allowed):
    monkeypatch.setattr(news, '_proves_coverage', lambda *_a: True)
    result = news.NewsResult(news.NewsStatus(status), 'test evidence',
                            snapshot=news.CalendarSnapshot(NOW, ()))
    monkeypatch.setattr(news, 'evaluate_news', lambda *_a, **_k: result)
    risk_environment(monkeypatch)
    assert (run_risk()['decision'] == 'APPROVED') is allowed
    placed, sent = sizing._place(monkeypatch)
    assert placed['success'] is allowed
    assert len(sent) == (1 if allowed else 0)


@pytest.mark.parametrize('final_status', ['UNKNOWN', 'BLACKOUT'])
def test_clear_risk_then_changed_execution_never_sends(monkeypatch, final_status):
    results = iter([news.NewsResult(news.NewsStatus.CLEAR, 'earlier'),
                    news.NewsResult(news.NewsStatus(final_status), 'changed')])
    monkeypatch.setattr(news, 'evaluate_news', lambda *_a, **_k: next(results))
    risk_environment(monkeypatch)
    assert run_risk()['decision'] == 'APPROVED'
    result, sent = sizing._place(monkeypatch)
    assert not result['success'] and sent == []
    assert f'NEWS {final_status}' in result['error']


@pytest.mark.parametrize('bad', [None, ('CLEAR', True), news.NewsResult('invalid', 'bad'),
                                news.NewsResult(news.NewsStatus.UNKNOWN, 'bad', fail_closed=None)])
def test_invalid_helper_result_blocks_both_callers(monkeypatch, bad):
    monkeypatch.setattr(news, 'evaluate_news', lambda *_a, **_k: bad)
    risk_environment(monkeypatch)
    assert run_risk()['decision'] == 'REJECTED'
    result, sent = sizing._place(monkeypatch)
    assert not result['success'] and sent == []


def test_unexpected_helper_exception_blocks_even_failopen(monkeypatch):
    (news.CONFIG_FILE.parent / 'local_config.yaml').write_text('global:\n  news_fail_closed: false\n')
    def broken(*_a, **_k):
        raise RuntimeError('unexpected helper failure')
    monkeypatch.setattr(news, 'evaluate_news', broken)
    risk_environment(monkeypatch)
    assert run_risk()['decision'] == 'REJECTED'
    result, sent = sizing._place(monkeypatch)
    assert not result['success'] and not sent


def test_real_unknown_failopen_is_explicit_at_both_callers(monkeypatch, caplog):
    (news.CONFIG_FILE.parent / 'local_config.yaml').write_text('global:\n  news_fail_closed: false\n')
    real = news.evaluate_news
    monkeypatch.setattr(news, 'evaluate_news', lambda symbol: real(symbol, now=NOW))
    risk_environment(monkeypatch)
    with caplog.at_level(logging.INFO):
        assert run_risk()['decision'] == 'APPROVED'
        result, sent = sizing._place(monkeypatch)
    assert result['success'] and len(sent) == 1
    assert caplog.text.count('NEWS UNKNOWN / FAIL-OPEN') >= 2
    assert 'NEWS CLEAR' not in caplog.text


@pytest.mark.parametrize('suffix,flag', [('@amr', 'asian_exit_done'), ('@mon', 'monday_exit_done')])
@pytest.mark.parametrize('status,closes', [('CLEAR', True), ('UNKNOWN', True), ('BLACKOUT', False)])
def test_scheduled_exit_distinguishes_unknown_from_blackout(monkeypatch, suffix, flag, status, closes):
    monkeypatch.setattr(news, 'evaluate_news', lambda *_a, **_k:
                        news.NewsResult(news.NewsStatus(status), 'test'))
    calls = []
    monkeypatch.setattr(main, 'close_trade', lambda *_a, **_k: calls.append(1) or True)
    state = {'open_trades': [{'strategy_key': 'GBPUSD'+suffix, 'symbol': 'GBPUSD', 'ticket': 7}]}
    main.step_asian_time_exit(state, LOG, suffix=suffix, done_flag=flag)
    assert bool(calls) is closes and bool(state.get(flag)) is closes


def test_scheduled_exit_helper_exception_still_closes(monkeypatch):
    monkeypatch.setattr(news, 'evaluate_news', lambda *_a, **_k: unavailable())
    calls = []
    monkeypatch.setattr(main, 'close_trade', lambda *_a, **_k: calls.append(1) or True)
    state = {'open_trades': [{'strategy_key': 'AUDJPY@amr', 'symbol': 'AUDJPY', 'ticket': 7}]}
    main.step_asian_time_exit(state, LOG)
    assert calls == [1] and state['asian_exit_done']


def test_unknown_news_does_not_enter_friday_or_accounting_path(monkeypatch):
    monkeypatch.setattr(news, 'evaluate_news', lambda *_a, **_k:
                        news.NewsResult(news.NewsStatus.UNKNOWN, 'offline'))
    trade = friday._trade(1)
    state = friday._state(trade)
    friday._broker(monkeypatch, {1: [True, True, True, False]})
    calls = []
    monkeypatch.setattr(main, 'close_trade', lambda *_a, **_k: calls.append(1) or len(calls) == 2)
    main.step_friday_close(state, LOG)
    assert not state['friday_close_done']
    main.step_friday_close(state, LOG)
    assert state['friday_close_done'] and calls == [1, 1]
    closed = {**trade, 'exit_price': 1.21, 'exit_time': NOW.isoformat(),
              'exit_reason': 'FRIDAY_CLOSE', 'exit_pnl': 10.0}
    exits = []
    monkeypatch.setattr(main, '_check_untracked_positions', lambda *_a, **_k: None)
    monkeypatch.setattr(main, 'monitor_positions', lambda *_a: ([], [closed]))
    monkeypatch.setattr(main.tj, 'log_exit', lambda item: exits.append(item['ticket']))
    main.step_monitor_positions(state, LOG)
    main.step_monitor_positions(state, LOG)
    assert exits == [1] and state['closed_today'] == [closed]
    assert state['daily_pnl'] == 10.0 and state['open_trades'] == []


def test_unknown_news_does_not_break_reconciliation_reporting_or_journal(monkeypatch):
    from core import trade_journal as journal
    from src.agents import agent_reporting as reporting
    monkeypatch.setattr(news, '_fetch_feed', unavailable)
    assert news.evaluate_news('EURUSD').status is news.NewsStatus.UNKNOWN
    state = main._fresh_state(NOW.date().isoformat())
    reconciled = []
    monkeypatch.setattr(main, 'initialize_and_validate', lambda _log: True)
    monkeypatch.setattr(main, '_check_untracked_positions',
                        lambda *_a, **_k: reconciled.append(1))
    assert main.step_pre_entry_reconciliation(state, LOG) and reconciled == [1]
    monkeypatch.setattr(reporting, 'mt5', NS(initialize=lambda: False))
    assert reporting.run(state)['success']
    monkeypatch.setattr(journal, 'MT5_AVAILABLE', False)
    assert journal.market_context('EURUSD')['minutes_to_next_high_news'] is None
    journal.log_event('task018b_unknown_news_test', {'ticket': 18001})
    assert 'task018b_unknown_news_test' in journal.JOURNAL_FILE.read_text()


def test_unknown_news_does_not_block_sltp_management(monkeypatch):
    monkeypatch.setattr(news, 'evaluate_news', lambda *_a, **_k:
                        news.NewsResult(news.NewsStatus.UNKNOWN, 'offline'))
    sent = []
    monkeypatch.setattr(execution, 'mt5', NS(TRADE_ACTION_SLTP=6, TRADE_RETCODE_DONE=10009,
                        order_send=lambda req: sent.append(req) or NS(retcode=10009)))
    position = NS(price_current=1.3, price_open=1.2, sl=1.1, tp=1.4, ticket=18)
    assert execution._apply_breakeven(position, {'symbol': 'EURUSD', 'direction': 'BUY'}, LOG)
    assert len(sent) == 1 and sent[0]['action'] == 6 and sent[0]['position'] == 18


def test_management_functions_do_not_use_entry_news_gate():
    # Guard the scope of the new execution gate, including close and SL/TP sends.
    functions = ast.parse(Path(execution.__file__).read_text(encoding='utf-8'))
    for node in functions.body:
        if isinstance(node, ast.FunctionDef) and node.name != 'place_trade':
            assert 'evaluate_news' not in ast.unparse(node)


@pytest.mark.parametrize('family', ['amr', 'monday', 'arb', 'eurusd'])
@pytest.mark.parametrize('stage', ['risk', 'execution'])
@pytest.mark.parametrize('status', ['UNKNOWN', 'BLACKOUT'])
def test_news_rejection_never_acknowledges_or_consumes(monkeypatch, family, stage, status):
    state = main._fresh_state('2026-09-02')
    state['trade_allowed'] = True
    state['eurusd']['ema_pullback_pending'] = True
    state['eurusd']['ema_pullback_dir'] = 'BUY'
    signal = {'signal': 'BUY', 'sl_pips': 50, 'tp_pips': 100, 'entry_price': 1.1,
              'reason': 'test setup', 'strategy': 'EMA', 'trigger_bar_close': 1.1}
    key = {'amr': 'EURUSD@amr', 'monday': 'EURUSD@mon',
           'arb': 'EURUSD@arb', 'eurusd': 'EURUSD'}[family]
    state['session_data'] = {key: {'sl_pips': 50, 'tp_pips': 100}}
    state['london_traded'][key] = False
    monkeypatch.setattr(main, 'AMR_KEYS', [key])
    monkeypatch.setattr(main, 'BREAKOUT_KEYS', [key])
    monkeypatch.setattr(main, 'EURUSD_PAIR', 'EURUSD')
    monkeypatch.setattr(main, 'check_asian_reversion', lambda *_a: deepcopy(signal))
    monkeypatch.setattr(main, 'check_breakout', lambda *_a: deepcopy(signal))
    monkeypatch.setattr(main, 'check_eurusd_signals', lambda eu, _open: ([deepcopy(signal)], dict(eu)))
    monkeypatch.setattr(main, 'allow_or_log_entry', lambda *_a: True)
    rejected = news.NewsResult(news.NewsStatus(status), 'unavailable or blackout')
    sequence = iter([rejected] if stage == 'risk' else [
        news.NewsResult(news.NewsStatus.CLEAR, 'earlier'), rejected])
    monkeypatch.setattr(news, 'evaluate_news', lambda *_a, **_k: next(sequence))
    risk_environment(monkeypatch)
    monkeypatch.setattr(main, 'run_risk', lambda *_a, **_k: run_risk())
    sent = []
    def place(*_a, **_k):
        result, orders = sizing._place(monkeypatch)
        sent.extend(orders)
        return result
    monkeypatch.setattr(main, 'place_trade', place)
    acknowledgements, entries = [], []
    monkeypatch.setattr(main, 'acknowledge_trade', lambda *_a: acknowledgements.append(1))
    monkeypatch.setattr(main.tj, 'log_entry', lambda **_k: entries.append(1))
    monkeypatch.setattr(main.tj, 'log_signal', lambda **_k: None)
    monkeypatch.setattr(main.tj, 'log_rejection', lambda **_k: None)
    before = deepcopy(state)
    if family == 'arb':
        main.step_check_breakouts(state, 'london', LOG)
    elif family == 'eurusd':
        main.step_check_eurusd(state, LOG)
    else:
        main.step_check_asian_reversion(state, LOG, keys=[key],
                                       flag='monday_traded' if family == 'monday' else 'asian_traded')
    assert not acknowledgements and not entries and not sent
    assert state == before
