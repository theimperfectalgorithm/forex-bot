"""Task018C attacks, exercised against real gates with offline broker fakes."""
from datetime import datetime, timedelta, timezone
import io
import json
from types import SimpleNamespace as NS

import numpy as np
import pytest

from core import news_calendar as news, runtime_paths, trading_mode
from strategies import sma_ema_combined as sma
from src.agents import main_agent as main
import test_task015_actual_risk_sizing as sizing
from test_task018b_news_failclosed import NOW, LOG, event, payload, cache, risk_environment, run_risk
REAL_FETCH = news._fetch_feed


@pytest.fixture(autouse=True)
def clock(monkeypatch, _isolated_news):
    monkeypatch.setattr(news, '_utc_now', lambda: NOW)


def test_bracketing_and_forged_completeness_cannot_authorize_entry(monkeypatch):
    stored = cache(payload())
    stored.update(complete=True, coverage_start=(NOW-timedelta(days=3)).isoformat(),
                  coverage_end=(NOW+timedelta(days=3)).isoformat())
    news.CACHE_FILE.write_text(json.dumps(stored))
    result = news.evaluate_news('EURUSD')
    assert result.status is news.NewsStatus.UNKNOWN and result.snapshot is not None
    assert 'completeness' in result.reason
    risk_environment(monkeypatch)
    assert run_risk()['decision'] == 'REJECTED'
    result, sent = sizing._place(monkeypatch)
    assert not result['success'] and sent == []


@pytest.mark.parametrize('advance,expected', [(0, 'CLEAR'), (1, 'BLACKOUT'), (2, 'BLACKOUT')])
def test_final_send_crosses_301_seconds_using_same_snapshot(monkeypatch, advance, expected):
    # Synthetic completeness oracle isolates timing from the provider blocker.
    # The actual FF adapter cannot yield CLEAR; no production trust flag exists.
    monkeypatch.setattr(news, '_proves_coverage', lambda *_a: True)
    stamp = [NOW]
    monkeypatch.setattr(news, '_utc_now', lambda: stamp[0])
    fetches = []
    monkeypatch.setattr(news, '_fetch_feed', lambda:
                        fetches.append(1) or payload(event(NOW+timedelta(seconds=301))))
    initial = news.evaluate_news('EURUSD')
    assert initial.status is news.NewsStatus.CLEAR
    risk_environment(monkeypatch)
    assert run_risk()['decision'] == 'APPROVED'

    def ticks():
        yield NS(ask=1.1010, bid=1.1008)
        stamp[0] += timedelta(seconds=advance)
        yield NS(ask=1.1010, bid=1.1008)

    result, sent = sizing._place(monkeypatch, ticks=ticks())
    assert news.reevaluate_news(initial, 'EURUSD').status.value == expected
    assert result['success'] is (expected == 'CLEAR')
    assert len(sent) == (1 if expected == 'CLEAR' else 0)
    assert fetches == [1]


def test_final_snapshot_expiry_blocks_after_prop_query(monkeypatch):
    monkeypatch.setattr(news, '_proves_coverage', lambda *_a: True)
    snapshot = news.CalendarSnapshot(NOW-timedelta(hours=6)+timedelta(seconds=1), ())
    initial = news.NewsResult(news.NewsStatus.CLEAR, 'synthetic trusted snapshot', snapshot=snapshot)
    monkeypatch.setattr(news, 'evaluate_news', lambda *_a, **_k: initial)
    # _place installs its own prop fake; wrap the actual final news function to
    # verify it observes time advanced by the prop query in a direct setup.
    broker, sent = sizing.fake_mt5()
    monkeypatch.setattr(sizing.ex, 'mt5', broker)
    monkeypatch.setattr(sizing.ex, '_connect_for_entry', lambda *_a: True)
    stamp = [NOW]
    monkeypatch.setattr(news, '_utc_now', lambda: stamp[0])
    def prop(*_a, **_k):
        stamp[0] = NOW+timedelta(seconds=1)
        return NS(allowed=True, reason='mock prop query')
    monkeypatch.setattr(sizing.ex, 'evaluate_prop_risk', prop)
    result = sizing.ex.place_trade('EURUSD', {'signal': 'BUY'}, 9.99,
        {'sl_pips': 50, 'tp_pips': 100, 'use_live_anchor': True}, 'asian', 100, 2)
    assert not result['success'] and not sent and 'stale' in result['error']


@pytest.mark.parametrize('key,first,last', [
    ('news_fail_closed', 'true', 'false'), ('news_filter', 'true', 'false'),
    ('news_window_min', '5', '1'), ('unrelated', '1', '2')])
@pytest.mark.parametrize('which', ['global', 'local'])
def test_duplicate_yaml_rejects_both_real_entry_gates(monkeypatch, key, first, last, which):
    target = news.CONFIG_FILE if which == 'global' else news.CONFIG_FILE.parent/'local_config.yaml'
    target.write_text(f'global:\n  {key}: {first}\n  {key}: {last}\n')
    result = news.evaluate_news('EURUSD')
    assert result.source == 'config' and 'duplicate YAML key' in result.reason
    assert not result.entries_allowed
    risk_environment(monkeypatch)
    assert run_risk()['decision'] == 'REJECTED'
    result, sent = sizing._place(monkeypatch)
    assert not result['success'] and not sent


@pytest.mark.parametrize('text', [
    'global:\n  news_fail_closed: true\nglobal:\n  news_fail_closed: false\n',
    'other:\n  nested: 1\n  nested: 2\n',
    'base: &base\n  news_filter: true\nglobal:\n  <<: *base\n  news_filter: false\n'])
def test_duplicate_parent_nested_and_merged_keys_are_ambiguous(text):
    (news.CONFIG_FILE.parent/'local_config.yaml').write_text(text)
    assert not news.evaluate_news('EURUSD').entries_allowed


def cross_strategy(monkeypatch, session=False, trend=0):
    strategy = object.__new__(sma.SmaEmaCombined)
    strategy.pair = 'EURUSD'
    monkeypatch.setattr(strategy, '_connect', lambda _log: True)
    # Flat averages followed by one fall create an adverse BUY cross.
    bars = np.array([(i, 1.0 if i < 200 else .5) for i in range(201)],
                    dtype=[('time', 'i8'), ('close', 'f8')])
    monkeypatch.setattr(strategy, '_m15_bars', lambda: bars)
    monkeypatch.setattr(strategy, '_h1_ema50_trend', lambda _log: trend)
    monkeypatch.setattr(sma, 'london_ny_overlap', lambda _now: session)
    return strategy


@pytest.mark.parametrize('gate', ['trade_allowed', 'news', 'reconciliation', 'PAUSED', 'SHADOW'])
@pytest.mark.parametrize('session,trend', [(False, 1), (True, 0)])
def test_real_strategy_and_orchestrator_cross_exit_ignores_entry_gates(monkeypatch, gate, session, trend):
    state = main._fresh_state('2026-09-02')
    state['trade_allowed'] = gate != 'trade_allowed'
    state['ny_news_flag'] = gate == 'news'
    state['open_trades'] = [{'ticket': 18, 'symbol': 'EURUSD', 'strategy': 'SMA', 'direction': 'BUY'}]
    strategy = cross_strategy(monkeypatch, session, trend)
    monkeypatch.setattr(strategy, '_h1_ema50_trend',
                        lambda *_a: pytest.fail('entry-only H1 query before confirmed exit'))
    monkeypatch.setattr(main, 'EURUSD_PAIR', 'EURUSD')
    monkeypatch.setattr(main, 'check_eurusd_signals', strategy.check_signals)
    monkeypatch.setattr(trading_mode, 'get_trading_mode', lambda:
                        trading_mode.TradingModeStatus(gate, False, 'test'))
    monkeypatch.setattr(news, '_fetch_feed', lambda: pytest.fail('exit attempted calendar refresh'))
    assert news.evaluate_news('EURUSD', refresh=False).status is news.NewsStatus.UNKNOWN
    closes = []
    monkeypatch.setattr(main, 'close_trade', lambda *a, **_k: closes.append(a) or True)
    main.step_check_eurusd(state, LOG, entries_allowed=gate != 'reconciliation')
    assert closes == [(18, 'EURUSD')]


def test_actual_main_loop_dispatches_exit_despite_reconciliation_and_session(monkeypatch):
    class FixedDate(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW.replace(hour=19)  # outside entry session
    state = main._fresh_state('2026-09-02')
    state.update(trade_allowed=False, market_ran=True, london_prep_done=True,
                 ny_prep_done=True, report_ran=True)
    state['open_trades'] = [{'ticket': 18, 'symbol': 'EURUSD', 'strategy': 'SMA', 'direction': 'BUY'}]
    strategy = cross_strategy(monkeypatch)
    monkeypatch.setattr(main, 'datetime', FixedDate)
    monkeypatch.setattr(main, 'EURUSD_PAIR', 'EURUSD')
    monkeypatch.setattr(main, 'check_eurusd_signals', strategy.check_signals)
    monkeypatch.setattr(main, 'setup_logger', lambda: LOG)
    monkeypatch.setattr(main, '_bind_mt5_terminal', lambda *_a: True)
    monkeypatch.setattr(main, 'log_startup_mode', lambda *_a: None)
    monkeypatch.setattr(main, 'load_state', lambda: state)
    monkeypatch.setattr(main, 'save_state', lambda *_a: None)
    monkeypatch.setattr(main, 'evaluate_prop_risk', lambda *_a, **_k: None)
    monkeypatch.setattr(main, 'step_pre_entry_reconciliation', lambda *_a: False)
    monkeypatch.setattr(main, 'server_utc_offset_hours', lambda: 0)
    monkeypatch.setattr(main, 'step_monitor_positions', lambda *_a: None)
    for name in ('AMR_KEYS', 'MON_KEYS', 'BREAKOUT_KEYS'):
        monkeypatch.setattr(main, name, [])
    def stop(*_a):
        raise KeyboardInterrupt
    monkeypatch.setattr(main, 'sleep_until_next_quarter', stop)
    closes = []
    monkeypatch.setattr(main, 'close_trade', lambda *a: closes.append(a) or True)
    main.main()
    assert closes == [(18, 'EURUSD')]


@pytest.mark.parametrize('suffix,flag', [('@amr', 'asian_exit_done'), ('@mon', 'monday_exit_done')])
@pytest.mark.parametrize('snapshot', ['missing', 'stale', 'blackout'])
def test_scheduled_exit_uses_only_validated_memory(monkeypatch, suffix, flag, snapshot):
    if snapshot != 'missing':
        fetched = NOW-timedelta(hours=6) if snapshot == 'stale' else NOW
        monkeypatch.setattr(news, '_memory_snapshot', news.CalendarSnapshot(fetched, (('USD', 'release', NOW),)))
    monkeypatch.setattr(news, '_fetch_feed', lambda: pytest.fail('network on exit'))
    monkeypatch.setattr(news, '_load_cache', lambda: pytest.fail('unvalidated disk cache on exit'))
    closes = []
    monkeypatch.setattr(main, 'close_trade', lambda *a, **_k: closes.append(a) or True)
    state = {'open_trades': [{'ticket': 18, 'symbol': 'EURUSD', 'strategy_key': 'EURUSD'+suffix}]}
    main.step_asian_time_exit(state, LOG, suffix=suffix, done_flag=flag)
    assert bool(closes) is (snapshot != 'blackout')
    assert bool(state.get(flag)) is (snapshot != 'blackout')
    main.tj.market_context('EURUSD')  # journal may not refresh either


@pytest.mark.parametrize('relative', ['data', 'data/state', 'data/logs/subdir', 'data/journal',
                                     'journals', 'logs', 'state', 'reports/runtime'])
def test_production_descendant_runtime_selection_rejected(monkeypatch, relative):
    monkeypatch.setenv(runtime_paths.DATA_DIR_ENV, str(runtime_paths.REPO_ROOT/relative))
    with pytest.raises(RuntimeError, match='pytest may not use production data'):
        runtime_paths.data_dir()


@pytest.mark.parametrize('relative', ['data/state/task018d_probe', 'data/logs/task018d_probe',
                                     'journals/task018d_probe', 'reports/task018d_probe'])
def test_hardcoded_operational_writes_are_blocked(relative):
    with pytest.raises(RuntimeError, match='production'):
        (runtime_paths.REPO_ROOT/relative).write_text('must never be written')


def test_slow_failure_has_one_fetch_per_cycle_and_recovers(monkeypatch):
    monotonic = [1000.0]
    monkeypatch.setattr(news.time, 'monotonic', lambda: monotonic[0])
    calls = []
    def slow_failure():
        calls.append(1)
        monotonic[0] += 1000  # simulated slow DNS/body; no real sleep/network
        raise TimeoutError('simulated slow response')
    monkeypatch.setattr(news, '_fetch_feed', slow_failure)
    for symbol in ('EURUSD', 'GBPUSD', 'AUDJPY', 'EURUSD'):
        assert news.evaluate_news(symbol).status is news.NewsStatus.UNKNOWN
    assert calls == [1]
    monotonic[0] += news.RETRY_SECONDS
    monkeypatch.setattr(news, '_fetch_feed', lambda: calls.append(1) or payload(event()))
    assert news.evaluate_news('EURUSD').status is news.NewsStatus.BLACKOUT
    assert calls == [1, 1]


def test_response_and_cache_size_are_bounded(monkeypatch):
    class Response(io.BytesIO):
        def read(self, size=-1):
            assert size == news.MAX_RESPONSE_BYTES+1
            return super().read(size)
    calls = []
    def open_response(req, timeout):
        calls.append(timeout)
        return Response(b' '*(news.MAX_RESPONSE_BYTES+2))
    monkeypatch.setattr(news.urllib.request, 'urlopen', open_response)
    # Restore real fetch because autouse isolation forbids a live one.
    monkeypatch.setattr(news, '_fetch_feed', REAL_FETCH)
    assert news.evaluate_news('EURUSD').status is news.NewsStatus.UNKNOWN
    assert calls == [news.NETWORK_TIMEOUT_SECONDS]
    news.CACHE_FILE.write_bytes(b' '*(news.MAX_RESPONSE_BYTES+2))
    with pytest.raises(ValueError, match='size limit'):
        news._load_cache()


def test_bare_clear_without_snapshot_cannot_authorize_send(monkeypatch):
    monkeypatch.setattr(news, 'evaluate_news', lambda *_a:
                        news.NewsResult(news.NewsStatus.CLEAR, 'no evidence'))
    result, sent = sizing._place(monkeypatch)
    assert not result['success'] and not sent
