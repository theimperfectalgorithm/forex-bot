"""Offline native-output fixtures; never run an MQL service or query a terminal."""
import ast
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
from pathlib import Path
import re

import pytest

from core import calendar_bridge as bridge

NOW = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)
STAMP = int(NOW.timestamp())
OFFSET = 10800
IDENTITY = bridge.ExpectedIdentity('26520700', 'FivePercentOnline-Real',
    'C:\\MT5-5ers', 'C:\\shadow-fixture\\fivepercent-data', 'fivepercent-shadow', 'Fixture company')


def generation(*events, started=STAMP, offset=OFFSET, boot='boot-fixture-001', sequence=1):
    return {
        'schema_version': 1, 'source': bridge.SOURCE,
        'instance_id': IDENTITY.instance_id, 'boot_id': boot, 'sequence': str(sequence),
        'identity': {'login': IDENTITY.login, 'server': IDENTITY.server,
                     'company': IDENTITY.company, 'terminal_path': IDENTITY.terminal_path,
                     'terminal_data_path': IDENTITY.terminal_data_path},
        'clock': {'generated_server_time': started+offset, 'generated_utc_time': started,
                  'server_utc_offset_seconds': offset, 'offset_sample_time': started,
                  'clock_status': 'VALID', 'clock_uncertainty_seconds': 0,
                  'offset_before_seconds': offset, 'offset_after_seconds': offset,
                  'quote_age_before_seconds': 1, 'quote_age_after_seconds': 1},
        'query': {'server_start': started-3600+offset, 'server_end': started+3600+offset,
                  'utc_start': started-3600, 'utc_end': started+3600,
                  'started_utc': started, 'elapsed_ms': 0, 'return_count': len(events),
                  'error_code': 0, 'query_success': True, 'failure_stage': ''},
        'health': {'terminal_connected': True, 'event_enrichment_complete': True,
                   'country_enrichment_complete': True, 'currency_catalog_valid': True,
                   'change_before': '100', 'change_after': '100',
                   'change_error_before': 0, 'change_error_after': 0},
        'coverage': {'utc_start': started-3600, 'utc_end': started+3600,
                     'supported_currencies': sorted(bridge.SUPPORTED_CURRENCIES),
                     'returned_event_count': len(events)},
        'events': list(events),
    }


def event(*, at=STAMP, currency='USD', importance='HIGH', mode='DATETIME', offset=OFFSET):
    return {'value_id': '101', 'event_id': '202', 'country_id': '840',
            'country_code': 'US', 'currency': currency, 'importance': importance,
            'time_mode': mode, 'name': 'Native fixture release', 'server_time': at+offset,
            'utc_time': at if mode == 'DATETIME' else None}


def publish(directory, payload, *, raw=None):
    raw = json.dumps(payload, separators=(',', ':'), ensure_ascii=False).encode() if raw is None else raw
    filename = f"calendar_{payload['boot_id']}_{payload['sequence']}.json"
    manifest = {'schema_version': 1, 'instance_id': payload['instance_id'],
                'boot_id': payload['boot_id'], 'sequence': payload['sequence'],
                'payload_filename': filename, 'payload_bytes': len(raw),
                'payload_sha256': hashlib.sha256(raw).hexdigest(),
                'published_utc': payload['clock']['generated_utc_time']}
    (directory/filename).write_bytes(raw)
    (directory/'manifest.json').write_text(json.dumps(manifest))
    return directory/filename, manifest


def read(tmp_path, payload, symbol='GBPUSD', *, now=NOW):
    publish(tmp_path, payload)
    return bridge.BridgeReader(tmp_path, IDENTITY).read(symbol, now=now)


def test_valid_generation_preserves_occurrence_and_parent_ids(tmp_path):
    result = read(tmp_path, generation(event()))
    assert result.state is bridge.EvidenceState.VALID and result.shadow_state == 'BLACKOUT'
    assert result.events[0].value_id == '101' and result.events[0].event_id == '202'
    assert result.matching_value_ids == ('101',) and not result.entries_allowed


def test_successful_native_zero_is_valid_but_never_entry_permission(tmp_path):
    result = read(tmp_path, generation())
    assert result.state is bridge.EvidenceState.VALID and result.shadow_state == 'CLEAR'
    assert result.events == () and not result.entries_allowed


@pytest.mark.parametrize('section,field,value', [
    ('query', 'return_count', -1), ('query', 'error_code', 5400),
    ('query', 'error_code', 5401), ('query', 'error_code', 5402),
    ('query', 'error_code', 4001), ('query', 'error_code', 4004),
    ('query', 'query_success', False), ('query', 'query_success', 1),
    ('query', 'error_code', False), ('query', 'return_count', 1),
    ('query', 'elapsed_ms', bridge.MAX_QUERY_MS+1), ('query', 'failure_stage', 'native'),
    ('coverage', 'returned_event_count', 1),
    ('health', 'event_enrichment_complete', False),
    ('health', 'country_enrichment_complete', False),
    ('health', 'terminal_connected', False), ('health', 'currency_catalog_valid', False),
    ('health', 'change_after', '101'), ('health', 'change_error_before', 5400),
    ('health', 'change_error_after', 5401), ('health', 'change_before', '0'),
    ('clock', 'clock_status', 'UNKNOWN'), ('clock', 'offset_after_seconds', 7200),
    ('clock', 'offset_before_seconds', 7200), ('clock', 'quote_age_before_seconds', 11),
    ('clock', 'quote_age_after_seconds', 11), ('clock', 'clock_uncertainty_seconds', 3),
    ('clock', 'generated_server_time', STAMP+OFFSET+3600),
    ('clock', 'generated_utc_time', STAMP+60), ('clock', 'offset_sample_time', STAMP-1),
    ('query', 'utc_start', STAMP-7200), ('query', 'server_end', STAMP+7200+OFFSET),
    ('coverage', 'utc_end', STAMP+7200),
    ('coverage', 'supported_currencies', ['AUD', 'CAD', 'EUR', 'GBP', 'JPY']),
])
def test_rejects_failed_or_inconsistent_evidence(tmp_path, section, field, value):
    payload = generation()
    payload[section][field] = value
    result = read(tmp_path, payload)
    assert result.state is bridge.EvidenceState.INVALID and result.shadow_state == 'UNKNOWN'


@pytest.mark.parametrize('field,value', [
    ('login', '999999'), ('server', 'Demo-Server'), ('company', 'Wrong company'),
    ('terminal_path', 'C:\\MT5-Demo'), ('terminal_data_path', 'C:\\demo-data'),
])
def test_identity_mismatch(tmp_path, field, value):
    payload = generation()
    payload['identity'][field] = value
    assert read(tmp_path, payload).state is bridge.EvidenceState.IDENTITY_MISMATCH


def test_wrong_instance_is_identity_mismatch(tmp_path):
    payload = generation()
    payload['instance_id'] = 'demo-instance'
    assert read(tmp_path, payload).state is bridge.EvidenceState.IDENTITY_MISMATCH


@pytest.mark.parametrize('age,expected', [(89, 'VALID'), (90, 'STALE'), (3600, 'STALE')])
def test_expiry_from_query_start(tmp_path, age, expected):
    payload = generation(started=STAMP-age)
    assert read(tmp_path, payload).state.value == expected


def test_heartbeat_and_file_mtime_do_not_renew_evidence(tmp_path):
    payload = generation(started=STAMP-100)
    _, manifest = publish(tmp_path, payload)
    manifest['published_utc'] = STAMP
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    assert bridge.BridgeReader(tmp_path, IDENTITY).read('GBPUSD', now=NOW).state.value == 'STALE'


def test_sequence_rollback_and_same_sequence_rewrite(tmp_path):
    reader = bridge.BridgeReader(tmp_path, IDENTITY)
    publish(tmp_path, generation(sequence=2))
    assert reader.read('GBPUSD', now=NOW).state.value == 'VALID'
    assert reader.read('GBPUSD', now=NOW).state.value == 'VALID'
    publish(tmp_path, generation(sequence=1))
    assert reader.read('GBPUSD', now=NOW).state.value == 'INVALID'
    publish(tmp_path, generation(event(), sequence=2))
    assert reader.read('GBPUSD', now=NOW).state.value == 'INVALID'


def test_boot_change_quarantines_and_rejects_retired_boot(tmp_path):
    reader = bridge.BridgeReader(tmp_path, IDENTITY)
    publish(tmp_path, generation())
    assert reader.read('GBPUSD', now=NOW).state.value == 'VALID'
    publish(tmp_path, generation(boot='boot-fixture-002'))
    assert reader.read('GBPUSD', now=NOW).state.value == 'INVALID'
    assert reader.read('GBPUSD', now=NOW).state.value == 'INVALID'
    publish(tmp_path, generation(boot='boot-fixture-002', sequence=2, started=STAMP+1))
    assert reader.read('GBPUSD', now=NOW+timedelta(seconds=1)).state.value == 'VALID'
    publish(tmp_path, generation(sequence=99, started=STAMP+2))
    assert reader.read('GBPUSD', now=NOW+timedelta(seconds=2)).state.value == 'INVALID'


def test_query_time_rollback_rejected_even_with_increasing_sequence(tmp_path):
    reader = bridge.BridgeReader(tmp_path, IDENTITY)
    publish(tmp_path, generation())
    reader.read('GBPUSD', now=NOW)
    publish(tmp_path, generation(sequence=2, started=STAMP-1))
    assert reader.read('GBPUSD', now=NOW).state.value == 'INVALID'


@pytest.mark.parametrize('fault', ['missing', 'short', 'digest', 'manifest', 'payload', 'partial',
                                  'manifest_duplicate', 'payload_duplicate', 'traversal',
                                  'oversized_manifest', 'oversized_payload'])
def test_publication_faults_never_validate(tmp_path, fault):
    path, manifest = publish(tmp_path, generation())
    if fault == 'missing':
        path.unlink()
    elif fault == 'short':
        manifest['payload_bytes'] += 1
    elif fault == 'digest':
        manifest['payload_sha256'] = '0'*64
    elif fault == 'manifest':
        (tmp_path/'manifest.json').write_text('{')
    elif fault == 'payload':
        publish(tmp_path, generation(), raw=b'{invalid}')
    elif fault == 'partial':
        path.write_bytes(path.read_bytes()[:-5])
    elif fault == 'manifest_duplicate':
        (tmp_path/'manifest.json').write_text('{"schema_version":1,'+json.dumps(manifest)[1:])
    elif fault == 'payload_duplicate':
        publish(tmp_path, generation(), raw=b'{"schema_version":1,'+path.read_bytes()[1:])
    elif fault == 'traversal':
        manifest['payload_filename'] = '../another.json'
    elif fault == 'oversized_manifest':
        (tmp_path/'manifest.json').write_bytes(b' '*(bridge.MAX_MANIFEST_BYTES+1))
    else:
        manifest['payload_bytes'] = bridge.MAX_PAYLOAD_BYTES+1
    if fault in ('short', 'digest', 'traversal', 'oversized_payload'):
        (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    assert bridge.BridgeReader(tmp_path, IDENTITY).read('GBPUSD', now=NOW).state.value == 'INVALID'


def test_unavailable_manifest(tmp_path):
    assert bridge.BridgeReader(tmp_path, IDENTITY).read('GBPUSD', now=NOW).state.value == 'UNAVAILABLE'


def test_manifest_changes_during_read(tmp_path, monkeypatch):
    publish(tmp_path, generation())
    reader = bridge.BridgeReader(tmp_path, IDENTITY)
    original = reader._read
    calls = []
    def changed(name, cap):
        data = original(name, cap)
        if name == 'manifest.json':
            calls.append(1)
            if len(calls) == 2:
                return data+b' '
        return data
    monkeypatch.setattr(reader, '_read', changed)
    assert reader.read('GBPUSD', now=NOW).state.value == 'INVALID'


@pytest.mark.parametrize('importance', ['NONE', 'UNKNOWN'])
def test_unset_importance_is_unresolved(tmp_path, importance):
    result = read(tmp_path, generation(event(importance=importance)))
    assert result.state.value == 'VALID' and result.shadow_state == 'UNKNOWN'
    assert result.events[0].importance == importance


@pytest.mark.parametrize('mode', ['DATE', 'NOTIME', 'TENTATIVE', 'UNKNOWN'])
def test_uncertain_timing_preserved_without_invented_utc(tmp_path, mode):
    result = read(tmp_path, generation(event(mode=mode)))
    assert result.state.value == 'VALID' and result.shadow_state == 'UNKNOWN'
    assert result.events[0].time_mode == mode and result.events[0].utc_time is None


@pytest.mark.parametrize('field,value', [
    ('event_id', '0'), ('value_id', 'bad'), ('country_id', None),
    ('country_code', ''), ('currency', 'ABC'), ('importance', 'SURPRISE'),
    ('time_mode', 'UNKNOWN_MODE'), ('name', ''), ('utc_time', STAMP+OFFSET),
    ('server_time', 0),
])
def test_malformed_event_is_not_dropped(tmp_path, field, value):
    row = event()
    row[field] = value
    assert read(tmp_path, generation(row)).state.value == 'INVALID'


def test_uncertain_event_with_precise_utc_is_invalid(tmp_path):
    row = event(mode='TENTATIVE')
    row['utc_time'] = STAMP
    assert read(tmp_path, generation(row)).state.value == 'INVALID'


@pytest.mark.parametrize('offset', [7200, 10800, 19800, -18000])
def test_exact_once_offset_conversion(tmp_path, offset):
    result = read(tmp_path, generation(event(offset=offset), offset=offset))
    assert result.state.value == 'VALID' and result.events[0].utc_time == STAMP


@pytest.mark.parametrize('at', [datetime(2026, 9, 3, tzinfo=timezone.utc),
                              datetime(2026, 9, 7, tzinfo=timezone.utc)])
def test_midnight_and_week_rollover_coverage(tmp_path, at):
    stamp = int(at.timestamp())
    result = read(tmp_path, generation(event(at=stamp), started=stamp), now=at)
    assert result.state.value == 'VALID' and result.shadow_state == 'BLACKOUT'


def test_event_bracketing_does_not_replace_coverage(tmp_path):
    payload = generation()
    payload['query']['server_end'] = STAMP+OFFSET+299
    payload['query']['utc_end'] = STAMP+299
    payload['coverage']['utc_end'] = STAMP+299
    assert read(tmp_path, payload).state.value == 'INVALID'


@pytest.mark.parametrize('symbol', ['CHFJPY', 'ABCUSD', 'EURUSD.a', 'USDUSD', 'XAUJPY', ''])
def test_unsupported_requests_rejected(tmp_path, symbol):
    assert read(tmp_path, generation(), symbol).shadow_state == 'UNKNOWN'


def test_xauusd_uses_only_usd(tmp_path):
    result = read(tmp_path, generation(event(currency='EUR')), 'XAUUSD')
    assert result.shadow_state == 'CLEAR'
    assert read(tmp_path, generation(event()), 'XAUUSD').shadow_state == 'BLACKOUT'


def test_clock_uncertainty_near_boundary_is_unknown(tmp_path):
    payload = generation(event(at=STAMP+300))
    payload['clock']['clock_uncertainty_seconds'] = 2
    assert read(tmp_path, payload).shadow_state == 'UNKNOWN'


def test_observer_is_bounded_and_does_not_refresh_existing_news(tmp_path, caplog, monkeypatch):
    from core import news_calendar as news
    monkeypatch.setattr(news, 'evaluate_news', lambda *_a, **_k: pytest.fail('observer refreshed news'))
    publish(tmp_path, generation(event()))
    reader = bridge.BridgeReader(tmp_path, IDENTITY)
    reporter = bridge.ShadowReporter()
    with caplog.at_level(logging.INFO):
        for tick in (0, 1, 59, 60):
            reporter.compare(reader, 'GBPUSD', 'UNKNOWN', logging.getLogger('bridge-test'),
                             now=NOW, monotonic=tick)
    assert caplog.text.count('CALENDAR BRIDGE SHADOW ONLY') == 2
    assert 'existing_task018=UNKNOWN' in caplog.text and 'set_hash=' in caplog.text


def test_valid_bridge_cannot_enable_real_production_clear_or_order(tmp_path, monkeypatch):
    from core import news_calendar as news
    from test_task018b_news_failclosed import payload, risk_environment, run_risk
    import test_task015_actual_risk_sizing as sizing
    observed = read(tmp_path, generation())
    assert observed.shadow_state == 'CLEAR' and not observed.entries_allowed
    monkeypatch.setattr(news, '_utc_now', lambda: NOW)
    monkeypatch.setattr(news, '_fetch_feed', lambda: payload())
    assert news.evaluate_news('EURUSD').status is news.NewsStatus.UNKNOWN
    risk_environment(monkeypatch)
    assert run_risk()['decision'] == 'REJECTED'
    result, sent = sizing._place(monkeypatch)
    assert not result['success'] and not sent


def test_bridge_evidence_is_not_a_news_permission_result(tmp_path, monkeypatch):
    from core import news_calendar as news
    import test_task015_actual_risk_sizing as sizing
    result = read(tmp_path, generation())
    monkeypatch.setattr(news, 'evaluate_news', lambda *_a: result)
    placed, sent = sizing._place(monkeypatch)
    assert not placed['success'] and not sent


def test_no_production_dependency_on_bridge():
    root = Path(__file__).resolve().parents[1]
    for directory in ('core', 'src/agents', 'strategies'):
        for path in (root/directory).glob('*.py'):
            if path.name == 'calendar_bridge.py':
                continue
            assert 'calendar_bridge' not in path.read_text(encoding='utf-8')
    for name in ('core/calendar_bridge.py', 'src/calendar_bridge_shadow.py'):
        tree = ast.parse((root/name).read_text())
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        text = ' '.join(ast.unparse(n) for n in imports)
        assert not any(forbidden in text for forbidden in ('MetaTrader5', 'news_calendar', 'urllib', 'requests', 'socket'))


def test_mql_service_has_no_execution_api_and_uses_parent_event_id():
    source = (Path(__file__).resolve().parents[1]/'mql5/CalendarBridgeShadow.mq5').read_text()
    assert '#property service' in source
    forbidden = r'\b(OrderSend|OrderSendAsync|CTrade|PositionClose|PositionModify|MqlTradeRequest)\b'
    assert re.search(forbidden, source, re.I) is None
    assert 'CalendarEventById(values[i].event_id,event)' in source
    assert 'count>=0 && error==0' in source and 'error==5400' in source
    assert re.search(r'if\s*\(\s*CalendarValueHistory\(', source) is None


def test_production_operational_writes_still_blocked():
    from core.runtime_paths import PRODUCTION_DATA_DIR
    with pytest.raises(RuntimeError, match='production'):
        (PRODUCTION_DATA_DIR/'state/task018f-forbidden').write_text('forbidden')


def test_schema_does_not_accept_forged_complete_flag(tmp_path):
    payload = generation()
    payload['complete'] = True
    assert read(tmp_path, payload).state.value == 'INVALID'
