"""Read-only Task018F bridge evidence. NEVER a production permission provider.

No imports of MT5, news_calendar, strategy code or networking. A caller must
explicitly supply the directory and pinned identity. See docs/task018f_bridge.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import ntpath
import os
from pathlib import Path
import re
import time

SCHEMA_VERSION = 1
SOURCE = 'mql5-calendar-shadow'
SUPPORTED_CURRENCIES = frozenset({'AUD', 'CAD', 'EUR', 'GBP', 'JPY', 'USD'})
KNOWN_CALENDAR_CURRENCIES = SUPPORTED_CURRENCIES | frozenset({
    'CHF', 'CNY', 'NZD', 'HKD', 'SGD', 'ZAR', 'BRL', 'MXN', 'KRW', 'INR',
    'NOK', 'SEK', 'PLN', 'TRY', 'RUB', 'DKK', 'HUF', 'CZK', 'ILS', 'IDR'})
REFRESH_SECONDS = 60
EXPIRY_SECONDS = 90
MAX_QUERY_MS = 15000
MAX_CLOCK_UNCERTAINTY = 2
MAX_QUOTE_AGE = 10
MAX_PAYLOAD_BYTES = 1024 * 1024
MAX_MANIFEST_BYTES = 8192
MAX_EVENTS = 5000
MAX_RETIRED_BOOTS = 16


class EvidenceState(str, Enum):
    VALID = 'VALID'
    INVALID = 'INVALID'
    STALE = 'STALE'
    IDENTITY_MISMATCH = 'IDENTITY_MISMATCH'
    UNAVAILABLE = 'UNAVAILABLE'


@dataclass(frozen=True)
class ExpectedIdentity:
    login: str
    server: str
    terminal_path: str
    terminal_data_path: str
    instance_id: str
    company: str | None = None  # optional until independently observed; never invented


@dataclass(frozen=True)
class BridgeEvent:
    value_id: str
    event_id: str
    country_id: str
    country_code: str
    currency: str
    importance: str
    time_mode: str
    name: str
    server_time: int
    utc_time: int | None


@dataclass(frozen=True)
class BridgeEvidence:
    state: EvidenceState
    reason: str
    shadow_state: str = 'UNKNOWN'
    events: tuple[BridgeEvent, ...] = ()
    matching_value_ids: tuple[str, ...] = ()
    boot_id: str | None = None
    sequence: int | None = None

    @property
    def entries_allowed(self) -> bool:
        """Task018F is observation only, including VALID / shadow CLEAR."""
        return False


class _Rejected(ValueError):
    def __init__(self, reason, state=EvidenceState.INVALID):
        super().__init__(reason)
        self.state = state


def _require(condition, reason):
    if not condition:
        raise _Rejected(reason)


def _keys(value, names):
    _require(type(value) is dict and set(value) == set(names.split()), 'schema keys mismatch')
    return value


def _int(value, low=0, high=2**63-1):
    _require(type(value) is int and low <= value <= high, 'invalid integer')
    return value


def _text(value, limit=512):
    _require(type(value) is str and 0 < len(value) <= limit
             and not any(ord(c) < 32 for c in value), 'invalid text')
    return value


def _id(value):
    _require(type(value) is str and re.fullmatch(r'[1-9][0-9]{0,19}', value) is not None,
             'invalid native identifier')
    _require(int(value) <= 2**64-1, 'identifier overflow')
    return int(value)


def _token(value):
    _require(type(value) is str and re.fullmatch(r'[A-Za-z0-9_-]{8,64}', value) is not None,
             'invalid instance/boot token')
    return value


def _windows_path(value):
    _text(value, 1024)
    _require(ntpath.isabs(value) and not value.startswith(('\\\\', '//')),
             'identity must name a local absolute Windows path')
    return ntpath.normcase(ntpath.normpath(value))


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def constant(_value):
        raise _Rejected('nonfinite JSON constant')
    return json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=constant)


def symbol_currencies(symbol):
    _text(symbol, 6)
    symbol = symbol.upper()
    if symbol == 'XAUUSD':
        return frozenset({'USD'})
    _require(len(symbol) == 6 and symbol[:3] != symbol[3:], 'unsupported symbol')
    result = frozenset({symbol[:3], symbol[3:]})
    _require(result <= SUPPORTED_CURRENCIES, 'unsupported currency request')
    return result


class BridgeReader:
    """Bounded local reads. Sequence history is retained for this reader lifetime.

    Repeated identical generations are allowed until expiry. A new boot is
    quarantined until a second, newer successful generation; retired boots
    cannot return. New processes must establish their own observation history.
    """
    def __init__(self, directory, expected: ExpectedIdentity, *, expiry_seconds=EXPIRY_SECONDS):
        path = str(directory)
        _require(not path.startswith(('\\\\', '//')), 'network directories are forbidden')
        self.directory = Path(directory).resolve()
        _require(not str(self.directory).startswith(('\\\\', '//')), 'network directories are forbidden')
        if os.name == 'nt':
            import ctypes
            drive_type = ctypes.windll.kernel32.GetDriveTypeW(str(self.directory.anchor))
            _require(drive_type in (2, 3, 6), 'bridge requires a local filesystem drive')
        _id(expected.login)
        _text(expected.server)
        _token(expected.instance_id)
        _windows_path(expected.terminal_path)
        _windows_path(expected.terminal_data_path)
        if expected.company is not None:
            _text(expected.company)
        self.expected = expected
        self.expiry_seconds = _int(expiry_seconds, 1, EXPIRY_SECONDS)
        self._boot = None
        self._sequence = 0
        self._digest = None
        self._query_start = 0
        self._pending = None
        self._retired = set()

    def _read(self, filename, limit):
        path = self.directory / filename
        _require(path.resolve().parent == self.directory, 'payload escapes pinned directory')
        with path.open('rb') as handle:
            raw = handle.read(limit + 1)
        _require(len(raw) <= limit, 'file exceeds size limit')
        return raw

    def read(self, symbol, *, now=None, window_seconds=300):
        try:
            currencies = symbol_currencies(symbol)
            window_seconds = _int(window_seconds, 1, 3600)
            now = datetime.now(timezone.utc) if now is None else now
            _require(isinstance(now, datetime) and now.utcoffset() is not None, 'UTC clock required')
            at = now.timestamp()
            try:
                manifest_raw = self._read('manifest.json', MAX_MANIFEST_BYTES)
            except FileNotFoundError:
                raise _Rejected('manifest unavailable', EvidenceState.UNAVAILABLE)
            manifest = _keys(_json(manifest_raw),
                'schema_version instance_id boot_id sequence payload_filename payload_bytes payload_sha256 published_utc')
            _require(type(manifest['schema_version']) is int and manifest['schema_version'] == SCHEMA_VERSION,
                     'unsupported manifest version')
            boot = _token(manifest['boot_id'])
            sequence = _id(manifest['sequence'])
            _token(manifest['instance_id'])
            _require(manifest['payload_filename'] == f'calendar_{boot}_{sequence}.json', 'invalid payload filename')
            size = _int(manifest['payload_bytes'], 1, MAX_PAYLOAD_BYTES)
            digest = manifest['payload_sha256']
            _require(type(digest) is str and re.fullmatch('[0-9a-f]{64}', digest) is not None,
                     'invalid digest encoding')
            raw = self._read(manifest['payload_filename'], MAX_PAYLOAD_BYTES)
            _require(len(raw) == size, 'payload length mismatch')
            _require(hashlib.sha256(raw).hexdigest() == digest, 'payload digest mismatch')
            _require(self._read('manifest.json', MAX_MANIFEST_BYTES) == manifest_raw,
                     'publication changed during read')
            payload = _json(raw)
            events, uncertainty = self._validate(payload, manifest, at, window_seconds, currencies)
            self._advance(boot, sequence, digest, payload['query']['started_utc'])
            matching, unresolved = [], False
            for event in events:
                if event.currency not in currencies:
                    continue
                if event.importance in ('LOW', 'MODERATE'):
                    continue
                if event.time_mode != 'DATETIME':
                    # No precise timestamp or invented midnight release. The
                    # whole queried generation is unresolved for this currency.
                    unresolved = True
                    continue
                distance = abs(at - event.utc_time)
                if distance <= window_seconds + uncertainty:
                    if event.importance == 'HIGH' and distance <= window_seconds - uncertainty:
                        matching.append(event.value_id)
                    else:
                        unresolved = True
            state = 'BLACKOUT' if matching else ('UNKNOWN' if unresolved else 'CLEAR')
            return BridgeEvidence(EvidenceState.VALID, 'validated shadow evidence only', state,
                                  events, tuple(matching), boot, sequence)
        except _Rejected as exc:
            return BridgeEvidence(exc.state, str(exc))
        except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
            return BridgeEvidence(EvidenceState.INVALID, f'bridge read/validation failed: {type(exc).__name__}')

    def _validate(self, p, m, now, window, currencies):
        _keys(p, 'schema_version source instance_id boot_id sequence identity clock query health coverage events')
        _require(type(p['schema_version']) is int and p['schema_version'] == SCHEMA_VERSION
                 and p['source'] == SOURCE, 'unsupported payload source/version')
        for field in ('instance_id', 'boot_id', 'sequence'):
            _require(p[field] == m[field], 'manifest/payload identity disagreement')
        identity = _keys(p['identity'], 'login server company terminal_path terminal_data_path')
        _id(identity['login'])
        for field in ('server', 'company'):
            _text(identity[field])
        expected = self.expected
        if (p['instance_id'] != expected.instance_id or identity['login'] != expected.login
                or identity['server'] != expected.server
                or _windows_path(identity['terminal_path']) != _windows_path(expected.terminal_path)
                or _windows_path(identity['terminal_data_path']) != _windows_path(expected.terminal_data_path)
                or (expected.company is not None and identity['company'] != expected.company)):
            raise _Rejected('pinned bridge identity mismatch', EvidenceState.IDENTITY_MISMATCH)

        q = _keys(p['query'], 'server_start server_end utc_start utc_end started_utc elapsed_ms return_count error_code query_success failure_stage')
        c = _keys(p['clock'], 'generated_server_time generated_utc_time server_utc_offset_seconds offset_sample_time clock_status clock_uncertainty_seconds offset_before_seconds offset_after_seconds quote_age_before_seconds quote_age_after_seconds')
        h = _keys(p['health'], 'terminal_connected event_enrichment_complete country_enrichment_complete currency_catalog_valid change_before change_after change_error_before change_error_after')
        coverage = _keys(p['coverage'], 'utc_start utc_end supported_currencies returned_event_count')
        for field in ('server_start', 'server_end', 'utc_start', 'utc_end', 'started_utc'):
            _int(q[field], 1)
        _int(q['elapsed_ms'], 0, MAX_QUERY_MS)
        _int(q['return_count'], 0, MAX_EVENTS)
        _require(q['query_success'] is True and type(q['error_code']) is int
                 and q['error_code'] == 0 and q['failure_stage'] == '', 'native query failed/incomplete')
        for field in ('terminal_connected', 'event_enrichment_complete',
                      'country_enrichment_complete', 'currency_catalog_valid'):
            _require(h[field] is True, f'health failure: {field}')
        _require(_id(h['change_before']) == _id(h['change_after']), 'calendar changed during generation')
        for field in ('change_error_before', 'change_error_after'):
            _require(type(h[field]) is int and h[field] == 0, 'calendar change probe failed')
        _require(c['clock_status'] == 'VALID', 'invalid clock evidence')
        offset = _int(c['server_utc_offset_seconds'], -43200, 50400)
        _require(offset % 900 == 0, 'unreasonable offset')
        for field in ('offset_before_seconds', 'offset_after_seconds'):
            _require(_int(c[field], -43200, 50400) == offset, 'offset changed during generation')
        uncertainty = _int(c['clock_uncertainty_seconds'], 0, MAX_CLOCK_UNCERTAINTY)
        for field in ('quote_age_before_seconds', 'quote_age_after_seconds'):
            _int(c[field], 0, MAX_QUOTE_AGE)
        generated = _int(c['generated_utc_time'], 1)
        _require(abs(_int(c['generated_server_time'], 1) - offset - generated) <= uncertainty,
                 'generated UTC conversion mismatch')
        _require(q['started_utc'] <= _int(c['offset_sample_time'], 1) <= generated,
                 'offset sample outside generation')
        _require(abs((generated-q['started_utc'])*1000-q['elapsed_ms']) <= 2000,
                 'host clock jump or inconsistent elapsed time')
        _require(q['started_utc'] <= generated <= _int(m['published_utc'], 1) <= now+uncertainty,
                 'future or inconsistent generation time')
        age = now-q['started_utc']
        if age >= self.expiry_seconds:
            raise _Rejected('calendar evidence expired from query start', EvidenceState.STALE)
        _require(age >= -uncertainty, 'query starts in future')
        _require(q['server_start']-offset == q['utc_start']
                 and q['server_end']-offset == q['utc_end'], 'query UTC conversion mismatch')
        _require(0 < q['utc_end']-q['utc_start'] <= 3*86400, 'invalid query interval')
        _require(_int(coverage['utc_start'], 1) == q['utc_start']
                 and _int(coverage['utc_end'], 1) == q['utc_end'], 'coverage exceeds proven query')
        _require(q['utc_start'] < now-window-uncertainty
                 and now+window+uncertainty < q['utc_end'], 'window not strictly inside queried coverage')
        covered = coverage['supported_currencies']
        _require(type(covered) is list and len(covered) == len(SUPPORTED_CURRENCIES)
                 and all(type(x) is str for x in covered)
                 and set(covered) == SUPPORTED_CURRENCIES and currencies <= set(covered),
                 'currency coverage incomplete')
        _require(type(p['events']) is list, 'events must be an array')
        _require(_int(coverage['returned_event_count'], 0, MAX_EVENTS)
                 == q['return_count'] == len(p['events']), 'event count mismatch')
        events, seen = [], set()
        for row in p['events']:
            _keys(row, 'value_id event_id country_id country_code currency importance time_mode name server_time utc_time')
            for field in ('value_id', 'event_id', 'country_id'):
                _id(row[field])
            _require(row['value_id'] not in seen, 'duplicate occurrence identifier')
            seen.add(row['value_id'])
            _require(type(row['country_code']) is str and re.fullmatch('[A-Z]{2}', row['country_code']) is not None,
                     'invalid country code')
            _require(type(row['currency']) is str and row['currency'] in KNOWN_CALENDAR_CURRENCIES,
                     'invalid currency')
            _require(row['importance'] in ('HIGH', 'MODERATE', 'LOW', 'NONE', 'UNKNOWN'), 'invalid importance encoding')
            _require(row['time_mode'] in ('DATETIME', 'DATE', 'NOTIME', 'TENTATIVE', 'UNKNOWN'), 'invalid time mode encoding')
            _text(row['name'])
            server_time = _int(row['server_time'])
            if row['time_mode'] == 'DATETIME':
                _require(server_time > 0 and _int(row['utc_time'], 1) == server_time-offset,
                         'event UTC conversion mismatch')
                _require(q['server_start'] <= server_time <= q['server_end'], 'event outside native query')
            else:
                _require(row['utc_time'] is None, 'uncertain event has manufactured precise UTC')
            events.append(BridgeEvent(**row))
        return tuple(events), uncertainty

    def _advance(self, boot, sequence, digest, started):
        _require(boot not in self._retired, 'retired boot replay')
        _require(started >= self._query_start, 'query-time rollback')
        if self._boot is not None and boot != self._boot:
            _require(len(self._retired) < MAX_RETIRED_BOOTS, 'boot history limit reached')
            if self._pending is None or self._pending[0] != boot:
                self._pending = (boot, sequence, started)
                raise _Rejected('new boot quarantined; require a later generation')
            _require(sequence > self._pending[1] and started > self._pending[2],
                     'new boot requires a later generation')
            self._retired.add(self._boot)
            self._sequence = 0
        _require(sequence >= self._sequence, 'sequence rollback')
        if sequence == self._sequence:
            _require(digest == self._digest, 'same sequence changed payload')
        self._boot, self._sequence, self._digest = boot, sequence, digest
        self._query_start = started
        self._pending = None


class ShadowReporter:
    """Explicit observer; caller supplies existing Task018 state, never refreshes it."""
    def __init__(self, *, minimum_interval=60):
        self.minimum_interval = _int(minimum_interval, 1, 3600)
        self._last = float('-inf')

    def compare(self, reader, symbol, existing_state, logger, *, now=None, monotonic=None):
        _require(existing_state in ('CLEAR', 'BLACKOUT', 'UNKNOWN'), 'invalid supplied Task018 state')
        result = reader.read(symbol, now=now)
        current = time.monotonic() if monotonic is None else monotonic
        if current-self._last >= self.minimum_interval:
            # Full set digest plus bounded sample; no growing log/sequence cache.
            rows = [(e.value_id, e.currency, e.importance, e.time_mode, e.utc_time) for e in result.events]
            digest = hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]
            logger.info('CALENDAR BRIDGE SHADOW ONLY symbol=%s bridge=%s candidate=%s '
                        'existing_task018=%s events=%s set_hash=%s sample=%s reason=%s',
                        symbol, result.state.value, result.shadow_state, existing_state,
                        len(rows), digest, rows[:5], result.reason[:240])
            self._last = current
        return result
