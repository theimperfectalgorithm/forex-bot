"""Validated Forex Factory calendar for entry protection.

UNKNOWN blocks entries in fail-closed mode but never blocks exit management.
The weekly export supplies positive event evidence, NOT completeness evidence.
No current provider can authorize CLEAR. See docs/task018d_news_safety.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import logging
import os
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml
from core.runtime_paths import data_dir

REPO_ROOT = Path(__file__).parent.parent
CACHE_FILE = data_dir() / 'news_calendar.json'
CONFIG_FILE = REPO_ROOT / 'config' / 'global_config.yaml'
LOGS_DIR = data_dir() / 'logs'
FEED_URL = 'https://nfs.faireconomy.media/ff_calendar_thisweek.json'
REFRESH_HOURS = 6
WINDOW_MIN = 5
FUTURE_TOLERANCE = timedelta(minutes=5)
CURRENCIES = frozenset({'USD', 'EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'NZD', 'CHF', 'CNY'})
IMPACTS = frozenset({'high', 'medium', 'low', 'holiday', 'non-economic'})
MAX_RESPONSE_BYTES = 1024 * 1024
NETWORK_TIMEOUT_SECONDS = 10
RETRY_SECONDS = 15 * 60
_memory_snapshot = None
_retry_after = 0.0


class StrictLoader(yaml.SafeLoader):
    """Reject ambiguous keys throughout both parsed configuration documents."""

    def construct_mapping(self, node, deep=False):
        self.flatten_mapping(node)
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                raise ValueError(f'duplicate YAML key: {key}')
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


@dataclass(frozen=True)
class CalendarSnapshot:
    fetched_at: datetime
    # Immutable parsed events: (currency, title, aware UTC time).
    events: tuple


def _proves_coverage(snapshot, start, end, currencies):
    """No registered source has an explicit, defensible completeness contract.

    A future trusted adapter must prove the entire inclusive interval, all
    relevant currencies and impacts, freshness, revisions and truncation status.
    JSON/cache flags and event bracketing are never such proof.
    """
    return False


def _setup_logger() -> logging.Logger:
    """
    BUG FIX 2026-07-21: this module previously did a bare
    logging.getLogger('NEWS') with no handlers attached -- unlike every
    other module in this codebase (agent_market.py, agent_risk.py, ...),
    which all wire a FileHandler to data/logs/trading.log via this same
    pattern. With no handler, every warning/error this module logs
    (feed fetch failures, "NO feed and NO cache", stale-cache fallback)
    went nowhere -- not to trading.log, not anywhere an operator would
    see it. This is how a live news-gate fail-closed rejection on the
    5ers account produced zero diagnostic trace: is_blackout() clearly
    ran its fail-closed branch (confirmed by the exact rejection text in
    the risk-reject log line), but the *why* -- the underlying fetch
    failure -- was silently swallowed.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('NEWS')
    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
        fmt.converter = time.gmtime
        fh = logging.FileHandler(LOGS_DIR / 'trading.log', encoding='utf-8')
        fh.setFormatter(fmt)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
        logger.setLevel(logging.INFO)
    return logger


log = _setup_logger()


class NewsStatus(str, Enum):
    CLEAR = 'CLEAR'
    BLACKOUT = 'BLACKOUT'
    UNKNOWN = 'UNKNOWN'


@dataclass(frozen=True)
class NewsResult:
    status: NewsStatus
    reason: str
    source: str = 'forexfactory'
    matching_events: tuple = ()
    fail_closed: bool = True
    filter_enabled: bool = True
    snapshot: CalendarSnapshot | None = None
    window_min: int = WINDOW_MIN

    @property
    def entries_allowed(self) -> bool:
        if type(self.fail_closed) is not bool or type(self.filter_enabled) is not bool:
            raise ValueError('invalid news policy flags')
        if self.status is NewsStatus.CLEAR:
            return True
        if self.status is NewsStatus.BLACKOUT:
            return False
        if self.status is NewsStatus.UNKNOWN:
            return not self.filter_enabled or not self.fail_closed
        raise ValueError('invalid news status')

    @property
    def entry_message(self) -> str:
        if self.status is NewsStatus.UNKNOWN and self.entries_allowed:
            action = 'FILTER DISABLED' if not self.filter_enabled else 'FAIL-OPEN'
        else:
            action = 'ENTRY PERMITTED' if self.entries_allowed else 'ENTRY BLOCKED'
        return f'NEWS {self.status.value} / {action}: {self.reason} (source={self.source})'


def _settings() -> dict:
    """Strict news-only configuration; malformed local overrides never vanish."""
    keys = ('news_filter', 'news_fail_closed', 'news_window_min')
    cfg = {}
    for path in (CONFIG_FILE, CONFIG_FILE.parent / 'local_config.yaml'):
        try:
            with open(path, encoding='utf-8') as f:
                document = yaml.load(f, Loader=StrictLoader)
        except FileNotFoundError:
            if path == CONFIG_FILE:
                raise
            continue
        if not isinstance(document, dict):
            raise ValueError(f'invalid configuration document: {path.name}')
        block = document.get('global', {})
        if not isinstance(block, dict):
            raise ValueError(f'invalid global configuration: {path.name}')
        cfg.update({key: block[key] for key in keys if key in block})
    if any(key not in cfg for key in keys):
        raise ValueError('missing required news configuration')
    for key in ('news_filter', 'news_fail_closed'):
        if type(cfg[key]) is not bool:
            raise ValueError(f'{key} must be a boolean')
    _validate_window(cfg['news_window_min'])
    return cfg


def _validate_window(value):
    if type(value) is not int or value <= 0:
        raise ValueError('news_window_min must be a positive integer')


def _aware(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError('timestamp must be an ISO string')
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError('timestamp must include a timezone')
    return result.astimezone(timezone.utc)


def _currencies(symbol: str) -> frozenset:
    if not isinstance(symbol, str):
        raise ValueError('unresolvable symbol')
    symbol = symbol.upper()
    if symbol == 'XAUUSD':
        return frozenset({'USD'})
    if (len(symbol) != 6 or symbol[:3] == symbol[3:]
            or symbol[:3] not in CURRENCIES or symbol[3:] not in CURRENCIES):
        raise ValueError(f'unsupported symbol: {symbol}')
    return frozenset({symbol[:3], symbol[3:]})


def _validate_raw(raw: list) -> tuple:
    """Require the entire response to classify; never silently drop a record.

    Dates validate records only. Neither their span nor freshness proves that
    the provider included every relevant event.
    """
    if not isinstance(raw, list) or not raw:
        raise ValueError('missing/ambiguous weekly payload')
    events, dates = [], []
    for ev in raw:
        if not isinstance(ev, dict):
            raise ValueError('malformed calendar record')
        impact, currency, title = ev.get('impact'), ev.get('country'), ev.get('title')
        if not isinstance(impact, str) or impact.lower() not in IMPACTS:
            raise ValueError('missing/unknown impact classification')
        if not isinstance(currency, str) or currency.upper() not in CURRENCIES:
            raise ValueError('missing/unsupported event currency')
        if not isinstance(title, str) or not title.strip():
            raise ValueError('missing event title')
        event_time = _aware(ev.get('date'))
        dates.append(event_time)
        if impact.lower() == 'high':
            events.append({'currency': currency.upper(), 'title': title,
                           'time_utc': event_time.isoformat()})
    start, end = min(dates), max(dates)
    if end - start > timedelta(days=7):
        raise ValueError('ambiguous weekly period')
    return events, start, end


def _load_cache():
    with open(CACHE_FILE, 'rb') as f:
        return _read_json(f)


def _read_json(stream):
    raw = stream.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError('calendar response exceeds size limit')
    return json.loads(raw.decode('utf-8'))


def _fetch_feed():
    req = urllib.request.Request(FEED_URL, headers={'User-Agent': 'Mozilla/5.0'})
    # Socket-operation timeout, NOT a DNS/whole-request wall-clock deadline.
    with urllib.request.urlopen(req, timeout=NETWORK_TIMEOUT_SECONDS) as response:
        return _read_json(response)


def _validate_snapshot(cache, now, start, end):
    if not isinstance(cache, dict) or cache.get('schema_version') != 1:
        raise ValueError('malformed/legacy cache without raw validation evidence')
    fetched = _aware(cache.get('fetched_at'))
    age = now - fetched
    if age < -FUTURE_TOLERANCE or age >= timedelta(hours=REFRESH_HOURS):
        raise ValueError('future or stale calendar timestamp')
    events, _, _ = _validate_raw(cache.get('raw_events'))
    return events


def _write_cache(cache):
    """Publish only a fully validated snapshot, preserving the previous file."""
    temporary = None
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         dir=CACHE_FILE.parent, delete=False) as f:
            temporary = Path(f.name)
            json.dump(cache, f, indent=2)
        os.replace(temporary, CACHE_FILE)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _fresh(snapshot, now):
    if not isinstance(snapshot, CalendarSnapshot):
        raise ValueError('no validated in-memory calendar')
    age = now - snapshot.fetched_at
    if age < -FUTURE_TOLERANCE or age >= timedelta(hours=REFRESH_HOURS):
        raise ValueError('future or stale calendar timestamp')


def _decode(cache, now):
    events = _validate_snapshot(cache, now, now, now)
    return CalendarSnapshot(_aware(cache['fetched_at']), tuple(
        (ev['currency'], ev['title'], _aware(ev['time_utc'])) for ev in events))


def _snapshot(now, *, refresh=True):
    global _memory_snapshot, _retry_after
    try:
        _fresh(_memory_snapshot, now)
        return _memory_snapshot, 'memory'
    except ValueError:
        pass
    if not refresh:
        raise ValueError('no fresh validated in-memory calendar; refresh prohibited')
    try:
        snapshot = _decode(_load_cache(), now)
        _memory_snapshot = snapshot
        return snapshot, 'cache'
    except Exception as exc:
        cache_reason = str(exc)
    if time.monotonic() < _retry_after:
        raise ValueError('calendar refresh backoff active; ' + cache_reason)
    # One attempt per cycle across all symbols/risk/execution consumers. Set
    # before AND after the attempt so slow failures do not immediately retry.
    _retry_after = time.monotonic() + RETRY_SECONDS
    try:
        raw = _fetch_feed()
        cache = {'schema_version': 1, 'fetched_at': now.isoformat(), 'raw_events': raw}
        snapshot = _decode(cache, now)
        cache['events'] = _validate_snapshot(cache, now, now, now)
        try:
            _write_cache(cache)
        except Exception as exc:
            log.warning(f'news_calendar: cache write failed ({exc})')
        _memory_snapshot = snapshot
        return snapshot, 'feed'
    except Exception as exc:
        raise ValueError(f'feed unusable: {exc}; cache unusable: {cache_reason}') from exc
    finally:
        _retry_after = time.monotonic() + RETRY_SECONDS


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _classify(snapshot, symbol, clock, at, window_min, source, policy):
    _fresh(snapshot, clock)
    _validate_window(window_min)
    currencies = _currencies(symbol)
    window = timedelta(minutes=window_min)
    matching = tuple({'currency': currency, 'title': title, 'time_utc': stamp.isoformat()}
                     for currency, title, stamp in snapshot.events
                     if currency in currencies and abs(at - stamp) <= window)
    evidence = dict(snapshot=snapshot, window_min=window_min, **policy)
    if matching:
        return NewsResult(NewsStatus.BLACKOUT,
                          f'relevant high-impact event within +/-{window_min}min',
                          source, matching, **evidence)
    if _proves_coverage(snapshot, at - window, at + window, currencies):
        return NewsResult(NewsStatus.CLEAR, 'trusted complete coverage of requested window',
                          source, **evidence)
    return NewsResult(NewsStatus.UNKNOWN,
                      'provider has no proven completeness/coverage contract', source, **evidence)


def evaluate_news(symbol: str, when: datetime | None = None,
                  window_min: int | None = None, *, now: datetime | None = None,
                  refresh: bool = True) -> NewsResult:
    """Entry refresh is bounded; exit callers must use refresh=False (memory only)."""
    try:
        cfg = _settings()
    except Exception as exc:
        return NewsResult(NewsStatus.UNKNOWN, f'invalid news configuration: {exc}',
                          source='config')
    policy = dict(fail_closed=cfg['news_fail_closed'], filter_enabled=cfg['news_filter'])
    if not cfg['news_filter']:
        return NewsResult(NewsStatus.UNKNOWN, 'news_filter explicitly disabled',
                          source='config', **policy)
    try:
        clock = _aware((now if now is not None else _utc_now()).isoformat())
        if when is not None:
            _aware(when.isoformat())
        window_min = cfg['news_window_min'] if window_min is None else window_min
        _validate_window(window_min)
        _currencies(symbol)
        snapshot, source = _snapshot(clock, refresh=refresh)
        clock = _aware((now if now is not None else _utc_now()).isoformat())
        at = _aware(when.isoformat()) if when is not None else clock
        return _classify(snapshot, symbol, clock, at, window_min, source, policy)
    except Exception as exc:
        return NewsResult(NewsStatus.UNKNOWN, str(exc), **policy)


def reevaluate_news(previous: NewsResult, symbol: str) -> NewsResult:
    """Final send gate: fresh UTC + retained immutable evidence; no I/O or fetch.

    Explicit disabled/fail-open policy remains visible. A bare CLEAR without a
    retained validated snapshot cannot authorize submission.
    """
    try:
        if not isinstance(previous, NewsResult):
            raise ValueError('invalid retained news result')
        previous.entries_allowed  # validate policy/status before honoring it
        policy = dict(fail_closed=previous.fail_closed, filter_enabled=previous.filter_enabled)
        if previous.snapshot is None:
            return NewsResult(NewsStatus.UNKNOWN, previous.reason, previous.source, **policy)
        clock = _aware(_utc_now().isoformat())
        try:
            return _classify(previous.snapshot, symbol, clock, clock,
                             previous.window_min, previous.source, policy)
        except Exception as exc:
            return NewsResult(NewsStatus.UNKNOWN, str(exc), previous.source, **policy)
    except Exception as exc:
        return NewsResult(NewsStatus.UNKNOWN, f'invalid retained news evidence: {exc}')


def _get_events_ex() -> tuple:
    """Observational journal/reporting view: in-memory only, never refresh."""
    try:
        snapshot, _ = _snapshot(_utc_now(), refresh=False)
        return [{'currency': currency, 'title': title, 'time_utc': stamp.isoformat()}
                for currency, title, stamp in snapshot.events], True
    except Exception:
        return [], False


def _get_events() -> list:
    return _get_events_ex()[0]


def is_blackout(symbol: str, when: datetime | None = None,
                window_min: int | None = None) -> tuple:
    """Legacy entry-policy wrapper. Exit callers must use evaluate_news."""
    result = evaluate_news(symbol, when, window_min)
    return not result.entries_allowed, result.entry_message
