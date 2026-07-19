"""
news_calendar -- high-impact news blackout gate (5ers rule: no trade
entries or bot-initiated exits within +/-5 minutes of high-impact news).

Data source: Forex Factory's free weekly calendar JSON
(https://nfs.faireconomy.media/ff_calendar_thisweek.json), cached to
data/news_calendar.json and refreshed every REFRESH_HOURS. On fetch
failure the stale cache is used (with a warning); with no cache at all
the gate FAILS OPEN (allows trading, logs loudly) -- during the demo
phase a missed gate is observable and cheap, while a fail-closed gate
would silently stop all trading on any network hiccup. REVISIT BEFORE
THE CHALLENGE: on a funded account this should fail CLOSED.

Enabled by `news_filter: true` in config/global_config.yaml.

Usage:
    from core.news_calendar import is_blackout
    blocked, reason = is_blackout('GBPUSD')          # now
    blocked, reason = is_blackout('XAUUSD', when=dt) # specific time
"""

from __future__ import annotations

import json
import logging
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

REPO_ROOT   = Path(__file__).parent.parent
CACHE_FILE  = REPO_ROOT / 'data' / 'news_calendar.json'
CONFIG_FILE = REPO_ROOT / 'config' / 'global_config.yaml'

FEED_URL      = 'https://nfs.faireconomy.media/ff_calendar_thisweek.json'
REFRESH_HOURS = 6
WINDOW_MIN    = 5          # +/- minutes around a high-impact event

log = logging.getLogger('NEWS')


def _cfg(key: str, default):
    """Read one key from global_config's `global:` block, overridden by
    the gitignored per-instance config/local_config.yaml (same schema)."""
    val = default
    try:
        with open(CONFIG_FILE, encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        val = cfg.get('global', {}).get(key, val)
    except Exception as e:
        log.warning(f"news_calendar: could not read global_config ({e})")
    try:
        with open(CONFIG_FILE.parent / 'local_config.yaml',
                  encoding='utf-8') as f:
            loc = (yaml.safe_load(f) or {}).get('global', {})
        if key in loc:
            val = loc[key]
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return val


def _news_filter_enabled() -> bool:
    return bool(_cfg('news_filter', False))


def _fail_closed() -> bool:
    """Challenge/funded instances set news_fail_closed: true in their
    local_config: a missing calendar then BLOCKS entries instead of
    allowing them (the 5ers deducts profits made near news; a silent
    feed outage must not create violations). Demo stays fail-open."""
    return bool(_cfg('news_fail_closed', False))


def _load_cache() -> dict | None:
    try:
        with open(CACHE_FILE, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _fetch_feed() -> list | None:
    try:
        req = urllib.request.Request(FEED_URL,
                                     headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode('utf-8'))
    except Exception as e:
        log.warning(f"news_calendar: feed fetch failed ({e})")
        return None


def _get_events_ex() -> tuple:
    """(events, available). available=False when there is neither a live
    feed nor a usable cache (usable = fetched within MAX_CACHE_AGE_H --
    an old cache crossing into a new week would miss this week's events).
    Each event: {'currency': 'USD', 'time_utc': iso-string}."""
    MAX_CACHE_AGE_H = 72
    cache = _load_cache()
    now = datetime.now(timezone.utc)

    fresh = (cache is not None
             and (now - datetime.fromisoformat(cache['fetched_at']))
             < timedelta(hours=REFRESH_HOURS))
    if not fresh:
        raw = _fetch_feed()
        if raw is not None:
            events = []
            for ev in raw:
                if str(ev.get('impact', '')).lower() != 'high':
                    continue
                try:
                    t = datetime.fromisoformat(ev['date'])
                    events.append({
                        'currency': str(ev.get('country', '')).upper(),
                        'title'   : ev.get('title', ''),
                        'time_utc': t.astimezone(timezone.utc).isoformat(),
                    })
                except Exception:
                    continue
            cache = {'fetched_at': now.isoformat(), 'events': events}
            try:
                CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                    json.dump(cache, f, indent=2)
            except Exception as e:
                log.warning(f"news_calendar: cache write failed ({e})")
        elif cache is not None:
            log.warning("news_calendar: using STALE cache after fetch failure")
        else:
            log.error("news_calendar: NO feed and NO cache")
            return [], False

    age_ok = (cache is not None
              and (now - datetime.fromisoformat(cache['fetched_at']))
              < timedelta(hours=MAX_CACHE_AGE_H))
    return (cache.get('events', []) if cache else []), bool(age_ok)


def _get_events() -> list:
    """Back-compat list-only view (used by core.trade_journal)."""
    return _get_events_ex()[0]


def is_blackout(symbol: str, when: datetime | None = None,
                window_min: int | None = None) -> tuple:
    """
    Returns (blocked: bool, reason: str). blocked=True when a high-impact
    event for either of the symbol's currencies falls within
    +/-window_min minutes of `when` (default: now, UTC).
    window_min defaults to config `news_window_min` (5 -- deliberately
    wider than the 5ers' official +/-2min server-time rule, as margin
    for clock skew). With `news_fail_closed: true` (challenge/funded
    instances), an unavailable calendar BLOCKS instead of allowing.
    """
    if not _news_filter_enabled():
        return False, 'news_filter disabled in global_config'

    if window_min is None:
        window_min = int(_cfg('news_window_min', WINDOW_MIN))
    when = when or datetime.now(timezone.utc)
    curs = {symbol[:3].upper(), symbol[3:6].upper()}
    window = timedelta(minutes=window_min)

    events, available = _get_events_ex()
    if not available:
        if _fail_closed():
            return True, ('news calendar unavailable -- FAIL-CLOSED '
                          '(entries blocked until the feed/cache recovers)')
        log.error("news_calendar: calendar unavailable -- FAILING OPEN "
                  "(demo mode; set news_fail_closed: true on funded accounts)")

    for ev in events:
        if ev['currency'] not in curs:
            continue
        t = datetime.fromisoformat(ev['time_utc'])
        if abs(when - t) <= window:
            return True, (f"high-impact {ev['currency']} news "
                          f"'{ev['title']}' at {t.strftime('%H:%M')} UTC "
                          f"(+/-{window_min}min blackout)")
    return False, 'no high-impact news in window'
