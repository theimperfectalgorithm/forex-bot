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


def _news_filter_enabled() -> bool:
    try:
        with open(CONFIG_FILE, encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        return bool(cfg.get('global', {}).get('news_filter', False))
    except Exception as e:
        log.warning(f"news_calendar: could not read global_config ({e}) -- "
                    f"treating news_filter as DISABLED")
        return False


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


def _get_events() -> list:
    """High-impact events for the current week, refreshing the cache as
    needed. Each event: {'currency': 'USD', 'time_utc': iso-string}."""
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
            log.error("news_calendar: NO feed and NO cache -- gate is "
                      "FAILING OPEN (trading allowed without news protection)")
            return []
    return cache.get('events', []) if cache else []


def is_blackout(symbol: str, when: datetime | None = None,
                window_min: int = WINDOW_MIN) -> tuple:
    """
    Returns (blocked: bool, reason: str). blocked=True when a high-impact
    event for either of the symbol's currencies falls within
    +/-window_min minutes of `when` (default: now, UTC).
    """
    if not _news_filter_enabled():
        return False, 'news_filter disabled in global_config'

    when = when or datetime.now(timezone.utc)
    curs = {symbol[:3].upper(), symbol[3:6].upper()}
    window = timedelta(minutes=window_min)

    for ev in _get_events():
        if ev['currency'] not in curs:
            continue
        t = datetime.fromisoformat(ev['time_utc'])
        if abs(when - t) <= window:
            return True, (f"high-impact {ev['currency']} news "
                          f"'{ev['title']}' at {t.strftime('%H:%M')} UTC "
                          f"(+/-{window_min}min blackout)")
    return False, 'no high-impact news in window'
