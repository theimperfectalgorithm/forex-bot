"""
Session window predicates -- single source of truth for the UTC trading
session boundaries that were previously duplicated/scattered across
agent_strategy.py (LONDON_START/END, NY_START/END, EURUSD_SESSION_START/END,
ASIAN_END_HOUR) and main_agent.py (T_LONDON_*, T_NY_*, T_FRIDAY_CLOSE).

These functions reproduce the exact hour boundaries used by the original
agent_strategy.py session checks inside check_breakout() and
check_eurusd_signals() -- no values were changed.

Note: main_agent.py's own T_LONDON_END (12:30) / T_NY_END (20:45) scheduling
constants are intentionally NOT merged into this module. Those control when
the orchestrator polls for breakouts (an operational cadence/safety-margin
choice), which is a different concern from "is this UTC time inside the
London/NY/Asian session" used by the strategy logic itself. Merging them
would change orchestrator polling behavior, which is out of scope for a
no-logic-change refactor.

EOD close history: this module previously exposed is_eod_close_time(), a
daily 17:30 UTC forced-close check. That was replaced by a Friday-only
close (is_friday_close_time() below) -- positions now run to their natural
SL/TP Monday-Thursday, and are only force-closed ahead of the weekend.
"""

from __future__ import annotations

from datetime import datetime

# -- agent_strategy.py session window constants (preserved verbatim)
LONDON_START_HOUR = 8
LONDON_END_HOUR   = 13
NY_START_HOUR     = 13
NY_END_HOUR       = 22
ASIAN_END_HOUR    = 7     # Asian session = 00:00 up to (not including) 07:00 UTC
OVERLAP_START_HOUR = 12   # EURUSD_SESSION_START
OVERLAP_END_HOUR   = 16   # EURUSD_SESSION_END (window is effectively 12:00-15:45)


def london_session(utc_time: datetime) -> bool:
    """True if utc_time falls inside the London session window (08:00-13:00 UTC)."""
    return LONDON_START_HOUR <= utc_time.hour < LONDON_END_HOUR


def ny_session(utc_time: datetime) -> bool:
    """True if utc_time falls inside the NY session window (13:00-22:00 UTC)."""
    return NY_START_HOUR <= utc_time.hour < NY_END_HOUR


def asian_session(utc_time: datetime) -> bool:
    """True if utc_time falls inside the Asian session window (00:00-07:00 UTC)."""
    return utc_time.hour < ASIAN_END_HOUR


def london_ny_overlap(utc_time: datetime) -> bool:
    """True if utc_time falls inside the London/NY overlap window used by
    EURUSD's dual strategy (12:00-16:00 UTC, i.e. 12:00-15:45 in practice
    since the orchestrator only checks on 15-minute boundaries)."""
    return OVERLAP_START_HOUR <= utc_time.hour < OVERLAP_END_HOUR


FRIDAY_CLOSE_HOUR   = 20   # 20:00 UTC
FRIDAY_CLOSE_MINUTE = 0


def is_friday(utc_time: datetime) -> bool:
    """True if utc_time falls on a Friday (UTC calendar day)."""
    return utc_time.weekday() == 4


def is_friday_close_time(utc_time: datetime) -> bool:
    """
    True only if utc_time is a Friday AND at or past 20:00 UTC.

    Uses >= semantics (fires and stays true for the rest of Friday) rather
    than an exact-minute match, matching the same robustness pattern the
    old EOD check used: the orchestrator only polls on 15-minute
    boundaries, so a >= comparison means a delayed or skipped poll cycle
    still catches the trigger. The orchestrator's own friday_close_done
    flag (reset fresh each day) ensures this only actually fires once.
    """
    if not is_friday(utc_time):
        return False
    now_minutes   = utc_time.hour * 60 + utc_time.minute
    close_minutes = FRIDAY_CLOSE_HOUR * 60 + FRIDAY_CLOSE_MINUTE
    return now_minutes >= close_minutes
