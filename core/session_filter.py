"""
Session window predicates -- single source of truth for the UTC trading
session boundaries that were previously duplicated/scattered across
agent_strategy.py (LONDON_START/END, NY_START/END, EURUSD_SESSION_START/END,
ASIAN_END_HOUR) and main_agent.py (T_LONDON_*, T_NY_*, T_EOD_CLOSE).

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


def is_eod_close_time(utc_time: datetime, eod_time_str: str) -> bool:
    """
    True once utc_time is at or past the EOD close time of day given as
    "HH:MM" (e.g. "17:30"). Matches the >= semantics main_agent.py uses for
    T_EOD_CLOSE (fires once and stays true for the rest of the day, gated
    by the orchestrator's own eod_close_done flag).
    """
    hh, mm = eod_time_str.split(':')
    eod_minutes = int(hh) * 60 + int(mm)
    now_minutes = utc_time.hour * 60 + utc_time.minute
    return now_minutes >= eod_minutes
