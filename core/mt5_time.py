"""Small, explicit helpers for MT5 server-clock timestamps.

MT5 deal/tick epoch values on this broker encode the broker server clock
(UTC+2/UTC+3), not real UTC.  Conversion deliberately requires the observed
offset at the time the value was retrieved: callers must not apply today's
offset to historical records.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


VALID_SERVER_UTC_OFFSETS = (2, 3)


def observed_server_utc_offset_hours(mt5_module, symbol: str = 'EURUSD') -> int | None:
    """Return the currently observed server offset, or ``None`` if unavailable."""
    try:
        if not mt5_module.initialize():
            return None
        tick = mt5_module.symbol_info_tick(symbol)
        if tick and tick.time:
            offset = round((tick.time - datetime.now(timezone.utc).timestamp()) / 3600)
            if offset in VALID_SERVER_UTC_OFFSETS:
                return offset
    except Exception:
        pass
    return None


def server_epoch_to_utc(server_epoch: int | float, observed_offset_hours: int) -> datetime:
    """Convert an MT5 server-clock epoch to an aware real-UTC datetime."""
    if observed_offset_hours not in VALID_SERVER_UTC_OFFSETS:
        raise ValueError(f'unsupported MT5 server UTC offset: {observed_offset_hours!r}')
    return (datetime.fromtimestamp(server_epoch, tz=timezone.utc)
            - timedelta(hours=observed_offset_hours))


def mt5_bar_time_to_utc(value: int | float | datetime,
                        observed_offset_hours: int) -> datetime:
    """Normalize one MT5 bar time exactly once.

    Numeric MT5 values are broker/server-clock epochs and require the observed
    offset.  An aware ``datetime`` is an explicit already-normalized value used
    by tests/offline callers and is converted to UTC without another shift.
    Naive datetimes are rejected because their clock basis is ambiguous.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError('naive MT5 bar datetime has ambiguous timezone')
        return value.astimezone(timezone.utc)
    return server_epoch_to_utc(value, observed_offset_hours)
