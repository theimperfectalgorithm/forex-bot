"""
Agent 1 -- Market Intelligence
==============================
Runs at 00:00 UTC each day.

Responsibilities:
  - Connect to MT5 and read current account balance
  - Calculate daily P&L vs starting balance
  - Verify account is above the $90,000 hard floor
  - Fetch the economic calendar for high-impact events
  - Flag any event within 30 minutes of London open (08:00) or NY open (13:00)
  - Return TRADE_DAY or AVOID with reason

Returns a dict -- consumed by the orchestrator.
"""

import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

# -- logging (shared file, separate logger name)
LOGS_DIR = Path(__file__).parent.parent.parent / 'data' / 'logs'

def _log() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger('MARKET')
    if not log.handlers:
        fmt = logging.Formatter('%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(LOGS_DIR / 'trading.log', encoding='utf-8')
        fh.setFormatter(fmt)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        log.addHandler(fh)
        log.addHandler(ch)
        log.setLevel(logging.INFO)
    return log


# -- constants
STARTING_BALANCE   = 100_000.00
HARD_FLOOR         = 90_000.00
MAX_DAILY_LOSS_PCT = 0.05

# currencies relevant to our pairs (GBPJPY, EURJPY, EURUSD)
WATCHED_CURRENCIES = ['USD', 'EUR', 'GBP', 'JPY']

# flag events within this many minutes of a session open
NEWS_WINDOW_MINS = 30

LONDON_OPEN_HOUR = 8    # 08:00 UTC
NY_OPEN_HOUR     = 13   # 13:00 UTC


# ---------------------------------------------------------------------------
# MT5 connection
# ---------------------------------------------------------------------------

def _connect(log: logging.Logger) -> bool:
    if mt5.initialize():
        return True
    log.error(f"MT5 init failed: {mt5.last_error()}")
    return False


# ---------------------------------------------------------------------------
# Account check
# ---------------------------------------------------------------------------

def _get_account_status(log: logging.Logger) -> dict:
    """Return current balance, equity, and daily P&L."""
    info = mt5.account_info()
    if info is None:
        log.error("Could not fetch account info")
        return {}

    balance = info.balance
    equity  = info.equity
    daily_pnl = balance - STARTING_BALANCE   # simplified: vs challenge start

    log.info(f"Balance: ${balance:,.2f}  Equity: ${equity:,.2f}  "
             f"vs start: ${daily_pnl:+,.2f}")

    return {
        'balance'  : balance,
        'equity'   : equity,
        'daily_pnl': daily_pnl,
    }


# ---------------------------------------------------------------------------
# Economic calendar
# ---------------------------------------------------------------------------

def _fetch_calendar_events(today: datetime, log: logging.Logger) -> list:
    """
    Fetch high-impact events for today using the MT5 calendar API.
    Returns a list of dicts with keys: time, currency, name, importance.

    MT5 calendar availability varies by broker. Failures are caught and
    logged -- the system defaults to 'no news' rather than crashing.
    """
    date_from = today.replace(hour=0, minute=0, second=0, microsecond=0)
    date_to   = date_from + timedelta(days=1)
    events    = []

    for currency in WATCHED_CURRENCIES:
        try:
            values = mt5.calendar_values_by_currency(currency, date_from, date_to)
            if values is None:
                continue

            for val in values:
                # Look up the parent event for importance rating
                event_info = mt5.calendar_event_by_id(val.id)
                if event_info is None:
                    continue

                # calendar_event_by_id may return a tuple or a single object
                if isinstance(event_info, (list, tuple)):
                    event_info = event_info[0]

                # importance: 1=low, 2=medium, 3=high
                if getattr(event_info, 'importance', 0) < 3:
                    continue

                event_time = datetime.fromtimestamp(val.time, tz=timezone.utc)
                events.append({
                    'time'      : event_time,
                    'currency'  : currency,
                    'name'      : getattr(event_info, 'name', 'Unknown'),
                    'importance': getattr(event_info, 'importance', 3),
                })
                log.info(f"  High-impact event: {currency} '{event_info.name}' "
                         f"at {event_time.strftime('%H:%M')} UTC")

        except Exception as e:
            log.warning(f"Calendar fetch failed for {currency}: {e}")

    return events


def _check_news_flags(events: list, today: datetime, log: logging.Logger) -> tuple:
    """
    Return (london_flag, ny_flag) -- True if a high-impact event falls
    within NEWS_WINDOW_MINS of either session open.
    """
    window = timedelta(minutes=NEWS_WINDOW_MINS)
    london_open = today.replace(hour=LONDON_OPEN_HOUR, minute=0, second=0)
    ny_open     = today.replace(hour=NY_OPEN_HOUR,     minute=0, second=0)

    london_flag = any(
        abs((ev['time'] - london_open).total_seconds()) <= window.total_seconds()
        for ev in events
    )
    ny_flag = any(
        abs((ev['time'] - ny_open).total_seconds()) <= window.total_seconds()
        for ev in events
    )

    if london_flag:
        log.warning("NEWS FLAG active: high-impact event within 30 min of London open (08:00)")
    if ny_flag:
        log.warning("NEWS FLAG active: high-impact event within 30 min of NY open (13:00)")

    return london_flag, ny_flag


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run() -> dict:
    """
    Main entry point called by the orchestrator at 00:00 UTC.

    Returns:
        {
          'status'           : 'TRADE_DAY' | 'AVOID',
          'reason'           : str | None,
          'balance'          : float,
          'daily_pnl'        : float,
          'above_hard_floor' : bool,
          'london_news_flag' : bool,
          'ny_news_flag'     : bool,
          'news_events'      : list,
        }
    """
    log = _log()
    log.info("Agent 1 -- Market Intelligence running")

    if not _connect(log):
        return {
            'status'           : 'AVOID',
            'reason'           : 'MT5 connection failed',
            'balance'          : 0.0,
            'daily_pnl'        : 0.0,
            'above_hard_floor' : False,
            'london_news_flag' : False,
            'ny_news_flag'     : False,
            'news_events'      : [],
        }

    try:
        # -- Account status
        acct = _get_account_status(log)
        balance   = acct.get('balance', 0.0)
        daily_pnl = acct.get('daily_pnl', 0.0)

        # -- Hard floor check
        above_hard_floor = balance > HARD_FLOOR
        if not above_hard_floor:
            log.critical(f"HARD FLOOR BREACHED: balance ${balance:,.2f} <= ${HARD_FLOOR:,.0f}")
            return {
                'status'           : 'AVOID',
                'reason'           : f"Hard floor breached: ${balance:,.2f}",
                'balance'          : balance,
                'daily_pnl'        : daily_pnl,
                'above_hard_floor' : False,
                'london_news_flag' : False,
                'ny_news_flag'     : False,
                'news_events'      : [],
            }

        # -- Economic calendar
        today = datetime.now(timezone.utc)
        events = _fetch_calendar_events(today, log)
        if not events:
            log.info("No high-impact events found today (or calendar unavailable)")

        london_flag, ny_flag = _check_news_flags(events, today, log)

        status = 'TRADE_DAY'
        reason = None

        log.info(f"Market verdict: {status}")
        return {
            'status'           : status,
            'reason'           : reason,
            'balance'          : balance,
            'daily_pnl'        : daily_pnl,
            'above_hard_floor' : True,
            'london_news_flag' : london_flag,
            'ny_news_flag'     : ny_flag,
            'news_events'      : [
                {**ev, 'time': ev['time'].isoformat()} for ev in events
            ],
        }

    except Exception as e:
        log.error(f"Agent 1 unexpected error: {e}", exc_info=True)
        return {
            'status'           : 'AVOID',
            'reason'           : f"Agent error: {e}",
            'balance'          : 0.0,
            'daily_pnl'        : 0.0,
            'above_hard_floor' : True,
            'london_news_flag' : False,
            'ny_news_flag'     : False,
            'news_events'      : [],
        }
    finally:
        # Leave MT5 connected -- other agents will use it
        pass
