"""
Agent 2 -- Strategy
===================
Called twice per day by the orchestrator:
  07:45 UTC  --  prepare_session('london')
  12:45 UTC  --  prepare_session('ny')

Then called every 15 minutes during the session window:
  check_breakout(pair, session_data, session)

Strategy logic:
  1. H4 trend filter: Close > SMA50 > SMA200 = BULLISH (+1)
                      Close < SMA50 < SMA200 = BEARISH (-1)
                      Otherwise             = NEUTRAL  (0, no trade)

  2. Asian session range: High and Low of M15 bars from 00:00-06:45 UTC

  3. Breakout confirmation (STRICTER than backtest):
     -- Bar CLOSE must be beyond the Asian range (not just a wick)
     -- Direction must match H4 trend
     -- Overshoot must be <= 20 pips past the range level
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np

# -- logging
LOGS_DIR = Path(__file__).parent.parent.parent / 'data' / 'logs'

def _log() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger('STRATEGY')
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


# -- pair configuration
PAIRS = {
    'GBPJPY': {'pip_size': 0.01},
    'EURJPY': {'pip_size': 0.01},
    'EURUSD': {'pip_size': 0.0001},
}

# -- strategy parameters
H4_SMA_FAST          = 50
H4_SMA_SLOW          = 200
H4_BARS_NEEDED       = 220    # enough history for SMA200
ASIAN_END_HOUR       = 7      # 00:00-06:45 UTC bars used for Asian range
MIN_ASIAN_RANGE_PIPS = 10     # skip days with tiny ranges
MAX_OVERSHOOT_PIPS   = 20     # cancel if breakout bar closed > 20p past level

# session window hours (UTC)
LONDON_START = 8
LONDON_END   = 13
NY_START     = 13
NY_END       = 22


# ---------------------------------------------------------------------------
# MT5 helpers
# ---------------------------------------------------------------------------

def _connect(log: logging.Logger) -> bool:
    if mt5.initialize():
        return True
    log.error(f"MT5 init failed: {mt5.last_error()}")
    return False


def _h4_bars(symbol: str, count: int = H4_BARS_NEEDED):
    """Fetch H4 bars as a list of dicts. Returns None on failure."""
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, count)
    if rates is None or len(rates) < H4_SMA_SLOW:
        return None
    return rates


def _m15_bars_today(symbol: str):
    """Fetch all M15 bars from UTC midnight until now."""
    now       = datetime.now(timezone.utc)
    midnight  = now.replace(hour=0, minute=0, second=0, microsecond=0)
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, midnight, now)
    if rates is None or len(rates) == 0:
        return None
    return rates


def _last_closed_m15(symbol: str):
    """Return the last CLOSED M15 bar (position 1 from current)."""
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 1, 1)
    if rates is None or len(rates) == 0:
        return None
    return rates[0]


# ---------------------------------------------------------------------------
# H4 trend calculation
# ---------------------------------------------------------------------------

def _h4_trend(symbol: str, log: logging.Logger) -> int:
    """
    Compute H4 trend for symbol.
    Returns +1 (BULLISH), -1 (BEARISH), or 0 (NEUTRAL).
    """
    bars = _h4_bars(symbol)
    if bars is None:
        log.warning(f"{symbol}: not enough H4 history ({H4_SMA_SLOW} bars needed)")
        return 0

    closes = np.array([b['close'] for b in bars])
    sma50  = np.mean(closes[-H4_SMA_FAST:])
    sma200 = np.mean(closes[-H4_SMA_SLOW:])
    last   = closes[-1]

    if last > sma50 > sma200:
        return 1    # BULLISH: price > fast SMA > slow SMA
    if last < sma50 < sma200:
        return -1   # BEARISH: price < fast SMA < slow SMA
    return 0        # NEUTRAL: mixed / transitioning


# ---------------------------------------------------------------------------
# Asian range calculation
# ---------------------------------------------------------------------------

def _asian_range(symbol: str, log: logging.Logger) -> dict | None:
    """
    Compute Asian session High/Low from today's M15 bars, 00:00-06:45 UTC.
    Returns None if not enough data or range is below minimum.
    """
    bars = _m15_bars_today(symbol)
    if bars is None:
        log.warning(f"{symbol}: no M15 data for today")
        return None

    pip_size  = PAIRS[symbol]['pip_size']
    asian_bars = [b for b in bars if
                  datetime.fromtimestamp(b['time'], tz=timezone.utc).hour < ASIAN_END_HOUR]

    if len(asian_bars) < 4:
        log.warning(f"{symbol}: insufficient Asian session bars ({len(asian_bars)})")
        return None

    high = max(b['high'] for b in asian_bars)
    low  = min(b['low']  for b in asian_bars)
    range_pips = (high - low) / pip_size

    if range_pips < MIN_ASIAN_RANGE_PIPS:
        log.info(f"{symbol}: Asian range too tight ({range_pips:.1f}p < {MIN_ASIAN_RANGE_PIPS}p)")
        return None

    sl_pips = range_pips * 0.50   # SL = 50% of range
    tp_pips = range_pips * 1.00   # TP = 100% of range (2:1 RR)

    return {
        'asian_high' : high,
        'asian_low'  : low,
        'range_pips' : round(range_pips, 1),
        'sl_pips'    : round(sl_pips, 1),
        'tp_pips'    : round(tp_pips, 1),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_session(session: str) -> dict:
    """
    Called at 07:45 (london) or 12:45 (ny) UTC.

    For each pair, computes:
      - H4 trend direction (+1/-1/0)
      - Asian session range (high, low, pips, sl_pips, tp_pips)

    Returns:
        {
          'GBPJPY': {h4_trend, asian_high, asian_low, range_pips, sl_pips, tp_pips},
          'EURJPY': {...},
          'EURUSD': {...},
        }
        -- pair is OMITTED from the dict if data is unavailable or trend is neutral.
    """
    log = _log()
    log.info(f"Agent 2 -- prepare_session({session.upper()})")

    if not _connect(log):
        return {}

    result = {}
    for symbol in PAIRS:
        trend = _h4_trend(symbol, log)
        if trend == 0:
            log.info(f"  {symbol}: NEUTRAL H4 trend -- skipping")
            continue

        asian = _asian_range(symbol, log)
        if asian is None:
            log.info(f"  {symbol}: Asian range unavailable -- skipping")
            continue

        result[symbol] = {
            'h4_trend'  : trend,
            **asian,
        }
        trend_label = 'BULLISH' if trend == 1 else 'BEARISH'
        log.info(f"  {symbol}: {trend_label}  range={asian['range_pips']:.1f}p  "
                 f"H={asian['asian_high']:.5f}  L={asian['asian_low']:.5f}  "
                 f"SL={asian['sl_pips']:.1f}p  TP={asian['tp_pips']:.1f}p")

    return result


def check_breakout(symbol: str, session_data: dict, session: str) -> dict:
    """
    Called every 15 minutes during a session window.

    Checks whether the LAST CLOSED M15 bar has its CLOSE price beyond the
    Asian range in the H4 trend direction (stricter than backtest: close
    required, not just a wick).

    Returns:
        {
          'signal'             : 'BUY' | 'SELL' | 'NO_SIGNAL',
          'reason'             : str,
          'trigger_bar_close'  : float,
          'trigger_bar_time'   : str (ISO),
          'entry_price'        : float,  -- Asian High (BUY) or Low (SELL)
          'overshoot_pips'     : float,
        }
    """
    log = _log()

    no_signal = lambda reason: {
        'signal'            : 'NO_SIGNAL',
        'reason'            : reason,
        'trigger_bar_close' : 0.0,
        'trigger_bar_time'  : '',
        'entry_price'       : 0.0,
        'overshoot_pips'    : 0.0,
    }

    if symbol not in session_data:
        return no_signal('pair not in session data (neutral trend or bad range)')

    data       = session_data[symbol]
    asian_high = data['asian_high']
    asian_low  = data['asian_low']
    trend      = data['h4_trend']
    pip_size   = PAIRS[symbol]['pip_size']

    if not _connect(log):
        return no_signal('MT5 connection failed')

    bar = _last_closed_m15(symbol)
    if bar is None:
        return no_signal('no M15 bar data')

    bar_time  = datetime.fromtimestamp(bar['time'], tz=timezone.utc)
    bar_close = bar['close']
    bar_h     = bar_time.hour

    # Verify bar is inside the expected session window
    if session == 'london' and not (LONDON_START <= bar_h < LONDON_END):
        return no_signal(f'bar at {bar_h:02d}:00 UTC outside London window')
    if session == 'ny' and not (NY_START <= bar_h < NY_END):
        return no_signal(f'bar at {bar_h:02d}:00 UTC outside NY window')

    # BUY: trend BULLISH, bar CLOSE above Asian High
    if trend == 1 and bar_close > asian_high:
        overshoot = (bar_close - asian_high) / pip_size
        if overshoot > MAX_OVERSHOOT_PIPS:
            return no_signal(f'BUY overshoot {overshoot:.0f}p exceeds {MAX_OVERSHOOT_PIPS}p limit')
        return {
            'signal'            : 'BUY',
            'reason'            : 'close above Asian High, H4 BULLISH',
            'trigger_bar_close' : bar_close,
            'trigger_bar_time'  : bar_time.isoformat(),
            'entry_price'       : asian_high,   # anchor SL/TP to range level
            'overshoot_pips'    : round(overshoot, 1),
        }

    # SELL: trend BEARISH, bar CLOSE below Asian Low
    if trend == -1 and bar_close < asian_low:
        overshoot = (asian_low - bar_close) / pip_size
        if overshoot > MAX_OVERSHOOT_PIPS:
            return no_signal(f'SELL overshoot {overshoot:.0f}p exceeds {MAX_OVERSHOOT_PIPS}p limit')
        return {
            'signal'            : 'SELL',
            'reason'            : 'close below Asian Low, H4 BEARISH',
            'trigger_bar_close' : bar_close,
            'trigger_bar_time'  : bar_time.isoformat(),
            'entry_price'       : asian_low,
            'overshoot_pips'    : round(overshoot, 1),
        }

    # No confirmed breakout on this bar
    trend_label = 'BULLISH' if trend == 1 else 'BEARISH'
    return no_signal(f'no {trend_label} close beyond range on bar {bar_time.strftime("%H:%M")}')
