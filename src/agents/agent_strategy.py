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

import time

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
        fmt.converter = time.gmtime
        fh = logging.FileHandler(LOGS_DIR / 'trading.log', encoding='utf-8')
        fh.setFormatter(fmt)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        log.addHandler(fh)
        log.addHandler(ch)
        log.setLevel(logging.INFO)
    return log


# -- breakout pair configuration (GBPJPY and EURJPY only -- EURUSD handled separately)
PAIRS = {
    'GBPJPY': {'pip_size': 0.01},
    'EURJPY': {'pip_size': 0.01},
}

# -- breakout strategy parameters
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

# -- EURUSD dual-strategy configuration
EURUSD_PIP_SIZE      = 0.0001
EURUSD_SESSION_START = 12     # 12:00 UTC
EURUSD_SESSION_END   = 16     # 16:00 UTC (window is 12:00-15:45)

# SMA Run 1 parameters
SMA_FAST         = 50
SMA_MID          = 100
SMA_SLOW         = 200
SMA_SL_PIPS      = 30
SMA_TP_PIPS      = 60
SMA_FLAT_PIPS    = 5          # skip if |SMA50-SMA100| < 5p (markets ranging)
SMA_BARS_NEEDED  = 250        # 250 M15 bars = enough for SMA200 + prev/curr

# EMA Pullback parameters (Test A -- no cross-exit)
EMA_FAST_N       = 5
EMA_MID_N        = 20
EMA_SLOW_N       = 50
EMA_SL_PIPS      = 15
EMA_TP_PIPS      = 30
EMA_TOUCH_PIPS   = 3          # pullback counts if price within 3p of EMA20

# Shared H1 EMA filter for both EURUSD strategies
H1_EMA_N         = 50
H1_BARS_NEEDED   = 60

# Per-strategy daily limits
MAX_SMA_DAILY    = 2
MAX_EMA_DAILY    = 2
MAX_EURUSD_CONSEC = 2


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
# EURUSD helpers
# ---------------------------------------------------------------------------

def _ewm_ema(values: np.ndarray, period: int) -> np.ndarray:
    """EWM EMA matching pandas ewm(span=n, adjust=False)."""
    alpha  = 2.0 / (period + 1)
    result = np.empty(len(values), dtype=float)
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
    return result


def _m15_bars_eurusd(count: int = SMA_BARS_NEEDED):
    """
    Fetch last `count` completed M15 bars for EURUSD.
    Starts at position 1 so the current forming bar is excluded.
    """
    rates = mt5.copy_rates_from_pos('EURUSD', mt5.TIMEFRAME_M15, 1, count)
    if rates is None or len(rates) < SMA_SLOW:
        return None
    return rates


def _h1_ema50_trend(log: logging.Logger) -> int:
    """
    H1 EMA50 trend for EURUSD.
    Returns +1 (bullish), -1 (bearish), 0 on data error.
    """
    rates = mt5.copy_rates_from_pos('EURUSD', mt5.TIMEFRAME_H1, 1, H1_BARS_NEEDED)
    if rates is None or len(rates) < H1_EMA_N:
        log.warning(f"EURUSD: not enough H1 data for EMA{H1_EMA_N}")
        return 0
    closes = np.array([b['close'] for b in rates])
    ema50  = _ewm_ema(closes, H1_EMA_N)
    last   = closes[-1]
    if last > ema50[-1]:
        return 1
    if last < ema50[-1]:
        return -1
    return 0


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


# ---------------------------------------------------------------------------
# EURUSD dual-strategy signal check
# ---------------------------------------------------------------------------

def check_eurusd_signals(eurusd_state: dict, open_trades: list) -> tuple:
    """
    Check EURUSD for SMA Run 1 and EMA Pullback signals.
    Called every 15 minutes during 12:00-15:45 UTC.

    Args:
        eurusd_state : state['eurusd'] dict from orchestrator
        open_trades  : state['open_trades'] list (to detect open EURUSD SMA trade)

    Returns:
        (signals, updated_eurusd_state)

        Each signal is a dict with keys:
          signal   : 'BUY' | 'SELL' | 'CROSS_EXIT'
          strategy : 'SMA' | 'EMA'
          sl_pips  : int
          tp_pips  : int
          reason   : str
          cross_exit_ticket : int  -- only present for CROSS_EXIT signals
    """
    log = _log()
    signals = []
    state   = dict(eurusd_state)

    if not _connect(log):
        return signals, state

    now  = datetime.now(timezone.utc)
    hour = now.hour

    # Window: 12:00 to 15:59 UTC
    if not (EURUSD_SESSION_START <= hour < EURUSD_SESSION_END):
        return signals, state

    # ---- Shared data ----
    bars = _m15_bars_eurusd()
    if bars is None:
        log.warning("EURUSD: insufficient M15 data for signal check")
        return signals, state

    h1_trend = _h1_ema50_trend(log)
    if h1_trend == 0:
        return signals, state

    closes = np.array([b['close'] for b in bars])

    # Pre-compute rolling SMAs via cumsum (matches backtest rolling mean)
    def _rolling(arr, n):
        cs = np.cumsum(arr, dtype=float)
        cs[n:] = cs[n:] - cs[:-n]
        out = np.full(len(arr), np.nan)
        out[n - 1:] = cs[n - 1:] / n
        return out

    sma50_arr  = _rolling(closes, SMA_FAST)
    sma100_arr = _rolling(closes, SMA_MID)
    sma200_arr = _rolling(closes, SMA_SLOW)

    sma50_prev  = sma50_arr[-2];  sma50_curr  = sma50_arr[-1]
    sma100_prev = sma100_arr[-2]; sma100_curr = sma100_arr[-1]
    sma200_curr = sma200_arr[-1]

    # ================================================================
    # SMA Run 1 strategy
    # ================================================================

    # Detect open SMA trade for cross-exit check
    open_sma = next(
        (t for t in open_trades
         if t.get('symbol') == 'EURUSD' and t.get('strategy') == 'SMA'),
        None
    )

    if open_sma is not None:
        # Check for adverse SMA50 x SMA100 cross
        bear_cross = sma50_prev >= sma100_prev and sma50_curr < sma100_curr
        bull_cross = sma50_prev <= sma100_prev and sma50_curr > sma100_curr

        adverse = (
            (open_sma['direction'] == 'BUY'  and bear_cross) or
            (open_sma['direction'] == 'SELL' and bull_cross)
        )
        if adverse:
            log.info(f"EURUSD SMA cross-exit: ticket={open_sma['ticket']} "
                     f"direction={open_sma['direction']}")
            signals.append({
                'signal'             : 'CROSS_EXIT',
                'strategy'           : 'SMA',
                'cross_exit_ticket'  : open_sma['ticket'],
                'sl_pips'            : SMA_SL_PIPS,
                'tp_pips'            : SMA_TP_PIPS,
                'reason'             : (f"SMA50 x SMA100 adverse cross against "
                                        f"{open_sma['direction']}"),
            })

    # New SMA entry (only when no open SMA trade and within limits)
    sma_ok = (
        open_sma is None
        and state.get('sma_daily_trades', 0) < MAX_SMA_DAILY
        and state.get('sma_consec_losses', 0) < MAX_EURUSD_CONSEC
    )

    if sma_ok and not np.isnan(sma200_curr):
        flat = abs(sma50_curr - sma100_curr) / EURUSD_PIP_SIZE < SMA_FLAT_PIPS
        if not flat:
            last_close    = closes[-1]
            bull_cross    = sma50_prev <= sma100_prev and sma50_curr > sma100_curr
            bear_cross    = sma50_prev >= sma100_prev and sma50_curr < sma100_curr
            sma_trend_ok  = (bull_cross and h1_trend == 1 and last_close > sma200_curr) or \
                            (bear_cross and h1_trend == -1 and last_close < sma200_curr)

            if bull_cross and h1_trend == 1 and last_close > sma200_curr:
                log.info("EURUSD SMA: BUY -- SMA50 x SMA100 bullish cross")
                signals.append({
                    'signal'  : 'BUY',
                    'strategy': 'SMA',
                    'sl_pips' : SMA_SL_PIPS,
                    'tp_pips' : SMA_TP_PIPS,
                    'reason'  : 'SMA50 crossed above SMA100, H1 EMA50 bullish, price > SMA200',
                })
            elif bear_cross and h1_trend == -1 and last_close < sma200_curr:
                log.info("EURUSD SMA: SELL -- SMA50 x SMA100 bearish cross")
                signals.append({
                    'signal'  : 'SELL',
                    'strategy': 'SMA',
                    'sl_pips' : SMA_SL_PIPS,
                    'tp_pips' : SMA_TP_PIPS,
                    'reason'  : 'SMA50 crossed below SMA100, H1 EMA50 bearish, price < SMA200',
                })

    # ================================================================
    # EMA Pullback strategy (Test A -- no cross-exit)
    # ================================================================

    open_ema = any(
        t.get('symbol') == 'EURUSD' and t.get('strategy') == 'EMA'
        for t in open_trades
    )
    ema_ok = (
        not open_ema
        and state.get('ema_daily_trades', 0) < MAX_EMA_DAILY
        and state.get('ema_consec_losses', 0) < MAX_EURUSD_CONSEC
    )

    if ema_ok:
        ema5  = _ewm_ema(closes, EMA_FAST_N)
        ema20 = _ewm_ema(closes, EMA_MID_N)
        ema50 = _ewm_ema(closes, EMA_SLOW_N)

        e5   = ema5[-1];  e20  = ema20[-1];  e50  = ema50[-1]
        last = closes[-1]

        bull_aligned = e5 > e20 > e50
        bear_aligned = e5 < e20 < e50

        touch = EMA_TOUCH_PIPS * EURUSD_PIP_SIZE
        pb_pending = state.get('ema_pullback_pending', False)
        pb_dir     = state.get('ema_pullback_dir', '')

        if not pb_pending:
            # Detect new pullback to EMA20
            if bull_aligned and h1_trend == 1:
                if abs(last - e20) <= touch or last < e20:
                    state['ema_pullback_pending'] = True
                    state['ema_pullback_dir']     = 'BUY'
                    log.info(f"EURUSD EMA: BUY pullback detected  "
                             f"close={last:.5f}  EMA20={e20:.5f}")
            elif bear_aligned and h1_trend == -1:
                if abs(last - e20) <= touch or last > e20:
                    state['ema_pullback_pending'] = True
                    state['ema_pullback_dir']     = 'SELL'
                    log.info(f"EURUSD EMA: SELL pullback detected  "
                             f"close={last:.5f}  EMA20={e20:.5f}")
        else:
            # Pullback pending -- check for confirmation bar
            if pb_dir == 'BUY':
                if last > e20 and bull_aligned and h1_trend == 1:
                    log.info("EURUSD EMA: BUY confirmed -- close above EMA20 after pullback")
                    signals.append({
                        'signal'  : 'BUY',
                        'strategy': 'EMA',
                        'sl_pips' : EMA_SL_PIPS,
                        'tp_pips' : EMA_TP_PIPS,
                        'reason'  : 'EMA pullback confirmed: close above EMA20, H1 bullish',
                    })
                    state['ema_pullback_pending'] = False
                    state['ema_pullback_dir']     = ''
                elif not bull_aligned or h1_trend != 1:
                    state['ema_pullback_pending'] = False
                    state['ema_pullback_dir']     = ''
                    log.info("EURUSD EMA: BUY pullback cancelled -- alignment broken")

            elif pb_dir == 'SELL':
                if last < e20 and bear_aligned and h1_trend == -1:
                    log.info("EURUSD EMA: SELL confirmed -- close below EMA20 after pullback")
                    signals.append({
                        'signal'  : 'SELL',
                        'strategy': 'EMA',
                        'sl_pips' : EMA_SL_PIPS,
                        'tp_pips' : EMA_TP_PIPS,
                        'reason'  : 'EMA pullback confirmed: close below EMA20, H1 bearish',
                    })
                    state['ema_pullback_pending'] = False
                    state['ema_pullback_dir']     = ''
                elif not bear_aligned or h1_trend != -1:
                    state['ema_pullback_pending'] = False
                    state['ema_pullback_dir']     = ''
                    log.info("EURUSD EMA: SELL pullback cancelled -- alignment broken")

    return signals, state
