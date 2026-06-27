"""
LondonBreakout strategy -- extracted from the original agent_strategy.py
breakout logic (GBPJPY and EURJPY). One instance is bound to a single pair
via pair_config['pair']; pair_manager creates one instance per active pair
that lists `strategy: london_breakout` in its YAML.

All parameters, thresholds, and decision logic are preserved EXACTLY as
they were in agent_strategy.py -- this is a structural move, not a
behavior change.

Strategy logic (unchanged from the original):
  1. H4 trend filter: Close > SMA50 > SMA200 = BULLISH (+1)
                      Close < SMA50 < SMA200 = BEARISH (-1)
                      Otherwise             = NEUTRAL  (0, no trade)
  2. Asian session range: High and Low of M15 bars from 00:00-06:45 UTC
  3. Breakout confirmation (stricter than backtest):
     -- Bar CLOSE must be beyond the Asian range (not just a wick)
     -- Direction must match H4 trend
     -- Overshoot must be <= 20 pips past the range level
  4. EOD close at 17:30 UTC (handled by the orchestrator, not this class --
     see main_agent.py step_eod_close(), unchanged)
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np

from strategies.base_strategy import BaseStrategy
from core.session_filter import london_session, ny_session

# -- logging (same setup/format as the original agent_strategy.py _log())
LOGS_DIR = Path(__file__).parent.parent / 'data' / 'logs'

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

# -- strategy parameters (preserved verbatim from agent_strategy.py)
H4_SMA_FAST          = 50
H4_SMA_SLOW          = 200
H4_BARS_NEEDED       = 220    # enough history for SMA200
ASIAN_END_HOUR       = 7      # 00:00-06:45 UTC bars used for Asian range
MIN_ASIAN_RANGE_PIPS = 10     # skip days with tiny ranges
MAX_OVERSHOOT_PIPS   = 20     # cancel if breakout bar closed > 20p past level

REQUIRED_KEYS = ['pair', 'strategy', 'active', 'timeframe', 'risk_percent',
                  'h4_filter', 'session', 'eod_close_utc']


class LondonBreakout(BaseStrategy):

    COMPATIBLE_PAIRS = ["GBPJPY", "EURJPY", "GBPUSD"]

    def __init__(self, pair_config: dict):
        super().__init__(pair_config)
        self.validate_config(pair_config)
        self.pip_size = 0.01 if 'JPY' in self.pair else 0.0001

    # ------------------------------------------------------------------
    # BaseStrategy required interface
    # ------------------------------------------------------------------

    def validate_config(self, pair_config: dict) -> None:
        missing = [k for k in REQUIRED_KEYS if k not in pair_config]
        if missing:
            raise ValueError(
                f"LondonBreakout config for {pair_config.get('pair', '?')} "
                f"missing required key(s): {missing}"
            )

    def get_session_windows(self) -> list:
        return [
            {'name': 'london', 'start': '08:00', 'end': '13:00'},
            {'name': 'ny',      'start': '13:00', 'end': '22:00'},
        ]

    def generate_signal(self, pair: str, current_price: float, h4_trend: int,
                         **context):
        """
        Required interface method. With full context (session_data + session
        kwargs, as supplied in production by agent_strategy.py), delegates to
        check_breakout() and returns 'BUY' / 'SELL' / None.

        Called with only the 3 base args (e.g. a smoke test with mock data),
        there is no Asian range / session context available to make a real
        breakout decision, so it safely returns None rather than hitting MT5.
        """
        session_data = context.get('session_data')
        session      = context.get('session')
        if session_data is None or session is None:
            return None

        result = self.check_breakout(session_data, session)
        if result['signal'] in ('BUY', 'SELL'):
            return result['signal']
        return None

    def calculate_sl(self, entry_price: float, direction: str, pair_config: dict) -> float:
        sl_pips = pair_config.get('sl_pips')
        if sl_pips is None:
            raise ValueError(
                f"LondonBreakout.calculate_sl: pair_config for {self.pair} has no "
                f"'sl_pips' -- breakout SL is derived from the Asian range at "
                f"prepare() time, not a static YAML value"
            )
        dist = sl_pips * self.pip_size
        return entry_price - dist if direction == 'BUY' else entry_price + dist

    def calculate_tp(self, entry_price: float, direction: str, pair_config: dict) -> float:
        tp_pips = pair_config.get('tp_pips')
        if tp_pips is None:
            raise ValueError(
                f"LondonBreakout.calculate_tp: pair_config for {self.pair} has no "
                f"'tp_pips' -- breakout TP is derived from the Asian range at "
                f"prepare() time, not a static YAML value"
            )
        dist = tp_pips * self.pip_size
        return entry_price + dist if direction == 'BUY' else entry_price - dist

    # ------------------------------------------------------------------
    # MT5 helpers (preserved verbatim from agent_strategy.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _connect(log) -> bool:
        if mt5.initialize():
            return True
        log.error(f"MT5 init failed: {mt5.last_error()}")
        return False

    def _h4_bars(self, count: int = H4_BARS_NEEDED):
        rates = mt5.copy_rates_from_pos(self.pair, mt5.TIMEFRAME_H4, 0, count)
        if rates is None or len(rates) < H4_SMA_SLOW:
            return None
        return rates

    def _m15_bars_today(self):
        now      = datetime.now(timezone.utc)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        rates = mt5.copy_rates_range(self.pair, mt5.TIMEFRAME_M15, midnight, now)
        if rates is None or len(rates) == 0:
            return None
        return rates

    def _last_closed_m15(self):
        rates = mt5.copy_rates_from_pos(self.pair, mt5.TIMEFRAME_M15, 1, 1)
        if rates is None or len(rates) == 0:
            return None
        return rates[0]

    # ------------------------------------------------------------------
    # H4 trend (preserved verbatim from agent_strategy.py _h4_trend)
    # ------------------------------------------------------------------

    def h4_trend(self, log) -> int:
        """Returns +1 (BULLISH), -1 (BEARISH), or 0 (NEUTRAL)."""
        bars = self._h4_bars()
        if bars is None:
            log.warning(f"{self.pair}: not enough H4 history ({H4_SMA_SLOW} bars needed)")
            return 0

        closes = np.array([b['close'] for b in bars])
        sma50  = np.mean(closes[-H4_SMA_FAST:])
        sma200 = np.mean(closes[-H4_SMA_SLOW:])
        last   = closes[-1]

        if last > sma50 > sma200:
            return 1
        if last < sma50 < sma200:
            return -1
        return 0

    # ------------------------------------------------------------------
    # Asian range (preserved verbatim from agent_strategy.py _asian_range)
    # ------------------------------------------------------------------

    def asian_range(self, log) -> dict | None:
        bars = self._m15_bars_today()
        if bars is None:
            log.warning(f"{self.pair}: no M15 data for today")
            return None

        asian_bars = [b for b in bars if
                      datetime.fromtimestamp(b['time'], tz=timezone.utc).hour < ASIAN_END_HOUR]

        if len(asian_bars) < 4:
            log.warning(f"{self.pair}: insufficient Asian session bars ({len(asian_bars)})")
            return None

        high = max(b['high'] for b in asian_bars)
        low  = min(b['low']  for b in asian_bars)
        range_pips = (high - low) / self.pip_size

        if range_pips < MIN_ASIAN_RANGE_PIPS:
            log.info(f"{self.pair}: Asian range too tight ({range_pips:.1f}p < {MIN_ASIAN_RANGE_PIPS}p)")
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

    # ------------------------------------------------------------------
    # prepare() -- combines H4 trend + Asian range, called once per session
    # (preserved logic from agent_strategy.py prepare_session(), per-pair)
    # ------------------------------------------------------------------

    def prepare(self, log) -> dict | None:
        if not self._connect(log):
            return None

        trend = self.h4_trend(log)
        if trend == 0:
            log.info(f"  {self.pair}: NEUTRAL H4 trend -- skipping")
            return None

        asian = self.asian_range(log)
        if asian is None:
            log.info(f"  {self.pair}: Asian range unavailable -- skipping")
            return None

        result = {'h4_trend': trend, **asian}
        trend_label = 'BULLISH' if trend == 1 else 'BEARISH'
        log.info(f"  {self.pair}: {trend_label}  range={asian['range_pips']:.1f}p  "
                 f"H={asian['asian_high']:.5f}  L={asian['asian_low']:.5f}  "
                 f"SL={asian['sl_pips']:.1f}p  TP={asian['tp_pips']:.1f}p")
        return result

    # ------------------------------------------------------------------
    # check_breakout() (preserved verbatim from agent_strategy.py check_breakout)
    # ------------------------------------------------------------------

    def check_breakout(self, session_data: dict, session: str) -> dict:
        """
        session_data : this pair's own data dict (h4_trend, asian_high,
                        asian_low, range_pips, sl_pips, tp_pips) -- already
                        scoped to self.pair by the caller.
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

        if not session_data:
            return no_signal('pair not in session data (neutral trend or bad range)')

        asian_high = session_data['asian_high']
        asian_low  = session_data['asian_low']
        trend      = session_data['h4_trend']

        if not self._connect(log):
            return no_signal('MT5 connection failed')

        bar = self._last_closed_m15()
        if bar is None:
            return no_signal('no M15 bar data')

        bar_time  = datetime.fromtimestamp(bar['time'], tz=timezone.utc)
        bar_close = bar['close']

        if session == 'london' and not london_session(bar_time):
            return no_signal(f'bar at {bar_time.hour:02d}:00 UTC outside London window')
        if session == 'ny' and not ny_session(bar_time):
            return no_signal(f'bar at {bar_time.hour:02d}:00 UTC outside NY window')

        if trend == 1 and bar_close > asian_high:
            overshoot = (bar_close - asian_high) / self.pip_size
            if overshoot > MAX_OVERSHOOT_PIPS:
                return no_signal(f'BUY overshoot {overshoot:.0f}p exceeds {MAX_OVERSHOOT_PIPS}p limit')
            return {
                'signal'            : 'BUY',
                'reason'            : 'close above Asian High, H4 BULLISH',
                'trigger_bar_close' : bar_close,
                'trigger_bar_time'  : bar_time.isoformat(),
                'entry_price'       : asian_high,
                'overshoot_pips'    : round(overshoot, 1),
            }

        if trend == -1 and bar_close < asian_low:
            overshoot = (asian_low - bar_close) / self.pip_size
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

        trend_label = 'BULLISH' if trend == 1 else 'BEARISH'
        return no_signal(f'no {trend_label} close beyond range on bar {bar_time.strftime("%H:%M")}')
