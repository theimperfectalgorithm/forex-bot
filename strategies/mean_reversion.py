"""
MeanReversion -- H1 RSI(14) mean-reversion entries during a RANGING H4
market (SMA50 and SMA200 within a tight band -- the opposite gate from
LondonBreakout / H4TrendPullback, which both require a trend; this
strategy explicitly requires the absence of one).

Strategy logic:
  1. H4 range filter: |SMA50 - SMA200| < H4_RANGE_THRESHOLD_PIPS (default
     20p) -- only trade when H4 is NOT trending. A trending H4 = no trade.
  2. Entry on H1: RSI(RSI_PERIOD) < RSI_OVERSOLD (default 30) -> BUY,
     expecting reversion up to the H1 20 SMA. RSI > RSI_OVERBOUGHT
     (default 70) -> SELL, expecting reversion down to the H1 20 SMA.
     Uses the last CLOSED H1 bar's close as entry (not a live tick),
     matching every other strategy in this codebase.
  3. Session filter: London only, SESSION_START_HOUR-SESSION_END_HOUR UTC
     (08:00-16:00) -- deliberately narrower than the full London/NY
     window to avoid the volatility spikes mean-reversion setups are
     most vulnerable to.
  4. TP = the live H1 20 SMA value at signal time (the "mean" being
     reverted to). SL = SL_MULTIPLIER x the entry-to-TP distance (default
     1.5x), on the opposite side of entry from TP -- asymmetric by
     design: mean-reversion setups tend to win more often than they
     lose, but a loss means the reversion thesis failed and price kept
     trending, which tends to travel further than a successful
     reversion does.
  5. Friday close at 20:00 UTC (handled by the orchestrator, not this
     class -- see main_agent.py step_friday_close()). Mon-Thu, positions
     run to natural SL/TP.
  6. One trade per day per pair.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
import numpy as np

from strategies.base_strategy import BaseStrategy

# -- logging (same setup/format as LondonBreakout's _log())
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

# -- strategy parameters
H4_SMA_FAST             = 50
H4_SMA_SLOW             = 200
H4_BARS_NEEDED          = 220
H4_RANGE_THRESHOLD_PIPS = 20     # SMA50/SMA200 must be within this many pips -- ranging H4

H1_SMA_PERIOD  = 20              # the "mean" target
RSI_PERIOD     = 14
RSI_OVERSOLD   = 30
RSI_OVERBOUGHT = 70
H1_BARS_NEEDED = max(H1_SMA_PERIOD, RSI_PERIOD) + 1

SESSION_START_HOUR = 8    # London only -- 08:00-16:00 UTC (deliberately
SESSION_END_HOUR   = 16   # narrower than NY to avoid its volatility)

SL_MULTIPLIER = 1.5       # SL = SL_MULTIPLIER x the entry-to-TP distance

REQUIRED_KEYS = ['pair', 'strategy', 'active', 'timeframe', 'risk_percent',
                  'h4_filter', 'session', 'friday_close']


def _session_window(utc_time: datetime) -> bool:
    """London-only entry session filter (08:00-16:00 UTC)."""
    return SESSION_START_HOUR <= utc_time.hour < SESSION_END_HOUR


class MeanReversion(BaseStrategy):

    NAME = "mean_reversion"
    SESSION = "london_ny"
    COMPATIBLE_PAIRS = ["AUDUSD", "NZDUSD", "EURUSD"]

    def __init__(self, pair_config: dict):
        super().__init__(pair_config)
        self.validate_config(pair_config)
        self.pip_size = 0.01 if 'JPY' in self.pair else 0.0001
        self._last_trade_date = None  # same-day dedup (see check_reversion below) --
                                       # the authoritative "one position at a time"
                                       # gate is the orchestrator's own open_trades
                                       # check in main_agent.py step_check_breakouts()

    # ------------------------------------------------------------------
    # BaseStrategy required interface
    # ------------------------------------------------------------------

    def validate_config(self, pair_config: dict) -> None:
        missing = [k for k in REQUIRED_KEYS if k not in pair_config]
        if missing:
            raise ValueError(
                f"MeanReversion config for {pair_config.get('pair', '?')} "
                f"missing required key(s): {missing}"
            )

    def get_session_windows(self) -> list:
        return [
            {'name': 'london', 'start': f'{SESSION_START_HOUR:02d}:00',
                                'end': f'{SESSION_END_HOUR:02d}:00'},
        ]

    def generate_signal(self, pair: str, current_price: float, h4_trend: int,
                         **context):
        """
        Required interface method. With full context (session_data +
        session kwargs, as supplied in production by agent_strategy.py),
        delegates to check_reversion() and returns 'BUY' / 'SELL' / None.

        Called with only the 3 base args (e.g. a smoke test with mock
        data), there is no RSI/SMA context available to make a real
        decision, so it safely returns None rather than hitting MT5.
        """
        session_data = context.get('session_data')
        session      = context.get('session')
        if session_data is None or session is None:
            return None

        result = self.check_reversion(session_data, session)
        if result['signal'] in ('BUY', 'SELL'):
            return result['signal']
        return None

    def calculate_sl(self, entry_price: float, direction: str, pair_config: dict) -> float:
        sl_pips = pair_config.get('sl_pips')
        if sl_pips is None:
            raise ValueError(
                f"MeanReversion.calculate_sl: pair_config for {self.pair} has no "
                f"'sl_pips' -- SL is derived from the entry-to-TP distance at "
                f"check_reversion() time, not a static YAML value"
            )
        dist = sl_pips * self.pip_size
        return entry_price - dist if direction == 'BUY' else entry_price + dist

    def calculate_tp(self, entry_price: float, direction: str, pair_config: dict) -> float:
        tp_pips = pair_config.get('tp_pips')
        if tp_pips is None:
            raise ValueError(
                f"MeanReversion.calculate_tp: pair_config for {self.pair} has no "
                f"'tp_pips' -- TP is the live H1 20 SMA at check_reversion() time, "
                f"not a static YAML value"
            )
        dist = tp_pips * self.pip_size
        return entry_price + dist if direction == 'BUY' else entry_price - dist

    # ------------------------------------------------------------------
    # MT5 helpers
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

    def _h1_bars(self, count: int = H1_BARS_NEEDED):
        """Most recent CLOSED H1 bars (pos=1 skips the still-forming current bar)."""
        rates = mt5.copy_rates_from_pos(self.pair, mt5.TIMEFRAME_H1, 1, count)
        if rates is None or len(rates) < H1_BARS_NEEDED:
            return None
        return rates

    def _last_closed_h1(self):
        rates = mt5.copy_rates_from_pos(self.pair, mt5.TIMEFRAME_H1, 1, 1)
        if rates is None or len(rates) == 0:
            return None
        return rates[0]

    # ------------------------------------------------------------------
    # H4 range filter (opposite polarity of a trend filter)
    # ------------------------------------------------------------------

    def h4_is_ranging(self, log) -> bool:
        bars = self._h4_bars()
        if bars is None:
            log.warning(f"{self.pair}: not enough H4 history ({H4_SMA_SLOW} bars needed)")
            return False

        closes = np.array([b['close'] for b in bars])
        sma50  = np.mean(closes[-H4_SMA_FAST:])
        sma200 = np.mean(closes[-H4_SMA_SLOW:])
        diff_pips = abs(sma50 - sma200) / self.pip_size

        return diff_pips < H4_RANGE_THRESHOLD_PIPS

    # ------------------------------------------------------------------
    # H1 RSI + SMA20 ("the mean")
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi(closes: np.ndarray, period: int) -> float:
        """Wilder's RSI over the last `period` deltas -- returns the latest value."""
        deltas = np.diff(closes)
        gains  = np.clip(deltas, 0, None)
        losses = -np.clip(deltas, None, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    # ------------------------------------------------------------------
    # prepare() -- H4 range gate + H1 RSI/SMA20, called once per session
    # ------------------------------------------------------------------

    def prepare(self, log) -> dict | None:
        if not self._connect(log):
            return None

        if not self.h4_is_ranging(log):
            log.info(f"  {self.pair}: H4 trending (not ranging) -- skipping")
            return None

        bars = self._h1_bars()
        if bars is None:
            log.info(f"  {self.pair}: insufficient H1 history -- skipping")
            return None

        closes = np.array([b['close'] for b in bars])
        sma20  = float(np.mean(closes[-H1_SMA_PERIOD:]))
        rsi    = self._rsi(closes, RSI_PERIOD)

        log.info(f"  {self.pair}: RANGING H4  RSI={rsi:.1f}  H1_SMA20={sma20:.5f}")
        return {'sma20': sma20, 'rsi': rsi}

    # ------------------------------------------------------------------
    # check_reversion() -- RSI extreme + reversion target at the H1 20 SMA
    # ------------------------------------------------------------------

    def check_reversion(self, session_data: dict, session: str) -> dict:
        """
        session_data : this pair's own data dict ({'sma20': float,
                        'rsi': float}) -- already scoped to self.pair by
                        the caller.
        """
        log = _log()

        no_signal = lambda reason: {
            'signal'      : 'NO_SIGNAL',
            'reason'      : reason,
            'entry_price' : 0.0,
            'sl_price'    : 0.0,
            'tp_price'    : 0.0,
        }

        if not session_data:
            return no_signal('pair not in session data (H4 trending, not ranging)')

        sma20 = session_data['sma20']
        rsi   = session_data['rsi']

        if not self._connect(log):
            return no_signal('MT5 connection failed')

        bar = self._last_closed_h1()
        if bar is None:
            return no_signal('no H1 bar data')

        bar_time = datetime.fromtimestamp(bar['time'], tz=timezone.utc)
        entry    = bar['close']

        if not _session_window(bar_time):
            return no_signal(f'bar at {bar_time.hour:02d}:00 UTC outside London-only window')

        if self._last_trade_date == bar_time.date():
            return no_signal('already traded this pair today (one trade/day limit)')

        if rsi < RSI_OVERSOLD:
            direction = 'BUY'
            tp_price  = sma20
            tp_pips   = (tp_price - entry) / self.pip_size
        elif rsi > RSI_OVERBOUGHT:
            direction = 'SELL'
            tp_price  = sma20
            tp_pips   = (entry - tp_price) / self.pip_size
        else:
            return no_signal(f'RSI {rsi:.1f} not extreme (needs < {RSI_OVERSOLD} or '
                             f'> {RSI_OVERBOUGHT})')

        if tp_pips <= 0:
            return no_signal(f'H1 20 SMA is on the wrong side of entry for a {direction} '
                             f'reversion (tp_pips={tp_pips:.1f}) -- skipping')

        sl_pips  = tp_pips * SL_MULTIPLIER
        sl_price = (entry - sl_pips * self.pip_size if direction == 'BUY'
                    else entry + sl_pips * self.pip_size)

        self._last_trade_date = bar_time.date()
        return {
            'signal'      : direction,
            'reason'      : f'RSI {rsi:.1f} extreme, reverting to H1 20 SMA {sma20:.5f}',
            'entry_price' : entry,
            'sl_price'    : sl_price,
            'tp_price'    : tp_price,
            'sl_pips'     : round(sl_pips, 1),
            'tp_pips'     : round(tp_pips, 1),
        }
