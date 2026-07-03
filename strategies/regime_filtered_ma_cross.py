"""
RegimeFilteredMACross -- NEW, UNTESTED strategy awaiting backtest validation.

Do NOT set active: true on any pairs/*.yaml until a full walk-forward
backtest has been run and the strategy passes the standard validation
criteria (profit factor > 1.3, max drawdown < 8%, >= 60% profitable
months over a 2-year in-sample period with a clean out-of-sample
forward test).

STRUCTURAL DIFFERENCE FROM H4TrendPullback (which failed):
  H4TrendPullback combined regime detection and entry signal into one
  calculation: the H4 trend detected the regime AND the H4 pullback to
  the H1 50 EMA was the entry trigger. These two responsibilities were
  coupled on the same timeframe. The backtest showed this produced late
  re-entries into trend moves, with poor SL geometry (the pullback SL
  was often too tight relative to the H4's own volatility) -- 0/81
  training configs passed the criteria in Run 1; in Run 2 (no daily EOD
  cut), the forward-test drawdown roughly doubled to 17.6%.

  RegimeFilteredMACross deliberately SEPARATES the two responsibilities:
    - H4 ONLY tells us which DIRECTION to trade (regime gate).
      It never determines WHEN to enter.
    - H1 ONLY tells us WHEN to enter (cross trigger).
      It is computed completely independently of the H4 calculation.
  A fresh H1 EMA9/EMA21 cross aligned with the H4 regime direction is
  a faster signal that catches moves earlier within the H4 structure,
  rather than waiting for the H4 itself to show a pullback.

Strategy logic:
  1. H4 REGIME (direction gate only -- not the entry trigger):
     H4 50/200 SMA. SMA50 > SMA200 → BULLISH regime (+1).
     SMA50 < SMA200 → BEARISH regime (-1). Equal / insufficient → NEUTRAL
     (0, no trade). Only fully closed H4 bars are used (pos=0, since the
     current bar is still forming).

  2. H1 ENTRY TRIGGER (timing gate only -- independent of H4):
     EMA9 / EMA21 cross on H1. A cross is FRESH only when:
       - EMA9 was BELOW EMA21 on bar[-2] and is ABOVE EMA21 on bar[-1]
         → fresh bullish cross → candidate BUY.
       - EMA9 was ABOVE EMA21 on bar[-2] and is BELOW EMA21 on bar[-1]
         → fresh bearish cross → candidate SELL.
     Stale alignment (EMA9 has been above EMA21 for many bars) is NOT
     treated as a signal -- only the bar on which the cross actually
     happens produces a candidate entry.

  3. GATE: candidate entry is only taken when H1 cross direction matches
     the H4 regime. A bullish H1 cross in a bearish H4 regime is silently
     rejected (and vice versa). No position is taken.

  4. Session filter: London (08:00-17:00 UTC). Trend moves on H1 are most
     directional and predictable within the London window. Deliberately
     narrower than VRT's window to avoid NY-session whipsaws on a cross
     signal that is already one bar old by the time it's acted on.

  5. Fixed SL and TP from the pair YAML (sl_pips / tp_pips). No trailing
     stop, no cross-exit. Cross-exit logic has failed across 4 prior
     strategy variants in this codebase by cutting profitable trades early.

  6. Friday close at 20:00 UTC handled by the orchestrator (see
     main_agent.py step_friday_close()). Mon-Thu, positions run to natural
     SL/TP.

  7. One trade per day per pair.

Design notes:
  - EMA periods 9/21 are standard fast-trigger parameters. They will
    fire more frequently than 20/50, which is appropriate here because
    the H4 regime gate is doing the heavy filtering -- we want the H1
    to trigger early in a move, not confirm it late.
  - If the backtest shows too many false H1 crosses within a valid H4
    regime, consider adding a minimum cross size threshold (EMA9 must be
    at least N pips above EMA21 to count), or a one-bar confirmation
    hold. These would be parameter additions, not structural changes.
  - This strategy has NOT been validated. Backtest thoroughly before use.
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

# -- logging (identical setup to LondonBreakout / MeanReversion)
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


# -- H4 regime parameters (direction gate)
H4_SMA_FAST    = 50
H4_SMA_SLOW    = 200
H4_BARS_NEEDED = 220

# -- H1 cross parameters (entry trigger -- independent of H4)
H1_EMA_FAST    = 9
H1_EMA_SLOW    = 21
H1_BARS_NEEDED = H1_EMA_SLOW + 3   # 2+ bars needed to detect a fresh cross

SESSION_START_HOUR = 8    # London open
SESSION_END_HOUR   = 17   # end before late NY

REQUIRED_KEYS = ['pair', 'strategy', 'active', 'timeframe', 'risk_percent',
                 'sl_pips', 'tp_pips', 'h4_filter', 'session', 'friday_close']


def _session_window(utc_time: datetime) -> bool:
    return SESSION_START_HOUR <= utc_time.hour < SESSION_END_HOUR


class RegimeFilteredMACross(BaseStrategy):

    NAME = "regime_filtered_ma_cross"
    SESSION = "london"
    COMPATIBLE_PAIRS = ["GBPJPY", "GBPUSD", "EURUSD", "USDCAD"]

    def __init__(self, pair_config: dict):
        super().__init__(pair_config)
        self.validate_config(pair_config)
        self.pip_size = 0.01 if 'JPY' in self.pair else 0.0001
        self._last_trade_date = None   # same-day dedup; authoritative gate
                                        # is orchestrator open_trades check

    # ------------------------------------------------------------------
    # BaseStrategy required interface
    # ------------------------------------------------------------------

    def validate_config(self, pair_config: dict) -> None:
        missing = [k for k in REQUIRED_KEYS if k not in pair_config]
        if missing:
            raise ValueError(
                f"RegimeFilteredMACross config for {pair_config.get('pair', '?')} "
                f"missing required key(s): {missing}"
            )

    def get_session_windows(self) -> list:
        return [
            {'name': 'london',
             'start': f'{SESSION_START_HOUR:02d}:00',
             'end':   f'{SESSION_END_HOUR:02d}:00'},
        ]

    def generate_signal(self, pair: str, current_price: float, h4_trend: int,
                        **context):
        """
        Returns 'BUY' / 'SELL' / None.
        Called with only the 3 base args (smoke test / verify_architecture.py),
        returns None safely without hitting MT5.
        """
        session_data = context.get('session_data')
        session      = context.get('session')
        if session_data is None or session is None:
            return None

        result = self.check_cross(session_data, session)
        if result['signal'] in ('BUY', 'SELL'):
            return result['signal']
        return None

    def calculate_sl(self, entry_price: float, direction: str, pair_config: dict) -> float:
        dist = pair_config.get('sl_pips', 0) * self.pip_size
        return entry_price - dist if direction == 'BUY' else entry_price + dist

    def calculate_tp(self, entry_price: float, direction: str, pair_config: dict) -> float:
        dist = pair_config.get('tp_pips', 0) * self.pip_size
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
        """Most recent CLOSED H1 bars (pos=1 skips the still-forming bar)."""
        rates = mt5.copy_rates_from_pos(self.pair, mt5.TIMEFRAME_H1, 1, count)
        if rates is None or len(rates) < H1_BARS_NEEDED:
            return None
        return rates

    # ------------------------------------------------------------------
    # H4 REGIME (direction gate -- sole purpose is to say which way)
    # ------------------------------------------------------------------

    def h4_regime(self, log) -> int:
        """
        Returns +1 (BULLISH), -1 (BEARISH), or 0 (NEUTRAL).
        This is a DIRECTION GATE ONLY. It does not determine when to enter.
        Structural note: keeping this completely separate from h1_cross()
        is the key difference from H4TrendPullback, which combined both.
        """
        bars = self._h4_bars()
        if bars is None:
            log.warning(f"{self.pair}: insufficient H4 history "
                        f"({H4_SMA_SLOW} bars needed for regime)")
            return 0

        closes = np.array([b['close'] for b in bars])
        sma50  = float(np.mean(closes[-H4_SMA_FAST:]))
        sma200 = float(np.mean(closes[-H4_SMA_SLOW:]))

        if sma50 > sma200:
            return 1
        if sma50 < sma200:
            return -1
        return 0

    # ------------------------------------------------------------------
    # H1 CROSS (entry trigger -- independent of H4)
    # ------------------------------------------------------------------

    @staticmethod
    def _ema_series(values: np.ndarray, period: int) -> np.ndarray:
        alpha = 2.0 / (period + 1)
        out   = np.empty(len(values))
        out[0] = values[0]
        for i in range(1, len(values)):
            out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
        return out

    def h1_fresh_cross(self, log):
        """
        Returns (+1, bar_time) for a fresh bullish cross,
                (-1, bar_time) for a fresh bearish cross,
                ( 0, bar_time) if no fresh cross on the most recent bar.

        'Fresh' means the cross state CHANGED on bars[-1] vs bars[-2].
        Stale alignment (EMA9 has been above EMA21 for many bars) is NOT
        treated as a signal -- this prevents re-triggering every cycle.
        """
        bars = self._h1_bars()
        if bars is None:
            log.info(f"  {self.pair}: insufficient H1 history "
                     f"({H1_BARS_NEEDED} bars needed for cross)")
            return 0, None

        closes   = np.array([b['close'] for b in bars])
        ema_fast = self._ema_series(closes, H1_EMA_FAST)
        ema_slow = self._ema_series(closes, H1_EMA_SLOW)

        bar_time   = datetime.fromtimestamp(bars[-1]['time'], tz=timezone.utc)
        prev_above = ema_fast[-2] > ema_slow[-2]
        curr_above = ema_fast[-1] > ema_slow[-1]

        if not prev_above and curr_above:
            log.info(f"  {self.pair}: fresh BULLISH H1 "
                     f"EMA{H1_EMA_FAST}/EMA{H1_EMA_SLOW} cross "
                     f"at {bar_time.strftime('%H:%M')} UTC "
                     f"(fast={ema_fast[-1]:.5f}  slow={ema_slow[-1]:.5f})")
            return 1, bar_time

        if prev_above and not curr_above:
            log.info(f"  {self.pair}: fresh BEARISH H1 "
                     f"EMA{H1_EMA_FAST}/EMA{H1_EMA_SLOW} cross "
                     f"at {bar_time.strftime('%H:%M')} UTC "
                     f"(fast={ema_fast[-1]:.5f}  slow={ema_slow[-1]:.5f})")
            return -1, bar_time

        return 0, bar_time

    # ------------------------------------------------------------------
    # prepare() -- H4 regime snapshot, called once per session
    # ------------------------------------------------------------------

    def prepare(self, log) -> dict | None:
        if not self._connect(log):
            return None

        regime = self.h4_regime(log)
        if regime == 0:
            log.info(f"  {self.pair}: NEUTRAL H4 regime -- skipping "
                     f"(SMA{H4_SMA_FAST}/{H4_SMA_SLOW} not separated)")
            return None

        label = 'BULLISH' if regime == 1 else 'BEARISH'
        log.info(f"  {self.pair}: H4 regime = {label}  "
                 f"(waiting for H1 EMA{H1_EMA_FAST}/{H1_EMA_SLOW} cross "
                 f"in same direction)")
        return {'h4_regime': regime}

    # ------------------------------------------------------------------
    # check_cross() -- H1 fresh cross gated by H4 regime
    # ------------------------------------------------------------------

    def check_cross(self, session_data: dict, session: str) -> dict:
        """
        session_data : this pair's own data dict ({'h4_regime': int})
                        -- already scoped to self.pair by the caller.

        The H4 regime was determined at session prep time (prepare()).
        The H1 cross is computed fresh here from live MT5 data.
        These two calculations are INDEPENDENT -- the separation is
        the structural fix vs H4TrendPullback.
        """
        log = _log()

        no_signal = lambda reason: {
            'signal'      : 'NO_SIGNAL',
            'reason'      : reason,
            'entry_price' : 0.0,
        }

        if not session_data:
            return no_signal('pair not in session data (NEUTRAL H4 regime)')

        h4_regime = session_data['h4_regime']

        if not self._connect(log):
            return no_signal('MT5 connection failed')

        cross, bar_time = self.h1_fresh_cross(log)
        if bar_time is None:
            return no_signal('no H1 bar data')

        if not _session_window(bar_time):
            return no_signal(
                f'bar at {bar_time.hour:02d}:00 UTC outside London window')

        if self._last_trade_date == bar_time.date():
            return no_signal('already traded this pair today (one trade/day limit)')

        if cross == 0:
            return no_signal(
                f'no fresh H1 EMA{H1_EMA_FAST}/{H1_EMA_SLOW} cross on last bar')

        # Gate: H1 cross direction must match H4 regime
        if cross == 1 and h4_regime == 1:
            rates = mt5.copy_rates_from_pos(self.pair, mt5.TIMEFRAME_H1, 1, 1)
            entry = float(rates[0]['close']) if rates is not None else 0.0
            self._last_trade_date = bar_time.date()
            return {
                'signal'      : 'BUY',
                'reason'      : (f'H1 EMA{H1_EMA_FAST} crossed above '
                                 f'EMA{H1_EMA_SLOW} -- BULLISH H4 regime confirmed'),
                'entry_price' : entry,
            }

        if cross == -1 and h4_regime == -1:
            rates = mt5.copy_rates_from_pos(self.pair, mt5.TIMEFRAME_H1, 1, 1)
            entry = float(rates[0]['close']) if rates is not None else 0.0
            self._last_trade_date = bar_time.date()
            return {
                'signal'      : 'SELL',
                'reason'      : (f'H1 EMA{H1_EMA_FAST} crossed below '
                                 f'EMA{H1_EMA_SLOW} -- BEARISH H4 regime confirmed'),
                'entry_price' : entry,
            }

        cross_dir  = 'bullish' if cross == 1 else 'bearish'
        regime_dir = 'BULLISH' if h4_regime == 1 else 'BEARISH'
        return no_signal(
            f'{cross_dir} H1 cross rejected: mismatched with {regime_dir} H4 regime')
