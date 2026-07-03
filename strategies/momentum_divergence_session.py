"""
MomentumDivergenceSession -- NEW, UNTESTED strategy awaiting backtest validation.

Do NOT set active: true on any pairs/*.yaml until a full walk-forward
backtest has been run and the strategy passes the standard validation
criteria (profit factor > 1.3, max drawdown < 8%, >= 60% profitable
months over a 2-year in-sample period with a clean out-of-sample
forward test).

Strategy logic:
  1. Session filter: London/NY overlap only (12:00-17:00 UTC). Divergence
     signals outside this window are ignored. The overlap provides the
     liquidity depth that allows divergence signals to resolve cleanly --
     wider session windows (especially Asian hours) produce false divergence
     from low-volume price drift rather than genuine exhaustion.
  2. RSI(14) divergence detection on H1 bars:
       Bullish divergence: price makes a LOWER low while RSI makes a
         HIGHER low at the corresponding swing lows. Interpretation: price
         momentum is bottoming before price itself -- anticipates a reversal
         up. Only valid when the first swing low had RSI < RSI_EXTREME_LOW
         (default 35) -- the initial extreme is required for conviction.
       Bearish divergence: price makes a HIGHER high while RSI makes a
         LOWER high at the corresponding swing highs. Interpretation: price
         momentum is topping before price itself -- anticipates a reversal
         down. Only valid when the first swing high had RSI > RSI_EXTREME_HIGH
         (default 65) -- same reasoning.
  3. Swing detection: scans the last SWING_LOOKBACK H1 bars (default 30
     bars, ~30 hours) for swing highs and swing lows. A swing high is a
     bar whose high exceeds both immediate neighbours; a swing low is a bar
     whose low is below both immediate neighbours. Requires at least 2
     qualifying swings of the relevant type separated by at least
     MIN_SWING_GAP bars (default 3) for a valid divergence signal.
  4. Fixed SL and TP from the pair YAML (sl_pips / tp_pips). No trailing
     stop, no cross-exit. Cross-exit logic has failed across 4 prior
     strategy variants in this codebase by killing profitable trades before
     the TP was reached.
  5. Friday close at 20:00 UTC handled by the orchestrator (see
     main_agent.py step_friday_close()). Mon-Thu, positions run to natural
     SL/TP.
  6. One trade per day per pair.

Design notes:
  - Divergence is a momentum-ahead-of-price signal. It is NOT a pure
    reversal bet: the signal says momentum is fading; the RSI extreme
    requirement is what gives it conviction (a divergence where RSI was
    only at 55 is noise; one where RSI was at 72 and now shows a lower
    high on the same price high is a genuine exhaustion signal).
  - The session restriction is critical. Divergence patterns that form
    during thin Asian hours are structurally different -- lower volume
    means swings are price-discovery artefacts rather than exhaustion.
  - This strategy is conceptually separate from mean_reversion.py (which
    requires a RANGING H4 market). MomentumDivergenceSession is agnostic
    to H4 regime -- the divergence signal is self-contained and does not
    require any market-structure precondition beyond the session window.
  - This strategy has NOT been validated. The parameter values (RSI 14,
    extreme thresholds 35/65, lookback 30 bars) are starting estimates
    only -- backtest and grid-search before use.
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


# -- strategy parameters
RSI_PERIOD       = 14
RSI_EXTREME_LOW  = 35    # bullish divergence only valid if first swing RSI < this
RSI_EXTREME_HIGH = 65    # bearish divergence only valid if first swing RSI > this

SWING_LOOKBACK   = 30    # H1 bars to scan for swings (~30 hours of history)
MIN_SWING_GAP    = 3     # minimum bars separating two qualifying swings

SESSION_START_HOUR = 12  # London/NY overlap
SESSION_END_HOUR   = 17

H1_BARS_NEEDED = SWING_LOOKBACK + RSI_PERIOD + 5

REQUIRED_KEYS = ['pair', 'strategy', 'active', 'timeframe',
                 'risk_percent', 'sl_pips', 'tp_pips', 'session', 'friday_close']


def _session_window(utc_time: datetime) -> bool:
    return SESSION_START_HOUR <= utc_time.hour < SESSION_END_HOUR


class MomentumDivergenceSession(BaseStrategy):

    NAME = "momentum_divergence_session"
    SESSION = "london_ny_overlap"
    COMPATIBLE_PAIRS = ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"]

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
                f"MomentumDivergenceSession config for {pair_config.get('pair', '?')} "
                f"missing required key(s): {missing}"
            )

    def get_session_windows(self) -> list:
        return [
            {'name': 'london_ny_overlap',
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

        result = self.check_divergence(session_data, session)
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

    def _h1_bars(self, count: int = H1_BARS_NEEDED):
        """Most recent CLOSED H1 bars (pos=1 skips the still-forming bar)."""
        rates = mt5.copy_rates_from_pos(self.pair, mt5.TIMEFRAME_H1, 1, count)
        if rates is None or len(rates) < H1_BARS_NEEDED:
            return None
        return rates

    # ------------------------------------------------------------------
    # RSI (Wilder's smoothing -- same as every backtest in this codebase)
    # ------------------------------------------------------------------

    @staticmethod
    def _rsi_series(closes: np.ndarray, period: int) -> np.ndarray:
        """
        Full RSI series using Wilder's EMA smoothing.
        Output length = len(closes) - 1 (one value per delta).
        First `period` values are NaN (seeding phase).
        """
        deltas   = np.diff(closes)
        gains    = np.clip(deltas, 0, None)
        losses   = -np.clip(deltas, None, 0)
        out      = np.full(len(deltas), np.nan)
        if len(deltas) < period:
            return out
        avg_gain = float(np.mean(gains[:period]))
        avg_loss = float(np.mean(losses[:period]))
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            rs       = avg_gain / avg_loss if avg_loss > 0 else np.inf
            out[i]   = 100.0 - (100.0 / (1.0 + rs))
        return out

    # ------------------------------------------------------------------
    # Swing detection
    # ------------------------------------------------------------------

    @staticmethod
    def _swing_highs(highs: np.ndarray) -> list:
        """Indices where high exceeds both immediate neighbours."""
        return [i for i in range(1, len(highs) - 1)
                if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]]

    @staticmethod
    def _swing_lows(lows: np.ndarray) -> list:
        """Indices where low is below both immediate neighbours."""
        return [i for i in range(1, len(lows) - 1)
                if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]]

    # ------------------------------------------------------------------
    # prepare() -- H1 bar snapshot, called once per session
    # ------------------------------------------------------------------

    def prepare(self, log) -> dict | None:
        if not self._connect(log):
            return None

        bars = self._h1_bars()
        if bars is None:
            log.info(f"  {self.pair}: insufficient H1 history "
                     f"({H1_BARS_NEEDED} bars needed)")
            return None

        closes = np.array([b['close'] for b in bars])
        highs  = np.array([b['high']  for b in bars])
        lows   = np.array([b['low']   for b in bars])
        rsi    = self._rsi_series(closes, RSI_PERIOD)

        # Trim NaN from RSI seed phase before logging
        valid_rsi = rsi[~np.isnan(rsi)]
        rsi_now   = float(valid_rsi[-1]) if len(valid_rsi) > 0 else float('nan')

        log.info(f"  {self.pair}: RSI={rsi_now:.1f}  close={closes[-1]:.5f}  "
                 f"swing_lookback={SWING_LOOKBACK}bars")

        return {
            'closes' : closes.tolist(),
            'highs'  : highs.tolist(),
            'lows'   : lows.tolist(),
            'rsi'    : rsi.tolist(),
        }

    # ------------------------------------------------------------------
    # check_divergence() -- swing comparison, called each 15-minute cycle
    # ------------------------------------------------------------------

    def check_divergence(self, session_data: dict, session: str) -> dict:
        """
        session_data : this pair's own data dict (closes, highs, lows, rsi)
                        -- already scoped to self.pair by the caller.
        """
        log = _log()

        no_signal = lambda reason: {
            'signal'      : 'NO_SIGNAL',
            'reason'      : reason,
            'entry_price' : 0.0,
        }

        if not session_data:
            return no_signal('pair not in session data')

        if not self._connect(log):
            return no_signal('MT5 connection failed')

        rates = mt5.copy_rates_from_pos(self.pair, mt5.TIMEFRAME_H1, 1, 1)
        if rates is None or len(rates) == 0:
            return no_signal('no H1 bar data')

        bar_time = datetime.fromtimestamp(rates[0]['time'], tz=timezone.utc)

        if not _session_window(bar_time):
            return no_signal(
                f'bar at {bar_time.hour:02d}:00 UTC outside London/NY overlap')

        if self._last_trade_date == bar_time.date():
            return no_signal('already traded this pair today (one trade/day limit)')

        # Use the most recent SWING_LOOKBACK bars from the session snapshot
        highs = np.array(session_data['highs'])[-SWING_LOOKBACK:]
        lows  = np.array(session_data['lows'])[-SWING_LOOKBACK:]
        rsi   = np.array(session_data['rsi'])[-SWING_LOOKBACK:]

        sh_idx = self._swing_highs(highs)
        sl_idx = self._swing_lows(lows)

        # ---- Bearish divergence: price higher high, RSI lower high ----
        if len(sh_idx) >= 2:
            i1, i2 = sh_idx[-2], sh_idx[-1]
            if (i2 - i1 >= MIN_SWING_GAP
                    and highs[i2] > highs[i1]
                    and not np.isnan(rsi[i1]) and not np.isnan(rsi[i2])
                    and rsi[i1] > RSI_EXTREME_HIGH   # first peak must have been extreme
                    and rsi[i2] < rsi[i1]):
                self._last_trade_date = bar_time.date()
                return {
                    'signal'      : 'SELL',
                    'reason'      : (f'bearish divergence: '
                                     f'price HH {highs[i1]:.5f}->{highs[i2]:.5f}  '
                                     f'RSI LH {rsi[i1]:.1f}->{rsi[i2]:.1f}'),
                    'entry_price' : float(rates[0]['close']),
                }

        # ---- Bullish divergence: price lower low, RSI higher low -----
        if len(sl_idx) >= 2:
            i1, i2 = sl_idx[-2], sl_idx[-1]
            if (i2 - i1 >= MIN_SWING_GAP
                    and lows[i2] < lows[i1]
                    and not np.isnan(rsi[i1]) and not np.isnan(rsi[i2])
                    and rsi[i1] < RSI_EXTREME_LOW    # first trough must have been extreme
                    and rsi[i2] > rsi[i1]):
                self._last_trade_date = bar_time.date()
                return {
                    'signal'      : 'BUY',
                    'reason'      : (f'bullish divergence: '
                                     f'price LL {lows[i1]:.5f}->{lows[i2]:.5f}  '
                                     f'RSI HL {rsi[i1]:.1f}->{rsi[i2]:.1f}'),
                    'entry_price' : float(rates[0]['close']),
                }

        rsi_now = float(rsi[-1]) if not np.isnan(rsi[-1]) else float('nan')
        return no_signal(
            f'no divergence in last {SWING_LOOKBACK} bars '
            f'(RSI={rsi_now:.1f}  sh={len(sh_idx)}  sl={len(sl_idx)})')
