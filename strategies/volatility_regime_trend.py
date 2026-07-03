"""
VolatilityRegimeTrend -- NEW, UNTESTED strategy awaiting backtest validation.

Do NOT set active: true on any pairs/*.yaml until a full walk-forward
backtest has been run and the strategy passes the standard validation
criteria (profit factor > 1.3, max drawdown < 8%, >= 60% profitable
months over a 2-year in-sample period with a clean out-of-sample
forward test).

Strategy logic:
  1. Volatility filter: ATR(14) on H1 must exceed ATR_MIN_PIPS (default 8p,
     configurable per pair via the YAML `atr_min_pips` key). Entries are
     blocked in low-volatility / choppy conditions where EMA signals tend
     to whipsaw and produce frequent small losses with no clear trend.
  2. EMA alignment: EMA20 must be above EMA50 (BUY) or below EMA50 (SELL),
     AND price must be on the correct side of EMA20 (close > EMA20 for BUY,
     close < EMA20 for SELL). This is an alignment gate -- it does NOT wait
     for a cross event (see RegimeFilteredMACross for the cross-trigger
     variant). Alignment is less precise but fires more frequently,
     appropriate for a volatility-regime strategy where the ATR filter is
     the primary quality gate.
  3. Session filter: London + early NY (08:00-20:00 UTC). The Asian session
     is deliberately excluded -- EMA alignment signals in thin Asian
     conditions have historically produced false positives on low-volume
     price drift that reverses at London open.
  4. Fixed SL and TP from the pair YAML (sl_pips / tp_pips). No trailing
     stop, no cross-exit. Cross-exit rules have failed across 4 prior
     strategy variants in this codebase by killing profitable trades before
     TP was reached.
  5. Friday close at 20:00 UTC handled by the orchestrator (see
     main_agent.py step_friday_close()). Mon-Thu, positions run to natural
     SL/TP.
  6. One trade per day per pair.

Design notes:
  - The volatility filter is the key structural feature that separates
    this from a plain EMA-alignment strategy. EMA20 > EMA50 fires equally
    in strong trends AND in flat ranging markets -- the ATR gate is what
    keeps the strategy out of chop.
  - ATR_MIN_PIPS should be calibrated per pair before going live: AUDUSD
    moves less in absolute pip terms than GBPJPY, so the threshold must be
    pair-specific (set in the YAML, not hardcoded here).
  - This strategy has NOT been validated. The structural idea is sound, but
    the specific parameter values (EMA periods 20/50, ATR period 14,
    threshold 8p) are starting-point estimates only -- backtest before use.
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
H1_EMA_FAST  = 20
H1_EMA_SLOW  = 50
ATR_PERIOD   = 14
ATR_MIN_PIPS = 8         # default; override per pair via YAML `atr_min_pips`

H1_BARS_NEEDED = H1_EMA_SLOW + ATR_PERIOD + 2

SESSION_START_HOUR = 8   # London open
SESSION_END_HOUR   = 20  # late NY; Asian session excluded

REQUIRED_KEYS = ['pair', 'strategy', 'active', 'timeframe',
                 'risk_percent', 'sl_pips', 'tp_pips', 'session', 'friday_close']


def _session_window(utc_time: datetime) -> bool:
    return SESSION_START_HOUR <= utc_time.hour < SESSION_END_HOUR


class VolatilityRegimeTrend(BaseStrategy):

    NAME = "volatility_regime_trend"
    SESSION = "london_ny"
    COMPATIBLE_PAIRS = ["GBPUSD", "EURUSD", "GBPJPY", "EURJPY", "AUDUSD"]

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
                f"VolatilityRegimeTrend config for {pair_config.get('pair', '?')} "
                f"missing required key(s): {missing}"
            )

    def get_session_windows(self) -> list:
        return [
            {'name': 'london_ny',
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

        result = self.check_signal(session_data, session)
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
    # EMA and ATR calculations
    # ------------------------------------------------------------------

    @staticmethod
    def _ema_series(values: np.ndarray, period: int) -> np.ndarray:
        alpha = 2.0 / (period + 1)
        out   = np.empty(len(values))
        out[0] = values[0]
        for i in range(1, len(values)):
            out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
        return out

    @staticmethod
    def _atr_wilder(bars, period: int) -> float:
        """Wilder's ATR -- same smoothing as every backtest in this codebase."""
        trs = []
        for i in range(1, len(bars)):
            h, l, pc = bars[i]['high'], bars[i]['low'], bars[i - 1]['close']
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        if len(trs) < period:
            return 0.0
        atr = float(np.mean(trs[:period]))
        for tr in trs[period:]:
            atr = (atr * (period - 1) + tr) / period
        return atr

    # ------------------------------------------------------------------
    # prepare() -- ATR + EMA snapshot, called once per session
    # ------------------------------------------------------------------

    def prepare(self, log) -> dict | None:
        if not self._connect(log):
            return None

        bars = self._h1_bars()
        if bars is None:
            log.info(f"  {self.pair}: insufficient H1 history "
                     f"({H1_BARS_NEEDED} bars needed)")
            return None

        closes   = np.array([b['close'] for b in bars])
        ema_fast = self._ema_series(closes, H1_EMA_FAST)
        ema_slow = self._ema_series(closes, H1_EMA_SLOW)
        atr      = self._atr_wilder(bars, ATR_PERIOD)
        atr_pips = atr / self.pip_size
        atr_min  = self.pair_config.get('atr_min_pips', ATR_MIN_PIPS)

        log.info(f"  {self.pair}: ATR={atr_pips:.1f}p (min={atr_min}p)  "
                 f"EMA{H1_EMA_FAST}={ema_fast[-1]:.5f}  "
                 f"EMA{H1_EMA_SLOW}={ema_slow[-1]:.5f}  "
                 f"close={closes[-1]:.5f}")

        if atr_pips < atr_min:
            log.info(f"  {self.pair}: ATR {atr_pips:.1f}p below "
                     f"threshold {atr_min}p -- skipping (low volatility)")
            return None

        return {
            'ema_fast' : float(ema_fast[-1]),
            'ema_slow' : float(ema_slow[-1]),
            'atr_pips' : round(atr_pips, 1),
            'close'    : float(closes[-1]),
        }

    # ------------------------------------------------------------------
    # check_signal() -- EMA alignment gate, called each 15-minute cycle
    # ------------------------------------------------------------------

    def check_signal(self, session_data: dict, session: str) -> dict:
        """
        session_data : this pair's own data dict (ema_fast, ema_slow,
                        atr_pips, close) -- already scoped to self.pair
                        by the caller.
        """
        log = _log()

        no_signal = lambda reason: {
            'signal'      : 'NO_SIGNAL',
            'reason'      : reason,
            'entry_price' : 0.0,
        }

        if not session_data:
            return no_signal(
                'pair not in session data (ATR below threshold or no H1 history)')

        ema_fast = session_data['ema_fast']
        ema_slow = session_data['ema_slow']
        close    = session_data['close']

        if not self._connect(log):
            return no_signal('MT5 connection failed')

        rates = mt5.copy_rates_from_pos(self.pair, mt5.TIMEFRAME_H1, 1, 1)
        if rates is None or len(rates) == 0:
            return no_signal('no H1 bar data')

        bar_time = datetime.fromtimestamp(rates[0]['time'], tz=timezone.utc)

        if not _session_window(bar_time):
            return no_signal(
                f'bar at {bar_time.hour:02d}:00 UTC outside London/NY window')

        if self._last_trade_date == bar_time.date():
            return no_signal('already traded this pair today (one trade/day limit)')

        if ema_fast > ema_slow and close > ema_fast:
            self._last_trade_date = bar_time.date()
            return {
                'signal'      : 'BUY',
                'reason'      : (f'ATR={session_data["atr_pips"]:.1f}p  '
                                 f'EMA{H1_EMA_FAST}({ema_fast:.5f}) > '
                                 f'EMA{H1_EMA_SLOW}({ema_slow:.5f})  '
                                 f'close({close:.5f}) > EMA{H1_EMA_FAST}'),
                'entry_price' : close,
            }

        if ema_fast < ema_slow and close < ema_fast:
            self._last_trade_date = bar_time.date()
            return {
                'signal'      : 'SELL',
                'reason'      : (f'ATR={session_data["atr_pips"]:.1f}p  '
                                 f'EMA{H1_EMA_FAST}({ema_fast:.5f}) < '
                                 f'EMA{H1_EMA_SLOW}({ema_slow:.5f})  '
                                 f'close({close:.5f}) < EMA{H1_EMA_FAST}'),
                'entry_price' : close,
            }

        return no_signal(
            f'EMA alignment not met: '
            f'EMA{H1_EMA_FAST}={ema_fast:.5f}  '
            f'EMA{H1_EMA_SLOW}={ema_slow:.5f}  '
            f'close={close:.5f}')
