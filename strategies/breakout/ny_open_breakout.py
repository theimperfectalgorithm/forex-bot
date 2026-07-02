"""
NyOpenBreakout -- London-range breakout traded at NY open (13:00 UTC), with
an H4 50/200 SMA trend filter (plus a minimum SMA-separation threshold so
flat/choppy H4 charts don't count as "trending").

VALIDATION STATUS: FAILED. See src/ny_open_breakout_backtest.py for the
full grid search / walk-forward / forward-test report on EURUSD H1,
2020-2024. Summary:
  - 0 of 27 training-grid configs (2020-01 to 2022-06) passed the pass
    criteria (win rate 40-50%, max DD < 4%, profit factor > 1.2, >= 60
    trades, >= 60% profitable months) -- actual win rates cluster at
    50-55% and profitable-months sits around 48-57%.
  - The top 3 training configs (ranked by profit factor desc, max DD asc)
    all decay out-of-sample on unseen 2022-07 to 2024-12 data: profit
    factor drops from 1.43/1.38/1.36 (train) to 1.14/1.01/1.17 (forward),
    and max drawdown crosses the 4% limit on every one of them.
  - Walk-forward across 6 equal sub-periods of the full 2020-2024 range
    shows 3 of 6 periods net negative for all three configs -- the edge
    is not consistent through time.
  - Conclusion: whatever edge appears in training does not carry forward
    cleanly. NOT validated for live trading in this form -- do not set
    `active: true` on a pairs/*.yaml using this strategy without
    re-running the backtest after any logic change.

Strategy logic (parameters below are the best-ranked TRAINING config from
the backtest above -- documented for reproducibility, not as a
recommendation to trade them):
  1. H4 trend filter: Close > SMA50 > SMA200 (BULLISH, +1) or
     Close < SMA50 < SMA200 (BEARISH, -1), AND |SMA50-SMA200| must be at
     least H4_THRESHOLD_PIPS -- otherwise NEUTRAL (0, no trade). Only H4
     bars that have fully closed before NY open are used.
  2. London range: High/Low of H1 bars from LONDON_RANGE_START_HOUR to
     LONDON_RANGE_END_HOUR UTC (default 08:00-12:00).
  3. NY-open breakout: the first H1 bar in the 13:00-14:45 UTC window
     (NY_ENTRY_START_HOUR to NY_ENTRY_END_HOUR) whose High/Low closes
     beyond the London range, in the H4 trend direction only.
  4. SL = SL_MULTIPLIER x London range beyond entry (default 2.0x).
     TP = RISK_REWARD x SL distance (default 2:1).
  5. EOD close: Friday-only, at 20:00 UTC (handled by the orchestrator,
     not this class -- see main_agent.py step_friday_close(), same as
     LondonBreakout. Mon-Thu, positions run to natural SL/TP.)
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

# -- strategy parameters (best-ranked TRAINING config -- NOT validated, see
#    module docstring)
H4_SMA_FAST             = 50
H4_SMA_SLOW             = 200
H4_BARS_NEEDED          = 220
H4_BAR_HOURS            = 4        # H4 bar duration, for "fully closed" checks
H4_THRESHOLD_PIPS       = 30       # min |SMA50-SMA200| separation required

LONDON_RANGE_START_HOUR = 8        # 08:00-12:00 UTC London range window
LONDON_RANGE_END_HOUR   = 12
MIN_LONDON_RANGE_PIPS   = 5        # skip degenerate near-zero ranges

NY_ENTRY_START_HOUR     = 13       # NY-open breakout scan window
NY_ENTRY_END_HOUR       = 15       # (H1 bars with hour in [13, 14])

SL_MULTIPLIER           = 2.0      # x London range
RISK_REWARD             = 2.0      # TP = RISK_REWARD x SL distance

REQUIRED_KEYS = ['pair', 'strategy', 'active', 'timeframe', 'risk_percent',
                  'h4_filter', 'session', 'friday_close']


def _ny_open_window(utc_time: datetime) -> bool:
    """NY-open breakout scan window (13:00-14:45 UTC, i.e. hour in [13, 15))."""
    return NY_ENTRY_START_HOUR <= utc_time.hour < NY_ENTRY_END_HOUR


class NyOpenBreakout(BaseStrategy):

    NAME = "ny_open_breakout"
    SESSION = "ny_open"
    COMPATIBLE_PAIRS = ["EURUSD", "USDJPY", "GBPUSD"]

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
                f"NyOpenBreakout config for {pair_config.get('pair', '?')} "
                f"missing required key(s): {missing}"
            )

    def get_session_windows(self) -> list:
        return [
            {'name': 'london',  'start': f'{LONDON_RANGE_START_HOUR:02d}:00',
                                 'end': f'{LONDON_RANGE_END_HOUR:02d}:00'},
            {'name': 'ny_open', 'start': f'{NY_ENTRY_START_HOUR:02d}:00',
                                 'end': '14:45'},
        ]

    def generate_signal(self, pair: str, current_price: float, h4_trend: int,
                         **context):
        """
        Required interface method. With full context (session_data +
        session kwargs, as supplied in production by agent_strategy.py),
        delegates to check_breakout() and returns 'BUY' / 'SELL' / None.

        Called with only the 3 base args (e.g. a smoke test with mock
        data), there is no London range / session context available to
        make a real breakout decision, so it safely returns None rather
        than hitting MT5.
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
                f"NyOpenBreakout.calculate_sl: pair_config for {self.pair} has no "
                f"'sl_pips' -- SL is derived from the London range at prepare() "
                f"time, not a static YAML value"
            )
        dist = sl_pips * self.pip_size
        return entry_price - dist if direction == 'BUY' else entry_price + dist

    def calculate_tp(self, entry_price: float, direction: str, pair_config: dict) -> float:
        tp_pips = pair_config.get('tp_pips')
        if tp_pips is None:
            raise ValueError(
                f"NyOpenBreakout.calculate_tp: pair_config for {self.pair} has no "
                f"'tp_pips' -- TP is derived from the London range at prepare() "
                f"time, not a static YAML value"
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

    def _h1_bars_today(self):
        now      = datetime.now(timezone.utc)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        rates = mt5.copy_rates_range(self.pair, mt5.TIMEFRAME_H1, midnight, now)
        if rates is None or len(rates) == 0:
            return None
        return rates

    def _last_closed_h1(self):
        rates = mt5.copy_rates_from_pos(self.pair, mt5.TIMEFRAME_H1, 1, 1)
        if rates is None or len(rates) == 0:
            return None
        return rates[0]

    # ------------------------------------------------------------------
    # H4 trend (mirrors LondonBreakout.h4_trend, plus a minimum SMA
    # separation threshold -- see module docstring)
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
        diff_pips = abs(sma50 - sma200) / self.pip_size

        if diff_pips < H4_THRESHOLD_PIPS:
            return 0
        if last > sma50 > sma200:
            return 1
        if last < sma50 < sma200:
            return -1
        return 0

    # ------------------------------------------------------------------
    # London range (High/Low of H1 bars in the London window)
    # ------------------------------------------------------------------

    def london_range(self, log) -> dict | None:
        bars = self._h1_bars_today()
        if bars is None:
            log.warning(f"{self.pair}: no H1 data for today")
            return None

        london_bars = [b for b in bars if
                       LONDON_RANGE_START_HOUR
                       <= datetime.fromtimestamp(b['time'], tz=timezone.utc).hour
                       < LONDON_RANGE_END_HOUR]

        if len(london_bars) < 2:
            log.warning(f"{self.pair}: insufficient London-window bars ({len(london_bars)})")
            return None

        high = max(b['high'] for b in london_bars)
        low  = min(b['low']  for b in london_bars)
        range_pips = (high - low) / self.pip_size

        if range_pips < MIN_LONDON_RANGE_PIPS:
            log.info(f"{self.pair}: London range too tight ({range_pips:.1f}p < "
                     f"{MIN_LONDON_RANGE_PIPS}p)")
            return None

        sl_pips = range_pips * SL_MULTIPLIER
        tp_pips = sl_pips * RISK_REWARD

        return {
            'london_high': high,
            'london_low' : low,
            'range_pips' : round(range_pips, 1),
            'sl_pips'    : round(sl_pips, 1),
            'tp_pips'    : round(tp_pips, 1),
        }

    # ------------------------------------------------------------------
    # prepare() -- combines H4 trend + London range, called once per
    # session (mirrors LondonBreakout.prepare())
    # ------------------------------------------------------------------

    def prepare(self, log) -> dict | None:
        if not self._connect(log):
            return None

        trend = self.h4_trend(log)
        if trend == 0:
            log.info(f"  {self.pair}: NEUTRAL H4 trend -- skipping")
            return None

        rng = self.london_range(log)
        if rng is None:
            log.info(f"  {self.pair}: London range unavailable -- skipping")
            return None

        result = {'h4_trend': trend, **rng}
        trend_label = 'BULLISH' if trend == 1 else 'BEARISH'
        log.info(f"  {self.pair}: {trend_label}  range={rng['range_pips']:.1f}p  "
                 f"H={rng['london_high']:.5f}  L={rng['london_low']:.5f}  "
                 f"SL={rng['sl_pips']:.1f}p  TP={rng['tp_pips']:.1f}p")
        return result

    # ------------------------------------------------------------------
    # check_breakout() -- NY-open breakout of the London range
    # ------------------------------------------------------------------

    def check_breakout(self, session_data: dict, session: str) -> dict:
        """
        session_data : this pair's own data dict (h4_trend, london_high,
                        london_low, range_pips, sl_pips, tp_pips) --
                        already scoped to self.pair by the caller.
        """
        log = _log()

        no_signal = lambda reason: {
            'signal'            : 'NO_SIGNAL',
            'reason'            : reason,
            'trigger_bar_close' : 0.0,
            'trigger_bar_time'  : '',
            'entry_price'       : 0.0,
        }

        if not session_data:
            return no_signal('pair not in session data (neutral trend or bad range)')

        london_high = session_data['london_high']
        london_low  = session_data['london_low']
        trend       = session_data['h4_trend']

        if not self._connect(log):
            return no_signal('MT5 connection failed')

        bar = self._last_closed_h1()
        if bar is None:
            return no_signal('no H1 bar data')

        bar_time  = datetime.fromtimestamp(bar['time'], tz=timezone.utc)
        bar_close = bar['close']

        if session == 'ny_open' and not _ny_open_window(bar_time):
            return no_signal(f'bar at {bar_time.hour:02d}:00 UTC outside NY-open window')

        if trend == 1 and bar_close > london_high:
            return {
                'signal'            : 'BUY',
                'reason'            : 'close above London High, H4 BULLISH',
                'trigger_bar_close' : bar_close,
                'trigger_bar_time'  : bar_time.isoformat(),
                'entry_price'       : london_high,
            }

        if trend == -1 and bar_close < london_low:
            return {
                'signal'            : 'SELL',
                'reason'            : 'close below London Low, H4 BEARISH',
                'trigger_bar_close' : bar_close,
                'trigger_bar_time'  : bar_time.isoformat(),
                'entry_price'       : london_low,
            }

        trend_label = 'BULLISH' if trend == 1 else 'BEARISH'
        return no_signal(f'no {trend_label} close beyond range on bar {bar_time.strftime("%H:%M")}')
