"""
AsianRangeBreakout -- Asian session range breakout traded at the
Tokyo/London overlap (07:00-08:30 UTC), with an H4 50/200 SMA trend
filter. Structurally close to LondonBreakout, just with the breakout
window shifted to immediately follow the Asian range itself instead of
London/NY hours.

Strategy logic:
  1. H4 trend filter: SMA50 vs SMA200 on H4 -- bullish if 50 above 200,
     bearish if 50 below 200 (pure SMA-sign test; matches
     H4TrendPullback's convention). Neutral = no trade. Only H4 bars
     that have fully closed before the signal bar are used.
  2. Asian session range: High and Low of H1 bars from 00:00-07:00 UTC.
  3. Breakout at the Tokyo/London overlap: the first H1 bar in
     OVERLAP_START_HOUR-OVERLAP_END_HOUR UTC (07:00-08:30, approximated
     at H1 resolution as bars with hour in {7, 8}) whose close moves
     beyond the Asian range, in the H4 trend direction only.
  4. SL = the opposite side of the Asian range (full range distance).
     TP = TP_MULTIPLIER x the Asian range distance (default 1.5x, so
     1.5:1 reward:risk).
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
H4_SMA_FAST          = 50
H4_SMA_SLOW          = 200
H4_BARS_NEEDED       = 220
H4_BAR_HOURS         = 4        # H4 bar duration, for "fully closed" checks

ASIAN_END_HOUR       = 7        # 00:00-07:00 UTC Asian range
OVERLAP_START_HOUR   = 7        # Tokyo/London overlap breakout window
OVERLAP_END_HOUR     = 9        # H1-resolution approximation of 08:30
                                 # (bars with hour in {7, 8})
MIN_ASIAN_RANGE_PIPS = 10       # skip degenerate near-zero ranges

TP_MULTIPLIER        = 1.5      # default TP = TP_MULTIPLIER x Asian range
                                 # distance; override per pair via the YAML
                                 # `tp_multiplier` key (the GBPJPY-validated
                                 # variant uses 2.0 -- see
                                 # pairs/GBPJPY_asianrange.yaml)

REQUIRED_KEYS = ['pair', 'strategy', 'active', 'timeframe', 'risk_percent',
                  'h4_filter', 'session', 'friday_close']


class AsianRangeBreakout(BaseStrategy):

    NAME = "asian_range_breakout"
    SESSION = "asian"
    # GBPJPY added 2026-07-04 (walk-forward validated, tp 2.0 / no H4);
    # CADJPY added 2026-07-05 (phase 6: IS PF 1.15 / OOS 1.38);
    # XAUUSD added 2026-07-05 (phase 7, PROVISIONAL: IS PF 1.45, OOS flat)
    COMPATIBLE_PAIRS = ["USDJPY", "AUDJPY", "NZDJPY", "AUDUSD", "GBPJPY",
                        "CADJPY", "XAUUSD"]

    def __init__(self, pair_config: dict):
        super().__init__(pair_config)
        self.validate_config(pair_config)
        self.pip_size = (0.1 if self.pair.startswith('XAU')
                         else 0.01 if 'JPY' in self.pair else 0.0001)
        self._last_trade_date = None  # same-day dedup (see check_breakout below) --
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
                f"AsianRangeBreakout config for {pair_config.get('pair', '?')} "
                f"missing required key(s): {missing}"
            )

    def get_session_windows(self) -> list:
        return [
            {'name': 'asian',   'start': '00:00', 'end': f'{ASIAN_END_HOUR:02d}:00'},
            {'name': 'overlap', 'start': f'{OVERLAP_START_HOUR:02d}:00', 'end': '08:30'},
        ]

    def generate_signal(self, pair: str, current_price: float, h4_trend: int,
                         **context):
        """
        Required interface method. With full context (session_data +
        session kwargs, as supplied in production by agent_strategy.py),
        delegates to check_breakout() and returns 'BUY' / 'SELL' / None.

        Called with only the 3 base args (e.g. a smoke test with mock
        data), there is no Asian range / session context available to
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
                f"AsianRangeBreakout.calculate_sl: pair_config for {self.pair} has no "
                f"'sl_pips' -- breakout SL is derived from the Asian range at "
                f"prepare() time, not a static YAML value"
            )
        dist = sl_pips * self.pip_size
        return entry_price - dist if direction == 'BUY' else entry_price + dist

    def calculate_tp(self, entry_price: float, direction: str, pair_config: dict) -> float:
        tp_pips = pair_config.get('tp_pips')
        if tp_pips is None:
            raise ValueError(
                f"AsianRangeBreakout.calculate_tp: pair_config for {self.pair} has no "
                f"'tp_pips' -- breakout TP is derived from the Asian range at "
                f"prepare() time, not a static YAML value"
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
    # H4 trend (pure SMA50-vs-SMA200 sign -- matches H4TrendPullback)
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

        if sma50 > sma200:
            return 1
        if sma50 < sma200:
            return -1
        return 0

    # ------------------------------------------------------------------
    # Asian range (High/Low of H1 bars, 00:00-07:00 UTC)
    # ------------------------------------------------------------------

    def asian_range(self, log) -> dict | None:
        bars = self._h1_bars_today()
        if bars is None:
            log.warning(f"{self.pair}: no H1 data for today")
            return None

        asian_bars = [b for b in bars if
                      datetime.fromtimestamp(b['time'], tz=timezone.utc).hour < ASIAN_END_HOUR]

        if len(asian_bars) < 2:
            log.warning(f"{self.pair}: insufficient Asian session bars ({len(asian_bars)})")
            return None

        high = max(b['high'] for b in asian_bars)
        low  = min(b['low']  for b in asian_bars)
        range_pips = (high - low) / self.pip_size

        min_range = self.pair_config.get('min_range_pips', MIN_ASIAN_RANGE_PIPS)
        if range_pips < min_range:
            log.info(f"{self.pair}: Asian range too tight ({range_pips:.1f}p < "
                     f"{min_range}p)")
            return None

        tp_mult = self.pair_config.get('tp_multiplier', TP_MULTIPLIER)
        sl_pips = range_pips                    # opposite side of the range
        tp_pips = range_pips * tp_mult

        return {
            'asian_high' : high,
            'asian_low'  : low,
            'range_pips' : round(range_pips, 1),
            'sl_pips'    : round(sl_pips, 1),
            'tp_pips'    : round(tp_pips, 1),
        }

    # ------------------------------------------------------------------
    # prepare() -- combines H4 trend + Asian range, called once per
    # session (mirrors LondonBreakout.prepare())
    # ------------------------------------------------------------------

    def prepare(self, log) -> dict | None:
        if not self._connect(log):
            return None

        # h4_filter: false in the pair YAML disables the trend gate --
        # breakouts are then taken in EITHER direction. The walk-forward
        # search found the H4 gate reduced GBPJPY performance (it lags the
        # early-London move this strategy trades); other pairs keep it on.
        use_h4 = self.pair_config.get('h4_filter', True)
        trend  = self.h4_trend(log) if use_h4 else 0
        if use_h4 and trend == 0:
            log.info(f"  {self.pair}: NEUTRAL H4 trend -- skipping")
            return None

        asian = self.asian_range(log)
        if asian is None:
            log.info(f"  {self.pair}: Asian range unavailable -- skipping")
            return None

        result = {'h4_trend': trend, 'h4_filter': use_h4, **asian}
        trend_label = ('BULLISH' if trend == 1
                       else 'BEARISH' if trend == -1 else 'ANY (no H4 gate)')
        log.info(f"  {self.pair}: {trend_label}  range={asian['range_pips']:.1f}p  "
                 f"H={asian['asian_high']:.5f}  L={asian['asian_low']:.5f}  "
                 f"SL={asian['sl_pips']:.1f}p  TP={asian['tp_pips']:.1f}p")
        return result

    # ------------------------------------------------------------------
    # check_breakout() -- Tokyo/London overlap breakout of the Asian range
    # ------------------------------------------------------------------

    def check_breakout(self, session_data: dict, session: str) -> dict:
        """
        session_data : this pair's own data dict (h4_trend, asian_high,
                        asian_low, range_pips, sl_pips, tp_pips) --
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

        asian_high = session_data['asian_high']
        asian_low  = session_data['asian_low']
        trend      = session_data['h4_trend']
        use_h4     = session_data.get('h4_filter', True)

        if not self._connect(log):
            return no_signal('MT5 connection failed')

        bar = self._last_closed_h1()
        if bar is None:
            return no_signal('no H1 bar data')

        bar_time  = datetime.fromtimestamp(bar['time'], tz=timezone.utc)
        bar_close = bar['close']

        if not (OVERLAP_START_HOUR <= bar_time.hour < OVERLAP_END_HOUR):
            return no_signal(f'bar at {bar_time.hour:02d}:00 UTC outside Tokyo/London '
                             f'overlap window')

        if self._last_trade_date == bar_time.date():
            return no_signal('already traded this pair today (one trade/day limit)')

        if bar_close > asian_high and (not use_h4 or trend == 1):
            return {
                'signal'            : 'BUY',
                'reason'            : ('close above Asian High, H4 BULLISH'
                                       if use_h4 else
                                       'close above Asian High (no H4 gate)'),
                'trigger_bar_close' : bar_close,
                'trigger_bar_time'  : bar_time.isoformat(),
                'entry_price'       : asian_high,
            }

        if bar_close < asian_low and (not use_h4 or trend == -1):
            return {
                'signal'            : 'SELL',
                'reason'            : ('close below Asian Low, H4 BEARISH'
                                       if use_h4 else
                                       'close below Asian Low (no H4 gate)'),
                'trigger_bar_close' : bar_close,
                'trigger_bar_time'  : bar_time.isoformat(),
                'entry_price'       : asian_low,
            }

        trend_label = ('BULLISH' if trend == 1
                       else 'BEARISH' if trend == -1 else 'either-direction')
        return no_signal(f'no {trend_label} close beyond range on bar {bar_time.strftime("%H:%M")}')

    def acknowledge_trade(self, signal: dict) -> None:
        """Commit same-day dedup only after the broker accepts the entry."""
        if signal.get('signal') not in ('BUY', 'SELL'):
            raise ValueError('cannot acknowledge a non-entry ARB signal')
        bar_time = datetime.fromisoformat(signal['trigger_bar_time'])
        self._last_trade_date = bar_time.date()
