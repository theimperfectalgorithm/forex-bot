"""
H4TrendPullback -- H4 trend filter, pullback-and-reclaim entry on the H1
50 EMA, session-filtered to London/NY hours.

VALIDATION STATUS: FAILED. See src/h4_trend_pullback_backtest.py for the
full grid search / walk-forward / forward-test report on GBPJPY H1,
2020-2024. Summary:
  - ALL 81 training-grid configs (2020-01 to 2022-06) were net LOSERS
    (profit factor < 1.0) -- this isn't a narrow miss on the pass
    criteria, the raw signal has negative edge across the entire grid.
    Best training config: EMA=20, depth=15p, H4_thr=30p, SL_lookback=5
    bars -- win rate 36.2%, profit factor 0.83, max DD 8.9%, P&L -$728.55
    over 412 trades.
  - Top 3 training configs stay net-losing on unseen 2022-07 to 2024-12
    forward data (profit factor 0.89 / 0.93 / 0.89), so there is no
    train/forward gap to speak of -- the strategy simply doesn't work in
    either period.
  - SPECIFIC FAILURE MODE (from the exit-reason breakdown of the #1
    config on training data, 412 trades): 213 trades (52%) hit the full
    stop loss (avg -$17.76), 112 trades (27%) got force-closed at the
    17:30 EOD cutoff averaging only +$3.65 (essentially breakeven --
    barely any follow-through), and just 87 trades (21%) reached the
    full 2:1 take-profit (avg +$30.40). The pullback-and-reclaim pattern
    does not have enough room to run 2x its stop distance before EOD
    closes it out -- losers get cut in full, winners get cut short.
  - Walk-forward across 6 equal sub-periods of the full 2020-2024 range:
    4 of 6 periods net negative for the #1 config -- no consistent edge
    through time either.
  - Correlation vs LondonBreakout on the same GBPJPY data: weakly
    positive (Pearson r = 0.19 on daily P&L, 67% win/loss agreement on
    days both traded) -- not that it matters much here, since this
    strategy loses money on its own regardless of what it's paired with.
  - Conclusion: NOT validated for live trading in this form. A fix would
    need to either loosen the EOD cutoff (give winners more room) or
    tighten the TP multiple to something the EOD window can realistically
    reach -- both are logic changes outside the scope of this backtest.

Strategy logic (parameters below are the best-ranked TRAINING config --
documented for reproducibility, not as a recommendation to trade them):
  1. H4 trend: SMA50 vs SMA200 on H4 -- bullish if 50 above 200, bearish
     if 50 below 200, AND the two SMAs must be separated by at least
     H4_THRESHOLD_PIPS (default 30p), otherwise neutral (no trade). Only
     H4 bars that have fully closed before the signal bar are used.
  2. Pullback + reclaim on H1, using an EMA(EMA_PERIOD) (default 20):
       bullish trend -> BUY  when a bar's Low  <= EMA + PULLBACK_DEPTH_PIPS
                         AND that same bar's Close  > EMA
       bearish trend -> SELL when a bar's High >= EMA - PULLBACK_DEPTH_PIPS
                         AND that same bar's Close  < EMA
  3. Session filter: only bars in SESSION_START_HOUR-SESSION_END_HOUR UTC
     (08:00-21:00) are eligible signal bars; combined with the 17:30 EOD
     close, entries are only scanned through the bar before EOD_HOUR (see
     ENTRY_CUTOFF_HOUR).
  4. SL = the swing low (BUY) / swing high (SELL) over the last
     SL_LOOKBACK H1 bars (default 5), ending at the signal bar.
     TP = RISK_REWARD x the SL distance (default 2:1).
  5. EOD close: Friday-only, at 20:00 UTC (handled by the orchestrator,
     not this class -- see main_agent.py step_friday_close(), same as
     LondonBreakout. Mon-Thu, positions run to natural SL/TP. NOTE: this
     strategy's own EMA/ENTRY_CUTOFF_HOUR parameters below were validated
     against the OLD daily-17:30-close backtest -- see VALIDATION STATUS
     above; they were not re-validated against the new Friday-only close
     and this strategy remains FAILED/inactive either way.)
  6. Only one trade per day per pair.
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
H4_SMA_FAST          = 50
H4_SMA_SLOW          = 200
H4_BARS_NEEDED       = 220
H4_BAR_HOURS         = 4        # H4 bar duration, for "fully closed" checks
H4_THRESHOLD_PIPS    = 30       # min |SMA50-SMA200| separation required

EMA_PERIOD           = 20
PULLBACK_DEPTH_PIPS  = 15       # how close a wick must get to the EMA to "touch" it
SL_LOOKBACK          = 5        # H1 bars for the swing-low/high SL

SESSION_START_HOUR   = 8        # London/NY session filter: 08:00-21:00 UTC
SESSION_END_HOUR     = 21
EOD_HOUR             = 17       # 17:00-18:00 H1 bar approximates the 17:30 EOD close
ENTRY_CUTOFF_HOUR    = EOD_HOUR # entries scanned only through the bar before EOD

RISK_REWARD          = 2.0      # TP = RISK_REWARD x SL distance

REQUIRED_KEYS = ['pair', 'strategy', 'active', 'timeframe', 'risk_percent',
                  'h4_filter', 'session', 'friday_close']


def _session_window(utc_time: datetime) -> bool:
    """London/NY session filter, capped before EOD (see ENTRY_CUTOFF_HOUR)."""
    return SESSION_START_HOUR <= utc_time.hour < ENTRY_CUTOFF_HOUR


class H4TrendPullback(BaseStrategy):

    NAME = "h4_trend_pullback"
    SESSION = "london_ny"
    COMPATIBLE_PAIRS = ["GBPJPY", "EURJPY", "EURUSD"]

    def __init__(self, pair_config: dict):
        super().__init__(pair_config)
        self.validate_config(pair_config)
        self.pip_size = 0.01 if 'JPY' in self.pair else 0.0001
        self._h1_history: list = []   # rolling H1 bars, oldest-first, for swing SL lookback
        self._last_trade_date = None  # enforces one trade per day per pair

    # ------------------------------------------------------------------
    # BaseStrategy required interface
    # ------------------------------------------------------------------

    def validate_config(self, pair_config: dict) -> None:
        missing = [k for k in REQUIRED_KEYS if k not in pair_config]
        if missing:
            raise ValueError(
                f"H4TrendPullback config for {pair_config.get('pair', '?')} "
                f"missing required key(s): {missing}"
            )

    def get_session_windows(self) -> list:
        return [
            {'name': 'london_ny', 'start': f'{SESSION_START_HOUR:02d}:00',
                                   'end': f'{SESSION_END_HOUR:02d}:00'},
        ]

    def generate_signal(self, pair: str, current_price: float, h4_trend: int,
                         **context):
        """
        Required interface method. With full context (session_data +
        session kwargs, as supplied in production by agent_strategy.py),
        delegates to check_pullback() and returns 'BUY' / 'SELL' / None.

        Called with only the 3 base args (e.g. a smoke test with mock
        data), there is no H1 bar history available to detect a real
        pullback, so it safely returns None rather than hitting MT5.
        """
        session_data = context.get('session_data')
        session      = context.get('session')
        if session_data is None or session is None:
            return None

        result = self.check_pullback(session_data, session)
        if result['signal'] in ('BUY', 'SELL'):
            return result['signal']
        return None

    def calculate_sl(self, entry_price: float, direction: str, pair_config: dict) -> float:
        sl_pips = pair_config.get('sl_pips')
        if sl_pips is None:
            raise ValueError(
                f"H4TrendPullback.calculate_sl: pair_config for {self.pair} has no "
                f"'sl_pips' -- SL is derived from the swing low/high at "
                f"check_pullback() time, not a static YAML value"
            )
        dist = sl_pips * self.pip_size
        return entry_price - dist if direction == 'BUY' else entry_price + dist

    def calculate_tp(self, entry_price: float, direction: str, pair_config: dict) -> float:
        tp_pips = pair_config.get('tp_pips')
        if tp_pips is None:
            raise ValueError(
                f"H4TrendPullback.calculate_tp: pair_config for {self.pair} has no "
                f"'tp_pips' -- TP is derived from the swing low/high at "
                f"check_pullback() time, not a static YAML value"
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

    def _h1_bars(self, count: int):
        rates = mt5.copy_rates_from_pos(self.pair, mt5.TIMEFRAME_H1, 0, count)
        if rates is None or len(rates) < count:
            return None
        return rates

    def _last_closed_h1(self):
        rates = mt5.copy_rates_from_pos(self.pair, mt5.TIMEFRAME_H1, 1, 1)
        if rates is None or len(rates) == 0:
            return None
        return rates[0]

    # ------------------------------------------------------------------
    # H4 trend (pure SMA50-vs-SMA200 sign, plus a minimum separation --
    # see module docstring)
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
        diff_pips = abs(sma50 - sma200) / self.pip_size

        if diff_pips < H4_THRESHOLD_PIPS:
            return 0
        if sma50 > sma200:
            return 1
        if sma50 < sma200:
            return -1
        return 0

    # ------------------------------------------------------------------
    # Pullback + reclaim detection on H1 (EMA touch + close-back)
    # ------------------------------------------------------------------

    def check_pullback(self, session_data: dict, session: str) -> dict:
        """
        session_data : this pair's own data dict ({'h4_trend': int}) --
                        already scoped to self.pair by the caller.
        """
        log = _log()

        no_signal = lambda reason: {
            'signal'       : 'NO_SIGNAL',
            'reason'       : reason,
            'entry_price'  : 0.0,
            'sl_price'     : 0.0,
            'tp_price'     : 0.0,
        }

        if not session_data:
            return no_signal('pair not in session data (neutral H4 trend)')

        trend = session_data['h4_trend']
        if trend == 0:
            return no_signal('neutral H4 trend')

        if not self._connect(log):
            return no_signal('MT5 connection failed')

        bars = self._h1_bars(SL_LOOKBACK + 1)
        if bars is None:
            return no_signal('not enough H1 history for swing SL lookback')

        bar = bars[-1]
        bar_time = datetime.fromtimestamp(bar['time'], tz=timezone.utc)

        if not _session_window(bar_time):
            return no_signal(f'bar at {bar_time.hour:02d}:00 UTC outside session window')

        if self._last_trade_date == bar_time.date():
            return no_signal('already traded this pair today (one trade/day limit)')

        closes = np.array([b['close'] for b in bars])
        ema = closes[0]
        alpha = 2.0 / (EMA_PERIOD + 1)
        for c in closes[1:]:
            ema = alpha * c + (1 - alpha) * ema

        low, high, close = bar['low'], bar['high'], bar['close']

        if trend == 1:
            touched   = low <= ema + PULLBACK_DEPTH_PIPS * self.pip_size
            reclaimed = close > ema
            if not (touched and reclaimed):
                return no_signal('no bullish pullback+reclaim of the H1 EMA')
            direction = 'BUY'
        else:
            touched   = high >= ema - PULLBACK_DEPTH_PIPS * self.pip_size
            reclaimed = close < ema
            if not (touched and reclaimed):
                return no_signal('no bearish pullback+reclaim of the H1 EMA')
            direction = 'SELL'

        entry = close
        swing_lows  = [b['low']  for b in bars[:-1]]
        swing_highs = [b['high'] for b in bars[:-1]]

        if direction == 'BUY':
            sl_price = min(swing_lows) if swing_lows else None
            if sl_price is None or sl_price >= entry:
                return no_signal('degenerate swing-low SL (>= entry)')
            tp_price = entry + RISK_REWARD * (entry - sl_price)
        else:
            sl_price = max(swing_highs) if swing_highs else None
            if sl_price is None or sl_price <= entry:
                return no_signal('degenerate swing-high SL (<= entry)')
            tp_price = entry - RISK_REWARD * (sl_price - entry)

        self._last_trade_date = bar_time.date()
        trend_label = 'BULLISH' if trend == 1 else 'BEARISH'
        return {
            'signal'      : direction,
            'reason'      : f'H1 EMA{EMA_PERIOD} pullback+reclaim, H4 {trend_label}',
            'entry_price' : entry,
            'sl_price'    : sl_price,
            'tp_price'    : tp_price,
        }
