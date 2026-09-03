"""
AsianHoursReversion (AMR) -- M15 z-score mean reversion on JPY crosses
during the Asian session (00:00-07:00 UTC). DEMO FORWARD-TEST ONLY.

Validation status (2026-07-05, src/phase3_session_structure_search.py +
src/phase3b_amr_jpy_refine.py; data/phase3_results.csv, phase3b_results.csv):
  - ALL 36 tested variants on GBPJPY/EURJPY/AUDJPY were out-of-sample
    profitable (PF 1.08-2.04, Jul 2025-Jun 2026), parameter-INSENSITIVE.
  - But in-sample PF (Jul 2023-Jun 2025) plateaus at 1.10-1.17 -- below
    the PF > 1.3 activation bar. Diagnosis: a REGIME-STRENGTHENING edge,
    not an overfit. Persistence is unproven.
  => Policy: run 2-3 months on the demo account and compare live results
     to the backtest before ANY challenge use. Do not grid-search further.

Strategy logic:
  1. Entry window: 00:00 UTC to `entry_end_hour` (per-pair YAML; 04:00 or
     06:00). JPY crosses only -- the edge did NOT exist on any USD major.
  2. Signal on the last CLOSED M15 bar: z = (close - SMA20) / STD20 of
     the last 20 closed M15 bars. z <= -z_threshold -> BUY (revert up);
     z >= +z_threshold -> SELL (revert down).
  3. TP = the SMA20 value at signal time (the mean being reverted to).
     SL = sl_multiplier x the entry-to-TP distance, opposite side.
     TP distance must be >= max(3 pips, 2.5x typical spread) or no trade.
  4. TIME EXIT at 07:00 UTC: any open AMR position is closed before
     London opens -- the reversion thesis is Asian-hours-only, and being
     flat before London protects the prop-firm daily drawdown from
     session-open gaps. (5ers news rule: if a high-impact event falls on
     the exit minute, the orchestrator's blackout gate must shift it.)
  5. One trade per day per pair.

ORCHESTRATOR INTEGRATION -- NOT YET WIRED (why active: false):
  The main loop currently has no polling steps between 00:00 and 07:45
  UTC. Wiring this strategy needs, in main_agent.py:
    a) a step_check_asian_reversion() called each 15-min cycle while
       hour < entry_end_hour (signal checks), and
    b) a step_asian_time_exit() at >= 07:00 that closes any open
       position whose strategy_key ends in '@amr' (mirror of
       step_friday_close, filtered by key).
  The strategy-cache key scheme already supports it: extend
  agent_strategy._get_strategies() to key these as '<PAIR>@amr'.
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
from core.mt5_time import (mt5_bar_time_to_utc,
                           observed_server_utc_offset_hours)

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


# -- strategy parameters (defaults; per-pair overrides via YAML)
SMA_PERIOD       = 20
Z_THRESHOLD      = 2.0     # YAML `z_threshold`
SL_MULTIPLIER    = 1.5     # YAML `sl_multiplier`
ENTRY_END_HOUR   = 4       # YAML `entry_end_hour` -- entries 00:00 to here
TIME_EXIT_HOUR   = 7       # flat before London open
MIN_TP_PIPS      = 3.0
M15_BARS_NEEDED  = SMA_PERIOD + 1

REQUIRED_KEYS = ['pair', 'strategy', 'active', 'timeframe', 'risk_percent',
                 'z_threshold', 'sl_multiplier', 'entry_end_hour',
                 'session', 'friday_close']


class AsianHoursReversion(BaseStrategy):

    NAME = "asian_hours_reversion"
    SESSION = "asian"
    # CADJPY added 2026-07-05 (phase 6: IS PF 1.10 / OOS 1.35)
    COMPATIBLE_PAIRS = ["GBPJPY", "EURJPY", "AUDJPY", "CADJPY"]

    def __init__(self, pair_config: dict):
        super().__init__(pair_config)
        self.validate_config(pair_config)
        self.pip_size = 0.01 if 'JPY' in self.pair else 0.0001
        self._last_trade_date = None

    # ------------------------------------------------------------------
    # BaseStrategy required interface
    # ------------------------------------------------------------------

    def validate_config(self, pair_config: dict) -> None:
        missing = [k for k in REQUIRED_KEYS if k not in pair_config]
        if missing:
            raise ValueError(
                f"AsianHoursReversion config for {pair_config.get('pair', '?')} "
                f"missing required key(s): {missing}"
            )

    def get_session_windows(self) -> list:
        end = self.pair_config.get('entry_end_hour', ENTRY_END_HOUR)
        return [
            {'name': 'asian', 'start': '00:00', 'end': f'{end:02d}:00'},
        ]

    def generate_signal(self, pair: str, current_price: float, h4_trend: int,
                        **context):
        """With full context, delegates to check_signal(); with only the 3
        base args (smoke test) returns None without hitting MT5."""
        session_data = context.get('session_data')
        session      = context.get('session')
        if session_data is None or session is None:
            return None
        result = self.check_signal(session_data, session)
        if result['signal'] in ('BUY', 'SELL'):
            return result['signal']
        return None

    def calculate_sl(self, entry_price: float, direction: str, pair_config: dict) -> float:
        sl_pips = pair_config.get('sl_pips')
        if sl_pips is None:
            raise ValueError(
                f"AsianHoursReversion.calculate_sl: SL is dynamic (distance to "
                f"the M15 SMA20 x sl_multiplier, computed in check_signal()) -- "
                f"use the sl_pips returned in the signal dict, not the YAML")
        dist = sl_pips * self.pip_size
        return entry_price - dist if direction == 'BUY' else entry_price + dist

    def calculate_tp(self, entry_price: float, direction: str, pair_config: dict) -> float:
        tp_pips = pair_config.get('tp_pips')
        if tp_pips is None:
            raise ValueError(
                f"AsianHoursReversion.calculate_tp: TP is dynamic (the M15 "
                f"SMA20 at signal time) -- use the tp_pips returned in the "
                f"signal dict, not the YAML")
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

    def _m15_bars(self, count: int = M15_BARS_NEEDED):
        """Most recent CLOSED M15 bars (pos=1 skips the forming bar)."""
        rates = mt5.copy_rates_from_pos(self.pair, mt5.TIMEFRAME_M15, 1, count)
        if rates is None or len(rates) < M15_BARS_NEEDED:
            return None
        return rates

    # ------------------------------------------------------------------
    # prepare() -- trivially ready; the signal is self-contained per cycle
    # ------------------------------------------------------------------

    def prepare(self, log) -> dict | None:
        if not self._connect(log):
            return None
        log.info(f"  {self.pair}: AsianHoursReversion armed "
                 f"(z>={self.pair_config['z_threshold']}, "
                 f"entries until {self.pair_config['entry_end_hour']:02d}:00 UTC, "
                 f"time exit {TIME_EXIT_HOUR:02d}:00)")
        return {'armed': True}

    # ------------------------------------------------------------------
    # check_signal() -- z-score vs M15 SMA20, called each 15-min cycle
    # ------------------------------------------------------------------

    def check_signal(self, session_data: dict, session: str) -> dict:
        log = _log()

        no_signal = lambda reason: {
            'signal'      : 'NO_SIGNAL',
            'reason'      : reason,
            'entry_price' : 0.0,
            'sl_pips'     : 0.0,
            'tp_pips'     : 0.0,
        }

        if not session_data:
            return no_signal('pair not in session data')
        if not self._connect(log):
            return no_signal('MT5 connection failed')

        bars = self._m15_bars()
        if bars is None:
            return no_signal(f'insufficient M15 history ({M15_BARS_NEEDED} bars)')

        offset_h = observed_server_utc_offset_hours(mt5, self.pair)
        if offset_h is None:
            return no_signal('could not establish MT5 server UTC offset')
        try:
            bar_time = mt5_bar_time_to_utc(bars[-1]['time'], offset_h)
        except (TypeError, ValueError) as e:
            return no_signal(f'invalid M15 bar timestamp: {e}')

        end_hour = self.pair_config.get('entry_end_hour', ENTRY_END_HOUR)
        if bar_time.hour >= end_hour:
            log.debug(
                "AMR DECISION pair=%s scheduler_utc=%s bar_utc=%s "
                "window=00:00-%02d:00 inside=False signal=NO_SIGNAL "
                "reason=outside entry window",
                self.pair, datetime.now(timezone.utc).isoformat(),
                bar_time.isoformat(), end_hour)
            return no_signal(
                f'bar at {bar_time.hour:02d}:{bar_time.minute:02d} UTC outside '
                f'00:00-{end_hour:02d}:00 entry window')

        if self._last_trade_date == bar_time.date():
            log.debug(
                "AMR DECISION pair=%s scheduler_utc=%s bar_utc=%s "
                "window=00:00-%02d:00 inside=True signal=NO_SIGNAL "
                "reason=already traded UTC date",
                self.pair, datetime.now(timezone.utc).isoformat(),
                bar_time.isoformat(), end_hour)
            return no_signal('already traded this pair today (one trade/day limit)')

        closes = np.array([b['close'] for b in bars], dtype=float)
        sma = float(np.mean(closes[-SMA_PERIOD:]))
        std = float(np.std(closes[-SMA_PERIOD:], ddof=1))
        if std <= 0:
            return no_signal('flat window (STD20 = 0)')

        close = float(closes[-1])
        z = (close - sma) / std
        z_thr   = self.pair_config.get('z_threshold', Z_THRESHOLD)
        sl_mult = self.pair_config.get('sl_multiplier', SL_MULTIPLIER)
        tp_pips = abs(sma - close) / self.pip_size

        if abs(z) < z_thr:
            log.debug(
                "AMR DECISION pair=%s scheduler_utc=%s bar_utc=%s z=%+.3f "
                "threshold=%.3f window=00:00-%02d:00 inside=True "
                "signal=NO_SIGNAL reason=below threshold",
                self.pair, datetime.now(timezone.utc).isoformat(),
                bar_time.isoformat(), z, z_thr, end_hour)
            return no_signal(f'|z|={abs(z):.2f} below threshold {z_thr}')
        if tp_pips < MIN_TP_PIPS:
            log.debug(
                "AMR DECISION pair=%s scheduler_utc=%s bar_utc=%s z=%+.3f "
                "threshold=%.3f window=00:00-%02d:00 inside=True "
                "signal=NO_SIGNAL reason=TP distance %.1fp below %.1fp",
                self.pair, datetime.now(timezone.utc).isoformat(),
                bar_time.isoformat(), z, z_thr, end_hour,
                tp_pips, MIN_TP_PIPS)
            return no_signal(f'TP distance {tp_pips:.1f}p below {MIN_TP_PIPS}p minimum')

        direction = 'BUY' if z <= -z_thr else 'SELL'
        log.debug(
            "AMR DECISION pair=%s scheduler_utc=%s bar_utc=%s z=%+.3f "
            "threshold=%.3f window=00:00-%02d:00 inside=True signal=%s",
            self.pair, datetime.now(timezone.utc).isoformat(),
            bar_time.isoformat(), z, z_thr, end_hour, direction)
        return {
            'signal'      : direction,
            'reason'      : (f'z={z:+.2f} vs SMA{SMA_PERIOD}={sma:.5f}  '
                             f'close={close:.5f}  reverting to mean'),
            'entry_price' : close,
            'sl_pips'     : round(tp_pips * sl_mult, 1),
            'tp_pips'     : round(tp_pips, 1),
            'signal_bar_time_utc': bar_time.isoformat(),
        }

    def acknowledge_trade(self, signal: dict) -> None:
        """Commit same-day dedup only after the broker accepts the entry."""
        if signal.get('signal') not in ('BUY', 'SELL'):
            raise ValueError('cannot acknowledge a non-entry AMR signal')
        bar_time = datetime.fromisoformat(signal['signal_bar_time_utc'])
        self._last_trade_date = bar_time.date()
