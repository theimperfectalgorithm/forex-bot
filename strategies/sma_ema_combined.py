"""
SmaEmaCombined strategy -- extracted from the original agent_strategy.py
EURUSD dual-strategy logic (SMA Run 1 + EMA Pullback). One instance is bound
to EURUSD via pair_config['pair'].

All parameters, thresholds, and decision logic are preserved EXACTLY as
they were in agent_strategy.py -- this is a structural move, not a
behavior change.

NOTE on pairs/EURUSD.yaml: the YAML's sl_pips/tp_pips (50/100) are NOT used
to drive this strategy's actual SL/TP. The real strategy runs TWO different
SL/TP pairs depending on which sub-strategy fired (SMA Run 1: 30/60 pips,
EMA Pullback: 15/30 pips) -- both hardcoded below exactly as in the
original agent_strategy.py. Wiring the YAML's single sl_pips/tp_pips value
into the live path would silently change real trading behavior, which
violates the "no parameter changes" constraint for this refactor. The YAML
fields are kept as-specified for schema completeness/future use, but this
class's internal SMA_SL_PIPS/SMA_TP_PIPS/EMA_SL_PIPS/EMA_TP_PIPS constants
remain authoritative.

Strategy logic (unchanged from the original):
  SMA Run 1     : 50/100/200 SMA M15 crossover, H1 EMA50 trend filter,
                  SL=30p / TP=60p, adverse SMA50xSMA100 cross-exit.
  EMA Pullback  : 5/20/50 EMA M15 pullback entry (2-step touch then
                  confirmation bar), H1 EMA50 trend filter, SL=15p / TP=30p,
                  no cross-exit.
  Session       : 12:00-15:45 UTC (London/NY overlap).
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
from core.session_filter import london_ny_overlap

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
EURUSD_PIP_SIZE      = 0.0001

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

# Shared H1 EMA filter
H1_EMA_N         = 50
H1_BARS_NEEDED   = 60

# Per-strategy daily limits
MAX_SMA_DAILY     = 2
MAX_EMA_DAILY     = 2
MAX_EURUSD_CONSEC = 2

REQUIRED_KEYS = ['pair', 'strategy', 'active', 'timeframe', 'h1_filter',
                  'risk_percent', 'sl_pips', 'tp_pips', 'session', 'friday_close']


class SmaEmaCombined(BaseStrategy):

    COMPATIBLE_PAIRS = ["EURUSD"]

    def __init__(self, pair_config: dict):
        super().__init__(pair_config)
        self.validate_config(pair_config)

    # ------------------------------------------------------------------
    # BaseStrategy required interface
    # ------------------------------------------------------------------

    def validate_config(self, pair_config: dict) -> None:
        missing = [k for k in REQUIRED_KEYS if k not in pair_config]
        if missing:
            raise ValueError(
                f"SmaEmaCombined config for {pair_config.get('pair', '?')} "
                f"missing required key(s): {missing}"
            )

    def get_session_windows(self) -> list:
        return [{'name': 'london_ny_overlap', 'start': '12:00', 'end': '15:45'}]

    def generate_signal(self, pair: str, current_price: float, h4_trend: int,
                         **context):
        """
        Required interface method. With full context (eurusd_state +
        open_trades kwargs, as supplied in production by agent_strategy.py),
        delegates to check_signals() and returns the first actionable
        BUY/SELL direction, or None.

        Called with only the 3 base args (e.g. a smoke test with mock data),
        there is no M15 bar history / state context available, so it
        safely returns None rather than hitting MT5.

        Note: check_signals() can also return CROSS_EXIT signals and/or
        multiple signals (SMA + EMA) in one call -- that richer behavior is
        preserved and used by agent_strategy.py directly; generate_signal()
        only narrows it down to satisfy the BaseStrategy BUY/SELL/None contract.
        """
        eurusd_state = context.get('eurusd_state')
        open_trades  = context.get('open_trades')
        if eurusd_state is None or open_trades is None:
            return None

        signals, _ = self.check_signals(eurusd_state, open_trades)
        for sig in signals:
            if sig['signal'] in ('BUY', 'SELL'):
                return sig['signal']
        return None

    def calculate_sl(self, entry_price: float, direction: str, pair_config: dict) -> float:
        sl_pips = pair_config.get('sl_pips', SMA_SL_PIPS)
        dist = sl_pips * EURUSD_PIP_SIZE
        return entry_price - dist if direction == 'BUY' else entry_price + dist

    def calculate_tp(self, entry_price: float, direction: str, pair_config: dict) -> float:
        tp_pips = pair_config.get('tp_pips', SMA_TP_PIPS)
        dist = tp_pips * EURUSD_PIP_SIZE
        return entry_price + dist if direction == 'BUY' else entry_price - dist

    # ------------------------------------------------------------------
    # Helpers (preserved verbatim from agent_strategy.py)
    # ------------------------------------------------------------------

    @staticmethod
    def _connect(log) -> bool:
        if mt5.initialize():
            return True
        log.error(f"MT5 init failed: {mt5.last_error()}")
        return False

    @staticmethod
    def _ewm_ema(values: np.ndarray, period: int) -> np.ndarray:
        """EWM EMA matching pandas ewm(span=n, adjust=False)."""
        alpha  = 2.0 / (period + 1)
        result = np.empty(len(values), dtype=float)
        result[0] = values[0]
        for i in range(1, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i - 1]
        return result

    def _m15_bars(self, count: int = SMA_BARS_NEEDED):
        rates = mt5.copy_rates_from_pos(self.pair, mt5.TIMEFRAME_M15, 1, count)
        if rates is None or len(rates) < SMA_SLOW:
            return None
        return rates

    def _h1_ema50_trend(self, log) -> int:
        """Returns +1 (bullish), -1 (bearish), 0 on data error."""
        rates = mt5.copy_rates_from_pos(self.pair, mt5.TIMEFRAME_H1, 1, H1_BARS_NEEDED)
        if rates is None or len(rates) < H1_EMA_N:
            log.warning(f"{self.pair}: not enough H1 data for EMA{H1_EMA_N}")
            return 0
        closes = np.array([b['close'] for b in rates])
        ema50  = self._ewm_ema(closes, H1_EMA_N)
        last   = closes[-1]
        if last > ema50[-1]:
            return 1
        if last < ema50[-1]:
            return -1
        return 0

    # ------------------------------------------------------------------
    # check_signals() (preserved verbatim from agent_strategy.py check_eurusd_signals)
    # ------------------------------------------------------------------

    def check_signals(self, eurusd_state: dict, open_trades: list) -> tuple:
        """
        Returns (signals, updated_eurusd_state). Each signal is a dict with
        keys: signal ('BUY'|'SELL'|'CROSS_EXIT'), strategy ('SMA'|'EMA'),
        sl_pips, tp_pips, reason, cross_exit_ticket (CROSS_EXIT only).
        """
        log = _log()
        signals = []
        state   = dict(eurusd_state)

        if not self._connect(log):
            return signals, state

        bars = self._m15_bars()
        if bars is None:
            log.warning(f"{self.pair}: insufficient M15 data for signal check")
            return signals, state

        closes = np.array([b['close'] for b in bars])

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

        # ============================================================
        # SMA Run 1 strategy
        # ============================================================

        open_sma = next(
            (t for t in open_trades
             if t.get('symbol') == self.pair and t.get('strategy') == 'SMA'),
            None
        )

        if open_sma is not None:
            bear_cross = sma50_prev >= sma100_prev and sma50_curr < sma100_curr
            bull_cross = sma50_prev <= sma100_prev and sma50_curr > sma100_curr

            adverse = (
                (open_sma['direction'] == 'BUY'  and bear_cross) or
                (open_sma['direction'] == 'SELL' and bull_cross)
            )
            if adverse:
                log.info(f"{self.pair} SMA cross-exit: ticket={open_sma['ticket']} "
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

        # Return a confirmed exit before any entry-only H1 query can delay or
        # raise away that signal. New entries can be evaluated next cycle.
        if signals:
            return signals, state

        # Existing-position crosses are independent of entry session/trend.
        now = datetime.now(timezone.utc)
        if not london_ny_overlap(now):
            return signals, state

        h1_trend = self._h1_ema50_trend(log)
        h1_label = {1: 'BULLISH', -1: 'BEARISH', 0: 'NEUTRAL'}[h1_trend]
        log.info(f"{self.pair}: H1 EMA50 = {h1_label}  (needed: BULLISH for BUY, BEARISH for SELL)")
        if h1_trend == 0:
            log.info(f"{self.pair}: H1 trend NEUTRAL -- no signals possible this bar")
            return signals, state

        sma_daily  = state.get('sma_daily_trades', 0)
        sma_consec = state.get('sma_consec_losses', 0)
        sma_ok = (
            open_sma is None
            and sma_daily < MAX_SMA_DAILY
            and sma_consec < MAX_EURUSD_CONSEC
        )

        if not sma_ok:
            if open_sma is not None:
                log.info(f"{self.pair} SMA: skip new entry -- trade already open (ticket={open_sma['ticket']})")
            elif sma_daily >= MAX_SMA_DAILY:
                log.info(f"{self.pair} SMA: skip -- daily trade limit reached ({sma_daily}/{MAX_SMA_DAILY})")
            elif sma_consec >= MAX_EURUSD_CONSEC:
                log.info(f"{self.pair} SMA: skip -- paused after {sma_consec} consecutive losses")

        if sma_ok and not np.isnan(sma200_curr):
            last_close = closes[-1]
            flat       = abs(sma50_curr - sma100_curr) / EURUSD_PIP_SIZE < SMA_FLAT_PIPS
            bull_cross = sma50_prev <= sma100_prev and sma50_curr > sma100_curr
            bear_cross = sma50_prev >= sma100_prev and sma50_curr < sma100_curr

            log.info(
                f"{self.pair} SMA: SMA50={sma50_curr:.5f}  SMA100={sma100_curr:.5f}  "
                f"SMA200={sma200_curr:.5f}  close={last_close:.5f}  "
                f"flat={flat}  bull_cross={bull_cross}  bear_cross={bear_cross}"
            )

            if flat:
                log.info(f"{self.pair} SMA: skip -- flat zone "
                         f"(|SMA50-SMA100|={abs(sma50_curr-sma100_curr)/EURUSD_PIP_SIZE:.1f}p "
                         f"< {SMA_FLAT_PIPS}p)")
            elif bull_cross and h1_trend == 1 and last_close > sma200_curr:
                log.info(f"{self.pair} SMA: BUY signal -- SMA50 x SMA100 bullish cross")
                signals.append({
                    'signal'  : 'BUY',
                    'strategy': 'SMA',
                    'sl_pips' : SMA_SL_PIPS,
                    'tp_pips' : SMA_TP_PIPS,
                    'reason'  : 'SMA50 crossed above SMA100, H1 EMA50 bullish, price > SMA200',
                })
            elif bear_cross and h1_trend == -1 and last_close < sma200_curr:
                log.info(f"{self.pair} SMA: SELL signal -- SMA50 x SMA100 bearish cross")
                signals.append({
                    'signal'  : 'SELL',
                    'strategy': 'SMA',
                    'sl_pips' : SMA_SL_PIPS,
                    'tp_pips' : SMA_TP_PIPS,
                    'reason'  : 'SMA50 crossed below SMA100, H1 EMA50 bearish, price < SMA200',
                })
            elif bull_cross or bear_cross:
                if bull_cross:
                    log.info(
                        f"{self.pair} SMA: bullish cross present but no signal -- "
                        f"H1={h1_label}  close>SMA200={last_close > sma200_curr}"
                    )
                else:
                    log.info(
                        f"{self.pair} SMA: bearish cross present but no signal -- "
                        f"H1={h1_label}  close<SMA200={last_close < sma200_curr}"
                    )
            else:
                log.info(f"{self.pair} SMA: no crossover this bar")

        # ============================================================
        # EMA Pullback strategy (Test A -- no cross-exit)
        # ============================================================

        open_ema   = any(
            t.get('symbol') == self.pair and t.get('strategy') == 'EMA'
            for t in open_trades
        )
        ema_daily  = state.get('ema_daily_trades', 0)
        ema_consec = state.get('ema_consec_losses', 0)
        ema_ok = (
            not open_ema
            and ema_daily  < MAX_EMA_DAILY
            and ema_consec < MAX_EURUSD_CONSEC
        )

        if not ema_ok:
            if open_ema:
                log.info(f"{self.pair} EMA: skip -- trade already open")
            elif ema_daily >= MAX_EMA_DAILY:
                log.info(f"{self.pair} EMA: skip -- daily trade limit reached ({ema_daily}/{MAX_EMA_DAILY})")
            elif ema_consec >= MAX_EURUSD_CONSEC:
                log.info(f"{self.pair} EMA: skip -- paused after {ema_consec} consecutive losses")

        if ema_ok:
            ema5  = self._ewm_ema(closes, EMA_FAST_N)
            ema20 = self._ewm_ema(closes, EMA_MID_N)
            ema50 = self._ewm_ema(closes, EMA_SLOW_N)

            e5   = ema5[-1];  e20  = ema20[-1];  e50  = ema50[-1]
            last = closes[-1]

            bull_aligned = e5 > e20 > e50
            bear_aligned = e5 < e20 < e50

            touch      = EMA_TOUCH_PIPS * EURUSD_PIP_SIZE
            pb_pending = state.get('ema_pullback_pending', False)
            pb_dir     = state.get('ema_pullback_dir', '')

            log.info(
                f"{self.pair} EMA: close={last:.5f}  EMA5={e5:.5f}  EMA20={e20:.5f}  EMA50={e50:.5f}  "
                f"bull_aligned={bull_aligned}  bear_aligned={bear_aligned}  "
                f"pb_pending={pb_pending}  pb_dir={pb_dir!r}"
            )

            if not pb_pending:
                if bull_aligned and h1_trend == 1:
                    dist_pips = abs(last - e20) / EURUSD_PIP_SIZE
                    if abs(last - e20) <= touch or last < e20:
                        state['ema_pullback_pending'] = True
                        state['ema_pullback_dir']     = 'BUY'
                        log.info(
                            f"{self.pair} EMA: BUY pullback TOUCH (step 1/2) -- "
                            f"close={last:.5f}  EMA20={e20:.5f}  dist={dist_pips:.1f}p  "
                            f"NO trade yet -- waiting for confirmation bar"
                        )
                    else:
                        log.info(
                            f"{self.pair} EMA: bullish alignment, H1 bullish, but no pullback touch -- "
                            f"dist={dist_pips:.1f}p from EMA20 (need <= {EMA_TOUCH_PIPS}p)"
                        )
                elif bear_aligned and h1_trend == -1:
                    dist_pips = abs(last - e20) / EURUSD_PIP_SIZE
                    if abs(last - e20) <= touch or last > e20:
                        state['ema_pullback_pending'] = True
                        state['ema_pullback_dir']     = 'SELL'
                        log.info(
                            f"{self.pair} EMA: SELL pullback TOUCH (step 1/2) -- "
                            f"close={last:.5f}  EMA20={e20:.5f}  dist={dist_pips:.1f}p  "
                            f"NO trade yet -- waiting for confirmation bar"
                        )
                    else:
                        log.info(
                            f"{self.pair} EMA: bearish alignment, H1 bearish, but no pullback touch -- "
                            f"dist={dist_pips:.1f}p from EMA20 (need <= {EMA_TOUCH_PIPS}p)"
                        )
                else:
                    log.info(
                        f"{self.pair} EMA: no setup -- "
                        f"bull_aligned={bull_aligned} (need True + H1 BULLISH)  "
                        f"bear_aligned={bear_aligned} (need True + H1 BEARISH)  "
                        f"H1={h1_label}"
                    )
            else:
                if pb_dir == 'BUY':
                    if last > e20 and bull_aligned and h1_trend == 1:
                        log.info(
                            f"{self.pair} EMA: BUY CONFIRMED (step 2/2) -- "
                            f"close={last:.5f} > EMA20={e20:.5f}  entry signal generated"
                        )
                        signals.append({
                            'signal'  : 'BUY',
                            'strategy': 'EMA',
                            'sl_pips' : EMA_SL_PIPS,
                            'tp_pips' : EMA_TP_PIPS,
                            'reason'  : 'EMA pullback confirmed: close above EMA20, H1 bullish',
                        })
                    elif not bull_aligned or h1_trend != 1:
                        log.info(
                            f"{self.pair} EMA: BUY pullback cancelled -- "
                            f"bull_aligned={bull_aligned}  H1={h1_label}"
                        )
                        state['ema_pullback_pending'] = False
                        state['ema_pullback_dir']     = ''
                    else:
                        log.info(
                            f"{self.pair} EMA: BUY pending, close={last:.5f} not yet above EMA20={e20:.5f} -- "
                            f"still waiting"
                        )

                elif pb_dir == 'SELL':
                    if last < e20 and bear_aligned and h1_trend == -1:
                        log.info(
                            f"{self.pair} EMA: SELL CONFIRMED (step 2/2) -- "
                            f"close={last:.5f} < EMA20={e20:.5f}  entry signal generated"
                        )
                        signals.append({
                            'signal'  : 'SELL',
                            'strategy': 'EMA',
                            'sl_pips' : EMA_SL_PIPS,
                            'tp_pips' : EMA_TP_PIPS,
                            'reason'  : 'EMA pullback confirmed: close below EMA20, H1 bearish',
                        })
                    elif not bear_aligned or h1_trend != -1:
                        log.info(
                            f"{self.pair} EMA: SELL pullback cancelled -- "
                            f"bear_aligned={bear_aligned}  H1={h1_label}"
                        )
                        state['ema_pullback_pending'] = False
                        state['ema_pullback_dir']     = ''
                    else:
                        log.info(
                            f"{self.pair} EMA: SELL pending, close={last:.5f} not yet below EMA20={e20:.5f} -- "
                            f"still waiting"
                        )

        return signals, state

    def acknowledge_trade(self, signal: dict, state: dict | None = None) -> dict:
        """Commit execution-dependent EURUSD state after broker acceptance."""
        updated = dict(state or {})
        if signal.get('signal') not in ('BUY', 'SELL'):
            raise ValueError('cannot acknowledge a non-entry EURUSD signal')
        if signal.get('strategy') == 'EMA':
            expected = signal['signal']
            if (not updated.get('ema_pullback_pending')
                    or updated.get('ema_pullback_dir') != expected):
                raise ValueError('EMA acknowledgement does not match pending setup')
            updated['ema_pullback_pending'] = False
            updated['ema_pullback_dir'] = ''
        return updated
