"""
MondayDrift -- GBPUSD Monday-session long. VALIDATED 2026-07-05
(src/phase8_monday_validation.py, data/phase8_monday_results.csv):
  IS  (Jul23-Jun25): 103 trades, PF 1.97, max DD 0.66%, 66.7% prof. months
  OOS (Jul25-Jun26):  52 trades, PF 3.08, +$2,573 at 0.25% risk, DD 0.42%
  Robust across the SL/TP grid (OOS PF 2.88-3.12); EURUSD control weak ->
  the effect is GBPUSD-specific. Discovered in the phase-7 calendar
  screen: +0.21%/day OOS Monday drift, t=+4.0, positive every year.

Strategy logic:
  1. BUY GBPUSD at the close of Monday's 00:00-01:00 UTC H1 bar.
     No entry any other day; one trade per Monday.
  2. SL = sl_atr_mult x ATR20d (rolling 20-day daily ATR, pips) below
     entry; TP = tp_atr_mult x ATR20d above (YAML-configurable;
     validated: 1.25 / 1.0).
  3. TIME EXIT 21:00 UTC Monday (orchestrator step_monday_time_exit) --
     never holds overnight.
  4. Long-only by design: the measured anomaly is a positive Monday
     drift, not a symmetric effect.
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


ATR_DAYS        = 20
H1_BARS_NEEDED  = 26 * 24          # ~26 calendar days of H1 for ATR20d
SL_ATR_MULT     = 1.25             # YAML `sl_atr_mult`
TP_ATR_MULT     = 1.0              # YAML `tp_atr_mult`
EXIT_HOUR       = 21               # orchestrator closes at 21:00 UTC Monday

REQUIRED_KEYS = ['pair', 'strategy', 'active', 'timeframe', 'risk_percent',
                 'sl_atr_mult', 'tp_atr_mult', 'session', 'friday_close']


class MondayDrift(BaseStrategy):

    NAME = "monday_drift"
    SESSION = "monday"
    COMPATIBLE_PAIRS = ["GBPUSD"]

    def __init__(self, pair_config: dict):
        super().__init__(pair_config)
        self.validate_config(pair_config)
        self.pip_size = 0.0001
        self._last_trade_date = None

    def validate_config(self, pair_config: dict) -> None:
        missing = [k for k in REQUIRED_KEYS if k not in pair_config]
        if missing:
            raise ValueError(
                f"MondayDrift config for {pair_config.get('pair', '?')} "
                f"missing required key(s): {missing}")

    def get_session_windows(self) -> list:
        return [{'name': 'monday', 'start': '00:00', 'end': f'{EXIT_HOUR:02d}:00'}]

    def generate_signal(self, pair: str, current_price: float, h4_trend: int,
                        **context):
        session_data = context.get('session_data')
        session      = context.get('session')
        if session_data is None or session is None:
            return None
        result = self.check_signal(session_data, session)
        return result['signal'] if result['signal'] == 'BUY' else None

    def calculate_sl(self, entry_price: float, direction: str, pair_config: dict) -> float:
        sl_pips = pair_config.get('sl_pips')
        if sl_pips is None:
            raise ValueError("MondayDrift SL is dynamic (ATR20d) -- use the "
                             "sl_pips returned by check_signal()")
        return entry_price - sl_pips * self.pip_size

    def calculate_tp(self, entry_price: float, direction: str, pair_config: dict) -> float:
        tp_pips = pair_config.get('tp_pips')
        if tp_pips is None:
            raise ValueError("MondayDrift TP is dynamic (ATR20d) -- use the "
                             "tp_pips returned by check_signal()")
        return entry_price + tp_pips * self.pip_size

    @staticmethod
    def _connect(log) -> bool:
        if mt5.initialize():
            return True
        log.error(f"MT5 init failed: {mt5.last_error()}")
        return False

    def _atr20d_pips(self) -> float | None:
        """Rolling 20-day daily ATR in pips from closed H1 bars."""
        rates = mt5.copy_rates_from_pos(self.pair, mt5.TIMEFRAME_H1, 1,
                                        H1_BARS_NEEDED)
        if rates is None or len(rates) < ATR_DAYS * 24:
            return None
        import pandas as pd
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        d = pd.DataFrame({'H': df['high'].resample('1D').max(),
                          'L': df['low'].resample('1D').min(),
                          'C': df['close'].resample('1D').last()}).dropna()
        if len(d) < ATR_DAYS + 1:
            return None
        pc = d['C'].shift(1)
        tr = np.maximum.reduce([(d['H'] - d['L']).to_numpy(),
                                (d['H'] - pc).abs().to_numpy(),
                                (d['L'] - pc).abs().to_numpy()])
        atr = float(np.nanmean(tr[-ATR_DAYS:]))
        return atr / self.pip_size

    def prepare(self, log) -> dict | None:
        if not self._connect(log):
            return None
        log.info(f"  {self.pair}: MondayDrift armed (entry after Monday "
                 f"00:00 H1 close, exit {EXIT_HOUR:02d}:00 UTC)")
        return {'armed': True}

    def check_signal(self, session_data: dict, session: str) -> dict:
        log = _log()
        no_signal = lambda reason: {'signal': 'NO_SIGNAL', 'reason': reason,
                                    'entry_price': 0.0, 'sl_pips': 0.0,
                                    'tp_pips': 0.0}
        if not session_data:
            return no_signal('pair not in session data')
        if not self._connect(log):
            return no_signal('MT5 connection failed')

        rates = mt5.copy_rates_from_pos(self.pair, mt5.TIMEFRAME_H1, 1, 1)
        if rates is None or len(rates) == 0:
            return no_signal('no H1 bar data')
        bar_time = datetime.fromtimestamp(rates[0]['time'], tz=timezone.utc)

        # the signal bar is EXACTLY Monday's 00:00-01:00 H1 bar
        if bar_time.weekday() != 0 or bar_time.hour != 0:
            return no_signal(f'last closed bar {bar_time.strftime("%a %H:%M")} '
                             f'is not the Monday 00:00 bar')
        if self._last_trade_date == bar_time.date():
            return no_signal('already traded this Monday')

        atr = self._atr20d_pips()
        if atr is None or atr <= 0:
            return no_signal('insufficient history for ATR20d')

        sl_mult = self.pair_config.get('sl_atr_mult', SL_ATR_MULT)
        tp_mult = self.pair_config.get('tp_atr_mult', TP_ATR_MULT)
        return {
            'signal'      : 'BUY',
            'reason'      : (f'Monday drift long: ATR20d={atr:.0f}p  '
                             f'SL={sl_mult}x  TP={tp_mult}x'),
            'entry_price' : float(rates[0]['close']),
            'sl_pips'     : round(atr * sl_mult, 1),
            'tp_pips'     : round(atr * tp_mult, 1),
            'signal_bar_time_utc': bar_time.isoformat(),
        }

    def acknowledge_trade(self, signal: dict) -> None:
        """Commit the weekly dedup marker only after broker acceptance."""
        if signal.get('signal') != 'BUY':
            raise ValueError('cannot acknowledge a non-entry Monday signal')
        bar_time = datetime.fromisoformat(signal['signal_bar_time_utc'])
        self._last_trade_date = bar_time.date()
