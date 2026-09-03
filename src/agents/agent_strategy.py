"""
Agent 2 -- Strategy
===================
Called twice per day by the orchestrator:
  07:45 UTC  --  prepare_session('london')
  12:45 UTC  --  prepare_session('ny')

Then called every 15 minutes during the session window:
  check_breakout(pair, session_data, session)

This module is now a thin shell: the real decision logic for each pair
lives in its assigned strategy class (strategies/london_breakout.py,
strategies/sma_ema_combined.py), loaded dynamically via
core.pair_manager / core.strategy_loader based on pairs/*.yaml. No
hardcoded pair-specific logic remains here.

The public functions below (prepare_session, check_breakout,
check_eurusd_signals) keep their exact original signatures so
main_agent.py and test_system.py need no changes. `_h4_trend` and `_log`
also keep their original module-level signatures, since test_system.py
imports them directly -- `_h4_trend` is a pair-agnostic utility (works for
any symbol regardless of which strategy is assigned to it) and is kept
standalone rather than routed through a specific strategy instance.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import time

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
import numpy as np

# -- path setup so this module can import the repo-root strategies/core packages
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.pair_manager import get_active_pairs
from core.runtime_paths import data_dir

# -- logging
LOGS_DIR = data_dir() / 'logs'

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


# ---------------------------------------------------------------------------
# Strategy instance cache (loaded once per process via pair_manager)
# ---------------------------------------------------------------------------

_STRATEGY_CACHE: dict | None = None


def _get_strategies() -> dict:
    """Lazily load active pairs via pair_manager; cached for the process lifetime.

    Returns {key: strategy_instance} where key is the plain pair name for
    every strategy EXCEPT asian_range_breakout, which gets '<PAIR>@arb'.
    The suffix exists because one pair can now run two active strategies
    (e.g. GBPJPY london_breakout AND GBPJPY@arb asian_range_breakout) --
    a plain pair-name dict would silently shadow one with the other.
    Non-ARB keys are unchanged so existing state files and logs keep
    working. Use key.split('@')[0] to recover the MT5 symbol."""
    global _STRATEGY_CACHE
    if _STRATEGY_CACHE is None:
        active = get_active_pairs(log=_log())
        cache = {}
        for name, inst, cfg in active:
            strat = cfg.get('strategy')
            if strat == 'asian_range_breakout':
                key = f"{name}@arb"
            elif strat == 'asian_hours_reversion':
                key = f"{name}@amr"
            elif strat == 'monday_drift':
                key = f"{name}@mon"
            else:
                key = name
            cache[key] = inst
        _STRATEGY_CACHE = cache
    return _STRATEGY_CACHE


def server_utc_offset_hours() -> int:
    """
    MT5 SERVER clock offset vs real UTC, in whole hours. MetaQuotes-style
    brokers run UTC+3 during US daylight saving and UTC+2 otherwise (so
    the trading day rolls at New York close). Every bar timestamp MT5
    returns -- and therefore every session-hour rule in the strategies
    and every backtest window -- is in SERVER time, while the orchestrator
    wall clock is real UTC. This helper is the single source of truth for
    converting between the two.

    Detection: live tick timestamp vs real UTC when the market is open
    (accepted only if it lands on a plausible value); DST-calendar
    fallback when ticks are stale (weekend).
    """
    from datetime import datetime, timezone
    try:
        if mt5.initialize():
            tick = mt5.symbol_info_tick('EURUSD')
            if tick and tick.time:
                diff = round((tick.time
                              - datetime.now(timezone.utc).timestamp()) / 3600)
                if diff in (2, 3):
                    return diff
    except Exception:
        pass
    # fallback: US DST runs Mar-Nov; +/- a week at the edges is acceptable
    return 3 if 3 <= datetime.now(timezone.utc).month <= 10 else 2


def amr_keys() -> list:
    """Keys of active AsianHoursReversion pairs ('<PAIR>@amr')."""
    from strategies.asian_hours_reversion import AsianHoursReversion
    return [k for k, inst in _get_strategies().items()
            if isinstance(inst, AsianHoursReversion)]


def check_asian_reversion(key: str) -> dict:
    """15-min cycle check for a self-contained strategy key (@amr, @mon --
    both read their own closed bars from MT5, need no session prep, and
    return DYNAMIC sl_pips/tp_pips in the signal dict)."""
    inst = _get_strategies().get(key)
    if inst is None:
        return {'signal': 'NO_SIGNAL', 'reason': f'{key}: not loaded',
                'entry_price': 0.0, 'sl_pips': 0.0, 'tp_pips': 0.0}
    return inst.check_signal({'armed': True}, 'asian')


def acknowledge_trade(key: str, signal: dict,
                      state: dict | None = None) -> dict | None:
    """Commit strategy-local consumed state after confirmed execution only."""
    inst = _get_strategies().get(key)
    if inst is None:
        raise KeyError(f'{key}: strategy not loaded for acknowledgement')
    acknowledge = getattr(inst, 'acknowledge_trade', None)
    if acknowledge is None:
        return state
    if state is None:
        acknowledge(signal)
        return None
    return acknowledge(signal, state)


def _breakout_pairs() -> list:
    """Keys of active pairs whose strategy follows the breakout interface
    (prepare() once per session + check_breakout() each cycle, identical
    session_data shape): london_breakout and asian_range_breakout.
    ARB prep timing rides the existing 07:45 London prep -- the Asian
    range is complete at 07:00 and ARB's hour-7/8 breakout bars close at
    08:00/09:00, inside the orchestrator's London polling window; its own
    check_breakout() rejects any bar outside the 07:00-09:00 window."""
    from strategies.london_breakout import LondonBreakout
    from strategies.asian_range_breakout import AsianRangeBreakout
    return [key for key, inst in _get_strategies().items()
            if isinstance(inst, (LondonBreakout, AsianRangeBreakout))]


# ---------------------------------------------------------------------------
# H4 trend (pair-agnostic utility -- preserved standalone since
# test_system.py imports it directly and uses it across all 3 pairs)
# ---------------------------------------------------------------------------

H4_SMA_FAST    = 50
H4_SMA_SLOW    = 200
H4_BARS_NEEDED = 220    # enough history for SMA200


def _connect(log: logging.Logger) -> bool:
    if mt5.initialize():
        return True
    log.error(f"MT5 init failed: {mt5.last_error()}")
    return False


def _h4_trend(symbol: str, log: logging.Logger) -> int:
    """
    Compute H4 trend for symbol. Returns +1 (BULLISH), -1 (BEARISH), or 0 (NEUTRAL).
    Pair-agnostic: works for any symbol MT5 has H4 data for.
    """
    if not _connect(log):
        return 0

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, H4_BARS_NEEDED)
    if rates is None or len(rates) < H4_SMA_SLOW:
        log.warning(f"{symbol}: not enough H4 history ({H4_SMA_SLOW} bars needed)")
        return 0

    closes = np.array([b['close'] for b in rates])
    sma50  = np.mean(closes[-H4_SMA_FAST:])
    sma200 = np.mean(closes[-H4_SMA_SLOW:])
    last   = closes[-1]

    if last > sma50 > sma200:
        return 1
    if last < sma50 < sma200:
        return -1
    return 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_session(session: str) -> dict:
    """
    Called at 07:45 (london) or 12:45 (ny) UTC.

    For each active london_breakout pair, delegates to that pair's
    LondonBreakout.prepare() to compute H4 trend + Asian range.

    Returns:
        {
          'GBPJPY': {h4_trend, asian_high, asian_low, range_pips, sl_pips, tp_pips},
          'EURJPY': {...},
        }
        -- pair is OMITTED from the dict if data is unavailable or trend is neutral.
    """
    log = _log()
    log.info(f"Agent 2 -- prepare_session({session.upper()})")

    strategies = _get_strategies()
    result = {}
    for pair in _breakout_pairs():
        data = strategies[pair].prepare(log)
        if data is not None:
            result[pair] = data

    return result


def check_breakout(symbol: str, session_data: dict, session: str) -> dict:
    """
    Called every 15 minutes during a session window. Delegates to the
    pair's LondonBreakout.check_breakout() with that pair's own slice of
    session_data.
    """
    no_signal = lambda reason: {
        'signal'            : 'NO_SIGNAL',
        'reason'            : reason,
        'trigger_bar_close' : 0.0,
        'trigger_bar_time'  : '',
        'entry_price'       : 0.0,
        'overshoot_pips'    : 0.0,
    }

    strategies = _get_strategies()
    strategy = strategies.get(symbol)
    if strategy is None:
        return no_signal(f'{symbol}: no active london_breakout strategy loaded')

    pair_session_data = session_data.get(symbol)
    if pair_session_data is None:
        return no_signal('pair not in session data (neutral trend or bad range)')

    return strategy.check_breakout(pair_session_data, session)


# ---------------------------------------------------------------------------
# EURUSD dual-strategy signal check
# ---------------------------------------------------------------------------

def check_eurusd_signals(eurusd_state: dict, open_trades: list) -> tuple:
    """
    Check EURUSD for SMA Run 1 and EMA Pullback signals. Delegates to the
    EURUSD pair's SmaEmaCombined.check_signals().

    Returns (signals, updated_eurusd_state) -- see SmaEmaCombined.check_signals
    for the signal dict shape.
    """
    strategies = _get_strategies()
    strategy = strategies.get('EURUSD')
    if strategy is None:
        _log().warning("EURUSD: no active sma_ema_combined strategy loaded")
        return [], dict(eurusd_state)

    return strategy.check_signals(eurusd_state, open_trades)
