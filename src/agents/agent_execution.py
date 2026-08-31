"""
Agent 4 -- Trade Execution
===========================
Two responsibilities:

1. place_trade()
   Called by the orchestrator when risk is APPROVED.
   Places a market order on MT5 with SL and TP anchored to the Asian
   session range (not to market entry price -- matches backtest design).
   Logs every trade to data/trades_log.csv.

2. monitor_positions()
   Called by the orchestrator every 15 minutes.
   - Detects positions closed by MT5 (SL or TP hit)
   - Moves stop loss to breakeven when trade is >= 25 pips in profit
   Returns (still_open, newly_closed).

Stop loss  = Asian High/Low - 50% of Asian range
Take profit = Asian High/Low + 100% of Asian range  (2:1 RR)
"""

from __future__ import annotations

import csv
import logging
import math
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import time

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

# -- paths
AGENTS_DIR  = Path(__file__).parent
BASE_DIR    = AGENTS_DIR.parent.parent
from core.runtime_paths import data_dir

DATA_DIR    = data_dir()
LOGS_DIR    = DATA_DIR / 'logs'
TRADES_LOG  = DATA_DIR / 'trades_log.csv'

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.mt5_time import observed_server_utc_offset_hours, server_epoch_to_utc
from core.mt5_connect import initialize_and_validate
from core.trade_cost_ledger import aggregate_position_deals, append_cost_record
from core.prop_loss_guard import evaluate_prop_risk

# -- logging
def _log() -> logging.Logger:
    log = logging.getLogger('EXECUTION')
    if not log.handlers:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
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


# -- constants
MAGIC_NUMBER      = 200001   # unique identifier for all system trades
DEVIATION         = 20       # max slippage in points
BREAKEVEN_PIPS    = 25       # move SL to entry when this many pips in profit
MAX_CLOSE_RETRIES = 3        # give up searching for exit deal after this many cycles
MIN_TP_HEADROOM   = 5        # reject order if live entry leaves < 5 pips to TP

PAIRS = {
    'GBPJPY': {'pip_size': 0.01,   'digits': 3},
    'EURJPY': {'pip_size': 0.01,   'digits': 3},
    'EURUSD': {'pip_size': 0.0001, 'digits': 5},
    'AUDJPY': {'pip_size': 0.01,   'digits': 3},
    'CADJPY': {'pip_size': 0.01,   'digits': 3},
    'NZDJPY': {'pip_size': 0.01,   'digits': 3},
    'GBPUSD': {'pip_size': 0.0001, 'digits': 5},
    # gold: 1 'pip' = $0.10; broker quotes 2 decimals
    'XAUUSD': {'pip_size': 0.1,    'digits': 2},
}


# ---------------------------------------------------------------------------
# Trade log (CSV)
# ---------------------------------------------------------------------------

TRADES_LOG_HEADERS = [
    'Timestamp', 'Pair', 'Direction', 'Session', 'Lots',
    'EntryPrice', 'SL', 'TP', 'AsianHigh', 'AsianLow',
    'RangePips', 'SLPips', 'TPPips', 'Ticket', 'Status',
    'ExitPrice', 'ExitTime', 'ExitReason', 'PnL', 'Balance',
]

def _write_trade_log(row: dict):
    """Append one row to trades_log.csv, creating the file with headers if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = TRADES_LOG.exists()

    with open(TRADES_LOG, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=TRADES_LOG_HEADERS,
                                extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# MT5 helpers
# ---------------------------------------------------------------------------

def _connect(log: logging.Logger) -> bool:
    if mt5.initialize():
        return True
    log.error(f"MT5 init failed: {mt5.last_error()}")
    return False


def _connect_for_entry(log: logging.Logger) -> bool:
    """Entry-only connection gate with independent account validation."""
    return initialize_and_validate(log)


def _broker_duplicate_for_symbol(symbol: str, log: logging.Logger):
    """Return a bot-owned same-symbol position, or raise on query failure."""
    positions = mt5.positions_get()
    if positions is None:
        raise RuntimeError(f"positions_get failed: {mt5.last_error()}")
    return next((p for p in positions
                 if p.symbol == symbol and p.magic == MAGIC_NUMBER), None)


def _price_round(price: float, symbol: str) -> float:
    digits = PAIRS[symbol]['digits']
    return round(price, digits)


def _get_live_price(symbol: str, signal: str) -> float | None:
    """Return ask (BUY) or bid (SELL) for market order."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None
    price = tick.ask if signal == 'BUY' else tick.bid
    return price if isinstance(price, (int, float)) and math.isfinite(price) and price > 0 else None


def _expected_loss(symbol: str, signal: str, volume: float,
                   entry: float, sl: float) -> float | None:
    """Broker-calculated account-currency loss, or None (fail closed)."""
    if not all(math.isfinite(v) and v > 0 for v in (volume, entry, sl)):
        return None
    order_type = mt5.ORDER_TYPE_BUY if signal == 'BUY' else mt5.ORDER_TYPE_SELL
    try:
        profit = mt5.order_calc_profit(order_type, symbol, volume, entry, sl)
    except Exception:
        return None
    if profit is None or not isinstance(profit, (int, float)) or not math.isfinite(profit):
        return None
    loss = -float(profit)
    return loss if loss > 0 else None


def _floor_volume(raw: float, volume_min: float, volume_step: float,
                  effective_max: float) -> float | None:
    """Normalize downward on the broker's min-anchored volume grid."""
    vals = (raw, volume_min, volume_step, effective_max)
    if not all(math.isfinite(v) and v > 0 for v in vals) or effective_max < volume_min:
        return None
    capped = min(raw, effective_max)
    grid_tolerance = max(1e-12, volume_step * 1e-10)
    if capped + grid_tolerance < volume_min:
        return None
    if capped < volume_min:  # mathematical equality obscured by float division
        capped = volume_min
    steps = math.floor(((capped - volume_min) / volume_step) + 1e-12)
    normalized = volume_min + steps * volume_step
    # Decimal places are presentation only; the floor above is authoritative.
    decimals = max(0, min(8, len(f"{volume_step:.8f}".rstrip('0').split('.')[-1])))
    return round(normalized, decimals)


def _size_for_risk(symbol: str, signal: str, entry: float, sl: float,
                   allowed_risk: float, bot_max_lot: float,
                   nominal_sl_pips: float, pip_size: float,
                   log: logging.Logger) -> tuple[float, float] | tuple[None, str]:
    """Find the largest broker-valid volume whose expected loss is safe."""
    if signal == 'BUY' and sl >= entry:
        return None, f"invalid BUY SL: entry={entry} SL={sl}"
    if signal == 'SELL' and sl <= entry:
        return None, f"invalid SELL SL: entry={entry} SL={sl}"
    if not math.isfinite(allowed_risk) or allowed_risk <= 0:
        return None, "invalid allowed monetary risk"
    info = mt5.symbol_info(symbol)
    try:
        vmin, vstep, vmax = float(info.volume_min), float(info.volume_step), float(info.volume_max)
    except (AttributeError, TypeError, ValueError):
        return None, f"missing/invalid broker volume metadata for {symbol}"
    if not all(math.isfinite(v) and v > 0 for v in (vmin, vstep, vmax)):
        return None, f"missing/invalid broker volume metadata for {symbol}"
    effective_max = min(vmax, bot_max_lot)
    min_loss = _expected_loss(symbol, signal, vmin, entry, sl)
    if min_loss is None:
        return None, f"broker loss calculation failed for {symbol}"
    if min_loss > allowed_risk + max(1e-9, allowed_risk * 1e-12):
        return None, (f"RISK REJECTED: broker minimum volume would exceed allowed monetary risk; "
                      f"symbol={symbol} allowed=${allowed_risk:.2f} minimum_volume={vmin:g} "
                      f"expected_loss=${min_loss:.2f} entry={entry} SL={sl}")
    # MT5 profit calculation is linear in volume. Derive from the known-valid
    # broker minimum instead of asking it to price 1.0 lot, which itself may
    # exceed an unusual symbol's volume_max.
    loss_per_lot = min_loss / vmin
    raw = allowed_risk / loss_per_lot
    volume = _floor_volume(raw, vmin, vstep, effective_max)
    if volume is None:
        return None, f"no valid safe broker volume for {symbol}"
    expected = _expected_loss(symbol, signal, volume, entry, sl)
    if expected is None:
        return None, f"broker loss calculation failed for {symbol}"
    actual_pips = abs(entry - sl) / pip_size
    log.info(f"RISK DISTANCE: {symbol} {signal} nominal={nominal_sl_pips:.1f}p "
             f"actual={actual_pips:.1f}p")
    log.info(f"BROKER SIZING: allowed=${allowed_risk:.2f} raw={raw:.8f} "
             f"min={vmin:g} step={vstep:g} broker_max={vmax:g} bot_max={bot_max_lot:g} "
             f"final={volume:g} expected_loss=${expected:.2f}")
    return volume, expected


def _get_filling_mode(symbol: str) -> int:
    """
    Return the best supported ORDER_FILLING_* constant for this symbol.

    MT5 symbol_info.filling_mode is a bitmask:
      bit 0 (value 1) = FOK supported
      bit 1 (value 2) = IOC supported

    Priority: FOK -> IOC -> RETURN (fallback).
    Different brokers support different modes per instrument; using an
    unsupported mode produces retcode 10030.
    """
    info = mt5.symbol_info(symbol)
    if info is not None:
        fm = info.filling_mode
        if fm & 1:
            return mt5.ORDER_FILLING_FOK
        if fm & 2:
            return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


def _confirm_fill_price(ticket: int, fallback_price: float, log: logging.Logger) -> float:
    """
    order_send()'s immediate result.price is unreliable on market-execution
    brokers (confirmed 2026-08: intermittently 0.0 on the 5ers/Five Percent
    Online account for genuinely successful, correctly-executed orders --
    real SL/TP/PnL were always correct since those never depended on this
    value, but the logged EntryPrice/fill_price corrupted slippage and
    R-multiple analysis for the affected trades). The just-opened
    position's price_open always reflects the broker's own confirmed
    fill; a few short retries cover the rare case where it isn't visible
    in the microseconds right after order_send() returns. fallback_price
    (result.price if it was non-zero, else the pre-order live-price
    snapshot) is used only if positions_get() never confirms -- strictly
    better than silently recording 0.0.
    """
    for _ in range(3):
        positions = mt5.positions_get(ticket=ticket)
        if positions and positions[0].price_open:
            return positions[0].price_open
        time.sleep(0.3)
    if fallback_price:
        log.warning(f"ticket {ticket}: could not confirm fill via positions_get() -- "
                   f"using fallback price {fallback_price}")
        return fallback_price
    log.warning(f"ticket {ticket}: could not confirm fill price at all -- logging 0.0")
    return 0.0


# ---------------------------------------------------------------------------
# Place trade
# ---------------------------------------------------------------------------

def place_trade(symbol: str, breakout: dict, lot_size: float,
                session_data: dict, session: str,
                allowed_risk_dollars: float | None = None,
                bot_max_lot: float | None = None) -> dict:
    """
    Place a market order on MT5.

    For GBPJPY/EURJPY (breakout): SL/TP anchored to Asian range level.
    For EURUSD (SMA/EMA): SL/TP anchored to live entry price when
      session_data['use_live_anchor'] is True.

    Returns:
        {
          'success'     : bool,
          'ticket'      : int,
          'entry_price' : float,
          'sl'          : float,
          'tp'          : float,
          'error'       : str | None,
        }
    """
    log = _log()

    failed = lambda err: {'success': False, 'ticket': 0,
                          'entry_price': 0.0, 'sl': 0.0, 'tp': 0.0, 'error': err}

    if not _connect_for_entry(log):
        return failed("MT5 expected-account identity validation failed")

    if not mt5.symbol_select(symbol, True):
        return failed(f"symbol_select({symbol}) failed")

    signal     = breakout['signal']
    sl_pips    = session_data['sl_pips']
    tp_pips    = session_data['tp_pips']
    pip_size   = PAIRS[symbol]['pip_size']
    use_live   = session_data.get('use_live_anchor', False)
    strategy   = session_data.get('strategy', '')

    # Get live entry price
    entry_price = _get_live_price(symbol, signal)
    if entry_price is None:
        return failed(f"Could not get live price for {symbol}")

    # Determine SL/TP anchor: live price (EURUSD) or Asian range level (others)
    if use_live:
        anchor = entry_price
    else:
        anchor = session_data['asian_high'] if signal == 'BUY' else session_data['asian_low']

    if signal == 'BUY':
        sl_price   = _price_round(anchor - sl_pips * pip_size, symbol)
        tp_price   = _price_round(anchor + tp_pips * pip_size, symbol)
        order_type = mt5.ORDER_TYPE_BUY
    else:
        sl_price   = _price_round(anchor + sl_pips * pip_size, symbol)
        tp_price   = _price_round(anchor - tp_pips * pip_size, symbol)
        order_type = mt5.ORDER_TYPE_SELL

    # Headroom check: reject if live entry leaves < MIN_TP_HEADROOM pips to TP.
    # This catches the case where the market has already moved most of the way
    # to the TP by the time the order is sent (e.g. entry at 215.010 with TP
    # at 215.013 -- 0.3 pip reward against 31.8 pip risk).
    if signal == 'BUY':
        headroom_pips = (tp_price - entry_price) / pip_size
    else:
        headroom_pips = (entry_price - tp_price) / pip_size

    if headroom_pips < MIN_TP_HEADROOM:
        err = (f"TP headroom too small: entry={entry_price}  tp={tp_price}  "
               f"headroom={headroom_pips:.1f}p < {MIN_TP_HEADROOM}p minimum -- order rejected")
        log.warning(err)
        return failed(err)

    # The legacy lot_size argument is intentionally not trusted. It was based
    # on nominal strategy distance. Production callers must provide the
    # unchanged monetary budget so this layer can size against broker truth.
    if allowed_risk_dollars is None or bot_max_lot is None:
        return failed("missing final broker-aware risk budget -- order rejected")
    sized = _size_for_risk(symbol, signal, entry_price, sl_price,
                           allowed_risk_dollars, bot_max_lot, sl_pips,
                           pip_size, log)
    if sized[0] is None:
        log.error(sized[1])
        return failed(sized[1])
    final_volume, pre_send_risk = sized

    comment = f"5ers_{session}_{signal}_{strategy}" if strategy else f"5ers_{session}_{signal}"

    request = {
        'action'      : mt5.TRADE_ACTION_DEAL,
        'symbol'      : symbol,
        'volume'      : final_volume,
        'type'        : order_type,
        'price'       : entry_price,
        'sl'          : sl_price,
        'tp'          : tp_price,
        'deviation'   : DEVIATION,
        'magic'       : MAGIC_NUMBER,
        'comment'     : comment,
        'type_time'   : mt5.ORDER_TIME_GTC,
        'type_filling': _get_filling_mode(symbol),
    }

    # Final broker-truth guard, deliberately adjacent to order_send. Stale
    # state or a crash/restart must not permit a second bot position on the
    # same symbol. Unrelated/manual magic numbers do not claim the slot.
    try:
        duplicate = _broker_duplicate_for_symbol(symbol, log)
    except Exception as e:
        err = f"broker-side duplicate guard query failed for {symbol}: {e} -- order rejected"
        log.error(err)
        return failed(err)
    if duplicate is not None:
        err = (f"broker-side duplicate guard: {symbol} already has bot position "
               f"ticket={duplicate.ticket} magic={duplicate.magic} -- order rejected")
        log.error(err)
        return failed(err)

    # Refresh executable price at the last practical point before submission.
    # SL/TP methodology remains unchanged; only risk and request price refresh.
    final_entry = _get_live_price(symbol, signal)
    if final_entry is None:
        return failed(f"missing final tick for {symbol} -- order rejected")
    final_risk = _expected_loss(symbol, signal, final_volume, final_entry, sl_price)
    tolerance = max(1e-9, allowed_risk_dollars * 1e-12)
    if final_risk is None or final_risk > allowed_risk_dollars + tolerance:
        err = (f"FINAL RISK REJECTED: {symbol} {signal} volume={final_volume:g} "
               f"entry={final_entry} SL={sl_price} expected_loss={final_risk} "
               f"allowed=${allowed_risk_dollars:.2f}")
        log.error(err)
        return failed(err)
    request['price'] = final_entry
    pre_send_risk = final_risk

    # Re-query account-wide broker truth using the final normalized expected
    # loss. This last entry gate is intentionally adjacent to order_send.
    prop = evaluate_prop_risk(final_risk, log=log)
    if not prop.allowed:
        return failed(f"Broker-authoritative prop loss guard: {prop.reason}")

    result = mt5.order_send(request)
    if result is None:
        return failed("order_send returned None")

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        err = f"order_send failed: retcode={result.retcode} comment={result.comment}"
        log.error(err)
        return failed(err)

    ticket       = result.order
    actual_entry = _confirm_fill_price(ticket, result.price or final_entry, log)
    # Observability is labelled actual only when broker position truth confirms
    # price_open. A request/result fallback remains useful for journalling but
    # must not be presented as a confirmed fill-risk measurement.
    confirmed_entry = None
    try:
        confirmed = mt5.positions_get(ticket=ticket)
        if confirmed and getattr(confirmed[0], 'price_open', 0):
            confirmed_entry = float(confirmed[0].price_open)
    except Exception:
        pass
    actual_risk = (_expected_loss(symbol, signal, final_volume,
                                  confirmed_entry, sl_price)
                   if confirmed_entry is not None else None)
    if actual_risk is None:
        log.error(f"POST-FILL RISK unavailable: {symbol} ticket={ticket} "
                  f"confirmed fill not available; journal_entry={actual_entry} SL={sl_price}")
    else:
        difference = actual_risk - allowed_risk_dollars
        pct_difference = difference / allowed_risk_dollars * 100.0
        log.info(f"POST-FILL RISK: {symbol} ticket={ticket} allowed=${allowed_risk_dollars:.2f} "
                 f"pre_send=${pre_send_risk:.2f} actual_fill_to_SL=${actual_risk:.2f} "
                 f"difference=${difference:+.2f} ({pct_difference:+.2f}%)")

    log.info(f"ORDER PLACED  {symbol} {signal}  {final_volume}L  "
             f"entry={actual_entry:.5f}  SL={sl_price}  TP={tp_price}  "
             f"ticket={ticket}")

    _write_trade_log({
        'Timestamp'  : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        'Pair'       : symbol,
        'Direction'  : signal,
        'Session'    : session.capitalize(),
        'Lots'       : final_volume,
        'EntryPrice' : actual_entry,
        'SL'         : sl_price,
        'TP'         : tp_price,
        'AsianHigh'  : session_data.get('asian_high', ''),
        'AsianLow'   : session_data.get('asian_low',  ''),
        'RangePips'  : session_data.get('range_pips', ''),
        'SLPips'     : sl_pips,
        'TPPips'     : tp_pips,
        'Ticket'     : ticket,
        'Status'     : 'OPEN',
    })

    return {
        'success'     : True,
        'ticket'      : ticket,
        'entry_price' : actual_entry,
        'sl'          : sl_price,
        'tp'          : tp_price,
        'volume'      : final_volume,
        'pre_send_risk': pre_send_risk,
        'actual_risk' : actual_risk,
        'error'       : None,
    }


# ---------------------------------------------------------------------------
# Close trade (used for SMA cross-exit on EURUSD)
# ---------------------------------------------------------------------------

def close_trade(ticket: int, symbol: str, comment: str = 'SMA_cross_exit') -> bool:
    """
    Send a market close order for an open position.
    Returns True if the close order was accepted by MT5.
    """
    log = _log()

    if not _connect(log):
        return False

    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        log.warning(f"close_trade: ticket {ticket} not found in open positions")
        return False

    pos  = positions[0]
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        log.error(f"close_trade: could not get tick for {symbol}")
        return False

    # Close direction is opposite to open direction
    if pos.type == mt5.ORDER_TYPE_BUY:
        close_type  = mt5.ORDER_TYPE_SELL
        close_price = tick.bid
    else:
        close_type  = mt5.ORDER_TYPE_BUY
        close_price = tick.ask

    request = {
        'action'      : mt5.TRADE_ACTION_DEAL,
        'symbol'      : symbol,
        'volume'      : pos.volume,
        'type'        : close_type,
        'price'       : close_price,
        'position'    : ticket,
        'deviation'   : DEVIATION,
        'magic'       : MAGIC_NUMBER,
        'comment'     : comment,
        'type_time'   : mt5.ORDER_TIME_GTC,
        'type_filling': _get_filling_mode(symbol),
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        log.info(f"CLOSE_TRADE  {symbol}  ticket={ticket}  price={close_price:.5f}")
        return True

    rc = result.retcode if result else 'None'
    log.error(f"close_trade failed for ticket {ticket}: retcode={rc}")
    return False


def position_is_open(ticket: int, log: logging.Logger) -> bool | None:
    """Return broker truth for one position, or ``None`` if unavailable.

    This deliberately distinguishes a confirmed empty lookup from connection,
    query, and broker failures.  Scheduled exits must remain retryable whenever
    the terminal cannot prove that a position is gone.
    """
    if not _connect(log):
        return None
    try:
        positions = mt5.positions_get(ticket=ticket)
    except Exception as e:
        log.error(f"Position verification failed for ticket {ticket}: {e}",
                  exc_info=True)
        return None
    if positions is None:
        log.error(f"Position verification failed for ticket {ticket}: "
                  "positions_get returned None")
        return None
    return len(positions) > 0


# ---------------------------------------------------------------------------
# Monitor open positions
# ---------------------------------------------------------------------------

def _apply_breakeven(position, trade: dict, log: logging.Logger) -> bool:
    """
    Move SL to entry price (breakeven) when >= BREAKEVEN_PIPS in profit.
    Returns True if breakeven was applied.
    """
    symbol   = trade['symbol']
    pip_size = PAIRS[symbol]['pip_size']

    if trade['direction'] == 'BUY':
        profit_pips = (position.price_current - position.price_open) / pip_size
    else:
        profit_pips = (position.price_open - position.price_current) / pip_size

    if profit_pips < BREAKEVEN_PIPS:
        return False

    # Only move if current SL is still below/above entry
    entry = position.price_open
    already_be = (
        (trade['direction'] == 'BUY'  and position.sl >= entry) or
        (trade['direction'] == 'SELL' and position.sl <= entry and position.sl > 0)
    )
    if already_be:
        return False

    be_request = {
        'action'  : mt5.TRADE_ACTION_SLTP,
        'position': position.ticket,
        'symbol'  : symbol,
        'sl'      : _price_round(entry, symbol),
        'tp'      : position.tp,
    }
    result = mt5.order_send(be_request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        log.info(f"BREAKEVEN  {symbol}  ticket={position.ticket}  "
                 f"profit={profit_pips:.1f}p  SL moved to {entry:.5f}")
        return True
    else:
        rc = result.retcode if result else 'None'
        log.warning(f"Breakeven failed for ticket {position.ticket}: retcode={rc}")
        return False


def _find_exit_deal(deals, ticket: int):
    """
    Search a sequence of deals for the exit deal matching ticket.
    Matches on position_id first (MT5 5 standard), then on order ticket
    as a fallback (guards against stored-ticket vs position-ticket mismatch).
    Returns the deal object or None.
    """
    # Primary: position_id is the position ticket (equals opening order ticket in MT5 5)
    match = next(
        (d for d in deals
         if d.position_id == ticket and d.entry == mt5.DEAL_ENTRY_OUT),
        None
    )
    if match:
        return match
    # Fallback: match by the closing order ticket in case stored ticket differs
    return next(
        (d for d in deals
         if d.order == ticket and d.entry == mt5.DEAL_ENTRY_OUT),
        None
    )


def _format_exit_deal(deal, position_deals, observed_offset_hours: int) -> dict:
    """Keep legacy gross P&L while carrying separate MT5 accounting fields."""
    accounting = aggregate_position_deals(position_deals, deal.position_id)
    # A history lookup failure must not invent a zero-gross close.  The exit
    # deal is already authoritative for the legacy field in that narrow case.
    if not accounting['deal_count']:
        accounting.update({
            'gross_pnl': round(float(deal.profit or 0.0), 2),
            'commission': round(float(getattr(deal, 'commission', 0.0) or 0.0), 2),
            'swap': round(float(getattr(deal, 'swap', 0.0) or 0.0), 2),
            'fee': round(float(getattr(deal, 'fee', 0.0) or 0.0), 2),
            'deal_count': 1,
        })
        accounting['net_pnl'] = round(
            accounting['gross_pnl'] + accounting['commission']
            + accounting['swap'] + accounting['fee'], 2)
    exit_time = server_epoch_to_utc(deal.time, observed_offset_hours).isoformat()
    return {
        'exit_price'  : deal.price,
        'exit_time'   : exit_time,
        'exit_reason' : ('TP'          if deal.reason == mt5.DEAL_REASON_TP  else
                         'SL'          if deal.reason == mt5.DEAL_REASON_SL  else
                         'MANUAL/OTHER'),
        # exit_pnl deliberately retains its historical GROSS semantics for
        # strategy analytics, health monitoring, and R calculations.
        'exit_pnl'    : accounting['gross_pnl'],
        **accounting,
        'server_offset_h': observed_offset_hours,
    }


def _position_deals(ticket: int, fallback_deals) -> list:
    """Return all deals for a position, falling back to the already-read window."""
    try:
        deals = mt5.history_deals_get(position=ticket)
        if deals is not None:
            return list(deals)
    except Exception:
        pass
    return [d for d in (fallback_deals or []) if getattr(d, 'position_id', None) == ticket]


def _observed_offset_or_fallback(log: logging.Logger) -> int:
    offset = observed_server_utc_offset_hours(mt5)
    if offset is not None:
        return offset
    # This is only a live-path contingency. Historical repair must supply an
    # independently known historical offset and is intentionally not done here.
    fallback = 3 if 3 <= datetime.now(timezone.utc).month <= 10 else 2
    log.warning(f"MT5 server offset could not be observed; using live DST fallback UTC+{fallback}")
    return fallback


def _get_closed_deal(ticket: int, log: logging.Logger) -> dict | None:
    """
    Look up the exit deal for a closed position.

    Uses calendar-day boundaries as search windows rather than
    datetime.now() as the upper bound.  The MT5 broker server clock
    can run several hours ahead of the Windows system clock; using
    datetime.now() as the end time then excludes deals whose timestamps
    are beyond the (lagging) system clock even though they are already
    recorded in MT5 history.  Anchoring to fixed calendar windows avoids
    that problem entirely.

    Search order:
      1. UTC midnight today  ->  midnight + 28 h
         (covers all same-day closes regardless of broker timezone offset)
      2. UTC midnight yesterday  ->  midnight + 28 h
         (catches trades that close in the evening near or past midnight UTC)

    Matches by position_id (primary) and order ticket (fallback) to handle
    any broker-specific difference between the two fields.
    """
    try:
        now      = datetime.now(timezone.utc)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # Pass 1: today's calendar window (midnight UTC -> midnight+28h)
        day_end = midnight + timedelta(hours=28)
        deals = mt5.history_deals_get(midnight, day_end)
        if deals:
            match = _find_exit_deal(deals, ticket)
            if match:
                offset = _observed_offset_or_fallback(log)
                return _format_exit_deal(match, _position_deals(ticket, deals), offset)

        # Pass 2: yesterday's calendar window (catches trades near day boundary)
        yest_midnight = midnight - timedelta(days=1)
        deals = mt5.history_deals_get(yest_midnight, yest_midnight + timedelta(hours=28))
        if deals:
            match = _find_exit_deal(deals, ticket)
            if match:
                log.info(f"ticket {ticket}: exit deal found in yesterday's window")
                offset = _observed_offset_or_fallback(log)
                return _format_exit_deal(match, _position_deals(ticket, deals), offset)

        return None
    except Exception as e:
        log.warning(f"Could not fetch exit deal for ticket {ticket}: {e}")
        return None


def monitor_positions(open_trades: list, log: logging.Logger,
                       friday_tickets: set | None = None) -> tuple:
    """
    Check all tracked open trades.
      - Moves SL to breakeven when >= BREAKEVEN_PIPS in profit
      - Detects positions that have been closed by MT5 (SL/TP hit)

    Args:
        open_trades    : list of trade dicts from daily_state['open_trades']
        log            : logger passed from orchestrator
        friday_tickets : tickets force-closed by the Friday 20:00 UTC close --
                         their exit_reason is reported as 'FRIDAY_CLOSE'
                         instead of 'MANUAL/OTHER' (MT5 records both as a
                         client-side close, so the trigger type can't be
                         told apart from the deal alone).

                         Historical trades closed before this change may
                         still carry the legacy 'EOD_CLOSE' label from the
                         old daily 17:30 UTC forced close -- that label is
                         never rewritten, it just won't be produced for any
                         new close.

    Returns:
        (still_open, newly_closed)
          still_open   -- updated list (breakeven_moved flags updated)
          newly_closed -- list of completed trade dicts with exit details
    """
    if not _connect(log):
        return open_trades, []

    friday_tickets = friday_tickets or set()

    still_open   = []
    newly_closed = []

    for trade in open_trades:
        ticket = trade['ticket']
        symbol = trade['symbol']

        # Look up this position in MT5
        positions = mt5.positions_get(ticket=ticket)

        if positions is None or len(positions) == 0:
            # Position no longer open -- find exit details
            exit_info = _get_closed_deal(ticket, log)
            if exit_info:
                if ticket in friday_tickets and exit_info['exit_reason'] == 'MANUAL/OTHER':
                    exit_info['exit_reason'] = 'FRIDAY_CLOSE'

                closed_trade = {**trade, **exit_info}
                newly_closed.append(closed_trade)

                ledger_record = {
                    'ticket': ticket,
                    'exit_time': exit_info['exit_time'],
                    'gross_pnl': exit_info['gross_pnl'],
                    'commission': exit_info['commission'],
                    'swap': exit_info['swap'],
                    'fee': exit_info['fee'],
                    'net_pnl': exit_info['net_pnl'],
                    'deal_count': exit_info['deal_count'],
                    'server_offset_h': exit_info['server_offset_h'],
                }
                if not append_cost_record(ledger_record):
                    log.info(f"Accounting ledger already contains or could not save ticket={ticket}")

                _write_trade_log({
                    'Timestamp'  : exit_info['exit_time'],
                    'Pair'       : symbol,
                    'Direction'  : trade['direction'],
                    'Session'    : trade['session'],
                    'Lots'       : trade['lots'],
                    'EntryPrice' : trade['entry_price'],
                    'SL'         : trade['sl'],
                    'TP'         : trade['tp'],
                    'AsianHigh'  : trade.get('asian_high', ''),
                    'AsianLow'   : trade.get('asian_low', ''),
                    'RangePips'  : trade.get('sl_pips', 0) * 2,
                    'SLPips'     : trade.get('sl_pips', ''),
                    'TPPips'     : trade.get('tp_pips', ''),
                    'Ticket'     : ticket,
                    'Status'     : 'CLOSED',
                    'ExitPrice'  : exit_info['exit_price'],
                    'ExitTime'   : exit_info['exit_time'],
                    'ExitReason' : exit_info['exit_reason'],
                    'PnL'        : exit_info['exit_pnl'],
                })
            else:
                retry = trade.get('close_retry', 0) + 1
                if retry >= MAX_CLOSE_RETRIES:
                    # Exhausted retries -- emit as UNKNOWN CLOSE so orchestrator
                    # removes it from open_trades and logs it; P&L is indeterminate
                    # (real balance impact visible on next agent_market balance read)
                    log.warning(
                        f"UNKNOWN CLOSE: ticket={ticket} {symbol} {trade['direction']} -- "
                        f"not in positions and no exit deal after {retry} attempts -- "
                        f"removing from monitoring"
                    )
                    now_str = datetime.now(timezone.utc).isoformat()
                    newly_closed.append({
                        **trade,
                        'exit_price'  : 0.0,
                        'exit_time'   : now_str,
                        'exit_reason' : 'UNKNOWN',
                        'exit_pnl'    : 0.0,
                        'gross_pnl'   : None,
                        'commission'  : None,
                        'swap'        : None,
                        'fee'         : None,
                        'net_pnl'     : None,
                        'accounting_coverage': 'incomplete',
                    })
                else:
                    log.warning(
                        f"ticket {ticket} not in positions, exit deal not found -- "
                        f"retry {retry}/{MAX_CLOSE_RETRIES}"
                    )
                    still_open.append({**trade, 'close_retry': retry})
            continue

        # Position is still open
        pos = positions[0]

        # Apply breakeven if not already done and profit threshold reached.
        # EXCLUDED for validated-book trades (strategy_key contains '@',
        # i.e. @arb/@amr/@mon): the phase-7 exit study showed baseline
        # SL/TP beats breakeven moves on 5 of 6 book strategies, and the
        # walk-forward validations were run WITHOUT any breakeven -- live
        # must match backtest. (Observed live 2026-07-10: the legacy BE
        # rule turned a +25p CADJPY@arb into a -$4.75 scratch instead of
        # letting it run to its 2:1 target.) Legacy-style trades keep the
        # original behavior.
        if ('@' not in str(trade.get('strategy_key', ''))
                and not trade.get('breakeven_moved', False)):
            moved = _apply_breakeven(pos, trade, log)
            if moved:
                trade = {**trade, 'breakeven_moved': True}

        still_open.append(trade)

    return still_open, newly_closed


def find_untracked_positions(open_trades: list, log: logging.Logger,
                             strict: bool = False) -> list:
    """
    Reconciliation check (2026-08-03): monitor_positions() above is
    one-directional -- it only detects "bot thinks a ticket is open, MT5
    shows it closed." It never detects the reverse: an MT5 position this
    bot placed (magic == MAGIC_NUMBER) that ISN'T in open_trades, e.g.
    after a crash/restart where state wasn't carried forward, or some
    other desync. That gap is exactly the class of bug a Reddit reviewer
    flagged and the July cross-terminal incident made concrete.

    Filters on magic number specifically so a manually-placed trade in
    the same MT5 account (different/zero magic) is not treated as a bot
    desync -- this function is read-only and never closes anything;
    callers decide what to do with the result.

    Returns a list of plain dicts (ticket, symbol, direction, lots,
    entry_price, sl, tp, open_time) for every untracked bot position.
    """
    if not _connect(log):
        if strict:
            raise RuntimeError("MT5 connection failed during reconciliation")
        return []
    known_tickets = {t['ticket'] for t in open_trades}
    positions = mt5.positions_get()
    if positions is None:
        if strict:
            raise RuntimeError(f"positions_get failed during reconciliation: {mt5.last_error()}")
        return []
    if not positions:
        return []

    untracked = []
    for pos in positions:
        if pos.magic != MAGIC_NUMBER or pos.ticket in known_tickets:
            continue
        untracked.append({
            'ticket'      : pos.ticket,
            'symbol'      : pos.symbol,
            'direction'   : 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
            'lots'        : pos.volume,
            'entry_price' : pos.price_open,
            'sl'          : pos.sl,
            'tp'          : pos.tp,
            'open_time'   : datetime.fromtimestamp(pos.time, tz=timezone.utc).isoformat(),
        })
    return untracked
