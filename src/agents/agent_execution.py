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
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import time

import MetaTrader5 as mt5

# -- paths
AGENTS_DIR  = Path(__file__).parent
BASE_DIR    = AGENTS_DIR.parent.parent
DATA_DIR    = BASE_DIR / 'data'
LOGS_DIR    = DATA_DIR / 'logs'
TRADES_LOG  = DATA_DIR / 'trades_log.csv'

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


def _price_round(price: float, symbol: str) -> float:
    digits = PAIRS[symbol]['digits']
    return round(price, digits)


def _get_live_price(symbol: str, signal: str) -> float | None:
    """Return ask (BUY) or bid (SELL) for market order."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None
    return tick.ask if signal == 'BUY' else tick.bid


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


# ---------------------------------------------------------------------------
# Place trade
# ---------------------------------------------------------------------------

def place_trade(symbol: str, breakout: dict, lot_size: float,
                session_data: dict, session: str) -> dict:
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

    if not _connect(log):
        return failed("MT5 connection failed")

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

    comment = f"5ers_{session}_{signal}_{strategy}" if strategy else f"5ers_{session}_{signal}"

    request = {
        'action'      : mt5.TRADE_ACTION_DEAL,
        'symbol'      : symbol,
        'volume'      : lot_size,
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

    result = mt5.order_send(request)
    if result is None:
        return failed("order_send returned None")

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        err = f"order_send failed: retcode={result.retcode} comment={result.comment}"
        log.error(err)
        return failed(err)

    ticket       = result.order
    actual_entry = result.price

    log.info(f"ORDER PLACED  {symbol} {signal}  {lot_size}L  "
             f"entry={actual_entry:.5f}  SL={sl_price}  TP={tp_price}  "
             f"ticket={ticket}")

    _write_trade_log({
        'Timestamp'  : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        'Pair'       : symbol,
        'Direction'  : signal,
        'Session'    : session.capitalize(),
        'Lots'       : lot_size,
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


def _format_exit_deal(deal) -> dict:
    return {
        'exit_price'  : deal.price,
        'exit_time'   : datetime.fromtimestamp(deal.time, tz=timezone.utc).isoformat(),
        'exit_reason' : ('TP'          if deal.reason == mt5.DEAL_REASON_TP  else
                         'SL'          if deal.reason == mt5.DEAL_REASON_SL  else
                         'MANUAL/OTHER'),
        'exit_pnl'    : deal.profit,
    }


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
                return _format_exit_deal(match)

        # Pass 2: yesterday's calendar window (catches trades near day boundary)
        yest_midnight = midnight - timedelta(days=1)
        deals = mt5.history_deals_get(yest_midnight, yest_midnight + timedelta(hours=28))
        if deals:
            match = _find_exit_deal(deals, ticket)
            if match:
                log.info(f"ticket {ticket}: exit deal found in yesterday's window")
                return _format_exit_deal(match)

        return None
    except Exception as e:
        log.warning(f"Could not fetch exit deal for ticket {ticket}: {e}")
        return None


def monitor_positions(open_trades: list, log: logging.Logger,
                       eod_tickets: set | None = None) -> tuple:
    """
    Check all tracked open trades.
      - Moves SL to breakeven when >= BREAKEVEN_PIPS in profit
      - Detects positions that have been closed by MT5 (SL/TP hit)

    Args:
        open_trades : list of trade dicts from daily_state['open_trades']
        log         : logger passed from orchestrator
        eod_tickets : tickets force-closed by the 17:30 UTC EOD close --
                      their exit_reason is reported as 'EOD_CLOSE' instead
                      of 'MANUAL/OTHER' (MT5 records both as a client-side
                      close, so the trigger type can't be told apart from
                      the deal alone)

    Returns:
        (still_open, newly_closed)
          still_open   -- updated list (breakeven_moved flags updated)
          newly_closed -- list of completed trade dicts with exit details
    """
    if not _connect(log):
        return open_trades, []

    eod_tickets = eod_tickets or set()

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
                if ticket in eod_tickets and exit_info['exit_reason'] == 'MANUAL/OTHER':
                    exit_info['exit_reason'] = 'EOD_CLOSE'

                closed_trade = {**trade, **exit_info}
                newly_closed.append(closed_trade)

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

        # Apply breakeven if not already done and profit threshold reached
        if not trade.get('breakeven_moved', False):
            moved = _apply_breakeven(pos, trade, log)
            if moved:
                trade = {**trade, 'breakeven_moved': True}

        still_open.append(trade)

    return still_open, newly_closed
