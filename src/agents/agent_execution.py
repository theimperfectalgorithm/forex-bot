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
        fh = logging.FileHandler(LOGS_DIR / 'trading.log', encoding='utf-8')
        fh.setFormatter(fmt)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        log.addHandler(fh)
        log.addHandler(ch)
        log.setLevel(logging.INFO)
    return log


# -- constants
MAGIC_NUMBER    = 200001   # unique identifier for all system trades
DEVIATION       = 20       # max slippage in points
BREAKEVEN_PIPS  = 25       # move SL to entry when this many pips in profit

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


# ---------------------------------------------------------------------------
# Place trade
# ---------------------------------------------------------------------------

def place_trade(symbol: str, breakout: dict, lot_size: float,
                session_data: dict, session: str) -> dict:
    """
    Place a market order on MT5.

    SL and TP are anchored to the Asian range breakout level (not entry),
    matching the backtest design:
      BUY:  entry ~ asian_high, SL = asian_high - sl_pips, TP = asian_high + tp_pips
      SELL: entry ~ asian_low,  SL = asian_low  + sl_pips, TP = asian_low  - tp_pips

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

    # Ensure the symbol is visible in the Market Watch
    if not mt5.symbol_select(symbol, True):
        return failed(f"symbol_select({symbol}) failed")

    signal      = breakout['signal']
    asian_high  = session_data['asian_high']
    asian_low   = session_data['asian_low']
    sl_pips     = session_data['sl_pips']
    tp_pips     = session_data['tp_pips']
    pip_size    = PAIRS[symbol]['pip_size']

    # Anchor levels come from the Asian range (not live market price)
    if signal == 'BUY':
        anchor     = asian_high
        sl_price   = _price_round(anchor - sl_pips * pip_size, symbol)
        tp_price   = _price_round(anchor + tp_pips * pip_size, symbol)
        order_type = mt5.ORDER_TYPE_BUY
    else:
        anchor     = asian_low
        sl_price   = _price_round(anchor + sl_pips * pip_size, symbol)
        tp_price   = _price_round(anchor - tp_pips * pip_size, symbol)
        order_type = mt5.ORDER_TYPE_SELL

    # Live entry price (ask/bid at moment of order)
    entry_price = _get_live_price(symbol, signal)
    if entry_price is None:
        return failed(f"Could not get live price for {symbol}")

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
        'comment'     : f'5ers_{session}_{signal}',
        'type_time'   : mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_IOC,
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

    # -- Log to CSV (entry row -- exit fields blank until close)
    _write_trade_log({
        'Timestamp'  : datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
        'Pair'       : symbol,
        'Direction'  : signal,
        'Session'    : session.capitalize(),
        'Lots'       : lot_size,
        'EntryPrice' : actual_entry,
        'SL'         : sl_price,
        'TP'         : tp_price,
        'AsianHigh'  : asian_high,
        'AsianLow'   : asian_low,
        'RangePips'  : session_data['range_pips'],
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


def _get_closed_deal(ticket: int, log: logging.Logger) -> dict | None:
    """
    Look up the exit deal for a closed position in MT5 history.
    Searches the last 24 hours.
    """
    try:
        from_time = datetime.now(timezone.utc) - timedelta(hours=24)
        deals = mt5.history_deals_get(from_time, datetime.now(timezone.utc))
        if deals is None:
            return None

        # DEAL_ENTRY_OUT = 1 (exit deal)
        exit_deal = next(
            (d for d in deals
             if d.position_id == ticket and d.entry == mt5.DEAL_ENTRY_OUT),
            None
        )
        if exit_deal is None:
            return None

        return {
            'exit_price'  : exit_deal.price,
            'exit_time'   : datetime.fromtimestamp(exit_deal.time,
                                                   tz=timezone.utc).isoformat(),
            'exit_reason' : 'TP' if exit_deal.reason == mt5.DEAL_REASON_TP
                            else ('SL' if exit_deal.reason == mt5.DEAL_REASON_SL
                                  else 'MANUAL/OTHER'),
            'exit_pnl'    : exit_deal.profit,
        }
    except Exception as e:
        log.warning(f"Could not fetch exit deal for ticket {ticket}: {e}")
        return None


def monitor_positions(open_trades: list, log: logging.Logger) -> tuple:
    """
    Check all tracked open trades.
      - Moves SL to breakeven when >= BREAKEVEN_PIPS in profit
      - Detects positions that have been closed by MT5 (SL/TP hit)

    Args:
        open_trades : list of trade dicts from daily_state['open_trades']
        log         : logger passed from orchestrator

    Returns:
        (still_open, newly_closed)
          still_open   -- updated list (breakeven_moved flags updated)
          newly_closed -- list of completed trade dicts with exit details
    """
    if not _connect(log):
        return open_trades, []

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
                closed_trade = {**trade, **exit_info}
                newly_closed.append(closed_trade)

                # Update the CSV row with exit details
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
                # Can't find exit -- keep in list to retry next cycle
                log.warning(f"ticket {ticket} not in positions but no exit deal found -- retrying")
                still_open.append(trade)
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
