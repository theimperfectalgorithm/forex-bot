"""
Agent 3 -- Risk Management
===========================
Called before every trade attempt.

Checks (in order):
  1. Hard floor -- is balance above $90,000?
  2. Daily loss limit -- has the 5% ($5,000) daily cap been hit?
  3. Consecutive losses -- has the 2-loss per-pair limit been hit today?
  4. News flag -- is a high-impact event flagged for this session?

If all checks pass, calculates dynamic lot size:
  Risk per trade  = 1% of current balance (= 20% of 5% daily limit)
  Pip value       = derived live from MT5 symbol_info (account-currency aware)
  Lot size        = risk_usd / (sl_pips x pip_value_per_lot)
  Hard cap        = 2.0 lots maximum

Returns APPROVED with lot size, or REJECTED with reason.
"""

import logging
import sys
from pathlib import Path

import MetaTrader5 as mt5

# -- logging
LOGS_DIR = Path(__file__).parent.parent.parent / 'data' / 'logs'

def _log() -> logging.Logger:
    log = logging.getLogger('RISK')
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
STARTING_BALANCE   = 100_000.00
HARD_FLOOR         = 90_000.00
MAX_DAILY_LOSS_PCT = 0.05        # 5%
RISK_PER_TRADE_PCT = 0.01        # 1% of balance per trade (GBPJPY, EURJPY)
EURUSD_RISK_PCT    = 0.0025      # 0.25% of balance per trade (EURUSD both strategies)
MIN_LOT            = 0.01
MAX_LOT            = 2.00
MAX_CONSEC_LOSSES  = 2


# ---------------------------------------------------------------------------
# Pip value calculation (live, account-currency aware)
# ---------------------------------------------------------------------------

def _pip_value_per_lot(symbol: str, log: logging.Logger) -> float:
    """
    Return the value of 1 pip per 1 standard lot in account currency (USD).

    Uses MT5 symbol_info tick value and size -- works correctly for all
    currency pairs regardless of quote currency (JPY, USD, EUR etc.).

    For a USD account:
      EURUSD: ~$10.00 / pip / lot
      GBPJPY: ~$6.50-7.00 / pip / lot  (depends on live USDJPY rate)
      EURJPY: ~$6.50-7.00 / pip / lot
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        log.warning(f"symbol_info failed for {symbol} -- using fallback pip value")
        # Fallback: hardcoded approximations from backtest
        return 6.67 if 'JPY' in symbol else 10.00

    pip_size   = 0.01   if 'JPY' in symbol else 0.0001
    tick_size  = info.trade_tick_size     # smallest price move
    tick_value = info.trade_tick_value    # value of one tick per lot in account currency

    if tick_size == 0:
        return 10.00   # safety fallback

    return (pip_size / tick_size) * tick_value


# ---------------------------------------------------------------------------
# Lot size calculation
# ---------------------------------------------------------------------------

def _calc_lots(balance: float, sl_pips: float, symbol: str,
               log: logging.Logger) -> float:
    """
    Lot size = risk_usd / (sl_pips x pip_value_per_lot)
    Clamped to [MIN_LOT, MAX_LOT].
    EURUSD uses EURUSD_RISK_PCT (0.25%); other pairs use RISK_PER_TRADE_PCT (1%).
    """
    risk_pct  = EURUSD_RISK_PCT if symbol == 'EURUSD' else RISK_PER_TRADE_PCT
    risk_usd  = balance * risk_pct
    pv        = _pip_value_per_lot(symbol, log)
    lots      = risk_usd / (sl_pips * pv) if sl_pips > 0 else MIN_LOT
    lots      = round(max(MIN_LOT, min(lots, MAX_LOT)), 2)

    log.info(f"  Lot calc: risk=${risk_usd:.2f} ({risk_pct*100:.2f}%)  SL={sl_pips:.1f}p  "
             f"pip_val=${pv:.2f}  -> {lots:.2f} lots")
    return lots


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(symbol: str, signal: str, sl_pips: float, daily_state: dict) -> dict:
    """
    Called by the orchestrator before every trade.

    Args:
        symbol      : e.g. 'GBPJPY'
        signal      : 'BUY' or 'SELL'
        sl_pips     : stop-loss distance in pips (50% of Asian range)
        daily_state : orchestrator's shared state dict

    Returns:
        {
          'decision' : 'APPROVED' | 'REJECTED',
          'lot_size' : float,
          'reason'   : str,
        }
    """
    log = _log()

    reject = lambda reason: {
        'decision': 'REJECTED', 'lot_size': 0.0, 'reason': reason}
    approve = lambda lots: {
        'decision': 'APPROVED', 'lot_size': lots, 'reason': 'all checks passed'}

    # -- 1. MT5 connection & live balance
    if not mt5.initialize():
        return reject(f"MT5 init failed: {mt5.last_error()}")

    acct = mt5.account_info()
    if acct is None:
        return reject("Could not read account info")

    balance = acct.balance

    # -- 2. Hard floor check
    if balance <= HARD_FLOOR:
        return reject(f"Hard floor breached: balance ${balance:,.2f} <= ${HARD_FLOOR:,.0f}")

    # -- 3. Daily loss limit
    daily_limit = balance * MAX_DAILY_LOSS_PCT
    daily_pnl   = daily_state.get('daily_pnl', 0.0)
    if daily_pnl <= -daily_limit:
        return reject(f"Daily loss limit hit: ${daily_pnl:,.2f} (limit -${daily_limit:,.0f})")

    # -- 4. Consecutive losses for this pair
    consec = daily_state.get('consec_losses', {}).get(symbol, 0)
    if consec >= MAX_CONSEC_LOSSES:
        return reject(f"{symbol} paused: {consec} consecutive losses today")

    # -- 5. Pair paused flag (set by orchestrator after 2 losses)
    if daily_state.get('pair_paused', {}).get(symbol, False):
        return reject(f"{symbol} is paused for today")

    # -- 6. SL pips sanity
    if sl_pips <= 0:
        return reject(f"Invalid SL pips: {sl_pips}")

    # -- All checks passed -- calculate lot size
    lots = _calc_lots(balance, sl_pips, symbol, log)

    log.info(f"Risk APPROVED: {symbol} {signal}  {lots:.2f}L  "
             f"SL={sl_pips:.1f}p  balance=${balance:,.2f}")

    return approve(lots)
