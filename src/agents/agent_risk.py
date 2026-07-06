"""
Agent 3 -- Risk Management
===========================
Called before every trade attempt.

Checks (in order):
  1. Hard floor -- are balance AND equity above $90,000? (equity matters:
     a large floating loss can breach the prop-firm floor before any
     trade is closed, so balance alone is blind to it)
  2. Daily loss limit -- has the 5% ($5,000) daily cap been hit?
     Two layers: (a) closed-trade daily_pnl (original check) and
     (b) equity drawdown vs the day's first-seen equity anchor, with a
     4% SOFT stop -- trading halts before the firm's 5% hard breach,
     leaving room for open-position slippage.
  3. Consecutive losses -- has the 2-loss per-pair limit been hit today?
  4. News flag -- is a high-impact event flagged for this session?
  5. Aggregate open risk -- sum of (entry -> SL) risk across all open
     positions plus the new trade must stay under 3% of balance, so a
     single bad day where every open position stops out cannot come
     close to the 5% daily limit on its own. Any open position missing
     an SL blocks new trades entirely.
  6. Currency concentration -- max 2 open positions sharing one currency
     (e.g. GBPJPY + EURJPY = 2 JPY exposures; a third JPY trade is
     rejected as one adverse JPY move would hit all three at once).

If all checks pass, calculates dynamic lot size:
  Risk per trade  = 1% of current balance (= 20% of 5% daily limit)
  Pip value       = derived live from MT5 symbol_info (account-currency aware)
  Lot size        = risk_usd / (sl_pips x pip_value_per_lot)
  Hard cap        = 2.0 lots maximum

Returns APPROVED with lot size, or REJECTED with reason.
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

# -- logging
LOGS_DIR = Path(__file__).parent.parent.parent / 'data' / 'logs'

def _log() -> logging.Logger:
    log = logging.getLogger('RISK')
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
STARTING_BALANCE   = 100_000.00
HARD_FLOOR         = 90_000.00
MAX_DAILY_LOSS_PCT = 0.05        # 5% -- the prop firm's hard daily limit
DAILY_EQUITY_SOFT_STOP_PCT = 0.04  # halt NEW trades at -4% equity on the day,
                                   # leaving a 1% buffer of open-position
                                   # slippage before the firm's 5% breach
MAX_TOTAL_OPEN_RISK_PCT = 0.03   # aggregate entry->SL risk across all open
                                 # positions + the new trade, as pct of balance
MAX_SAME_CURRENCY  = 2           # max open positions sharing one currency
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

    # gold: 1 'pip' = $0.10 (matches strategies + agent_execution PAIRS).
    # Computed from contract size, NOT trade_tick_value -- MetaQuotes
    # reports inconsistent tick values for metals when the market is
    # closed (observed: tick_value 0.1 vs contract 100 oz). For a
    # USD-quoted metal, pip value per lot = contract_size x pip_size
    # exactly ($10 for 100 oz x $0.10).
    if symbol.startswith('XAU'):
        return info.trade_contract_size * 0.1

    pip_size = 0.01 if 'JPY' in symbol else 0.0001
    tick_size  = info.trade_tick_size     # smallest price move
    tick_value = info.trade_tick_value    # value of one tick per lot in account currency

    if tick_size == 0:
        return 10.00   # safety fallback

    return (pip_size / tick_size) * tick_value


# ---------------------------------------------------------------------------
# Portfolio-level risk helpers
# ---------------------------------------------------------------------------

def _open_risk_usd(log: logging.Logger) -> tuple:
    """
    Sum the worst-case loss (entry -> SL) across every open position, in
    account currency. Returns (total_risk_usd, positions_without_sl).

    A position with no SL has unbounded risk -- callers should treat
    any positions_without_sl > 0 as a blocker for new trades.
    """
    positions = mt5.positions_get()
    if positions is None or len(positions) == 0:
        return 0.0, 0

    total, no_sl = 0.0, 0
    for p in positions:
        if not p.sl:
            no_sl += 1
            continue
        info = mt5.symbol_info(p.symbol)
        if info is None or info.trade_tick_size == 0:
            log.warning(f"  open-risk calc: no symbol_info for {p.symbol} -- "
                        f"counting as unbounded")
            no_sl += 1
            continue
        risk = (abs(p.price_open - p.sl) / info.trade_tick_size
                * info.trade_tick_value * p.volume)
        total += risk
    return total, no_sl


def _same_currency_count(symbol: str) -> int:
    """Number of open positions sharing either currency with `symbol`
    (e.g. for GBPJPY: any open *JPY* or *GBP* position counts)."""
    positions = mt5.positions_get()
    if positions is None:
        return 0
    base, quote = symbol[:3], symbol[3:6]
    n = 0
    for p in positions:
        sym = p.symbol[:6]
        if base in (sym[:3], sym[3:6]) or quote in (sym[:3], sym[3:6]):
            n += 1
    return n


# ---------------------------------------------------------------------------
# Lot size calculation
# ---------------------------------------------------------------------------

def _calc_lots(balance: float, sl_pips: float, symbol: str,
               log: logging.Logger, risk_pct: float | None = None) -> float:
    """
    Lot size = risk_usd / (sl_pips x pip_value_per_lot)
    Clamped to [MIN_LOT, MAX_LOT].
    risk_pct: the strategy's own risk fraction (from its pair YAML
    risk_percent), passed by the orchestrator. Fallback when None
    preserves legacy behaviour: EURUSD 0.25%, others 1%.
    """
    if risk_pct is None:
        risk_pct = EURUSD_RISK_PCT if symbol == 'EURUSD' else RISK_PER_TRADE_PCT
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

def run(symbol: str, signal: str, sl_pips: float, daily_state: dict,
        risk_pct: float | None = None) -> dict:
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
    equity  = acct.equity

    # -- 2. Hard floor check (balance AND equity -- floating losses count)
    if balance <= HARD_FLOOR:
        return reject(f"Hard floor breached: balance ${balance:,.2f} <= ${HARD_FLOOR:,.0f}")
    if equity <= HARD_FLOOR:
        return reject(f"Hard floor breached on EQUITY: ${equity:,.2f} <= "
                      f"${HARD_FLOOR:,.0f} (floating losses)")

    # -- 3a. Daily loss limit (closed trades -- original check)
    daily_limit = balance * MAX_DAILY_LOSS_PCT
    daily_pnl   = daily_state.get('daily_pnl', 0.0)
    if daily_pnl <= -daily_limit:
        return reject(f"Daily loss limit hit: ${daily_pnl:,.2f} (limit -${daily_limit:,.0f})")

    # -- 3b. Daily EQUITY drawdown soft stop (includes floating P&L).
    # Anchor = first equity seen today. daily_state is rebuilt fresh at each
    # UTC date rollover, so the anchor resets daily. Anchored at the first
    # risk call of the day rather than exactly 00:00 -- if equity has RISEN
    # since midnight this is stricter than the firm's rule (anchor is
    # higher), never looser than measuring from midnight balance would be
    # in a losing day, because losses only lower equity below any anchor.
    anchor = daily_state.get('day_start_equity')
    if not anchor:
        daily_state['day_start_equity'] = equity
        anchor = equity
    equity_dd = (anchor - equity) / anchor if anchor > 0 else 0.0
    if equity_dd >= DAILY_EQUITY_SOFT_STOP_PCT:
        return reject(
            f"Daily equity soft stop: equity ${equity:,.2f} is "
            f"{equity_dd*100:.2f}% below today's anchor ${anchor:,.2f} "
            f"(soft stop {DAILY_EQUITY_SOFT_STOP_PCT*100:.0f}%, firm limit "
            f"{MAX_DAILY_LOSS_PCT*100:.0f}%)")

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

    # -- 7. Aggregate open risk cap (portfolio-level)
    open_risk, no_sl = _open_risk_usd(log)
    if no_sl > 0:
        return reject(f"{no_sl} open position(s) have NO stop loss -- "
                      f"unbounded book risk, no new trades until fixed")
    eff_pct   = risk_pct if risk_pct is not None else (
        EURUSD_RISK_PCT if symbol == 'EURUSD' else RISK_PER_TRADE_PCT)
    new_risk  = balance * eff_pct
    max_total = balance * MAX_TOTAL_OPEN_RISK_PCT
    if open_risk + new_risk > max_total:
        return reject(
            f"Aggregate open risk cap: ${open_risk:,.0f} open + "
            f"${new_risk:,.0f} new > ${max_total:,.0f} "
            f"({MAX_TOTAL_OPEN_RISK_PCT*100:.0f}% of balance)")

    # -- 8. Currency concentration cap
    shared = _same_currency_count(symbol)
    if shared >= MAX_SAME_CURRENCY:
        return reject(
            f"Currency concentration: {shared} open position(s) already "
            f"share a currency with {symbol} (max {MAX_SAME_CURRENCY})")

    # -- 9. High-impact news blackout (5ers: no entries +/-5 min of news)
    try:
        from core.news_calendar import is_blackout
        blocked, news_reason = is_blackout(symbol)
        if blocked:
            return reject(f"News blackout: {news_reason}")
    except Exception as e:
        log.warning(f"news blackout check failed ({e}) -- proceeding")

    # -- All checks passed -- calculate lot size
    lots = _calc_lots(balance, sl_pips, symbol, log, risk_pct)

    log.info(f"Risk APPROVED: {symbol} {signal}  {lots:.2f}L  "
             f"SL={sl_pips:.1f}p  balance=${balance:,.2f}")

    return approve(lots)
