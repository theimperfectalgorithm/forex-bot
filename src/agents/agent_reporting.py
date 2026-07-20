"""
Agent 5 -- Daily Reporting
===========================
Runs at 21:00 UTC each day.

Responsibilities:
  - Compile the full day's trading summary from daily_state
  - Append a row to data/equity_curve.csv
  - Write data/daily_report.txt with a human-readable summary
  - Flag any unusual behaviour: consecutive losses, drawdown, system errors
  - Run a daily P&L reconciliation against actual MT5 balance change
  - Run a month-to-date reconciliation to surface unlogged exits early

Reads the orchestrator's daily_state dict directly (no MT5 calls needed
for reporting -- all data is already in state, except for live balance).

Exit reason labels: the TRADE DETAIL section below prints whatever
exit_reason string agent_execution.py already assigned to each trade
(SL, TP, FRIDAY_CLOSE, MANUAL/OTHER, UNKNOWN, or -- on trades closed
before the daily-EOD-close removal -- the legacy EOD_CLOSE label). This
module does no reason-specific branching, so both old and new labels
display correctly with no code path needed for either one.
"""

import csv
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import time

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

# -- paths
AGENTS_DIR    = Path(__file__).parent
BASE_DIR      = AGENTS_DIR.parent.parent
DATA_DIR      = BASE_DIR / 'data'
LOGS_DIR      = DATA_DIR / 'logs'
EQUITY_CSV    = DATA_DIR / 'equity_curve.csv'
REPORT_TXT    = DATA_DIR / 'daily_report.txt'
TRADES_CSV    = DATA_DIR / 'trades_log.csv'

# -- repo root on path so core.* is importable regardless of who imports
# this module (mirrors agent_strategy.py's own path setup)
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# -- constants: SINGLE SOURCE OF TRUTH in core/account_config.py.
# INCIDENT 2026-07-19: this module's STARTING_BALANCE was hardcoded at
# $100,000 while the $5,000 challenge clone's real balance is $5,000 --
# every reconciliation compared against the wrong baseline and reported
# a nonsensical "-$95,000 / 95% drawdown". Now derived from the same
# starting_balance every other agent uses.
from core.account_config import (STARTING_BALANCE, TARGET_BALANCE,
                                 HARD_FLOOR, MAX_DAILY_LOSS_PCT)
RECON_TOLERANCE    = 1.00   # dollar tolerance for P&L reconciliation

EQUITY_HEADERS = [
    'Date', 'Balance', 'DailyPnL', 'DailyPct',
    'Trades', 'Wins', 'Losses', 'WinRate', 'CumReturn',
]


# -- logging
def _log() -> logging.Logger:
    log = logging.getLogger('REPORTING')
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


# ---------------------------------------------------------------------------
# Equity curve CSV
# ---------------------------------------------------------------------------

def _append_equity_curve(row: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = EQUITY_CSV.exists()
    with open(EQUITY_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=EQUITY_HEADERS, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Reconciliation helpers
# ---------------------------------------------------------------------------

def _read_logged_pnl_for_date(date_str: str) -> float:
    """
    Sum PnL of all CLOSED trades in trades_log.csv whose ExitTime falls on
    date_str (format: 'YYYY-MM-DD').  Works for both date-only prefix matching
    ('2026-06-04') and month-only ('2026-06') for monthly totals.
    """
    if not TRADES_CSV.exists():
        return 0.0
    total = 0.0
    with open(TRADES_CSV, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('Status') != 'CLOSED':
                continue
            exit_time = (row.get('ExitTime') or '').strip()
            if exit_time.startswith(date_str):
                try:
                    total += float(row.get('PnL') or 0)
                except (ValueError, TypeError):
                    pass
    return round(total, 2)


def _read_prev_closing_balance(today: str) -> float:
    """
    Return the most recent Balance recorded in equity_curve.csv whose Date
    is NOT today.  Falls back to STARTING_BALANCE if the file is empty or
    missing (first trading day).
    """
    if not EQUITY_CSV.exists():
        return STARTING_BALANCE
    rows = []
    with open(EQUITY_CSV, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('Date', '') != today:
                rows.append(row)
    if not rows:
        return STARTING_BALANCE
    try:
        return float(rows[-1]['Balance'])
    except (KeyError, ValueError):
        return STARTING_BALANCE


def _reconciliation_check(date_str: str, logged_pnl: float,
                           current_balance: float, prev_balance: float,
                           log: logging.Logger) -> dict:
    """
    Compare logged P&L (sum of CSV trades) against actual balance change
    (current MT5 balance minus yesterday's closing balance).

    Returns a dict with: matched, logged_pnl, actual_change, discrepancy,
    prev_balance, current_balance.
    """
    actual_change = round(current_balance - prev_balance, 2)
    discrepancy   = round(actual_change - logged_pnl, 2)
    matched       = abs(discrepancy) <= RECON_TOLERANCE

    if not matched:
        log.warning(
            f"RECONCILIATION MISMATCH on {date_str}: "
            f"logged_pnl={logged_pnl:+.2f}  actual_change={actual_change:+.2f}  "
            f"discrepancy={discrepancy:+.2f} -- likely unlogged trade exits "
            f"(manual closes, crashes, or restarts); review MT5 history"
        )

    return {
        'matched'        : matched,
        'logged_pnl'     : logged_pnl,
        'actual_change'  : actual_change,
        'discrepancy'    : discrepancy,
        'prev_balance'   : prev_balance,
        'current_balance': current_balance,
    }


def _monthly_reconciliation_lines(month_str: str, current_balance: float) -> list:
    """
    Build the MONTH-TO-DATE RECONCILIATION report section.

    Compares:
      - Logged P&L: sum of all CLOSED trades in trades_log.csv for month_str
      - Actual P&L: current_balance minus the last equity_curve balance before
                    this month (or STARTING_BALANCE if no prior row exists)

    Returns a list of report-text lines ready to append to the daily report.
    """
    # Logged P&L: sum every CLOSED trade whose ExitTime is in this month
    monthly_logged = 0.0
    trade_count    = 0
    if TRADES_CSV.exists():
        with open(TRADES_CSV, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                if row.get('Status') != 'CLOSED':
                    continue
                exit_time = (row.get('ExitTime') or '').strip()
                if exit_time.startswith(month_str):
                    try:
                        monthly_logged += float(row.get('PnL') or 0)
                        trade_count    += 1
                    except (ValueError, TypeError):
                        pass
    monthly_logged = round(monthly_logged, 2)

    # Start-of-month balance: last equity_curve row whose Date is before this month
    start_balance = STARTING_BALANCE
    if EQUITY_CSV.exists():
        with open(EQUITY_CSV, newline='', encoding='utf-8') as f:
            prior_rows = [r for r in csv.DictReader(f)
                          if not r.get('Date', '').startswith(month_str)]
        if prior_rows:
            try:
                start_balance = float(prior_rows[-1]['Balance'])
            except (KeyError, ValueError):
                pass

    monthly_actual = round(current_balance - start_balance, 2)
    discrepancy    = round(monthly_actual - monthly_logged, 2)
    matched        = abs(discrepancy) <= RECON_TOLERANCE

    lines = [
        "",
        "  MONTH-TO-DATE RECONCILIATION",
        f"  Month                : {month_str}",
        f"  Logged trades P&L    : ${monthly_logged:>+10,.2f}   ({trade_count} closed trades in CSV)",
        f"  Actual P&L (vs bal)  : ${monthly_actual:>+10,.2f}   "
        f"(${current_balance:,.2f} - start ${start_balance:,.2f})",
    ]
    if matched:
        lines.append(f"  Discrepancy          : NONE (within ${RECON_TOLERANCE:.2f} tolerance)")
        lines.append(f"  STATUS               : PASS")
    else:
        lines.append(f"  Discrepancy          : ${discrepancy:>+10,.2f}")
        lines.append(f"  STATUS               : MISMATCH")
        lines.append(f"  NOTE: ${abs(discrepancy):,.2f} unaccounted for -- review MT5 history")
        lines.append(f"        for unlogged exits (manual closes, crashes, restarts).")

    return lines


# ---------------------------------------------------------------------------
# Daily stats from state
# ---------------------------------------------------------------------------

def _compute_stats(state: dict, balance: float) -> dict:
    closed = state.get('closed_today', [])
    wins   = [t for t in closed if t.get('exit_pnl', 0) > 0]
    losses = [t for t in closed if t.get('exit_pnl', 0) <= 0]
    total  = len(closed)

    win_rate  = (len(wins) / total * 100) if total > 0 else 0.0
    daily_pnl = state.get('daily_pnl', 0.0)
    daily_pct = (daily_pnl / STARTING_BALANCE) * 100
    cum_return = ((balance - STARTING_BALANCE) / STARTING_BALANCE) * 100

    return {
        'total'     : total,
        'wins'      : len(wins),
        'losses'    : len(losses),
        'win_rate'  : win_rate,
        'daily_pnl' : daily_pnl,
        'daily_pct' : daily_pct,
        'cum_return': cum_return,
        'balance'   : balance,
    }


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def _anomalies(state: dict, stats: dict) -> list:
    flags = []

    # Large drawdown from starting balance
    drawdown_pct = (STARTING_BALANCE - stats['balance']) / STARTING_BALANCE * 100
    if drawdown_pct >= 7.0:
        flags.append(f"WARNING: drawdown {drawdown_pct:.1f}% -- approaching 10% hard floor")
    elif drawdown_pct >= 5.0:
        flags.append(f"NOTICE: drawdown {drawdown_pct:.1f}% -- monitor closely")

    # Large single-day loss
    if stats['daily_pct'] <= -3.0:
        flags.append(f"WARNING: large daily loss {stats['daily_pct']:.1f}%")

    # Pairs paused from consecutive losses
    paused = [p for p, v in state.get('pair_paused', {}).items() if v]
    if paused:
        flags.append(f"NOTICE: pairs paused today from consecutive losses: {', '.join(paused)}")

    eurusd = state.get('eurusd', {})
    if eurusd.get('sma_consec_losses', 0) >= 2:
        flags.append("NOTICE: EURUSD-SMA paused from 2 consecutive losses")
    if eurusd.get('ema_consec_losses', 0) >= 2:
        flags.append("NOTICE: EURUSD-EMA paused from 2 consecutive losses")

    # Zero trades on a trade day
    if stats['total'] == 0 and state.get('trade_allowed', False):
        flags.append("NOTICE: trade day with zero trades -- check strategy or MT5 connectivity")

    # Low win rate (if meaningful sample)
    if stats['total'] >= 5 and stats['win_rate'] < 30:
        flags.append(f"WARNING: low win rate today {stats['win_rate']:.0f}%")

    # Progress toward target
    pct_to_target = ((stats['balance'] - STARTING_BALANCE) /
                     (TARGET_BALANCE - STARTING_BALANCE) * 100)
    if pct_to_target > 0:
        flags.append(f"INFO: {pct_to_target:.1f}% of the way to $110,000 target")

    return flags


# ---------------------------------------------------------------------------
# Report text
# ---------------------------------------------------------------------------

def _write_report(state: dict, stats: dict, flags: list, log: logging.Logger,
                  recon=None, monthly_lines=None):
    date_str = state.get('date', datetime.now(timezone.utc).date().isoformat())
    now_str  = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    lines = [
        "=" * 60,
        f"  The5ers Daily Report -- {date_str}",
        f"  Generated: {now_str}",
        "=" * 60,
        "",
        "  ACCOUNT SUMMARY",
        f"  Balance          : ${stats['balance']:>12,.2f}",
        f"  Daily P&L        : ${stats['daily_pnl']:>+12,.2f}  ({stats['daily_pct']:+.2f}%)",
        f"  Cumulative return: {stats['cum_return']:>+11.2f}%",
        f"  vs Target ${TARGET_BALANCE/1000:.0f}k    : ${stats['balance'] - TARGET_BALANCE:>+11,.2f}",
        f"  vs Hard Floor    : ${stats['balance'] - HARD_FLOOR:>+11,.2f}",
        "",
        "  TODAY'S TRADING",
        f"  Total trades     : {stats['total']}",
        f"  Wins / Losses    : {stats['wins']} / {stats['losses']}",
        f"  Win rate         : {stats['win_rate']:.1f}%",
        "",
        "  RISK STATE",
        f"  Trade allowed    : {'YES' if state.get('trade_allowed') else 'NO -- AVOID DAY'}",
        f"  London news flag : {'YES' if state.get('london_news_flag') else 'no'}",
        f"  NY news flag     : {'YES' if state.get('ny_news_flag') else 'no'}",
    ]

    # Per-pair summary
    consec = state.get('consec_losses', {})
    paused = state.get('pair_paused', {})
    eurusd = state.get('eurusd', {})
    if consec or eurusd:
        lines += ["", "  PER-PAIR STATE"]
        for pair in ['GBPJPY', 'EURJPY']:
            status = "PAUSED" if paused.get(pair) else "active"
            lines.append(f"  {pair:<8}  consec_losses={consec.get(pair, 0)}  {status}")
        if eurusd:
            sma_cl = eurusd.get('sma_consec_losses', 0)
            ema_cl = eurusd.get('ema_consec_losses', 0)
            sma_st = "PAUSED" if sma_cl >= 2 else "active"
            ema_st = "PAUSED" if ema_cl >= 2 else "active"
            lines.append(f"  EURUSD-SMA  consec_losses={sma_cl}  {sma_st}")
            lines.append(f"  EURUSD-EMA  consec_losses={ema_cl}  {ema_st}")

    # Trade detail
    closed = state.get('closed_today', [])
    if closed:
        lines += ["", "  TRADE DETAIL"]
        for t in closed:
            pnl    = t.get('exit_pnl', 0)
            result = "WIN " if pnl > 0 else "LOSS"
            lines.append(
                f"  {result}  {t.get('symbol','?'):<8} "
                f"{t.get('direction','?'):<5} {t.get('session','?'):<8} "
                f"{t.get('lots',0):.2f}L  "
                f"entry={t.get('entry_price',0):.5f}  "
                f"exit={t.get('exit_price',0):.5f}  "
                f"P&L=${pnl:+,.2f}  "
                f"reason={t.get('exit_reason','?')}"
            )

    # Daily reconciliation check
    if recon is not None:
        lines += ["", "  RECONCILIATION CHECK"]
        if recon['matched']:
            lines.append(f"  PASS -- logged P&L matches actual balance change")
            lines.append(f"  Logged P&L (CSV)   : ${recon['logged_pnl']:>+10,.2f}")
            lines.append(f"  Actual bal change  : ${recon['actual_change']:>+10,.2f}")
            lines.append(f"  Discrepancy        : none (within ${RECON_TOLERANCE:.2f} tolerance)")
        else:
            lines.append(f"  MISMATCH DETECTED")
            lines.append(f"  Logged P&L (CSV)   : ${recon['logged_pnl']:>+10,.2f}")
            lines.append(f"  Actual bal change  : ${recon['actual_change']:>+10,.2f}")
            lines.append(f"  Discrepancy        : ${recon['discrepancy']:>+10,.2f}")
            lines.append(f"  NOTE: Discrepancy likely indicates unlogged trade exits")
            lines.append(f"        (manual closes, crashes, or restarts).")
            lines.append(f"        Review MT5 history directly to identify missing trade(s).")

    # Anomaly flags
    if flags:
        lines += ["", "  FLAGS & ALERTS"]
        for flag in flags:
            lines.append(f"  {flag}")

    # Month-to-date reconciliation (always shown)
    if monthly_lines:
        lines += monthly_lines

    lines += ["", "=" * 60]

    report_text = "\n".join(lines) + "\n"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_TXT, 'w', encoding='utf-8') as f:
        f.write(report_text)

    # Also print to console/log
    log.info("\n" + report_text)

    return str(REPORT_TXT)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(daily_state: dict) -> dict:
    """
    Called by the orchestrator at 21:00 UTC.

    Args:
        daily_state : the orchestrator's shared state dict

    Returns:
        {'success': bool, 'report_path': str}
    """
    log = _log()
    log.info("Agent 5 -- Daily Reporting running")

    try:
        date_str  = daily_state.get('date', datetime.now(timezone.utc).date().isoformat())
        month_str = date_str[:7]   # 'YYYY-MM'

        # Yesterday's closing balance (read BEFORE appending today's row)
        prev_balance = _read_prev_closing_balance(date_str)

        # Get live balance from MT5 if available; fall back to prev + state estimate
        balance = prev_balance + daily_state.get('daily_pnl', 0.0)
        if mt5.initialize():
            acct = mt5.account_info()
            if acct:
                balance = acct.balance

        # Daily reconciliation: compare CSV-logged P&L against actual balance move
        logged_pnl    = _read_logged_pnl_for_date(date_str)
        recon         = _reconciliation_check(date_str, logged_pnl, balance, prev_balance, log)

        # Month-to-date reconciliation section for the report
        monthly_lines = _monthly_reconciliation_lines(month_str, balance)

        stats  = _compute_stats(daily_state, balance)
        flags  = _anomalies(daily_state, stats)

        # Append to equity curve CSV
        _append_equity_curve({
            'Date'      : date_str,
            'Balance'   : round(balance, 2),
            'DailyPnL'  : round(stats['daily_pnl'], 2),
            'DailyPct'  : round(stats['daily_pct'], 3),
            'Trades'    : stats['total'],
            'Wins'      : stats['wins'],
            'Losses'    : stats['losses'],
            'WinRate'   : round(stats['win_rate'], 1),
            'CumReturn' : round(stats['cum_return'], 3),
        })

        # Write human-readable report
        report_path = _write_report(daily_state, stats, flags, log,
                                    recon=recon, monthly_lines=monthly_lines)

        log.info(f"Reporting complete: {report_path}")
        return {'success': True, 'report_path': report_path}

    except Exception as e:
        log.error(f"Agent 5 error: {e}", exc_info=True)
        return {'success': False, 'report_path': ''}
