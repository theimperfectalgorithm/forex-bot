"""
Agent 5 -- Daily Reporting
===========================
Runs at 21:00 UTC each day.

Responsibilities:
  - Compile the full day's trading summary from daily_state
  - Append a row to data/equity_curve.csv
  - Write data/daily_report.txt with a human-readable summary
  - Flag any unusual behaviour: consecutive losses, drawdown, system errors

Reads the orchestrator's daily_state dict directly (no MT5 calls needed
for reporting -- all data is already in state).
"""

import csv
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5

# -- paths
AGENTS_DIR    = Path(__file__).parent
BASE_DIR      = AGENTS_DIR.parent.parent
DATA_DIR      = BASE_DIR / 'data'
LOGS_DIR      = DATA_DIR / 'logs'
EQUITY_CSV    = DATA_DIR / 'equity_curve.csv'
REPORT_TXT    = DATA_DIR / 'daily_report.txt'

# -- constants
STARTING_BALANCE   = 100_000.00
TARGET_BALANCE     = 110_000.00
HARD_FLOOR         = 90_000.00
MAX_DAILY_LOSS_PCT = 0.05

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

def _write_report(state: dict, stats: dict, flags: list, log: logging.Logger):
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
        f"  vs Target $110k  : ${stats['balance'] - TARGET_BALANCE:>+11,.2f}",
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
    if consec:
        lines += ["", "  PER-PAIR STATE"]
        for pair in ['GBPJPY', 'EURJPY', 'EURUSD']:
            status = "PAUSED" if paused.get(pair) else "active"
            lines.append(f"  {pair:<8}  consec_losses={consec.get(pair, 0)}  {status}")

    # Trade detail
    closed = state.get('closed_today', [])
    if closed:
        lines += ["", "  TRADE DETAIL"]
        for t in closed:
            pnl     = t.get('exit_pnl', 0)
            result  = "WIN " if pnl > 0 else "LOSS"
            lines.append(
                f"  {result}  {t.get('symbol','?'):<8} "
                f"{t.get('direction','?'):<5} {t.get('session','?'):<8} "
                f"{t.get('lots',0):.2f}L  "
                f"entry={t.get('entry_price',0):.5f}  "
                f"exit={t.get('exit_price',0):.5f}  "
                f"P&L=${pnl:+,.2f}  "
                f"reason={t.get('exit_reason','?')}"
            )

    # Anomaly flags
    if flags:
        lines += ["", "  FLAGS & ALERTS"]
        for flag in flags:
            lines.append(f"  {flag}")

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
        # Get live balance from MT5 if available; fall back to state estimate
        balance = STARTING_BALANCE + daily_state.get('daily_pnl', 0.0)
        if mt5.initialize():
            acct = mt5.account_info()
            if acct:
                balance = acct.balance

        stats  = _compute_stats(daily_state, balance)
        flags  = _anomalies(daily_state, stats)

        # Append to equity curve CSV
        _append_equity_curve({
            'Date'      : daily_state.get('date', ''),
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
        report_path = _write_report(daily_state, stats, flags, log)

        log.info(f"Reporting complete: {report_path}")
        return {'success': True, 'report_path': report_path}

    except Exception as e:
        log.error(f"Agent 5 error: {e}", exc_info=True)
        return {'success': False, 'report_path': ''}
