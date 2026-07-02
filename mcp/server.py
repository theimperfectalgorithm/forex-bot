"""
Forex Bot MCP Server
=====================
Exposes read-only forex-bot capabilities as MCP tools over HTTP
(streamable-http transport), so Claude (Desktop or Code) running on Mac
can query the live bot on the VPS: historical bars, bot status, daily
reports, trade history, equity curve, quick backtests, and log tail.

This is a separate, additive layer on top of the existing system:
  - It NEVER writes to trades_log.csv, equity_curve.csv, or daily_state.json.
  - It NEVER places, modifies, or closes an order.
  - It NEVER imports anything from src/agents/ (the live orchestrator/
    agents) -- only read-only helpers (core.data_loader.get_bars, direct
    MetaTrader5 read calls like account_info()/positions_get(), and plain
    file reads).
  - No existing bot file is modified by this server.

Run directly:
    python mcp/server.py
On the VPS, via Task Scheduler:
    mcp\\start_mcp.bat

Auth: every HTTP request must carry an X-API-Key header matching
MCP_API_KEY in mcp/.env (see mcp/.env.example). Missing or wrong key ->
401. All tool calls are logged (timestamp + tool name) to
data/logs/mcp_access.log.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from mcp.server.fastmcp import FastMCP

# ── Paths ─────────────────────────────────────────────────────────────────

MCP_DIR     = Path(__file__).parent
REPO_ROOT   = MCP_DIR.parent
DATA_DIR    = REPO_ROOT / 'data'
LOGS_DIR    = DATA_DIR / 'logs'
TRADES_CSV  = DATA_DIR / 'trades_log.csv'
EQUITY_CSV  = DATA_DIR / 'equity_curve.csv'
REPORT_TXT  = DATA_DIR / 'daily_report.txt'
TRADING_LOG = LOGS_DIR / 'trading.log'
MCP_LOG     = LOGS_DIR / 'mcp_access.log'

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))   # so core./strategies. imports resolve

import backtest_engine  # mcp/backtest_engine.py (same directory)

# ── API key ──────────────────────────────────────────────────────────────

load_dotenv(MCP_DIR / '.env')
API_KEY = os.environ.get('MCP_API_KEY')
if not API_KEY:
    raise RuntimeError(
        "MCP_API_KEY not set. Copy mcp/.env.example to mcp/.env and fill in "
        "a real key (see mcp/server.py header, or run:\n"
        "  python -c \"import secrets; print(secrets.token_hex(16))\""
    )

# ── Logging ──────────────────────────────────────────────────────────────

def _log() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger('MCP')
    if not log.handlers:
        fmt = logging.Formatter('%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
        fmt.converter = time.gmtime
        fh = logging.FileHandler(MCP_LOG, encoding='utf-8')
        fh.setFormatter(fmt)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        log.addHandler(fh)
        log.addHandler(ch)
        log.setLevel(logging.INFO)
    return log

log = _log()


def _log_call(tool_name: str, **kwargs) -> None:
    log.info(f"TOOL CALL  {tool_name}  args={kwargs}")


# ── Small shared helpers ────────────────────────────────────────────────

def _to_float(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _tail_lines(path: Path, n: int) -> list[str]:
    if not path.exists():
        return []
    with open(path, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    return [l.rstrip('\n') for l in lines[-n:]]


def _is_bot_process_running() -> dict:
    """
    Checks for a python.exe process via Windows Task Manager's CLI
    equivalent (tasklist). NOTE: this is deliberately the coarse check
    that was asked for -- it reports True if ANY python.exe process is
    running on the machine, not specifically main_agent.py, since
    tasklist alone (without /V and command-line parsing) can't
    distinguish which python.exe is the bot. On non-Windows hosts (e.g.
    local Mac testing) there is no Task Manager/tasklist, so this
    returns running=None with a note instead of guessing.
    """
    if sys.platform != 'win32':
        return {'running': None, 'note': 'not Windows -- tasklist unavailable on this host'}
    try:
        out = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'],
            capture_output=True, text=True, timeout=10,
        )
        lines = [l for l in out.stdout.splitlines() if l.strip()]
        running = len(lines) > 1   # header row + >=1 data row
        return {'running': running, 'process_count': max(len(lines) - 1, 0)}
    except Exception as e:
        return {'running': None, 'error': str(e)}


def _count_today_trades() -> int:
    """Counts unique tickets OPENED today (not row count -- each trade
    gets an OPEN row at entry and a separate CLOSED row at exit)."""
    if not TRADES_CSV.exists():
        return 0
    today = datetime.now(timezone.utc).date().isoformat()
    count = 0
    with open(TRADES_CSV, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('Status') == 'OPEN' and (row.get('Timestamp') or '').startswith(today):
                count += 1
    return count


# ── MCP server + tools ───────────────────────────────────────────────────

mcp = FastMCP("forex-bot", host="0.0.0.0", port=8000)


@mcp.tool()
def get_historical_bars(pair: str, timeframe: str, start_date: str, end_date: str) -> dict:
    """Fetch OHLCV bars for a pair and timeframe.

    pair: e.g. GBPJPY
    timeframe: M15, H1, H4
    start_date, end_date: YYYY-MM-DD format
    Returns: JSON array of bars with datetime, open, high, low, close, volume.
    Uses live MT5 data if available (VPS), otherwise falls back to the
    CSV files in data/historical/ (see core/data_loader.get_bars) -- the
    same fallback the bot's own backtest scripts use.
    """
    _log_call('get_historical_bars', pair=pair, timeframe=timeframe,
              start_date=start_date, end_date=end_date)
    try:
        from core import data_loader
        start = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end   = datetime.strptime(end_date, '%Y-%m-%d').replace(
            hour=23, minute=59, second=59, tzinfo=timezone.utc)
        df = data_loader.get_bars(pair, timeframe, start, end)

        bars = [
            {
                'datetime': idx.isoformat(),
                'open'    : round(float(row['Open']), 5),
                'high'    : round(float(row['High']), 5),
                'low'     : round(float(row['Low']), 5),
                'close'   : round(float(row['Close']), 5),
                'volume'  : int(row['tick_volume']),
            }
            for idx, row in df.iterrows()
        ]
        return {'success': True, 'pair': pair, 'timeframe': timeframe,
               'count': len(bars), 'bars': bars}
    except Exception as e:
        return {'success': False, 'error': str(e)}


@mcp.tool()
def get_bot_status() -> dict:
    """Current bot status: MT5 balance/equity, open positions with live
    P&L, last 10 lines of trading.log, whether the bot process is
    running, and today's trade count. Read-only -- never modifies bot
    state, never places or closes an order.
    """
    _log_call('get_bot_status')
    result = {
        'timestamp'        : datetime.now(timezone.utc).isoformat(),
        'mt5_available'    : False,
        'balance'          : None,
        'equity'           : None,
        'open_positions'   : [],
        'log_tail'         : [],
        'process_running'  : None,
        'today_trade_count': 0,
    }

    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            result['mt5_available'] = True
            acct = mt5.account_info()
            if acct:
                result['balance'] = acct.balance
                result['equity']  = acct.equity
            positions = mt5.positions_get()
            if positions:
                for pos in positions:
                    result['open_positions'].append({
                        'pair'         : pos.symbol,
                        'direction'    : 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
                        'entry_price'  : pos.price_open,
                        'current_price': pos.price_current,
                        'pnl'          : pos.profit,
                        'lots'         : pos.volume,
                        'ticket'       : pos.ticket,
                    })
        else:
            result['mt5_error'] = f'mt5.initialize() failed: {mt5.last_error()}'
    except ImportError:
        result['mt5_error'] = 'MetaTrader5 package not installed on this host'
    except Exception as e:
        result['mt5_error'] = str(e)

    result['log_tail']          = _tail_lines(TRADING_LOG, 10)
    result['process_running']   = _is_bot_process_running()
    result['today_trade_count'] = _count_today_trades()

    return result


@mcp.tool()
def get_daily_report(date: str = "today") -> dict:
    """Agent 5's daily report text for a given date (YYYY-MM-DD) or 'today'.

    LIMITATION: agent_reporting.py writes a single data/daily_report.txt
    that gets overwritten every day -- there is no per-date archive in
    the bot's current design (and this read-only MCP layer does not
    modify agent_reporting.py to add one). This tool can only serve
    whichever date is CURRENTLY in that file; requesting any other date
    returns success=False with a clear explanation and the date that IS
    available.
    """
    _log_call('get_daily_report', date=date)
    if not REPORT_TXT.exists():
        return {'success': False, 'error': 'data/daily_report.txt does not exist yet '
                '(Agent 5 has not run)'}

    text = REPORT_TXT.read_text(encoding='utf-8')
    report_date = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('The5ers Daily Report --'):
            report_date = line.split('--', 1)[1].strip()
            break

    requested = datetime.now(timezone.utc).date().isoformat() if date == 'today' else date

    if report_date != requested:
        return {
            'success': False,
            'error': (f"No archived report for {requested} -- only the latest report "
                     f"({report_date}) is available. data/daily_report.txt is "
                     f"overwritten daily by Agent 5; there is no historical archive."),
            'available_date': report_date,
        }

    return {'success': True, 'date': report_date, 'report': text}


@mcp.tool()
def get_trade_history(start_date: str, end_date: str) -> dict:
    """All CLOSED trades from trades_log.csv with an exit date between
    start_date and end_date (YYYY-MM-DD, inclusive).
    Returns: JSON array with date, pair, direction, entry, exit, pnl, reason.
    """
    _log_call('get_trade_history', start_date=start_date, end_date=end_date)
    if not TRADES_CSV.exists():
        return {'success': True, 'count': 0, 'trades': []}

    trades = []
    with open(TRADES_CSV, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('Status') != 'CLOSED':
                continue
            exit_date = (row.get('ExitTime') or '')[:10]
            if not exit_date or not (start_date <= exit_date <= end_date):
                continue
            trades.append({
                'date'     : exit_date,
                'pair'     : row.get('Pair'),
                'direction': row.get('Direction'),
                'session'  : row.get('Session'),
                'lots'     : _to_float(row.get('Lots')),
                'entry'    : _to_float(row.get('EntryPrice')),
                'exit'     : _to_float(row.get('ExitPrice')),
                'pnl'      : _to_float(row.get('PnL')),
                'reason'   : row.get('ExitReason'),
            })
    return {'success': True, 'count': len(trades), 'trades': trades}


@mcp.tool()
def get_equity_curve() -> dict:
    """Full balance history from equity_curve.csv.

    NOTE: the underlying CSV tracks end-of-day Balance only (written once
    daily by Agent 5, from mt5.account_info().balance) -- there is no
    separate intraday 'equity' series recorded in the bot's current
    design, so this returns 'balance' rather than fabricating a distinct
    'equity' field.
    """
    _log_call('get_equity_curve')
    if not EQUITY_CSV.exists():
        return {'success': True, 'count': 0, 'curve': []}

    rows = []
    with open(EQUITY_CSV, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append({
                'date'          : row.get('Date'),
                'balance'       : _to_float(row.get('Balance')),
                'daily_pnl'     : _to_float(row.get('DailyPnL')),
                'daily_pct'     : _to_float(row.get('DailyPct')),
                'trades'        : int(row.get('Trades') or 0),
                'win_rate'      : _to_float(row.get('WinRate')),
                'cum_return_pct': _to_float(row.get('CumReturn')),
            })
    return {'success': True, 'count': len(rows), 'curve': rows}


@mcp.tool()
def run_backtest(strategy: str, pair: str, start_date: str, end_date: str,
                 config: dict = {}) -> dict:
    """Runs a quick backtest using a strategy from STRATEGY_REGISTRY.

    strategy: strategy name (e.g. london_breakout, ny_open_breakout,
      asian_range_breakout). Only strategies that share a common
      "H4 trend + session-range breakout" shape are supported by this
      generic engine -- others (sma_ema_combined, h4_trend_pullback,
      mean_reversion, or any stub) return a clear "not supported"
      response rather than a misleading approximation.
    pair: e.g. GBPJPY
    start_date, end_date: YYYY-MM-DD
    config: optional dict of parameter overrides (e.g. sl_multiplier,
      tp_multiplier, h4_threshold_pips, min_range_pips) -- merged over
      that strategy's real module defaults.
    Returns: win rate, P&L, drawdown, profit factor, trade count.

    This is a lightweight tool for quick interactive queries, reusing
    the same look-ahead-safe H4 trend logic validated in this repo's
    src/*_backtest.py research scripts -- but it is not a substitute for
    those scripts' full grid-search/walk-forward/forward-test protocol.
    """
    _log_call('run_backtest', strategy=strategy, pair=pair,
              start_date=start_date, end_date=end_date, config=config)
    try:
        return backtest_engine.run(strategy, pair, start_date, end_date, config)
    except Exception as e:
        return {'success': False, 'error': str(e)}


@mcp.tool()
def get_log_tail(lines: int = 50) -> dict:
    """Returns the last N lines of trading.log -- useful for checking
    what the bot is doing right now.
    """
    _log_call('get_log_tail', lines=lines)
    tail = _tail_lines(TRADING_LOG, lines)
    return {'success': True, 'path': str(TRADING_LOG), 'line_count': len(tail), 'lines': tail}


# ── FastAPI wrapper: API-key auth + uvicorn hosting on port 8000 ───────────
#
# mcp.streamable_http_app() returns the MCP protocol's Starlette ASGI app
# (routes at /mcp by default). We wrap it in a FastAPI app so the whole
# thing serves via `uvicorn`/FastAPI as specified, and add API-key auth
# as middleware on the OUTER app so it covers every request regardless of
# which inner route handles it. The MCP app's own lifespan (its session
# manager) is wired into the FastAPI app's lifespan explicitly -- mounting
# it without doing this is a common pitfall that silently breaks the
# streamable-http session manager.

mcp_asgi_app = mcp.streamable_http_app()


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with mcp_asgi_app.router.lifespan_context(mcp_asgi_app):
        yield


app = FastAPI(title="Forex Bot MCP Server", lifespan=lifespan)


@app.middleware("http")
async def api_key_auth(request: Request, call_next):
    if request.headers.get("x-api-key") != API_KEY:
        log.warning(f"AUTH FAILED  {request.method} {request.url.path}  "
                   f"from {request.client.host if request.client else '?'}")
        return JSONResponse({"detail": "Unauthorized -- missing or invalid X-API-Key"},
                           status_code=401)
    return await call_next(request)


@app.get("/health")
async def health():
    """Unauthenticated-by-middleware-order health check would defeat the
    purpose, so this still requires X-API-Key like everything else --
    it's just a lightweight liveness probe, not a bot data endpoint."""
    return {"status": "ok", "server": "forex-bot-mcp", "time": datetime.now(timezone.utc).isoformat()}


app.mount("/", mcp_asgi_app)


if __name__ == '__main__':
    import uvicorn
    log.info("Starting Forex Bot MCP server on 0.0.0.0:8000 (MCP endpoint: /mcp)")
    uvicorn.run(app, host="0.0.0.0", port=8000)
