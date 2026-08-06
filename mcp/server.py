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

# Load mcp/.env by absolute path (relative to this file, not the process's
# cwd) before anything else -- this must run before any MCP_API_KEY
# reference, and cwd isn't reliable when launched via Task Scheduler.
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / '.env')

import csv
import json
import logging
import os
import secrets
import subprocess
import sys
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
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
JOURNAL_JSONL = DATA_DIR / 'journal' / 'events.jsonl'
NEWS_JSON     = DATA_DIR / 'news_calendar.json'
STATE_JSON    = DATA_DIR / 'state' / 'daily_state.json'

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))   # so core./strategies. imports resolve

import core.mt5_connect  # noqa: F401 -- pins mt5.initialize() to THIS clone's
                         # terminal; without it, this server's bare initialize()
                         # can attach to the OTHER account's terminal when two
                         # MT5 terminals run on the same VPS (see that module's
                         # 2026-07-21 incident writeup)

import backtest_engine  # mcp/backtest_engine.py (same directory)

# ── API key / port / dashboard token ─────────────────────────────────────

API_KEY = os.environ.get('MCP_API_KEY')
if not API_KEY:
    raise RuntimeError(
        "MCP_API_KEY not set. Copy mcp/.env.example to mcp/.env and fill in "
        "a real key (see mcp/server.py header, or run:\n"
        "  python -c \"import secrets; print(secrets.token_hex(16))\""
    )

PORT = int(os.environ.get('MCP_PORT', '8000'))

# Dashboard auth is a SEPARATE token from the MCP key: the dashboard token
# rides in a phone-bookmark URL (?key=...) and browser fetch headers, so it
# must never be the MCP key. If unset, /dash and /api/* simply stay 401 --
# the MCP side keeps working, which is the safe default on a clone where
# the dashboard hasn't been configured yet.
DASH_TOKEN = os.environ.get('DASH_TOKEN')
DASH_LABEL = os.environ.get('DASH_LABEL', 'Account')

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


# Approximate USD value of 1 pip per 1.0 lot -- used ONLY as the R-multiple
# FALLBACK when the journal has no risk_usd_intended for a ticket. Real risk
# comes from the journal's entry events; this is a documented approximation
# (JPY-quote pip value floats with USDJPY; 6.7 is a mid-2026 ballpark).
PIP_VALUE_USD = {'default': 10.0, 'JPY': 6.7, 'XAUUSD': 10.0}


def _pip_value_usd(pair: str) -> float:
    if pair == 'XAUUSD':
        return PIP_VALUE_USD['XAUUSD']
    if pair.endswith('JPY'):
        return PIP_VALUE_USD['JPY']
    return PIP_VALUE_USD['default']


def _closed_trades() -> list[dict]:
    """All CLOSED rows from trades_log.csv, full columns, oldest first."""
    if not TRADES_CSV.exists():
        return []
    with open(TRADES_CSV, newline='', encoding='utf-8') as f:
        return [r for r in csv.DictReader(f) if r.get('Status') == 'CLOSED']


def _journal_events(kinds: set | None = None) -> list[dict]:
    """Parsed events.jsonl lines (malformed lines skipped), oldest first."""
    if not JOURNAL_JSONL.exists():
        return []
    out = []
    with open(JOURNAL_JSONL, encoding='utf-8') as f:
        for line in f:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if kinds is None or ev.get('kind') in kinds:
                out.append(ev)
    return out


def _risk_usd_map() -> dict:
    """Ticket -> intended risk USD from journal 'entry' events. Entries with
    a null risk_usd_intended are skipped (the pip-value fallback covers them)."""
    m = {}
    for ev in _journal_events({'entry'}):
        risk = _to_float(ev.get('risk_usd_intended'))
        if ev.get('ticket') is not None and risk:
            m[str(ev['ticket'])] = risk
    return m


def _hold_hours_map() -> dict:
    """Ticket -> hold_hours from journal 'exit' events."""
    m = {}
    for ev in _journal_events({'exit'}):
        if ev.get('ticket') is not None and ev.get('hold_hours') is not None:
            m[str(ev['ticket'])] = ev['hold_hours']
    return m


def _trade_r(row: dict, risk_map: dict) -> tuple:
    """(r, source) for a CLOSED trades_log row. r = PnL / risk-at-entry.
    Risk source: the journal's risk_usd_intended when available ('journal'),
    else SLPips * Lots * pip_value ('fallback' -- see PIP_VALUE_USD), else
    (None, 'none')."""
    pnl = _to_float(row.get('PnL'))
    if pnl is None:
        return None, 'none'
    risk = risk_map.get(str(row.get('Ticket') or ''))
    source = 'journal'
    if not risk:
        sl_pips = _to_float(row.get('SLPips'))
        lots    = _to_float(row.get('Lots'))
        if not sl_pips or not lots:
            return None, 'none'
        risk = abs(sl_pips) * lots * _pip_value_usd(row.get('Pair') or '')
        source = 'fallback'
    if not risk:
        return None, 'none'
    return round(pnl / risk, 2), source


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

mcp = FastMCP("forex-bot", host="0.0.0.0", port=PORT)


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


def _status_payload() -> dict:
    """Shared body of get_bot_status -- also used by the /api/* dashboard
    endpoints so the MCP tool and REST route can't drift apart."""
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
def get_bot_status() -> dict:
    """Current bot status: MT5 balance/equity, open positions with live
    P&L, last 10 lines of trading.log, whether the bot process is
    running, and today's trade count. Read-only -- never modifies bot
    state, never places or closes an order.
    """
    _log_call('get_bot_status')
    return _status_payload()


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


def _equity_rows() -> list[dict]:
    """Shared body of get_equity_curve -- also used by /api/equity and
    /api/summary (previous-close lookup)."""
    if not EQUITY_CSV.exists():
        return []
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
    return rows


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
    rows = _equity_rows()
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


# ── Per-IP rate limiting on FAILED auth only ────────────────────────────
#
# 2026-08-03: the open port draws constant internet-scanner traffic (bots
# probing /, /_next, /api, etc. with random/no keys). None of it has ever
# gotten past auth, but a large burst piling up unauthenticated
# connections was part of what starved the demo server during that day's
# hang. A valid key ALWAYS bypasses this entirely -- only repeated
# *failures* from one IP get throttled, so a shared/NAT'd IP with a
# legitimate key is never affected.
_AUTH_FAIL_WINDOW_S = 60
_AUTH_FAIL_MAX      = 15
_auth_fail_log: dict[str, deque] = defaultdict(deque)


def _rate_limited(ip: str) -> bool:
    dq = _auth_fail_log.get(ip)
    if not dq:
        return False
    now = time.time()
    while dq and now - dq[0] > _AUTH_FAIL_WINDOW_S:
        dq.popleft()
    if not dq:
        del _auth_fail_log[ip]
        return False
    return len(dq) >= _AUTH_FAIL_MAX


def _record_auth_fail(ip: str) -> None:
    _auth_fail_log[ip].append(time.time())


def _dash_authorized(request: Request) -> bool:
    """/dash and /api/* accept the dashboard token either as an
    X-Dash-Key header (what the page's own fetch() calls send) or a
    ?key= query param (what the phone-bookmark URL carries -- only the
    initial page load uses this form)."""
    if not DASH_TOKEN:
        return False
    supplied = request.headers.get('x-dash-key') or request.query_params.get('key') or ''
    return secrets.compare_digest(supplied, DASH_TOKEN)


@app.middleware("http")
async def api_key_auth(request: Request, call_next):
    if request.headers.get("x-api-key") == API_KEY:
        return await call_next(request)
    path = request.url.path
    if (path == '/dash' or path.startswith('/api/')) and _dash_authorized(request):
        return await call_next(request)

    ip = request.client.host if request.client else '?'
    if _rate_limited(ip):
        # No warning log here -- an IP already over the threshold has
        # been logged plenty; this just cheaply short-circuits the flood
        # without doing any more work per request.
        return JSONResponse({"detail": "Too many failed attempts -- try again later"},
                           status_code=429)
    _record_auth_fail(ip)
    log.warning(f"AUTH FAILED  {request.method} {path}  from {ip}")
    return JSONResponse({"detail": "Unauthorized -- missing or invalid X-API-Key"},
                       status_code=401)


# CORS must be OUTSIDE the auth middleware so it answers preflight OPTIONS
# itself (preflights never carry auth headers). add_middleware() prepends,
# so adding CORS *after* defining api_key_auth puts it outermost -- correct
# by construction. Wildcard origin is fine here: auth is a bearer-style
# token (no cookies), everything is read-only, and the dashboard page on
# port A must be able to fetch the other account's API on port B.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["x-dash-key"],
)


@app.get("/health")
async def health():
    """Unauthenticated-by-middleware-order health check would defeat the
    purpose, so this still requires X-API-Key like everything else --
    it's just a lightweight liveness probe, not a bot data endpoint."""
    return {"status": "ok", "server": "forex-bot-mcp", "time": datetime.now(timezone.utc).isoformat()}


# ── Dashboard: read-only JSON API + static page ─────────────────────────
#
# All routes here MUST be declared before the app.mount("/", ...) catch-all
# below, or the MCP app swallows them. Auth: DASH_TOKEN (see middleware).
# One server process per VPS clone (MT5's python API is a per-process
# singleton pinned by core.mt5_connect) -- the page fetches BOTH clones'
# APIs (ports 8000 + 8001) client-side and renders a tab per account.
#
# INCIDENT 2026-08-03: these routes do blocking file I/O (open/csv/json)
# and were declared `async def` -- a blocking call inside an async route
# runs directly on uvicorn's single event loop and stalls EVERY other
# request for its duration, including MCP protocol traffic. That's the
# likely cause of the demo dashboard hanging until its process was
# restarted (compounded by a pile-up of scanner connections on the open
# port). Fix: every route below is a plain `def`, not `async def` --
# FastAPI/Starlette automatically runs plain-def routes in a thread
# pool, so a slow read can never block other requests. Do not add
# `async` back to these without also removing their blocking I/O.

@app.get("/api/summary")
def api_summary():
    """Everything the dashboard needs above the chart, in one call:
    live status + this clone's account constants + today's derived
    numbers (realized/unrealized P&L, daily-loss usage, target progress).
    """
    from core import account_config as ac   # reads this clone's local_config.yaml
    _log_call('api_summary')
    s = _status_payload()
    rows = _equity_rows()
    # Previous EOD close is the baseline for "today's P&L". Before the
    # first equity_curve row exists, fall back to the starting balance.
    prev_close = rows[-1]['balance'] if rows else float(ac.STARTING_BALANCE)
    bal, eq = s['balance'], s['equity']
    today_realized = round(bal - prev_close, 2) if bal is not None else None
    today_total    = round(eq  - prev_close, 2) if eq  is not None else None
    loss_limit_usd = round(prev_close * ac.MAX_DAILY_LOSS_PCT, 2)
    return {
        'label'    : DASH_LABEL,
        'timestamp': s['timestamp'],
        'account'  : {
            'starting_balance'  : ac.STARTING_BALANCE,
            'hard_floor'        : ac.HARD_FLOOR,
            'target_balance'    : ac.TARGET_BALANCE,
            'profit_target_pct' : ac.PROFIT_TARGET_PCT,
            'max_daily_loss_pct': ac.MAX_DAILY_LOSS_PCT,
        },
        'status'   : {k: s[k] for k in ('mt5_available', 'balance', 'equity',
                                        'open_positions', 'process_running',
                                        'today_trade_count')},
        'today'    : {
            'realized_pnl'  : today_realized,
            'total_pnl'     : today_total,     # incl. floating
            'unrealized_pnl': (round(eq - bal, 2)
                               if bal is not None and eq is not None else None),
            'loss_limit_usd': loss_limit_usd,
            'loss_limit_used_pct': (round(max(0.0, -today_total) / loss_limit_usd * 100, 1)
                                    if today_total is not None and loss_limit_usd else 0.0),
        },
        'progress' : {
            'to_target_pct': (round(max(0.0, eq - ac.STARTING_BALANCE)
                                    / (ac.TARGET_BALANCE - ac.STARTING_BALANCE) * 100, 1)
                              if eq is not None and ac.TARGET_BALANCE > ac.STARTING_BALANCE
                              else None),
            'floor_distance_usd': (round(eq - ac.HARD_FLOOR, 2)
                                   if eq is not None else None),
        },
    }


@app.get("/api/equity")
def api_equity():
    """EOD equity curve plus one live point (today + current equity),
    computed server-side because only the server has live MT5 access."""
    _log_call('api_equity')
    rows = _equity_rows()
    live_point = None
    s = _status_payload()
    if s['equity'] is not None:
        live_point = {'date': datetime.now(timezone.utc).date().isoformat(),
                      'balance': s['equity']}
    return {'success': True, 'count': len(rows), 'curve': rows,
            'live_point': live_point}


@app.get("/api/trades")
def api_trades(limit: int = 20):
    """Last N CLOSED trades from trades_log.csv, newest first, each with an
    R-multiple (see _trade_r for the risk-source rules)."""
    _log_call('api_trades', limit=limit)
    risk_map = _risk_usd_map()
    trades = []
    for row in _closed_trades():
        r, _src = _trade_r(row, risk_map)
        trades.append({
            'date'     : (row.get('ExitTime') or '')[:10],
            'pair'     : row.get('Pair'),
            'direction': row.get('Direction'),
            'session'  : row.get('Session'),
            'lots'     : _to_float(row.get('Lots')),
            'entry'    : _to_float(row.get('EntryPrice')),
            'exit'     : _to_float(row.get('ExitPrice')),
            'pnl'      : _to_float(row.get('PnL')),
            'r'        : r,
            'reason'   : row.get('ExitReason'),
        })
    trades = trades[-max(limit, 1):][::-1]
    return {'success': True, 'count': len(trades), 'trades': trades}


@app.get("/api/stats")
def api_stats(days: int = 30, start: str = '', end: str = ''):
    """Closed-trade statistics over a period. `days=N` = rolling window ending
    today (0 = all history); explicit `start`/`end` (YYYY-MM-DD, inclusive)
    override `days` -- the weekly-report generator passes an exact Mon-Fri.

    Definitions (these numbers get quoted publicly -- keep them defensible):
      - win            : PnL > 0 (zero-PnL scratches count as losses)
      - profit_factor  : gross_wins / |gross_losses|; null when no losses
                         (client may render as infinity)
      - expectancy_r   : mean R over trades with a computable R; r_coverage
                         reports how many Rs came from the journal's exact
                         risk_usd_intended vs the pip-value fallback vs none
      - avg_planned_rr : mean(|TPPips| / |SLPips|) -- PLANNED ratio at entry,
                         not realized
      - avg_win_r / avg_loss_r : realized mean R of winners / losers
      - max_drawdown_pct: max peak-to-trough % decline of equity_curve.csv's
                         EOD Balance series (no intraday series exists, so
                         intraday drawdown is not represented)
      - pnl.week/month : sum of DailyPnL rows in the current ISO week /
                         calendar month, plus today's live realized delta
                         (balance - last EOD close) when MT5 is up
    """
    _log_call('api_stats', days=days, start=start, end=end)
    today = datetime.now(timezone.utc).date()
    if start and end:
        lo, hi = start, end
    elif days and days > 0:
        lo = (today - timedelta(days=days - 1)).isoformat()
        hi = today.isoformat()
    else:
        lo, hi = '0000-01-01', '9999-12-31'

    risk_map = _risk_usd_map()
    hold_map = _hold_hours_map()

    rows = [r for r in _closed_trades()
            if lo <= (r.get('ExitTime') or '')[:10] <= hi]

    wins, losses = [], []
    rs, planned_rrs = [], []
    r_cov = {'journal': 0, 'fallback': 0, 'none': 0}
    best = worst = None
    for row in rows:
        pnl = _to_float(row.get('PnL'))
        if pnl is None:
            continue
        (wins if pnl > 0 else losses).append(pnl)
        r, src = _trade_r(row, risk_map)
        r_cov[src] += 1
        if r is not None:
            rs.append(r)
        sl = _to_float(row.get('SLPips'))
        tp = _to_float(row.get('TPPips'))
        if sl and tp:
            planned_rrs.append(abs(tp) / abs(sl))
        item = {'date': (row.get('ExitTime') or '')[:10],
                'pair': row.get('Pair'), 'direction': row.get('Direction'),
                'pnl': pnl, 'r': r, 'reason': row.get('ExitReason'),
                'hold_hours': hold_map.get(str(row.get('Ticket') or ''))}
        if best is None or pnl > best['pnl']:
            best = item
        if worst is None or pnl < worst['pnl']:
            worst = item

    gross_win  = round(sum(wins), 2)
    gross_loss = round(sum(losses), 2)
    win_rs  = [r for r in rs if r > 0]
    loss_rs = [r for r in rs if r <= 0]
    count = len(wins) + len(losses)

    # equity-derived numbers
    eq = _equity_rows()
    balances = [r['balance'] for r in eq if r['balance'] is not None]
    max_dd = 0.0
    peak = balances[0] if balances else 0.0
    for b in balances:
        peak = max(peak, b)
        if peak:
            max_dd = max(max_dd, (peak - b) / peak * 100)

    iso_year, iso_week, _ = today.isocalendar()
    week_pnl = sum((r['daily_pnl'] or 0) for r in eq
                   if r['date'] and
                   datetime.strptime(r['date'], '%Y-%m-%d').date()
                   .isocalendar()[:2] == (iso_year, iso_week))
    month_pnl = sum((r['daily_pnl'] or 0) for r in eq
                    if (r['date'] or '').startswith(today.strftime('%Y-%m')))
    # today's live realized delta (same derivation as /api/summary)
    s = _status_payload()
    if s['balance'] is not None and eq:
        live_delta = s['balance'] - (eq[-1]['balance'] or s['balance'])
        week_pnl  += live_delta
        month_pnl += live_delta

    return {
        'success': True, 'days': days,
        'period': {'start': lo if lo != '0000-01-01' else (eq[0]['date'] if eq else None),
                   'end': hi if hi != '9999-12-31' else today.isoformat()},
        'trades': {
            'count': count, 'wins': len(wins), 'losses': len(losses),
            'win_rate_pct': round(len(wins) / count * 100, 1) if count else None,
            'gross_win_usd': gross_win, 'gross_loss_usd': gross_loss,
            'profit_factor': (round(gross_win / abs(gross_loss), 2)
                              if gross_loss else None),
            'avg_win_usd' : round(gross_win / len(wins), 2) if wins else None,
            'avg_loss_usd': round(gross_loss / len(losses), 2) if losses else None,
            'expectancy_r': round(sum(rs) / len(rs), 2) if rs else None,
            'avg_planned_rr': (round(sum(planned_rrs) / len(planned_rrs), 2)
                               if planned_rrs else None),
            'avg_win_r' : round(sum(win_rs) / len(win_rs), 2) if win_rs else None,
            'avg_loss_r': round(sum(loss_rs) / len(loss_rs), 2) if loss_rs else None,
            'r_coverage': r_cov,
        },
        'best_trade': best, 'worst_trade': worst,
        'pnl': {'period_usd': round(gross_win + gross_loss, 2),
                'week_usd': round(week_pnl, 2), 'month_usd': round(month_pnl, 2)},
        'max_drawdown_pct': round(max_dd, 2),
        'sparklines': {'week' : balances[-7:], 'month': balances[-30:]},
    }


@app.get("/api/journal")
def api_journal(limit: int = 5):
    """Latest journal cards: entry events (newest first) joined to their exit
    events by ticket. All text comes verbatim from the bot's own journal --
    nothing is fabricated."""
    _log_call('api_journal', limit=limit)
    entries = _journal_events({'entry'})[::-1][:max(limit, 1)]
    exits   = {str(ev.get('ticket')): ev for ev in _journal_events({'exit'})
               if ev.get('ticket') is not None}
    risk_map = _risk_usd_map()
    cards = []
    for ev in entries:
        ticket = str(ev.get('ticket'))
        ex = exits.get(ticket)
        exit_obj = None
        if ex:
            pnl  = _to_float(ex.get('pnl'))
            risk = risk_map.get(ticket)
            if not risk:
                sl, lots = _to_float(ev.get('sl_pips')), _to_float(ev.get('lots'))
                if sl and lots:
                    risk = abs(sl) * lots * _pip_value_usd(ev.get('symbol') or '')
            exit_obj = {
                'ts_utc': ex.get('ts_utc'), 'exit_price': ex.get('exit_price'),
                'exit_reason': ex.get('exit_reason'), 'pnl': pnl,
                'r': (round(pnl / risk, 2) if pnl is not None and risk else None),
                'hold_hours': ex.get('hold_hours'),
            }
        cards.append({
            'ticket': ev.get('ticket'), 'symbol': ev.get('symbol'),
            'direction': ev.get('direction'), 'ts_utc': ev.get('ts_utc'),
            'lots': ev.get('lots'), 'fill_price': ev.get('fill_price'),
            'slippage_pips': ev.get('slippage_pips'),
            'sl_pips': ev.get('sl_pips'), 'tp_pips': ev.get('tp_pips'),
            'reason': ev.get('strategy_reason'),
            'context': {
                'spread_pips': ev.get('spread_pips'),
                'atr14_h1_pips': ev.get('atr14_h1_pips'),
                'minutes_to_next_high_news': ev.get('minutes_to_next_high_news'),
            },
            'exit': exit_obj,
        })
    return {'success': True, 'count': len(cards), 'cards': cards}


# UTC session windows -- same convention already used client-side in
# dashboard.html's session card: London 07:00-16:00, NY 12:00-21:00,
# everything else counted as Asian.
def _session_from_event(ev: dict) -> str:
    hour_server = ev.get('hour_server')
    offset      = ev.get('server_offset_h')
    if hour_server is None or offset is None:
        return 'Unknown'
    utc_hour = (hour_server - offset) % 24
    if 7 <= utc_hour < 16:
        return 'London'
    if 12 <= utc_hour < 21:
        return 'NY'
    return 'Asian'


@app.get("/api/slippage")
def api_slippage():
    """Slippage aggregation (avg / worst, overall + by pair + by session).

    Sign convention (set by core.trade_journal.log_entry): slippage_pips
    = (fill_price - signal_price)/pip, sign-adjusted so POSITIVE always
    means the fill cost you pips vs. the strategy's signal price,
    regardless of BUY/SELL direction; negative means a better-than-
    expected fill. "Worst" is therefore simply the maximum value.
    Session is derived from each entry's own hour_server/server_offset_h
    (not joined to trades_log.csv), so open (not yet closed) trades are
    included too.
    """
    _log_call('api_slippage')
    entries = [ev for ev in _journal_events({'entry'})
              if _to_float(ev.get('slippage_pips')) is not None]

    def _agg(rows: list) -> dict:
        vals = [_to_float(r['slippage_pips']) for r in rows]
        return {
            'avg_pips'    : round(sum(vals) / len(vals), 2),
            'avg_abs_pips': round(sum(abs(v) for v in vals) / len(vals), 2),
            'worst_pips'  : round(max(vals), 2),
            'count'       : len(vals),
        }

    if not entries:
        return {'success': True, 'count': 0,
                'overall': None, 'by_pair': [], 'by_session': []}

    by_pair_map, by_session_map = {}, {}
    for ev in entries:
        by_pair_map.setdefault(ev.get('symbol') or '?', []).append(ev)
        by_session_map.setdefault(_session_from_event(ev), []).append(ev)

    by_pair = sorted(
        [{'pair': k, **_agg(v)} for k, v in by_pair_map.items()],
        key=lambda r: r['worst_pips'], reverse=True)
    session_order = {'Asian': 0, 'London': 1, 'NY': 2, 'Unknown': 3}
    by_session = sorted(
        [{'session': k, **_agg(v)} for k, v in by_session_map.items()],
        key=lambda r: session_order.get(r['session'], 9))

    return {'success': True, 'count': len(entries),
            'overall': _agg(entries), 'by_pair': by_pair,
            'by_session': by_session}


@app.get("/api/news")
def api_news(limit: int = 5):
    """Upcoming high-impact news from the bot's own calendar cache
    (data/news_calendar.json, ForexFactory feed, high-impact only)."""
    _log_call('api_news', limit=limit)
    try:
        with open(NEWS_JSON, encoding='utf-8') as f:
            cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {'success': True, 'fetched_at': None, 'stale': True, 'events': []}

    now = datetime.now(timezone.utc)
    fetched_at = cache.get('fetched_at')
    stale = True
    if fetched_at:
        try:
            age_h = (now - datetime.fromisoformat(fetched_at)).total_seconds() / 3600
            stale = age_h > 72
        except ValueError:
            pass

    upcoming = []
    for ev in cache.get('events', []):
        try:
            t = datetime.fromisoformat(ev['time_utc'])
        except (KeyError, ValueError):
            continue
        if t >= now:
            upcoming.append({'currency': ev.get('currency'),
                             'title': ev.get('title'),
                             'time_utc': ev['time_utc'],
                             'minutes_until': int((t - now).total_seconds() // 60)})
    upcoming.sort(key=lambda e: e['minutes_until'])
    return {'success': True, 'fetched_at': fetched_at, 'stale': stale,
            'events': upcoming[:max(limit, 1)]}


@app.get("/api/state")
def api_state():
    """Read-only subset of the orchestrator's daily_state.json (session prep
    flags, trade-allowed verdict, paused pairs). exists=false when the file
    is absent (fresh day) or unreadable (orchestrator mid-write)."""
    _log_call('api_state')
    empty = {'success': True, 'exists': False, 'date': None,
             'trade_allowed': None, 'avoid_reason': None,
             'london_prep_done': None, 'ny_prep_done': None,
             'london_news_flag': None, 'ny_news_flag': None,
             'pair_paused': {}, 'consec_losses': {}, 'daily_pnl': None}
    try:
        with open(STATE_JSON, encoding='utf-8') as f:
            st = json.load(f)
    except FileNotFoundError:
        return empty
    except (json.JSONDecodeError, OSError):
        return {**empty, 'note': 'state file unreadable (possibly mid-write)'}
    return {'success': True, 'exists': True,
            'date': st.get('date'),
            'trade_allowed': st.get('trade_allowed'),
            'avoid_reason': st.get('avoid_reason'),
            'london_prep_done': st.get('london_prep_done'),
            'ny_prep_done': st.get('ny_prep_done'),
            'london_news_flag': st.get('london_news_flag'),
            'ny_news_flag': st.get('ny_news_flag'),
            'pair_paused': st.get('pair_paused') or {},
            'consec_losses': st.get('consec_losses') or {},
            'daily_pnl': st.get('daily_pnl')}


@app.get("/dash")
def dash_page():
    """The read-only mobile dashboard (single self-contained HTML file)."""
    return FileResponse(MCP_DIR / 'dashboard.html', media_type='text/html')


app.mount("/", mcp_asgi_app)


if __name__ == '__main__':
    import uvicorn
    log.info(f"Starting Forex Bot MCP server on 0.0.0.0:{PORT} "
             f"(MCP endpoint: /mcp, dashboard: /dash)")
    # limit_concurrency: caps simultaneous in-flight requests (excess get
    # a fast 503 instead of queuing unbounded) -- a second layer behind
    # the per-IP rate limiter above, protecting against a burst that
    # isn't just one IP repeatedly failing auth (2026-08-03 incident).
    uvicorn.run(app, host="0.0.0.0", port=PORT, limit_concurrency=100)
