"""
Forex Bot MCP Server -- test harness.

Connects to a running mcp/server.py instance as a REAL MCP client (using
the official mcp SDK's streamable-http client, not just raw HTTP), does
the protocol initialize handshake, lists tools, and calls each of the 7
tools directly. Also checks that a missing/wrong X-API-Key is rejected
with 401.

Usage (server must already be running, e.g. `python mcp/server.py`):
    python mcp/test_mcp.py
    python mcp/test_mcp.py --url http://vps-host:8000/mcp --key <API_KEY>

Read-only: calls the same 7 read-only tools the server exposes. Does not
place orders or modify any bot file/state.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

MCP_DIR = Path(__file__).parent
load_dotenv(MCP_DIR / '.env')

PASS = "[PASS]"
FAIL = "[FAIL]"


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


async def test_auth_rejection(base_url: str) -> bool:
    """A request with no/wrong X-API-Key must get 401."""
    section("AUTH CHECK -- missing/wrong X-API-Key must be rejected")
    ok = True
    async with httpx.AsyncClient(timeout=10) as client:
        for label, headers in [
            ("no key", {}),
            ("wrong key", {"X-API-Key": "not-the-real-key"}),
        ]:
            try:
                resp = await client.post(base_url, headers={**headers, "Accept": "application/json, text/event-stream"})
                passed = resp.status_code == 401
                ok = ok and passed
                print(f"  {PASS if passed else FAIL}  {label} -> HTTP {resp.status_code} "
                     f"(expected 401)")
            except Exception as e:
                ok = False
                print(f"  {FAIL}  {label} -> request error: {e}")
    return ok


async def run_tests(url: str, api_key: str) -> None:
    results: list[bool] = []

    auth_ok = await test_auth_rejection(url)
    results.append(auth_ok)

    section("MCP PROTOCOL -- connect, initialize, list tools")
    headers = {"X-API-Key": api_key}

    async with streamablehttp_client(url, headers=headers) as (read, write, _get_sid):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print(f"  {PASS}  initialize() handshake succeeded")

            tools_result = await session.list_tools()
            tool_names = sorted(t.name for t in tools_result.tools)
            expected = sorted([
                'get_historical_bars', 'get_bot_status', 'get_daily_report',
                'get_trade_history', 'get_equity_curve', 'run_backtest', 'get_log_tail',
            ])
            all_present = set(expected).issubset(set(tool_names))
            results.append(all_present)
            print(f"  {PASS if all_present else FAIL}  tools/list returned all 7 tools  "
                 f"-- got {tool_names}")

            # ── Tool 1: get_historical_bars ──────────────────────────
            section("TOOL 1 -- get_historical_bars")
            end = datetime.now(timezone.utc).date()
            start = end - timedelta(days=7)
            r = await session.call_tool('get_historical_bars', {
                'pair': 'GBPJPY', 'timeframe': 'H1',
                'start_date': str(start), 'end_date': str(end),
            })
            data = json.loads(r.content[0].text)
            ok = data.get('success') is True and data.get('count', 0) > 0
            results.append(ok)
            print(f"  {PASS if ok else FAIL}  {data.get('count', 0)} bars returned  "
                 f"(success={data.get('success')})")
            if not ok:
                print(f"         {data}")

            # ── Tool 2: get_bot_status ────────────────────────────────
            section("TOOL 2 -- get_bot_status")
            r = await session.call_tool('get_bot_status', {})
            status = json.loads(r.content[0].text)
            ok = 'balance' in status and 'open_positions' in status and 'log_tail' in status
            results.append(ok)
            print(f"  {PASS if ok else FAIL}  response has the expected shape")
            print("  Example response:")
            print("  " + json.dumps(status, indent=2, default=str).replace("\n", "\n  "))

            # ── Tool 3: get_daily_report ──────────────────────────────
            section("TOOL 3 -- get_daily_report")
            r = await session.call_tool('get_daily_report', {'date': 'today'})
            report = json.loads(r.content[0].text)
            ok = 'success' in report   # True (report exists) or a clean False + error
            results.append(ok)
            print(f"  {PASS if ok else FAIL}  response has 'success' key  "
                 f"(success={report.get('success')})")
            print("  Example response:")
            print("  " + json.dumps(report, indent=2, default=str)[:1500].replace("\n", "\n  "))

            # ── Tool 4: get_trade_history ─────────────────────────────
            section("TOOL 4 -- get_trade_history")
            r = await session.call_tool('get_trade_history', {
                'start_date': '2020-01-01', 'end_date': str(end),
            })
            hist = json.loads(r.content[0].text)
            ok = hist.get('success') is True and 'trades' in hist
            results.append(ok)
            print(f"  {PASS if ok else FAIL}  {hist.get('count', 0)} trades returned")

            # ── Tool 5: get_equity_curve ──────────────────────────────
            section("TOOL 5 -- get_equity_curve")
            r = await session.call_tool('get_equity_curve', {})
            curve = json.loads(r.content[0].text)
            ok = curve.get('success') is True and 'curve' in curve
            results.append(ok)
            print(f"  {PASS if ok else FAIL}  {curve.get('count', 0)} equity rows returned")

            # ── Tool 6: run_backtest ──────────────────────────────────
            section("TOOL 6 -- run_backtest")
            r = await session.call_tool('run_backtest', {
                'strategy': 'london_breakout', 'pair': 'GBPJPY',
                'start_date': '2024-01-01', 'end_date': '2024-03-31',
                'config': {},
            })
            bt = json.loads(r.content[0].text)
            ok = bt.get('success') is True and 'stats' in bt
            results.append(ok)
            print(f"  {PASS if ok else FAIL}  backtest ran  "
                 f"(trades={bt.get('trade_count')}, stats={bt.get('stats')})")

            # ── Tool 7: get_log_tail ──────────────────────────────────
            section("TOOL 7 -- get_log_tail")
            r = await session.call_tool('get_log_tail', {'lines': 5})
            tail = json.loads(r.content[0].text)
            ok = tail.get('success') is True and 'lines' in tail
            results.append(ok)
            print(f"  {PASS if ok else FAIL}  {tail.get('line_count', 0)} log lines returned")

    section("SUMMARY")
    passed = sum(1 for r in results if r)
    print(f"  {passed} / {len(results)} checks passed")
    sys.exit(0 if passed == len(results) else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the Forex Bot MCP server")
    parser.add_argument('--url', default='http://localhost:8000/mcp',
                        help='MCP endpoint URL (default: http://localhost:8000/mcp)')
    parser.add_argument('--key', default=os.environ.get('MCP_API_KEY'),
                        help='X-API-Key (default: MCP_API_KEY from mcp/.env)')
    args = parser.parse_args()

    if not args.key:
        print("ERROR: no API key -- pass --key or set MCP_API_KEY in mcp/.env")
        sys.exit(1)

    asyncio.run(run_tests(args.url, args.key))


if __name__ == '__main__':
    main()
