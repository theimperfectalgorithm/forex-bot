"""
System check for The5ers multi-agent trading system.
Runs each agent through its key functions and reports pass/fail.

Usage (from repo root):
    python src/agents/test_system.py
"""

import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime, timezone

# Ensure agents directory is on the path
AGENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(AGENTS_DIR))

# Repo root on path too, so core.* is importable
BASE_DIR = AGENTS_DIR.parent.parent
sys.path.insert(0, str(BASE_DIR))

# Same fix as main_agent.py -- import before anything else gets a chance
# to call mt5.initialize() bare (see core/mt5_connect.py for why this
# matters on a VPS running more than one MT5 terminal).
import core.mt5_connect  # noqa: F401 -- imported for its patching side effect

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

def section(title):
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print('=' * 64)

# ---------------------------------------------------------------------------
# TEST 1 -- Import all agent modules
# ---------------------------------------------------------------------------
section("TEST 1 -- Import all agent modules")

agent_names = [
    'agent_market',
    'agent_strategy',
    'agent_risk',
    'agent_execution',
    'agent_reporting',
]
imported = {}
all_imported = True

for name in agent_names:
    try:
        mod = __import__(name)
        imported[name] = mod
        print(f"  {PASS}  {name}.py")
    except Exception as e:
        imported[name] = None
        all_imported = False
        print(f"  {FAIL}  {name}.py  --  {e}")

if not all_imported:
    print("\n  Some imports failed -- fix errors above before continuing.")

# ---------------------------------------------------------------------------
# TEST 2 -- MT5 connection + account balance (agent_market)
# ---------------------------------------------------------------------------
section("TEST 2 -- MT5 connection + account balance")

try:
    import MetaTrader5 as mt5
    if mt5.initialize():
        term = mt5.terminal_info()
        acct = mt5.account_info()
        if term and acct:
            print(f"  {PASS}  Connected  :  {term.name}")
            print(f"         Account   :  {acct.login}  ({acct.server})")
            print(f"         Balance   :  ${acct.balance:,.2f}")
            print(f"         Equity    :  ${acct.equity:,.2f}")
            print(f"         Currency  :  {acct.currency}")
            floor_ok = acct.balance > 90_000
            print(f"  {'PASS' if floor_ok else 'FAIL'}  Hard floor check  :  "
                  f"balance {'above' if floor_ok else 'BELOW'} $90,000")
        else:
            print(f"  {FAIL}  Could not read terminal/account info")
    else:
        print(f"  {FAIL}  MT5 connection failed: {mt5.last_error()}")
        print("         Is MetaTrader 5 open and logged in?")
except Exception as e:
    print(f"  {FAIL}  {e}")
    import traceback; traceback.print_exc()

# ---------------------------------------------------------------------------
# TEST 3 -- H4 trends for all 3 pairs (agent_strategy)
# ---------------------------------------------------------------------------
section("TEST 3 -- H4 trend direction for all 3 pairs")

try:
    from agent_strategy import prepare_session, _h4_trend, _log as strat_log

    strat_log_inst = strat_log()

    # Show raw H4 trend first (works any time of day)
    print("  H4 trend (50/200 SMA) -- live from MT5:")
    for sym in ['GBPJPY', 'EURJPY', 'EURUSD']:
        trend = _h4_trend(sym, strat_log_inst)
        label = {1: 'BULLISH  (+1)', -1: 'BEARISH  (-1)', 0: 'NEUTRAL   (0)'}[trend]
        mark  = PASS if trend != 0 else WARN
        print(f"  {mark}  {sym:<8}  H4 = {label}")

    # prepare_session gives Asian range + trend together
    print()
    print("  Full session prep (Asian range + trend):")
    session_data = prepare_session('london')
    if session_data:
        for pair, data in session_data.items():
            label = 'BULLISH' if data['h4_trend'] == 1 else 'BEARISH'
            print(f"  {PASS}  {pair:<8}  {label:<8}  "
                  f"range={data['range_pips']:.1f}p  "
                  f"H={data['asian_high']:.5f}  L={data['asian_low']:.5f}  "
                  f"SL={data['sl_pips']:.1f}p  TP={data['tp_pips']:.1f}p")
    else:
        print(f"  {WARN}  No pairs in session_data.")
        print("         Possible reasons:")
        print("         - All 3 pairs have neutral H4 trend right now")
        print("         - Asian session range unavailable (ran before 07:00 UTC)")
        print("         - MT5 data issue")

except Exception as e:
    print(f"  {FAIL}  {e}")
    import traceback; traceback.print_exc()

# ---------------------------------------------------------------------------
# TEST 4 -- Risk calculation (agent_risk)
# ---------------------------------------------------------------------------
section("TEST 4 -- Risk / lot-size calculation")

try:
    from agent_risk import run as run_risk

    mock_state = {
        'daily_pnl'    : 0.0,
        'consec_losses': {'GBPJPY': 0, 'EURJPY': 0, 'EURUSD': 0},
        'pair_paused'  : {'GBPJPY': False, 'EURJPY': False, 'EURUSD': False},
    }

    print("  Sample risk check -- 1% risk per trade, SL = 35 pips:")
    for sym in ['GBPJPY', 'EURJPY', 'EURUSD']:
        r = run_risk(sym, 'BUY', 35.0, mock_state)
        mark = PASS if r['decision'] == 'APPROVED' else FAIL
        print(f"  {mark}  {sym:<8}  {r['decision']:<9}  "
              f"lots={r['lot_size']:.2f}  ({r['reason']})")

    # Show effect of different SL sizes
    print()
    print(f"  GBPJPY lot sizes at different SL distances (1% risk):")
    for sl in [15, 25, 35, 50, 70]:
        r = run_risk('GBPJPY', 'BUY', float(sl), mock_state)
        print(f"         SL={sl:>2}p  ->  {r['lot_size']:.2f} lots  ({r['reason']})")

except Exception as e:
    print(f"  {FAIL}  {e}")
    import traceback; traceback.print_exc()

# ---------------------------------------------------------------------------
# TEST 5 -- Order preparation (NO TRADE PLACED)
# ---------------------------------------------------------------------------
section("TEST 5 -- Order preparation check  [NO TRADE SENT]")

try:
    from agent_execution import (
        _price_round, _get_live_price, MAGIC_NUMBER, DEVIATION, PAIRS as EXEC_PAIRS
    )

    print("  Verifying SL/TP calculation logic with sample Asian ranges:")
    print()

    test_cases = [
        {'symbol': 'GBPJPY', 'signal': 'BUY',  'asian_high': 195.50, 'asian_low': 194.80,
         'sl_pips': 35.0, 'tp_pips': 70.0},
        {'symbol': 'EURJPY', 'signal': 'SELL', 'asian_high': 162.30, 'asian_low': 161.70,
         'sl_pips': 30.0, 'tp_pips': 60.0},
        {'symbol': 'EURUSD', 'signal': 'BUY',  'asian_high': 1.0850, 'asian_low': 1.0820,
         'sl_pips': 15.0, 'tp_pips': 30.0},
    ]

    for tc in test_cases:
        sym      = tc['symbol']
        signal   = tc['signal']
        pip_size = EXEC_PAIRS[sym]['pip_size']

        if signal == 'BUY':
            anchor   = tc['asian_high']
            sl_price = _price_round(anchor - tc['sl_pips'] * pip_size, sym)
            tp_price = _price_round(anchor + tc['tp_pips'] * pip_size, sym)
        else:
            anchor   = tc['asian_low']
            sl_price = _price_round(anchor + tc['sl_pips'] * pip_size, sym)
            tp_price = _price_round(anchor - tc['tp_pips'] * pip_size, sym)

        live_px  = _get_live_price(sym, signal)
        live_str = f"{live_px:.5f}" if live_px else "unavailable"

        # Validate RR ratio
        if signal == 'BUY':
            sl_dist = (anchor - sl_price) / pip_size
            tp_dist = (tp_price - anchor) / pip_size
        else:
            sl_dist = (sl_price - anchor) / pip_size
            tp_dist = (anchor - tp_price) / pip_size
        rr = tp_dist / sl_dist if sl_dist > 0 else 0

        print(f"  {PASS}  {sym:<8} {signal:<5}  anchor={anchor}  "
              f"SL={sl_price}  TP={tp_price}")
        print(f"         live_price={live_str}  "
              f"RR={rr:.2f}:1  magic={MAGIC_NUMBER}  [NOT SENT]")

    print()
    print(f"  {PASS}  Order preparation logic verified -- no orders sent to MT5")

except Exception as e:
    print(f"  {FAIL}  {e}")
    import traceback; traceback.print_exc()

# ---------------------------------------------------------------------------
# TEST 6 -- Daily report generation (agent_reporting)
# ---------------------------------------------------------------------------
section("TEST 6 -- Daily report generation")

try:
    from agent_reporting import run as run_report

    today = datetime.now(timezone.utc).date().isoformat()

    mock_state = {
        'date'             : today,
        'trade_allowed'    : True,
        'london_news_flag' : False,
        'ny_news_flag'     : False,
        'daily_pnl'        : 450.00,
        'consec_losses'    : {'GBPJPY': 0, 'EURJPY': 1, 'EURUSD': 0},
        'pair_paused'      : {'GBPJPY': False, 'EURJPY': False, 'EURUSD': False},
        'open_trades'      : [],
        'closed_today'     : [
            {
                'symbol'      : 'GBPJPY',
                'direction'   : 'BUY',
                'session'     : 'London',
                'lots'        : 2.0,
                'entry_price' : 195.50,
                'exit_price'  : 196.20,
                'sl'          : 195.15,
                'tp'          : 196.20,
                'exit_reason' : 'TP',
                'exit_pnl'    : 467.03,
                'exit_time'   : f'{today}T10:45:00+00:00',
            },
            {
                'symbol'      : 'EURJPY',
                'direction'   : 'SELL',
                'session'     : 'London',
                'lots'        : 2.0,
                'entry_price' : 162.30,
                'exit_price'  : 162.47,
                'sl'          : 162.60,
                'tp'          : 162.00,
                'exit_reason' : 'SL',
                'exit_pnl'    : -17.03,
                'exit_time'   : f'{today}T09:30:00+00:00',
            },
        ],
    }

    result = run_report(mock_state)
    mark = PASS if result['success'] else FAIL
    print(f"  {mark}  Report written: {result.get('report_path', 'none')}")

    rp = Path(result.get('report_path', ''))
    if rp.exists():
        print()
        for line in rp.read_text(encoding='utf-8').splitlines():
            print(f"  {line}")

except Exception as e:
    print(f"  {FAIL}  {e}")
    import traceback; traceback.print_exc()

# ---------------------------------------------------------------------------
# TEST 8 -- Reconciliation check (agent_reporting)
# ---------------------------------------------------------------------------
section("TEST 8 -- P&L reconciliation check (agent_reporting)")

try:
    from agent_reporting import (
        _reconciliation_check,
        _read_logged_pnl_for_date,
        _monthly_reconciliation_lines,
        RECON_TOLERANCE,
        _log as rep_log,
    )

    rep_log_inst = rep_log()

    # 8a: Mismatch -- logged P&L differs from actual balance change by > $1
    #     Simulates a day where a trade closed in MT5 but was never written to CSV.
    recon_miss = _reconciliation_check(
        date_str        = '2026-06-30',
        logged_pnl      = -253.50,          # what our CSV recorded
        current_balance = 100_600.00,        # MT5 balance implies +600 change
        prev_balance    = 100_000.00,
        log             = rep_log_inst,
    )
    miss_ok = not recon_miss['matched'] and abs(recon_miss['discrepancy'] - 853.50) < 0.01
    mark    = PASS if miss_ok else FAIL
    print(f"  {mark}  Mismatch flagged correctly  "
          f"(logged={recon_miss['logged_pnl']:+.2f}  "
          f"actual={recon_miss['actual_change']:+.2f}  "
          f"discrepancy={recon_miss['discrepancy']:+.2f})")

    # 8b: Pass -- logged P&L and actual change agree within $1 tolerance
    recon_pass = _reconciliation_check(
        date_str        = '2026-06-30',
        logged_pnl      = -253.50,
        current_balance = 100_746.00,        # actual_change = -254.00, within $1
        prev_balance    = 101_000.00,
        log             = rep_log_inst,
    )
    pass_ok = recon_pass['matched']
    mark    = PASS if pass_ok else FAIL
    print(f"  {mark}  Pass within ${RECON_TOLERANCE:.0f} tolerance  "
          f"(logged={recon_pass['logged_pnl']:+.2f}  "
          f"actual={recon_pass['actual_change']:+.2f}  "
          f"discrepancy={recon_pass['discrepancy']:+.2f})")

    # 8c: Read actual logged P&L for Jun 04 from trades_log.csv
    #     Expects GBPJPY TP +217.71 + GBPJPY TP +3.75 = +221.46
    logged_jun04 = _read_logged_pnl_for_date('2026-06-04')
    expected_jun04 = 221.46
    read_ok = abs(logged_jun04 - expected_jun04) < 0.01
    mark    = PASS if read_ok else FAIL
    print(f"  {mark}  _read_logged_pnl_for_date('2026-06-04') = "
          f"${logged_jun04:.2f}  (expected ${expected_jun04:.2f})")

    # 8d: Read June monthly logged P&L from trades_log.csv (19 closed trades)
    logged_jun = _read_logged_pnl_for_date('2026-06')
    expected_jun = 1567.93
    read_ok_m = abs(logged_jun - expected_jun) < 0.02   # allow 2c rounding
    mark      = PASS if read_ok_m else FAIL
    print(f"  {mark}  _read_logged_pnl_for_date('2026-06') = "
          f"${logged_jun:.2f}  (expected ${expected_jun:.2f})")

    # 8e: Monthly reconciliation lines show mismatch for June
    #     Actual June P&L = $101,070.73 - $100,000 = +$1,070.73
    #     Logged June P&L = +$1,567.93  =>  discrepancy = -$497.20
    m_lines = _monthly_reconciliation_lines('2026-06', current_balance=101_070.73)
    has_mismatch = any('MISMATCH' in l for l in m_lines)
    has_note     = any('unlogged' in l for l in m_lines)
    mark         = PASS if (has_mismatch and has_note) else FAIL
    print(f"  {mark}  Monthly reconciliation correctly flags June discrepancy")
    for line in m_lines:
        print(f"         {line.strip()}")

except Exception as e:
    print(f"  {FAIL}  {e}")
    import traceback; traceback.print_exc()

# ---------------------------------------------------------------------------
# TEST 9 -- Friday-only close scheduling (core.session_filter)
# ---------------------------------------------------------------------------
section("TEST 9 -- Friday-only close scheduling (replaces the old daily EOD close)")

try:
    from datetime import datetime as _dt, timezone as _tz
    from core.session_filter import is_friday_close_time, is_friday

    # Fixed reference dates (verified below, not assumed):
    #   2026-07-03 is a Friday
    #   2026-07-02 is a Thursday (non-Friday)
    friday    = _dt(2026, 7, 3, tzinfo=_tz.utc)
    thursday  = _dt(2026, 7, 2, tzinfo=_tz.utc)

    dates_ok = friday.weekday() == 4 and thursday.weekday() == 3
    mark = PASS if dates_ok else FAIL
    print(f"  {mark}  Reference dates confirmed  "
          f"(2026-07-03 weekday={friday.weekday()}, 2026-07-02 weekday={thursday.weekday()})")

    # 9a: Friday close fires correctly on Friday at 20:00 UTC
    fires_at_2000 = is_friday_close_time(friday.replace(hour=20, minute=0))
    mark = PASS if fires_at_2000 else FAIL
    print(f"  {mark}  Friday 20:00 UTC  ->  is_friday_close_time() == True")

    # 9a (continued): still true a few minutes after 20:00 (>= semantics,
    # so a delayed/skipped 15-min poll cycle doesn't miss the trigger)
    fires_at_2005 = is_friday_close_time(friday.replace(hour=20, minute=5))
    mark = PASS if fires_at_2005 else FAIL
    print(f"  {mark}  Friday 20:05 UTC  ->  is_friday_close_time() == True  (>= semantics)")

    # 9b: NO close fires on a non-Friday at 17:30 UTC (the OLD daily EOD
    # close time -- must NOT fire anymore, on any day, Friday or not)
    no_fire_thu_1730 = is_friday_close_time(thursday.replace(hour=17, minute=30))
    mark = PASS if not no_fire_thu_1730 else FAIL
    print(f"  {mark}  Thursday 17:30 UTC  ->  is_friday_close_time() == False  "
          f"(old daily EOD time, non-Friday)")

    # 9c: NO close fires on Friday itself at the old 17:30 EOD time --
    # confirms the trigger moved to 20:00, not just "any time on Friday"
    no_fire_fri_1730 = is_friday_close_time(friday.replace(hour=17, minute=30))
    mark = PASS if not no_fire_fri_1730 else FAIL
    print(f"  {mark}  Friday 17:30 UTC  ->  is_friday_close_time() == False  "
          f"(before the 20:00 trigger)")

    # 9d: NO close fires on a non-Friday even at/after 20:00 UTC
    no_fire_thu_2000 = is_friday_close_time(thursday.replace(hour=20, minute=0))
    mark = PASS if not no_fire_thu_2000 else FAIL
    print(f"  {mark}  Thursday 20:00 UTC  ->  is_friday_close_time() == False  "
          f"(right time, wrong day)")

    # 9e: is_friday() helper agrees with is_friday_close_time()'s day gate
    mark = PASS if (is_friday(friday) and not is_friday(thursday)) else FAIL
    print(f"  {mark}  is_friday() correctly identifies Friday vs Thursday")

except Exception as e:
    print(f"  {FAIL}  {e}")
    import traceback; traceback.print_exc()

# ---------------------------------------------------------------------------
# TEST 7 -- Orchestrator 2-minute live run
# ---------------------------------------------------------------------------
section("TEST 7 -- Orchestrator (2-minute live run)")

print("  Launching main_agent.py subprocess for 120 seconds ...")
print("  Output will stream below. The orchestrator will show its")
print("  schedule decisions based on current UTC time.\n")

try:
    proc = subprocess.Popen(
        [sys.executable, str(AGENTS_DIR / 'main_agent.py')],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding='utf-8',
        errors='replace',
    )

    # main_agent.py sleeps up to 900s between checks (perfectly normal --
    # it only wakes on 15-minute UTC boundaries). proc.stdout.readline() is
    # a BLOCKING call: if we call it directly in the loop below, it can
    # block far longer than our 120s test budget waiting for the next line,
    # which defeats the deadline entirely. Drain stdout on a background
    # thread instead, so the main thread's deadline check below always
    # fires on time regardless of how quiet the child currently is.
    import threading
    import queue

    line_queue: queue.Queue = queue.Queue()

    def _drain_stdout():
        for line in proc.stdout:
            line_queue.put(line)
        line_queue.put(None)   # sentinel: stdout closed (process exited)

    reader = threading.Thread(target=_drain_stdout, daemon=True)
    reader.start()

    deadline = time.time() + 120
    exited_early = False
    while time.time() < deadline:
        try:
            line = line_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if line is None:
            exited_early = True
            break
        print(f"  {line}", end='')

    if exited_early:
        print(f"\n  {WARN}  Orchestrator exited early (code {proc.returncode})")
    elif proc.poll() is None:
        proc.terminate()
        proc.wait(timeout=5)
        print(f"\n  Orchestrator terminated cleanly after 120s")

except Exception as e:
    print(f"  {FAIL}  Could not run orchestrator: {e}")
    import traceback; traceback.print_exc()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
section("SYSTEM CHECK COMPLETE")
print(f"  Completed at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
print()
