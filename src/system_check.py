"""
System Check -- The5ers Multi-Agent Trading System
====================================================
Run from repo root:  python src/system_check.py

Checks (in order):
  1. File existence
  2. Module imports
  3. MT5 connection + account info
  4. Trend signals (GBPJPY, EURJPY H4 SMA; EURUSD H1 EMA50)
  5. Lot size calculations (all 4 risk scenarios)
  6. Order preparation (SL/TP math, no real orders placed)
  7. daily_state.json structure (create if missing)
  8. Two-minute orchestrator dry-run

Results are printed as a PASS/FAIL table.
"""

import sys
import os
import json
import time
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path

# -- ensure agents directory is on the path
REPO_ROOT  = Path(__file__).parent.parent
AGENTS_DIR = REPO_ROOT / 'src' / 'agents'
DATA_DIR   = REPO_ROOT / 'data'
STATE_DIR  = DATA_DIR / 'state'
sys.path.insert(0, str(AGENTS_DIR))

# suppress MT5 console noise
logging.disable(logging.CRITICAL)

RESULTS = []   # (test_name, status, detail)

def record(name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    RESULTS.append((name, status, detail))
    mark = "[+]" if passed else "[-]"
    print(f"  {mark} {name}: {status}" + (f"  -- {detail}" if detail else ""))
    return passed


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ===========================================================================
# 1. File existence
# ===========================================================================
section("1. FILE CHECK")

REQUIRED_FILES = [
    AGENTS_DIR / 'main_agent.py',
    AGENTS_DIR / 'agent_market.py',
    AGENTS_DIR / 'agent_strategy.py',
    AGENTS_DIR / 'agent_risk.py',
    AGENTS_DIR / 'agent_execution.py',
    AGENTS_DIR / 'agent_reporting.py',
]

for fp in REQUIRED_FILES:
    record(fp.name, fp.exists(), "" if fp.exists() else "file not found")

# Create missing data files
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

trades_log = DATA_DIR / 'trades_log.csv'
if not trades_log.exists():
    headers = ('Timestamp,Pair,Direction,Session,Lots,EntryPrice,SL,TP,'
               'AsianHigh,AsianLow,RangePips,SLPips,TPPips,Ticket,Status,'
               'ExitPrice,ExitTime,ExitReason,PnL,Balance\n')
    trades_log.write_text(headers, encoding='utf-8')
    record("data/trades_log.csv", True, "created (was missing)")
else:
    record("data/trades_log.csv", True, "exists")

equity_csv = DATA_DIR / 'equity_curve.csv'
record("data/equity_curve.csv", equity_csv.exists(), "" if equity_csv.exists() else "missing")

daily_state_file = STATE_DIR / 'daily_state.json'
record("data/state/ directory", STATE_DIR.exists(), "")


# ===========================================================================
# 2. Module imports
# ===========================================================================
section("2. IMPORT CHECK")

modules = {}
for mod_name in ['agent_market', 'agent_strategy', 'agent_risk',
                 'agent_execution', 'agent_reporting']:
    try:
        import importlib
        mod = importlib.import_module(mod_name)
        modules[mod_name] = mod
        record(f"import {mod_name}", True)
    except Exception as e:
        record(f"import {mod_name}", False, str(e))

# Check key symbols are importable
try:
    from agent_strategy  import prepare_session, check_breakout, check_eurusd_signals
    from agent_risk      import run as run_risk, _calc_lots, _pip_value_per_lot
    from agent_execution import place_trade, monitor_positions, close_trade, PAIRS as EXEC_PAIRS
    record("key symbol imports", True)
except Exception as e:
    record("key symbol imports", False, str(e))
    print(f"\n  FATAL: cannot continue without agent imports\n")
    sys.exit(1)


# ===========================================================================
# 3. MT5 connection
# ===========================================================================
section("3. MT5 CONNECTION")

MT5_OK    = False
balance   = 0.0
acct_num  = 0

try:
    import MetaTrader5 as mt5
    if mt5.initialize():
        acct = mt5.account_info()
        if acct:
            balance   = acct.balance
            acct_num  = acct.login
            server    = acct.server
            currency  = acct.currency
            MT5_OK    = True
            record("MT5 initialize()", True,
                   f"acct={acct_num}  server={server}  balance=${balance:,.2f} {currency}")
        else:
            record("MT5 initialize()", False, "account_info() returned None")
        mt5.shutdown()
    else:
        err = mt5.last_error()
        record("MT5 initialize()", False, f"error: {err}")
except Exception as e:
    record("MT5 initialize()", False, str(e))

if not MT5_OK:
    print("\n  NOTE: MT5 not connected -- trend/lot/order checks will use fallback values\n")


# ===========================================================================
# 4. Trend signals
# ===========================================================================
section("4. TREND SIGNALS")

import numpy as np

def mt5_connect():
    try:
        import MetaTrader5 as mt5
        return mt5.initialize()
    except Exception:
        return False

def fetch_h4_trend(symbol: str):
    """Returns (trend_int, sma50, sma200, last_close) or None."""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return None
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 220)
        if rates is None or len(rates) < 200:
            return None
        closes  = np.array([b['close'] for b in rates])
        sma50   = float(np.mean(closes[-50:]))
        sma200  = float(np.mean(closes[-200:]))
        last    = float(closes[-1])
        if last > sma50 > sma200:
            trend = 1
        elif last < sma50 < sma200:
            trend = -1
        else:
            trend = 0
        return trend, sma50, sma200, last
    except Exception:
        return None

def fetch_eurusd_h1_ema(period: int = 50):
    """Returns (trend_int, last_close, ema_val) or None."""
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return None
        rates = mt5.copy_rates_from_pos('EURUSD', mt5.TIMEFRAME_H1, 1, 60)
        if rates is None or len(rates) < period:
            return None
        closes = np.array([b['close'] for b in rates], dtype=float)
        alpha  = 2.0 / (period + 1)
        ema    = closes[0]
        for c in closes[1:]:
            ema = alpha * c + (1 - alpha) * ema
        last  = float(closes[-1])
        trend = 1 if last > ema else (-1 if last < ema else 0)
        return trend, last, float(ema)
    except Exception:
        return None
    finally:
        try: mt5.shutdown()
        except: pass

TREND_LABELS = {1: 'BULLISH', -1: 'BEARISH', 0: 'NEUTRAL'}
trends = {}

for sym in ['GBPJPY', 'EURJPY']:
    res = fetch_h4_trend(sym)
    if res:
        trend, sma50, sma200, last = res
        label = TREND_LABELS[trend]
        detail = (f"close={last:.3f}  SMA50={sma50:.3f}  SMA200={sma200:.3f}  "
                  f"=> {label}")
        trends[sym] = trend
        record(f"{sym} H4 SMA50/200 trend", True, detail)
    else:
        trends[sym] = None
        record(f"{sym} H4 SMA50/200 trend", not MT5_OK,
               "MT5 unavailable (expected)" if not MT5_OK else "fetch failed")

res = fetch_eurusd_h1_ema()
if res:
    trend, last, ema_val = res
    label = TREND_LABELS[trend]
    detail = f"close={last:.5f}  H1_EMA50={ema_val:.5f}  => {label}"
    trends['EURUSD'] = trend
    record("EURUSD H1 EMA50 trend", True, detail)
else:
    trends['EURUSD'] = None
    record("EURUSD H1 EMA50 trend", not MT5_OK,
           "MT5 unavailable (expected)" if not MT5_OK else "fetch failed")


# ===========================================================================
# 5. Lot size calculations
# ===========================================================================
section("5. LOT SIZE CALCULATIONS")

STARTING_BALANCE  = balance if balance > 0 else 100_000.0
HARD_FLOOR        = 90_000.0

LOT_CASES = [
    ('GBPJPY',    0.01,   25,  'JPY', 1.0  ),   # 1% risk, 25p SL (typical)
    ('EURJPY',    0.01,   25,  'JPY', 1.0  ),
    ('EURUSD-SMA',0.0025, 30,  'USD', 0.0001),
    ('EURUSD-EMA',0.0025, 15,  'USD', 0.0001),
]

def pip_value_fallback(sym: str) -> float:
    return 6.67 if 'JPY' in sym else 10.00

def pip_value_live(sym: str) -> float:
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize():
            return pip_value_fallback(sym)
        real_sym = sym.replace('-SMA','').replace('-EMA','')
        info = mt5.symbol_info(real_sym)
        if info is None:
            return pip_value_fallback(sym)
        pip_size   = 0.01 if 'JPY' in sym else 0.0001
        tick_size  = info.trade_tick_size
        tick_value = info.trade_tick_value
        if tick_size == 0:
            return 10.0
        return (pip_size / tick_size) * tick_value
    except Exception:
        return pip_value_fallback(sym)
    finally:
        try: mt5.shutdown()
        except: pass

for label, risk_pct, sl_pips, quote_ccy, _ in LOT_CASES:
    risk_usd = STARTING_BALANCE * risk_pct
    pv       = pip_value_live(label) if MT5_OK else pip_value_fallback(label)
    lots     = round(max(0.01, min(risk_usd / (sl_pips * pv), 2.00)), 2)
    detail   = (f"balance=${STARTING_BALANCE:,.0f}  "
                f"risk={risk_pct*100:.2f}%=${risk_usd:.2f}  "
                f"SL={sl_pips}p  pip_val=${pv:.2f}  "
                f"=> {lots:.2f} lots")
    in_range = 0.01 <= lots <= 2.00
    record(f"Lot calc {label}", in_range, detail)


# ===========================================================================
# 6. Order preparation (math only -- no real orders)
# ===========================================================================
section("6. ORDER PREPARATION (no real trades)")

def prepare_order(symbol, direction, sl_pips, tp_pips, use_live_anchor=False):
    """Return SL/TP prices given a sample entry price and anchor logic."""
    SAMPLE_PRICES = {
        'GBPJPY': {'BUY': 197.50, 'SELL': 197.50},
        'EURJPY': {'BUY': 161.20, 'SELL': 161.20},
        'EURUSD': {'BUY': 1.0850, 'SELL': 1.0850},
    }
    sym_clean = symbol.replace('-SMA','').replace('-EMA','')
    pip       = 0.01 if 'JPY' in sym_clean else 0.0001
    digits    = 3   if 'JPY' in sym_clean else 5
    entry     = SAMPLE_PRICES[sym_clean][direction]

    # Asian-range anchoring uses a fixed range level (simulated)
    if not use_live_anchor:
        anchor = entry - 0.05 if direction == 'BUY' else entry + 0.05
    else:
        anchor = entry   # live anchor = entry price

    if direction == 'BUY':
        sl = round(anchor - sl_pips * pip, digits)
        tp = round(anchor + tp_pips * pip, digits)
    else:
        sl = round(anchor + sl_pips * pip, digits)
        tp = round(anchor - tp_pips * pip, digits)

    rr_ok = abs(tp - anchor) / abs(sl - anchor) > 1.5   # expect ~2:1
    return entry, sl, tp, rr_ok


ORDER_CASES = [
    ('GBPJPY', 'BUY',  False, 'H4_BULLISH',  None),
    ('EURJPY', 'SELL', False, 'H4_BEARISH',  None),
    ('EURUSD', 'BUY',  True,  'SMA_BUY',  (30, 60)),
    ('EURUSD', 'SELL', True,  'EMA_SELL', (15, 30)),
]

for sym, direction, live_anchor, label, overrides in ORDER_CASES:
    sym_clean = sym.replace('-SMA','').replace('-EMA','')
    if sym_clean == 'GBPJPY' or sym_clean == 'EURJPY':
        # Standard breakout: sl=50% of range, tp=100%  (use 25p/50p sample)
        sl_pips, tp_pips = 25, 50
    else:
        sl_pips, tp_pips = overrides if overrides else (30, 60)

    entry, sl, tp, rr_ok = prepare_order(sym_clean, direction, sl_pips, tp_pips,
                                          use_live_anchor=live_anchor)
    detail = (f"entry={entry}  SL={sl}  TP={tp}  "
              f"SL_dist={sl_pips}p  TP_dist={tp_pips}p  "
              f"RR={'ok' if rr_ok else 'CHECK'}")
    record(f"Order prep {sym} {direction} ({label})", rr_ok, detail)


# ===========================================================================
# 7. daily_state.json check / creation
# ===========================================================================
section("7. DAILY STATE FILE")

BREAKOUT_PAIRS = ['GBPJPY', 'EURJPY']
today = datetime.now(timezone.utc).date().isoformat()

FRESH_STATE = {
    'date'             : today,
    'market_ran'       : False,
    'london_prep_done' : False,
    'ny_prep_done'     : False,
    'report_ran'       : False,
    'trade_allowed'    : None,
    'avoid_reason'     : None,
    'london_news_flag' : False,
    'ny_news_flag'     : False,
    'session_data'     : {},
    'london_traded'    : {p: False for p in BREAKOUT_PAIRS},
    'ny_traded'        : {p: False for p in BREAKOUT_PAIRS},
    'open_trades'      : [],
    'closed_today'     : [],
    'daily_pnl'        : 0.0,
    'consec_losses'    : {p: 0 for p in BREAKOUT_PAIRS},
    'pair_paused'      : {p: False for p in BREAKOUT_PAIRS},
    'eurusd': {
        'sma_daily_trades'    : 0,
        'ema_daily_trades'    : 0,
        'sma_consec_losses'   : 0,
        'ema_consec_losses'   : 0,
        'ema_pullback_pending': False,
        'ema_pullback_dir'    : '',
    },
}

REQUIRED_KEYS = set(FRESH_STATE.keys())

if not daily_state_file.exists():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(daily_state_file, 'w', encoding='utf-8') as f:
        json.dump(FRESH_STATE, f, indent=2)
    record("daily_state.json exists", True, "created with fresh state")
    state_data = FRESH_STATE
else:
    try:
        with open(daily_state_file, encoding='utf-8') as f:
            state_data = json.load(f)
        record("daily_state.json readable", True)
    except Exception as e:
        record("daily_state.json readable", False, str(e))
        state_data = {}

# Check all required keys present
missing = REQUIRED_KEYS - set(state_data.keys())
if missing:
    # Patch missing keys
    for k in missing:
        state_data[k] = FRESH_STATE[k]
    with open(daily_state_file, 'w', encoding='utf-8') as f:
        json.dump(state_data, f, indent=2)
    record("daily_state.json structure", True, f"patched missing keys: {missing}")
else:
    record("daily_state.json structure", True,
           f"all {len(REQUIRED_KEYS)} required keys present")

# Check date freshness
state_date = state_data.get('date', '')
date_ok    = state_date == today
record("daily_state.json date", date_ok,
       f"state date={state_date}  today={today}" +
       ("" if date_ok else "  => will reset at next orchestrator start"))

# Check eurusd sub-state
eurusd_state = state_data.get('eurusd', {})
eurusd_keys  = {'sma_daily_trades','ema_daily_trades','sma_consec_losses',
                'ema_consec_losses','ema_pullback_pending','ema_pullback_dir'}
eurusd_ok    = eurusd_keys.issubset(set(eurusd_state.keys()))
record("daily_state.json eurusd sub-state", eurusd_ok,
       f"keys present: {sorted(eurusd_state.keys())}")


# ===========================================================================
# 8. Two-minute orchestrator dry-run
# ===========================================================================
section("8. ORCHESTRATOR DRY-RUN (2 minutes)")

print("  Starting orchestrator loop -- watch for agent firings...")
print("  (MT5 is live; no orders will be sent unless a real signal fires)\n")

# Re-enable logging for orchestrator output
logging.disable(logging.NOTSET)

# Set up a simple console logger
fmt = logging.Formatter('  %(asctime)s  %(name)-12s  %(message)s',
                        datefmt='%H:%M:%S')
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt)

for logger_name in ['ORCHESTRATOR','STRATEGY','RISK','EXECUTION','REPORTING','MARKET']:
    lg = logging.getLogger(logger_name)
    if not lg.handlers:
        lg.addHandler(ch)
    lg.setLevel(logging.INFO)

# Import orchestrator pieces
try:
    from main_agent import (load_state, save_state,
                            step_market, step_session_prep, step_check_breakouts,
                            step_check_eurusd, step_monitor_positions,
                            minutes_since_midnight,
                            T_MARKET_AGENT, T_LONDON_PREP, T_LONDON_START, T_LONDON_END,
                            T_NY_PREP, T_NY_START, T_NY_END,
                            T_EURUSD_START, T_EURUSD_END, T_REPORT,
                            PAIRS as ORCH_PAIRS)
    record("orchestrator module import", True, f"PAIRS={ORCH_PAIRS}")
except Exception as e:
    record("orchestrator module import", False, str(e))
    print("\n  Cannot run dry-run without orchestrator.\n")
    # Jump to summary
    ORCH_PAIRS = None

orch_log = logging.getLogger('ORCHESTRATOR')

RUN_OK      = True
LOOP_COUNT  = 0
agent_fires = []

if ORCH_PAIRS is not None:
    orch_log.info("Dry-run started -- 2-minute window")
    start_ts = time.time()
    deadline = start_ts + 120   # 2 minutes

    while time.time() < deadline:
        try:
            now   = datetime.now(timezone.utc)
            state = load_state()
            t     = minutes_since_midnight(now)
            LOOP_COUNT += 1

            # Agent 1 -- Market Intelligence (once per day)
            if t >= T_MARKET_AGENT and not state['market_ran']:
                orch_log.info("--- Agent 1: Market Intelligence ---")
                try:
                    step_market(state, orch_log)
                    save_state(state)
                    agent_fires.append('Agent1-Market')
                except Exception as e:
                    orch_log.error(f"Agent 1 error: {e}")
                    RUN_OK = False

            # Agent 2 -- London prep
            if t >= T_LONDON_PREP and not state.get('london_prep_done'):
                orch_log.info("--- Agent 2: London prep ---")
                try:
                    if state.get('trade_allowed', False):
                        step_session_prep(state, 'london', orch_log)
                    else:
                        state['london_prep_done'] = True
                        orch_log.info("London prep skipped -- AVOID day")
                    save_state(state)
                    agent_fires.append('Agent2-LondonPrep')
                except Exception as e:
                    orch_log.error(f"Agent 2 London prep error: {e}")
                    RUN_OK = False

            # London breakout checks
            if T_LONDON_START <= t <= T_LONDON_END and state.get('london_prep_done'):
                try:
                    step_check_breakouts(state, 'london', orch_log)
                    save_state(state)
                    agent_fires.append('LondonBreakoutCheck')
                except Exception as e:
                    orch_log.error(f"London breakout error: {e}")
                    RUN_OK = False

            # Agent 2 -- NY prep
            if t >= T_NY_PREP and not state.get('ny_prep_done'):
                orch_log.info("--- Agent 2: NY prep ---")
                try:
                    if state.get('trade_allowed', False):
                        step_session_prep(state, 'ny', orch_log)
                    else:
                        state['ny_prep_done'] = True
                        orch_log.info("NY prep skipped -- AVOID day")
                    save_state(state)
                    agent_fires.append('Agent2-NYPrep')
                except Exception as e:
                    orch_log.error(f"Agent 2 NY prep error: {e}")
                    RUN_OK = False

            # NY breakout checks
            if T_NY_START <= t <= T_NY_END and state.get('ny_prep_done'):
                try:
                    step_check_breakouts(state, 'ny', orch_log)
                    save_state(state)
                    agent_fires.append('NYBreakoutCheck')
                except Exception as e:
                    orch_log.error(f"NY breakout error: {e}")
                    RUN_OK = False

            # EURUSD dual-strategy checks
            if T_EURUSD_START <= t <= T_EURUSD_END:
                try:
                    step_check_eurusd(state, orch_log)
                    save_state(state)
                    agent_fires.append('EURUSDCheck')
                except Exception as e:
                    orch_log.error(f"EURUSD check error: {e}")
                    RUN_OK = False

            # Position monitor
            try:
                step_monitor_positions(state, orch_log)
                save_state(state)
            except Exception as e:
                orch_log.error(f"Monitor error: {e}")
                RUN_OK = False

            remaining = int(deadline - time.time())
            orch_log.info(f"Loop {LOOP_COUNT} complete -- {remaining}s remaining in dry-run")

            # Short sleep so we don't spam; real orchestrator sleeps to next bar boundary
            time.sleep(10)

        except Exception as e:
            orch_log.error(f"Orchestrator loop error: {e}")
            traceback.print_exc()
            RUN_OK = False
            time.sleep(5)

    orch_log.info("Dry-run complete")

    fired_set  = sorted(set(agent_fires))
    agent_desc = ', '.join(fired_set) if fired_set else 'none (schedule not in window)'
    record("Orchestrator ran for 2 min without crash", RUN_OK, f"loops={LOOP_COUNT}")
    record("Agents/checks fired", True, agent_desc)


# ===========================================================================
# Final summary table
# ===========================================================================
print("\n")
print("=" * 70)
print("  SYSTEM CHECK RESULTS")
print("=" * 70)

col_w = max(len(r[0]) for r in RESULTS) + 2
total  = len(RESULTS)
passed = sum(1 for r in RESULTS if r[1] == 'PASS')
failed = total - passed

print(f"\n  {'TEST':<{col_w}}  {'STATUS':<8}  DETAIL")
print(f"  {'-'*col_w}  {'-'*8}  {'-'*30}")
for name, status, detail in RESULTS:
    mark = "[+]" if status == 'PASS' else "[-]"
    trunc = detail[:60] + '...' if len(detail) > 60 else detail
    print(f"  {mark} {name:<{col_w}}  {status:<8}  {trunc}")

print()
print(f"  {'='*col_w}")
print(f"  TOTAL: {total}  PASSED: {passed}  FAILED: {failed}")
print(f"  {'='*col_w}")

if failed == 0:
    print("\n  SYSTEM READY FOR DEMO TRADING")
else:
    print(f"\n  {failed} CHECK(S) FAILED -- review above before going live")

print()
