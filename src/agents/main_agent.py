"""
The5ers Multi-Agent Live Trading System -- Main Orchestrator
============================================================
Runs continuously. Wakes each sub-agent at scheduled UTC times and
coordinates the full trading day. Errors in any agent are caught and
logged; the loop always continues.

Daily schedule (UTC):
  00:00  Agent 1 -- Market Intelligence (account + news check)
  07:45  Agent 2 -- Strategy prep for London session
  08:00-12:30  Orchestrator -- London M15 bar checks (breakout detection)
  12:45  Agent 2 -- Strategy prep for NY session
  13:00-20:45  Orchestrator -- NY M15 bar checks
  Every 15 min  Orchestrator -- Monitor open positions, apply breakeven
  21:00  Agent 5 -- Daily reporting

Run from repo root:
  python src/agents/main_agent.py
"""

import sys
import time
import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

# -- path setup so agents can import each other when run from any directory
AGENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(AGENTS_DIR))

from agent_market    import run as run_market
from agent_strategy  import prepare_session, check_breakout
from agent_risk      import run as run_risk
from agent_execution import place_trade, monitor_positions
from agent_reporting import run as run_reporting

# -- directory paths
BASE_DIR   = AGENTS_DIR.parent.parent          # repo root
DATA_DIR   = BASE_DIR / 'data'
STATE_DIR  = DATA_DIR / 'state'
LOGS_DIR   = DATA_DIR / 'logs'
STATE_FILE = STATE_DIR / 'daily_state.json'

# -- trading pairs
PAIRS = ['GBPJPY', 'EURJPY', 'EURUSD']

# -- schedule thresholds (minutes since UTC midnight)
T_MARKET_AGENT   =  0 * 60 +  0    # 00:00
T_LONDON_PREP    =  7 * 60 + 45    # 07:45
T_LONDON_START   =  8 * 60 +  0    # 08:00
T_LONDON_END     = 12 * 60 + 30    # 12:30  (last bar check before NY prep)
T_NY_PREP        = 12 * 60 + 45    # 12:45
T_NY_START       = 13 * 60 +  0    # 13:00
T_NY_END         = 20 * 60 + 45    # 20:45  (last bar check)
T_REPORT         = 21 * 60 +  0    # 21:00


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / 'trading.log'

    fmt = logging.Formatter('%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)

    log = logging.getLogger('ORCHESTRATOR')
    log.setLevel(logging.INFO)
    log.addHandler(fh)
    log.addHandler(ch)
    return log


# ---------------------------------------------------------------------------
# Daily state management
# ---------------------------------------------------------------------------

def _fresh_state(today: str) -> dict:
    return {
        'date'             : today,
        # agent completion flags (reset each day)
        'market_ran'       : False,
        'london_prep_done' : False,
        'ny_prep_done'     : False,
        'report_ran'       : False,
        # market agent output
        'trade_allowed'    : None,
        'avoid_reason'     : None,
        'london_news_flag' : False,
        'ny_news_flag'     : False,
        # strategy agent output -- Asian range + H4 trend per pair
        'session_data'     : {},
        # per-pair session trade flags
        'london_traded'    : {p: False for p in PAIRS},
        'ny_traded'        : {p: False for p in PAIRS},
        # live position tracking
        'open_trades'      : [],
        'closed_today'     : [],
        # running daily P&L and risk state
        'daily_pnl'        : 0.0,
        'consec_losses'    : {p: 0 for p in PAIRS},
        'pair_paused'      : {p: False for p in PAIRS},
    }


def load_state() -> dict:
    """Load today's state from disk; create fresh state if date changed."""
    today = datetime.now(timezone.utc).date().isoformat()
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, encoding='utf-8') as f:
                state = json.load(f)
            if state.get('date') == today:
                return state
        except Exception:
            pass  # corrupt file -- start fresh

    state = _fresh_state(today)
    save_state(state)
    return state


def save_state(state: dict):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Step functions -- each wraps one agent call
# ---------------------------------------------------------------------------

def step_market(state: dict, log: logging.Logger):
    log.info("--- Agent 1: Market Intelligence ---")
    try:
        result = run_market()
        state['trade_allowed']    = (result['status'] == 'TRADE_DAY')
        state['avoid_reason']     = result.get('reason')
        state['london_news_flag'] = result.get('london_news_flag', False)
        state['ny_news_flag']     = result.get('ny_news_flag', False)
        state['market_ran']       = True

        log.info(f"Market verdict : {result['status']}")
        if result.get('reason'):
            log.info(f"Reason         : {result['reason']}")
        log.info(f"Account balance: ${result.get('balance', 0):,.2f}")
        if result.get('london_news_flag'):
            log.warning("NEWS FLAG: High-impact event near London open (08:00)")
        if result.get('ny_news_flag'):
            log.warning("NEWS FLAG: High-impact event near NY open (13:00)")
    except Exception as e:
        log.error(f"Agent 1 failed: {e}", exc_info=True)
        # Default to AVOID on market agent failure -- safety first
        state['trade_allowed'] = False
        state['avoid_reason']  = f"Agent 1 error: {e}"
        state['market_ran']    = True


def step_session_prep(state: dict, session: str, log: logging.Logger):
    log.info(f"--- Agent 2: Strategy prep ({session.upper()}) ---")
    try:
        session_data = prepare_session(session)
        state['session_data'] = session_data
        state[f'{session}_prep_done'] = True

        for pair, data in session_data.items():
            trend_label = {1: 'BULLISH', -1: 'BEARISH', 0: 'NEUTRAL'}[data['h4_trend']]
            log.info(f"  {pair}: range={data['range_pips']:.1f}p "
                     f"H={data['asian_high']:.5f} L={data['asian_low']:.5f} "
                     f"H4={trend_label}")
    except Exception as e:
        log.error(f"Agent 2 ({session}) failed: {e}", exc_info=True)
        state[f'{session}_prep_done'] = True   # mark done to avoid infinite retry


def step_check_breakouts(state: dict, session: str, log: logging.Logger):
    """Check the last closed M15 bar for breakout signals on each untraded pair."""
    if not state.get('session_data'):
        return

    for pair in PAIRS:
        # Skip if already traded this session
        if state[f'{session}_traded'][pair]:
            continue

        # For NY: skip if a London position for this pair is still open
        if session == 'ny':
            london_open = any(
                t['symbol'] == pair and t['session'] == 'London'
                for t in state['open_trades']
            )
            if london_open:
                continue

        # Skip if pair paused after 2 consecutive losses
        if state['pair_paused'][pair]:
            continue

        try:
            breakout = check_breakout(pair, state['session_data'], session)
        except Exception as e:
            log.error(f"Agent 2 check_breakout {pair}: {e}", exc_info=True)
            continue

        if breakout['signal'] == 'NO_SIGNAL':
            continue

        log.info(f"BREAKOUT CONFIRMED  {pair} {session.upper()} "
                 f"{breakout['signal']}  bar_close={breakout['trigger_bar_close']:.5f}  "
                 f"+{breakout['overshoot_pips']:.0f}p past range")

        # Daily loss limit guard
        daily_limit = 0.05 * 100_000   # $5,000 -- absolute floor check
        if state['daily_pnl'] <= -daily_limit:
            log.warning(f"Daily loss limit reached -- skipping {pair}")
            continue

        # News flag guard
        if session == 'london' and state.get('london_news_flag'):
            log.warning(f"London news flag active -- skipping {pair} this session")
            continue
        if session == 'ny' and state.get('ny_news_flag'):
            log.warning(f"NY news flag active -- skipping {pair} this session")
            continue

        # Risk management
        try:
            sl_pips = state['session_data'][pair]['sl_pips']
            risk = run_risk(pair, breakout['signal'], sl_pips, state)
        except Exception as e:
            log.error(f"Agent 3 risk check {pair}: {e}", exc_info=True)
            continue

        if risk['decision'] == 'REJECTED':
            log.warning(f"Risk REJECTED {pair}: {risk['reason']}")
            continue

        log.info(f"Risk APPROVED {pair}: {risk['lot_size']:.2f} lots")

        # Trade execution
        try:
            result = place_trade(pair, breakout, risk['lot_size'],
                                 state['session_data'][pair], session)
        except Exception as e:
            log.error(f"Agent 4 place_trade {pair}: {e}", exc_info=True)
            continue

        if result['success']:
            log.info(f"TRADE PLACED  {pair} {breakout['signal']}  "
                     f"{risk['lot_size']:.2f}L  "
                     f"entry={result['entry_price']:.5f}  "
                     f"SL={result['sl']:.5f}  TP={result['tp']:.5f}  "
                     f"ticket={result['ticket']}")

            state[f'{session}_traded'][pair] = True
            state['open_trades'].append({
                'ticket'        : result['ticket'],
                'symbol'        : pair,
                'direction'     : breakout['signal'],
                'session'       : session.capitalize(),
                'lots'          : risk['lot_size'],
                'entry_price'   : result['entry_price'],
                'sl'            : result['sl'],
                'tp'            : result['tp'],
                'asian_high'    : state['session_data'][pair]['asian_high'],
                'asian_low'     : state['session_data'][pair]['asian_low'],
                'sl_pips'       : sl_pips,
                'tp_pips'       : state['session_data'][pair]['tp_pips'],
                'breakeven_moved': False,
                'entry_time'    : datetime.now(timezone.utc).isoformat(),
            })
        else:
            log.error(f"Trade placement FAILED {pair}: {result['error']}")


def step_monitor_positions(state: dict, log: logging.Logger):
    """Check all open positions: move to breakeven, detect closures."""
    if not state['open_trades']:
        return
    try:
        still_open, newly_closed = monitor_positions(state['open_trades'], log)
        state['open_trades'] = still_open

        for trade in newly_closed:
            pnl = trade.get('exit_pnl', 0.0)
            state['daily_pnl'] += pnl
            state['closed_today'].append(trade)

            pair = trade['symbol']
            if pnl <= 0:
                state['consec_losses'][pair] += 1
                if state['consec_losses'][pair] >= 2:
                    state['pair_paused'][pair] = True
                    log.warning(f"{pair} paused for today: 2 consecutive losses")
            else:
                state['consec_losses'][pair] = 0

            log.info(f"POSITION CLOSED  {pair} {trade['direction']}  "
                     f"exit={trade.get('exit_price', 0):.5f}  "
                     f"reason={trade.get('exit_reason', '?')}  "
                     f"PnL=${pnl:+,.2f}  daily=${state['daily_pnl']:+,.2f}")
    except Exception as e:
        log.error(f"Position monitoring error: {e}", exc_info=True)


def step_report(state: dict, log: logging.Logger):
    log.info("--- Agent 5: Daily Reporting ---")
    try:
        result = run_reporting(state)
        state['report_ran'] = True
        log.info(f"Report written: {result.get('report_path', '?')}")
    except Exception as e:
        log.error(f"Agent 5 reporting failed: {e}", exc_info=True)
        state['report_ran'] = True   # don't retry tonight


# ---------------------------------------------------------------------------
# Scheduler helper
# ---------------------------------------------------------------------------

def minutes_since_midnight(now: datetime) -> int:
    return now.hour * 60 + now.minute


def sleep_until_next_quarter(log: logging.Logger):
    """Sleep until 5 seconds past the next 15-minute UTC boundary."""
    now = datetime.now(timezone.utc)
    mins = now.minute
    next_min = ((mins // 15) + 1) * 15

    if next_min >= 60:
        next_boundary = (now + timedelta(hours=1)).replace(
            minute=0, second=5, microsecond=0)
    else:
        next_boundary = now.replace(minute=next_min, second=5, microsecond=0)

    wait = (next_boundary - datetime.now(timezone.utc)).total_seconds()
    wait = max(wait, 1)
    log.info(f"Sleeping {wait:.0f}s -- next check at {next_boundary.strftime('%H:%M:%S')} UTC")
    time.sleep(wait)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    log = setup_logger()
    log.info("=" * 60)
    log.info("The5ers Trading System -- Orchestrator Started")
    log.info(f"Pairs: {', '.join(PAIRS)}")
    log.info("=" * 60)

    while True:
        try:
            now   = datetime.now(timezone.utc)
            state = load_state()
            t     = minutes_since_midnight(now)

            # -- 00:00  Market Intelligence (once per day)
            if t >= T_MARKET_AGENT and not state['market_ran']:
                step_market(state, log)
                save_state(state)

            # -- 07:45  London session prep (once per day)
            if t >= T_LONDON_PREP and not state['london_prep_done']:
                if state.get('trade_allowed', False):
                    step_session_prep(state, 'london', log)
                else:
                    log.info("London prep skipped -- market agent flagged AVOID")
                    state['london_prep_done'] = True
                save_state(state)

            # -- 08:00-12:30  London breakout checks (every 15 min)
            if T_LONDON_START <= t <= T_LONDON_END and state['london_prep_done']:
                step_check_breakouts(state, 'london', log)
                save_state(state)

            # -- 12:45  NY session prep (once per day)
            if t >= T_NY_PREP and not state['ny_prep_done']:
                if state.get('trade_allowed', False):
                    step_session_prep(state, 'ny', log)
                else:
                    log.info("NY prep skipped -- market agent flagged AVOID")
                    state['ny_prep_done'] = True
                save_state(state)

            # -- 13:00-20:45  NY breakout checks (every 15 min)
            if T_NY_START <= t <= T_NY_END and state['ny_prep_done']:
                step_check_breakouts(state, 'ny', log)
                save_state(state)

            # -- 21:00  Daily report (once per day)
            if t >= T_REPORT and not state['report_ran']:
                step_report(state, log)
                save_state(state)

            # -- Always: monitor open positions for breakeven / closures
            step_monitor_positions(state, log)
            save_state(state)

        except Exception as e:
            log.critical(f"Orchestrator loop error: {e}", exc_info=True)
            # Never crash -- just log and continue

        sleep_until_next_quarter(log)


if __name__ == '__main__':
    main()
