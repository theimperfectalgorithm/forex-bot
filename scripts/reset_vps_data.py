"""
Resets trades_log.csv, equity_curve.csv, and daily_state.json to a clean
baseline, after backing up the existing files.

Schemas are imported directly from the real bot code rather than
hand-duplicated here, so this script can never drift out of sync with
it:
  - TRADES_LOG_HEADERS         from src.agents.agent_execution
  - EQUITY_HEADERS, STARTING_BALANCE   from src.agents.agent_reporting
  - _fresh_state()             from src.agents.main_agent -- called
                                directly, not hand-reproduced, so if
                                that schema ever changes this script
                                picks up the new one automatically.

Safety guards (both must pass before anything is touched):
  - Aborts if MT5 is available and reports any open positions.
  - Aborts if main_agent.py is currently running as a process.

Each run's backups get a unique timestamp suffix, so re-running the
script never overwrites a previous backup.

Usage:
    python scripts/reset_vps_data.py
    python scripts/reset_vps_data.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT  = Path(__file__).parent.parent
AGENTS_DIR = REPO_ROOT / 'src' / 'agents'
DATA_DIR   = REPO_ROOT / 'data'
STATE_DIR  = DATA_DIR / 'state'

# src/agents/ modules import each other as flat names (agent_market,
# agent_strategy, ...) -- put the directory on sys.path the same way
# main_agent.py does for itself before importing it.
sys.path.insert(0, str(AGENTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from agent_execution import TRADES_LOG_HEADERS
from agent_reporting import EQUITY_HEADERS, STARTING_BALANCE
from main_agent import _fresh_state

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

BASELINE_DATE = '2026-07-01'


# ---------------------------------------------------------------------------
# Safety checks
# ---------------------------------------------------------------------------

def _open_positions_count() -> int | None:
    """Returns the number of open MT5 positions, or None if MT5 isn't
    available (e.g. running this off the VPS) -- in which case the
    caller skips the check rather than guessing."""
    if not MT5_AVAILABLE or not mt5.initialize():
        return None
    positions = mt5.positions_get()
    return len(positions) if positions is not None else 0


def _bot_process_running() -> bool:
    """Checks whether main_agent.py is currently running, by matching
    process command lines rather than doing a coarse 'any python.exe
    is running' check (main_agent.py isn't the only python process on
    the VPS)."""
    try:
        if sys.platform == 'win32':
            out = subprocess.run(
                ['powershell', '-NoProfile', '-Command',
                 "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
                 "| Select-Object -ExpandProperty CommandLine"],
                capture_output=True, text=True, timeout=15, check=False,
            )
        else:
            # Non-Windows (local dev/testing only -- the real bot only
            # ever runs on the Windows VPS, see requirements-windows.txt).
            out = subprocess.run(
                ['ps', 'ax', '-o', 'command'],
                capture_output=True, text=True, timeout=15, check=False,
            )
        # Require BOTH: the process image is a python interpreter, AND one
        # of its arguments is a path whose filename is exactly
        # main_agent.py. A raw substring/token check on the whole command
        # line would false-positive on any unrelated process that merely
        # mentions the filename in its arguments (an editor, a grep, a
        # shell wrapping a test command that references the path).
        for line in out.stdout.splitlines():
            tokens = [t.strip('"\'') for t in line.split()]
            if not tokens:
                continue
            if not Path(tokens[0]).name.lower().startswith('python'):
                continue
            if any(Path(t).name == 'main_agent.py' for t in tokens[1:]):
                return True
        return False
    except Exception as e:
        print(f'  WARNING: could not check for a running bot process ({e}) -- '
              f'treating as running, to be safe.')
        return True


def run_safety_checks() -> None:
    open_positions = _open_positions_count()
    if open_positions is None:
        print('  MT5 not available -- skipping open-position check.')
    elif open_positions > 0:
        sys.exit(f'ABORTED: {open_positions} open position(s) detected. '
                  f'Close all trades before resetting data files.')
    else:
        print('  No open positions detected.')

    if _bot_process_running():
        sys.exit('ABORTED: ForexBot process is running. Stop the bot before '
                  'resetting data files.')
    print('  ForexBot process is not running.')


# ---------------------------------------------------------------------------
# Reset actions
# ---------------------------------------------------------------------------

def backup(path: Path, timestamp: str) -> None:
    if not path.exists():
        print(f'  (no existing {path.name} -- nothing to back up)')
        return
    backup_path = path.with_stem(f'{path.stem}_backup_{timestamp}')
    shutil.copy(path, backup_path)
    print(f'  Backed up {path.name} -> {backup_path.name}')


def reset_trades_log(timestamp: str) -> None:
    path = DATA_DIR / 'trades_log.csv'
    backup(path, timestamp)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(TRADES_LOG_HEADERS)
    print(f'  Reset {path.name}  (header: {TRADES_LOG_HEADERS})')


def reset_equity_curve(timestamp: str) -> None:
    path = DATA_DIR / 'equity_curve.csv'
    backup(path, timestamp)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    baseline_row = {
        'Date'      : BASELINE_DATE,
        'Balance'   : f'{STARTING_BALANCE:.2f}',
        'DailyPnL'  : '0.00',
        'DailyPct'  : '0.00',
        'Trades'    : 0,
        'Wins'      : 0,
        'Losses'    : 0,
        'WinRate'   : '0.0',
        'CumReturn' : '0.0',
    }
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=EQUITY_HEADERS)
        writer.writeheader()
        writer.writerow(baseline_row)
    print(f'  Reset {path.name}  (header: {EQUITY_HEADERS})')
    print(f'  Baseline row: {baseline_row}')


def reset_daily_state(timestamp: str) -> dict:
    path = STATE_DIR / 'daily_state.json'
    backup(path, timestamp)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    clean_state = _fresh_state(BASELINE_DATE)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(clean_state, f, indent=2, default=str)
    print(f'  Reset {path.name}  (via main_agent._fresh_state({BASELINE_DATE!r}))')
    return clean_state


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def _count_data_rows(path: Path) -> int:
    """Row count excluding the header. 0 if the file doesn't exist."""
    if not path.exists():
        return 0
    with open(path, newline='', encoding='utf-8') as f:
        return max(sum(1 for _ in csv.reader(f)) - 1, 0)


def dry_run() -> None:
    print(f'DRY RUN -- would reset data files to a clean {BASELINE_DATE} baseline.\n')

    trades_rows = _count_data_rows(DATA_DIR / 'trades_log.csv')
    equity_rows = _count_data_rows(DATA_DIR / 'equity_curve.csv')
    state_path  = STATE_DIR / 'daily_state.json'
    state_desc  = 'daily_state.json' if state_path.exists() else 'daily_state.json (does not exist yet)'

    summary = (f'Would reset trades_log.csv ({trades_rows} rows), '
               f'equity_curve.csv ({equity_rows} rows), {state_desc}')

    open_positions = _open_positions_count()
    if open_positions:
        print(f'{summary} -- {open_positions} open position(s) detected, would ABORT')
        return

    if _bot_process_running():
        print(f'{summary} -- ForexBot process is running, would ABORT')
        return

    note = '' if open_positions == 0 else ' (MT5 not available -- open-position check would be skipped)'
    print(f'{summary}{note}')
    print('No blockers detected -- a live run would proceed.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true',
                         help='Show what would be reset without doing it')
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    print('Running safety checks...')
    run_safety_checks()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print(f'\nResetting data files to a clean {BASELINE_DATE} baseline...\n')

    print('trades_log.csv:')
    reset_trades_log(timestamp)

    print('\nequity_curve.csv:')
    reset_equity_curve(timestamp)

    print('\ndaily_state.json:')
    clean_state = reset_daily_state(timestamp)

    print('\nDone -- all data files reset to a clean baseline.\n')
    print('clean daily_state.json content (from _fresh_state()):')
    print(json.dumps(clean_state, indent=2, default=str))


if __name__ == '__main__':
    main()
