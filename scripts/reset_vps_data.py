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

Usage:
    python scripts/reset_vps_data.py
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
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

BASELINE_DATE = '2026-07-01'


def backup(path: Path) -> None:
    if not path.exists():
        print(f'  (no existing {path.name} -- nothing to back up)')
        return
    backup_path = path.with_stem(path.stem + '_backup_pre_reset')
    shutil.copy(path, backup_path)
    print(f'  Backed up {path.name} -> {backup_path.name}')


def reset_trades_log() -> None:
    path = DATA_DIR / 'trades_log.csv'
    backup(path)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(TRADES_LOG_HEADERS)
    print(f'  Reset {path.name}  (header: {TRADES_LOG_HEADERS})')


def reset_equity_curve() -> None:
    path = DATA_DIR / 'equity_curve.csv'
    backup(path)
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


def reset_daily_state() -> dict:
    path = STATE_DIR / 'daily_state.json'
    backup(path)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    clean_state = _fresh_state(BASELINE_DATE)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(clean_state, f, indent=2, default=str)
    print(f'  Reset {path.name}  (via main_agent._fresh_state({BASELINE_DATE!r}))')
    return clean_state


def main() -> None:
    print(f'Resetting data files to a clean {BASELINE_DATE} baseline...\n')

    print('trades_log.csv:')
    reset_trades_log()

    print('\nequity_curve.csv:')
    reset_equity_curve()

    print('\ndaily_state.json:')
    clean_state = reset_daily_state()

    print('\nDone -- all data files reset to a clean baseline.\n')
    print('clean daily_state.json content (from _fresh_state()):')
    print(json.dumps(clean_state, indent=2, default=str))


if __name__ == '__main__':
    main()
