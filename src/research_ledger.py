"""
Experiment ledger -- append-only, git-tracked record of every research
experiment (never lost, never overwritten). Lives at
experiments/experiments.csv (repo root, NOT under data/ so it isn't
caught by data/*.csv in .gitignore -- this file is meant to be
committed and to accumulate across the whole project's life).

One row per experiment. Status vocabulary matches the strategy-status
system: RESEARCHING, BASELINE, PROMISING, OVERFIT_RISK, FAILED,
VALIDATION, OUT_OF_SAMPLE, PAPER_TRADING, LIVE_CANDIDATE, REJECTED.

Usage:
    from research_ledger import log_experiment, next_experiment_id
    exp_id = next_experiment_id()
    log_experiment(exp_id, family='pdh_pdl', hypothesis='...', pair='EURUSD',
                   session='london', params={'sl_atr': 1.5, ...},
                   is_metrics=mi, oos_metrics=mo, decision='REJECT',
                   status='FAILED', notes='...')
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

LEDGER_DIR  = Path(__file__).parent.parent / 'experiments'
LEDGER_FILE = LEDGER_DIR / 'experiments.csv'

FIELDS = [
    'experiment_id', 'date', 'family', 'hypothesis', 'pair', 'session',
    'timeframe', 'params_json', 'dataset_start', 'dataset_end',
    'is_trades', 'is_wr', 'is_pf', 'is_dd', 'is_prof_months',
    'oos_trades', 'oos_wr', 'oos_pf', 'oos_pnl', 'oos_dd',
    'wf_folds', 'wf_consistency_pct', 'mc_risk_of_ruin_pct',
    'decision', 'status', 'notes',
]


def next_experiment_id() -> str:
    """EXP-001, EXP-002, ... -- scans the existing ledger for the max."""
    if not LEDGER_FILE.exists():
        return 'EXP-001'
    n = 0
    with open(LEDGER_FILE, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try:
                n = max(n, int(row['experiment_id'].split('-')[1]))
            except Exception:
                continue
    return f'EXP-{n + 1:03d}'


def log_experiment(experiment_id: str, family: str, hypothesis: str,
                   pair: str, session: str = '', timeframe: str = 'H1',
                   params: dict | None = None,
                   dataset_start=None, dataset_end=None,
                   is_metrics: dict | None = None,
                   oos_metrics: dict | None = None,
                   wf_folds: int | None = None,
                   wf_consistency_pct: float | None = None,
                   mc_risk_of_ruin_pct: float | None = None,
                   decision: str = 'INVESTIGATE', status: str = 'RESEARCHING',
                   notes: str = ''):
    """Appends one row. NEVER overwrites or deletes prior rows -- if you
    need to revise an experiment, log a NEW row referencing the old
    experiment_id in `notes` rather than editing history."""
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    is_metrics = is_metrics or {}
    oos_metrics = oos_metrics or {}
    row = {
        'experiment_id': experiment_id,
        'date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
        'family': family, 'hypothesis': hypothesis, 'pair': pair,
        'session': session, 'timeframe': timeframe,
        'params_json': json.dumps(params or {}),
        'dataset_start': str(dataset_start) if dataset_start else '',
        'dataset_end': str(dataset_end) if dataset_end else '',
        'is_trades': is_metrics.get('trades', 0),
        'is_wr': is_metrics.get('win_rate', 0),
        'is_pf': is_metrics.get('pf', 0),
        'is_dd': is_metrics.get('max_dd_pct', 0),
        'is_prof_months': is_metrics.get('prof_months_pct', 0),
        'oos_trades': oos_metrics.get('trades', 0),
        'oos_wr': oos_metrics.get('win_rate', 0),
        'oos_pf': oos_metrics.get('pf', 0),
        'oos_pnl': oos_metrics.get('pnl', 0),
        'oos_dd': oos_metrics.get('max_dd_pct', 0),
        'wf_folds': wf_folds if wf_folds is not None else '',
        'wf_consistency_pct': wf_consistency_pct if wf_consistency_pct is not None else '',
        'mc_risk_of_ruin_pct': mc_risk_of_ruin_pct if mc_risk_of_ruin_pct is not None else '',
        'decision': decision, 'status': status, 'notes': notes,
    }
    file_exists = LEDGER_FILE.exists()
    with open(LEDGER_FILE, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not file_exists:
            w.writeheader()
        w.writerow(row)
    return row
