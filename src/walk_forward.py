"""
Walk-forward evaluator -- reusable across any strategy's signal list.

HONEST DATA-CONSTRAINT NOTE: classical walk-forward (e.g. train 1-2yr /
test 1yr, rolled forward across many folds) assumes 5-10+ years of
history. This project's MT5 feed provides ~3 years (see `fetch()` in
strategy_matrix_backtest.py). With 3 years, a traditional multi-fold
roll isn't honestly available -- so this module adapts window size to
whatever span is passed in (per the master-prompt's own instruction:
"if data availability differs, dynamically adjust the periods while
maintaining chronological separation") and REPORTS the number of folds
it actually achieved. Fewer than ~4 folds should be treated as weak
walk-forward evidence, not disqualifying but not strong confirmation
either -- say so plainly in any report that uses this.

Usage:
    from walk_forward import run_walk_forward
    folds = run_walk_forward(bars, candidates_by_bar, pip, spread,
                              risk_pct, train_months=12, test_months=6)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    test_metrics: dict


def run_walk_forward(bars: pd.DataFrame, candidates: list, pip: float,
                     spread_pips: float, risk_pct: float,
                     train_months: int = 12, test_months: int = 6,
                     time_exit_hour: int | None = None) -> pd.DataFrame:
    """
    Rolls a train_months/test_months window forward by test_months each
    step across the full bar span. The strategy's SIGNAL LIST is fixed
    (candidates were generated once over the whole history -- this
    function does not re-select parameters per fold, it only measures
    how a FIXED rule performs across different historical stretches,
    which is what matters for robustness: a real edge should not need
    to be re-tuned every fold to survive).

    Returns one row per fold with that fold's TEST-period metrics, plus
    a final 'ALL_FOLDS' summary row (mean/std/consistency).
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from strategy_matrix_backtest import run_sim, metrics

    tdf, eq = run_sim(bars, candidates, pip, spread_pips, risk_pct,
                      time_exit_hour=time_exit_hour)

    span_start = bars.index[0]
    span_end   = bars.index[-1]
    total_months = (span_end.year - span_start.year) * 12 + \
                   (span_end.month - span_start.month)

    rows = []
    fold_id = 0
    cursor = span_start + pd.DateOffset(months=train_months)
    while cursor + pd.DateOffset(months=test_months) <= span_end:
        test_start = cursor
        test_end   = cursor + pd.DateOffset(months=test_months)
        m = metrics(tdf, eq, test_start, test_end)
        rows.append({
            'fold': fold_id, 'test_start': test_start.date(),
            'test_end': test_end.date(),
            'trades': m.get('trades', 0), 'win_rate': m.get('win_rate', 0),
            'pf': m.get('pf', 0), 'pnl': m.get('pnl', 0),
            'max_dd_pct': m.get('max_dd_pct', 0),
        })
        fold_id += 1
        cursor += pd.DateOffset(months=test_months)

    wf = pd.DataFrame(rows)
    if wf.empty:
        return wf

    n_folds = len(wf)
    pf_vals = wf['pf'].replace([np.inf, -np.inf], np.nan).dropna()
    profitable_folds = (wf['pnl'] > 0).sum()
    summary = pd.DataFrame([{
        'fold': 'SUMMARY', 'test_start': None, 'test_end': None,
        'trades': wf['trades'].sum(),
        'win_rate': round(wf['win_rate'].mean(), 1),
        'pf': round(pf_vals.mean(), 2) if len(pf_vals) else 0,
        'pnl': round(wf['pnl'].sum(), 2),
        'max_dd_pct': round(wf['max_dd_pct'].max(), 2),
    }])
    wf = pd.concat([wf, summary], ignore_index=True)
    wf.attrs['n_folds'] = n_folds
    wf.attrs['profitable_folds'] = int(profitable_folds)
    wf.attrs['consistency_pct'] = round(profitable_folds / n_folds * 100, 1) if n_folds else 0
    wf.attrs['data_constraint_note'] = (
        f'{n_folds} fold(s) achievable from available history -- '
        f'{"WEAK evidence (fewer than 4 folds)" if n_folds < 4 else "adequate fold count"}')
    return wf
