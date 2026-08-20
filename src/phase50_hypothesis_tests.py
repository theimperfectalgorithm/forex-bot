"""
Phase 50 -- H1 (conditional JPY) and H2 (concurrency concentration)
prospective, prediction-time-safe hypothesis tests. Diagnostic only.
"""
import numpy as np
import pandas as pd


def welch_test(hi: np.ndarray, lo: np.ndarray):
    n1, n2 = len(hi), len(lo)
    if n1 < 2 or n2 < 2:
        return None
    m1, m2 = hi.mean(), lo.mean()
    v1, v2 = hi.var(ddof=1), lo.var(ddof=1)
    se = np.sqrt(v1 / n1 + v2 / n2)
    diff = m1 - m2
    ci95 = 1.96 * se
    return {'mean_hi': round(m1, 4), 'mean_lo': round(m2, 4), 'n_hi': n1, 'n_lo': n2,
            'effect': round(diff, 4), 'se': round(se, 4), 'ci95_lo': round(diff - ci95, 4), 'ci95_hi': round(diff + ci95, 4)}


def h1_cellwise_test(pred: pd.DataFrame, min_cell: int) -> pd.DataFrame:
    """H1: within each vol_state(T-1) x concurrency-bucket(T-1) cell,
    compare T_total_R for JPY-high(T-1) vs JPY-low(T-1)."""
    rows = []
    for vs in ['LOW', 'NORMAL', 'HIGH']:
        for cf in [True, False]:
            cell = pred[(pred.T_minus_1_vol_state == vs) & (pred.T_minus_1_conc_4plus == cf)]
            hi = cell[cell.T_minus_1_jpy_high]['T_total_R'].values
            lo = cell[~cell.T_minus_1_jpy_high]['T_total_R'].values
            res = welch_test(hi, lo)
            if res is None or len(hi) < min_cell or len(lo) < min_cell:
                rows.append({'vol_state_T-1': vs, 'concurrency_4plus_T-1': cf, 'n_hi': len(hi), 'n_lo': len(lo), 'evidence': 'INSUFFICIENT SAMPLE'})
                continue
            rows.append({'vol_state_T-1': vs, 'concurrency_4plus_T-1': cf, **res, 'evidence': 'ADEQUATE SAMPLE',
                         'stress_rate_hi_pct': round(cell[cell.T_minus_1_jpy_high]['T_is_stress_day'].mean() * 100, 1),
                         'stress_rate_lo_pct': round(cell[~cell.T_minus_1_jpy_high]['T_is_stress_day'].mean() * 100, 1)})
    return pd.DataFrame(rows)


def h1_pooled_test(pred: pd.DataFrame, min_cell: int) -> dict:
    """Pooled H1 test across all cells (weighted by simple pooling of
    within-cell differences is avoided -- report the direct full-sample
    JPY-high-vs-low comparison as the headline pooled statistic,
    disclosed as NOT conditioning-adjusted, alongside the cellwise table
    which IS conditioning-adjusted (the actual primary test)."""
    hi = pred[pred.T_minus_1_jpy_high]['T_total_R'].values
    lo = pred[~pred.T_minus_1_jpy_high]['T_total_R'].values
    res = welch_test(hi, lo)
    return res or {'n_hi': len(hi), 'n_lo': len(lo), 'evidence': 'INSUFFICIENT SAMPLE'}


def h2_test(pred: pd.DataFrame, min_cell: int) -> dict:
    """H2 primary test on the FULL population (not the stress subset)."""
    hi = pred[pred.T_minus_1_conc_4plus]['T_total_R'].values
    lo = pred[~pred.T_minus_1_conc_4plus]['T_total_R'].values
    res = welch_test(hi, lo)
    if res is None or len(hi) < min_cell or len(lo) < min_cell:
        return {'n_hi': len(hi), 'n_lo': len(lo), 'evidence': 'INSUFFICIENT SAMPLE'}
    res['evidence'] = 'ADEQUATE SAMPLE'
    res['stress_rate_hi_pct'] = round(pred[pred.T_minus_1_conc_4plus]['T_is_stress_day'].mean() * 100, 1)
    res['stress_rate_lo_pct'] = round(pred[~pred.T_minus_1_conc_4plus]['T_is_stress_day'].mean() * 100, 1)
    return res


def h1_h2_interaction(pred: pd.DataFrame, min_cell: int) -> dict:
    both = pred[pred.T_minus_1_jpy_high & pred.T_minus_1_conc_4plus]['T_total_R'].values
    other = pred[~(pred.T_minus_1_jpy_high & pred.T_minus_1_conc_4plus)]['T_total_R'].values
    res = welch_test(both, other)
    if res is None or len(both) < min_cell or len(other) < min_cell:
        return {'n_both': len(both), 'n_other': len(other), 'evidence': 'INSUFFICIENT SAMPLE', 'label': 'EXPLORATORY'}
    res['evidence'] = 'ADEQUATE SAMPLE'
    res['label'] = 'EXPLORATORY'
    return res
