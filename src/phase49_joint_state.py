"""
Phase 49 -- joint-state combination testing (the 12 preregistered
combinations in reports/phase49_preregistration.md section 7). Reusable,
diagnostic-only module: no strategy/live modification.
"""
import itertools

import pandas as pd

MIN_CELL = 10

# Exactly the 12 combinations frozen in the preregistration -- no others.
COMBOS = {
    'A_vol_x_concurrency': ['vol_high', 'conc_4plus'],
    'B_vol_x_jpy': ['vol_high', 'jpy_high'],
    'C_vol_x_amr': ['vol_high', 'amr_high'],
    'D_vol_x_direction': ['vol_high', 'long_heavy'],
    'E_concurrency_x_jpy': ['conc_4plus', 'jpy_high'],
    'F_concurrency_x_amr': ['conc_4plus', 'amr_high'],
    'G_concurrency_x_direction': ['conc_4plus', 'long_heavy'],
    'H_amr_x_jpy': ['amr_high', 'jpy_high'],
    'I_vol_x_conc_x_amr': ['vol_high', 'conc_4plus', 'amr_high'],
    'J_vol_x_conc_x_jpy': ['vol_high', 'conc_4plus', 'jpy_high'],
    'K_vol_x_conc_x_direction': ['vol_high', 'conc_4plus', 'long_heavy'],
    'L_vol_x_conc_x_jpy_x_direction': ['vol_high', 'conc_4plus', 'jpy_high', 'long_heavy'],
}


def add_binary_flags(ledger: pd.DataFrame) -> pd.DataFrame:
    """Adds the binary state columns the 12 combinations are built from.
    Does not mutate the input in place -- returns a new DataFrame."""
    led = ledger.copy()
    led['vol_high'] = led.vol_state == 'HIGH'
    led['conc_4plus'] = led.max_concurrent >= 4
    led['jpy_high'] = led.jpy_share_pct >= led.jpy_share_pct.median()
    led['amr_high'] = led.amr_share_pct >= led.amr_share_pct.median()
    led['long_heavy'] = led.long_share_pct > led.short_share_pct
    return led


def run_joint_state_analysis(ledger: pd.DataFrame, min_cell: int = MIN_CELL) -> pd.DataFrame:
    """Runs all 12 preregistered combinations and every state (2^k per
    combination), returning one row per combination-state. Cells below
    min_cell are reported as INSUFFICIENT SAMPLE, never interpreted."""
    led = add_binary_flags(ledger)
    rows = []
    for name, cols in COMBOS.items():
        for state in itertools.product([True, False], repeat=len(cols)):
            mask = pd.Series(True, index=led.index)
            for c, s in zip(cols, state):
                mask &= (led[c] == s)
            sub = led[mask]
            if len(sub) < min_cell:
                rows.append({'combination': name, 'state': str(dict(zip(cols, state))), 'n_days': len(sub),
                             'mean_R': None, 'worst_R': None, 'evidence': 'INSUFFICIENT SAMPLE'})
                continue
            rows.append({'combination': name, 'state': str(dict(zip(cols, state))), 'n_days': len(sub),
                         'mean_R': round(sub.total_R.mean(), 4), 'worst_R': round(sub.total_R.min(), 4),
                         'evidence': 'ADEQUATE SAMPLE'})
    return pd.DataFrame(rows)
