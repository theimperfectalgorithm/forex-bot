"""
Phase 49 -- temporal (chronological-midpoint) validation of marginal
factor effects. Diagnostic only.
"""
import numpy as np
import pandas as pd

MIN_CELL = 10


def chronological_split(ledger: pd.DataFrame):
    """Splits by trade-day chronological order at the midpoint -- no
    random split, no re-splitting after seeing results."""
    led = ledger.sort_values('date').reset_index(drop=True)
    mid = len(led) // 2
    return led.iloc[:mid].copy(), led.iloc[mid:].copy()


def _hi_lo(half: pd.DataFrame, col: str):
    if col in ('conc_4plus', 'vol_high'):
        return half[half[col]], half[~half[col]]
    med = half[col].median()
    return half[half[col] >= med], half[half[col] < med]


def run_temporal_validation(ledger: pd.DataFrame, factors=None, min_cell: int = MIN_CELL) -> pd.DataFrame:
    """For each factor, computes the hi-vs-lo daily-R effect in the
    earlier and later chronological halves independently. A finding
    'survives' only if both halves show the same effect DIRECTION with
    adequate sample in both -- never inferred from one half alone."""
    if factors is None:
        factors = [('JPY exposure (median split)', 'jpy_share_pct'), ('AMR exposure (median split)', 'amr_share_pct'),
                   ('Concurrency 4+', 'conc_4plus'), ('HIGH volatility', 'vol_high')]
    earlier, later = chronological_split(ledger)
    rows = []
    for name, col in factors:
        for half_name, half in [('earlier', earlier), ('later', later)]:
            hi, lo = _hi_lo(half, col)
            eff = round(hi.total_R.mean() - lo.total_R.mean(), 4) if len(hi) >= min_cell and len(lo) >= min_cell else None
            rows.append({'factor': name, 'half': half_name, 'n_days': len(half), 'n_hi': len(hi), 'n_lo': len(lo), 'effect_hi_minus_lo': eff})
    df = pd.DataFrame(rows)
    survival = []
    for name in df.factor.unique():
        sub = df[df.factor == name]
        e1 = sub[sub.half == 'earlier']['effect_hi_minus_lo'].iloc[0]
        e2 = sub[sub.half == 'later']['effect_hi_minus_lo'].iloc[0]
        survives = (np.sign(e1) == np.sign(e2)) if (e1 is not None and e2 is not None) else None
        survival.append({'factor': name, 'earlier_effect': e1, 'later_effect': e2, 'survives_temporal_validation': survives})
    df.attrs['survival'] = pd.DataFrame(survival)
    return df
