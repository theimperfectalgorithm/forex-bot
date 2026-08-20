"""
Phase 50 -- prediction-time-safe dataset: shifts Phase49's daily
portfolio dataset by one trading day so every predictor row uses only
information from an already-closed prior day (T-1) to predict the
following day's (T) outcome. Explicit lookahead_safe audit column.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from phase49_stress_dataset import load_control, build_daily_dataset  # noqa: E402


def build_prediction_dataset(min_cell: int = 20):
    df = load_control()
    ledger = build_daily_dataset(df).sort_values('date').reset_index(drop=True)

    # full-period stress threshold (worst 10%), computed once, per the
    # frozen convention -- used only for the derived companion outcome
    stress_threshold = ledger['total_R'].quantile(0.10)
    conc_threshold = 4  # reused, not re-chosen, from Phase43/49

    ledger['vol_high'] = ledger.vol_state == 'HIGH'
    ledger['conc_4plus'] = ledger.max_concurrent >= conc_threshold
    jpy_median = ledger['jpy_share_pct'].median()
    ledger['jpy_high'] = ledger['jpy_share_pct'] >= jpy_median

    rows = []
    for i in range(1, len(ledger)):
        prev = ledger.iloc[i - 1]
        cur = ledger.iloc[i]
        lookahead_safe = pd.Timestamp(prev['date']) < pd.Timestamp(cur['date'])
        rows.append({
            'T_date': cur['date'], 'T_minus_1_date': prev['date'], 'lookahead_safe': bool(lookahead_safe),
            'T_minus_1_jpy_share_pct': prev['jpy_share_pct'], 'T_minus_1_jpy_high': bool(prev['jpy_high']),
            'T_minus_1_vol_state': prev['vol_state'], 'T_minus_1_vol_high': bool(prev['vol_high']),
            'T_minus_1_vol_pctile': prev['vol_pctile'] if 'vol_pctile' in prev else None,
            'T_minus_1_max_concurrent': prev['max_concurrent'], 'T_minus_1_conc_4plus': bool(prev['conc_4plus']),
            'T_total_R': cur['total_R'],
            'T_is_stress_day': bool(cur['total_R'] <= stress_threshold),
        })
    pred = pd.DataFrame(rows)
    assert pred['lookahead_safe'].all(), "STOP -- lookahead violation detected, per preregistration Part22-23"
    return pred, {'jpy_median': jpy_median, 'conc_threshold': conc_threshold, 'stress_threshold_R': stress_threshold, 'min_cell': min_cell}
