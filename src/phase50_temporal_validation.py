"""
Phase 50 -- chronological discovery/validation split and walk-forward
folds for H1/H2. No random splitting, no re-splitting after results.
"""
import pandas as pd


def discovery_validation_split(pred: pd.DataFrame):
    pred_sorted = pred.sort_values('T_date').reset_index(drop=True)
    mid = len(pred_sorted) // 2
    return pred_sorted.iloc[:mid].copy(), pred_sorted.iloc[mid:].copy()


def walk_forward_folds(pred: pd.DataFrame):
    """2-fold expanding-window walk-forward, per the preregistration's
    sample-size-driven fold count (a 3rd fold would under-power the
    final validation window)."""
    pred_sorted = pred.sort_values('T_date').reset_index(drop=True)
    n = len(pred_sorted)
    third = n // 3
    fold1_disc = pred_sorted.iloc[:third]
    fold1_val = pred_sorted.iloc[third:2 * third]
    fold2_disc = pred_sorted.iloc[:2 * third]
    fold2_val = pred_sorted.iloc[2 * third:]
    return [('fold1', fold1_disc, fold1_val), ('fold2', fold2_disc, fold2_val)]
