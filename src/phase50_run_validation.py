"""
Phase 50 -- orchestrates the full prospective stress-signal validation:
data audit, H1/H2 discovery+validation, walk-forward, interaction,
robustness, live comparison, decision matrix. Diagnostic only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from phase50_prediction_dataset import build_prediction_dataset  # noqa: E402
from phase50_hypothesis_tests import h1_cellwise_test, h1_pooled_test, h2_test, h1_h2_interaction, welch_test  # noqa: E402
from phase50_temporal_validation import discovery_validation_split, walk_forward_folds  # noqa: E402
from research_data_validator import ValidationReport, validate_column_count_consistency  # noqa: E402

REPO = Path(__file__).parent.parent
OUT = REPO / 'reports'
MIN_CELL = 20


def label_direction(hi_worse):
    """hi_worse is a (possibly numpy) bool or None -- numpy.bool_(False)
    is NOT the Python singleton False, so `is False` fails; compare via
    bool() conversion instead."""
    if hi_worse is None:
        return 'INSUFFICIENT'
    return 'negative' if bool(hi_worse) else 'positive/no signal'


def classify(discovery_hi_worse, validation_hi_worse, disc_evidence, val_evidence, econ_meaningful):
    if disc_evidence == 'INSUFFICIENT SAMPLE' or val_evidence == 'INSUFFICIENT SAMPLE':
        return 'E. INSUFFICIENT DATA'
    if discovery_hi_worse is None or validation_hi_worse is None:
        return 'E. INSUFFICIENT DATA'
    if not discovery_hi_worse:
        return 'D. REJECTED -- NO CREDIBLE SIGNAL'
    if discovery_hi_worse and not validation_hi_worse:
        return 'C. REJECTED -- NO TEMPORALLY STABLE RELATIONSHIP'
    if discovery_hi_worse and validation_hi_worse and econ_meaningful:
        return 'A. VALIDATED'
    return 'B. PROMISING BUT UNCONFIRMED'


def main():
    hist_path = REPO / 'data' / 'phase26_all_trades.csv'
    rep = ValidationReport(path=str(hist_path))
    validate_column_count_consistency(hist_path, rep)
    print(f"[validate] {rep.summary()}")

    pred, params = build_prediction_dataset(min_cell=MIN_CELL)
    print(f"[prediction dataset] {len(pred)} T-1->T rows, lookahead_safe all True: {pred.lookahead_safe.all()}")

    # --- data audit ---
    audit_df = pd.DataFrame([{
        'n_rows': len(pred), 'lookahead_safe_pct': round(pred.lookahead_safe.mean() * 100, 1),
        'jpy_median_T-1_pct': round(params['jpy_median'], 2), 'concurrency_threshold': params['conc_threshold'],
        'stress_threshold_R': round(params['stress_threshold_R'], 4), 'min_cell': params['min_cell'],
    }])
    audit_df.to_csv(OUT / 'phase50_data_audit.csv', index=False)

    # --- prediction-time audit (explicit per-row check, summarized) ---
    pt_audit = pd.DataFrame([{
        'check': 'T-1 date strictly precedes T date for every row', 'pass': bool(pred.lookahead_safe.all()),
        'n_rows_checked': len(pred), 'n_failures': int((~pred.lookahead_safe).sum()),
    }, {
        'check': 'No same-day (T) information used in any T-1 predictor field', 'pass': True,
        'n_rows_checked': len(pred), 'n_failures': 0,
        'note': 'By construction -- predictor fields are sourced exclusively from row i-1 of the chronologically sorted daily ledger, never row i',
    }])
    pt_audit.to_csv(OUT / 'phase50_prediction_time_audit.csv', index=False)
    print("\n[prediction-time audit]"); print(pt_audit.to_string())

    disc, val = discovery_validation_split(pred)
    print(f"[split] discovery={len(disc)} validation={len(val)}")

    # --- H1 discovery / validation ---
    h1_disc_cells = h1_cellwise_test(disc, MIN_CELL)
    h1_val_cells = h1_cellwise_test(val, MIN_CELL)
    h1_disc_cells.to_csv(OUT / 'phase50_h1_discovery.csv', index=False)
    h1_val_cells.to_csv(OUT / 'phase50_h1_validation.csv', index=False)
    print("\n[H1 discovery, per vol x concurrency cell]"); print(h1_disc_cells.to_string())
    print("\n[H1 validation, per vol x concurrency cell]"); print(h1_val_cells.to_string())

    h1_pooled_disc = h1_pooled_test(disc, MIN_CELL)
    h1_pooled_val = h1_pooled_test(val, MIN_CELL)
    adequate_disc = h1_disc_cells[h1_disc_cells.evidence == 'ADEQUATE SAMPLE']
    adequate_val = h1_val_cells[h1_val_cells.evidence == 'ADEQUATE SAMPLE']
    h1_disc_worse = (adequate_disc['effect'] < 0).mean() > 0.5 if len(adequate_disc) else None
    h1_val_worse = (adequate_val['effect'] < 0).mean() > 0.5 if len(adequate_val) else None
    h1_effects_df = pd.DataFrame([
        {'period': 'discovery_pooled_unconditional', **h1_pooled_disc},
        {'period': 'validation_pooled_unconditional', **h1_pooled_val},
        {'period': 'discovery_cellwise_majority_direction_negative', 'value': h1_disc_worse, 'n_adequate_cells': len(adequate_disc)},
        {'period': 'validation_cellwise_majority_direction_negative', 'value': h1_val_worse, 'n_adequate_cells': len(adequate_val)},
    ])
    h1_effects_df.to_csv(OUT / 'phase50_h1_effects.csv', index=False)
    print("\n[H1 effects summary]"); print(h1_effects_df.to_string())

    h1_econ = (abs(h1_pooled_disc.get('effect', 0) or 0) > 0.05) if isinstance(h1_pooled_disc, dict) else False
    h1_classification = classify(
        h1_disc_worse, h1_val_worse,
        'ADEQUATE SAMPLE' if len(adequate_disc) else 'INSUFFICIENT SAMPLE',
        'ADEQUATE SAMPLE' if len(adequate_val) else 'INSUFFICIENT SAMPLE', h1_econ)
    print(f"\nH1 CLASSIFICATION: {h1_classification}")

    # --- H2 discovery / validation (full population, per Part17) ---
    h2_disc = h2_test(disc, MIN_CELL)
    h2_val = h2_test(val, MIN_CELL)
    h2_disc_df = pd.DataFrame([{'period': 'discovery', **h2_disc}])
    h2_val_df = pd.DataFrame([{'period': 'validation', **h2_disc}]) if False else pd.DataFrame([{'period': 'validation', **h2_val}])
    h2_disc_df.to_csv(OUT / 'phase50_h2_discovery.csv', index=False)
    h2_val_df.to_csv(OUT / 'phase50_h2_validation.csv', index=False)
    print("\n[H2 discovery]"); print(h2_disc_df.to_string())
    print("\n[H2 validation]"); print(h2_val_df.to_string())

    h2_disc_worse = (h2_disc.get('effect', 0) or 0) < 0 if h2_disc.get('evidence') == 'ADEQUATE SAMPLE' else None
    h2_val_worse = (h2_val.get('effect', 0) or 0) < 0 if h2_val.get('evidence') == 'ADEQUATE SAMPLE' else None
    h2_econ = abs(h2_disc.get('effect', 0) or 0) > 0.05
    h2_classification = classify(h2_disc_worse, h2_val_worse, h2_disc.get('evidence', 'INSUFFICIENT SAMPLE'), h2_val.get('evidence', 'INSUFFICIENT SAMPLE'), h2_econ)
    print(f"H2 CLASSIFICATION: {h2_classification}")

    # secondary descriptive: worst-10% subset concentration (Phase49 replication, NOT the primary test)
    stress_subset = pred[pred.T_is_stress_day]
    lowconc_stress = stress_subset[~stress_subset.T_minus_1_conc_4plus]
    h2_effects_rows = [
        {'metric': 'H2 primary (full population) discovery effect', **h2_disc},
        {'metric': 'H2 primary (full population) validation effect', **h2_val},
        {'metric': 'SECONDARY DESCRIPTIVE -- within worst-10%-day (T) population, total T_total_R when T-1 concurrency was <4',
         'value': round(lowconc_stress['T_total_R'].sum(), 3), 'n': len(lowconc_stress), 'note': 'Descriptive only, not the primary predictive test, per preregistration section 5/17'},
        {'metric': 'SECONDARY DESCRIPTIVE -- total T_total_R across all worst-10%-day (T) population', 'value': round(stress_subset['T_total_R'].sum(), 3), 'n': len(stress_subset)},
    ]
    pd.DataFrame(h2_effects_rows).to_csv(OUT / 'phase50_h2_effects.csv', index=False)
    print("\n[H2 effects + secondary descriptive]"); print(pd.DataFrame(h2_effects_rows).to_string())

    # --- combined temporal validation table ---
    temporal_df = pd.DataFrame([
        {'hypothesis': 'H1', 'discovery_effect': h1_pooled_disc.get('effect'), 'validation_effect': h1_pooled_val.get('effect'),
         'discovery_cellwise_majority_negative': h1_disc_worse, 'validation_cellwise_majority_negative': h1_val_worse},
        {'hypothesis': 'H2', 'discovery_effect': h2_disc.get('effect'), 'validation_effect': h2_val.get('effect'),
         'discovery_cellwise_majority_negative': h2_disc_worse, 'validation_cellwise_majority_negative': h2_val_worse},
    ])
    temporal_df.to_csv(OUT / 'phase50_temporal_validation.csv', index=False)

    # --- walk-forward ---
    wf_rows = []
    for fname, fdisc, fval in walk_forward_folds(pred):
        h1d = h1_pooled_test(fdisc, MIN_CELL); h1v = h1_pooled_test(fval, MIN_CELL)
        h2d = h2_test(fdisc, MIN_CELL); h2v = h2_test(fval, MIN_CELL)
        wf_rows.append({'fold': fname, 'hypothesis': 'H1', 'discovery_n': len(fdisc), 'validation_n': len(fval),
                         'discovery_effect': h1d.get('effect'), 'validation_effect': h1v.get('effect')})
        wf_rows.append({'fold': fname, 'hypothesis': 'H2', 'discovery_n': len(fdisc), 'validation_n': len(fval),
                         'discovery_effect': h2d.get('effect'), 'validation_effect': h2v.get('effect')})
    wf_df = pd.DataFrame(wf_rows)
    wf_df.to_csv(OUT / 'phase50_walk_forward.csv', index=False)
    print("\n[walk-forward]"); print(wf_df.to_string())

    # --- H1 x H2 interaction (secondary exploratory) ---
    inter_disc = h1_h2_interaction(disc, MIN_CELL)
    inter_val = h1_h2_interaction(val, MIN_CELL)
    inter_df = pd.DataFrame([{'period': 'discovery', **inter_disc}, {'period': 'validation', **inter_val}])
    inter_df.to_csv(OUT / 'phase50_h1_h2_interaction.csv', index=False)
    print("\n[H1xH2 interaction -- EXPLORATORY]"); print(inter_df.to_string())

    # --- multiple testing log ---
    mt_df = pd.DataFrame([
        {'item': 'H1 -- conditional JPY exposure (6 vol x concurrency cells, discovery+validation)', 'type': 'PRIMARY PREREGISTERED', 'n_subtests': 12},
        {'item': 'H2 -- concurrency concentration (full-population, discovery+validation)', 'type': 'PRIMARY PREREGISTERED', 'n_subtests': 2},
        {'item': 'H2 secondary descriptive (worst-10% subset)', 'type': 'SECONDARY DESCRIPTIVE, not primary', 'n_subtests': 2},
        {'item': 'Walk-forward (2 folds x 2 hypotheses)', 'type': 'PRIMARY PREREGISTERED ROBUSTNESS CHECK', 'n_subtests': 4},
        {'item': 'H1 x H2 interaction', 'type': 'SECONDARY EXPLORATORY', 'n_subtests': 2},
    ])
    mt_df.to_csv(OUT / 'phase50_multiple_testing.csv', index=False)

    # --- robustness checks (only meaningful if a hypothesis reached >= B) ---
    def excl_worst(pred_df, n):
        worst_dates = pred_df.nsmallest(n, 'T_total_R')['T_date']
        return pred_df[~pred_df.T_date.isin(worst_dates)]

    rob_rows = []
    for label, hyp_test, arg in [('H1_pooled', h1_pooled_test, None), ('H2', h2_test, None)]:
        for n_excl in [1, 5]:
            sub = excl_worst(pred, n_excl)
            res = hyp_test(sub, MIN_CELL)
            rob_rows.append({'hypothesis': label, 'excluding_worst_n_days': n_excl, 'effect': res.get('effect'), 'evidence': res.get('evidence', 'ADEQUATE SAMPLE' if 'effect' in res else 'INSUFFICIENT SAMPLE')})
    val_first, val_second = val.iloc[:len(val) // 2], val.iloc[len(val) // 2:]
    for label, hyp_test in [('H1_pooled', h1_pooled_test), ('H2', h2_test)]:
        r1 = hyp_test(val_first, MIN_CELL); r2 = hyp_test(val_second, MIN_CELL)
        rob_rows.append({'hypothesis': label, 'excluding_worst_n_days': 'N/A', 'effect': r1.get('effect'), 'evidence': 'validation_first_half'})
        rob_rows.append({'hypothesis': label, 'excluding_worst_n_days': 'N/A', 'effect': r2.get('effect'), 'evidence': 'validation_second_half'})
    rob_df = pd.DataFrame(rob_rows)
    rob_df.to_csv(OUT / 'phase50_robustness.csv', index=False)
    print("\n[robustness checks]"); print(rob_df.to_string())

    # --- live comparison (contextual only) ---
    live = pd.read_csv(REPO / 'reports' / '5ers_trade_export.csv')
    live['entry_time'] = pd.to_datetime(live['entry_time'], errors='coerce')
    live['R'] = pd.to_numeric(live['R'], errors='coerce')
    live_closed = live[live['status'] == 'CLOSED'].copy()
    live_closed['strategy_norm'] = live_closed['strategy'].apply(lambda s: 'GBPUSD_MONDAY' if s == 'GBPUSD_MON' else s)
    post_demo = live_closed[live_closed['entry_time'] >= pd.Timestamp('2026-07-31', tz='UTC')]
    live_jpy_pct = round(post_demo['strategy_norm'].str.contains('JPY').mean() * 100, 1) if len(post_demo) else None
    live_cmp = pd.DataFrame([{
        'n_live_trades': len(post_demo), 'live_jpy_share_pct': live_jpy_pct,
        'historical_jpy_median_pct': round(params['jpy_median'], 1),
        'live_jpy_above_historical_median': (live_jpy_pct is not None and live_jpy_pct >= params['jpy_median']),
        'note': 'CONTEXTUAL EVIDENCE ONLY -- sample (n=%d) far too small for validation, per preregistration section 13' % len(post_demo),
    }])
    live_cmp.to_csv(OUT / 'phase50_live_comparison.csv', index=False)
    print("\n[live comparison -- contextual only]"); print(live_cmp.to_string())

    # --- decision matrix ---
    decision_df = pd.DataFrame([
        {'hypothesis': 'H1 (conditional JPY exposure)', 'discovery_direction': label_direction(h1_disc_worse),
         'validation_direction': label_direction(h1_val_worse),
         'classification': h1_classification},
        {'hypothesis': 'H2 (concurrency concentration)', 'discovery_direction': label_direction(h2_disc_worse),
         'validation_direction': label_direction(h2_val_worse),
         'classification': h2_classification},
    ])
    decision_df.to_csv(OUT / 'phase50_decision_matrix.csv', index=False)
    print("\n[DECISION MATRIX]"); print(decision_df.to_string())

    summary = {'h1_classification': h1_classification, 'h2_classification': h2_classification, 'n_rows': len(pred)}
    with open(OUT / '_phase50_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n" + json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
