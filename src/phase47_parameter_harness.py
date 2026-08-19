"""
Phase 47 Stage B -- parameter-perturbation harness (SANITY-TESTED ONLY
in this phase; no final robustness conclusions are produced here, per
reports/phase47_preregistration.md section 9/21).

Identifies each strategy's continuous/discrete perturbable parameters
and provides run_with_perturbation(), which changes EXACTLY one
parameter and verifies no other config value is touched.
"""
from pathlib import Path

REPO = Path(__file__).parent.parent

# Perturbable parameters per strategy, frozen per the preregistration.
# h4_filter and session-window boundaries are explicitly EXCLUDED
# (categorical / structural, not +/-20% perturbation targets).
PERTURBABLE_PARAMS = {
    'AUDJPY_AMR': ['z_threshold', 'sl_multiplier'],
    'CADJPY_AMR': ['z_threshold', 'sl_multiplier'],
    'EURJPY_AMR': ['z_threshold', 'sl_multiplier'],
    'GBPJPY_AMR': ['z_threshold', 'sl_multiplier'],
    'CADJPY_ARB': ['tp_multiplier', 'min_range_pips'],
    'GBPUSD_MONDAY': ['sl_atr_mult', 'tp_atr_mult'],
}

NON_PERTURBABLE = {
    'h4_filter': 'categorical (binary on/off) -- not a +/-20% target, per preregistration section 9',
    'entry_end_hour': 'discrete, per-pair structural choice -- not perturbed per Part 18',
    'session': 'structural session-window definition -- not perturbed per Part 18',
    'friday_close': 'categorical execution rule, not a strategy parameter',
    'risk_percent': 'position-sizing input, not a signal parameter -- out of this harness scope',
}


def perturb_config(base_config: dict, param: str, pct: float) -> dict:
    """Return a NEW config dict with exactly one parameter changed by pct
    (e.g. -0.20 or +0.20). Does not mutate base_config. Never touches the
    live YAML file on disk -- this operates on an in-memory dict only."""
    if param not in base_config:
        raise KeyError(f"{param} not present in base_config")
    new_config = dict(base_config)  # shallow copy -- new dict object
    new_config[param] = round(base_config[param] * (1 + pct), 6)
    return new_config


def sanity_check(base_config: dict, param: str, pct: float = 0.20) -> dict:
    """Software-correctness test: perturbing `param` must change ONLY
    that key; every other key must be identical to base_config, and
    base_config itself must be unmodified (no in-place mutation)."""
    base_snapshot = dict(base_config)
    perturbed = perturb_config(base_config, param, pct)

    unrelated_changed = [k for k in base_config if k != param and perturbed.get(k) != base_config[k]]
    base_mutated = base_config != base_snapshot
    target_changed = perturbed[param] != base_config[param]

    return {
        'param': param, 'pct': pct,
        'target_param_changed': target_changed,
        'unrelated_params_unchanged': len(unrelated_changed) == 0,
        'unrelated_changed_keys': unrelated_changed,
        'base_config_not_mutated': not base_mutated,
        'PASS': target_changed and len(unrelated_changed) == 0 and not base_mutated,
    }
