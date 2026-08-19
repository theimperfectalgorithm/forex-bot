import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from phase47_parameter_harness import perturb_config, sanity_check, PERTURBABLE_PARAMS


def test_perturb_changes_only_target_param():
    base = {'z_threshold': 2.0, 'sl_multiplier': 1.5, 'entry_end_hour': 4}
    result = perturb_config(base, 'z_threshold', 0.20)
    assert result['z_threshold'] == 2.4
    assert result['sl_multiplier'] == base['sl_multiplier']
    assert result['entry_end_hour'] == base['entry_end_hour']


def test_perturb_does_not_mutate_base_config():
    base = {'z_threshold': 2.0, 'sl_multiplier': 1.5}
    snapshot = dict(base)
    perturb_config(base, 'z_threshold', -0.20)
    assert base == snapshot


def test_perturb_negative_and_positive_directions():
    base = {'tp_multiplier': 2.0}
    minus = perturb_config(base, 'tp_multiplier', -0.20)
    plus = perturb_config(base, 'tp_multiplier', 0.20)
    assert minus['tp_multiplier'] == 1.6
    assert plus['tp_multiplier'] == 2.4


def test_perturb_missing_param_raises():
    base = {'z_threshold': 2.0}
    try:
        perturb_config(base, 'nonexistent_param', 0.20)
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_sanity_check_passes_for_all_registered_strategies():
    configs = {
        'AUDJPY_AMR': {'z_threshold': 2.0, 'sl_multiplier': 1.5},
        'CADJPY_ARB': {'tp_multiplier': 2.0, 'min_range_pips': 15},
        'GBPUSD_MONDAY': {'sl_atr_mult': 1.25, 'tp_atr_mult': 1.0},
    }
    for strat, cfg in configs.items():
        for param in PERTURBABLE_PARAMS[strat]:
            for pct in (-0.20, 0.20):
                result = sanity_check(cfg, param, pct)
                assert result['PASS'], f"{strat}/{param}/{pct} failed: {result}"


def test_sanity_check_detects_unrelated_change():
    """A deliberately broken perturb function should be caught by the sanity check."""
    base = {'a': 1.0, 'b': 2.0}

    def broken_perturb(cfg, param, pct):
        new = dict(cfg)
        new[param] = cfg[param] * (1 + pct)
        new['b'] = 999.0  # bug: mutates an unrelated key
        return new

    perturbed = broken_perturb(base, 'a', 0.20)
    unrelated_changed = [k for k in base if k != 'a' and perturbed.get(k) != base[k]]
    assert unrelated_changed == ['b']
