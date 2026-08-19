import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from phase47_cost_harness import apply_cost_multiplier, sanity_check


def test_cost_multiplier_changes_only_r_multiple():
    trade = {'r_multiple': 1.2, 'sl_pips': 40, 'tp_pips': 60, 'direction': 'BUY'}
    stressed = apply_cost_multiplier(trade, 2.0)
    assert stressed['r_multiple'] != trade['r_multiple']
    assert stressed['sl_pips'] == trade['sl_pips']
    assert stressed['tp_pips'] == trade['tp_pips']
    assert stressed['direction'] == trade['direction']


def test_cost_multiplier_does_not_mutate_original_trade():
    trade = {'r_multiple': 1.2, 'sl_pips': 40}
    snapshot = dict(trade)
    apply_cost_multiplier(trade, 2.0)
    assert trade == snapshot


def test_higher_multiplier_reduces_r_more():
    trade = {'r_multiple': 1.2}
    r1 = apply_cost_multiplier(trade, 1.0)['r_multiple']
    r1_5 = apply_cost_multiplier(trade, 1.5)['r_multiple']
    r2 = apply_cost_multiplier(trade, 2.0)['r_multiple']
    assert r1 > r1_5 > r2


def test_sanity_check_passes_at_stressed_multipliers():
    trade = {'r_multiple': -1.0, 'sl_pips': 30, 'tp_pips': 45}
    for mult in (1.5, 2.0):
        result = sanity_check(trade, mult)
        assert result['PASS'], result


def test_sanity_check_baseline_multiplier_is_a_noop():
    """At multiplier=1.0 the stressed R should equal the original (no change) --
    this is the expected identity case, not a failure of the harness."""
    trade = {'r_multiple': 0.5}
    result = sanity_check(trade, 1.0)
    assert result['r_multiple_changed'] is False
    assert result['protected_fields_unchanged'] is True
