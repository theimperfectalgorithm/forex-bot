"""
Phase 47 Stage B -- cost-stress harness (SANITY-TESTED ONLY in this
phase; no final robustness conclusions produced here, per
reports/phase47_preregistration.md section 9/21).

Applies a cost multiplier to a trade's R-multiple WITHOUT touching
signal generation, SL/TP, or trade timing -- verified by the sanity
test below.
"""
COST_PER_TRADE_R = 0.02  # placeholder flat-cost-in-R unit, consistent with
                          # this project's convention of a small fixed cost
                          # subtracted from the raw move (see Phase26-40's
                          # COST=0.00018 price-unit convention, expressed
                          # here in R-space since this dataset's r_multiple
                          # is already risk-normalized)


def apply_cost_multiplier(trade: dict, multiplier: float) -> dict:
    """Return a NEW trade dict with only r_multiple adjusted for the
    stressed cost; entry/exit/SL/TP/timing fields are copied unchanged."""
    new_trade = dict(trade)
    raw_r = trade['r_multiple'] + COST_PER_TRADE_R  # back out the baseline cost already embedded
    stressed_r = raw_r - COST_PER_TRADE_R * multiplier
    new_trade['r_multiple'] = round(stressed_r, 4)
    return new_trade


def sanity_check(trade: dict, multiplier: float = 2.0) -> dict:
    """Software-correctness test: only r_multiple may change; every
    other field (signal/SL/TP/timing) must be identical, and the
    original trade dict must not be mutated in place."""
    base_snapshot = dict(trade)
    stressed = apply_cost_multiplier(trade, multiplier)

    protected_fields = [k for k in trade if k != 'r_multiple']
    unrelated_changed = [k for k in protected_fields if stressed.get(k) != trade[k]]
    base_mutated = trade != base_snapshot
    r_changed = stressed['r_multiple'] != trade['r_multiple']

    return {
        'multiplier': multiplier,
        'r_multiple_changed': r_changed,
        'protected_fields_unchanged': len(unrelated_changed) == 0,
        'unrelated_changed_fields': unrelated_changed,
        'base_trade_not_mutated': not base_mutated,
        'PASS': r_changed and len(unrelated_changed) == 0 and not base_mutated,
    }
