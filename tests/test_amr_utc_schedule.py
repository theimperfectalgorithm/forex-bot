"""
Regression test for the AMR (AsianHoursReversion) orchestrator timezone
bug: AMR entry/exit gating must run on real UTC (`t`), not MT5 server
minutes (`srv`) -- confirmed from the trading log, a CADJPY AMR TIME EXIT
fired at 04:00 UTC while the broker was UTC+3 (should have been 07:00
UTC). ARB/monday_drift remain correctly server-clock gated and must not
be touched by this fix.

Run directly: python tests/test_amr_utc_schedule.py
Also pytest-discoverable (test_ functions, plain asserts).
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.agents import main_agent as ma


def test_amr_utc_constants_exist_and_are_correct():
    assert ma.T_AMR_START == 0 * 60
    assert ma.T_AMR_END == 6 * 60
    assert ma.T_AMR_EXIT == 7 * 60


def test_no_amr_orchestration_condition_uses_srv():
    """The AMR entry-poll and AMR time-exit `if` conditions in main() must
    gate on `t` (real UTC), never on `srv` (server time)."""
    src = inspect.getsource(ma.main)
    assert 'if (t <= T_AMR_END and AMR_KEYS' in src, \
        "AMR entry-poll condition must gate on real-UTC `t`, not `srv`"
    assert 'if (t >= T_AMR_EXIT and AMR_KEYS' in src, \
        "AMR time-exit condition must gate on real-UTC `t`, not `srv`"
    assert 'srv <= T_ASIAN_END' not in src
    assert 'srv >= T_ASIAN_EXIT' not in src
    assert not hasattr(ma, 'T_ASIAN_END')
    assert not hasattr(ma, 'T_ASIAN_EXIT')


def test_monday_and_london_ny_remain_server_clock_gated():
    """Only AMR moved to real UTC -- Monday Drift and London/NY scheduling
    must still be server-clock (`srv`) gated, unmodified by this fix."""
    src = inspect.getsource(ma.main)
    assert 'srv <= T_MONDAY_END' in src
    assert 'srv >= T_MONDAY_EXIT' in src
    assert 'srv >= T_LONDON_PREP' in src
    assert 'T_LONDON_START <= srv <= T_LONDON_END' in src
    assert 'srv >= T_NY_PREP' in src
    assert 'T_NY_START <= srv <= T_NY_END' in src
    assert ma.T_MONDAY_END == 2 * 60
    assert ma.T_MONDAY_EXIT == 21 * 60


def test_server_utc_offset_hours_untouched():
    """The fix must not touch server_utc_offset_hours() -- it is still
    required for ARB/monday_drift/London/NY server-time gating."""
    from src.agents import agent_strategy
    assert hasattr(agent_strategy, 'server_utc_offset_hours')


def test_amr_entry_window_covers_widest_configured_entry_end_hour():
    """T_AMR_END (06:00 UTC) must be >= every AMR pair's entry_end_hour so
    the orchestrator polls widely enough for the strategy's own per-pair
    check_signal() cutoff to be reachable."""
    import yaml
    pairs_dir = os.path.join(os.path.dirname(__file__), '..', 'pairs')
    max_entry_end_hour = 0
    for fname in os.listdir(pairs_dir):
        if 'asianrev' not in fname:
            continue
        with open(os.path.join(pairs_dir, fname), encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        if cfg.get('strategy') == 'asian_hours_reversion' and cfg.get('active'):
            max_entry_end_hour = max(max_entry_end_hour, cfg.get('entry_end_hour', 0))
    assert max_entry_end_hour > 0, 'expected at least one active asian_hours_reversion pair config'
    assert ma.T_AMR_END >= max_entry_end_hour * 60


TESTS = [
    test_amr_utc_constants_exist_and_are_correct,
    test_no_amr_orchestration_condition_uses_srv,
    test_monday_and_london_ny_remain_server_clock_gated,
    test_server_utc_offset_hours_untouched,
    test_amr_entry_window_covers_widest_configured_entry_end_hour,
]


def main():
    failed = 0
    for t in TESTS:
        try:
            t()
            print(f'PASS {t.__name__}')
        except AssertionError as e:
            failed += 1
            print(f'FAIL {t.__name__}: {e}')
    print(f'\n{len(TESTS) - failed}/{len(TESTS)} passed')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
