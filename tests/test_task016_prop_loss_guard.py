from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace as NS
import math
import uuid

import pytest

from core.prop_loss_guard import (PropRules, evaluate_prop_risk,
                                  make_daily_snapshot, prop_day_start_utc,
                                  rules_from_config, write_daily_snapshot)
from core.runtime_paths import data_dir
from src.agents import agent_execution as execution


ROOT = Path(__file__).parents[1]
NOW = datetime(2026, 8, 31, 10, 30, tzinfo=timezone.utc)
RULES = PropRules(5000, .10, .05, "higher_of_day_start_balance_or_equity",
                  "mt5_server_midnight")


class FakeMT5:
    DEAL_TYPE_BUY = 0
    DEAL_TYPE_SELL = 1
    DEAL_TYPE_BALANCE = 2
    DEAL_TYPE_CREDIT = 3
    DEAL_TYPE_COMMISSION = 7
    DEAL_TYPE_CHARGE = 15
    DEAL_TYPE_CORRECTION = 17
    DEAL_TYPE_BONUS = 18
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1

    def __init__(self, balance=4800, equity=None, deals=(), positions=()):
        self.account = NS(login=26520700, server="FivePercentOnline-Real",
                          balance=balance, equity=balance if equity is None else equity)
        self.deals = deals
        self.positions = positions
        self.history_args = None
        self.calc = lambda typ, symbol, volume, price, sl: -100 * volume

    def account_info(self): return self.account
    def history_deals_get(self, start, end):
        self.history_args = (start, end)
        return self.deals
    def positions_get(self): return self.positions
    def symbol_info(self, symbol): return NS(name=symbol)
    def symbol_info_tick(self, symbol): return NS(bid=1.1, ask=1.1002)
    def order_calc_profit(self, *args): return self.calc(*args)


def deal(kind=0, profit=0, commission=0, swap=0, fee=0):
    return NS(type=kind, profit=profit, commission=commission, swap=swap, fee=fee)


def position(ticket=1, sl=1.09, volume=1, kind=0, symbol="EURUSD"):
    return NS(ticket=ticket, sl=sl, volume=volume, type=kind, symbol=symbol)


def check(api=None, candidate=10, identity=True, rules=RULES, offset=3):
    api = api or FakeMT5()
    snapshot_path = data_dir() / "state" / f"prop-{uuid.uuid4()}.json"
    if (api.account is not None
            and math.isfinite(float(api.account.balance))
            and math.isfinite(float(api.account.equity))):
        impacts = sum(sum(float(getattr(d, f, 0) or 0)
                          for f in ("profit", "commission", "swap", "fee"))
                      for d in (api.deals or ()))
        opening = float(api.account.balance) - impacts
        snap = make_daily_snapshot(
            account=api.account,
            day_start=prop_day_start_utc(NOW, offset), offset=offset,
            balance=opening, equity=opening,
            source="broker_reconstruction_no_midnight_positions",
            captured_at=prop_day_start_utc(NOW, offset))
        write_daily_snapshot(snap, path=snapshot_path)
    return evaluate_prop_risk(candidate, rules=rules, api=api, now_utc=NOW,
                              identity_validator=lambda log: identity,
                              offset_provider=lambda unused: offset,
                              snapshot_path=snapshot_path)


def test_healthy_expected_account_allows():
    assert check().allowed


def test_wrong_account_blocks():
    assert not check(identity=False).allowed


def test_account_info_unavailable_blocks():
    api = FakeMT5(); api.account = None
    assert not check(api).allowed


def test_history_failure_blocks_but_empty_success_allows():
    assert not check(FakeMT5(deals=None)).allowed
    assert check(FakeMT5(deals=())).allowed


@pytest.mark.parametrize("equity", [4500, 4499])
def test_overall_equity_at_or_below_floor_blocks(equity):
    assert not check(FakeMT5(balance=4800, equity=equity), candidate=0).allowed


def test_floating_loss_can_breach_overall_floor():
    result = check(FakeMT5(balance=4800, equity=4490), candidate=0)
    assert not result.allowed and result.values["floating_pnl"] == -310


def test_candidate_crosses_overall_floor():
    assert not check(FakeMT5(balance=4600), candidate=100).allowed


def test_candidate_fits_overall_headroom():
    assert check(FakeMT5(balance=4600), candidate=99).allowed


@pytest.mark.parametrize("loss,allowed", [(200, True), (250, False), (251, False)])
def test_daily_realized_loss_boundary(loss, allowed):
    api = FakeMT5(balance=4800-loss, deals=(deal(profit=-loss),))
    assert check(api, candidate=0).allowed is allowed


def test_floating_loss_consumes_daily_headroom():
    assert not check(FakeMT5(balance=4800, equity=4550), candidate=0).allowed


def test_restart_reconstructs_identical_baseline():
    api1 = FakeMT5(balance=4700, deals=(deal(profit=-100),))
    api2 = FakeMT5(balance=4700, deals=(deal(profit=-100),))
    assert check(api1).values["daily_baseline"] == check(api2).values["daily_baseline"] == 4800


def test_losses_before_process_start_are_included():
    result = check(FakeMT5(balance=4680, deals=(deal(profit=-120),)), candidate=0)
    assert result.values["daily_used"] == 120


def test_manual_untracked_deal_is_included_without_magic_filter():
    d = deal(profit=-80); d.magic = 999
    assert check(FakeMT5(balance=4720, deals=(d,)), candidate=0).values["daily_used"] == 80


@pytest.mark.parametrize("field,value", [("commission", -7), ("swap", -3), ("fee", -2)])
def test_trade_cost_components_are_included(field, value):
    kwargs = {field: value}
    result = check(FakeMT5(balance=4800+value, deals=(deal(**kwargs),)), candidate=0)
    assert result.values["daily_used"] == -value


def test_deposit_does_not_expand_capacity():
    api = FakeMT5(balance=4900, deals=(deal(2, profit=100),))
    result = check(api, candidate=250)
    assert not result.allowed and result.values["adjusted_equity"] == 4800


def test_withdrawal_consumes_capacity_conservatively():
    api = FakeMT5(balance=4700, deals=(deal(2, profit=-100),))
    assert check(api, candidate=0).values["daily_used"] == 100


def test_unknown_deal_type_blocks():
    assert not check(FakeMT5(deals=(deal(999),))).allowed


@pytest.mark.parametrize("offset,expected_hour", [(2, 22), (3, 21)])
def test_mt5_server_midnight_boundaries(offset, expected_hour):
    assert prop_day_start_utc(NOW, offset).hour == expected_hour


def test_server_offset_transition_changes_boundary_not_local_timezone():
    winter = prop_day_start_utc(datetime(2026, 1, 15, 23, tzinfo=timezone.utc), 2)
    summer = prop_day_start_utc(datetime(2026, 8, 15, 23, tzinfo=timezone.utc), 3)
    assert (winter.hour, summer.hour) == (22, 21)


def test_history_receives_aware_complete_prop_day_window():
    api = FakeMT5(); check(api)
    assert api.history_args == (datetime(2026, 8, 30, 21, tzinfo=timezone.utc), NOW)


def test_positions_get_failure_blocks():
    assert not check(FakeMT5(positions=None)).allowed


def test_valid_open_position_sl_risk_reserved():
    result = check(FakeMT5(positions=(position(volume=1),)), candidate=10)
    assert result.allowed and result.values["open_reserved_risk"] == 100


def test_open_position_without_sl_blocks():
    assert not check(FakeMT5(positions=(position(sl=0),))).allowed


@pytest.mark.parametrize("failure", ["symbol", "tick", "calc"])
def test_position_loss_broker_failures_block(failure):
    api = FakeMT5(positions=(position(),))
    if failure == "symbol": api.symbol_info = lambda symbol: None
    if failure == "tick": api.symbol_info_tick = lambda symbol: None
    if failure == "calc": api.calc = lambda *args: None
    assert not check(api).allowed


def test_multiple_positions_aggregate_reservation():
    api = FakeMT5(positions=(position(1), position(2)))
    result = check(api, candidate=49)
    assert result.allowed and result.values["open_reserved_risk"] == 200
    assert not check(api, candidate=50).allowed


def test_all_active_and_legacy_paths_use_common_risk_then_execution_guard():
    main = (ROOT / "src/agents/main_agent.py").read_text(encoding="utf-8")
    assert main.count("run_risk(") >= 3 and main.count("place_trade(") >= 3
    source = (ROOT / "src/agents/agent_execution.py").read_text(encoding="utf-8")
    assert source.index("evaluate_prop_risk(final_risk") < source.index("mt5.order_send(request)")
    # All configured symbols funnel through one of these three entry families;
    # each family invokes both shared functions rather than sending directly.
    for family in ("step_check_breakouts", "step_check_asian_reversion",
                   "step_check_eurusd"):
        start = main.index(f"def {family}")
        body = main[start:main.find("\ndef step_", start + 5)]
        assert "run_risk(" in body and "place_trade(" in body


def test_guard_blocks_before_order_send(monkeypatch):
    source = (ROOT / "src/agents/agent_execution.py").read_text(encoding="utf-8")
    assert "if not prop.allowed" in source
    assert source.index("if not prop.allowed") < source.index("result = mt5.order_send(request)")


def test_guard_does_not_wrap_monitoring_or_exit_order_sends():
    source = (ROOT / "src/agents/agent_execution.py").read_text(encoding="utf-8")
    monitor = source[source.index("def monitor_positions"):]
    assert "evaluate_prop_risk" not in monitor


@pytest.mark.parametrize("field,value", [("balance", math.nan), ("equity", math.inf)])
def test_nonfinite_account_values_block(field, value):
    api = FakeMT5(); setattr(api.account, field, value)
    assert not check(api).allowed


def test_ambiguous_repository_daily_rule_requires_configuration():
    rules, error = rules_from_config({"starting_balance": 5000,
                                      "hard_floor_pct": .10,
                                      "daily_loss_pct": .05,
                                      "prop_day_boundary_model":
                                      "mt5_server_midnight"})
    assert rules is None and "CONFIGURATION REQUIRED" in error


@pytest.mark.parametrize("selector", [
    "broker_day_start_balance", "previous_day_closing_equity",
    "min_previous_close_balance_equity", "max_previous_close_balance_equity",
])
def test_no_unproven_balance_equity_selector_can_enable_live_guard(selector):
    rules, error = rules_from_config({
        "starting_balance": 5000, "hard_floor_pct": .10,
        "daily_loss_pct": .05, "prop_day_boundary_model":
        "mt5_server_midnight", "prop_daily_reference_model": selector,
    })
    assert rules is None and "CONFIGURATION REQUIRED" in error


@pytest.mark.parametrize("balance,equity,candidates", [
    (5000, 5000, {5000}),
    (5000, 4900, {5000, 4900}),
    (5000, 5100, {5000, 5100}),
])
def test_overnight_positions_expose_unresolved_reference(balance, equity,
                                                          candidates):
    # Balance-only/equity-only/min/max collapse when equal, but produce the
    # shown distinct candidates with floating loss/profit. Official wording
    # supplied to this task does not say which candidate is authoritative.
    assert {balance, equity, min(balance, equity), max(balance, equity)} == candidates


def test_production_guard_has_no_local_accounting_dependencies():
    source = (ROOT / "core/prop_loss_guard.py").read_text(encoding="utf-8")
    for forbidden in ("trades_log.csv", "events.jsonl", "daily_state.json",
                      "equity_curve.csv", "trade_costs.jsonl"):
        assert forbidden not in source


def test_tests_inject_fake_api_and_never_initialize_terminal():
    source = Path(__file__).read_text(encoding="utf-8")
    forbidden = "initial" + "ize("
    executable = "terminal64" + ".exe"
    assert forbidden not in source and executable not in source
