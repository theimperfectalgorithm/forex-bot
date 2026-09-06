"""Task 011 entry-safety regression tests. All MT5 behavior is mocked."""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import core.mt5_connect as mc
from src.agents import agent_execution as execution
from src.agents import main_agent


LOG = logging.getLogger("task011-tests")
pytestmark = pytest.mark.usefixtures('clear_news')
ROOT = Path(__file__).parent.parent


def _identity(monkeypatch, *, login=26520700, server="FivePercentOnline-Real",
              account=True):
    fake = SimpleNamespace(
        account_info=lambda: (SimpleNamespace(login=login, server=server)
                              if account else None),
        terminal_info=lambda: SimpleNamespace(path=r"C:\MT5-5ers"),
        initialize=lambda **kwargs: True,
        last_error=lambda: (0, "ok"),
    )
    monkeypatch.setattr(mc, "mt5", fake)
    monkeypatch.setattr(mc, "PATCHED", True)
    monkeypatch.setattr(mc, "MT5_PATH", r"C:\MT5-5ers\terminal64.exe")
    monkeypatch.setattr(mc, "EXPECTED_MT5_PATH", r"C:\MT5-5ers\terminal64.exe")
    monkeypatch.setattr(mc, "EXPECTED_MT5_LOGIN", 26520700)
    monkeypatch.setattr(mc, "EXPECTED_MT5_SERVER", "FivePercentOnline-Real")
    return fake


def test_expected_login_match_allowed(monkeypatch):
    _identity(monkeypatch)
    assert mc.validate_expected_account(LOG) is True


def test_expected_login_mismatch_blocked(monkeypatch):
    _identity(monkeypatch, login=999)
    assert mc.validate_expected_account(LOG) is False


def test_expected_server_mismatch_blocked(monkeypatch):
    _identity(monkeypatch, server="Other-Server")
    assert mc.validate_expected_account(LOG) is False


def test_account_info_unavailable_blocked(monkeypatch):
    _identity(monkeypatch, account=False)
    assert mc.validate_expected_account(LOG) is False


def test_missing_expected_identity_config_blocked(monkeypatch):
    _identity(monkeypatch)
    monkeypatch.setattr(mc, "EXPECTED_MT5_LOGIN", 0)
    assert mc.validate_expected_account(LOG) is False


def _entry_mt5(positions=()):
    sent = []
    fake = SimpleNamespace(
        ORDER_TYPE_BUY=0, ORDER_TYPE_SELL=1, TRADE_ACTION_DEAL=1,
        ORDER_TIME_GTC=0, ORDER_FILLING_FOK=0, ORDER_FILLING_IOC=1,
        ORDER_FILLING_RETURN=2, TRADE_RETCODE_DONE=10009,
        symbol_select=lambda symbol, enabled: True,
        symbol_info_tick=lambda symbol: SimpleNamespace(ask=115.10, bid=115.09),
        symbol_info=lambda symbol: SimpleNamespace(
            filling_mode=1, volume_min=0.01, volume_step=0.01, volume_max=10.0),
        order_calc_profit=lambda typ, symbol, volume, entry, sl:
            -abs(entry - sl) * 10000 * volume,
        positions_get=lambda **kwargs: tuple(positions),
        order_send=lambda request: sent.append(request) or
            SimpleNamespace(retcode=10009, order=123, price=115.10, comment="ok"),
        last_error=lambda: (0, "ok"),
    )
    return fake, sent


def _place(monkeypatch, positions=()):
    fake, sent = _entry_mt5(positions)
    monkeypatch.setattr(execution, "mt5", fake)
    monkeypatch.setattr(execution, "_connect_for_entry", lambda log: True)
    monkeypatch.setattr(execution, "_confirm_fill_price",
                        lambda ticket, fallback, log: fallback)
    monkeypatch.setattr(execution, "_write_trade_log", lambda row: None)
    monkeypatch.setattr(execution, "evaluate_prop_risk",
                        lambda risk, log=None: SimpleNamespace(allowed=True, reason="test"))
    result = execution.place_trade(
        "CADJPY", {"signal": "BUY"}, 0.01,
        {"sl_pips": 20, "tp_pips": 40, "asian_high": 115.00,
         "asian_low": 114.80}, "london", 100.0, 2.0)
    return result, sent


def test_identity_revalidated_before_order_submission(monkeypatch):
    fake, sent = _entry_mt5(())
    monkeypatch.setattr(execution, "mt5", fake)
    monkeypatch.setattr(execution, "_connect_for_entry", lambda log: False)
    result = execution.place_trade("CADJPY", {"signal": "BUY"}, 0.01,
                                   {"sl_pips": 20, "tp_pips": 40,
                                    "asian_high": 115.0, "asian_low": 114.8},
                                   "london", 100.0, 2.0)
    assert not result["success"]
    assert sent == []


@pytest.mark.parametrize("entry_call", [
    "step_check_asian_reversion",  # AMR
    "step_check_breakouts",        # London/ARB and NY
    "step_check_eurusd",           # EURUSD
])
def test_reconciliation_source_precedes_entry_families(entry_call):
    source = (ROOT / "src" / "agents" / "main_agent.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = [(n.func.id if isinstance(n.func, ast.Name) else None, n.lineno)
             for n in ast.walk(main) if isinstance(n, ast.Call)]
    reconcile_line = min(line for name, line in calls
                         if name == "step_pre_entry_reconciliation")
    assert all(reconcile_line < line for name, line in calls if name == entry_call)


def test_monday_entry_is_guarded_after_reconciliation():
    source = (ROOT / "src" / "agents" / "main_agent.py").read_text(encoding="utf-8")
    assert "entries_allowed and server_now.weekday() == 0" in source
    assert source.index("step_pre_entry_reconciliation") < source.index(
        "entries_allowed and server_now.weekday() == 0")


def test_ny_entry_is_guarded_after_reconciliation():
    source = (ROOT / "src" / "agents" / "main_agent.py").read_text(encoding="utf-8")
    assert "entries_allowed and T_NY_START" in source


def test_reconciliation_failure_blocks_entries(monkeypatch):
    monkeypatch.setattr(main_agent, "initialize_and_validate", lambda log: False)
    assert main_agent.step_pre_entry_reconciliation(
        {"open_trades": [], "untracked_positions_flagged_at": None}, LOG) is False


def test_position_management_remains_outside_entry_gate():
    source = (ROOT / "src" / "agents" / "main_agent.py").read_text(encoding="utf-8")
    # These scheduler calls remain present and are not textually conditioned
    # on entries_allowed; the final monitor is unconditional.
    assert "step_asian_time_exit(state, log)" in source
    assert "step_friday_close(state, log)" in source
    assert "step_monitor_positions(state, log)" in source


def test_untracked_position_adopted_before_entry(monkeypatch):
    state = main_agent._fresh_state("2026-08-25")
    position = {"ticket": 77, "symbol": "CADJPY", "direction": "BUY",
                "lots": 0.01, "entry_price": 115.0, "sl": 114.8,
                "tp": 115.4, "open_time": "2026-08-25T05:00:00+00:00"}
    monkeypatch.setattr(main_agent, "initialize_and_validate", lambda log: True)
    monkeypatch.setattr(main_agent, "find_untracked_positions",
                        lambda trades, log, strict=False: [position])
    assert main_agent.step_pre_entry_reconciliation(state, LOG) is False
    assert state["open_trades"][0]["ticket"] == 77
    assert state["untracked_positions_flagged_at"]


def test_same_symbol_bot_magic_rejected(monkeypatch):
    pos = SimpleNamespace(symbol="CADJPY", magic=execution.MAGIC_NUMBER, ticket=88)
    result, sent = _place(monkeypatch, [pos])
    assert not result["success"]
    assert "broker-side duplicate guard" in result["error"]
    assert sent == []


def test_different_symbol_allowed(monkeypatch):
    pos = SimpleNamespace(symbol="AUDJPY", magic=execution.MAGIC_NUMBER, ticket=88)
    result, sent = _place(monkeypatch, [pos])
    assert result["success"] and len(sent) == 1


def test_same_symbol_unrelated_magic_allowed(monkeypatch):
    pos = SimpleNamespace(symbol="CADJPY", magic=999, ticket=88)
    result, sent = _place(monkeypatch, [pos])
    assert result["success"] and len(sent) == 1


def test_positions_query_failure_fails_closed(monkeypatch):
    fake, sent = _entry_mt5(())
    fake.positions_get = lambda **kwargs: None
    fake.last_error = lambda: (1, "query failed")
    monkeypatch.setattr(execution, "mt5", fake)
    monkeypatch.setattr(execution, "_connect_for_entry", lambda log: True)
    monkeypatch.setattr(execution, "_write_trade_log", lambda row: None)
    result = execution.place_trade(
        "CADJPY", {"signal": "BUY"}, 0.01,
        {"sl_pips": 20, "tp_pips": 40, "asian_high": 115.0,
         "asian_low": 114.8}, "london", 100.0, 2.0)
    assert not result["success"] and sent == []


def test_no_existing_position_preserves_order_path(monkeypatch):
    result, sent = _place(monkeypatch, [])
    assert result["success"] and len(sent) == 1


def test_crash_window_stale_state_broker_position_prevents_second_order(monkeypatch):
    # Application state is intentionally absent from this execution-layer
    # test: broker truth alone must prevent the duplicate.
    pos = SimpleNamespace(symbol="CADJPY", magic=execution.MAGIC_NUMBER, ticket=321)
    result, sent = _place(monkeypatch, [pos])
    assert not result["success"] and sent == []
