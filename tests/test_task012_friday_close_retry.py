"""Task 012 Friday-close retry regression tests. No live MT5 calls."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path

from src.agents import agent_execution as execution
from src.agents import main_agent as ma


LOG = logging.getLogger("task012-tests")
ROOT = Path(__file__).parent.parent


def _trade(ticket: int, symbol: str = "GBPUSD", key: str = "GBPUSD") -> dict:
    return {
        "ticket": ticket, "symbol": symbol, "strategy_key": key,
        "direction": "BUY", "session": "London", "lots": 0.1,
        "entry_price": 1.2, "sl": 1.19, "tp": 1.22,
        "breakeven_moved": False,
    }


def _state(*trades: dict) -> dict:
    return {
        "open_trades": list(trades), "closed_today": [], "daily_pnl": 0.0,
        "consec_losses": {t["symbol"]: 0 for t in trades},
        "pair_paused": {t["symbol"]: False for t in trades},
        "eurusd": {}, "friday_close_done": False,
        "friday_closed_tickets": [],
    }


def _broker(monkeypatch, answers: dict[int, list[bool | None]]):
    calls = []

    def verify(ticket, log):
        calls.append(ticket)
        values = answers[ticket]
        return values.pop(0) if len(values) > 1 else values[0]

    monkeypatch.setattr(ma, "position_is_open", verify)
    return calls


def test_no_open_positions_is_done(monkeypatch):
    state = _state()
    monkeypatch.setattr(ma, "close_trade", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("close must not be called")))
    ma.step_friday_close(state, LOG)
    assert state["friday_close_done"] is True


def test_success_is_done_only_after_broker_confirms_absent(monkeypatch):
    state = _state(_trade(1))
    _broker(monkeypatch, {1: [True, False]})
    monkeypatch.setattr(ma, "close_trade", lambda *a, **k: True)
    ma.step_friday_close(state, LOG)
    assert state["friday_close_done"] is True
    assert state["friday_closed_tickets"] == [1]


def test_close_false_remains_retryable(monkeypatch):
    state = _state(_trade(1))
    _broker(monkeypatch, {1: [True, True]})
    monkeypatch.setattr(ma, "close_trade", lambda *a, **k: False)
    ma.step_friday_close(state, LOG)
    assert state["friday_close_done"] is False
    assert state["open_trades"] == [_trade(1)]
    assert state["friday_closed_tickets"] == []


def test_close_exception_does_not_complete(monkeypatch):
    state = _state(_trade(1))
    _broker(monkeypatch, {1: [True]})
    monkeypatch.setattr(ma, "close_trade", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("send failed")))
    ma.step_friday_close(state, LOG)
    assert state["friday_close_done"] is False
    assert state["open_trades"] == [_trade(1)]


def test_one_ticket_exception_does_not_block_other_tickets(monkeypatch):
    state = _state(_trade(1), _trade(2, "USDJPY"))
    _broker(monkeypatch, {1: [True], 2: [True, False]})
    calls = []

    def close(ticket, symbol, comment):
        calls.append(ticket)
        if ticket == 1:
            raise RuntimeError("first ticket failed")
        return True

    monkeypatch.setattr(ma, "close_trade", close)
    ma.step_friday_close(state, LOG)
    assert calls == [1, 2]
    assert state["friday_close_done"] is False
    assert state["friday_closed_tickets"] == [2]


def test_partial_success_retries_only_still_open_ticket(monkeypatch):
    state = _state(_trade(1), _trade(2, "USDJPY"))
    verify_calls = _broker(monkeypatch, {
        1: [True, False, False],
        2: [True, True, True, False],
    })
    close_calls = []

    def close(ticket, symbol, comment):
        close_calls.append(ticket)
        return ticket == 1 or close_calls.count(2) == 2

    monkeypatch.setattr(ma, "close_trade", close)
    ma.step_friday_close(state, LOG)
    assert state["friday_close_done"] is False
    assert close_calls == [1, 2]
    assert state["friday_closed_tickets"] == [1]
    assert state["open_trades"] == [_trade(1), _trade(2, "USDJPY")]

    ma.step_friday_close(state, LOG)
    assert state["friday_close_done"] is True
    assert close_calls == [1, 2, 2]
    assert verify_calls.count(1) == 3
    assert state["friday_closed_tickets"] == [1, 2]


def test_broker_verification_failure_fails_safe_without_close(monkeypatch):
    state = _state(_trade(1))
    _broker(monkeypatch, {1: [None]})
    close_calls = []
    monkeypatch.setattr(ma, "close_trade", lambda *a, **k: close_calls.append(a))
    ma.step_friday_close(state, LOG)
    assert state["friday_close_done"] is False
    assert close_calls == []


def test_position_verification_distinguishes_connection_and_query_failures(monkeypatch):
    monkeypatch.setattr(execution, "_connect", lambda log: False)
    assert execution.position_is_open(1, LOG) is None

    monkeypatch.setattr(execution, "_connect", lambda log: True)
    monkeypatch.setattr(execution.mt5, "positions_get", lambda **kwargs: None)
    assert execution.position_is_open(1, LOG) is None

    def raises(**kwargs):
        raise RuntimeError("terminal query failed")

    monkeypatch.setattr(execution.mt5, "positions_get", raises)
    assert execution.position_is_open(1, LOG) is None


def test_position_verification_distinguishes_open_from_confirmed_absent(monkeypatch):
    monkeypatch.setattr(execution, "_connect", lambda log: True)
    monkeypatch.setattr(execution.mt5, "positions_get", lambda **kwargs: [])
    assert execution.position_is_open(1, LOG) is False
    monkeypatch.setattr(execution.mt5, "positions_get", lambda **kwargs: [object()])
    assert execution.position_is_open(1, LOG) is True


def test_accepted_but_still_open_retries_then_skips_once_closed(monkeypatch):
    state = _state(_trade(1))
    _broker(monkeypatch, {1: [True, True, False]})
    close_calls = []
    monkeypatch.setattr(ma, "close_trade",
                        lambda ticket, *a, **k: close_calls.append(ticket) or True)
    ma.step_friday_close(state, LOG)
    assert state["friday_close_done"] is False
    ma.step_friday_close(state, LOG)
    assert state["friday_close_done"] is True
    assert close_calls == [1]
    assert state["friday_closed_tickets"] == [1]


def test_monitor_accounts_successful_friday_close_once(monkeypatch):
    trade = _trade(1)
    state = _state(trade)
    state["friday_closed_tickets"] = [1]
    closed = {**trade, "exit_price": 1.21, "exit_time": "2026-08-21T20:00:00+00:00",
              "exit_reason": "FRIDAY_CLOSE", "exit_pnl": 10.0}
    seen = []

    def monitor(open_trades, log, friday_tickets):
        seen.append(friday_tickets)
        return [], [closed]

    monkeypatch.setattr(ma, "_check_untracked_positions", lambda *a, **k: None)
    monkeypatch.setattr(ma, "monitor_positions", monitor)
    monkeypatch.setattr(ma.tj, "log_exit", lambda item: seen.append(item["ticket"]))
    ma.step_monitor_positions(state, LOG)
    ma.step_monitor_positions(state, LOG)
    assert seen == [{1}, 1]
    assert state["open_trades"] == []
    assert state["closed_today"] == [closed]
    assert state["daily_pnl"] == 10.0


def test_amr_monday_exit_code_and_friday_timing_are_unchanged():
    source = (ROOT / "src" / "agents" / "main_agent.py").read_text(encoding="utf-8")
    assert "step_asian_time_exit(state, log)" in source
    assert "done_flag='monday_exit_done'" in source
    assert ma.T_AMR_EXIT == 7 * 60
    assert ma.T_FRIDAY_CLOSE == 20 * 60
    assert ma.is_friday_close_time(datetime(2026, 8, 21, 19, 59,
                                             tzinfo=timezone.utc)) is False
    assert ma.is_friday_close_time(datetime(2026, 8, 21, 20, 0,
                                             tzinfo=timezone.utc)) is True
