"""Task017 global entry mode tests; all broker APIs are deterministic fakes."""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from core import trading_mode
from src.agents import agent_execution as execution
from src.agents import main_agent


def write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.parametrize("configured,expected,allowed", [
    ("LIVE", "LIVE", True), ("paused", "PAUSED", False),
    ("Shadow", "SHADOW", False),
])
def test_exact_modes_are_normalized(tmp_path, configured, expected, allowed):
    status = trading_mode.resolve_trading_mode(
        write(tmp_path / "global.yaml", f"trading:\n  mode: {configured}\n"),
        tmp_path / "absent-local.yaml")
    assert (status.mode, status.entries_allowed) == (expected, allowed)


def test_missing_mode_fails_closed(tmp_path):
    status = trading_mode.resolve_trading_mode(
        write(tmp_path / "global.yaml", "global: {}\n"), tmp_path / "none")
    assert status.mode is None and not status.entries_allowed


def test_invalid_mode_fails_closed(tmp_path):
    status = trading_mode.resolve_trading_mode(
        write(tmp_path / "global.yaml", "trading:\n  mode: ENABLED\n"),
        tmp_path / "none")
    assert status.mode is None and not status.entries_allowed


@pytest.mark.parametrize("kind", ["malformed", "unreadable"])
def test_bad_configuration_fails_closed(tmp_path, kind):
    cfg = tmp_path / "global.yaml"
    if kind == "malformed":
        write(cfg, "trading: [\n")
    else:
        cfg.mkdir()
    status = trading_mode.resolve_trading_mode(cfg, tmp_path / "none")
    assert status.mode is None and not status.entries_allowed


def test_local_mode_is_authoritative_override(tmp_path):
    global_cfg = write(tmp_path / "global.yaml", "trading:\n  mode: PAUSED\n")
    local_cfg = write(tmp_path / "local.yaml", "trading:\n  mode: LIVE\n")
    assert trading_mode.resolve_trading_mode(global_cfg, local_cfg).entries_allowed


@pytest.mark.parametrize("mode,message", [
    ("PAUSED", "ENTRY BLOCKED AUDJPY@amr BUY -- trading mode PAUSED"),
    ("SHADOW", "SHADOW SIGNAL AUDJPY@amr BUY -- no live order sent"),
])
def test_orchestrator_gate_logs_and_blocks(monkeypatch, caplog, mode, message):
    monkeypatch.setattr(trading_mode, "get_trading_mode",
                        lambda: trading_mode.TradingModeStatus(mode, False, "test"))
    with caplog.at_level(logging.WARNING):
        assert not trading_mode.allow_or_log_entry(
            logging.getLogger("mode-test"), "AUDJPY@amr", "AUDJPY", "BUY")
    assert message in caplog.text


def test_live_orchestrator_gate_permits_entry(monkeypatch):
    monkeypatch.setattr(trading_mode, "get_trading_mode",
                        lambda: trading_mode.TradingModeStatus("LIVE", True, "test"))
    assert trading_mode.allow_or_log_entry(
        logging.getLogger("mode-test"), "AUDJPY@amr", "AUDJPY", "BUY")


@pytest.mark.parametrize("mode", ["PAUSED", "SHADOW", None])
def test_execution_side_bypass_gate_blocks_before_mt5(monkeypatch, mode):
    called = []
    monkeypatch.setattr(execution, "get_trading_mode",
                        lambda: trading_mode.TradingModeStatus(mode, False, "test"))
    monkeypatch.setattr(execution, "_connect_for_entry",
                        lambda _log: called.append("connect") or True)
    result = execution.place_trade(
        "AUDJPY", {"signal": "BUY"}, .01,
        {"sl_pips": 10, "tp_pips": 20, "use_live_anchor": True},
        "asian", 10, .5)
    assert not result["success"] and called == []


def close_fake():
    sent = []
    api = NS(
        ORDER_TYPE_BUY=0, ORDER_TYPE_SELL=1, TRADE_ACTION_DEAL=1,
        ORDER_TIME_GTC=0, ORDER_FILLING_FOK=0, ORDER_FILLING_IOC=1,
        ORDER_FILLING_RETURN=2, TRADE_RETCODE_DONE=10009,
        positions_get=lambda **_kw: (NS(type=0, volume=.1),),
        symbol_info_tick=lambda _s: NS(bid=100.0, ask=100.1),
        symbol_info=lambda _s: NS(filling_mode=1),
        order_send=lambda req: sent.append(req) or NS(retcode=10009),
    )
    return api, sent


@pytest.mark.parametrize("mode", ["PAUSED", "SHADOW"])
def test_non_live_modes_do_not_block_position_close(monkeypatch, mode):
    api, sent = close_fake()
    monkeypatch.setattr(execution, "mt5", api)
    monkeypatch.setattr(execution, "_connect", lambda _log: True)
    monkeypatch.setattr(execution, "get_trading_mode",
                        lambda: trading_mode.TradingModeStatus(mode, False, "test"))
    assert execution.close_trade(42, "AUDJPY")
    assert sent[0]["position"] == 42


def test_shadow_gate_creates_no_trade_or_accounting_records(monkeypatch):
    monkeypatch.setattr(trading_mode, "get_trading_mode",
                        lambda: trading_mode.TradingModeStatus("SHADOW", False, "test"))
    writes = []
    monkeypatch.setattr(main_agent, "AMR_KEYS", ["AUDJPY@amr"])
    monkeypatch.setattr(main_agent, "check_asian_reversion", lambda _key: {
        "signal": "BUY", "sl_pips": 10.0, "tp_pips": 20.0,
        "entry_price": 100.0, "reason": "test signal"})
    monkeypatch.setattr(main_agent, "run_risk",
                        lambda *_a, **_k: writes.append("risk"))
    monkeypatch.setattr(main_agent, "place_trade",
                        lambda *_a, **_k: writes.append("order"))
    monkeypatch.setattr(main_agent.tj, "log_signal",
                        lambda **_k: writes.append("journal"))
    state = {"asian_traded": {}, "open_trades": [], "pair_paused": {},
             "daily_pnl": 0.0}
    main_agent.step_check_asian_reversion(state, logging.getLogger("mode-test"))
    assert writes == []
    assert state["asian_traded"] == {} and state["open_trades"] == []


def test_final_gate_is_entry_only_and_adjacent_to_entry_path():
    source = Path(execution.__file__).read_text(encoding="utf-8")
    place = source[source.index("def place_trade"):source.index("def close_trade")]
    assert place.index("get_trading_mode()") < place.index("mt5.order_send(request)")
    close = source[source.index("def close_trade"):source.index("def position_is_open")]
    assert "get_trading_mode" not in close
