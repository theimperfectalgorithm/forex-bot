"""Task 015 broker-aware sizing tests. No live MT5 or production files."""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from src.agents import agent_execution as ex
from src.agents import agent_risk as risk


LOG = logging.getLogger("task015")


def fake_mt5(*, ask=1.1010, bid=1.1008, vmin=.01, step=.01, vmax=10.0,
             loss_per_price_lot=100000.0, calc_valid=True, positions=()):
    sent = []

    def calc(_typ, _symbol, volume, entry, stop):
        if not calc_valid:
            return float("nan")
        return -abs(entry - stop) * loss_per_price_lot * volume

    mt5 = SimpleNamespace(
        ORDER_TYPE_BUY=0, ORDER_TYPE_SELL=1, TRADE_ACTION_DEAL=1,
        ORDER_TIME_GTC=0, ORDER_FILLING_FOK=0, ORDER_FILLING_IOC=1,
        ORDER_FILLING_RETURN=2, TRADE_RETCODE_DONE=10009,
        symbol_info=lambda _s: SimpleNamespace(filling_mode=1, volume_min=vmin,
                                                volume_step=step, volume_max=vmax),
        symbol_info_tick=lambda _s: SimpleNamespace(ask=ask, bid=bid),
        order_calc_profit=calc,
        symbol_select=lambda _s, _enabled: True,
        positions_get=lambda **_kw: tuple(positions),
        order_send=lambda req: sent.append(req) or SimpleNamespace(
            retcode=10009, order=7, price=req["price"], comment="ok"),
        last_error=lambda: (0, "ok"),
    )
    return mt5, sent


def size(monkeypatch, signal, entry, stop, allowed=100.0, nominal=10.0,
         symbol="EURUSD", bot_max=2.0, **kwargs):
    mt5, _ = fake_mt5(**kwargs)
    monkeypatch.setattr(ex, "mt5", mt5)
    return ex._size_for_risk(symbol, signal, entry, stop, allowed, bot_max,
                             nominal, .01 if "JPY" in symbol else .0001, LOG)


def test_buy_sizes_executable_ask_to_sl(monkeypatch):
    volume, expected = size(monkeypatch, "BUY", 1.1010, 1.0960)
    assert volume == pytest.approx(.20) and expected == pytest.approx(100)


def test_sell_sizes_executable_bid_to_sl(monkeypatch):
    volume, expected = size(monkeypatch, "SELL", 1.1008, 1.1058)
    assert volume == pytest.approx(.20) and expected == pytest.approx(100)


def test_cadjpy_actual_distance_reduces_nominal_lot(monkeypatch):
    volume, _ = size(monkeypatch, "SELL", 114.711, 115.100, allowed=58.35,
                     nominal=27.2, symbol="CADJPY", loss_per_price_lot=15000)
    nominal_volume = 58.35 / (.272 * 15000)
    assert volume == pytest.approx(.01)
    assert volume < nominal_volume


def test_volume_normalization_floors():
    assert ex._floor_volume(.067, .01, .01, 2.0) == pytest.approx(.06)


def test_minimum_lot_over_budget_rejected(monkeypatch):
    result = size(monkeypatch, "BUY", 1.1, 1.09, allowed=9.99)
    assert result[0] is None and "minimum volume" in result[1]


def test_minimum_lot_exactly_in_budget_allowed(monkeypatch):
    volume, loss = size(monkeypatch, "BUY", 1.1, 1.09, allowed=10.0)
    assert volume == pytest.approx(.01) and loss == pytest.approx(10.0)


def test_broker_max_respected(monkeypatch):
    volume, _ = size(monkeypatch, "BUY", 1.1, 1.099, allowed=1000, vmax=.3)
    assert volume == pytest.approx(.3)


def test_bot_max_respected(monkeypatch):
    volume, _ = size(monkeypatch, "BUY", 1.1, 1.099, allowed=1000, bot_max=.2)
    assert volume == pytest.approx(.2)


@pytest.mark.parametrize("signal,entry,stop", [
    ("BUY", 1.1, 1.1), ("BUY", 1.1, 1.2),
    ("SELL", 1.1, 1.1), ("SELL", 1.1, 1.0),
])
def test_invalid_stop_orientation_rejected(monkeypatch, signal, entry, stop):
    assert size(monkeypatch, signal, entry, stop)[0] is None


def test_missing_metadata_fails_closed(monkeypatch):
    mt5, _ = fake_mt5()
    mt5.symbol_info = lambda _s: None
    monkeypatch.setattr(ex, "mt5", mt5)
    assert ex._size_for_risk("EURUSD", "BUY", 1.1, 1.09, 100, 2, 10, .0001, LOG)[0] is None


def test_nonfinite_broker_loss_fails_closed(monkeypatch):
    assert size(monkeypatch, "BUY", 1.1, 1.09, calc_valid=False)[0] is None


def _place(monkeypatch, signal="BUY", *, ask=1.1010, bid=1.1008,
           ticks=None, allowed=100.0, confirmed_fill=None):
    mt5, sent = fake_mt5(ask=ask, bid=bid)
    if ticks is not None:
        sequence = iter(ticks)
        mt5.symbol_info_tick = lambda _s: next(sequence)
    if confirmed_fill is not None:
        pos = SimpleNamespace(price_open=confirmed_fill)
        mt5.positions_get = lambda **kw: (pos,) if "ticket" in kw else ()
    monkeypatch.setattr(ex, "mt5", mt5)
    monkeypatch.setattr(ex, "_connect_for_entry", lambda _log: True)
    monkeypatch.setattr(ex, "_write_trade_log", lambda _row: None)
    monkeypatch.setattr(ex, "evaluate_prop_risk",
                        lambda risk, log=None: SimpleNamespace(allowed=True, reason="test"))
    monkeypatch.setattr(ex, "_confirm_fill_price",
                        lambda _ticket, fallback, _log: confirmed_fill or fallback)
    session = {"sl_pips": 50, "tp_pips": 100, "use_live_anchor": True,
               "strategy": "AMR"}
    return ex.place_trade("EURUSD", {"signal": signal}, 9.99, session,
                          "asian", allowed, 2.0), sent


def test_missing_tick_fails_closed(monkeypatch):
    mt5, sent = fake_mt5()
    mt5.symbol_info_tick = lambda _s: None
    monkeypatch.setattr(ex, "mt5", mt5)
    monkeypatch.setattr(ex, "_connect_for_entry", lambda _log: True)
    result = ex.place_trade("EURUSD", {"signal": "BUY"}, 1,
                            {"sl_pips": 50, "tp_pips": 100,
                             "use_live_anchor": True}, "ny", 100, 2)
    assert not result["success"] and not sent


def test_buy_uses_ask(monkeypatch):
    result, sent = _place(monkeypatch, "BUY")
    assert result["success"] and sent[0]["price"] == pytest.approx(1.1010)


def test_sell_uses_bid(monkeypatch):
    result, sent = _place(monkeypatch, "SELL")
    assert result["success"] and sent[0]["price"] == pytest.approx(1.1008)


def test_final_risk_over_budget_does_not_send(monkeypatch):
    tick1 = SimpleNamespace(ask=1.1010, bid=1.1008)
    tick2 = SimpleNamespace(ask=1.1020, bid=1.1018)
    result, sent = _place(monkeypatch, "BUY", ticks=[tick1, tick2], allowed=100)
    assert not result["success"] and sent == []
    assert "FINAL RISK REJECTED" in result["error"]


def test_final_safe_risk_allows_order(monkeypatch):
    result, sent = _place(monkeypatch, "BUY")
    assert result["success"] and len(sent) == 1
    assert result["volume"] != pytest.approx(9.99)  # legacy nominal lot ignored


def test_post_fill_diagnostic_uses_confirmed_fill(monkeypatch):
    result, _ = _place(monkeypatch, "BUY", confirmed_fill=1.1020)
    expected = ex._expected_loss("EURUSD", "BUY", result["volume"],
                                 1.1020, result["sl"])
    assert result["actual_risk"] == pytest.approx(expected)


def test_effective_risk_budget_represents_scale(monkeypatch):
    monkeypatch.setattr(risk, "RISK_SCALE", .5)
    allowed, base, effective = risk._risk_budget(4736, "CADJPY", .0025)
    assert base == pytest.approx(.0025)
    assert effective == pytest.approx(.00125)
    assert allowed == pytest.approx(5.92)


def test_strategy_sources_and_parameters_unchanged():
    # The hardening is confined to risk/execution/orchestration; signal and
    # SL/TP implementations/configuration remain untouched.
    import subprocess
    changed = subprocess.run(
        ["git", "diff", "--name-only", "--", "strategies", "pairs", "config"],
        capture_output=True, text=True, check=True).stdout.strip()
    assert changed in ("", "config/global_config.yaml")
