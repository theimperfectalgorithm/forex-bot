"""Task017B: detected signals are consumed only after broker success."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
from types import SimpleNamespace as NS

import numpy as np
import pytest

from core import trading_mode
from strategies import asian_hours_reversion as amr_mod
from strategies.asian_range_breakout import AsianRangeBreakout
from strategies.monday_drift import MondayDrift
from strategies import sma_ema_combined as eurusd_mod
from strategies.sma_ema_combined import SmaEmaCombined
from src.agents import main_agent


LOG = logging.getLogger("task017b")
DAY = datetime(2026, 8, 24, 1, 0, tzinfo=timezone.utc)  # Monday


def amr_strategy(monkeypatch):
    strategy = amr_mod.AsianHoursReversion({
        "pair": "AUDJPY", "strategy": "asian_hours_reversion",
        "active": True, "timeframe": "M15", "risk_percent": .25,
        "z_threshold": 2.0, "sl_multiplier": 1.5,
        "entry_end_hour": 4, "session": "asian", "friday_close": True,
    })
    closes = [100.0] * 19 + [99.0]
    values = [closes[0]] + closes
    start = DAY - timedelta(minutes=15 * (len(values) - 1))
    rates = np.array([
        (int((start + timedelta(minutes=15*i) + timedelta(hours=3)).timestamp()), v)
        for i, v in enumerate(values)
    ], dtype=[("time", "i8"), ("close", "f8")])
    monkeypatch.setattr(strategy, "_connect", lambda _log: True)
    monkeypatch.setattr(strategy, "_m15_bars", lambda: rates)
    monkeypatch.setattr(amr_mod, "observed_server_utc_offset_hours",
                        lambda _api, _symbol: 3)
    return strategy


def arb_strategy(monkeypatch):
    strategy = AsianRangeBreakout({
        "pair": "CADJPY", "strategy": "asian_range_breakout",
        "active": True, "timeframe": "H1", "risk_percent": .5,
        "h4_filter": False, "session": "asian", "friday_close": True,
    })
    bar = {"time": datetime(2026, 8, 24, 8, tzinfo=timezone.utc).timestamp(),
           "close": 101.0}
    monkeypatch.setattr(strategy, "_connect", lambda _log: True)
    monkeypatch.setattr(strategy, "_last_closed_h1", lambda: bar)
    return strategy, {"asian_high": 100.0, "asian_low": 99.0,
                      "h4_trend": 0, "h4_filter": False}


def monday_strategy(monkeypatch):
    strategy = MondayDrift({
        "pair": "GBPUSD", "strategy": "monday_drift", "active": True,
        "timeframe": "H1", "risk_percent": .25, "sl_atr_mult": 1.25,
        "tp_atr_mult": 1.0, "session": "monday", "friday_close": True,
    })
    rates = np.array([(int(DAY.replace(hour=0).timestamp()), 1.2)],
                     dtype=[("time", "i8"), ("close", "f8")])
    fake = NS(TIMEFRAME_H1=1, copy_rates_from_pos=lambda *_a: rates)
    monkeypatch.setattr("strategies.monday_drift.mt5", fake)
    monkeypatch.setattr(strategy, "_connect", lambda _log: True)
    monkeypatch.setattr(strategy, "_atr20d_pips", lambda: 80.0)
    return strategy


@pytest.mark.parametrize("mode", ["PAUSED", "SHADOW"])
def test_amr_non_live_detection_is_not_consumed(monkeypatch, mode):
    strategy = amr_strategy(monkeypatch)
    monkeypatch.setattr(trading_mode, "get_trading_mode", lambda:
                        trading_mode.TradingModeStatus(mode, False, "test"))
    first = strategy.check_signal({"armed": True}, "asian")
    assert not trading_mode.allow_or_log_entry(LOG, "AUDJPY@amr", "AUDJPY",
                                                first["signal"])
    assert strategy._last_trade_date is None
    assert strategy.check_signal({"armed": True}, "asian")["signal"] == "BUY"


def test_amr_success_acknowledges_but_risk_rejection_does_not(monkeypatch):
    strategy = amr_strategy(monkeypatch)
    signal = strategy.check_signal({"armed": True}, "asian")
    # A risk rejection performs no acknowledgement.
    assert strategy.check_signal({"armed": True}, "asian")["signal"] == "BUY"
    strategy.acknowledge_trade(signal)
    assert "already traded" in strategy.check_signal(
        {"armed": True}, "asian")["reason"]


@pytest.mark.parametrize("mode", ["PAUSED", "SHADOW"])
def test_arb_non_live_detection_is_not_consumed(monkeypatch, mode):
    strategy, session = arb_strategy(monkeypatch)
    monkeypatch.setattr(trading_mode, "get_trading_mode", lambda:
                        trading_mode.TradingModeStatus(mode, False, "test"))
    first = strategy.check_breakout(session, "london")
    assert not trading_mode.allow_or_log_entry(LOG, "CADJPY@arb", "CADJPY",
                                                first["signal"])
    assert first["signal"] == "BUY" and strategy._last_trade_date is None
    assert strategy.check_breakout(session, "london")["signal"] == "BUY"


def test_arb_success_consumes_only_after_ack(monkeypatch):
    strategy, session = arb_strategy(monkeypatch)
    signal = strategy.check_breakout(session, "london")
    strategy.acknowledge_trade(signal)
    assert "already traded" in strategy.check_breakout(session, "london")["reason"]


@pytest.mark.parametrize("outcome", ["PAUSED", "SHADOW", "RISK_REJECTED"])
def test_monday_block_or_rejection_does_not_consume(monkeypatch, outcome):
    strategy = monday_strategy(monkeypatch)
    first = strategy.check_signal({"armed": True}, "asian")
    if outcome in ("PAUSED", "SHADOW"):
        monkeypatch.setattr(trading_mode, "get_trading_mode", lambda:
                            trading_mode.TradingModeStatus(outcome, False, "test"))
        assert not trading_mode.allow_or_log_entry(
            LOG, "GBPUSD@mon", "GBPUSD", first["signal"])
    assert first["signal"] == "BUY" and strategy._last_trade_date is None
    assert strategy.check_signal({"armed": True}, "asian")["signal"] == "BUY"


def test_monday_success_ack_suppresses_same_monday(monkeypatch):
    strategy = monday_strategy(monkeypatch)
    signal = strategy.check_signal({"armed": True}, "asian")
    strategy.acknowledge_trade(signal)
    assert strategy.check_signal({"armed": True}, "asian")["reason"] == "already traded this Monday"


def test_ema_confirmation_retained_until_success_ack(monkeypatch):
    strategy = object.__new__(SmaEmaCombined)
    strategy.pair = "EURUSD"
    monkeypatch.setattr(strategy, "_connect", lambda _log: True)
    bars = np.array([(i, 3.0) for i in range(201)],
                    dtype=[("time", "i8"), ("close", "f8")])
    monkeypatch.setattr(strategy, "_m15_bars", lambda: bars)
    monkeypatch.setattr(strategy, "_h1_ema50_trend", lambda _log: 1)
    monkeypatch.setattr(eurusd_mod, "london_ny_overlap", lambda _now: True)
    monkeypatch.setattr(strategy, "_ewm_ema", lambda closes, period:
                        np.full(len(closes), {5: 3.0, 20: 2.0, 50: 1.0}[period]))
    state = {"ema_pullback_pending": True, "ema_pullback_dir": "BUY"}
    signals, blocked = strategy.check_signals(state, [])
    signal = next(s for s in signals if s.get("strategy") == "EMA")
    assert blocked["ema_pullback_pending"] is True
    assert blocked["ema_pullback_dir"] == "BUY"
    acknowledged = strategy.acknowledge_trade(signal, blocked)
    assert acknowledged["ema_pullback_pending"] is False
    assert acknowledged["ema_pullback_dir"] == ""
    assert state["ema_pullback_pending"] is True


def base_state():
    return {"asian_traded": {}, "open_trades": [], "pair_paused": {},
            "daily_pnl": 0.0}


@pytest.mark.parametrize("success", [False, True])
def test_orchestrator_ack_boundary(monkeypatch, success):
    calls = []
    monkeypatch.setattr(main_agent, "AMR_KEYS", ["AUDJPY@amr"])
    monkeypatch.setattr(main_agent, "check_asian_reversion", lambda _key: {
        "signal": "BUY", "sl_pips": 10.0, "tp_pips": 20.0,
        "entry_price": 100.0, "reason": "test",
        "signal_bar_time_utc": DAY.isoformat()})
    monkeypatch.setattr(main_agent, "allow_or_log_entry", lambda *_a: True)
    monkeypatch.setattr(main_agent, "run_risk", lambda *_a, **_k: {
        "decision": "APPROVED", "lot_size": .1,
        "allowed_risk_dollars": 10.0, "max_lot": .5})
    result = {"success": success, "error": "rejected"}
    if success:
        result.update(ticket=7, volume=.1, entry_price=100.0, sl=99.0, tp=102.0)
    monkeypatch.setattr(main_agent, "place_trade", lambda *_a, **_k: result)
    monkeypatch.setattr(main_agent, "acknowledge_trade",
                        lambda key, signal: calls.append((key, signal)))
    monkeypatch.setattr(main_agent.tj, "log_signal", lambda **_k: None)
    monkeypatch.setattr(main_agent.tj, "log_entry", lambda **_k: None)
    monkeypatch.setattr(main_agent.tj, "log_rejection", lambda **_k: None)
    main_agent.step_check_asian_reversion(base_state(), LOG)
    assert len(calls) == (1 if success else 0)


def test_logging_failure_success_still_acknowledges_and_tracks(monkeypatch):
    calls = []
    state = base_state()
    monkeypatch.setattr(main_agent, "AMR_KEYS", ["AUDJPY@amr"])
    monkeypatch.setattr(main_agent, "check_asian_reversion", lambda _key: {
        "signal": "BUY", "sl_pips": 10.0, "tp_pips": 20.0,
        "entry_price": 100.0, "reason": "test",
        "signal_bar_time_utc": DAY.isoformat()})
    monkeypatch.setattr(main_agent, "allow_or_log_entry", lambda *_a: True)
    monkeypatch.setattr(main_agent, "run_risk", lambda *_a, **_k: {
        "decision": "APPROVED", "lot_size": .1,
        "allowed_risk_dollars": 10.0, "max_lot": .5})
    # This is the contract returned by place_trade after its internal CSV
    # failure: broker success and ticket information remain authoritative.
    monkeypatch.setattr(main_agent, "place_trade", lambda *_a, **_k: {
        "success": True, "ticket": 77, "volume": .1,
        "entry_price": 100.0, "sl": 99.0, "tp": 102.0, "error": None})
    monkeypatch.setattr(main_agent, "acknowledge_trade",
                        lambda key, signal: calls.append((key, signal)))
    monkeypatch.setattr(main_agent.tj, "log_signal", lambda **_k: None)
    monkeypatch.setattr(main_agent.tj, "log_entry", lambda **_k: None)

    main_agent.step_check_asian_reversion(state, LOG)

    assert len(calls) == 1
    assert state["asian_traded"]["AUDJPY@amr"] is True
    assert [trade["ticket"] for trade in state["open_trades"]] == [77]
