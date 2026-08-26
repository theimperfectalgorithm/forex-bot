"""Task 013D AMR UTC parity regressions. All MT5 behavior is fake."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging

import numpy as np
import pytest

from core.mt5_time import mt5_bar_time_to_utc
from strategies import asian_hours_reversion as amr
from src.agents import main_agent as ma


HISTORICAL_CASES = [
    ("CADJPY", 4, 2.0, 1.5, datetime(2026, 8, 25, 2, 45, tzinfo=timezone.utc),
     [114.918,114.951,114.935,114.933,114.918,114.929,114.970,114.974,114.943,114.946,
      114.944,114.974,114.928,114.924,114.944,114.968,114.979,114.978,115.009,115.047],
     "SELL", 114.9556, 0.03249680146203691, 2.8125844971778977, 9.1, 13.7),
    ("AUDJPY", 4, 2.0, 1.5, datetime(2026, 8, 26, 1, 0, tzinfo=timezone.utc),
     [114.082,114.063,114.070,114.039,114.021,113.985,114.009,114.049,114.027,114.049,
      114.064,114.080,114.094,114.100,114.060,114.113,114.083,114.030,113.998,113.969],
     "BUY", 114.04925, 0.039293597978616916, -2.0423174290040236, 8.0, 12.0),
    ("CADJPY", 4, 2.0, 1.5, datetime(2026, 8, 26, 1, 0, tzinfo=timezone.utc),
     [115.063,115.060,115.064,115.039,114.978,115.019,115.009,115.067,115.049,115.069,
      115.079,115.088,115.078,115.092,115.066,115.060,115.052,114.998,114.987,114.931],
     "BUY", 115.0424, 0.042261591371636424, -2.635963208777051, 11.1, 16.7),
    ("AUDJPY", 4, 2.0, 1.5, datetime(2026, 8, 25, 0, 0, tzinfo=timezone.utc),
     [113.774,113.760,113.764,113.776,113.765,113.753,113.746,113.655,113.705,113.708,
      113.689,113.738,113.762,113.761,113.757,113.723,113.749,113.809,113.838,113.872],
     "SELL", 113.7552, 0.04871463629622696, 2.397636703880275, 11.7, 17.5),
    ("EURJPY", 6, 2.0, 1.5, datetime(2026, 8, 25, 0, 45, tzinfo=timezone.utc),
     [185.612,185.623,185.587,185.569,185.492,185.488,185.523,185.500,185.579,185.596,
      185.590,185.586,185.560,185.593,185.642,185.672,185.697,185.692,185.685,185.747],
     "SELL", 185.60165, 0.07179743068266445, 2.0244457025549596, 14.5, 21.8),
    ("EURJPY", 6, 2.0, 1.5, datetime(2026, 8, 26, 0, 45, tzinfo=timezone.utc),
     [185.848,185.858,185.866,185.879,185.812,185.834,185.863,185.812,185.887,185.872,
      185.886,185.891,185.905,185.901,185.907,185.870,185.884,185.871,185.800,185.784],
     "BUY", 185.8615, 0.03582927353582754, -2.1630357624345065, 7.8, 11.6),
    ("GBPJPY", 4, 2.5, 1.25, datetime(2026, 8, 26, 0, 45, tzinfo=timezone.utc),
     [217.268,217.294,217.292,217.299,217.306,217.293,217.227,217.227,217.290,217.284,
      217.291,217.297,217.319,217.319,217.311,217.252,217.287,217.280,217.216,217.176],
     "BUY", 217.2764, 0.037921525618877155, -2.647572806249582, 10.0, 12.6),
]


def _strategy(pair, end_hour, z=2.0, sl_mult=1.5):
    return amr.AsianHoursReversion({
        "pair": pair, "strategy": "asian_hours_reversion", "active": True,
        "timeframe": "M15", "risk_percent": 0.25, "z_threshold": z,
        "sl_multiplier": sl_mult, "entry_end_hour": end_hour,
        "session": "asian", "friday_close": True,
    })


def _rates(closes, bar_utc, offset=3):
    # Production asks for 21 bars but computes over the final 20 closes.
    vals = [closes[0]] + list(closes)
    start = bar_utc - timedelta(minutes=15 * (len(vals) - 1))
    return np.array([
        (int((start + timedelta(minutes=15*i) + timedelta(hours=offset)).timestamp()), v)
        for i, v in enumerate(vals)
    ], dtype=[("time", "i8"), ("close", "f8")])


def _run(monkeypatch, strategy, rates, offset=3):
    monkeypatch.setattr(strategy, "_connect", lambda log: True)
    monkeypatch.setattr(strategy, "_m15_bars", lambda: rates)
    monkeypatch.setattr(amr, "observed_server_utc_offset_hours",
                        lambda mt5, symbol: offset)
    return strategy.check_signal({"armed": True}, "asian")


@pytest.mark.parametrize(
    "pair,end_hour,zthr,sl_mult,bar_utc,closes,direction,sma,std,z,tp,sl",
    HISTORICAL_CASES,
)
def test_aug23_26_signals_through_production_path(
        monkeypatch, pair, end_hour, zthr, sl_mult, bar_utc, closes,
        direction, sma, std, z, tp, sl):
    strategy = _strategy(pair, end_hour, zthr, sl_mult)
    result = _run(monkeypatch, strategy, _rates(closes, bar_utc))
    assert np.mean(closes) == pytest.approx(sma)
    assert np.std(closes, ddof=1) == pytest.approx(std)
    assert (closes[-1] - sma) / std == pytest.approx(z)
    assert result["signal"] == direction
    assert result["entry_price"] == pytest.approx(closes[-1])
    assert result["signal_bar_time_utc"] == bar_utc.isoformat()
    assert result["tp_pips"] == tp
    assert result["sl_pips"] == sl


@pytest.mark.parametrize("pair,end_hour,eligible,rejected", [
    ("AUDJPY", 4, 3, 4), ("CADJPY", 4, 3, 4),
    ("GBPJPY", 4, 3, 4), ("EURJPY", 6, 5, 6),
])
def test_entry_window_boundaries(monkeypatch, pair, end_hour, eligible, rejected):
    closes = [100.0] * 19 + [99.0]
    strategy = _strategy(pair, end_hour)
    for hour, minute in [(0, 0), (1, 0), (2, 45), (eligible, 45)]:
        strategy._last_trade_date = None
        bar = datetime(2026, 8, 25, hour, minute, tzinfo=timezone.utc)
        assert _run(monkeypatch, strategy, _rates(closes, bar))["signal"] == "BUY"
    strategy._last_trade_date = None
    bar = datetime(2026, 8, 25, rejected, 0, tzinfo=timezone.utc)
    result = _run(monkeypatch, strategy, _rates(closes, bar))
    assert result["signal"] == "NO_SIGNAL"
    assert "outside" in result["reason"]


def test_numeric_server_time_converted_once_and_aware_utc_not_shifted_again():
    utc = datetime(2026, 8, 26, 1, 0, tzinfo=timezone.utc)
    server_epoch = (utc + timedelta(hours=3)).timestamp()
    assert mt5_bar_time_to_utc(server_epoch, 3) == utc
    assert mt5_bar_time_to_utc(utc, 3) == utc
    with pytest.raises(ValueError, match="naive"):
        mt5_bar_time_to_utc(utc.replace(tzinfo=None), 3)


def test_offset_unavailable_fails_closed(monkeypatch):
    strategy = _strategy("AUDJPY", 4)
    rates = _rates([100.0] * 19 + [99.0],
                   datetime(2026, 8, 25, 1, 0, tzinfo=timezone.utc))
    monkeypatch.setattr(strategy, "_connect", lambda log: True)
    monkeypatch.setattr(strategy, "_m15_bars", lambda: rates)
    monkeypatch.setattr(amr, "observed_server_utc_offset_hours",
                        lambda mt5, symbol: None)
    result = strategy.check_signal({"armed": True}, "asian")
    assert result["signal"] == "NO_SIGNAL"
    assert "offset" in result["reason"]


def test_one_trade_dedup_uses_normalized_utc_date(monkeypatch):
    strategy = _strategy("AUDJPY", 4)
    closes = [100.0] * 19 + [99.0]
    first = _rates(closes, datetime(2026, 8, 25, 0, 0, tzinfo=timezone.utc))
    assert _run(monkeypatch, strategy, first)["signal"] == "BUY"
    later = _rates(closes, datetime(2026, 8, 25, 2, 45, tzinfo=timezone.utc))
    result = _run(monkeypatch, strategy, later)
    assert result["signal"] == "NO_SIGNAL"
    assert "already traded" in result["reason"]


def test_debug_observability_contains_utc_window_and_decision(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG, logger="task013d")
    logger = logging.getLogger("task013d")
    monkeypatch.setattr(amr, "_log", lambda: logger)
    strategy = _strategy("AUDJPY", 4)
    rates = _rates([100.0] * 19 + [99.0],
                   datetime(2026, 8, 25, 2, 45, tzinfo=timezone.utc))
    assert _run(monkeypatch, strategy, rates)["signal"] == "BUY"
    assert "pair=AUDJPY" in caplog.text
    assert "bar_utc=2026-08-25T02:45:00+00:00" in caplog.text
    assert "window=00:00-04:00" in caplog.text
    assert "signal=BUY" in caplog.text


def test_0700_exit_gate_uses_real_utc_not_server_clock():
    before = datetime(2026, 8, 25, 6, 59, tzinfo=timezone.utc)
    at = datetime(2026, 8, 25, 7, 0, tzinfo=timezone.utc)
    assert ma.minutes_since_midnight(before) < ma.T_AMR_EXIT
    assert ma.minutes_since_midnight(at) >= ma.T_AMR_EXIT
    # The corresponding broker clocks are 09:59/10:00; neither can move
    # eligibility back to 04:00 because main() gates on real-UTC `t`.
    assert ma.minutes_since_midnight(before + timedelta(hours=3)) != ma.T_AMR_EXIT
    assert "if (t >= T_AMR_EXIT and AMR_KEYS" in __import__("inspect").getsource(ma.main)
