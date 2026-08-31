"""Task 016C: authoritative MAX(balance, equity) daily reference snapshots."""
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from core.prop_loss_guard import (DAILY_REFERENCE_MODEL, DAY_BOUNDARY_MODEL,
    PropRules, evaluate_prop_risk, load_daily_snapshot, make_daily_snapshot,
    prop_day_start_utc, rules_from_config, write_daily_snapshot)


NOW = datetime(2026, 8, 31, 10, 30, tzinfo=timezone.utc)
DAY_START = datetime(2026, 8, 30, 21, tzinfo=timezone.utc)
RULES = PropRules(5000, .10, .05, DAILY_REFERENCE_MODEL, DAY_BOUNDARY_MODEL)


def account(balance=5000, equity=5000, login=26520700,
            server="FivePercentOnline-Real"):
    return NS(balance=balance, equity=equity, login=login, server=server)


def snapshot(balance, equity, *, acct=None,
             source="authoritative_server_midnight"):
    return make_daily_snapshot(account=acct or account(), day_start=DAY_START,
        offset=3, balance=balance, equity=equity, source=source,
        captured_at=DAY_START)


@pytest.mark.parametrize("balance,equity,reference,floor", [
    (5000, 5000, 5000, 4750),
    (5000, 4900, 5000, 4750),
    (5000, 5100, 5100, 4850),
    (4800, 4750, 4800, 4550),
])
def test_authoritative_examples(balance, equity, reference, floor):
    snap = snapshot(balance, equity)
    assert snap["daily_reference"] == reference
    assert reference - 5000 * .05 == floor


def test_exact_configuration_is_accepted():
    rules, error = rules_from_config({
        "starting_balance": 5000, "hard_floor_pct": .10,
        "daily_loss_pct": .05,
        "prop_daily_reference_model": DAILY_REFERENCE_MODEL,
        "prop_day_boundary_model": DAY_BOUNDARY_MODEL})
    assert error is None and rules == RULES


@pytest.mark.parametrize("offset,hour", [(2, 22), (3, 21)])
def test_server_midnight_not_utc_or_windows_local(offset, hour):
    start = prop_day_start_utc(NOW, offset)
    assert start.hour == hour and start.tzinfo == timezone.utc and start.hour != 0


@pytest.mark.parametrize("offset", [None, 0, 2.5, 4])
def test_unavailable_or_ambiguous_offset_blocks(offset, tmp_path):
    api = Broker()
    result = evaluate_prop_risk(0, rules=RULES, api=api, now_utc=NOW,
        identity_validator=lambda log: True,
        offset_provider=lambda unused: offset,
        snapshot_path=tmp_path / "snapshot.json")
    assert not result.allowed and "offset" in result.reason


def test_valid_snapshot_round_trip_and_exact_day_identity(tmp_path):
    path = tmp_path / "snapshot.json"
    write_daily_snapshot(snapshot(5000, 5100), path=path)
    loaded, error = load_daily_snapshot(login=26520700,
        server="FivePercentOnline-Real", day_id="2026-08-31", offset=3,
        path=path)
    assert error is None and loaded["daily_reference"] == 5100
    assert loaded["server_day"] == "2026-08-31"
    assert loaded["server_offset_hours"] == 3


@pytest.mark.parametrize("mutation", [
    lambda value: value.pop("day_start_equity"),
    lambda value: value.update(login=999),
    lambda value: value.update(server="Wrong"),
    lambda value: value.update(server_day="2026-08-30"),
    lambda value: value.update(day_start_balance=math.nan),
    lambda value: value.update(day_start_equity=math.inf),
    lambda value: value.update(daily_reference=9999),
])
def test_invalid_snapshot_blocks_load(tmp_path, mutation):
    value = snapshot(5000, 5000)
    mutation(value)
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    loaded, error = load_daily_snapshot(login=26520700,
        server="FivePercentOnline-Real", day_id="2026-08-31", offset=3,
        path=path)
    assert loaded is None and error


def test_partial_write_is_corrupt(tmp_path):
    path = tmp_path / "snapshot.json"
    path.write_text('{"server_day":', encoding="utf-8")
    loaded, error = load_daily_snapshot(login=26520700,
        server="FivePercentOnline-Real", day_id="2026-08-31", offset=3,
        path=path)
    assert loaded is None and "corrupt" in error


def test_write_uses_atomic_replace(tmp_path, monkeypatch):
    calls = []
    original = os.replace
    monkeypatch.setattr(os, "replace", lambda src, dst: (calls.append((src, dst)),
                                                          original(src, dst))[1])
    path = tmp_path / "snapshot.json"
    write_daily_snapshot(snapshot(5000, 5000), path=path)
    assert calls and path.exists() and not list(tmp_path.glob("*.tmp"))


def test_same_day_snapshot_cannot_be_replaced_by_favorable_equity(tmp_path):
    path = tmp_path / "snapshot.json"
    write_daily_snapshot(snapshot(5000, 4900), path=path)
    with pytest.raises(FileExistsError):
        write_daily_snapshot(snapshot(5000, 5500), path=path)
    assert json.loads(path.read_text())["daily_reference"] == 5000


def test_next_day_snapshot_atomically_replaces_old_day(tmp_path):
    path = tmp_path / "snapshot.json"
    write_daily_snapshot(snapshot(5000, 5000), path=path)
    next_start = datetime(2026, 8, 31, 21, tzinfo=timezone.utc)
    next_snap = make_daily_snapshot(account=account(), day_start=next_start,
        offset=3, balance=4900, equity=4900, source="next_day",
        captured_at=next_start)
    write_daily_snapshot(next_snap, path=path)
    assert json.loads(path.read_text())["server_day"] == "2026-09-01"


class Broker:
    DEAL_TYPE_BUY = 0; DEAL_TYPE_SELL = 1; DEAL_TYPE_BALANCE = 2
    DEAL_TYPE_CREDIT = 3; DEAL_TYPE_COMMISSION = 7; DEAL_TYPE_CHARGE = 15
    DEAL_TYPE_CORRECTION = 17; DEAL_TYPE_BONUS = 18
    POSITION_TYPE_BUY = 0; POSITION_TYPE_SELL = 1
    ORDER_TYPE_BUY = 0; ORDER_TYPE_SELL = 1
    def __init__(self, *, positions=(), deals=(), balance=4800, equity=4800):
        self.account = account(balance, equity); self.positions = positions
        self.deals = deals
    def account_info(self): return self.account
    def positions_get(self): return self.positions
    def history_deals_get(self, start, end): return self.deals
    def symbol_info(self, symbol): return NS(name=symbol)
    def symbol_info_tick(self, symbol): return NS(bid=1.1, ask=1.1002)
    def order_calc_profit(self, *args): return -10


def evaluate(api, path, *, write=True, now=NOW):
    return evaluate_prop_risk(0, rules=RULES, api=api, now_utc=now,
        identity_validator=lambda log: True, offset_provider=lambda unused: 3,
        snapshot_path=path, snapshot_write=write)


def test_missing_snapshot_safe_reconstruction_and_restart(tmp_path):
    path = tmp_path / "snapshot.json"
    first = evaluate(Broker(), path)
    second = evaluate(Broker(), path)
    assert first.allowed and second.allowed
    assert first.values["daily_baseline"] == second.values["daily_baseline"] == 4800
    assert second.values["snapshot_source"] == "broker_reconstruction_no_midnight_positions"


def test_missing_snapshot_possible_overnight_position_blocks(tmp_path):
    pos = NS(ticket=1, symbol="EURUSD", sl=1.09, volume=0.1, type=0)
    result = evaluate(Broker(positions=(pos,)), tmp_path / "snapshot.json")
    assert not result.allowed and "overnight position" in result.reason


@pytest.mark.parametrize("seconds", [1, 5, 59, 60, 61])
@pytest.mark.parametrize("observed_equity", [5120, 5050])
def test_delayed_midnight_observation_never_becomes_reference(
        tmp_path, seconds, observed_equity):
    pos = NS(ticket=1, symbol="EURUSD", sl=1.09, volume=0.1, type=0)
    observed = DAY_START + timedelta(seconds=seconds)
    path = tmp_path / "snapshot.json"
    result = evaluate(Broker(positions=(pos,), balance=5000,
                             equity=observed_equity), path,
                      now=observed)
    assert not result.allowed and "overnight position" in result.reason
    assert "daily_baseline" not in result.values
    assert not path.exists()


def test_legacy_delayed_observation_snapshot_is_rejected(tmp_path):
    path = tmp_path / "snapshot.json"
    delayed = snapshot(5000, 5120, source="broker_midnight_observation")
    delayed["captured_at_utc"] = (DAY_START + timedelta(seconds=5)).isoformat()
    path.write_text(json.dumps(delayed), encoding="utf-8")
    loaded, error = load_daily_snapshot(login=26520700,
        server="FivePercentOnline-Real", day_id="2026-08-31", offset=3,
        path=path)
    assert loaded is None and "invalid" in error


def test_exact_source_with_delayed_timestamp_is_rejected(tmp_path):
    path = tmp_path / "snapshot.json"
    delayed = snapshot(5000, 5100)
    delayed["captured_at_utc"] = (DAY_START + timedelta(seconds=1)).isoformat()
    path.write_text(json.dumps(delayed), encoding="utf-8")
    loaded, error = load_daily_snapshot(login=26520700,
        server="FivePercentOnline-Real", day_id="2026-08-31", offset=3,
        path=path)
    assert loaded is None and error


def test_proven_flat_reconstruction_never_uses_current_equity(tmp_path):
    path = tmp_path / "snapshot.json"
    result = evaluate(Broker(balance=4800, equity=4900), path)
    assert result.allowed
    assert result.values["day_start_balance"] == 4800
    assert result.values["day_start_equity"] == 4800
    assert result.values["daily_baseline"] != 4900


def test_corrupt_snapshot_does_not_fall_back_when_overnight_possible(tmp_path):
    path = tmp_path / "snapshot.json"; path.write_text("{", encoding="utf-8")
    pos = NS(ticket=1, symbol="EURUSD", sl=1.09, volume=0.1, type=0)
    result = evaluate(Broker(positions=(pos,)), path)
    assert not result.allowed and "midnight equity unavailable" in result.reason


def test_valid_higher_equity_snapshot_drives_floor_and_exact_floor_blocks(tmp_path):
    path = tmp_path / "snapshot.json"
    write_daily_snapshot(snapshot(5000, 5100), path=path)
    result = evaluate(Broker(balance=4850, equity=4850), path)
    assert not result.allowed
    assert result.values["daily_baseline"] == 5100
    assert result.values["daily_floor"] == 4850
    assert result.values["daily_used"] == 250
