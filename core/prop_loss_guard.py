"""Broker-authoritative prop loss protection for new entries.

No local journal/CSV/state value participates in the decision.  Production
configuration must explicitly select the daily reference and day boundary;
an absent or ambiguous selection blocks entries rather than guessing rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import logging
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

from core.account_config import GLOBAL_CFG, STARTING_BALANCE, HARD_FLOOR_PCT, MAX_DAILY_LOSS_PCT
from core.mt5_connect import validate_expected_account
from core.mt5_time import observed_server_utc_offset_hours
from core.runtime_paths import data_dir

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - production dependency
    mt5 = None


DAILY_REFERENCE_MODEL = "higher_of_day_start_balance_or_equity"
DAY_BOUNDARY_MODEL = "mt5_server_midnight"
SNAPSHOT_VERSION = 1


@dataclass(frozen=True)
class PropRules:
    start_balance: float
    max_loss_pct: float
    daily_loss_pct: float
    daily_reference_model: str
    day_boundary_model: str


@dataclass
class PropRiskResult:
    allowed: bool
    reason: str
    values: dict[str, Any]


def rules_from_config(config: dict | None = None) -> tuple[PropRules | None, str | None]:
    cfg = GLOBAL_CFG if config is None else config
    daily_reference = cfg.get("prop_daily_reference_model")
    day_boundary = cfg.get("prop_day_boundary_model", DAY_BOUNDARY_MODEL)
    if daily_reference != DAILY_REFERENCE_MODEL or day_boundary != DAY_BOUNDARY_MODEL:
        return None, (
            "CONFIGURATION REQUIRED: explicitly set prop_daily_reference_model="
            f"{DAILY_REFERENCE_MODEL} and prop_day_boundary_model={DAY_BOUNDARY_MODEL}")
    try:
        rules = PropRules(float(cfg.get("starting_balance", STARTING_BALANCE)),
                          float(cfg.get("hard_floor_pct", HARD_FLOOR_PCT)),
                          float(cfg.get("daily_loss_pct", MAX_DAILY_LOSS_PCT)),
                          daily_reference, day_boundary)
    except (TypeError, ValueError):
        return None, "invalid prop loss configuration"
    vals = (rules.start_balance, rules.max_loss_pct, rules.daily_loss_pct)
    if not all(math.isfinite(v) and v > 0 for v in vals) \
            or rules.max_loss_pct >= 1 or rules.daily_loss_pct >= 1:
        return None, "invalid prop loss configuration"
    return rules, None


def prop_day_start_utc(now_utc: datetime, server_offset_hours: int) -> datetime:
    """Return midnight in the currently observed MT5 server clock."""
    if now_utc.tzinfo is None or now_utc.utcoffset() is None:
        raise ValueError("now_utc must be timezone-aware")
    if server_offset_hours not in (2, 3):
        raise ValueError("ambiguous or unavailable MT5 server UTC offset")
    server_now = now_utc.astimezone(timezone.utc) + timedelta(hours=server_offset_hours)
    server_midnight = server_now.replace(hour=0, minute=0, second=0, microsecond=0)
    return server_midnight - timedelta(hours=server_offset_hours)


def _snapshot_file(path: Path | None = None) -> Path:
    return path or (data_dir() / "state" / "prop_daily_reference.json")


def _server_day_id(day_start: datetime, offset: int) -> str:
    return (day_start + timedelta(hours=offset)).date().isoformat()


def _snapshot_values(raw: Any, *, login: int, server: str, day_id: str,
                     offset: int) -> tuple[dict[str, Any] | None, str | None]:
    required = {"version", "server_day", "day_start_balance", "day_start_equity",
                "daily_reference", "login", "server", "captured_at_utc",
                "server_offset_hours", "source"}
    if not isinstance(raw, dict) or not required.issubset(raw):
        return None, "daily reference snapshot missing required fields"
    try:
        balance = _finite(raw["day_start_balance"])
        equity = _finite(raw["day_start_equity"])
        reference = _finite(raw["daily_reference"])
        captured = datetime.fromisoformat(str(raw["captured_at_utc"]))
        valid = (raw["version"] == SNAPSHOT_VERSION and int(raw["login"]) == login
                 and str(raw["server"]).casefold() == server.casefold()
                 and raw["server_day"] == day_id
                 and int(raw["server_offset_hours"]) == offset
                 and captured.tzinfo is not None
                 and abs(reference - max(balance, equity)) <= 1e-9)
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        return None, "daily reference snapshot identity/content invalid"
    return raw, None


def load_daily_snapshot(*, login: int, server: str, day_id: str, offset: int,
                        path: Path | None = None) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with _snapshot_file(path).open(encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        return None, "daily reference snapshot missing"
    except (OSError, ValueError, json.JSONDecodeError):
        return None, "daily reference snapshot corrupt/unreadable"
    return _snapshot_values(raw, login=login, server=server, day_id=day_id,
                            offset=offset)


def write_daily_snapshot(snapshot: dict[str, Any], *, path: Path | None = None) -> None:
    """Atomically publish a snapshot; never replace one for the same day."""
    target = _snapshot_file(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            existing = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            raise FileExistsError("existing daily reference snapshot is unreadable")
        if existing.get("server_day") == snapshot.get("server_day"):
            raise FileExistsError("daily reference snapshot already exists")
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp",
                                     dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, sort_keys=True, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            if os.path.exists(temporary):
                os.unlink(temporary)
        finally:
            raise


def make_daily_snapshot(*, account: Any, day_start: datetime, offset: int,
                        balance: float, equity: float, source: str,
                        captured_at: datetime) -> dict[str, Any]:
    balance, equity = _finite(balance), _finite(equity)
    return {"version": SNAPSHOT_VERSION,
            "server_day": _server_day_id(day_start, offset),
            "day_start_balance": balance, "day_start_equity": equity,
            "daily_reference": max(balance, equity),
            "login": int(account.login), "server": str(account.server),
            "captured_at_utc": captured_at.astimezone(timezone.utc).isoformat(),
            "server_offset_hours": offset, "source": source}


def _finite(value: Any) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("nonfinite broker value")
    return value


def _deal_impact(deal: Any) -> float:
    return sum(_finite(getattr(deal, field, 0.0) or 0.0)
               for field in ("profit", "commission", "swap", "fee"))


def _deal_kind(deal: Any, api: Any) -> str:
    deal_type = getattr(deal, "type", None)
    trade_types = {getattr(api, "DEAL_TYPE_BUY", 0), getattr(api, "DEAL_TYPE_SELL", 1)}
    cashflow_names = ("DEAL_TYPE_BALANCE", "DEAL_TYPE_CREDIT", "DEAL_TYPE_CORRECTION",
                      "DEAL_TYPE_BONUS")
    cost_names = ("DEAL_TYPE_COMMISSION", "DEAL_TYPE_COMMISSION_DAILY",
                  "DEAL_TYPE_COMMISSION_MONTHLY", "DEAL_TYPE_COMMISSION_AGENT_DAILY",
                  "DEAL_TYPE_COMMISSION_AGENT_MONTHLY", "DEAL_TYPE_CHARGE",
                  "DEAL_TYPE_INTEREST")
    cashflow_types = {getattr(api, name) for name in cashflow_names if hasattr(api, name)}
    cost_types = {getattr(api, name) for name in cost_names if hasattr(api, name)}
    if deal_type in trade_types:
        return "trade"
    if deal_type in cost_types:
        return "cost"
    if deal_type in cashflow_types:
        return "cashflow"
    return "unknown"


def _reserve_open_positions(api: Any, positions: Any) -> tuple[float | None, str | None]:
    total = 0.0
    for position in positions:
        symbol = str(getattr(position, "symbol", ""))
        sl = _finite(getattr(position, "sl", 0.0) or 0.0)
        if not symbol or sl <= 0:
            return None, f"open position {getattr(position, 'ticket', '?')} has no SL"
        info = api.symbol_info(symbol)
        if info is None:
            return None, f"symbol_info unavailable for open position {symbol}"
        tick = api.symbol_info_tick(symbol)
        if tick is None:
            return None, f"symbol_info_tick unavailable for open position {symbol}"
        ptype = getattr(position, "type", None)
        if ptype == api.POSITION_TYPE_BUY:
            order_type, price = api.ORDER_TYPE_BUY, _finite(tick.bid)
        elif ptype == api.POSITION_TYPE_SELL:
            order_type, price = api.ORDER_TYPE_SELL, _finite(tick.ask)
        else:
            return None, f"unknown open position type for {symbol}"
        volume = _finite(getattr(position, "volume", 0.0))
        if volume <= 0 or price <= 0:
            return None, f"invalid open position data for {symbol}"
        pnl = api.order_calc_profit(order_type, symbol, volume, price, sl)
        if pnl is None:
            return None, f"order_calc_profit failed for open position {symbol}"
        pnl = _finite(pnl)
        total += max(0.0, -pnl)
    return total, None


def evaluate_prop_risk(candidate_risk: float, *, rules: PropRules | None = None,
                       api: Any = None, now_utc: datetime | None = None,
                       identity_validator: Callable[[logging.Logger], bool] | None = None,
                       offset_provider: Callable[[Any], int | None] | None = None,
                       snapshot_path: Path | None = None,
                       snapshot_write: bool = True,
                       log: logging.Logger | None = None) -> PropRiskResult:
    """Return an account-wide allow/block decision; every broker error blocks."""
    logger = log or logging.getLogger("PROP_RISK")
    api = api or mt5
    values: dict[str, Any] = {"candidate_risk": candidate_risk}

    def block(reason: str) -> PropRiskResult:
        values["decision"] = "BLOCK"
        logger.warning("PROP RISK: %s reason=%s", _format_values(values), reason)
        return PropRiskResult(False, reason, values)

    try:
        candidate_risk = _finite(candidate_risk)
        if candidate_risk < 0:
            return block("invalid candidate risk")
        values["candidate_risk"] = candidate_risk
        validator = identity_validator or validate_expected_account
        if api is None or not validator(logger):
            return block("wrong or unverifiable expected account identity")
        account = api.account_info()
        if account is None:
            return block("account_info unavailable")
        values.update(login=int(account.login), server=str(account.server),
                      balance=_finite(account.balance), equity=_finite(account.equity))

        if rules is None:
            rules, config_error = rules_from_config()
            if config_error:
                # The overall rule is unambiguous even when the daily model is
                # not. Include its broker-equity diagnostics in the fail-closed
                # result without pretending the daily rule was resolved.
                start = _finite(GLOBAL_CFG.get("starting_balance", STARTING_BALANCE))
                max_loss_pct = _finite(GLOBAL_CFG.get("hard_floor_pct", HARD_FLOOR_PCT))
                overall_floor = start * (1.0 - max_loss_pct)
                values.update(start=start, max_loss_pct=max_loss_pct,
                              overall_floor=overall_floor,
                              overall_headroom=values["equity"] - overall_floor,
                              overall_headroom_pct=(values["equity"] - overall_floor) / start)
                return block(config_error)
        values.update(start=rules.start_balance,
                      max_loss_pct=rules.max_loss_pct,
                      daily_loss_pct=rules.daily_loss_pct,
                      daily_reference=rules.daily_reference_model,
                      day_boundary=rules.day_boundary_model)
        overall_floor = rules.start_balance * (1.0 - rules.max_loss_pct)
        values["overall_floor"] = overall_floor

        positions = api.positions_get()
        if positions is None:
            return block("positions_get failed")
        reserved, reserve_error = _reserve_open_positions(api, positions)
        if reserve_error:
            return block(reserve_error)
        values["open_reserved_risk"] = reserved

        now = now_utc or datetime.now(timezone.utc)
        provider = offset_provider or observed_server_utc_offset_hours
        offset = provider(api)
        if offset not in (2, 3):
            return block("MT5 server UTC offset unavailable or ambiguous")
        day_start = prop_day_start_utc(now, int(offset))
        day_id = _server_day_id(day_start, int(offset))
        values.update(server_offset_h=int(offset), server_day=day_id,
                      prop_day_start=day_start.isoformat())
        deals = api.history_deals_get(day_start, now)
        if deals is None:
            return block("history_deals_get failed")

        trade_net = positive_cashflows = negative_cashflows = all_impacts = 0.0
        realized_gross = commission = swap = fees = 0.0
        for deal in deals:
            impact = _deal_impact(deal)
            kind = _deal_kind(deal, api)
            if kind in ("trade", "cost"):
                trade_net += impact
                realized_gross += _finite(getattr(deal, "profit", 0.0) or 0.0)
                commission += _finite(getattr(deal, "commission", 0.0) or 0.0)
                swap += _finite(getattr(deal, "swap", 0.0) or 0.0)
                fees += _finite(getattr(deal, "fee", 0.0) or 0.0)
            elif kind == "cashflow":
                if impact >= 0:
                    positive_cashflows += impact
                else:
                    negative_cashflows += impact
            else:
                return block(f"unclassified broker deal type {getattr(deal, 'type', None)}")
            all_impacts += impact

        # Reconstruct the broker-day opening balance after restart. Positive
        # cash flows are removed from current equity so deposits/credits cannot
        # expand capacity. Withdrawals remain in adjusted equity and therefore
        # consume headroom conservatively.
        reconstructed_balance = values["balance"] - all_impacts
        adjusted_equity = values["equity"] - positive_cashflows
        floating = values["equity"] - values["balance"]
        daily_limit = rules.start_balance * rules.daily_loss_pct
        snapshot, snapshot_error = load_daily_snapshot(
            login=values["login"], server=values["server"], day_id=day_id,
            offset=int(offset), path=snapshot_path)
        if snapshot is None:
            trade_deals = [d for d in deals if _deal_kind(d, api) == "trade"]
            seconds_after_midnight = (now - day_start).total_seconds()
            if 0 <= seconds_after_midnight <= 60:
                snapshot = make_daily_snapshot(
                    account=account, day_start=day_start, offset=int(offset),
                    balance=values["balance"], equity=values["equity"],
                    source="broker_midnight_observation", captured_at=now)
            elif not positions and not trade_deals:
                snapshot = make_daily_snapshot(
                    account=account, day_start=day_start, offset=int(offset),
                    balance=reconstructed_balance, equity=reconstructed_balance,
                    source="broker_reconstruction_no_midnight_positions",
                    captured_at=now)
            else:
                return block(f"midnight equity unavailable: {snapshot_error}; "
                             "overnight position cannot be excluded")
            if snapshot_write:
                try:
                    write_daily_snapshot(snapshot, path=snapshot_path)
                except FileExistsError:
                    # A concurrent writer won. Only the validated persisted
                    # value may be used; never prefer this later observation.
                    snapshot, snapshot_error = load_daily_snapshot(
                        login=values["login"], server=values["server"],
                        day_id=day_id, offset=int(offset), path=snapshot_path)
                    if snapshot is None:
                        return block(snapshot_error or "snapshot publication failed")
                except OSError as exc:
                    return block(f"daily reference snapshot write failed: {exc}")
        baseline = _finite(snapshot["daily_reference"])
        day_start_balance = _finite(snapshot["day_start_balance"])
        day_start_equity = _finite(snapshot["day_start_equity"])
        daily_floor = baseline - daily_limit
        daily_used = max(0.0, baseline - adjusted_equity)
        values.update(realized_daily_gross=realized_gross,
                      realized_daily_commission=commission,
                      realized_daily_swap=swap, realized_daily_fees=fees,
                      realized_daily_net=trade_net,
                      positive_cashflows=positive_cashflows,
                      negative_cashflows=negative_cashflows,
                      day_start_balance=day_start_balance,
                      day_start_equity=day_start_equity,
                      daily_baseline=baseline, daily_floor=daily_floor,
                      snapshot_source=snapshot["source"],
                      adjusted_equity=adjusted_equity,
                      floating_pnl=floating, daily_limit=daily_limit,
                      daily_used=daily_used,
                      overall_headroom=values["equity"] - overall_floor,
                      overall_headroom_pct=(values["equity"] - overall_floor)
                                           / rules.start_balance,
                      daily_headroom=daily_limit - daily_used)

        tolerance = 1e-9
        if values["equity"] <= overall_floor + tolerance:
            return block("overall equity is at/below configured floor")
        if daily_used >= daily_limit - tolerance:
            return block("daily loss is at/above configured limit")
        worst_equity = values["equity"] - reserved - candidate_risk
        worst_adjusted_equity = adjusted_equity - reserved - candidate_risk
        values.update(worst_equity=worst_equity,
                      worst_daily_used=max(0.0, baseline - worst_adjusted_equity))
        if worst_equity <= overall_floor + tolerance:
            return block("open reserved risk plus candidate crosses overall floor")
        if values["worst_daily_used"] >= daily_limit - tolerance:
            return block("open reserved risk plus candidate crosses daily limit")
        values["decision"] = "ALLOW"
        logger.info("PROP RISK: %s reason=within broker-authoritative limits",
                    _format_values(values))
        return PropRiskResult(True, "within broker-authoritative limits", values)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        return block(f"invalid broker data: {exc}")
    except Exception as exc:  # broker API errors are entry blockers
        return block(f"broker risk snapshot failed: {exc}")


def _format_values(values: dict[str, Any]) -> str:
    keys = ("start", "balance", "equity", "overall_floor", "overall_headroom",
            "overall_headroom_pct", "daily_reference", "day_boundary",
            "realized_daily_gross", "realized_daily_commission",
            "realized_daily_swap", "realized_daily_fees", "realized_daily_net",
            "daily_baseline", "daily_limit", "daily_used", "daily_headroom",
            "open_reserved_risk", "candidate_risk", "decision")
    return " ".join(f"{key}={values[key]}" for key in keys if key in values)
