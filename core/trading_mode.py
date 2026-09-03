"""Fail-closed, process-lifetime global control for new trade entries."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
GLOBAL_CONFIG_FILE = CONFIG_DIR / "global_config.yaml"
LOCAL_CONFIG_FILE = CONFIG_DIR / "local_config.yaml"
VALID_MODES = frozenset({"LIVE", "PAUSED", "SHADOW"})


@dataclass(frozen=True)
class TradingModeStatus:
    mode: str | None
    entries_allowed: bool
    reason: str


def resolve_trading_mode(global_file: Path = GLOBAL_CONFIG_FILE,
                         local_file: Path = LOCAL_CONFIG_FILE) -> TradingModeStatus:
    """Resolve ``trading.mode`` once, with local config overriding the default.

    Any read, parse, shape, or validation problem fails closed.  A missing
    optional local file is normal; a present but unreadable/malformed one is
    not ignored because doing so could unexpectedly enable entries.
    """
    try:
        with Path(global_file).open(encoding="utf-8") as handle:
            base = yaml.safe_load(handle)
        if not isinstance(base, dict):
            raise ValueError("global configuration root must be a mapping")
        trading = base.get("trading")
        if trading is not None and not isinstance(trading, dict):
            raise ValueError("trading configuration must be a mapping")

        local_path = Path(local_file)
        if local_path.exists():
            with local_path.open(encoding="utf-8") as handle:
                local = yaml.safe_load(handle)
            if not isinstance(local, dict):
                raise ValueError("local configuration root must be a mapping")
            if "trading" in local:
                if not isinstance(local["trading"], dict):
                    raise ValueError("local trading configuration must be a mapping")
                trading = local["trading"]

        if not isinstance(trading, dict) or "mode" not in trading:
            return TradingModeStatus(None, False, "trading.mode is missing")
        raw_mode = trading["mode"]
        if not isinstance(raw_mode, str) or not raw_mode.strip():
            return TradingModeStatus(None, False, "trading.mode must be a non-empty string")
        mode = raw_mode.strip().upper()
        if mode not in VALID_MODES:
            return TradingModeStatus(None, False,
                                     f"unsupported trading.mode {raw_mode!r}")
        return TradingModeStatus(mode, mode == "LIVE", f"configured mode {mode}")
    except Exception as exc:
        return TradingModeStatus(None, False,
                                 f"trading mode configuration unavailable: {exc}")


@lru_cache(maxsize=1)
def get_trading_mode() -> TradingModeStatus:
    """Return the process-lifetime mode; config changes require a restart."""
    return resolve_trading_mode()


def log_startup_mode(log) -> TradingModeStatus:
    status = get_trading_mode()
    if status.mode == "LIVE":
        log.info("TRADING MODE: LIVE")
    elif status.mode == "PAUSED":
        log.warning("TRADING MODE: PAUSED -- new entries disabled; position management remains active")
    elif status.mode == "SHADOW":
        log.warning("TRADING MODE: SHADOW -- signals evaluated; real entries disabled; position management remains active")
    else:
        log.critical("TRADING MODE: INVALID/UNAVAILABLE -- new entries blocked; position management remains active; %s",
                     status.reason)
    return status


def allow_or_log_entry(log, strategy_key: str, symbol: str,
                       direction: str) -> bool:
    """Authoritative orchestrator gate after signal discovery."""
    status = get_trading_mode()
    if status.entries_allowed:
        return True
    if status.mode == "SHADOW":
        log.warning("SHADOW SIGNAL %s %s -- no live order sent",
                    strategy_key, direction)
    else:
        label = status.mode or "INVALID/UNAVAILABLE"
        log.warning("ENTRY BLOCKED %s %s -- trading mode %s%s",
                    strategy_key, direction, label,
                    "" if status.mode else f" ({status.reason})")
    return False
