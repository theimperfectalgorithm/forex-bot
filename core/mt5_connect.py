"""
mt5_connect -- forces EVERY MetaTrader5.initialize() call in this
process to target THIS instance's configured terminal/account,
regardless of which file makes the call or what arguments (if any) it
passes.

INCIDENT 2026-07-21: main_agent.py's own _bind_mt5_terminal() correctly
bound the demo clone to its terminal (account 5052472770) at process
startup -- confirmed by the log. Moments later, in the SAME process,
agent_market.py's own bare mt5.initialize() call (no path argument)
returned account info for the OTHER terminal (the 5ers challenge
account, 26520700) running concurrently on the same VPS. The assumption
that "a later bare mt5.initialize() call is a no-op against the
established connection" is FALSE when two terminal processes are
running side by side -- MT5's Python API does not reliably keep a bare
initialize() pinned to a specific terminal once more than one is
running on the machine.

This is not a one-off: ~20 files (agent_market.py, agent_risk.py,
agent_execution.py -- including order placement --, agent_strategy.py,
agent_reporting.py, core/trade_journal.py, core/data_loader.py, and
every strategies/*.py file) each make their own bare mt5.initialize()
call. Editing all of them individually is exactly the kind of change
that's easy to get wrong or miss a call site on (see the
account_config.py incident for a prior example of duplicated-constants
drift). Since every one of those files does `import MetaTrader5 as mt5`
and Python caches that as ONE shared module object process-wide,
patching `mt5.initialize` itself, once, here, fixes every call site
everywhere -- including any not yet written.

Usage: import this module for its side effect (the patch), as early as
possible in the process -- before the first real mt5.initialize() call,
not necessarily before every other file's own `import MetaTrader5`
statement (attribute lookups on the shared module happen at CALL time).
main_agent.py imports this first, so every agent/strategy module it
pulls in afterward is automatically covered.

    import core.mt5_connect  # side-effect only: patches MetaTrader5.initialize

Config: same resolution as core/account_config.py -- global_config.yaml
`global:` block, overridden by the gitignored per-instance
local_config.yaml. New keys used here: mt5_terminal_path (already used
by main_agent.py's _bind_mt5_terminal()), plus optional mt5_login /
mt5_server for an explicit account login (password from the
MT5_PASSWORD env var -- never the config file, this repo is public).
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

import yaml

_CONFIG_DIR  = Path(__file__).parent.parent / 'config'
_GLOBAL_FILE = _CONFIG_DIR / 'global_config.yaml'
_LOCAL_FILE  = _CONFIG_DIR / 'local_config.yaml'


def _load_cfg() -> dict:
    cfg = {}
    try:
        with open(_GLOBAL_FILE, encoding='utf-8') as f:
            cfg = (yaml.safe_load(f) or {}).get('global', {})
    except Exception:
        pass
    try:
        with open(_LOCAL_FILE, encoding='utf-8') as f:
            cfg.update((yaml.safe_load(f) or {}).get('global', {}))
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return cfg


_CFG       = _load_cfg()
MT5_PATH   = (_CFG.get('mt5_terminal_path') or '').strip()
MT5_LOGIN  = int(_CFG.get('mt5_login') or 0)
MT5_SERVER = str(_CFG.get('mt5_server', ''))
EXPECTED_MT5_LOGIN = int(_CFG.get('expected_mt5_login') or 0)
EXPECTED_MT5_SERVER = str(_CFG.get('expected_mt5_server', '')).strip()
EXPECTED_MT5_PATH = str(_CFG.get('expected_mt5_terminal_path', '')).strip()

PATCHED = False   # importable flag so callers/tests can confirm the patch applied

try:
    import MetaTrader5 as mt5

    if not getattr(mt5.initialize, '_mt5_connect_patched', False):
        _original_initialize = mt5.initialize

        def _pinned_initialize(*args, **kwargs):
            """
            Forces this instance's configured path/login onto every
            initialize() call in the process, overriding whatever (if
            anything) the caller passed -- on a multi-terminal VPS, a
            caller-supplied path is far more likely to be a stale/bare
            call than an intentional override, so this instance's
            configured target always wins. Positional args are dropped
            for the same reason (the only positional MT5 accepts here
            is `path`, which this replaces anyway).
            """
            if MT5_PATH:
                kwargs['path'] = MT5_PATH
            if MT5_LOGIN:
                kwargs.setdefault('login', MT5_LOGIN)
                kwargs.setdefault('server', MT5_SERVER)
                kwargs.setdefault('password', os.environ.get('MT5_PASSWORD', ''))
            return _original_initialize(**kwargs)

        _pinned_initialize._mt5_connect_patched = True
        mt5.initialize = _pinned_initialize

    PATCHED = True
except ImportError:
    PATCHED = False   # MetaTrader5 not installed (e.g. Mac dev) -- nothing to patch


def _normalise_path(value: str) -> str:
    """Return a comparison form for a configured Windows terminal path."""
    return os.path.normcase(os.path.abspath(os.path.expandvars(value.strip())))


def validate_expected_account(log: logging.Logger | None = None) -> bool:
    """Fail-closed validation of this instance's terminal/account identity.

    The ``expected_*`` settings are assertions only. They never cause an
    account login or expose a password. Callers must initialize MT5 first.
    """
    log = log or logging.getLogger('MT5_IDENTITY')
    missing = []
    if not MT5_PATH:
        missing.append('mt5_terminal_path')
    if not EXPECTED_MT5_PATH:
        missing.append('expected_mt5_terminal_path')
    if not EXPECTED_MT5_LOGIN:
        missing.append('expected_mt5_login')
    if not EXPECTED_MT5_SERVER:
        missing.append('expected_mt5_server')
    if missing:
        log.critical("MT5 identity validation FAILED: missing required config: %s",
                     ', '.join(missing))
        return False
    if _normalise_path(MT5_PATH) != _normalise_path(EXPECTED_MT5_PATH):
        log.critical("MT5 identity validation FAILED: configured terminal path=%r "
                     "expected=%r", MT5_PATH, EXPECTED_MT5_PATH)
        return False
    if not PATCHED:
        log.critical("MT5 identity validation FAILED: MetaTrader5 unavailable")
        return False
    acct = mt5.account_info()
    if acct is None:
        log.critical("MT5 identity validation FAILED: account_info unavailable; "
                     "expected login=%s server=%r",
                     EXPECTED_MT5_LOGIN, EXPECTED_MT5_SERVER)
        return False
    actual_login = int(getattr(acct, 'login', 0) or 0)
    actual_server = str(getattr(acct, 'server', '') or '').strip()
    if actual_login != EXPECTED_MT5_LOGIN:
        log.critical("MT5 identity validation FAILED: expected login=%s actual=%s "
                     "expected server=%r actual=%r",
                     EXPECTED_MT5_LOGIN, actual_login,
                     EXPECTED_MT5_SERVER, actual_server)
        return False
    if actual_server.casefold() != EXPECTED_MT5_SERVER.casefold():
        log.critical("MT5 identity validation FAILED: login=%s expected server=%r "
                     "actual=%r", actual_login, EXPECTED_MT5_SERVER, actual_server)
        return False
    terminal_info = mt5.terminal_info()
    actual_terminal = str(getattr(terminal_info, 'path', '') or '').strip()
    if actual_terminal:
        expected_dir = str(Path(EXPECTED_MT5_PATH).parent)
        if _normalise_path(actual_terminal) != _normalise_path(expected_dir):
            log.critical("MT5 identity validation FAILED: expected terminal dir=%r "
                         "actual=%r", expected_dir, actual_terminal)
            return False
    return True


def initialize_and_validate(log: logging.Logger | None = None) -> bool:
    """Initialize the pinned terminal and validate its expected identity."""
    log = log or logging.getLogger('MT5_IDENTITY')
    if not PATCHED:
        log.critical("MT5 initialization FAILED: MetaTrader5 unavailable")
        return False
    if not mt5.initialize():
        log.critical("MT5 initialization FAILED for expected terminal %r: %s",
                     EXPECTED_MT5_PATH or MT5_PATH, mt5.last_error())
        return False
    return validate_expected_account(log)
