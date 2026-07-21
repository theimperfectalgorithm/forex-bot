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
