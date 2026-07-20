"""
account_config -- SINGLE SOURCE OF TRUTH for per-instance account numbers
(starting balance, hard floor, daily loss limit, profit target, lot cap,
risk scale). Every agent that needs one of these must import it from
HERE, never hardcode it.

Why this module exists (2026-07-19 incident): agent_risk.py was
correctly parameterized during the 2026-07-09 multi-account work, but
agent_market.py and agent_reporting.py each carried their OWN hardcoded
$100k-account constants that nobody updated. Result on the $5,000
challenge clone: agent_market compared the real $5,000 balance against
its hardcoded $90,000 floor, declared HARD FLOOR BREACHED, and returned
AVOID for every single day -- the bot never traded. agent_reporting
independently compared balance against its own hardcoded $100,000
baseline, producing a nonsensical "-$95,000 / 95% drawdown"
reconciliation warning. Two symptoms, one root cause: constants
duplicated instead of shared.

Config resolution (identical to core/news_calendar.py's pattern):
config/global_config.yaml `global:` block, overridden by the gitignored
per-instance config/local_config.yaml (same schema). Edit ONLY the
local file on a VPS clone -- never the tracked global file.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_CONFIG_DIR  = Path(__file__).parent.parent / 'config'
_CONFIG_FILE = _CONFIG_DIR / 'global_config.yaml'
_LOCAL_FILE  = _CONFIG_DIR / 'local_config.yaml'


def _load_global_config() -> dict:
    cfg = {}
    try:
        with open(_CONFIG_FILE, encoding='utf-8') as f:
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


GLOBAL_CFG = _load_global_config()

# -- account-scale constants, all derived from ONE starting_balance ---------
STARTING_BALANCE = float(GLOBAL_CFG.get('starting_balance', 100_000.00))
HARD_FLOOR_PCT   = float(GLOBAL_CFG.get('hard_floor_pct', 0.10))
HARD_FLOOR       = STARTING_BALANCE * (1.0 - HARD_FLOOR_PCT)
MAX_DAILY_LOSS_PCT = float(GLOBAL_CFG.get('daily_loss_pct', 0.05))
PROFIT_TARGET_PCT  = float(GLOBAL_CFG.get('profit_target_pct', 0.08))  # 5ers Classic step 1
TARGET_BALANCE     = STARTING_BALANCE * (1.0 + PROFIT_TARGET_PCT)

MAX_LOT     = float(GLOBAL_CFG.get('max_lot', 2.00))
RISK_SCALE  = float(GLOBAL_CFG.get('risk_scale', 1.0))
