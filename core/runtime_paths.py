"""Injectable runtime storage root with unchanged production defaults."""
from __future__ import annotations

import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_DATA_DIR = (REPO_ROOT / "data").resolve()
DATA_DIR_ENV = "FOREX_BOT_DATA_DIR"
PRODUCTION_RUNTIME_ROOTS = tuple((REPO_ROOT / name).resolve() for name in
                                ('data', 'logs', 'state', 'journal', 'journals', 'reports'))


def is_production_runtime_path(value) -> bool:
    path = Path(value).expanduser().resolve()
    return any(path == root or root in path.parents for root in PRODUCTION_RUNTIME_ROOTS)


def data_dir() -> Path:
    configured = os.environ.get(DATA_DIR_ENV)
    path = Path(configured).expanduser().resolve() if configured else PRODUCTION_DATA_DIR
    if "pytest" in sys.modules and is_production_runtime_path(path):
        raise RuntimeError(
            f"pytest may not use production data; set {DATA_DIR_ENV} before imports")
    return path
