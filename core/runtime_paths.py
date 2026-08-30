"""Injectable runtime storage root with unchanged production defaults."""
from __future__ import annotations

import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_DATA_DIR = (REPO_ROOT / "data").resolve()
DATA_DIR_ENV = "FOREX_BOT_DATA_DIR"


def data_dir() -> Path:
    configured = os.environ.get(DATA_DIR_ENV)
    path = Path(configured).expanduser().resolve() if configured else PRODUCTION_DATA_DIR
    if "pytest" in sys.modules and path == PRODUCTION_DATA_DIR:
        raise RuntimeError(
            f"pytest may not use production data; set {DATA_DIR_ENV} before imports")
    return path
