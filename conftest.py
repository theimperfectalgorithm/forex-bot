"""Repository-wide pytest isolation, established before test collection."""
from __future__ import annotations

import atexit
import os
from pathlib import Path
import shutil
import tempfile

_TEST_DATA_ROOT = Path(tempfile.mkdtemp(prefix="forex-bot-pytest-data-"))
os.environ["FOREX_BOT_DATA_DIR"] = str(_TEST_DATA_ROOT)


def _block_live_mt5() -> None:
    try:
        import MetaTrader5 as mt5
    except ImportError:
        return
    if not getattr(mt5.initialize, "_pytest_blocked", False):
        def blocked_initialize(*args, **kwargs):
            raise AssertionError("live MetaTrader5.initialize() is forbidden under pytest")
        blocked_initialize._pytest_blocked = True
        mt5.initialize = blocked_initialize


# Collection-time imports are covered, not only test execution.
_block_live_mt5()


@atexit.register
def _cleanup() -> None:
    shutil.rmtree(_TEST_DATA_ROOT, ignore_errors=True)


def pytest_sessionstart(session):
    from core.runtime_paths import PRODUCTION_DATA_DIR, data_dir
    isolated = data_dir().resolve()
    if isolated == PRODUCTION_DATA_DIR or PRODUCTION_DATA_DIR in isolated.parents:
        raise RuntimeError(f"unsafe pytest data root: {isolated}")


def pytest_runtest_setup(item):
    _block_live_mt5()
