"""Repository-wide pytest isolation, established before test collection."""
from __future__ import annotations

import atexit
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

import pytest

_TEST_DATA_ROOT = Path(tempfile.mkdtemp(prefix="forex-bot-pytest-data-"))
os.environ["FOREX_BOT_DATA_DIR"] = str(_TEST_DATA_ROOT)
from core.runtime_paths import is_production_runtime_path


def _protect_operational_writes(event, args):
    """Collection-time guard for ordinary Python filesystem APIs, including
    hardcoded legacy paths. Reads/source fixtures remain available. This is a
    regression safety guard, not a sandbox for native code or subprocesses.
    """
    def protected(value):
        return (isinstance(value, (str, bytes, os.PathLike))
                and is_production_runtime_path(os.fsdecode(value)))

    if event == 'open':
        path, mode, flags = args
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            if protected(path):
                raise RuntimeError(f'pytest forbids production operational write: {path}')
    elif event in ('os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.utime'):
        if protected(args[0]):
            raise RuntimeError(f'pytest forbids production operational mutation: {args[0]}')
    elif event in ('os.rename', 'os.link', 'os.symlink'):
        if any(protected(value) for value in args[:2]):
            raise RuntimeError('pytest forbids production operational move/link')


sys.addaudithook(_protect_operational_writes)


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


@pytest.fixture(autouse=True)
def _isolated_strategy_logs(monkeypatch):
    """Legacy strategy loggers predate runtime_paths; keep regression tests safe."""
    for name, module in tuple(sys.modules.items()):
        if name.startswith('strategies.') and hasattr(module, 'LOGS_DIR'):
            monkeypatch.setattr(module, 'LOGS_DIR', _TEST_DATA_ROOT / 'logs')

    original_init = logging.FileHandler.__init__
    production = (Path(__file__).parent / 'data').resolve()

    def guarded_init(handler, filename, *args, **kwargs):
        target = Path(filename).resolve()
        if target == production or production in target.parents:
            raise AssertionError(f'production log handler forbidden under pytest: {target}')
        original_init(handler, filename, *args, **kwargs)

    monkeypatch.setattr(logging.FileHandler, '__init__', guarded_init)


@pytest.fixture(autouse=True)
def _isolated_news(monkeypatch, tmp_path):
    """No test can read the production calendar/config or fetch live news."""
    from core import news_calendar as news
    config = tmp_path / 'global_config.yaml'
    config.write_text('global:\n  news_filter: true\n  news_fail_closed: true\n'
                      '  news_window_min: 5\n', encoding='utf-8')
    monkeypatch.setattr(news, 'CONFIG_FILE', config)
    monkeypatch.setattr(news, 'CACHE_FILE', tmp_path / 'news_calendar.json')
    monkeypatch.setattr(news, '_memory_snapshot', None)
    monkeypatch.setattr(news, '_retry_after', 0.0)

    def offline():
        raise AssertionError('live calendar fetch forbidden under pytest')

    monkeypatch.setattr(news, '_fetch_feed', offline)


@pytest.fixture
def clear_news(monkeypatch, _isolated_news):
    """Explicit CLEAR evidence for pre-existing order-path regression tests."""
    from core import news_calendar as news
    from datetime import datetime, timezone
    snapshot = news.CalendarSnapshot(datetime.now(timezone.utc), ())
    # A test-only completeness oracle, never a claim about the FF provider.
    monkeypatch.setattr(news, '_proves_coverage', lambda *_a: True)
    monkeypatch.setattr(news, 'evaluate_news', lambda *_a, **_kw:
                        news.NewsResult(news.NewsStatus.CLEAR, 'explicit test calendar',
                                        snapshot=snapshot))


@pytest.fixture(autouse=True)
def _explicit_live_mode_for_legacy_execution_tests(monkeypatch):
    """Existing order-path tests opt into LIVE explicitly.

    Task017 tests override this where they exercise fail-closed modes. This
    changes no config file and retains the repository's safe PAUSED default.
    """
    from src.agents import agent_execution
    monkeypatch.setattr(
        agent_execution, "get_trading_mode",
        lambda: SimpleNamespace(mode="LIVE", entries_allowed=True,
                                reason="explicit pytest LIVE mode"))
