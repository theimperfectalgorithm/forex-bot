"""Task 015B: representative production writers stay in the pytest root."""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from core import runtime_paths
from core import trade_cost_ledger as ledger
from core import trade_journal as journal
from src.agents import agent_execution as execution
from src.agents import agent_reporting as reporting
from src.agents import main_agent


def _inside(child: Path, parent: Path) -> bool:
    return child.resolve() == parent.resolve() or parent.resolve() in child.resolve().parents


def test_all_runtime_write_roots_are_isolated():
    isolated = runtime_paths.data_dir()
    assert not _inside(isolated, runtime_paths.PRODUCTION_DATA_DIR)
    paths = [execution.DATA_DIR, execution.LOGS_DIR, execution.TRADES_LOG,
             reporting.DATA_DIR, reporting.LOGS_DIR, reporting.EQUITY_CSV,
             reporting.REPORT_TXT, reporting.TRADES_CSV,
             main_agent.DATA_DIR, main_agent.STATE_FILE,
             journal.JOURNAL_DIR, journal.JOURNAL_FILE, ledger.LEDGER_FILE]
    assert all(_inside(Path(path), isolated) for path in paths)


def test_production_root_guard_fails_loudly(monkeypatch):
    monkeypatch.setenv(runtime_paths.DATA_DIR_ENV,
                       str(runtime_paths.PRODUCTION_DATA_DIR))
    with pytest.raises(RuntimeError, match="pytest may not use production data"):
        runtime_paths.data_dir()


def test_execution_logging_is_isolated():
    log = execution._log()
    log.info("isolated representative execution event")
    targets = [Path(h.baseFilename) for h in log.handlers
               if isinstance(h, logging.FileHandler)]
    assert targets and all(_inside(path, runtime_paths.data_dir()) for path in targets)


def test_reporting_outputs_are_isolated(monkeypatch):
    class OfflineMT5:
        @staticmethod
        def initialize():
            return False
    monkeypatch.setattr(reporting, "mt5", OfflineMT5())
    state = {"date": "2099-01-01", "daily_pnl": 0.0, "closed_today": [],
             "open_trades": [], "consec_losses": {}, "pair_paused": {},
             "market_outlook": {}, "london_news_flag": False,
             "ny_news_flag": False}
    result = reporting.run(state)
    assert _inside(Path(result["report_path"]), runtime_paths.data_dir())
    assert reporting.REPORT_TXT.exists()
    assert reporting.EQUITY_CSV.exists()


def test_journal_and_cost_ledger_defaults_are_isolated(monkeypatch):
    monkeypatch.setattr(journal, "MT5_AVAILABLE", False)
    journal.log_event("test_isolation", {"ticket": 7000001})
    assert journal.JOURNAL_FILE.exists()
    record = {"ticket": 7000001, "gross_pnl": 1.0, "commission": 0.0,
              "swap": 0.0, "fee": 0.0, "net_pnl": 1.0}
    assert ledger.append_cost_record(record)
    assert ledger.LEDGER_FILE.exists()


def test_state_persistence_is_isolated():
    state = main_agent._fresh_state("2099-01-01")
    main_agent.save_state(state)
    assert main_agent.STATE_FILE.exists()
    assert _inside(main_agent.STATE_FILE, runtime_paths.data_dir())


def test_stress_test_import_does_not_call_mt5_or_write(monkeypatch):
    import importlib
    import src.stress_test as stress
    monkeypatch.setattr(stress, "connect_mt5", lambda: (_ for _ in ()).throw(
        AssertionError("must not execute during import")))
    importlib.reload(stress)
    assert callable(stress.main)
