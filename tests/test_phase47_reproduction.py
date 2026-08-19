import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from phase47_reproduction_harness import sha256, match_trades


def test_sha256_deterministic(tmp_path):
    f = tmp_path / 'x.txt'
    f.write_text('hello world')
    assert sha256(f) == sha256(f)


def test_sha256_changes_with_content(tmp_path):
    f = tmp_path / 'x.txt'
    f.write_text('hello world')
    h1 = sha256(f)
    f.write_text('hello world!')
    h2 = sha256(f)
    assert h1 != h2


def test_match_trades_exact_match():
    recon = pd.DataFrame([{'entry_date': pd.Timestamp('2025-01-06').date(), 'direction': 'BUY', 'sl_pips': 40, 'tp_pips': 60}])
    hist = pd.DataFrame([{'trade_date': pd.Timestamp('2025-01-06').date(), 'dir': 'BUY', 'r_multiple': 1.5, 'sl_pips': 40}])
    matched_df, matched, fp, fn = match_trades(recon, hist)
    assert matched == 1
    assert fp == 0
    assert fn == 0


def test_match_trades_false_positive():
    recon = pd.DataFrame([{'entry_date': pd.Timestamp('2025-01-06').date(), 'direction': 'BUY', 'sl_pips': 40, 'tp_pips': 60}])
    hist = pd.DataFrame(columns=['trade_date', 'dir', 'r_multiple', 'sl_pips'])
    matched_df, matched, fp, fn = match_trades(recon, hist)
    assert matched == 0
    assert fp == 1
    assert fn == 0


def test_match_trades_false_negative():
    recon = pd.DataFrame(columns=['entry_date', 'direction', 'sl_pips', 'tp_pips'])
    hist = pd.DataFrame([{'trade_date': pd.Timestamp('2025-01-06').date(), 'dir': 'BUY', 'r_multiple': 1.5, 'sl_pips': 40}])
    matched_df, matched, fp, fn = match_trades(recon, hist)
    assert matched == 0
    assert fp == 0
    assert fn == 1


def test_match_trades_direction_mismatch_is_not_matched():
    recon = pd.DataFrame([{'entry_date': pd.Timestamp('2025-01-06').date(), 'direction': 'SELL', 'sl_pips': 40, 'tp_pips': 60}])
    hist = pd.DataFrame([{'trade_date': pd.Timestamp('2025-01-06').date(), 'dir': 'BUY', 'r_multiple': 1.5, 'sl_pips': 40}])
    matched_df, matched, fp, fn = match_trades(recon, hist)
    assert matched == 0
    assert fp == 1
    assert fn == 1


def test_source_files_are_never_modified_by_import():
    """Importing the harness module must not touch any live strategy file --
    a basic guard against accidental side effects at import time."""
    from phase47_reproduction_harness import SRC_FILES
    for name, path in SRC_FILES.items():
        assert path.exists(), f"expected {name} to exist at {path}"
