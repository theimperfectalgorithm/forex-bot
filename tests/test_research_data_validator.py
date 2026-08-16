"""
Regression tests for src/research_data_validator.py.

Includes a reconstruction of the exact incident that motivated this module
(reports/current_6_strategy_revalidation.csv, commit 6fd93a3): a hand-composed
CSV with an unquoted comma inside a text field, which silently corrupted 4 of
7 rows under pandas' default/on_bad_lines='warn' parsers. Every test here
either proves the validator catches a specific corruption class, or proves a
proper (csv.DictWriter-based) round trip preserves data exactly.
"""
import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from research_data_validator import (  # noqa: E402
    ResearchDataError,
    ValidationReport,
    validate_allowed_values,
    validate_column_count_consistency,
    validate_date_range,
    validate_datetime_columns,
    validate_lifecycle_pairing,
    validate_no_duplicate_key,
    validate_no_duplicate_rows,
    validate_no_missing_values,
    validate_no_unexpected_columns,
    validate_numeric_columns,
    validate_required_columns,
    validate_row_count,
    validate_roundtrip,
)


@pytest.fixture
def tmp_csv(tmp_path):
    def _write(text: str, name: str = 'test.csv') -> Path:
        p = tmp_path / name
        p.write_text(text, encoding='utf-8')
        return p
    return _write


# ---------------------------------------------------------------------------
# The exact incident: unquoted comma in a text field
# ---------------------------------------------------------------------------

def test_reproduces_the_original_incident(tmp_csv):
    """Reconstructs the exact malformed row shape from
    reports/current_6_strategy_revalidation.csv: a value like
    'STABLE (broad plateau, both z_thr and sl_mult)' written WITHOUT
    quoting, which splits into two fields and shifts everything after it."""
    text = (
        "strategy,pf,parameter_stability_status,final_note\n"
        "GBPJPY_AMR,1.426,STABLE (broad plateau, both z_thr and sl_mult),PASS\n"
    )
    p = tmp_csv(text)
    report = ValidationReport(path=str(p))
    with pytest.raises(ResearchDataError, match="column-count mismatch"):
        validate_column_count_consistency(p, report)


def test_properly_quoted_comma_passes(tmp_csv):
    """The same text field, correctly quoted (as csv.writer/DictWriter
    would produce automatically), must NOT raise."""
    text = (
        'strategy,pf,parameter_stability_status,final_note\n'
        'GBPJPY_AMR,1.426,"STABLE (broad plateau, both z_thr and sl_mult)",PASS\n'
    )
    p = tmp_csv(text)
    report = ValidationReport(path=str(p))
    validate_column_count_consistency(p, report)  # must not raise
    assert 'column_count_consistency' in report.checks_passed


# ---------------------------------------------------------------------------
# Round-trip proof: commas/quotes/negatives/timestamps/text cannot corrupt
# a dataset written through a proper CSV library
# ---------------------------------------------------------------------------

def test_roundtrip_preserves_commas_quotes_negatives_timestamps(tmp_path):
    fieldnames = ['trade_id', 'strategy', 'entry_time', 'R', 'notes']
    rows = [
        {
            'trade_id': '588709831',
            'strategy': 'CADJPY_ARB',
            'entry_time': '2026-08-13T05:00:05+00:00',
            'R': '-0.66',
            'notes': 'SL exit, spread was 0.8 pips, note: "tight" fill, regime=HIGH, vol>avg',
        },
        {
            'trade_id': '578493052',
            'strategy': 'AUDJPY_AMR',
            'entry_time': '2026-07-20T21:15:05+00:00',
            'R': '-1.02',
            'notes': 'Contains, multiple, commas, and a "quoted phrase", plus -12.5 negative numbers',
        },
        {
            'trade_id': '1',
            'strategy': 'GBPUSD_MONDAY',
            'entry_time': '2026-08-09T22:00:14+00:00',
            'R': '0.11',
            'notes': '',
        },
    ]
    out = tmp_path / 'roundtrip.csv'
    validate_roundtrip(rows, out, fieldnames)  # must not raise

    # Independently re-verify with the validator's own reader, not just trust
    # validate_roundtrip's internal check.
    with open(out, newline='', encoding='utf-8') as f:
        read_back = list(csv.DictReader(f))
    assert len(read_back) == len(rows)
    for original, got in zip(rows, read_back):
        for k in fieldnames:
            assert got[k] == original[k], f"field {k!r} corrupted: {original[k]!r} -> {got[k]!r}"


def test_roundtrip_detects_corruption_if_bypassing_csv_writer(tmp_path):
    """Negative control: hand-writing (not using csv.DictWriter) the same
    comma-bearing text WITHOUT quoting must be caught by
    validate_column_count_consistency -- proving the round-trip test isn't
    vacuously passing."""
    p = tmp_path / 'hand_written.csv'
    p.write_text(
        "trade_id,strategy,notes\n"
        "1,AUDJPY_AMR,Contains, multiple, unquoted, commas\n",
        encoding='utf-8')
    report = ValidationReport(path=str(p))
    with pytest.raises(ResearchDataError):
        validate_column_count_consistency(p, report)


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_required_columns_missing_raises(tmp_csv):
    p = tmp_csv("a,b\n1,2\n")
    report = ValidationReport(path=str(p))
    with pytest.raises(ResearchDataError, match="missing required columns"):
        validate_required_columns(p, {'a', 'b', 'c'}, report)


def test_required_columns_present_passes(tmp_csv):
    p = tmp_csv("a,b,c\n1,2,3\n")
    report = ValidationReport(path=str(p))
    validate_required_columns(p, {'a', 'b'}, report)


def test_unexpected_columns_raises(tmp_csv):
    p = tmp_csv("a,b,unexpected\n1,2,3\n")
    report = ValidationReport(path=str(p))
    with pytest.raises(ResearchDataError, match="unexpected columns"):
        validate_no_unexpected_columns(p, {'a', 'b'}, report)


# ---------------------------------------------------------------------------
# Row count: exact (frozen artifact) vs range (growing artifact)
# ---------------------------------------------------------------------------

def test_row_count_exact_mismatch_raises(tmp_csv):
    p = tmp_csv("a\n1\n2\n3\n")
    with pytest.raises(ResearchDataError, match="expected exactly 5"):
        validate_row_count(p, expected=5)


def test_row_count_range_allows_growth(tmp_csv):
    p = tmp_csv("a\n1\n2\n3\n")
    n = validate_row_count(p, min_rows=1, max_rows=100)
    assert n == 3


def test_row_count_below_minimum_raises(tmp_csv):
    p = tmp_csv("a\n1\n")
    with pytest.raises(ResearchDataError, match="at least 5"):
        validate_row_count(p, min_rows=5)


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def test_duplicate_rows_detected(tmp_csv):
    p = tmp_csv("a,b\n1,2\n1,2\n3,4\n")
    report = ValidationReport(path=str(p))
    with pytest.raises(ResearchDataError, match="fully-duplicate"):
        validate_no_duplicate_rows(p, report)


def test_duplicate_key_strict_detects_ticket_appearing_thrice(tmp_csv):
    p = tmp_csv("ticket,status\n1,OPEN\n1,CLOSED\n1,CLOSED\n")
    with pytest.raises(ResearchDataError, match="repeated more than"):
        validate_no_duplicate_key(p, 'ticket', allow_repeats=2)


def test_duplicate_key_allows_open_closed_pairing(tmp_csv):
    p = tmp_csv("ticket,status\n1,OPEN\n1,CLOSED\n2,OPEN\n2,CLOSED\n")
    validate_no_duplicate_key(p, 'ticket', allow_repeats=2)  # must not raise


# ---------------------------------------------------------------------------
# Numeric / datetime / missing-value validation
# ---------------------------------------------------------------------------

def test_numeric_column_rejects_non_numeric(tmp_csv):
    p = tmp_csv("r\n1.5\nnot_a_number\n-2.3\n")
    with pytest.raises(ResearchDataError, match="non-numeric"):
        validate_numeric_columns(p, {'r'})


def test_numeric_column_allows_sentinel(tmp_csv):
    p = tmp_csv("r\n1.5\nNOT_AVAILABLE\n-2.3\n")
    validate_numeric_columns(p, {'r'}, allow_missing_sentinel='NOT_AVAILABLE')


def test_datetime_column_rejects_malformed(tmp_csv):
    p = tmp_csv("t\n2026-08-13T05:00:05+00:00\nnot-a-date\n")
    with pytest.raises(ResearchDataError, match="unparseable datetime"):
        validate_datetime_columns(p, {'t'})


def test_missing_values_detected(tmp_csv):
    p = tmp_csv("strategy\nAUDJPY_AMR\n\nCADJPY_ARB\n")
    with pytest.raises(ResearchDataError, match="missing value"):
        validate_no_missing_values(p, {'strategy'})


# ---------------------------------------------------------------------------
# Strategy-name / allowed-value and date-range validation
# ---------------------------------------------------------------------------

def test_allowed_values_rejects_unknown_strategy(tmp_csv):
    p = tmp_csv("strategy\nAUDJPY_AMR\nUNKNOWN_STRAT\n")
    with pytest.raises(ResearchDataError, match="outside the allowed set"):
        validate_allowed_values(p, 'strategy', {'AUDJPY_AMR', 'CADJPY_ARB'})


def test_date_range_rejects_out_of_range(tmp_csv):
    from datetime import datetime, timezone
    p = tmp_csv("entry_time\n2026-06-01T00:00:00+00:00\n2026-08-01T00:00:00+00:00\n")
    with pytest.raises(ResearchDataError, match="before allowed minimum"):
        validate_date_range(p, 'entry_time', min_date=datetime(2026, 7, 31, tzinfo=timezone.utc))


# ---------------------------------------------------------------------------
# Lifecycle pairing (production trade export shape)
# ---------------------------------------------------------------------------

def test_lifecycle_pairing_passes_for_clean_export(tmp_csv):
    p = tmp_csv("trade_id,status\n1,OPEN\n1,CLOSED\n2,OPEN\n2,CLOSED\n")
    validate_lifecycle_pairing(p, 'trade_id', 'status')


def test_lifecycle_pairing_detects_missing_closed_row(tmp_csv):
    p = tmp_csv("trade_id,status\n1,OPEN\n1,CLOSED\n2,OPEN\n")  # ticket 2 never closed
    with pytest.raises(ResearchDataError, match="irregular lifecycle"):
        validate_lifecycle_pairing(p, 'trade_id', 'status')
