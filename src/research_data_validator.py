"""
Research data validator -- fails loudly, never silently drops/shifts/coerces.

Built after tracing a real data-corruption incident: reports/
current_6_strategy_revalidation.csv (commit 6fd93a3) was hand-composed text,
not written through a CSV-serialization library, so a comma inside
`parameter_stability_status` ("STABLE (broad plateau, both z_thr and
sl_mult)") was never auto-quoted. pandas' default C parser raised a
tokenization error; the python engine with on_bad_lines='warn' silently
DROPPED 4 of 7 rows (see git history / reports/research_data_integrity_policy.md
for the full incident writeup). No round-trip validation existed at the time
to catch this before it propagated into a downstream analysis (phase27/28/29).

This module is the fix: every CSV a research script reads or writes should
pass through here first. Every check either passes or raises
ResearchDataError -- there is no silent-continue path anywhere in this file.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable


class ResearchDataError(Exception):
    """Raised on any data-integrity failure. Always fatal -- callers must
    not catch this and continue with partial data."""


@dataclass
class ValidationReport:
    path: str
    checks_run: list = field(default_factory=list)
    checks_passed: list = field(default_factory=list)

    def record(self, name: str):
        self.checks_run.append(name)
        self.checks_passed.append(name)

    def summary(self) -> str:
        return f"{self.path}: {len(self.checks_passed)}/{len(self.checks_run)} checks passed"


def _read_raw_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    """Reads with the stdlib csv module (which raises/records ragged rows
    explicitly, unlike pandas' C engine which can silently misparse) and
    returns (header, data_rows) with NO row dropped or realigned. Column-
    count mismatches are returned as-is so validate_column_count_consistency
    can report them -- this function never fixes anything."""
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ResearchDataError(f"{path}: file is empty")
    return rows[0], rows[1:]


def validate_column_count_consistency(path: Path, report: ValidationReport) -> None:
    """FAILS LOUDLY if any data row has a different field count than the
    header. This is the exact check that would have caught the
    current_6_strategy_revalidation.csv incident immediately -- that file
    has 4 rows with 30 fields against a 29-field header."""
    header, data_rows = _read_raw_rows(path)
    bad = [(i + 2, len(row)) for i, row in enumerate(data_rows) if len(row) != len(header)]
    if bad:
        detail = ', '.join(f"line {ln} has {n} fields" for ln, n in bad[:10])
        raise ResearchDataError(
            f"{path}: column-count mismatch on {len(bad)} row(s) (expected {len(header)} fields) -- "
            f"{detail}. This is very likely an unquoted comma/quote inside a text field. "
            f"DO NOT read this file with a parser that silently drops or realigns bad rows.")
    report.record('column_count_consistency')


def validate_required_columns(path: Path, required: set[str], report: ValidationReport) -> None:
    header, _ = _read_raw_rows(path)
    missing = required - set(header)
    if missing:
        raise ResearchDataError(f"{path}: missing required columns: {sorted(missing)}")
    report.record('required_columns')


def validate_no_unexpected_columns(path: Path, allowed: set[str], report: ValidationReport) -> None:
    header, _ = _read_raw_rows(path)
    unexpected = set(header) - allowed
    if unexpected:
        raise ResearchDataError(f"{path}: unexpected columns not in schema: {sorted(unexpected)}")
    report.record('no_unexpected_columns')


def validate_row_count(path: Path, expected: int | None = None, min_rows: int | None = None,
                        max_rows: int | None = None, report: ValidationReport = None) -> int:
    """expected is an exact match (use for frozen/versioned artifacts).
    min_rows/max_rows are a range (use for growing artifacts like a live
    trade export, per the structural-vs-snapshot distinction in
    reports/research_data_integrity_policy.md)."""
    _, data_rows = _read_raw_rows(path)
    n = len(data_rows)
    if expected is not None and n != expected:
        raise ResearchDataError(f"{path}: expected exactly {expected} rows, got {n}")
    if min_rows is not None and n < min_rows:
        raise ResearchDataError(f"{path}: expected at least {min_rows} rows, got {n}")
    if max_rows is not None and n > max_rows:
        raise ResearchDataError(f"{path}: expected at most {max_rows} rows, got {n}")
    if report:
        report.record('row_count')
    return n


def validate_no_duplicate_rows(path: Path, report: ValidationReport) -> None:
    header, data_rows = _read_raw_rows(path)
    seen = set()
    dupes = []
    for i, row in enumerate(data_rows):
        key = tuple(row)
        if key in seen:
            dupes.append(i + 2)
        seen.add(key)
    if dupes:
        raise ResearchDataError(f"{path}: {len(dupes)} fully-duplicate row(s) at lines {dupes[:10]}")
    report.record('no_duplicate_rows')


def validate_no_duplicate_key(path: Path, key_column: str, allow_repeats: int | None = None,
                               report: ValidationReport = None) -> None:
    """Duplicate-ticket/trade-id detection. allow_repeats=2 permits the
    OPEN+CLOSED lifecycle pairing used in reports/5ers_trade_export.csv
    (each real trade legitimately appears twice); pass None to require
    strict uniqueness (e.g. for a CLOSED-only export)."""
    header, data_rows = _read_raw_rows(path)
    if key_column not in header:
        raise ResearchDataError(f"{path}: key column '{key_column}' not found in header {header}")
    idx = header.index(key_column)
    counts: dict[str, int] = {}
    for row in data_rows:
        counts[row[idx]] = counts.get(row[idx], 0) + 1
    limit = allow_repeats if allow_repeats is not None else 1
    bad = {k: v for k, v in counts.items() if v > limit}
    if bad:
        raise ResearchDataError(
            f"{path}: key column '{key_column}' has values repeated more than "
            f"{limit} time(s): {dict(list(bad.items())[:10])}")
    if report:
        report.record('no_duplicate_key')


def validate_numeric_columns(path: Path, columns: set[str], allow_missing_sentinel: str | None = None,
                              report: ValidationReport = None) -> None:
    """allow_missing_sentinel lets a literal string (e.g. 'NOT_AVAILABLE')
    pass without failing -- everything else must parse as float."""
    header, data_rows = _read_raw_rows(path)
    idxs = {c: header.index(c) for c in columns if c in header}
    missing_cols = columns - set(idxs)
    if missing_cols:
        raise ResearchDataError(f"{path}: numeric columns not found: {sorted(missing_cols)}")
    bad = []
    for i, row in enumerate(data_rows):
        for col, idx in idxs.items():
            v = row[idx]
            if v == '' or (allow_missing_sentinel and v == allow_missing_sentinel):
                continue
            try:
                float(v)
            except ValueError:
                bad.append((i + 2, col, v))
    if bad:
        detail = ', '.join(f"line {ln} col={c!r} value={v!r}" for ln, c, v in bad[:10])
        raise ResearchDataError(f"{path}: {len(bad)} non-numeric value(s) in numeric column(s): {detail}")
    if report:
        report.record('numeric_columns')


def validate_datetime_columns(path: Path, columns: set[str], allow_missing_sentinel: str | None = None,
                               report: ValidationReport = None) -> None:
    header, data_rows = _read_raw_rows(path)
    idxs = {c: header.index(c) for c in columns if c in header}
    missing_cols = columns - set(idxs)
    if missing_cols:
        raise ResearchDataError(f"{path}: datetime columns not found: {sorted(missing_cols)}")
    bad = []
    for i, row in enumerate(data_rows):
        for col, idx in idxs.items():
            v = row[idx]
            if v == '' or (allow_missing_sentinel and v == allow_missing_sentinel):
                continue
            try:
                datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                bad.append((i + 2, col, v))
    if bad:
        detail = ', '.join(f"line {ln} col={c!r} value={v!r}" for ln, c, v in bad[:10])
        raise ResearchDataError(f"{path}: {len(bad)} unparseable datetime value(s): {detail}")
    if report:
        report.record('datetime_columns')


def validate_no_missing_values(path: Path, columns: set[str], allow_sentinel: str | None = None,
                                report: ValidationReport = None) -> None:
    header, data_rows = _read_raw_rows(path)
    idxs = {c: header.index(c) for c in columns if c in header}
    missing_cols = columns - set(idxs)
    if missing_cols:
        raise ResearchDataError(f"{path}: columns not found: {sorted(missing_cols)}")
    bad = []
    for i, row in enumerate(data_rows):
        for col, idx in idxs.items():
            v = row[idx] if idx < len(row) else ''  # a short/blank row is a missing value, not a crash
            if v == '' and v != allow_sentinel:
                bad.append((i + 2, col))
    if bad:
        raise ResearchDataError(f"{path}: {len(bad)} missing value(s) in required column(s): {bad[:10]}")
    if report:
        report.record('no_missing_values')


def validate_allowed_values(path: Path, column: str, allowed: set[str], report: ValidationReport = None) -> None:
    """Strategy-name validation and similar closed-set checks."""
    header, data_rows = _read_raw_rows(path)
    if column not in header:
        raise ResearchDataError(f"{path}: column '{column}' not found")
    idx = header.index(column)
    seen_bad = {row[idx] for row in data_rows if row[idx] not in allowed}
    if seen_bad:
        raise ResearchDataError(f"{path}: column '{column}' has values outside the allowed set: {sorted(seen_bad)}")
    if report:
        report.record('allowed_values')


def validate_date_range(path: Path, column: str, min_date: datetime | None = None,
                         max_date: datetime | None = None, report: ValidationReport = None) -> None:
    header, data_rows = _read_raw_rows(path)
    if column not in header:
        raise ResearchDataError(f"{path}: date-range column '{column}' not found")
    idx = header.index(column)
    for i, row in enumerate(data_rows):
        v = row[idx]
        if not v or v == 'NOT_AVAILABLE':
            continue
        try:
            dt = datetime.fromisoformat(v.replace('Z', '+00:00'))
        except ValueError:
            raise ResearchDataError(f"{path}: unparseable date at line {i + 2} in '{column}': {v!r}")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=min_date.tzinfo if min_date else None)
        if min_date and dt < min_date:
            raise ResearchDataError(f"{path}: line {i + 2} date {v} is before allowed minimum {min_date}")
        if max_date and dt > max_date:
            raise ResearchDataError(f"{path}: line {i + 2} date {v} is after allowed maximum {max_date}")
    if report:
        report.record('date_range')


def validate_lifecycle_pairing(path: Path, key_column: str, status_column: str,
                                statuses: tuple[str, str] = ('OPEN', 'CLOSED'),
                                report: ValidationReport = None) -> None:
    """Production trade-export-specific: every key (ticket) should appear
    exactly once per lifecycle status (e.g. one OPEN row + one CLOSED row),
    never zero or more than one of either."""
    header, data_rows = _read_raw_rows(path)
    for col in (key_column, status_column):
        if col not in header:
            raise ResearchDataError(f"{path}: column '{col}' not found")
    kidx, sidx = header.index(key_column), header.index(status_column)
    by_key: dict[str, dict[str, int]] = {}
    for row in data_rows:
        by_key.setdefault(row[kidx], {}).setdefault(row[sidx], 0)
        by_key[row[kidx]][row[sidx]] += 1
    bad = {k: counts for k, counts in by_key.items()
           if any(counts.get(s, 0) != 1 for s in statuses)}
    if bad:
        raise ResearchDataError(
            f"{path}: {len(bad)} key(s) with irregular lifecycle status counts "
            f"(expected exactly one of each of {statuses}): {dict(list(bad.items())[:10])}")
    if report:
        report.record('lifecycle_pairing')


def validate_roundtrip(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    """Writes `rows` to `path` via csv.DictWriter (which auto-quotes),
    reads them back, and asserts row count / column set / every value is
    preserved exactly. Raises ResearchDataError on any mismatch. This is
    the check that proves 'commas inside text fields cannot corrupt the
    dataset' -- see tests/test_research_data_validator.py."""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    header, data_rows = _read_raw_rows(path)
    if header != fieldnames:
        raise ResearchDataError(f"round-trip failed: header mismatch {header} != {fieldnames}")
    if len(data_rows) != len(rows):
        raise ResearchDataError(f"round-trip failed: wrote {len(rows)} rows, read back {len(data_rows)}")
    for i, (original, read_back) in enumerate(zip(rows, data_rows)):
        expected = [str(original[c]) for c in fieldnames]
        if expected != read_back:
            raise ResearchDataError(f"round-trip failed at row {i}: wrote {expected}, read back {read_back}")


def run_full_validation(path: Path, checks: list[Callable[[Path, ValidationReport], None]]) -> ValidationReport:
    """Runs a list of (path, report) -> None validator functions in order.
    Stops at the FIRST failure (ResearchDataError propagates immediately --
    this function does not catch or continue past a failed check)."""
    report = ValidationReport(path=str(path))
    for check in checks:
        check(path, report)
    return report
