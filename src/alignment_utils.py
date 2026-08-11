"""
Safe cross-symbol / cross-timeframe alignment utilities.

Created 2026-08-11 as a direct result of the NZDJPY/USDJPY alignment bug
(see reports/phase13b_alignment_fix_report.md, EXP-034): two independently
fetched H1 bar series were joined by raw array position instead of by
timestamp, silently mispairing 84% of bars for nearly the entire dataset.

RULE: any time two different symbols' or two different timeframes'
series are combined, the join MUST go through timestamps -- never through
a shared loop index `i` or truncated positional slicing. Use the helpers
below rather than hand-rolling a reindex/searchsorted/merge_asof each time.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


class AlignmentError(ValueError):
    pass


def assert_valid_index(df_or_series, name: str, require_tz: bool = True) -> None:
    """Fail loudly on the properties that make positional joins unsafe:
    non-monotonic timestamps, duplicate timestamps, and (optionally)
    timezone-naive timestamps mixed into a pipeline that assumes UTC."""
    idx = df_or_series.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise AlignmentError(f"{name}: index is not a DatetimeIndex ({type(idx)})")
    if not idx.is_monotonic_increasing:
        raise AlignmentError(f"{name}: timestamps are not monotonic increasing")
    dupes = idx[idx.duplicated()]
    if len(dupes):
        raise AlignmentError(f"{name}: {len(dupes)} duplicate timestamps, e.g. {dupes[:3].tolist()}")
    if require_tz and idx.tz is None:
        raise AlignmentError(f"{name}: index is timezone-naive; this pipeline requires explicit tz")


def assert_same_tz(a: pd.DatetimeIndex, b: pd.DatetimeIndex, name_a: str, name_b: str) -> None:
    if a.tz != b.tz:
        raise AlignmentError(f"timezone mismatch: {name_a} tz={a.tz} vs {name_b} tz={b.tz}")


def safe_align(target_index: pd.DatetimeIndex, source: pd.Series, name: str = "series",
               max_missing_frac: float = 1.0) -> pd.Series:
    """Reindex `source` onto `target_index` by TIMESTAMP (never by position).

    Bars in target_index with no exact timestamp match in `source` become
    NaN -- this is intentional fail-safe behavior (skip, don't guess) and
    mirrors the fix applied to phase10_jpy_london_ny.py's proxy join.

    Raises if more than `max_missing_frac` of the resulting series is NaN,
    since a very high missing-rate after reindexing usually means the two
    series don't actually share a comparable timestamp grid (e.g. one is
    tz-naive, one is tz-aware, or they're on different timeframes and need
    merge_asof instead of reindex).
    """
    assert_valid_index(source.to_frame() if isinstance(source, pd.Series) else source, name,
                        require_tz=(target_index.tz is not None))
    if target_index.tz != source.index.tz:
        raise AlignmentError(
            f"safe_align({name}): target tz={target_index.tz} vs source tz={source.index.tz} -- "
            f"normalize timezones before aligning"
        )
    aligned = source.reindex(target_index)
    missing_frac = aligned.isna().mean()
    if missing_frac > max_missing_frac:
        raise AlignmentError(
            f"safe_align({name}): {missing_frac:.1%} of bars have no timestamp match "
            f"(threshold {max_missing_frac:.1%}) -- check timeframe/timezone/date-range compatibility"
        )
    return aligned


def safe_asof_align(target_index: pd.DatetimeIndex, source: pd.Series, name: str = "series",
                     direction: str = "backward") -> np.ndarray:
    """Bring the latest known value of a LOWER-frequency series (e.g. H4
    trend) onto a HIGHER-frequency index (e.g. H1/M15 bars) via merge_asof,
    which is timestamp-based and tolerant of differing bar counts. Prefer
    this over reindex() when the two series are on different timeframes."""
    assert_valid_index(source.to_frame(), name, require_tz=(target_index.tz is not None))
    left = pd.DataFrame({'time': target_index})
    right = source.dropna().rename('val').reset_index()
    right.columns = ['time', 'val']
    merged = pd.merge_asof(left, right, on='time', direction=direction)
    return merged['val'].values


def log_cross_symbol_signal(signal_timestamp, source_symbol: str, source_symbol_timestamp,
                             target_symbol: str, target_symbol_timestamp) -> dict:
    """Return a record for any cross-symbol signal so future audits can
    verify alignment without re-deriving it. Callers should append these
    to a list/CSV alongside their normal trade candidates when a signal
    on one symbol depends on another symbol's data (e.g. the NZDJPY/
    USDJPY momentum signal)."""
    return {
        'signal_timestamp': signal_timestamp,
        'source_symbol': source_symbol,
        'source_symbol_timestamp': source_symbol_timestamp,
        'target_symbol': target_symbol,
        'target_symbol_timestamp': target_symbol_timestamp,
        'stale': source_symbol_timestamp != target_symbol_timestamp,
    }
