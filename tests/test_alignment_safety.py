"""
Regression tests for the cross-symbol/cross-timeframe alignment bug class
uncovered by the NZDJPY/USDJPY investigation (EXP-034,
reports/phase13b_alignment_fix_report.md).

No pytest dependency in this repo -- run directly:
    python tests/test_alignment_safety.py

Each test function starts with test_ and returns nothing on success,
raises AssertionError on failure. main() runs them all and reports a
pass/fail summary, non-zero exit code on any failure (CI-friendly).
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from alignment_utils import safe_align, safe_asof_align, assert_valid_index, AlignmentError


def _h1_index(start, n, tz='UTC'):
    return pd.date_range(start, periods=n, freq='h', tz=tz)


# ── Test 1: two symbols with different missing candles ──────────────────────
def test_1_missing_candles_timestamp_alignment():
    idx_a = _h1_index('2024-01-01', 100)
    idx_b = idx_a.delete([10, 20, 30])          # symbol B missing 3 bars
    a = pd.Series(np.arange(100, dtype=float), index=idx_a)
    b = pd.Series(np.arange(97, dtype=float), index=idx_b)

    aligned = safe_align(idx_a, b, name='B')
    # bars 10/20/30 should be NaN (no match), everything else should equal
    # B's own value at that exact timestamp, not some shifted neighbor.
    assert aligned.isna().sum() == 3, "expected exactly 3 unmatched bars"
    for i in [0, 5, 50, 99]:
        if i not in (10, 20, 30):
            expected = b.loc[idx_a[i]]
            assert aligned.iloc[i] == expected, f"bar {i} misaligned: {aligned.iloc[i]} != {expected}"


# ── Test 2: extra holiday candles reproduce the exact NZDJPY bug shape ──────
def test_2_holiday_extra_candles_positional_join_fails_reindex_succeeds():
    idx_a = _h1_index('2024-01-01', 200)              # traded pair, e.g. NZDJPY
    # symbol B (proxy, e.g. USDJPY) has 5 EXTRA bars inserted mid-series
    # (simulating low-liquidity holiday bars the other symbol lacks) --
    # this shifts every later position by 5.
    extra = pd.date_range('2024-01-03 00:30', periods=5, freq='min', tz='UTC')
    idx_b = idx_a.union(extra).sort_values()
    a_val = np.arange(len(idx_a), dtype=float)
    b_val = np.arange(len(idx_b), dtype=float)         # B's OWN sequential values
    a = pd.Series(a_val, index=idx_a)
    b = pd.Series(b_val, index=idx_b)

    # The buggy pattern: positional join truncated to min length.
    n = min(len(a), len(b))
    positional_pairs = list(zip(a.to_numpy()[:n], b.to_numpy()[:n]))

    # The fixed pattern: timestamp join.
    aligned = safe_align(idx_a, b, name='B')
    timestamp_pairs = list(zip(a.to_numpy(), aligned.to_numpy()))

    # After the divergence point, positional pairing must diverge from the
    # correct timestamp-based pairing -- this is exactly the class of
    # silent corruption that hid in phase10 for 84% of the dataset.
    diverge_from = idx_a.get_indexer([extra[0]])[0]  # first index after the insert point
    mismatches = 0
    for i in range(diverge_from, n):
        bt = b.get(idx_a[i], np.nan)          # ground truth via a real timestamp lookup
        if not np.isnan(bt) and positional_pairs[i][1] != bt:
            mismatches += 1
    assert mismatches > 0, "test setup should reproduce positional misalignment after the insert point"

    # The timestamp-based join must NOT reproduce those mismatches.
    for i in range(diverge_from, n):
        bt = b.get(idx_a[i], np.nan)
        got = timestamp_pairs[i][1]
        if np.isnan(bt):
            assert np.isnan(got), f"bar {i}: expected NaN (no match), got {got}"
        else:
            assert got == bt, f"bar {i}: timestamp join mismatch {got} != {bt}"


# ── Test 3: different timeframe alignment (H4 trend onto H1 bars) ──────────
def test_3_multi_timeframe_alignment():
    h1_idx = _h1_index('2024-01-01', 48)
    h4_idx = pd.date_range('2024-01-01', periods=12, freq='4h', tz='UTC')
    h4_trend = pd.Series(np.arange(12, dtype=float), index=h4_idx)

    out = safe_asof_align(h1_idx, h4_trend, name='h4_trend')
    assert len(out) == len(h1_idx)
    # every H1 bar should carry the most recent H4 value at or before it
    for i, ts in enumerate(h1_idx):
        expected = h4_trend[h4_trend.index <= ts].iloc[-1]
        assert out[i] == expected, f"bar {i} ({ts}): got {out[i]}, expected {expected}"
    # bars before the first H4 bar should be NaN, not garbage
    early = pd.date_range('2023-12-31 20:00', periods=2, freq='h', tz='UTC')
    out2 = safe_asof_align(early, h4_trend, name='h4_trend')
    assert np.isnan(out2).all()


# ── Test 4: timezone differences must be caught, not silently coerced ──────
def test_4_timezone_mismatch_detected():
    idx_naive = pd.date_range('2024-01-01', periods=10, freq='h')             # tz-naive
    idx_utc = pd.date_range('2024-01-01', periods=10, freq='h', tz='UTC')      # tz-aware
    s_naive = pd.Series(np.arange(10, dtype=float), index=idx_naive)

    raised = False
    try:
        safe_align(idx_utc, s_naive, name='naive_source')
    except AlignmentError:
        raised = True
    assert raised, "safe_align must reject mixing tz-naive and tz-aware indexes"

    # Correctly normalized (both UTC) must succeed.
    s_utc = s_naive.tz_localize('UTC')
    aligned = safe_align(idx_utc, s_utc, name='utc_source')
    assert aligned.isna().sum() == 0


# ── Test 5: duplicate timestamps must be rejected explicitly ───────────────
def test_5_duplicate_timestamps_rejected():
    idx = _h1_index('2024-01-01', 10)
    idx_dup = idx.insert(5, idx[5])            # duplicate bar 5
    s = pd.Series(np.arange(11, dtype=float), index=idx_dup)

    raised = False
    try:
        assert_valid_index(s.to_frame(), 'dup_series')
    except AlignmentError:
        raised = True
    assert raised, "assert_valid_index must reject duplicate timestamps"


# ── Test 6: missing timestamps must not be silently forward-filled ─────────
def test_6_missing_timestamps_not_silently_filled():
    idx_a = _h1_index('2024-01-01', 50)
    idx_b = idx_a.delete([25])                  # one gap
    b = pd.Series(np.arange(49, dtype=float), index=idx_b)

    aligned = safe_align(idx_a, b, name='B')
    # the gap bar must be NaN, NOT forward-filled from bar 24's value --
    # safe_align must never call .ffill() implicitly.
    assert np.isnan(aligned.iloc[25]), "gap bar was silently filled instead of left NaN"
    assert aligned.iloc[24] == b.iloc[24], "sanity: neighboring bar should still be correct"


# ── Test 7: deliberately shifted timestamps must be detectable ─────────────
def test_7_shifted_timestamps_detected_via_missing_fraction():
    idx_a = _h1_index('2024-01-01', 100)
    # symbol B's clock is shifted by 30 minutes for its entire history --
    # every timestamp is now off-grid relative to A.
    idx_b = idx_a + pd.Timedelta(minutes=30)
    b = pd.Series(np.arange(100, dtype=float), index=idx_b)

    aligned = safe_align(idx_a, b, name='shifted_B')
    # with a full-series shift, essentially nothing lines up on the hour grid
    assert aligned.isna().mean() > 0.99, "shifted series should produce near-total misalignment, not a silent partial match"

    raised = False
    try:
        safe_align(idx_a, b, name='shifted_B', max_missing_frac=0.5)
    except AlignmentError:
        raised = True
    assert raised, "safe_align must raise when missing fraction exceeds the caller's threshold"


TESTS = [v for k, v in sorted(globals().items()) if k.startswith('test_')]


def main():
    passed, failed = 0, []
    for fn in TESTS:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            failed.append(fn.__name__)
        except Exception as e:
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
            failed.append(fn.__name__)
    print(f"\n{passed}/{len(TESTS)} passed")
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()
