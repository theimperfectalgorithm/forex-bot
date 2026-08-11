# Data Integrity & Alignment Audit — Full Codebase

**Date:** 2026-08-11
**Trigger:** the NZDJPY/USDJPY investigation found `build_usdjpy_proxy()`/
`signals_xmomentum()` in `src/phase10_jpy_london_ny.py` joined two symbols'
H1 bar arrays by raw array **position** instead of timestamp, silently
mispairing 84% of bars (16,287/19,352) for nearly the whole dataset (see
`reports/phase13b_alignment_fix_report.md`, EXP-034). This audit exists to
determine whether that was an isolated mistake or a systemic pattern.

**Scope:** every file in `src/` (47 scripts), `strategies/` (live
production code), `core/` (data loading / session / pair management),
`src/agents/` (live trading agent), and `mcp/` — audited by reading each
file's actual join logic, not by grepping for a fixed string. Three
independent passes were run (research scripts; live strategies + core;
agent + MCP), then I spot-checked a sample of the "SAFE" claims directly
against source (see Verification below) before writing this report.

## Bug class searched for

Two independently-sourced series (different symbol, different timeframe,
different session, different data source, or different timezone
representation) combined via:
- a shared loop index `i` into two separately-fetched arrays
- `min(len(a), len(b))` truncation followed by positional pairing
- `.iloc[]`/bare `.values`/`.to_numpy()` cross-indexing without a
  timestamp join
- `zip()` of two independently-sourced series
- forward-fill applied across a symbol boundary
- resample grids assumed identical across symbols without verification
- timezone-naive vs. timezone-aware timestamps mixed silently

## Result: one confirmed instance, already fixed. No others found.

### CONFIRMED AFFECTED (now fixed)

| File | Issue | Status |
|---|---|---|
| `src/phase10_jpy_london_ny.py` (`build_usdjpy_proxy`, `signals_xmomentum`) | Positional join of NZDJPY↔USDJPY H1 arrays | **Fixed** commit `77d90b6` — now returns/consumes timestamp-indexed `pd.Series` joined via `.reindex()` |
| `src/phase10b_xmo_refine.py`, `phase12_nzdjpy_validation_gate.py`, `phase13_nzdjpy_portfolio_analysis.py` | Consumed the buggy functions above by import | **Inherits the fix** automatically (no code of their own changed); historical console output from before the fix is superseded, not deleted — see `experiments/experiments.csv` EXP-034/035/036 |

No other file in the 60+ files reviewed exhibits this pattern.

### Repo-wide pattern that explains why this was isolated

Every other place in the codebase that combines two different symbols or
two different timeframes uses one of three genuinely timestamp-safe
techniques:

1. **`pd.merge_asof(..., on='time', direction='backward')`** —
   `h1_bollinger_walk_forward.py:align_to_h1`, `m15_walk_forward_search.py:align_to_m15`
2. **`.reindex(union).ffill().reindex(target)`** —
   `combined_strategy_backtest.py`, `fake_breakout_backtest.py`,
   `triple_ema_pullback_backtest.py`, `triple_ma_backtest.py`
3. **`np.searchsorted()` against an explicit `close_time` array** —
   `strategy_matrix_backtest.py:h4_regime_series` (the shared dependency
   for phase2/6/7/9's ARB/AMR signal functions), `h4_trend_pullback_backtest.py`,
   `ny_open_breakout_backtest.py`, `phase6_portfolio_model.py`,
   `prop_firm_backtest.py`, `revalidate_eurusd_live.py`, `the5ers_backtest.py`

`phase10_jpy_london_ny.py` was the one place that hand-rolled a bare
Python loop over `range(n)` with plain numpy arrays instead of using
pandas' index machinery — every other multi-series join in the repo went
through pandas/numpy operations that carry timestamps along for the ride.

**Live strategies (`strategies/*.py`) are categorically immune to this bug
class**: every strategy fetches exactly one symbol's own bars per
calculation. Multi-timeframe logic (H4 trend + H1 entry) reduces the H4
side to a **scalar** (trend integer / EMA float) before combining with H1
— there is no array-to-array positional join anywhere in production
strategy code. Cross-symbol state in the live agent (`src/agents/*.py`)
is dict-keyed by symbol/ticket string, not array-indexed, which is
inherently immune. `core/data_loader.py` and `core/health_monitor.py`
(the shared infrastructure other code should reuse) both already do
proper DatetimeIndex/ticket-key joins.

### POTENTIALLY AFFECTED

None found. All ambiguous cases were traced to their actual join
mechanism and resolved to SAFE or NOT APPLICABLE (see file-by-file table
below); nothing was left as an open question.

## Verification (not just agent-reported — spot-checked directly)

I independently re-read the join code for a sample of the "SAFE" claims
rather than trust the summary:
- `strategy_matrix_backtest.py:238-253` (`h4_regime_series`) — confirmed
  `np.searchsorted(h4_close_time.values, h1_close_time.values, side='right') - 1`,
  genuinely timestamp-based, and the dependency root for phase2/6/7/9.
- `h4_trend_pullback_backtest.py:180-185` (`get_h4_trend_at`) — confirmed
  `np.searchsorted(close_times, ts.to_datetime64(), side='right') - 1`.
- `h1_bollinger_walk_forward.py:147-154` and `m15_walk_forward_search.py:155-162`
  — confirmed genuine `pd.merge_asof(..., direction='backward')`.

## File-by-file classification

### `src/` (research/backtest scripts)

| File | Class | Reason |
|---|---|---|
| phase10_jpy_london_ny.py | CONFIRMED AFFECTED → FIXED | positional NZDJPY/USDJPY join, now `.reindex()` |
| phase10b_xmo_refine.py, phase12_nzdjpy_validation_gate.py, phase13_nzdjpy_portfolio_analysis.py | CONFIRMED AFFECTED → FIXED (inherited) | import the fixed functions |
| phase13b_alignment_fix_recheck.py | SAFE | this IS the fix's validation script |
| combined_strategy_backtest.py, fake_breakout_backtest.py, triple_ema_pullback_backtest.py, triple_ma_backtest.py | SAFE | reindex+ffill by timestamp |
| h1_bollinger_walk_forward.py, m15_walk_forward_search.py | SAFE | merge_asof by timestamp |
| h4_trend_pullback_backtest.py, ny_open_breakout_backtest.py, phase6_portfolio_model.py, prop_firm_backtest.py, revalidate_eurusd_live.py, the5ers_backtest.py, strategy_matrix_backtest.py, phase2_meanrev_arb_search.py | SAFE | searchsorted/`.loc`/`.at` by timestamp |
| multi_pair_backtest.py | SAFE | merge_asof per pair, pairs never cross-mixed |
| backtest.py, backtest_rsi.py, h1_atr_*.py, h1_backtest.py, h1_sl_tp_backtest.py, intraday_strategy.py, m15_atr_sl_tp_backtest.py, main.py, phase3_session_structure_search.py, phase3b_amr_jpy_refine.py, phase4_pro_eurusd_gbpusd.py, phase5_ict_backtest.py, phase7_exits_calendar_gold.py, phase8_monday_validation.py, phase9_5k_challenge_sim.py, phase11_pdh_pdl.py, rsi_optimise.py, sma_20_50_backtest.py, strategy.py, stress_test.py, system_check.py | NOT APPLICABLE | single symbol, no cross-series join |
| monte_carlo.py, walk_forward.py, research_ledger.py, reset_state.py | NOT APPLICABLE | operate on already-built trade/equity series, no raw symbol arrays |

### `strategies/` (live production)

All 21 files: **NOT APPLICABLE** — every strategy is single-symbol; H4/H1
combination is done via scalar reduction (trend int / indicator float),
never array-to-array positional join. Several are unimplemented stubs.

### `core/`

`data_loader.py` and `health_monitor.py`: **SAFE** (proper DatetimeIndex /
ticket-key joins — this is the pattern to keep using). All others: **NOT
APPLICABLE** (no bar-array joins at all).

### `src/agents/` and `mcp/`

All 11 files: **NOT APPLICABLE** or **SAFE** — live agent state is
dict-keyed by symbol/ticket, never positionally array-joined;
`mcp/backtest_engine.py`'s one cross-timeframe lookup uses `searchsorted`.

## Strategies/experiments potentially affected by this bug class

- **CONFIRMED AFFECTED:** NZDJPY cross-asset-momentum candidate only
  (phase10/10b/12/13). **Never live** on demo or the 5ers account — no
  production exposure. Already corrected and reclassified FAILED
  (EXP-034/035/036).
- **NOT APPLICABLE (no dependency on cross-symbol joins at all):** every
  live demo/5ers strategy — GBPJPY ARB, CADJPY ARB, XAUUSD ARB, GBPJPY
  AMR, EURJPY AMR, AUDJPY AMR, CADJPY AMR, GBPUSD Monday Drift. Each
  trades one symbol using only that symbol's own H1/H4/M15 bars.

## Fixes made this session

1. `src/phase10_jpy_london_ny.py` — root-cause fix, timestamp-safe reindex
   (already committed `77d90b6`, prior to this audit).
2. `src/alignment_utils.py` — new shared helper module: `safe_align()`
   (reindex + tz/monotonic/duplicate assertions + missing-fraction guard),
   `safe_asof_align()` (merge_asof wrapper for cross-timeframe joins),
   `assert_valid_index()`, and `log_cross_symbol_signal()` for recording
   `signal_timestamp` / `source_symbol_timestamp` / `target_symbol_timestamp`
   provenance on any future cross-symbol signal.
3. `tests/test_alignment_safety.py` — 7 regression tests (see below).

No strategy logic was modified. No new strategy research was started.

## Regression test summary

`tests/test_alignment_safety.py`, run via `python tests/test_alignment_safety.py`
(no pytest dependency in this repo):

```
PASS  test_1_missing_candles_timestamp_alignment
PASS  test_2_holiday_extra_candles_positional_join_fails_reindex_succeeds
PASS  test_3_multi_timeframe_alignment
PASS  test_4_timezone_mismatch_detected
PASS  test_5_duplicate_timestamps_rejected
PASS  test_6_missing_timestamps_not_silently_filled
PASS  test_7_shifted_timestamps_detected_via_missing_fraction

7/7 passed
```

Test 2 is the direct regression test for the NZDJPY bug shape: it builds
two series where one has extra "holiday" bars inserted mid-series (same
shape as NZDJPY's 6 extra Christmas-2023 bars), explicitly demonstrates
that a positional/truncated join produces wrong pairings after the insert
point, and then asserts `safe_align()` does **not** reproduce those
mismatches. If `safe_align()` (or any future replacement) regresses to
positional joining, this test fails.

These tests exercise `src/alignment_utils.py` directly — they don't (yet)
monkey-patch every legacy script's own hand-rolled `merge_asof`/
`searchsorted` call sites to prove each one individually, since those were
verified safe by direct code reading (see Verification section) and share
only three well-understood, already-correct patterns. The regression
surface that actually failed before (phase10's bare positional loop) is
what these tests target.

## Final integrity classification

# **RESEARCH PIPELINE TRUSTWORTHY**

One real instance of this bug class existed, in one file, affecting one
research candidate that was never live. It has been fixed at the root
(not worked around), independently re-verified two ways (a standalone
corrected re-check and a full re-run of the frozen validation gate
through the fixed module), and is now covered by an automated regression
test that specifically reproduces its failure shape. Every other
cross-symbol/cross-timeframe join in the codebase — research scripts,
live strategies, live agent, MCP layer — was read line-by-line and uses
one of three genuinely timestamp-safe join patterns. No strategy
currently live on demo or the 5ers account depends on any cross-symbol
join at all.

Recommend: safe to resume strategy research. New cross-symbol/cross-
timeframe work should import `src/alignment_utils.py` rather than
hand-roll a new join.
