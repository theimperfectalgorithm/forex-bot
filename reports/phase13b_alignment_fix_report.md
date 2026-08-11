# Critical Finding — NZDJPY/USDJPY Alignment Bug (supersedes phase10/10b/12/13)

**Experiments:** EXP-034 (bug + corrected re-check), EXP-035/036 (independent
re-run of the frozen gate through the fixed core module). Ledger:
`experiments/experiments.csv`.

## What happened

Part 1 of the portfolio-analysis request asked, as a mandatory first step,
to verify "whether there is any possibility of lookahead bias" in exactly
how the NZDJPY strategy uses USDJPY. Running that check surfaced this:

```
NZDJPY bars: 19,356   USDJPY bars: 19,352   common timestamps: 19,350
Fully aligned: False
```

`build_usdjpy_proxy()` / `signals_xmomentum()` in `phase10_jpy_london_ny.py`
(used unchanged by phase10, phase10b, and phase12) built the USDJPY proxy
as a plain numpy array and joined it to the traded pair by **raw array
position** (`usdjpy_move[i]`), not by timestamp. That's safe only if both
symbols have bar-for-bar identical timestamp sequences.

They don't. NZDJPY has 6 extra bars around 2023-12-25 (Christmas,
low-liquidity hours where NZDJPY's feed kept ticking but USDJPY's didn't).
From that point forward, position `i` in one array stopped meaning the
same calendar bar as position `i` in the other. I measured the damage
directly: **16,287 of 19,352 overlapping positions (84%) had mismatched
timestamps**, spanning essentially the entire post-Christmas-2023 dataset
— which is to say, nearly all of both the in-sample and out-of-sample
windows the original finding was built on.

## Fix

Rewrote the proxy construction to return timestamp-indexed pandas Series
and join via `.reindex()` (bars with no exact USDJPY match become NaN and
are skipped, rather than silently paired with the wrong bar). Applied the
fix in two places:
- `src/phase13b_alignment_fix_recheck.py` — standalone corrected re-check
- `src/phase10_jpy_london_ny.py` itself (`build_usdjpy_proxy` /
  `signals_xmomentum`) — so the bug can't recur if this module is reused
  by future research. Historical phase10/10b/12/13 console output predates
  this fix and is superseded by this report, not deleted.

Re-running the frozen phase12 gate through the now-fixed core module
independently reproduces the same conclusion (EXP-035/036), confirming
this wasn't specific to the phase13b script.

## Corrected numbers vs. the original (buggy) numbers

| Metric | Buggy (phase12) | Corrected |
|---|---|---|
| IS PF | 1.49 | 1.03–1.08 |
| OOS PF | 1.20 (profitable, +$7,731) | **0.87–0.94 (LOSING, -$1,590 to -$4,463)** |
| Permutation test | beats 100.0% of 1,000 shuffles | beats 95.9% of shuffles, but real PF itself is now <1.0 |
| CADJPY replication | PF 1.23 OOS ("PARTIALLY REPLICATED") | **PF 0.89 OOS (losing)** |
| Parameter plateau | all 18 neighbor cells IS PF ≥1.25 | all neighbor cells cluster near/below 1.0 OOS |
| Mechanical scorecard | 7/9 PASS → VALIDATED | 5/9 PASS → **FAILED** |

## What this invalidates

Everything computed on the buggy signal downstream of Part 1 in
`phase13_nzdjpy_portfolio_analysis.py` — the portfolio correlation matrix,
the JPY factor regression (R²=0.020, "POTENTIAL DIVERSIFICATION"), the
exploratory AMR-regime check, and the "PROMISING BUT INSUFFICIENT +
POTENTIAL DIVERSIFIER" dual classification (EXP-033) — describes a
strategy that, once correctly computed, does not have a standalone edge.
Analyzing the portfolio characteristics of a dead strategy isn't
meaningful, so none of those numbers should be relied on. They remain on
record in `reports/phase13_nzdjpy_portfolio_report.md` for audit purposes
only, explicitly flagged as superseded.

## Corrected classification

**Strategy: FAILED.** OOS is losing, cost/perturbation stress makes it
worse, and the out-of-family replication (CADJPY) also fails under the
corrected mechanism. This is not "regime-fading" the way AMR/the original
NZDJPY read was — it's a strategy that never had the standalone edge the
buggy signal appeared to show.

**Portfolio: MOOT.** A failed strategy has no diversification value worth
assessing; Parts 3–5 are not being re-run on the corrected-but-rejected
signal, since that would be optimizing/analyzing a candidate that should
instead be closed out, consistent with this project's own research-budget
discipline (prefer a clean negative result over further work on a dead
lead).

## Recommendation

Close the NZDJPY cross-asset-momentum hypothesis as **FAILED**, logged via
EXP-034/035/036, explicitly superseding EXP-030/031/032/033. Nothing here
was ever live, so no live/demo strategy needs to change. Per the research
philosophy that opened this whole line of work, this is a complete,
useful, negative result — the validation process caught a serious data
bug before any deployment decision was made on it, which is exactly what
it's for.
