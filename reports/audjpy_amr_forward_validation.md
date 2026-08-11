# AUDJPY AMR BUY-only — Prospective Forward Validation Tracker

**Status as of this report: DATA COLLECTION JUST STARTED. Zero prospective observations recorded.**
**This is not a research report — it is a tracking log, updated only at meaningful checkpoints.**

**Experiments:** EXP-090, EXP-091, `experiments/experiments.csv`.
**Tracker script:** `src/amr_forward_tracker.py`. **Immutable trade log:** `data/audjpy_amr_forward_trades.csv` (append-only, currently empty). **State file:** `data/audjpy_amr_forward_state.json`. **Audit log:** `data/audjpy_amr_forward_audit_log.jsonl`.

No strategy was modified. No parameter was searched. This phase is data
collection only, per explicit instruction — no new hypotheses, no
optimization, no filter changes.

## Frozen strategy specification (permanent, do not modify)

- **Instrument:** AUDJPY.
- **Base:** existing live AUDJPY AMR — `signals_amr_v(z_thr=2.0,
  sl_mult=1.5, end_hour=4)`, spread 2.0 pips, risk 0.25%.
- **Modification under test:** BUY-only — SELL candidates are excluded;
  BUY entry/exit/stop/target logic is byte-for-byte identical to the
  original.
- **Code reference:** `Model B` from `src/phase22_audjpy_amr_confirmatory.py`,
  reproduced unmodified in the tracker. Strategy version string recorded
  on every trade row: `phase22_model_B_buy_only@55e301e353ef271b00a766cee34b294bf66edc81`.
- **No filters of any kind** (ATR, trend, session, or otherwise) are
  applied beyond the BUY/SELL split.

## Start time (frozen)

**2026-08-11 09:45:00 UTC.** No bar before this timestamp is eligible
to generate a recorded prospective signal. (Bars before this timestamp
are used only as rolling-indicator lookback context — SMA20/STD20 need
20 prior bars to compute a value at any given bar, exactly as the live
strategy already does; this is not "future information," it's the same
backward-looking calculation AMR always performs.)

## Side-by-side recording methodology

Every qualifying AMR signal at or after the start time is evaluated on
two paths simultaneously, using **identical** entry timestamp, entry
price, spread, slippage, stop, target, and holding period:

- **ORIGINAL AMR** — eligible on every signal (BUY or SELL).
- **BUY-ONLY** — eligible only when the signal direction is BUY.

Each recorded trade row carries both eligibility flags
(`original_amr_eligible`, `buy_only_eligible`), so ORIGINAL and BUY-ONLY
performance can always be reconstructed separately from the same
immutable log without re-running anything.

## Immutability policy

`data/audjpy_amr_forward_trades.csv` is **append-only**. The tracker
script never edits or deletes an existing row. If a data or execution
error is ever objectively identified in a recorded trade, the original
row is preserved and a correction is documented as a new entry in
`data/audjpy_amr_forward_audit_log.jsonl` — never a silent overwrite.

## Pre-registered validation-completion criterion (frozen BEFORE any prospective data exists)

Per instruction, this criterion is defined now, with zero trades
recorded, precisely so it cannot be chosen to fit a result. This project
has no existing standard "minimum forward-test size" gate to reuse (the
project's historical validation gates — e.g. `phase12_nzdjpy_validation_gate.py`,
`phase8_monday_validation.py` — are all *historical* IS/OOS designs, not
prospective/live-forward designs), so a conservative fixed criterion is
documented here instead. **All of the following must be satisfied
before this forward test can be declared complete, in either direction:**

1. **Minimum 50 independent, fully-closed BUY-only prospective trades.**
   At AUDJPY AMR's historical BUY-only frequency (~137 BUY trades/year
   pooled across phase 20-22's 3-year sample, ≈ 11-12/month), this
   corresponds to roughly 4-5 months of collection — small enough to be
   practical, large enough to exceed the `MIN_SAMPLE=20` per-cell bar
   this project has used throughout phases 20-22, by a comfortable margin.
2. **Minimum 120 calendar days of chronological coverage** from the
   frozen start time, regardless of trade count — guards against an
   unusually fast burst of signals satisfying criterion 1 from a single
   short, unrepresentative stretch.
3. **Coverage spanning at least 2 different calendar quarters** — guards
   against the entire sample coming from one narrow market regime.
4. **Cost realism check:** the recorded spread assumption must remain
   consistent with the frozen 2.0-pip specification; if live/demo
   execution data becomes available showing materially different real
   spreads, that must be reported alongside the frozen-assumption
   numbers, not substituted for them silently.
5. **Stability check:** BUY-only's advantage over ORIGINAL AMR must be
   assessed using the same bootstrap-CI methodology already used in
   `reports/audjpy_amr_confirmatory_filter.md` (Part 9) — the forward
   test does not require the CI to exclude zero to be considered
   *complete* (that's a separate question from whether it's *validated*),
   but the CI must be computed and reported honestly at every checkpoint,
   including if it still crosses zero.

**Until all five conditions are met, every checkpoint report must
classify the forward test as `INSUFFICIENT FOR VALIDATION` — this is
the default state, not a fallback for a disappointing result.** Meeting
all five conditions makes the forward test *complete* (enough evidence
exists to render a verdict); it does not by itself mean BUY-only
*passed* — the checkpoint reviewing the complete data still has to look
at whether the direction and magnitude of the effect held up.

## Current status

| Metric | Value |
|---|---|
| Prospective trades recorded (either path) | **0** |
| Calendar days elapsed since start | **< 1** |
| Quarters covered | **0 of 2 required** |
| Criterion 1 (≥50 trades) | Not met |
| Criterion 2 (≥120 days) | Not met |
| Criterion 3 (≥2 quarters) | Not met |
| **Overall status** | **INSUFFICIENT FOR VALIDATION (expected — data collection has just begun)** |

## How this report is updated

This is a living document. Re-run `python src/amr_forward_tracker.py`
periodically (the script is idempotent — safe to run as often as
convenient, e.g. daily) to pull new bars and append any newly-closed
trades to the immutable log. This report itself should only be
re-written at meaningful checkpoints (not after every tracker run) —
the next scheduled checkpoint is **2026-08-25**, per the existing AMR
checkpoint. At that checkpoint, this report must be regenerated from
the accumulated `data/audjpy_amr_forward_trades.csv` and must state,
without exception:

1. Number of prospective observations (ORIGINAL and BUY-ONLY counts,
   separately).
2. ORIGINAL AMR performance on the prospective sample.
3. BUY-ONLY performance on the prospective sample.
4. The difference (total R, expectancy, PF, drawdown) between them.
5. Whether the directional asymmetry (SELL weaker than BUY) found in
   phase 21 remains consistent in the fresh data.
6. Whether the five-condition completion criterion above has been met.
7. Remaining uncertainty, reported honestly — including if the sample
   is still too small, or if the bootstrap CI still crosses zero even
   after the criterion is met.

**No result at that checkpoint — however favorable or unfavorable —
authorizes changing the BUY-only rule, restoring SELL trades, adding a
filter, or modifying the demo account.** The frozen specification in
this document is permanent for the duration of this forward test.

---

## What I did NOT do (per instructions)

- Did not modify AUDJPY AMR, CADJPY AMR, GBPJPY AMR, EURJPY AMR, ARB,
  Monday Drift, or XAUUSD ARB.
- Did not change the demo account.
- Did not choose the completion criterion after seeing any result — it
  is documented above with zero trades recorded.
- Did not declare BUY-only validated, promising, or failed — there is
  no prospective evidence yet to support any such classification.
- Did not optimize, search a parameter, or add a filter to the frozen
  candidate.
