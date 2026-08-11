# AUDJPY AMR BUY-Only — Final Validation Gate

**Experiments:** EXP-087 through EXP-089, `experiments/experiments.csv`.
**Git commit at time of this validation attempt:** `55e301e353ef271b00a766cee34b294bf66edc81`
(the exact commit that produced `reports/audjpy_amr_confirmatory_filter.md`).

**No strategy was modified. No parameter was searched. No optimization
occurred. The demo account was not touched.** This report's primary
finding is a data-availability constraint, not a performance result —
and per instructions, that constraint is reported honestly rather than
worked around.

## 1. Frozen specification

**Instrument:** AUDJPY. **Base strategy:** existing live AUDJPY AMR
(`signals_amr_v`, `z_thr=2.0`, `sl_mult=1.5`, `end_hour=4`, spread 2.0
pips, risk 0.25% — identical to the live YAML parameters used in phases
20/21/22, unchanged). **Modification:** BUY-only — SELL/rally-fading
candidates are excluded entirely; the BUY entry/exit/stop/target logic
is byte-for-byte unchanged. **Code reference:** the filter is exactly
`Model B` from `src/phase22_audjpy_amr_confirmatory.py`
(`tdf[tdf['dir'] == 'BUY']`), commit `55e301e`. No line of that logic is
touched by this report. Execution assumptions (spread, slippage model,
risk sizing, session window 00:00–04:00 UTC, 1.5×ATR stop, mean-reversion
target) are identical to the frozen baseline in every respect except the
SELL exclusion.

## 2. Data separation methodology

This is the central finding of this validation attempt, so it is stated
plainly before anything else: **there is no genuinely fresh historical
data available.**

`src/phase22_audjpy_amr_confirmatory.py`'s FINAL OOS window
(2025-08-07 → 2026-08-11) already consumed every bar available up to
the present moment. Re-fetching AUDJPY M15 data just now (for this
report) confirms:

```
Latest available bar: 2026-08-11 09:45:00+00:00
Fetch time (now):     2026-08-11 09:59:49+00:00
```

The most recent bar available from the data source is effectively "now"
— there has been no passage of calendar time between phase 22's OOS
window closing and this validation attempt. Every historical bar that
exists has already been used in either TRAIN, VALIDATION, or FINAL OOS
in phase 22. **Constructing a new "final OOS" window from this same
historical data would necessarily overlap with data already used for
model comparison and selection in phase 22 — this would not be a
genuine held-out test, it would be re-using the same evidence and
calling it new.** Per explicit instruction, this is not being
manufactured.

## 3. Exact validation period

**Historical final OOS: not available — insufficient fresh data (0
genuinely untouched calendar days beyond phase 22's existing FINAL OOS
window, which itself extends through the present moment).**

**Prospective forward-validation period (established per instructions'
explicit fallback):**

- **Start:** 2026-08-11 09:45:00 UTC (the latest bar available at the
  time of this report — i.e., beginning now).
- **End:** not yet reached — this period is by definition still in the
  future relative to this report.
- **Frozen rule for the entire period:** AUDJPY AMR BUY-only, exactly as
  specified in Section 1, with zero changes permitted regardless of
  interim results.
- **Trades observed in this period as of this report: zero** (no
  calendar time has passed since the freeze point).

## 4. Control results (existing AUDJPY AMR, for reference — unchanged from phase 22)

| Period | n | win rate | PF | expectancy |
|---|---|---|---|---|
| FINAL OOS (2025-08-07 → 2026-08-11, phase 22) | 208 | 69.2% | 1.14 | +12.86 |

## 5. BUY-only results (for reference — unchanged from phase 22, not re-run)

| Period | n | win rate | PF | expectancy |
|---|---|---|---|---|
| FINAL OOS (2025-08-07 → 2026-08-11, phase 22) | 140 | 77.9% | 1.74 | +49.79 |

These are restated from phase 22 for completeness, not re-derived —
re-running the identical simulation on the identical data would not
constitute new evidence.

## 6. Cost stress

Not re-run. Phase 22 already tested normal / 1.5x / 2x spread / 1-bar
delay on this exact frozen rule (Section 9 of
`reports/audjpy_amr_confirmatory_filter.md`): BUY-only was the only
model of the three tested that remained profitable at 2x spread (PF
1.08). That evidence stands; repeating it on the same data would not
add information.

## 7. Statistical uncertainty

Restated from phase 22, honestly, without softening: the OOS bootstrap
95% confidence interval on the expectancy difference between BUY-only
and control was **[-5.95, +81.31]** — it includes zero. P(BUY-only >
control) = 95.4%. **This confidence interval crossing zero is exactly
why phase 22 classified the result SUPPORTED rather than VALIDATED, and
exactly why this report cannot upgrade that classification without new
data.** No new statistical test was run here, because no new data exists
to run it on.

## 8. Chronological consistency

Not re-assessed here — phase 22 already reported this (10 of 11
walk-forward windows favored BUY-only, full log in
`reports/phase22_confirmatory_log.txt`). Repeating it would use the same
windows already examined.

## 9. Directional sanity check

Not applicable in this report — there is no untouched validation period
with trades to check BUY vs. SELL performance on. The historical BUY vs.
SELL comparison from phase 21/22 (SELL population independently
net-losing, PF 0.70) stands as prior evidence, not fresh confirmation.

## 10. Strongest evidence FOR (carried forward from phase 22, not re-derived)

Large, walk-forward-consistent, year-consistent, cost-stress-robust
improvement; the excluded SELL population is an independently-confirmed
historical net loser, not an arbitrarily chosen subgroup.

## 11. Strongest evidence AGAINST

The OOS bootstrap confidence interval on expectancy still crosses zero
— the single most rigorous statistical test applied to this candidate
has not cleared a formal significance bar. **And, as of this report,
there is no fresh data with which to attempt to clear it.**

## 12. Remaining uncertainty

Everything phase 22 already flagged remains unresolved: the formal
significance test is marginal, not decisive. In addition, this report
adds one new source of uncertainty: **the candidate has never been
observed on data that postdates its own selection.** Every number
reported for BUY-only so far comes from a period that also informed
(via phase 21's mechanism research, which used overlapping historical
data) the decision to test BUY-only in the first place. This does not
mean the finding is wrong — phase 22's design already tried to guard
against this with a strict chronological split and out-of-sample
testing within the available history — but it does mean no data exists
yet that is fully independent of the entire research process that
produced this candidate. Only the passage of real calendar time resolves
that.

## 13. August 25 checkpoint summary

*(Prepared as input to the existing checkpoint — this report does not
make the portfolio decision.)*

1. **Original AUDJPY AMR:** live, unchanged, FINAL OOS PF 1.14,
   expectancy +12.86/trade (phase 22, Section 4 above).
2. **BUY-only candidate:** not live, frozen specification (Section 1),
   FINAL OOS PF 1.74, expectancy +49.79/trade — nearly 4x the control's
   expectancy on the same held-out period.
3. **Mechanism evidence (phase 21):** AUDJPY's volatility-regime
   deterioration survives conditioning on trend in all 3 trend terciles;
   SELL trades collapse hardest in high volatility (PF 0.58); MAE
   exceeds MFE in the HIGH regime — directional asymmetry (BUY strong,
   SELL weak) is corroborated from multiple independent angles, not just
   the raw win-rate split.
4. **Historical validation evidence (phase 22):** BUY-only beats control
   in TRAIN, VALIDATION, and FINAL OOS; 10 of 11 walk-forward windows;
   all 3 tested years; uniquely survives 2x spread stress; retains 201.5%
   of control's total R on 63.2% of the trade count.
5. **Fresh validation evidence (this report):** **none available.** No
   calendar time has passed since phase 22's OOS window closed.
6. **Cost stress:** BUY-only remains profitable through 2x spread;
   control and the volatility-filter alternative (Model A) both do not.
7. **Remaining uncertainty:** the OOS bootstrap CI on expectancy still
   crosses zero (P=95.4%, not a clean two-sided 95%); no data exists yet
   that postdates the entire research process that selected this
   candidate.
8. **Recommended decision for the checkpoint to consider (not made
   here):** given (a) the size and consistency of the historical
   evidence, (b) the explicit absence of any way to obtain a genuinely
   fresh historical validation right now, and (c) that "D. VALIDATED FOR
   DEMO FORWARD TESTING" is explicitly defined as authorizing *only* a
   controlled, no-real-money demo forward test (not deployment) — the
   checkpoint may reasonably choose to begin the prospective
   forward-validation period defined in Section 3 on the demo account,
   under the frozen rule, with the explicit understanding that no
   performance-based changes are permitted during that period and that
   this itself is not a portfolio decision. That is a decision for the
   checkpoint to make, not one this report is authorized to make.

## 14. Final classification

# **B. INSUFFICIENT FRESH DATA**

Not A (FAILED) — nothing failed; no valid test was possible to run.
Not C (PROMISING BUT NOT VALIDATED) alone — this undersells the specific,
concrete reason validation could not proceed (there is literally zero
untouched historical data, confirmed by re-fetching at report time).
Not D (VALIDATED FOR DEMO FORWARD TESTING) — that classification would
require an actual fresh-data test having been run and passed; none was
possible to run.

**This is the classification the instructions explicitly anticipated
and required as the correct outcome when fresh data does not exist,
rather than manufacturing a validation window from already-used data.**

## 15. Exact recommendation

1. **Do not deploy BUY-only.** It remains a research candidate, not a
   live strategy.
2. **Do not modify AUDJPY AMR, CADJPY AMR, GBPJPY AMR, EURJPY AMR, ARB,
   Monday Drift, or XAUUSD ARB.** None were touched by this report.
3. **Bring this report, and the August 25 checkpoint summary in Section
   13, to the 2026-08-25 checkpoint.** The checkpoint — not this
   report — should decide whether to open the prospective forward-
   validation period defined in Section 3 (starting 2026-08-11 09:45
   UTC) on the demo account under the fully frozen rule.
4. **If the checkpoint opens the prospective period:** no performance-
   based changes are permitted during it, per instructions; it should
   run for long enough to accumulate a meaningful, genuinely
   independent trade sample before any classification upgrade is
   considered — that determination belongs to whoever reviews it once
   real data exists, not to this report.
5. **Do not attempt another historical validation gate before real
   calendar time has passed** — repeating this exercise tomorrow would
   face the identical data-insufficiency problem.

---

## Auditability record

- **Git commit:** `55e301e353ef271b00a766cee34b294bf66edc81`
- **Frozen strategy specification:** Section 1 of this report; code
  reference `src/phase22_audjpy_amr_confirmatory.py`, `Model B`
  (`tdf[tdf['dir'] == 'BUY']`), unmodified.
- **Historical data range used across all AUDJPY AMR research to
  date:** 2023-07-31 to 2026-08-11 (phase 20/21/22).
- **Validation range attempted in this report:** none available
  (historical); prospective period frozen to begin 2026-08-11 09:45 UTC.
- **Experiment IDs:** EXP-087 (data-separation check), EXP-088
  (checkpoint-summary compilation), EXP-089 (final classification).
- Fully reproducible: re-running `src/phase22_audjpy_amr_confirmatory.py`
  against the same commit and the same `data_loader.get_bars` call
  reproduces every number cited here.

## What I did NOT do (per instructions)

- Did not modify AUDJPY AMR, CADJPY AMR, GBPJPY AMR, EURJPY AMR, ARB,
  Monday Drift, or XAUUSD ARB.
- Did not change the demo account.
- Did not fabricate or manufacture an OOS period from already-used data.
- Did not optimize, search a parameter, or add a filter.
- Did not make the portfolio decision — Section 13 is evidence prepared
  for the checkpoint, not a decision.
- Did not deploy anything.
