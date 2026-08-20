# Phase 49 Preregistration — Portfolio Stress Mechanism & Contribution Audit

**Frozen before any stress-analysis result is inspected. Committed separately, before any Phase 49 result exists. Not modified after seeing results.**

PORTFOLIO DIAGNOSTIC RESEARCH ONLY. No live strategy code, YAML, risk, or position sizing modified. No filter, control, or threshold deployed. All findings are diagnostic; any possible intervention is recorded only as a FUTURE HYPOTHESIS.

---

## 1. The six frozen strategies (unchanged from Phases 45-48)

`AUDJPY_AMR`, `CADJPY_AMR`, `EURJPY_AMR`, `GBPJPY_AMR`, `CADJPY_ARB`, `GBPUSD_MONDAY`.

## 2. Primary data source

`data/phase26_all_trades.csv` — the same 2,712-trade historical control reconstruction used as the primary diagnostic population in Phases 31-48, unchanged. Live data (the freshest local `reports/5ers_trade_export.csv`, as used in Phase 45/46/48) is used **only** for the separate, explicitly-labeled live-comparison section (§26) — never pooled into the primary diagnostic population, per the task's explicit instruction.

## 3. Portfolio-day construction (frozen, per Part 6)

One row per UTC calendar day with at least one trade entry. For every day: total portfolio R, trade/entry/exit counts, max and average concurrent positions (interval-overlap count, identical methodology to Phase 41/43), JPY/AMR/ARB/Monday trade-share, long/short trade-share, Asian/London/NY session-share (using the ledger's own `session` column — confirmed in Phase 41/45 to contain only ASIAN/LONDON values; NY share will be reported as a confirmed-zero fact, not silently omitted), volatility level/percentile/state/transition (reusing the ledger's own `atr_pctile`/`vol_tercile` fields and Phase 42's exact daily-aggregation and tercile methodology), count of distinct strategies active, count of simultaneous JPY-linked positions, count of simultaneous AMR positions.

## 4. Stress / normal definitions (frozen, per Part 7, identical convention to Phases 41-44)

Worst 1%/5%/10% of daily portfolio R, thresholds computed once on the full historical period, never re-chosen after seeing which threshold looks most interesting. Normal = above the worst-20% threshold, per the established project convention.

## 5. Concurrency, exposure, direction, session, strategy-family definitions

Identical to Phases 41-46: concurrency = interval-overlap count at any point in the day; JPY exposure = trades on instruments containing JPY; direction = the `dir` field; mechanism families parsed from the strategy-name suffix (`_AMR`/`_ARB`/`_MONDAY`).

## 6. Volatility-transition definition (unchanged from Phase 42)

LOW/NORMAL/HIGH terciles of the daily volatility percentile (computed once on the full period); transition = the (previous-day-state, current-day-state) pair.

## 7. Joint-state combinations tested (frozen, exactly the 12 listed in Part 9, no others)

A. vol×concurrency, B. vol×JPY, C. vol×AMR, D. vol×direction, E. concurrency×JPY, F. concurrency×AMR, G. concurrency×direction, H. AMR×JPY, I. vol×concurrency×AMR, J. vol×concurrency×JPY, K. vol×concurrency×direction, L. vol×concurrency×JPY×direction. A cell with fewer than 10 days is reported as INSUFFICIENT SAMPLE, never interpreted.

## 8. Pre-stress exposure methodology (frozen, per Part 11, with a disclosed data-granularity limitation)

T-1 (previous trading day) exposure state is fully computable from the daily ledger. **T-60/T-30/T-15-minute intraday pre-stress exposure is NOT COMPUTABLE from this project's trade-level ledger** (no continuous intraday position-snapshot series exists at that granularity — the same class of disclosed limitation as Phase 42's lead-lag analysis and Phase 43's build-up/decay analysis) — reported as UNKNOWN BY DATA LIMITATION, not approximated.

## 9. Statistical approach (frozen, per Part 22)

Effect size (absolute/relative difference between stress and normal populations) is the primary reported quantity, with sample size and a qualitative evidence-strength label, consistent with every prior phase's convention. No p-value-hunting; where a simple two-sample comparison is reported, it is descriptive, not a formal hypothesis test with a claimed significance threshold.

## 10. Multi-factor explanatory model (frozen, per Part 20)

An ordinary-least-squares regression of daily portfolio R on the standardized (z-scored) predictors: volatility percentile, concurrency, JPY trade-share, AMR trade-share, ARB trade-share, long-share, strategy count — fit via `numpy.linalg.lstsq` (no `statsmodels`/`sklearn` dependency available in this environment; standard errors computed directly from the OLS residual covariance formula). Explicitly **explanatory/diagnostic only** — never used to generate a trading signal. Coefficients, standard errors, and R² are reported together; a coefficient is never interpreted without its uncertainty.

## 11. Temporal validation methodology (frozen, per Part 21)

The historical control is split chronologically at its midpoint (by trade count) into an EARLIER period (hypothesis identification) and a LATER period (validation) — no random split, no re-splitting after seeing results. A marginal or joint-state finding from the earlier half is only reported as surviving temporal validation if the same direction (not necessarily the same magnitude) holds in the later half; sample-size floors from §7 apply to each half independently. Findings failing this check, or where either half is under-sampled, are labeled EXPLORATORY ONLY, never CONFIRMED.

## 12. Evidence classification (frozen, per Part 28)

CONFIRMED (survives temporal validation with adequate sample in both halves) / STRONG (large effect, adequate sample, not independently temporally validated) / MODERATE (moderate effect, adequate sample) / PLAUSIBLE (small effect or borderline sample) / EXPLORATORY (secondary/interaction test, not preregistered as primary) / NO EVIDENCE / CONTRADICTED (opposite of a prior phase's finding) / INSUFFICIENT SAMPLE.

## 13. Descriptive counterfactual rule (frozen, per Part 25)

Any "if X had not occurred" estimate is computed directly from the historical daily ledger (removing the relevant days/trades and re-summing), labeled **DESCRIPTIVE COUNTERFACTUAL**, never **VALIDATED CONTROL**. No threshold is chosen to make a counterfactual look most favorable — only pre-registered stress/factor definitions from §4-§7 are used as the basis for any counterfactual.

## 14. Phase 44 connection (frozen, per Part 27)

This phase does not re-test, re-optimize, or attempt to overturn Phase 44's NO PORTFOLIO CONTROL JUSTIFIED finding. It only asks what mechanism a *future* control would need to target, if one were ever independently tested.

## 15. No-intervention rule

Any finding, however strong, results only in a recorded FUTURE HYPOTHESIS. No filter, limit, weight change, or strategy modification is implemented in this phase.

---

*No amendment has been made to this document after any Phase 49 result was produced.*
