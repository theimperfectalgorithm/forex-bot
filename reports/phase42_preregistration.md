# Phase 42 Preregistration — Volatility Stress Decomposition

**Frozen before any substantive analysis. Committed separately, before any Phase 42 result exists. Not modified after seeing results.**

FORENSIC/OBSERVATIONAL ANALYSIS ONLY. No new strategy, no backtest, no intervention, no portfolio change. Extends Phase 41's H. NO SINGLE DOMINANT FACTOR finding by decomposing its one MODERATE-evidence factor (HIGH-volatility trade share) into 8 preregistered sub-hypotheses.

---

## 1. Control portfolio (unchanged from Phase 41)

`data/phase26_all_trades.csv`, 2,712 trades, 6 strategies (`EURJPY_AMR`, `AUDJPY_AMR`, `CADJPY_AMR`, `GBPJPY_AMR`, `CADJPY_ARB`, `GBPUSD_MONDAY`), 2023-08-01 to 2026-08-13. No candidate strategy (AUDUSD Monday LONG, Phase 38 H1/H2, Phase 40) is added.

## 2. Historical windows (frozen, per Part 6, identical convention to Phase 41)

**A. Full historical control period**: the entire reconstruction. **B. Pre-demotion**: entry_time < 2026-07-31, within the same reconstruction. **C. Post-demotion live period**: `reports/5ers_portfolio_update_aug13_trade_level.csv`, 19 real trades — reported separately, never pooled with A/B, explicitly labeled INSUFFICIENT LIVE SAMPLE given n=19.

## 3. Daily aggregation (identical to Phase 41)

One row per calendar trading day (`entry_time.dt.date`, UTC), P&L/R attributed to entry date.

## 4. Volatility measurement (frozen, per Part 7 — reusing existing infrastructure, not inventing a new indicator)

**Source**: the control data's own already-validated per-trade `atr_pctile` field (a continuous 0-1 ATR-based volatility percentile, computed by the original AMR/ARB strategy pipeline at entry time) — the same "already validated realized-volatility infrastructure" referenced in the task instructions, reused as-is. **Daily volatility level**: the simple mean of `atr_pctile` across all trades entered that day. Where a day has zero trades, its volatility level is carried forward as UNKNOWN (not interpolated) for that day's row, and such days are excluded from volatility-conditioned calculations (disclosed in §Missing-data handling). **Daily volatility percentile**: the day's volatility level re-ranked as a percentile against the full-period distribution of daily volatility levels (i.e., a percentile-of-percentiles, since the underlying `atr_pctile` is itself already a percentile — this two-stage construction is disclosed, not hidden, and is necessary because the unit of analysis in this phase is the *trading day*, not the individual trade). **Daily volatility state**: LOW/NORMAL/HIGH terciles of the daily volatility percentile, computed once on the full period (§2A) — a coarser, discrete companion to the continuous percentile, used for the transition-matrix (§17) and state-based hypotheses (H2).

## 5. Volatility change / acceleration (frozen, per Parts 10-11)

**Δ volatility** (day *t*): daily volatility percentile(*t*) − daily volatility percentile(*t*-1), using the immediately preceding **trading** day (not calendar day, since the control does not trade every calendar day). **Volatility acceleration**: Δ(Δ volatility) — the day-over-day change in Δ volatility. Both are classified into LOW/NORMAL/HIGH terciles of their own full-period distribution, computed once (§2A).

## 6. Stress definition (identical convention to Phase 41)

Worst 1%/5%/10%/20% of daily portfolio R, fixed thresholds computed once on the full period.

## 7. Concurrent-position definition (identical to Phase 41)

Count of trades whose `[entry_time, exit_time]` interval overlaps any point in the given calendar day. Predefined buckets per Part 12: 0-1, 2, 3, 4, 5, 6.

## 8. Session / direction / JPY / mechanism definitions

Identical to Phase 41 (`session` column values ASIAN/LONDON only; `dir` BUY/SELL; JPY = instrument base or quote is JPY; mechanism parsed from the strategy-name suffix `_AMR`/`_ARB`/`_MONDAY`). **Disclosed limitation for H5 (volatility × session)**: the control data contains no London/NY-overlap or New-York-session trades at all (confirmed in Phase 41's data integrity check), so any session-transition analysis in this phase is necessarily limited to the Asian→London transition only — London→NY is UNKNOWN by data absence, not a null finding, and is reported as such rather than silently omitted.

## 9. Eight primary hypotheses (frozen, per Parts 9-16)

H1 (absolute volatility, continuous, per Part 9) · H2 (volatility change/state-transition, per Part 10) · H3 (volatility acceleration, per Part 11) · H4 (volatility × concurrent exposure, per Part 12) · H5 (volatility × session, per Part 13, with the disclosed NY-session-absence limitation) · H6 (volatility × direction, per Part 14) · H7 (volatility × JPY, per Part 15) · H8 (volatility × mechanism, per Part 16). Each tested exactly as specified in the corresponding task-instruction part, no additional variant.

## 10. Statistical approach (frozen, per Part 25)

Effect size (absolute and relative difference between compared buckets) is the primary reported quantity, not a p-value threshold. Confidence is expressed via sample size and a qualitative CONFIRMED/STRONG/MODERATE/WEAK/NO RELATIONSHIP/INSUFFICIENT scale (Part 9's classification, reused for every hypothesis) — consistent with every prior phase's practice in this project. A large effect on an insufficient sample (< 20 days in the relevant bucket) is labeled PROMISING/INSUFFICIENT EVIDENCE, never promoted to CONFIRMED.

## 11. Lead-lag windows (frozen, per Part 18)

Given the control data's trade-level (not continuous intraday tick/bar) granularity, only **day-level** lag windows are computable without fabricating false intraday precision: **same day, previous trading day**. "Previous session," "previous 4 hours," and "previous 8 hours" are **not computable from this dataset** (no continuous intraday volatility series is stored per session/hour) and are reported as UNKNOWN BY DATA LIMITATION, not silently dropped or approximated.

## 12. Threshold robustness (frozen, per Part 20)

Descriptive sensitivity only, at the 70th/80th/90th/95th percentile of the daily volatility-percentile distribution — never optimized, never used to select a "best" threshold.

## 13. Extreme-day sensitivity (frozen, per Part 21)

Primary findings (H1, H4 — the two hypotheses most likely to show a strong effect per Phase 41) re-run after excluding the worst 1/5/10 days.

## 14. Regime robustness (frozen, per Part 22)

Reuses this project's established historical periods (2019-2020, 2021-2022, 2023-2024, 2025, 2026 YTD) where the control's own 2023-08-2026-08 date range permits — the control does not extend back to 2019, so periods before 2023-08 are UNKNOWN BY DATA ABSENCE for this specific control (disclosed, not a null finding about those years).

## 15. Missing-data handling

19 trades have a null `atr_pctile`/`vol_tercile` in the source (1 disclosed in Phase 41; re-verified here) — excluded from volatility-specific calculations only, retained in R/count-based tables. Trading days with a null day-level volatility average (no trades entered that day, or all entries null) are excluded from volatility-conditioned rows and disclosed in the relevant CSV's row count vs. the full 774-day ledger.

## 16. Multiple-testing policy (frozen, per Part 26)

H1-H8 are the 8 primary preregistered hypotheses. Threshold-robustness (§12), extreme-day sensitivity (§13), and regime robustness (§14) are **robustness checks on the primary hypotheses**, not new hypotheses. Lead-lag (§11) and any transition-matrix cell not directly tied to H1-H8 are EXPLORATORY, labeled as such throughout.

## 17. Evidence classification (frozen, per Part 34 of the task instructions, reusing Part 27's decision framework)

Final volatility classification: one of A (CONFIRMED) / B (PROBABLE) / C (MODERATE/PROMISING BUT NOT CONFIRMED) / D (WEAK/INCONSISTENT) / E (NO EVIDENCE) / F (INSUFFICIENT DATA) — selected by whether H1-H8's effects are large, consistent across thresholds/regimes/extreme-day exclusion, and adequately sampled. No causal language ("volatility causes losses") — only "associated with"/"coincides with"/"preceded."

## 18. No-intervention rule

Any finding, however strong, is recorded as an observation and a FUTURE RESEARCH HYPOTHESIS only. No filter, scaling rule, or portfolio control is implemented or tested in this phase.

---

*No amendment has been made to this document after any Phase 42 result was produced.*
