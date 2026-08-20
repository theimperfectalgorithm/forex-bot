# Phase 50 Preregistration — Prospective Stress Signal Validation

**Frozen before any hypothesis result is inspected. Committed separately, before any Phase 50 result exists. Not modified after seeing results.**

DIAGNOSTIC VALIDATION ONLY. No live strategy code, YAML, risk, position sizing, or portfolio weight modified. No filter, limit, or control deployed. Tests exactly two hypotheses (H1, H2) carried forward from Phase 49 — no new hypothesis is introduced.

---

## 1. Population (unchanged from Phases 45-49)

`AUDJPY_AMR`, `CADJPY_AMR`, `EURJPY_AMR`, `GBPJPY_AMR`, `CADJPY_ARB`, `GBPUSD_MONDAY`, using `data/phase26_all_trades.csv` (2,712 trades, 774 trading days, 2023-08-01 to 2026-08-13) — the same historical control used throughout Phases 31-49.

## 2. Prediction-time-safe dataset construction (frozen, per Parts 6/13/22)

Reuses Phase 49's daily portfolio dataset construction exactly (`src/phase49_stress_dataset.py`), then adds a **one-trading-day lag**: for every day *T* with a valid predecessor trading day *T-1* in the ledger, the predictor row uses *T-1*'s already-closed state (JPY trade-share, volatility state/percentile, max concurrent positions) to predict *T*'s outcome. Since *T-1* is a fully completed prior trading day, `predictor_timestamp (end of T-1) < outcome_start (start of T)` holds by construction for every row — verified explicitly via an audit column `lookahead_safe`, computed as `T-1_date < T_date`, asserted `True` for 100% of rows before any hypothesis is tested (per Part 22-23). Any row failing this check is excluded and reported, not silently dropped.

## 3. Primary outcome (frozen, per Part 8/14 — exactly ONE)

**Next-trading-day total portfolio R** (`T_total_R`), continuous. A directly-derived companion measure, **next-day stress-day indicator** (`T_total_R` in the worst-10% of the full-period total-R distribution, threshold computed once on the full period per Phase 41-49's convention), is reported alongside as a secondary transform of the same outcome — not a second competing outcome window, per Part 8's "use ONE primary outcome" instruction.

## 4. H1 — conditional JPY exposure (frozen, per Parts 5-11)

**Predictor**: `T-1` portfolio JPY-linked trade-share (`jpy_share_pct`). **Conditioning variables**: `T-1` volatility state (LOW/NORMAL/HIGH tercile, Phase 42's exact methodology) and `T-1` max concurrent positions, bucketed at the Phase 43/49-established 4+ threshold (no new threshold invented). **Primary test**: within each of the 6 volatility-state × concurrency-bucket cells with ≥`MIN_CELL` (20) days in a given period, compare mean `T_total_R` for JPY-high (`T-1` JPY share ≥ full-period median) vs. JPY-low (`T-1` JPY share < median) subgroups. The median split threshold is computed once on the full period, never re-chosen per period or per cell, per Part 11's no-threshold-mining rule.

## 5. H2 — concurrency concentration (frozen, per Parts 12-17)

**Predictor**: `T-1` max concurrent positions (pre-outcome by construction). **Primary test, run on the FULL eligible population** (not the worst-10% subset, per Part 17's explicit anti-selection-bias instruction): compare mean `T_total_R` for high-concurrency (`T-1` max concurrent ≥ 4, the same threshold Phase 43/49 already established, not re-chosen here) vs. low-concurrency (`T-1` max concurrent < 4) subgroups. The worst-10% subset is used **only** for a clearly labeled SECONDARY DESCRIPTIVE re-statement of Phase 49's original within-stress-population concentration finding — never as the basis for the primary predictive test.

## 6. Discovery / validation split (frozen, per Part 10)

Chronological midpoint of the 774-day dataset (identical split point to Phase 49's temporal validation, for direct comparability) — EARLIER half = DISCOVERY (first ~387 days), LATER half = VALIDATION (last ~387 days). No re-splitting after seeing results. A relationship is **DISCOVERY-SUPPORTED** if it holds directionally in the earlier half; **VALIDATED** only if the same direction (not necessarily the same magnitude) also holds in the untouched later half.

## 7. Walk-forward check (frozen, per Part 18)

Given the ~774-day sample, a 3-fold expanding-window walk-forward is used: Fold 1 (discovery = first 258 days, validate on days 259-516), Fold 2 (discovery = first 516 days, validate on days 517-774) — 2 folds given the sample size (a 3rd fold would leave an under-powered final validation window; if either fold's validation population falls below `MIN_CELL` for a given cell, that fold is reported as INSUFFICIENT SAMPLE for that cell, not omitted from the report).

## 8. Minimum sample size (frozen, per Part 9/15/24)

`MIN_CELL = 20` days per subgroup per period (discovery or validation, whole-population or per-conditioning-cell) for a comparison to be reported as anything other than INSUFFICIENT SAMPLE — a stricter floor than Phase 41-49's 10/8-day floors, reflecting this phase's explicitly higher evidentiary bar (this phase can validate/reject, not just describe).

## 9. Effect size and statistical reporting (frozen, per Part 24)

For every comparison: mean difference (high − low), a 95% confidence interval computed via the standard Welch two-sample formula (unequal variances, no equal-variance assumption), sample sizes of both groups, and baseline vs. conditional stress-day rate (using the binary companion outcome from §3). No p-value is used alone to determine classification, per the explicit instruction.

## 10. H1×H2 interaction (frozen as SECONDARY EXPLORATORY, per Part 20)

Tested only after both primary tests are complete: `T-1` JPY-high AND `T-1` concurrency-high jointly, vs. all other combinations, on `T_total_R` — labeled EXPLORATORY throughout, never used to rescue a failed primary hypothesis.

## 11. Robustness checks (frozen, per Part 28)

For any hypothesis reaching at least PROMISING BUT UNCONFIRMED: re-run excluding the single worst day, re-run excluding the worst 5 days, and report early-vs-late-validation-period stability (splitting the validation half itself in two). These are robustness checks on an already-validated (or nearly-validated) finding, never a re-search for a better specification.

## 12. Classification rules (frozen, per Part 25, verbatim from the task)

A. VALIDATED / B. PROMISING BUT UNCONFIRMED / C. REJECTED — NO TEMPORALLY STABLE RELATIONSHIP / D. REJECTED — NO CREDIBLE SIGNAL / E. INSUFFICIENT DATA — applied exactly as defined in the task instructions, reproduced in `reports/phase50_decision_matrix.csv`.

## 13. Live comparison (frozen, per Part 29)

The freshest local, uncommitted `reports/5ers_trade_export.csv` (post-demotion closed current-6 trades, as used in Phase 45/46/48/49) is examined descriptively for whether its pre-stress JPY/concurrency characteristics resemble any pattern found historically — explicitly **contextual evidence only, never validation**, given its small sample (19 trades).

## 14. Phase 44 connection (frozen, per Part 30)

This phase does not re-test, re-optimize, or overturn Phase 44's NO PORTFOLIO CONTROL JUSTIFIED finding. Only if a hypothesis reaches A. VALIDATED does the report state that evidence is now sufficient to justify a *separate, future* intervention-design phase — no intervention is designed here.

## 15. Decision tree (frozen, per Part 31, verbatim)

Both fail → recommend stopping portfolio-control research for now, continue observation, do not begin a new strategy search. One passes → recommend a dedicated intervention-design phase on that mechanism only. Both pass → recommend a dedicated intervention-design phase on their joint effect. Either PROMISING BUT UNCONFIRMED → continue observation, collect more evidence, no intervention design yet. Insufficient data → do not manufacture confidence.

---

*No amendment has been made to this document after any Phase 50 result was produced.*
