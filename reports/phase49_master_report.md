# Phase 49 — Portfolio Stress Mechanism & Contribution Audit (Master Report)

**PORTFOLIO DIAGNOSTIC RESEARCH ONLY. No live strategy code, YAML, risk, or position sizing modified. No filter, control, or threshold deployed.**

---

## 1. Executive summary

The single most striking finding of this phase: **within the worst 10% of historical portfolio days, restricting to only the concurrency<4 subset reduces total loss from -290.4R to -25.9R — a ~91% concentration of stress-population losses in the concurrency≥4 subset.** This is a materially sharper, more targeted number than anything in Phases 41-43, but it describes the *already-stressed population*, not a general property of high-concurrency trading — the marginal concurrency-vs-daily-R relationship across the *whole* dataset remains weak and non-monotonic (Phase 43's finding, reconfirmed here), and the concurrency-4+ effect does **not** survive temporal validation as a standalone marginal factor (its sign flips between the earlier and later half of the historical record). Volatility remains the most temporally-robust marginal factor. A new, real finding: **JPY exposure shows a consistent, temporally-validated negative pattern once controlling for volatility state and concurrency** — invisible in Phase 41's simple marginal comparison, but present in every adequately-sampled joint cell here. GBPJPY_AMR's own daily R correlates strongly with total portfolio R (0.727) on the days it trades, but its mere presence does not dramatically predict worse days — consistent with it *moving with* portfolio-wide stress more than *independently causing* it. The explanatory multi-factor model explains only 3.8% of daily-R variance even with seven predictors — reconfirming Phase 41's H. NO SINGLE DOMINANT FACTOR verdict via an independent methodology. **Final classification: C. MULTI-FACTOR STRESS MECHANISM**, with the concurrency-within-stress concentration as the most concrete, quantified new lead for any future (untested, unimplemented) intervention research.

## 2. Phase 48 context

Phase 48 closed the parameter/cost robustness gap for all six live strategies (zero sign reversals, all cost-robust) but confirmed 4 of 6 remain CORRELATED on drawdown-correlation, with GBPJPY_AMR (the largest R-contributor) showing the strongest correlation of the six. Phase 49 investigates the mechanism behind that finding directly.

## 3. Research question

What combination of portfolio states actually produces the worst drawdown days, well enough to know what a future intervention phase would need to test?

## 4. Preregistration

`reports/phase49_preregistration.md`, committed separately (`5b408f2`) before any result. No amendment required.

## 5. Data

`data/phase26_all_trades.csv` (2,712 trades, 774 trading days) — the same historical control used throughout Phases 31-48, unchanged. Live data used only for §25's separate, explicitly-labeled comparison.

## 6. Stress definition

`reports/phase49_stress_definition.csv` — worst 1% (8 days), 5% (39 days), 10% (78 days), normal (619 days), thresholds computed once on the full period.

## 7. Marginal factors

`reports/phase49_marginal_stress_factors.csv`. At the worst-5% level: volatility percentile (+15.6 pts, MODERATE), concurrency (+1.07, MODERATE), strategy count (+1.11, MODERATE), simultaneous JPY/AMR positions (+1.07/+0.87, MODERATE), ARB exposure share (+1.4 pts, MODERATE). JPY share and AMR share themselves (as simple trade-percentage marginals) are only PLAUSIBLE — a materially weaker signal than the *conditional* JPY finding in §16.

## 8. Joint-state analysis

`reports/phase49_joint_state_analysis.csv` — all 12 preregistered combinations, 72 combination-states, 66 adequately sampled (≥10 days). No single cell dominates; see §14-16 for the most informative individual combinations (JPY-controlling-for-vol/concurrency, GBPJPY-specific).

## 9. Stress clusters

`reports/phase49_stress_clusters.csv` — D. MIXED (36.4% of worst-10% days occur within 5 days of another stress day) — neither cleanly isolated nor cleanly regime-clustered.

## 10. Pre-stress exposure

`reports/phase49_pre_stress_exposure.csv` — T-1 (previous trading day) state characterized for all 39 worst-5% days; no obviously distinct T-1 signature emerges from simple inspection (EXPLORATORY, not formally tested against a normal-day T-1 population in this phase). **Intraday T-60/T-30/T-15-minute exposure is UNKNOWN BY DATA LIMITATION** — no continuous intraday position-snapshot series exists in this project's trade-level ledger, disclosed before any result was computed.

## 11. Transition analysis

`reports/phase49_transition_analysis.csv`, re-testing Phase 42's finding with concurrency control where sample permits. `HIGH_to_NORMAL` remains the worst transition (mean R -0.248), and its low-concurrency subset is close to breakeven (-0.017) while its high-concurrency subset is materially worse (-0.354) — **the HIGH_to_NORMAL effect appears to be substantially a concurrency-interaction, not a pure volatility-transition effect**, a genuine refinement of Phase 42's original finding.

## 12. Concurrency

`reports/phase49_concurrency_analysis.csv` — mean R across 1+ through 6+ thresholds is non-monotonic (0.251→0.246→0.237→0.216→0.375→0.328) — no clean "more concurrency = worse" marginal pattern across the whole dataset, consistent with Phase 43's prior finding.

## 13. Strategy combinations

`reports/phase49_strategy_combinations.csv` — 15 combinations with ≥10 days. The most common (`AUDJPY_AMR+CADJPY_AMR+EURJPY_AMR+GBPJPY_AMR`, all four AMR pairs simultaneously active, 184 days) shows a below-average mean R (-0.090) and an elevated stress-day rate (15.8% vs. the ~10% baseline for worst-10%) — the largest, most common multi-strategy overlap is also somewhat worse than average, though not dramatically so (EXPLORATORY, per the preregistration's classification of this data-driven grouping as secondary).

## 14. GBPJPY_AMR analysis

`reports/phase49_gbpjpy_analysis.csv`. GBPJPY_AMR-active-day mean R (0.173) is modestly lower than inactive-day mean R (0.336) — a real but not dramatic difference. Its own daily R correlates strongly with total portfolio R on active days (0.727). **Interpretation, per the phase's own three-option framing**: this is most consistent with option 2/3 — GBPJPY_AMR does not independently and dramatically cause portfolio drawdown by its mere presence; rather, it is highly synchronized with whatever the broader portfolio is doing, and (per Phase 48's drawdown-correlation finding) that synchronization is itself stronger during the portfolio's own worst days.

## 15. JPY analysis

`reports/phase49_jpy_analysis.csv` — **the most important new finding of this phase**. Once controlling for volatility state and concurrency simultaneously, JPY-high days underperform JPY-low days in **every one of the 6 adequately-sampled cells**, most dramatically in HIGH-vol+high-concurrency (-0.345 vs. +0.368). This directly refines Phase 41's original marginal finding ("JPY concentration did NOT materially increase during stress... NO CLEAR ASSOCIATION") — the *marginal* (unconditional) relationship is genuinely weak (confirmed again in §7), but a real, temporally-validated *conditional* relationship exists once volatility and concurrency are accounted for.

## 16. AMR analysis

`reports/phase49_amr_analysis.csv`. AMR-high vs. AMR-low shows its clearest divergence specifically within the HIGH-volatility state, consistent with Phase 42/46's AMR/HIGH-vol-concentration finding, though the temporal-validation magnitude is unstable (near-zero in the earlier half, much larger in the later half) — directionally consistent, not confidently stable in magnitude.

## 17. Directional asymmetry

`reports/phase49_directional_analysis.csv`. Long-heavy outperforms short-heavy under normal volatility (+0.538 vs. +0.066), high volatility (+0.131 vs. -0.075), and high concurrency (+0.411 vs. -0.071) — but on stress days specifically, **both directions are almost equally bad** (-3.81 long-heavy vs. -3.64 short-heavy) — the directional asymmetry Phase 42/43 found in ordinary conditions largely disappears once a day is already a stress day.

## 18. Session analysis

`reports/phase49_session_analysis.csv`. **Zero of 774 trading days show any New York-session trade** — directly confirmed from raw timestamps, not assumed. The portfolio is structurally 100% Asian/London.

## 19. Multi-factor model

`reports/phase49_multifactor_model.csv`. OLS on 7 standardized predictors, n=774, **R² = 0.038** — a very low explanatory power even with every candidate predictor included simultaneously. `vol_pctile` is the only individually well-identified coefficient (t=-2.71, negative — higher volatility associated with lower daily R, consistent with §7/§11). `jpy_share_pct`, `amr_share_pct`, and `arb_share_pct` show near-identical t-statistics (~1.69) — a disclosed sign of multicollinearity (these three shares are mechanically related, since they partition the day's trade mix), meaning their individual coefficients should not be over-interpreted in isolation.

## 20. Temporal validation

`reports/phase49_temporal_validation.csv`. Chronological midpoint split (387 days each half). **Survives**: JPY exposure (negative both halves), AMR exposure (negative both halves, unstable magnitude), HIGH volatility (negative both halves). **Does NOT survive**: Concurrency 4+ as a standalone marginal factor (sign flips: -0.20 earlier, +0.01 later) — an important, disclosed negative result that tempers §12's already-weak marginal finding further.

## 21. Multiple-testing controls

`reports/phase49_multiple_testing.csv` — 11 categories of tests logged, distinguishing PRIMARY PREREGISTERED tests (marginal factors, joint-state combinations, transitions, concurrency thresholds, GBPJPY/JPY/AMR/direction/session analyses, the OLS model, temporal validation) from the one EXPLORATORY category (data-driven strategy-combination grouping, §13).

## 22. Worst-day decomposition

`reports/phase49_worst_day_decomposition.csv` — PROPORTIONAL classification at every stress-bucket level (max single-strategy share 20.0-25.7%), confirming Phase 41's independent prior finding via a fresh computation.

## 23. Loss sequence

`reports/phase49_loss_sequence.csv` — **100% of worst-1/5/10%-day populations have 2+ simultaneously losing strategies**, averaging 3.7-5.1 simultaneous losers per stress day. Stress in this portfolio is definitionally a multi-strategy, correlated-loss phenomenon, not an isolated single-strategy event.

## 24. Descriptive counterfactuals

`reports/phase49_descriptive_counterfactuals.csv` — **DESCRIPTIVE COUNTERFACTUAL, not a validated control**, per the explicit labeling rule. Removing GBPJPY_AMR's trades from worst-10% days improves that population's total R from -290.4 to -230.6 (a ~21% improvement — real but partial, consistent with §14's "correlated, not independently causal" interpretation). Restricting the worst-10% population to concurrency<4 days only improves it to -25.9 (a ~91% improvement) — the single most striking number in this report, discussed fully in §1 and `reports/phase49_phase44_connection.md`.

## 25. Live comparison

`reports/phase49_live_comparison.csv`. The post-demotion live sample (19 trades) shows 89.5% JPY share and 78.9% AMR share — both high, consistent with the historical portfolio's structural composition — but the sample is explicitly too small for a confident comparison to any of this phase's conditional findings (DESCRIPTIVE ONLY).

## 26. Phase 44 implications

`reports/phase49_phase44_connection.md` — full discussion. In summary: this phase does not overturn or retest Phase 44's NO PORTFOLIO CONTROL JUSTIFIED finding. It identifies what a future control would need to address (selectivity within the stress population specifically, not broad suppression; a genuinely multi-factor, low-R² mechanism; careful treatment of GBPJPY_AMR as a correlated rather than independently-causal contributor) and explicitly states that **no sufficiently stable, prospectively-actionable mechanism was found this phase** that would, by itself, justify designing a new intervention with confidence.

## 27. Evidence matrix

`reports/phase49_evidence_matrix.csv` — 16 findings classified across the full CONFIRMED→INSUFFICIENT SAMPLE hierarchy, with temporal-validation status recorded for every finding where it was tested.

## 28. What is confirmed

HIGH-volatility's negative association with daily R (temporally validated). Losses are proportional across strategies, not concentrated in one (confirmed independently of Phase 41). 100% of stress days show 2+ simultaneous losers. Zero NY-session exposure (a structural fact).

## 29. What is moderate

Volatility percentile's marginal stress association. JPY exposure's conditional (vol/concurrency-controlled) negative pattern. GBPJPY_AMR's correlation-not-causation profile. The HIGH_to_NORMAL transition's apparent concurrency-interaction refinement.

## 30. What is exploratory

The strategy-combination grouping (data-driven, not preregistered a priori). Pre-stress T-1 exposure characterization (not formally tested against a normal-day comparison population this phase).

## 31. What is contradicted

Concurrency 4+ as a standalone, temporally-stable marginal predictor of daily R — its sign flips between the earlier and later half of the historical record, directly contradicting a naive reading of its marginal-only appearance in §7.

## 32. What remains unknown

Whether the concurrency-within-stress concentration (§24) would hold under a genuinely out-of-sample test (this phase's temporal validation tested marginal factors, not this specific descriptive counterfactual). Intraday (T-60/30/15-minute) pre-stress exposure. Whether the JPY-controlling-for-vol/concurrency pattern (§15) would strengthen or weaken with a larger, more recent sample.

## 33. Future research hypotheses (NOT tested, NOT implemented)

1. Independently test the concurrency-within-stress concentration finding (§24) via genuine temporal out-of-sample validation, not just a single-period descriptive counterfactual.
2. Investigate the JPY-controlling-for-vol/concurrency conditional pattern (§15) as a standalone research question, given it newly survived temporal validation here.
3. Investigate the HIGH_to_NORMAL-transition-as-concurrency-interaction refinement (§11) with a larger sample.
4. If a future portfolio-control phase is ever undertaken, design it around selectivity-within-the-stress-population specifically (per `reports/phase49_phase44_connection.md`), not broad prospective suppression, given Phase 44's specific failure mode.

## 34. Limitations

- The OLS model (§19) has genuinely low explanatory power (R²=0.038) and disclosed multicollinearity among the three mechanism-share predictors — its coefficients should be read as suggestive, not definitive.
- The concurrency-within-stress descriptive counterfactual (§24) is the single most striking number in this report but was not independently temporally validated as a stress-population-specific relationship (only the marginal concurrency effect was temporally tested, and that one did not survive).
- Strategy-combination analysis (§13) is data-driven/exploratory, not preregistered a priori, and should not be treated with the same confidence as the primary preregistered tests.
- Pre-stress exposure (§10) lacks the intraday granularity the task requested; this is a genuine, disclosed data limitation of this project's trade-level ledger, not a computational shortcut.

## 35. Final verdict

### Answers to the 25 required questions

1. **What characterizes the worst days?** Elevated volatility, elevated concurrency, elevated strategy-count/simultaneous-JPY/AMR positions (all MODERATE marginal effects) — and, most sharply, worst-day losses concentrate heavily (~91%) in the concurrency≥4 subset of the stress population specifically.
2. **Is volatility still the strongest stress factor?** Among temporally-validated marginal factors, yes — the most consistently robust.
3. **Level or transition more informative?** Both matter, and they interact: the HIGH_to_NORMAL transition effect appears substantially concurrency-driven (§11), not a pure independent transition effect.
4. **Does concurrency materially amplify losses?** Not as a standalone marginal factor (does not survive temporal validation) — but decisively yes *within the already-stressed population* (§24's descriptive counterfactual).
5. **Does JPY exposure independently explain stress?** Not marginally (PLAUSIBLE only) — but yes, conditionally, once controlling for volatility and concurrency (MODERATE, temporally validated) — a genuinely new finding.
6. **Does AMR exposure independently explain stress?** Weakly and specifically within HIGH volatility; not temporally stable in magnitude.
7. **Does GBPJPY_AMR drive or co-occur with stress?** Co-occurs and correlates strongly (0.727) — does not independently, dramatically predict worse days by its mere presence.
8. **Does direction matter?** Yes in ordinary conditions (long-heavy outperforms); the asymmetry largely disappears on stress days themselves.
9. **Does session matter?** The portfolio has zero NY exposure — a structural fact, not a stress-specific finding.
10. **Which strategy combinations are most associated with stress?** The full four-AMR-pair combination (most common, 184 days) shows a modestly elevated stress-day rate — EXPLORATORY only.
11. **Isolated or clustered?** MIXED (36.4% clustering rate) — neither cleanly.
12. **Concentrated or spread across strategies?** PROPORTIONAL — spread across strategies, confirmed independently.
13. **How much of the worst 1% comes from simultaneous losses?** 100% of worst-1% days have 2+ simultaneous losers, averaging 5.1 per day.
14. **Can pre-stress exposure identify conditions before losses occur?** Not conclusively established this phase (T-1 exploratory only; intraday unavailable).
15. **Does any relationship survive temporal validation?** Yes — HIGH volatility, JPY exposure (conditional), and AMR exposure (directionally) all survive; concurrency (marginal, standalone) does not.
16. **Which findings remain only exploratory?** Strategy-combination grouping; pre-stress T-1 characterization.
17. **Does the live losing period resemble historical stress?** Descriptively consistent in composition (high JPY/AMR share) but the sample (n=19) is too small for confidence.
18. **Does Phase 49 provide a credible explanation for portfolio stress?** A multi-factor, partial explanation — not a single dominant mechanism, and even the best combination of factors explains only ~3.8% of daily-R variance.
19. **Does it justify any intervention?** No — per the explicit no-intervention rule, and because no sufficiently stable, prospectively-actionable mechanism was identified (see `reports/phase49_phase44_connection.md`).
20. **What should NOT be changed?** Nothing — this phase implements no change, per its own scope.
21. **What future intervention hypothesis is worth testing?** The four items in §33, none tested here.
22. **Strongest remaining uncertainty?** Whether the concurrency-within-stress concentration (§24) is a genuine, exploitable pattern or an artifact of this specific historical sample — untested out-of-sample.
23. **Is another strategy search justified?** No — unchanged from Phase 39/45/46/48.
24. **What should Phase 50 investigate?** Either of §33's items 1-2 (concurrency-within-stress OOS validation, or the JPY-conditional pattern), the two most concrete new leads from this phase.
25. **Should Phase 50 investigate anything at all?** Optional — no urgent gap forces immediate further work; the evidence base is now unusually complete for a research program of this type.

### Final classification

## **C. MULTI-FACTOR STRESS MECHANISM**

No single dominant factor was found (ruling out B), but the evidence is far from "no clear mechanism" (ruling out A) — volatility, concurrency-within-stress, and a newly-identified conditional JPY effect together characterize the portfolio's worst days better than any one factor alone, though the overall explanatory power remains modest (R²=0.038) and no finding was validated as prospectively actionable (this is **not** D, since no specific, temporally-validated, ready-to-test intervention mechanism emerged — only research leads for a possible future phase).

---

## Safety check confirmation

Preregistration committed (`5b408f2`) before results, unchanged after · historical control unchanged · six strategies unchanged · no strategy parameters changed · no risk changed · no portfolio control deployed · no strategy filter deployed · no optimization performed · stress definition frozen (worst 1/5/10%, computed once) · primary tests separated from the one exploratory test (§13) · multiple testing logged (`reports/phase49_multiple_testing.csv`) · temporal validation performed for all preregistered marginal factors · live data kept separate from historical data throughout · counterfactuals explicitly labeled DESCRIPTIVE, never VALIDATED CONTROL · Phase 44 not overturned or re-optimized (see `reports/phase49_phase44_connection.md`) · raw production 5ers export not committed.

---

*No live trading change authorized. No filter, control, or threshold implemented. 4 future research hypotheses recorded, none tested.*
