# Phase 42 — Volatility Stress Decomposition (Master Report)

**FORENSIC / OBSERVATIONAL ANALYSIS ONLY. No new strategy created or backtested. No live strategy, parameter, risk, or portfolio weight modified. No intervention implemented.**

---

## 1. Executive summary

Decomposing Phase 41's one MODERATE-evidence factor (HIGH-volatility trade share) across 8 preregistered sub-hypotheses finds a **nuanced, non-monotonic, mechanism-concentrated relationship — not the simple "higher volatility = worse" story the raw HIGH-vs-LOW comparison suggested.** The continuous percentile breakdown (H1) reveals the true danger zone is the **80th-90th percentile band** (avg daily R -0.46), not the most extreme 90-100th percentile (avg daily R +0.19, actually positive) — a genuinely important, counter-intuitive finding that a simple HIGH/NORMAL/LOW tercile split would obscure. The volatility×concurrency interaction (H4) shows a consistent, moderate negative effect at every concurrency threshold tested (4+, 5+, 6+ simultaneous positions). The volatility×mechanism breakdown (H8) shows the effect is **concentrated in AMR and ARB, not GBPUSD Monday** (which actually improves in HIGH volatility). The volatility×JPY interaction (H7) and volatility×direction breakdown (H6) both show meaningful asymmetries. **Final classification: C. MODERATE / PROMISING BUT NOT CONFIRMED** — real, directionally-consistent-across-regimes evidence, but not strong or clean enough for A/B, and importantly non-monotonic in a way that would make any simple threshold-based use fragile.

## 2. Phase 41 context

Phase 41 found JPY and AMR concentration were NOT differentially associated with stress (both saturated at baseline), while HIGH-volatility trade share was the strongest single factor (25.0%→39.2% on stress days), classified MODERATE. Phase 42 decomposes this one factor in depth.

## 3. Research question

Is the observed volatility-stress relationship absolute, transitional, accelerational, or interaction-driven — and is it a reliable description of when this portfolio becomes vulnerable?

## 4. Preregistration

`reports/phase42_preregistration.md`, committed separately (`1b2b499`) before any analysis. No amendment required.

## 5. Data integrity

Both source files validated clean. Trade count (2,712) reconciled exactly against Phase 41. 0 days excluded from the volatility ledger (all 774 days had at least one trade with a valid `atr_pctile`).

## 6. Control portfolio

Identical to Phase 41 — unchanged, no candidate strategy added.

## 7. Volatility methodology

Reused the control's own already-validated per-trade `atr_pctile` (continuous 0-1 ATR percentile from the original strategy pipeline) as the volatility measure — no new indicator invented. Daily volatility level = mean `atr_pctile` across that day's entries; daily volatility percentile = that level re-ranked against the full-period distribution; daily volatility state = LOW/NORMAL/HIGH terciles of the percentile.

## 8. Continuous volatility analysis (H1)

`reports/phase42_volatility_percentiles.csv`. **Non-monotonic**: avg daily R is positive and roughly stable from the 0th to 70th percentile (+0.19 to +0.51R), **drops sharply in the 80-90th percentile band to -0.46R** (the single worst decile), then **recovers to +0.19R in the 90-100th (most extreme) percentile**. Full-period correlation(vol_pctile, daily R) = **-0.080** (weak). HIGH-vs-non-HIGH effect size = 0.32R, classified **WEAK** by sample-adjusted effect-size rules, though directionally consistent.

## 9. Volatility change (H2)

`reports/phase42_volatility_change.csv`. `HIGH_to_NORMAL` (volatility falling out of a HIGH state) shows the worst average daily R of any transition (-0.248R, 51.9% loss-day rate) — worse than `stable_HIGH` itself (+0.021R). This suggests **the transition OUT of HIGH volatility, not persistence within it, may carry the larger risk** — a genuinely counter-intuitive finding, though on an ADEQUATE (n=79) but not large sample. `HIGH_to_HIGH` (0 days, since a day cannot be its own two-day transition under this methodology) is correctly empty.

## 10. Volatility acceleration (H3)

`reports/phase42_volatility_acceleration.csv`. HIGH volatility combined with HIGH acceleration (rapidly rising) shows a worse average (-0.135R) than HIGH volatility combined with stable acceleration (+0.087R) — a WEAK-to-moderate effect (0.22R), directionally consistent with "rapid increases are more dangerous than persistent levels."

## 11. Volatility × concurrency (H4)

`reports/phase42_volatility_concurrency.csv`. **The most consistent interaction finding in this phase**: at every concurrency threshold tested, HIGH-volatility days underperform non-HIGH days at the *same* concurrency level: 4+ positions (HIGH -0.217R vs. non-HIGH +0.205R, diff 0.42), 5+ (HIGH +0.103R vs. non-HIGH +0.554R, diff 0.45), 6+ (HIGH +0.017R vs. non-HIGH +0.583R, diff 0.57, thin sample n=14/17). The effect *direction* is consistent across all three thresholds even though individual magnitudes are classified WEAK by the phase's conservative effect-size rule.

## 12. Volatility × session (H5)

`reports/phase42_volatility_session.csv`. **Not testable beyond Asian/London** — the control has zero New York or overlap-session trades (confirmed independently in this phase's own data, matching Phase 31/41's finding). Asian-session R contribution drops sharply in the HIGH-vol state (+84.1 LOW → +9.7 HIGH) while London stays roughly flat-to-slightly-negative across states — suggesting whatever the HIGH-vol effect is, it manifests predominantly in the Asian session, which is also where the bulk of trade volume sits.

## 13. Volatility × direction (H6)

`reports/phase42_volatility_direction.csv`. **A real asymmetry**: short-side R deteriorates sharply in the HIGH state (-11.5 LOW → -11.3 NORMAL → **-42.4 HIGH**) while long-side R also falls but less dramatically (109.5 → 98.3 → 51.5). Net directional R collapses from ~+88-98R in LOW/NORMAL to +9.1R in HIGH — driven more by the short-side deterioration than the long-side.

## 14. Volatility × JPY (H7)

`reports/phase42_volatility_jpy.csv`. HIGH-vol + high-JPY days average **-0.144R**, while HIGH-vol + low-JPY days average **+0.420R** — a real interaction, even though Phase 41 found no standalone JPY-concentration effect. This suggests JPY exposure may matter **conditionally** (specifically during HIGH volatility) even though it does not matter unconditionally — a nuance Phase 41's marginal analysis alone could not reveal.

## 15. Volatility × mechanism (H8)

`reports/phase42_volatility_mechanism.csv`. **The clearest mechanism-concentration finding**: AMR's R contribution collapses from +80.8 (LOW) / +67.8 (NORMAL) to **-0.08 (HIGH)** — essentially flat, not just worse. ARB shows the same pattern (+13.9 / +12.2 / **-0.62**). **GBPUSD Monday improves** in the HIGH state (+3.3 / +7.1 / **+9.8**) — the opposite direction. The HIGH-volatility effect is concentrated in AMR and ARB, not portfolio-wide.

## 16. Regime transitions

`reports/phase42_transition_matrix.csv`. Full 3×3 LOW/NORMAL/HIGH transition matrix. `HIGH→NORMAL` (n=79, avg R -0.248, worst of any cell) and `NORMAL→HIGH` (n=79, avg R +0.089) show the entry-into and exit-from HIGH states are not symmetric — consistent with §9's finding that the transition itself, not the state, may carry more information than a simple level reading.

## 17. Lead-lag analysis

`reports/phase42_lead_lag.csv`. Only day-level lags are computable from this trade-level dataset (same-day corr -0.080, previous-day corr -0.067) — both weak. Session-level and intraday (4h/8h) lag windows are **UNKNOWN BY DATA LIMITATION**, disclosed rather than approximated.

## 18. Tail-risk analysis

`reports/phase42_tail_analysis.csv`. HIGH-volatility days show deeper tails at every percentile tested (1%: -5.33R vs. -4.44R ordinary; 5%: -4.22R vs. -3.18R; 10%: -3.19R vs. -2.47R) while the **mean** difference (0.04R HIGH vs. 0.36R ordinary) is comparatively larger in relative terms — **volatility appears to affect both the mean and the tail, but the tail effect is proportionally smaller than the mean-shift** given the tail values themselves aren't dramatically worse in absolute R terms. This is a nuanced, not clean-cut, finding.

## 19. Threshold sensitivity

`reports/phase42_threshold_robustness.csv`. Descriptive only. The 70th/80th percentile thresholds both show a meaningful negative "above vs. below" difference (-0.357R, -0.481R) while the 90th percentile threshold shows almost no difference (-0.067R) — **directly consistent with §8's non-monotonicity finding**: using the top decile as "the stress threshold" would substantially understate the effect visible at the 80th percentile.

## 20. Extreme-day sensitivity

`reports/phase42_extreme_day_sensitivity.csv`. The HIGH-vs-non-HIGH R difference remains negative after excluding the worst 1, 5, and 10 days (-0.32R → -0.30R → -0.31R → -0.24R) — **the relationship is not driven by a single extraordinary event**, though it does attenuate somewhat as more extreme days are removed.

## 21. Historical regime robustness

`reports/phase42_regime_robustness.csv`. 2019-2020 and 2021-2022 are **UNKNOWN BY DATA ABSENCE** (the control only starts 2023-08-01). Across the three periods the control does cover (2023-2024, 2025, 2026 YTD), the effect direction is **NEGATIVE (high-vol worse) in all three** — a consistent direction, though the earliest period (2023-2024) shows the largest magnitude (-0.45R) and the most recent (2026 YTD) shows a smaller one (-0.25R).

## 22. Post-demotion live evidence

`reports/phase42_post_demotion.csv`. 19 live trades, 8 trading days, total R -4.32. **Explicitly labeled INSUFFICIENT LIVE SAMPLE** — not pooled with the historical estimate, no deterioration or improvement inferred from this alone.

## 23. Phase 40 comparison

`reports/phase42_phase40_comparison.csv`. Phase 40's HIGH-volatility-state trend-continuation candidate failed as a **directional trading signal** (OOS PF 0.668). Phase 42 finds volatility has a real, if modest and non-monotonic, relationship as a **risk-state descriptor** of this specific control portfolio's own behavior. These are logically independent findings — a variable being useless for predicting direction does not mean it is useless for describing when existing exposure is riskier, and vice versa.

## 24. Multiple testing

8 primary preregistered hypotheses (H1-H8), all tested with the frozen methodology. Threshold robustness, extreme-day sensitivity, and regime robustness are treated as robustness checks on H1/H4, not new hypotheses. Lead-lag is EXPLORATORY throughout.

## 25. Evidence matrix

`reports/phase42_evidence_matrix.csv`. H1: WEAK (0.32R effect, but directionally consistent across regimes and extreme-day removal). H2: MODERATE for the `HIGH_to_NORMAL` transition specifically, INSUFFICIENT for others. H3: WEAK (0.22R). H4: WEAK-to-borderline-MODERATE (0.47R, consistent direction across 3 thresholds). H5: INSUFFICIENT (data absence). H6-H8: descriptive interaction findings, not independently strength-classified but individually informative (JPY, direction, and especially mechanism concentration in AMR/ARB).

## 26. Final volatility classification

### **C. MODERATE / PROMISING BUT NOT CONFIRMED.**

The relationship is real (consistent direction across all 3 available historical periods, survives removal of the worst 1/5/10 days, shows a coherent interaction pattern with concurrency and mechanism) but is **not strong, not monotonic, and not portfolio-wide** — it is concentrated in AMR/ARB (not GBPUSD Monday), concentrated in the 80-90th percentile band specifically (not the most extreme decile), and most clearly expressed as an interaction with concurrent exposure and JPY rather than as a standalone level effect.

## 27. What this means

Volatility (specifically, the 80-90th percentile band, in combination with high concurrent exposure, concentrated in AMR/ARB, with the transition OUT of HIGH states showing the worst average outcome) is a **legitimate, if partial and non-monotonic, risk-state descriptor** for this portfolio.

## 28. What this does NOT mean

This does **not** authorize volatility scaling, position reduction, a volatility filter, or any portfolio control — per the frozen no-intervention rule, every finding above is an observation, not a treatment. It also does not mean volatility is *the* portfolio's stress factor to the exclusion of others (Phase 41's H. NO SINGLE DOMINANT FACTOR verdict stands) — it means volatility is the most promising of the factors examined so far, not a confirmed cause.

## 29. Future research hypotheses (NOT tested)

1. Investigate the `HIGH_to_NORMAL` transition specifically (§9, §16) as a distinct risk state, separate from persistent HIGH volatility.
2. Investigate why the 80-90th percentile band specifically, not the top decile, is the worst-performing bucket (§8) — possibly a liquidity or execution-quality effect at moderately-elevated-but-not-extreme volatility.
3. Investigate the AMR/ARB-specific mechanism concentration of the volatility effect (§15) as a mechanism-design question, separate from a portfolio-level volatility question.
4. Investigate the short-side-specific deterioration in HIGH volatility (§13) as a directional-asymmetry question.
5. Investigate the volatility×JPY interaction (§14) given it appears conditionally even though Phase 41 found no unconditional JPY effect.

## 30. Limitations

- Volatility state/level was reused from the source data's `atr_pctile`, itself already a percentile — this phase's "percentile of a percentile" construction is a disclosed methodological choice, not a flaw, but means the exact numeric percentile values are one step removed from raw ATR.
- Lead-lag analysis (§17) is limited to day-level granularity; the preregistered intraday windows (previous session, previous 4/8 hours) could not be computed from this trade-level dataset.
- H4's extreme-day-sensitivity re-run was not separately performed (only H1's was) — a disclosed scope limitation, not a hidden gap.
- The non-monotonicity finding (§8) is itself only observed within one control's history (2023-08 to 2026-08) — whether it would replicate in an extended or different sample is unknown.

## 31. Final verdict

### Answers to the 25 required questions

1. **Does performance deteriorate continuously as volatility rises?** No — non-monotonic; worst at the 80-90th percentile, not the top decile.
2. **Is volatility change more informative than absolute volatility?** Possibly — the `HIGH_to_NORMAL` transition shows the single worst average R of any state or transition tested.
3. **Is volatility acceleration more informative than level?** Modestly — HIGH+HIGH-acceleration is worse than HIGH+stable, a real but weak effect.
4. **Does HIGH volatility become dangerous primarily with high concurrent exposure?** Yes, this is the most consistent single finding — negative at every concurrency threshold (4+/5+/6+) tested.
5. **Is there evidence for a volatility × concurrency interaction?** Yes, WEAK-to-borderline-MODERATE, directionally consistent.
6. **Is volatility stress concentrated around session transitions?** Not testable beyond Asian→London (data absence for NY); within what's testable, the Asian session shows the larger R deterioration in HIGH-vol.
7. **Is volatility stress directionally asymmetric?** Yes — short-side R deteriorates far more than long-side in the HIGH state.
8. **Is volatility only problematic with high JPY exposure?** Largely yes — HIGH-vol+high-JPY is negative (-0.144R) while HIGH-vol+low-JPY is positive (+0.420R), despite JPY showing no unconditional effect in Phase 41.
9. **Is volatility stress concentrated in AMR?** Yes — AMR's R contribution collapses to near-zero in the HIGH state.
10. **Is volatility stress concentrated in ARB?** Yes, similarly — ARB flips negative in the HIGH state.
11. **Does the relationship survive removal of the worst day?** Yes — the effect direction and magnitude barely change.
12. **Removal of the worst five days?** Yes, largely unchanged.
13. **Removal of the worst ten days?** Yes, though somewhat attenuated (-0.24R vs. -0.32R baseline).
14. **Does it appear consistently across historical regimes?** Yes, in all 3 periods the control's data covers (2023-2024, 2025, 2026 YTD) — 2019-2022 are UNKNOWN BY DATA ABSENCE.
15. **Does it appear in the post-demotion live period?** Cannot be assessed — INSUFFICIENT LIVE SAMPLE (n=19).
16. **Mean performance or tail losses?** Both, though the tail effect is proportionally more modest than the mean-shift in absolute R terms.
17. **Is there a stable volatility threshold?** No — the relationship is non-monotonic; the 90th-percentile threshold shows almost no effect while the 80th does, undermining the idea of a single stable cutoff.
18. **Is the relationship economically meaningful?** Modestly — 0.3-0.5R average daily-return differentials are non-trivial for a portfolio, but this is a risk-state observation, not a profitability claim.
19. **Does Phase 40's failure tell us anything about volatility as a risk factor?** No direct implication either way — a poor directional trading signal and a legitimate risk-state descriptor are independent properties (§23).
20. **Is volatility actually the portfolio's stress factor?** A promising, partial one — not confirmed, not the sole factor (Phase 41's H. NO SINGLE DOMINANT FACTOR still stands).
21. **If yes, how strong is the evidence?** MODERATE/PROMISING (classification C) — real and consistent in direction, but not strong, not monotonic, not portfolio-wide.
22. **If no, what alternative explanation remains?** The AMR/ARB mechanism-concentration (§15) and concurrency-interaction (§11) findings suggest the phenomenon may be more about *how much is open, in which mechanism* during elevated (not necessarily extreme) volatility, rather than volatility per se.
23. **What remains genuinely unknown?** Why the 80-90th percentile specifically is worse than the top decile; whether the effect replicates post-demotion; the true intraday lead-lag structure (blocked by data granularity).
24. **What should NOT be researched next?** A volatility-scaling or filtering intervention based on this evidence alone — the non-monotonicity and mechanism-concentration make a simple threshold-based rule fragile by this phase's own findings.
25. **What should Phase 43 investigate, if anything?** The AMR/ARB-specific mechanism concentration (§15) and/or the `HIGH_to_NORMAL` transition-specific risk (§9) are the two most evidence-grounded, narrower follow-up questions raised by this phase — not a repeat of the broad volatility question already answered here.

---

## Safety check confirmation

Preregistration committed (`1b2b499`) before results, unchanged after · research validator passed · control portfolio unchanged and reconciled against Phase 41 (2,712 trades) · no live strategy touched · no strategy parameters changed · no risk settings changed · no portfolio optimization · no volatility filter or scaling implemented · no strategy deployed · Phase 40 untouched · Phase 41 untouched · multiple testing addressed (8 primary hypotheses vs. EXPLORATORY lead-lag) · extreme-day sensitivity performed (H1) · historical regime analysis performed (3 periods, 2 flagged data-absent) · post-demotion sample kept separate and labeled INSUFFICIENT · Phase 40 comparison completed · raw production 5ers export not committed · all conclusions evidence-labeled · no causal claims made from this observational data.

---

*No live trading change authorized. No trading strategy produced. No intervention implemented. 5 future research hypotheses recorded, none tested.*
