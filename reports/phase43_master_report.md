# Phase 43 — Exposure × Volatility Stress Attribution (Master Report)

**FORENSIC / OBSERVATIONAL ANALYSIS ONLY. No new strategy created or backtested. No live strategy, parameter, risk, or portfolio weight modified. No intervention implemented.**

---

## 1. Executive summary

Testing whether Phase 42's MODERATE volatility finding is actually an exposure phenomenon produces a **mixed, genuinely nuanced result — not a clean confirmation**. The single strongest, most consistent finding: **the fraction of stress days that are HIGH-volatility-AND-4+-concurrent-positions rises monotonically with tail severity** (32.3% of worst-20% days → 62.5% of worst-1% days, vs. ~33% baseline expected by chance) — a real tail-concentration signal. But several other hypotheses did **not** confirm the expected direction: exposure accumulated *before* a volatility-expansion event was associated with **better**, not worse, subsequent 3-day performance (+0.47R vs. -0.32R, the opposite of H4's stated hypothesis). Unconditional concurrency (ignoring volatility state) shows **no** simple "more positions = worse" pattern — average R actually *rises* with concurrency level. JPY and AMR exposure are **too collinear with total open-position count in this control** to cleanly separate as independent variables (confirmed directly — several JPY/AMR × exposure combination cells are empty or near-empty). **Final classification: C. MODERATE / PROMISING BUT NOT CONFIRMED** — unchanged from Phase 42's own classification, now with more precision about *which* exposure dimension (concurrency-in-tail-days) carries the signal and which (pre-expansion buildup, unconditional concurrency, JPY/mechanism as separable factors) do not.

## 2. Phase 41 context

H. NO SINGLE DOMINANT FACTOR; HIGH-volatility trade share was the strongest single factor, classified MODERATE.

## 3. Phase 42 context

Volatility-stress relationship confirmed non-monotonic (worst at 80-90th percentile), concentrated in AMR/ARB, with a consistent volatility×concurrency interaction at every threshold (4+/5+/6+). Classification: C. MODERATE / PROMISING BUT NOT CONFIRMED.

## 4. Research question

Is the portfolio's vulnerability actually an exposure problem (position count, open risk, correlated risk, pre-existing exposure, direction, JPY, mechanism) rather than volatility per se?

## 5. Preregistration

`reports/phase43_preregistration.md`, committed separately (`fae47d2`) before any analysis. No amendment required.

## 6. Data integrity

Both source files validated clean. Trade count (2,712) reconciled against Phases 41/42. **One implementation bug was caught and fixed before results were interpreted**: a timezone-aware/naive datetime comparison error in the volatility-expansion-event lookback calculation — fixed by explicitly localizing the daily ledger's date index to UTC before comparison; this is a code-correctness fix, not a methodology change, and no result existed under the buggy version before it was corrected.

## 7. Control portfolio

Identical to Phases 41/42 — unchanged.

## 8. Exposure definitions

Per the preregistration, **total open risk in R equals open-position count for this dataset** (fixed fractional-risk-per-trade sizing, confirmed directly: `corr(open_risk_R, position_count) = 1.0`, `reports/phase43_open_risk.csv`) — H2 as originally framed (position count vs. open risk as *separate* variables) is **not testable** with this dataset; this is itself an informative finding about this control's structure, not a gap.

## 9. Exposure at entry

`reports/phase43_exposure_at_entry.csv` (2,712 trades, per-trade prior-exposure state). Raw pattern: mean R at entry rises with prior open-position count (0-1: 0.052R, 2: 0.061R, 3: 0.077R, 4: 0.174R) — the opposite of "more prior exposure predicts worse subsequent trades" at the individual-trade level. Only 2 trades ever entered with 5 already open, and none with 6+ — the control's own AMR/ARB entry logic appears to self-limit concurrent exposure below 6.

## 10. Exposure before volatility expansion (H4)

`reports/phase43_exposure_before_vol_expansion.csv`, 106 volatility-expansion events (Phase 42's `LOW_to_HIGH`/`NORMAL_to_HIGH` transitions). **High pre-expansion exposure (n=68) shows subsequent 3-day R averaging +0.469, while low pre-expansion exposure (n=38) averages -0.317** — a MODERATE effect size (0.79R) but in the **opposite direction from H4's stated hypothesis** ("exposure before expansion is more dangerous"). This is reported honestly as a genuine, if counter-intuitive, finding, not suppressed or reframed.

## 11. Position count (H1)

`reports/phase43_position_count.csv`. HIGH-vol + 4-position bucket shows the worst single cell in the entire table (-0.217R, n=86); HIGH-vol + 6+ shows a near-zero result (+0.017R, n=14, thin). Aggregating HIGH-vol+4-or-more (n=182) vs. HIGH-vol+0-to-3 (n=76): **effect size 0.306R, classified WEAK** by the phase's conservative sample-adjusted rule, though the specific 4-position cell is the most extreme.

## 12. Total open risk

`reports/phase43_open_risk.csv`. Degenerate — see §8.

## 13. Correlated open risk (H3)

`reports/phase43_correlated_risk.csv`. Average maximum-currency-concentration is **already very high on normal days** (0.953) and rises only modestly through the stress buckets (0.954 → 0.961 → 0.969 → 0.961, non-monotonic across the four stress tiers) — consistent with Phase 41/42's saturation finding: currency concentration in this JPY-heavy control has little room to differentiate stress from normal days.

## 14. Directional exposure (H6)

`reports/phase43_directional_exposure.csv`. **A refinement of Phase 42's short-side finding**: short-heavy days are negative in *every* volatility state (LOW -0.034R, NORMAL -0.179R, HIGH -0.204R) — short-side underperformance appears largely **volatility-independent**, a persistent structural weakness, not a HIGH-vol-specific interaction. Long-heavy days, by contrast, **do** degrade substantially with volatility (LOW +0.540R → NORMAL +0.530R → HIGH +0.110R) — the volatility-sensitivity Phase 42 attributed to "the short side" appears, on this more granular breakdown, to be more attributable to the **long side's own degradation** in HIGH volatility.

## 15. JPY exposure (H7)

`reports/phase43_jpy_exposure.csv`. **A collinearity problem, not a clean test**: the "HIGH-vol + high-JPY + low-open-risk" cell is **empty (n=0)** — in this control, high JPY exposure and high total exposure occur together by construction (since ~94% of all positions are JPY-linked at any given time, per Phase 41). HIGH-vol+high-JPY+high-open-risk (n=164) averages -0.102R; the two low-open-risk cells (regardless of JPY) both average positive (+0.378R, +0.251R). This is consistent with exposure (not JPY specifically) being the operative variable, but cannot be cleanly disentangled given the collinearity.

## 16. Mechanism exposure (H8)

`reports/phase43_mechanism_exposure.csv`. Same collinearity pattern as JPY: the "low-AMR + high-open-risk" cell has only 3 observations (too thin to interpret). HIGH-vol+high-AMR+high-open-risk (n=179, the best-sampled cell) averages -0.075R — consistent with, but not independently confirming beyond, Phase 42's AMR-concentration finding.

## 17. Exposure build-up

`reports/phase43_exposure_build_up.csv`. Around the worst 5 individual days, average open-position count rises modestly from T-24h (0.4) to a peak around T-8h to T-12h (1.2-1.4), then declines toward the event. **This is a very thin sample (5 event-days)** and the absolute position counts involved are small — reported as **EXPLORATORY, INSUFFICIENT for a confident build-up conclusion**, not a confirmed pattern.

## 18. Exposure decay

`reports/phase43_exposure_decay.csv`. Position count after the worst days' peak drops to near-zero within 12-24 hours and stays low through 72 hours (with one modest bounce at 48h) — consistent with the portfolio **not persistently over-exposed** after a stress episode; the dangerous state (to the extent build-up is real at all) appears **temporary, not persistent**. Same thin-sample caveat as §17.

## 19. Concurrent-position lifecycle

`reports/phase43_concurrency_lifecycle.csv`. **Unconditionally** (not conditioned on volatility), average R actually **rises** with concurrency level (level 1: 0.076R → level 5: 0.174R) and loss probability **falls** (34.1% → 28.8%) — directly contradicting a simple "more concurrent positions = worse" story when volatility is not accounted for. This reinforces that concurrency alone is not the driver — it is specifically the **HIGH-volatility + high-concurrency combination** (§11) that shows a negative pattern, not concurrency in isolation.

## 20. Count vs. risk

`reports/phase43_count_vs_risk.csv`. Degenerate matrix, per §8 — a single diagonal, not a 2D surface.

## 21. Correlated exposure

`reports/phase43_correlated_exposure_matrix.csv`. Every "no_jpy" cell is empty (n=0) at every position-count level — confirming the control has essentially no non-JPY concurrent-exposure state to compare against, at any concurrency level. This is a structural property of the control (JPY dominance is not merely a "stress day" phenomenon, it is the portfolio's permanent condition), not a new finding beyond §15.

## 22. Volatility × exposure surfaces

`reports/phase43_volatility_exposure_surfaces.csv`. Average position count rises slightly with volatility state (3.47 LOW → 3.94 HIGH); average net directional count (long minus short) falls substantially (1.29 → 0.65) — the portfolio becomes more balanced (relatively more short exposure) as volatility rises, consistent with §14's finding that long-side conviction weakens in HIGH volatility.

## 23. Tail analysis

`reports/phase43_tail_analysis.csv`. **The single most decisive finding of this phase**: the percentage of days that are BOTH HIGH-volatility AND 4+-concurrent-positions rises monotonically and substantially with tail severity — 32.3% (worst-20%) → 34.6% (worst-10%) → 53.8% (worst-5%) → **62.5% (worst-1%)** — nearly double the ~33% baseline rate expected if HIGH-vol-and-4+ were unrelated to tail severity. This concentration-in-the-tail pattern is more compelling than the raw mean-difference effect sizes reported elsewhere in this phase.

## 24. Extreme-day robustness

`reports/phase43_extreme_day_robustness.csv`. The HIGH-vol+4plus vs. non-HIGH-vol+4plus difference remains negative after excluding the worst 1, 5, and 10 days (-0.43R → -0.40R → -0.43R → -0.33R) — **not driven by a single extraordinary event**.

## 25. Historical regime robustness

`reports/phase43_regime_robustness.csv`. NEGATIVE direction (HIGH-vol+4plus worse than non-HIGH-vol+4plus) in **all 3 available historical periods** (2023-2024, 2025, 2026 YTD) — directionally consistent, though the earliest and most recent periods show larger effect magnitudes than 2025. 2019-2022 UNKNOWN BY DATA ABSENCE.

## 26. Post-demotion evidence

`reports/phase43_post_demotion.csv`. 19 live trades, 8 days — explicitly INSUFFICIENT LIVE SAMPLE, not separately testable for this phase's interaction hypotheses.

## 27. Phase 41/42 reconciliation

`reports/phase43_phase41_42_reconciliation.csv`. Phase 43 **REFINES** rather than confirms or rejects Phase 42's finding: the volatility×concurrency interaction survives (and is most visible in the tail-concentration measure, §23), but several of the more specific exposure sub-hypotheses (pre-expansion buildup, unconditional concurrency, JPY/mechanism as independently separable factors) either showed the opposite direction from expected or could not be cleanly tested due to collinearity within this control's structure.

## 28. Multiple testing

8 primary preregistered hypotheses tested per the frozen methodology. H2 and several cells of H7/H8 were found **degenerate/collinear by the data's own structure**, not through any search or threshold-tuning — disclosed as a data-structural finding, not treated as a null result to be explained away.

## 29. Evidence matrix

`reports/phase43_evidence_matrix.csv`. H1: WEAK by mean-difference, but STRONG by the tail-concentration measure (§23) — a genuine tension between two reasonable ways of measuring the same relationship, both reported. H4: MODERATE, but in the opposite direction from the stated hypothesis. H2, H7, H8: DEGENERATE/collinear, not independently testable with this dataset.

## 30. Final exposure classification

### **C. MODERATE / PROMISING BUT NOT CONFIRMED** (unchanged from Phase 42's classification of the underlying volatility relationship).

Phase 43 does not upgrade this to STRONG or CONFIRMED — the tail-concentration finding (§23) is the strongest single piece of evidence, but it coexists with a genuine counter-finding (H4, pre-expansion exposure showing a *better*, not worse, subsequent outcome) and several hypotheses that could not be cleanly tested due to structural collinearity in this specific control (position count ≡ open risk; JPY exposure ≡ total exposure; AMR exposure ≡ total exposure, each nearly by construction).

## 31. What the evidence means

The portfolio's worst days are disproportionately likely to also be HIGH-volatility-AND-high-concurrency days (§23) — a real, tail-concentrated, regime-robust pattern. Within that, the short side shows a persistent (not volatility-specific) weakness, while the long side specifically weakens during HIGH volatility (§14).

## 32. What this does NOT mean

This does **not** establish that reducing position count, capping risk, or filtering entries during HIGH volatility would improve outcomes — per the frozen no-intervention rule, none of that is tested here, and the counter-finding in §10 (pre-expansion exposure associated with *better* subsequent outcomes) specifically argues against assuming any exposure-reduction intervention would obviously help. It also does not mean JPY or AMR/ARB mechanism exposure are independently confirmed drivers — the collinearity found in §15-16 means this phase cannot separate "is it JPY" from "is it just total exposure," and that ambiguity should not be silently resolved in either direction.

## 33. Future research hypotheses (NOT tested)

`reports/phase43_future_research_ideas.csv` — 3 recorded, none implemented: (1) a **dedicated future intervention-testing phase** to investigate whether limiting concurrent HIGH-vol-state positions reduces tail severity (explicitly flagged as requiring its own phase, not extrapolated from this diagnostic evidence); (2) investigating currency-factor concentration as a more precise exposure measure than raw position count; (3) investigating whether pre-existing exposure specifically (as opposed to new entries) drives the H4 finding, given the counter-intuitive direction found here.

## 34. Limitations

- The exposure build-up/decay timeline (§17-18) rests on only 5 event-days and should not be treated as a confirmed temporal pattern.
- H2, H7, and H8's collinearity findings are a property of this specific control's structure (fixed-fractional sizing, near-universal JPY exposure, AMR-dominated volume) and may not generalize to a differently-constructed portfolio.
- The tail-concentration finding (§23) and the mean-difference effect size (§11) point toward somewhat different conclusions about how strong the evidence is — both are reported rather than reconciled into a single number, since they measure genuinely different things (frequency-in-tail vs. average magnitude).
- The H4 counter-finding (§10) is based on 106 expansion events split into two groups (68/38) — an adequate but not large sample for such a specific claim.

## 35. Final verdict

### Answers to the 26 required questions

1. **Is HIGH-vol + high position count associated with materially worse outcomes?** Modestly by mean difference (WEAK, 0.31R), more clearly by tail-concentration (§23, near-doubling in the worst 1%).
2. **Position count or total open risk the stronger variable?** Not separable — identical by construction in this dataset.
3. **Is correlated open risk more informative than total open risk?** Not clearly — currency concentration is already saturated on normal days, leaving little room to differentiate.
4. **Does exposure accumulate before volatility expansion?** Only weak, thin-sample evidence (§17); not confirmed.
5. **Does exposure entered before expansion perform worse?** **No — the opposite.** High pre-expansion exposure was associated with *better* subsequent 3-day performance.
6. **Are entries during volatility transitions disproportionately associated with losses?** See `phase43_exposure_at_entry.csv` — no clear disproportionate pattern found at the individual-trade level; if anything, entries with more prior open exposure showed higher, not lower, mean R.
7. **Is short exposure asymmetry robust after controlling for exposure?** Refined, not simply confirmed — short-side weakness appears largely volatility-independent (a persistent trait), while long-side weakness is the more volatility-sensitive component.
8. **Does JPY matter after controlling for total exposure?** Cannot be cleanly answered — JPY and total exposure are too collinear in this control to separate.
9. **Does AMR/ARB matter after controlling for exposure?** Same collinearity caveat as JPY.
10. **Does the volatility relationship disappear once exposure is controlled?** No — it persists and is, if anything, more visible in the tail-concentration measure.
11. **Survives removal of the worst day?** Yes.
12. **Worst five days?** Yes.
13. **Worst ten days?** Yes, modestly attenuated.
14. **Consistent across historical regimes?** Yes, directionally, in all 3 available periods.
15. **Post-demotion?** Cannot be assessed — insufficient live sample.
16. **Number of positions, amount of risk, or correlated risk?** Cannot be distinguished — identical/collinear in this dataset.
17. **Danger concentrated BEFORE expansion?** No — the opposite direction was found.
18. **DURING transition?** Not separately isolated in this phase beyond the expansion-event test.
19. **AFTER expansion?** Not the focus of this phase's design; §18's decay analysis (thin sample) suggests exposure does not persist elevated afterward.
20. **Stable, interpretable exposure pattern?** Partially — the tail-concentration pattern (§23) is the most stable and interpretable single result; several other sub-hypotheses were inconclusive or collinear.
21. **Or still too unstable?** A fair characterization for the finer-grained sub-hypotheses (H4's direction, H7/H8's collinearity) even though the headline volatility×concurrency pattern itself held up.
22. **Does Phase 43 strengthen the case for portfolio-control research?** Modestly, via the tail-concentration finding — but the H4 counter-finding argues for caution before assuming any specific intervention direction.
23. **Does Phase 43 weaken the volatility hypothesis?** No, but it complicates it — the hypothesis survives as a tail-concentration phenomenon while several of its proposed mechanisms (pre-expansion buildup, JPY/mechanism separability) did not confirm as expected.
24. **What remains unknown?** Whether the tail-concentration pattern reflects a real causal exposure mechanism or a byproduct of this control's own AMR-heavy, JPY-saturated construction; the true build-up/decay dynamics (thin sample); post-demotion behavior.
25. **What should NOT be researched next?** A position-count cap or risk-reduction filter designed and tested as if H4's original hypothesis (pre-expansion exposure is dangerous) were confirmed — it was not; that specific premise is not supported by this phase's evidence.
26. **What should Phase 44 investigate, if anything?** Given the accumulating diagnostic evidence (Phases 41-43) without a portfolio-control intervention yet tested, and per the future-research-ideas recorded here, a natural next step — not decided in this phase — would be a dedicated, narrowly-scoped intervention-testing phase (if the research program chooses to pursue one) that explicitly tests rather than assumes the direction of any exposure-based control, given this phase's own counter-intuitive H4 finding.

---

## Safety check confirmation

Preregistration committed (`fae47d2`) before results, unchanged after · research validator passed · control portfolio unchanged and reconciled against Phase 41/42 (2,712 trades) · no live strategy touched · no strategy parameters changed · no risk settings changed · no portfolio optimization · no position-count optimization · no risk-threshold optimization · no volatility filter · no exposure filter · no strategy deployed · Phase 40/41/42 untouched · multiple testing addressed (8 primary hypotheses, degenerate/collinear cells disclosed rather than hidden) · extreme-day robustness performed · historical regime analysis performed (3 periods, 2 flagged data-absent) · post-demotion separated and labeled INSUFFICIENT · causality warnings included throughout (association/preceded, never "caused") · raw production 5ers export not committed.

---

*No live trading change authorized. No trading strategy or portfolio control produced. 3 future research hypotheses recorded, none tested.*
