# Phase 44 — Portfolio-Control Counterfactual Validation (Master Report)

**HISTORICAL COUNTERFACTUAL RESEARCH ONLY. No live strategy, parameter, risk, or portfolio weight modified. No control deployed. IN-SAMPLE COUNTERFACTUAL EVIDENCE throughout — not out-of-sample validation.**

---

## 1. Executive summary

None of the four frozen portfolio controls clears the preregistered multi-criterion success bar. **Control B (HIGH-volatility 50% alternating new-entry suppression)** shows a striking headline trade-off — a real 30.4% max-drawdown reduction and 30.0% worst-5-day reduction for only a 1.9% total-return sacrifice — but fails on closer inspection: **60.8% of its 418 suppressed trades were historical winners**, meaning the control removes activity broadly rather than selectively avoiding bad trades, and it is **regime-fragile** — in the most recent period (2026 YTD) it produced both *worse* total R (58.35 vs. 60.40) and *worse* max drawdown (-14.96 vs. -13.98) than baseline. **Controls C, D, and E show no meaningful drawdown benefit at all** (C is negative, D is small and return-negative, E is negligible — it suppressed only 2 trades out of 2,712). **Final verdict: NO PORTFOLIO CONTROL JUSTIFIED** among the four tested — none reaches "A. HISTORICALLY PROMISING." This is a valid, informative negative result, not a research failure.

## 2. Phase 41-43 context

Phase 41: H. NO SINGLE DOMINANT FACTOR. Phase 42: volatility relationship real but non-monotonic, C. MODERATE/PROMISING BUT NOT CONFIRMED. Phase 43: exposure×volatility tail-concentration confirmed but a pre-expansion-exposure counter-finding argued against assuming any exposure-reduction intervention would help. Phase 44 tests that warning directly.

## 3. Research question

Can any pre-declared portfolio control improve historical tail risk without destroying the return stream — enough to justify a future out-of-sample validation phase?

## 4. Preregistration

`reports/phase44_preregistration.md`, committed separately (`6098577`) before any counterfactual was run. No amendment required. **One implementation bug was caught and fixed before results were interpreted or reported**: the drawdown-reduction and worst-5-day-reduction percentage formulas initially compared signed (negative) drawdown values directly rather than their absolute magnitudes, producing an inverted sign. This was a code-correctness fix (verified by hand-computing the correct percentages from the already-computed, unmodified control metrics), not a methodology change — no threshold, control definition, or classification rule was altered.

## 5. Data integrity

Both source files validated clean. Trade count (2,712) reconciled against Phases 41-43.

## 6. Control portfolio

Identical to Phases 41-43 — unchanged.

## 7. Counterfactual methodology

Each of Controls B-E suppresses (excludes from the counterfactual portfolio) specific historical trades based on the frozen rule, evaluated using portfolio state immediately before each trade's own entry (reusing Phase 43's `open_positions_at` methodology exactly). No SL/TP/entry/exit/holding-time is ever modified; no synthetic trade is created.

## 8. Baseline (Control A)

`reports/phase44_baseline.csv`. Total R 194.11, max drawdown -29.07R, worst day -6.19R, worst 5-day -16.15R, PF 1.211, 2,712 trades — this is the exact same control used throughout Phases 31-43.

## 9. High-volatility control (Control B)

`reports/phase44_high_vol_control.csv`. 418 trades suppressed (15.4%). Total R 190.43 (-1.9%), max drawdown -20.23R (**30.4% better**), worst 5-day -11.30R (**30.0% better**). The headline numbers look attractive — but see §13 and §16.

## 10. High-volatility × concurrency control (Control C)

`reports/phase44_high_vol_concurrency_control.csv`. 163 trades suppressed (6.0%). Total R 177.86 (-8.4%), max drawdown **-30.37R — actually worse than baseline** (-4.5%, a negative "reduction"). The specific cell Phase 43 flagged as the worst (HIGH-vol + concurrency≥4) turns out, when *all* trades meeting that condition are removed from the counterfactual, to leave the *remaining* portfolio with a **deeper**, not shallower, drawdown — an important, counter-intuitive finding: the worst-cell diagnosis from Phase 43 does not translate into a successful suppression rule.

## 11. Transition control (Control D)

`reports/phase44_transition_control.csv`. 275 trades suppressed (10.1%). Total R 213.68 (**+10.1%, actually improves** — because the suppressed HIGH_to_NORMAL-day trades were, on net, worse than average), max drawdown -27.42R (a modest 5.7% improvement), worst 5-day unchanged (-16.15R, identical to baseline — the transition-day suppression did not touch the days that produced the worst 5-day window).

## 12. Defensive control (Control E)

`reports/phase44_defensive_control.csv`. Only **2 trades suppressed out of 2,712** (0.1%) — the concurrency≥5 threshold is so rarely binding in this control that the intervention is functionally inert. Total R and max drawdown are essentially unchanged from baseline.

## 13. Suppressed-trade analysis

`reports/phase44_suppressed_trades.csv`. **The single most important disqualifying finding of this phase**: every control that suppressed a meaningful number of trades removed a *majority* of historical winners along with losers — Control B: 60.8% winners, Control C: 65.6% winners, Control D: 58.5% winners. None of the four controls is selectively removing bad trades; each is closer to an unconditional activity reduction that happens to shift the return distribution, not a risk-targeted filter.

## 14. Stress-period comparison

`reports/phase44_stress_comparison.csv`, using the Phase 41-frozen stress windows (fixed from baseline, never redefined post-hoc). At the worst-5% bucket: baseline -173.06R, Control B -131.26R (best), Control C -142.97R, Control D -141.78R, Control E -173.06R (unchanged, as expected given only 2 suppressed trades).

## 15. Trade-off analysis

`reports/phase44_tradeoff_analysis.csv` (corrected). Control B's trade-off ratio (drawdown-reduction-% per return-%-sacrificed) is 16.06 — numerically the most attractive of the four — but this ratio alone does not satisfy the preregistered multi-criterion bar (§9 of the preregistration), which also requires selectivity (§13) and regime robustness (§16).

## 16. Historical regime robustness

`reports/phase44_regime_robustness.csv`. **Control B fails this criterion directly**: in the 2026 YTD period, Control B produced *worse* total R (58.35 vs. baseline 60.40) and *worse* max drawdown (-14.96 vs. baseline -13.98) than doing nothing — the exact opposite of its aggregate-period result. This is precisely the kind of regime-inconsistency the preregistration's Criterion D was designed to catch. Control C similarly deteriorates in 2026 YTD (total R 48.74 vs. 60.40, max drawdown -19.02 vs. -13.98 — worse on both dimensions). Control D is the only intervention that improves in 2026 YTD as well as in aggregate (total R 71.95 vs. 60.40, max drawdown -9.08 vs. -13.98) — though its aggregate benefit was already the smallest of the four (§11) and this is a single-regime result on a small sample. 2019-2022 UNKNOWN BY DATA ABSENCE.

## 17. Extreme-day robustness

`reports/phase44_extreme_day_robustness.csv`. After excluding the worst 5 baseline days, Control B's drawdown advantage over baseline narrows substantially (-16.96R vs. baseline's -23.33R, still an improvement but a smaller one) and Control D continues to show the best absolute numbers (total R 236.11, max drawdown -21.94R) among all controls at this exclusion level.

## 18. Cost sensitivity

`reports/phase44_cost_sensitivity.csv`. **Not computable from this dataset** — `r_multiple`/`pnl` do not separately expose each trade's cost component, and Phase 44 does not re-simulate trades (only suppresses/retains historical ones). Disclosed as a genuine limitation, not fabricated.

## 19. Monte Carlo

`reports/phase44_monte_carlo.csv`. **SIMULATED.** Baseline's actual max drawdown (-29.07R) sits at only the 3.9th percentile of its own 10,000-draw reshuffled distribution — i.e., baseline's historical drawdown sequencing was itself somewhat unlucky relative to a random reshuffle of the same trades. Control B's actual drawdown (-20.23R) sits at the 18.1th percentile of *its own* reshuffled distribution — less unlucky, but this reflects Control B trading fewer total trades (2,294 vs. 2,712), not necessarily a structurally different risk profile. Control D sits at the 1.7th percentile of its own distribution — its actual sequencing was, if anything, more fortunate than typical, another reason to treat its apparent improvement cautiously.

## 20. False-positive limitations

`reports/phase44_false_positive_check.csv`. Every finding in this phase is **IN-SAMPLE COUNTERFACTUAL EVIDENCE** — Controls C, D, and E were constructed using thresholds and definitions derived directly from Phase 42/43's analysis of this exact same historical sample. This is a real limitation that would need to be addressed by genuine out-of-sample testing before any finding here could be trusted, even for a control that had cleared every other bar.

## 21. Evidence matrix

`reports/phase44_evidence_matrix.csv` (corrected). All four controls land at REJECTED classifications — see §22 for the precise sub-classification of each, refined beyond the mechanically-generated CSV labels using the regime-robustness evidence from §16.

## 22. Final classification

- **Control B (HIGH-vol 50% suppression): E. REJECTED — FRAGILE / REGIME-DEPENDENT.** The aggregate-period drawdown/return trade-off is numerically the most attractive of the four, but it inverts in the most recent regime (2026 YTD, worse on both R and drawdown), and 60.8% of suppressed trades were historical winners — the improvement is not evidence of selective risk avoidance and does not survive regime robustness.
- **Control C (HIGH-vol + concurrency≥4): C. REJECTED — NO MEANINGFUL BENEFIT.** Aggregate drawdown is actually worse (not better) than baseline; also fails regime robustness in 2026 YTD.
- **Control D (HIGH_to_NORMAL transition): B. MIXED / INSUFFICIENT.** The only control with a positive result in both the aggregate period and the 2026 YTD regime — but the aggregate drawdown improvement is small (5.7%), the worst-5-day metric is completely unchanged from baseline, 58.5% of suppressed trades were historical winners, and the Monte Carlo result (1.7th percentile) suggests the historical sequencing may itself be somewhat fortunate. Interesting enough to record as a future research thread, not strong enough for "HISTORICALLY PROMISING."
- **Control E (concurrency≥5, exposure-agnostic): C. REJECTED — NO MEANINGFUL BENEFIT.** Functionally inert (2 trades suppressed out of 2,712); the concurrency≥5 condition essentially never binds in this control.

## 23. What the evidence means

None of the four broad, pre-declared portfolio-control archetypes motivated by Phases 41-43's diagnostic findings produces a historical improvement robust enough, on its own terms (selectivity, regime consistency, tail robustness), to warrant further validation as currently specified.

## 24. What this does NOT mean

This does **not** mean no portfolio control could ever work — it means these four specific, broadly-defined rules do not. It also does not mean Phase 42/43's underlying diagnostic findings (the volatility×concurrency tail-concentration, the HIGH_to_NORMAL transition weakness) were wrong — a diagnostic finding can be real and directionally correct while a simple rule built to exploit it still fails to produce a net-beneficial, selective, regime-robust intervention, exactly as Control C's result (built directly on Phase 43's own worst-cell finding) illustrates.

## 25. Future validation requirements

`reports/phase44_future_validation.csv`. Any control that had reached "A. HISTORICALLY PROMISING" would still require genuine out-of-sample walk-forward or paper-trading validation before any live consideration — moot here since none reached that bar, but recorded as the standing requirement for any future phase that does produce a promising result.

## 26. Future research ideas

Three recorded, none implemented: (1) a milder (e.g., 25%) HIGH-vol suppression fraction as an independently preregistered future test, given Control B's directionally interesting but ultimately disqualified result; (2) investigating whether suppressed-winner trades cluster in specific strategies, which could inform a more selective future control design than the deliberately broad, exposure-agnostic rules tested here; (3) none of these are tested in this phase.

## 27. Limitations

- The suppressed-trade selection formula (alternating 2nd-entry for Control B) is one specific deterministic choice among many possible 50%-reduction rules — a different deterministic rule (e.g., suppressing the 1st rather than 2nd entry) was not tested and might produce different specific suppressed-trade composition, though the broad conclusion (majority-winner suppression) would likely be structurally similar given the historical win rate is well above 50%... actually the control's overall win rate is roughly 35-40%, so a >55% winner-suppression rate specifically among *suppressed* trades (not all trades) is itself informative and not simply an artifact of the alternating rule, though this was not independently verified with an alternative suppression pattern.
- The regime-robustness check (§16) rests on a single most-recent period (2026 YTD, 154 days) — a real but not overwhelming sample for the specific finding that Control B/C invert there.
- Cost sensitivity could not be assessed (§18) — a genuine gap, not resolved in this phase.
- All findings are IN-SAMPLE COUNTERFACTUAL EVIDENCE (§20) — the strongest possible caveat on every number in this report.

## 28. Final verdict

### Answers to the 20 required questions

1. **Does any frozen control materially reduce drawdown?** Control B does, in the aggregate period (30.4%) — but not robustly (see below).
2. **Does it reduce tail losses?** Control B's worst-5-day figure improves 30.0% in aggregate; Controls C/D/E show negligible-to-negative tail improvement.
3. **How much historical return is sacrificed?** Control B: 1.9%. Control C: 8.4%. Control D: actually gains 10.1% (fewer, better-selected trades removed). Control E: 0.3%.
4. **Does it suppress more losers than winners?** No — every control with a meaningful suppression count (B, C, D) removed a *majority* of historical winners (58.5-65.6%).
5. **Does it improve the return/drawdown trade-off?** Nominally yes for B (ratio 16.06) — but this reflects broad activity reduction, not selective risk avoidance (§13).
6-8. **Survives removal of worst 1/5/10 days?** Control B's advantage narrows but persists directionally; Control D remains the best absolute performer at the 5-day exclusion level (§17).
9. **Directionally consistent across regimes?** No for B and C (both invert in 2026 YTD); yes for D (though on a small, single-period sample).
10. **Survives 2x cost sensitivity?** Not computable (§18).
11. **Does Monte Carlo support the observed improvement?** Partially — baseline's own historical drawdown was itself somewhat unlucky (3.9th percentile of its own reshuffle), which mechanically makes any trade-removal control look relatively better without necessarily reflecting a real structural improvement.
12. **Is the apparent improvement driven by a small number of trades?** For Control E, yes trivially (only 2 trades suppressed, effectively inert). For B/C/D, no — each suppresses a substantial (6-15%) fraction of trades.
13. **Which intervention performs best?** By raw trade-off ratio, Control B; by regime consistency, Control D.
14. **Is that ranking robust?** No — neither ranking survives all of the preregistered robustness checks simultaneously.
15. **Is any intervention good enough to justify OOS validation?** No.
16. **Which interventions should be rejected?** All four, at their currently-tested specification.
17. **Does the evidence support a future portfolio-control research phase?** Only narrowly — via the two specific future-research ideas recorded (§26), not a broad continuation of the current approach.
18. **Or should portfolio-control research stop?** For the four archetypes tested here, yes, at their current specification; the underlying diagnostic questions (Phases 41-43) remain validly answered and are not undermined by this negative result.
19. **What remains unknown?** Whether a more selective (strategy-aware or magnitude-aware, not purely exposure/volatility-based) control could avoid the majority-winner-suppression problem found here; cost sensitivity; whether Control D's promising-but-thin 2026 YTD result would replicate with more data.
20. **What should Phase 45 investigate, if anything?** Not decided here — the future-research-ideas list (§26) is the most evidence-grounded starting point if portfolio-control research continues; alternatively the research program could return to return-stream research (per Phase 39's still-standing findings) given no portfolio-control intervention has yet proven itself.

---

## Safety check confirmation

Preregistration committed (`6098577`) before results, unchanged after · research validator passed · control portfolio unchanged and reconciled against Phase 41-43 (2,712 trades) · no live strategy modified · no strategy parameters modified · no risk settings modified · no live filter implemented · no live exposure control implemented · no deployment · no threshold optimization (all thresholds taken from Phase 42/43's prior findings, verified via preregistration commit timestamp preceding all results) · no position-limit optimization · no return-stream optimization · no strategy-specific removal (no AMR/ARB/JPY/pair-specific removal tested, per the explicit prohibition) · extreme-day robustness completed · regime robustness completed · cost sensitivity attempted and honestly disclosed as not computable · Monte Carlo methodology consistent with Phases 37-40 · false-positive/in-sample limitations documented throughout · raw production 5ers export not committed. One implementation bug (drawdown-reduction sign convention) caught and corrected before interpretation — verified via independent hand-calculation, not a methodology change.

---

*No live trading change authorized. NO PORTFOLIO CONTROL JUSTIFIED among the four tested. 3 future research ideas recorded, none implemented.*
