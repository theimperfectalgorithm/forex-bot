# Phase 40 — Volatility-Conditioned Return-Stream Research (Master Report)

**RESEARCH ONLY. No live strategy, parameter, risk, or portfolio weight modified. AMR/ARB/GBPUSD Monday/AUDUSD Monday LONG untouched. No candidate deployed.**

---

## 1. Executive summary

**HIGH-volatility-state trend continuation (EURUSD/GBPUSD/AUDUSD/USDCAD, New York session) is decisively rejected: OOS PF 0.668 on 2,228 trades — by a wide margin the largest and most statistically decisive sample tested in this project's entire research history.** The candidate fails not only Gate 1 (no credible edge) but also the specific gate it was designed to test: it **materially deteriorates in HIGH volatility** (Classification C), the opposite of its intended purpose, since this candidate trades *exclusively* in the HIGH-volatility state by construction. It is also CORRELATED with the control portfolio's drawdown days (0.251 vs. 0.090 normal) and, when integrated into the portfolio at any tested weight, **catastrophically worsens** both total return and maximum drawdown (control total_R 126.72 → combined -266.99 at 1.0x weight; control max_dd -14.53 → combined -279.10). **Final classification: B. REJECTED — NO CREDIBLE OOS EDGE.**

This is a clean, informative null result on the largest sample this project has ever tested, obtained without any parameter search, rescue attempt, or variant substitution — exactly the "successful research result" the phase's own instructions anticipated as an acceptable outcome.

## 2. Phase 39 context

Phase 39 concluded C. FX TECHNICAL RESEARCH CEILING REACHED FOR NOW (for undifferentiated mechanism search) and recommended self-calculated volatility-conditioning as the highest-priority, immediately-researchable next direction (priority score 75.0, the only class READY FOR PREREGISTRATION without new infrastructure).

## 3. Preregistration

`reports/phase40_preregistration.md`, committed separately (`bea0a31`) before any backtest ran. No amendment required — the implementation matched the frozen definition exactly (see §6).

## 4. Structural independence

`reports/phase40_structural_independence.md`. **B. RELATED BUT MEANINGFULLY DIFFERENT.** The closest prior work (Phase 35 H2's NY-session momentum, Phase 35 H5's ATR-scaled exit) each share one element but neither uses realized volatility as a genuine trade-**activation gate** the way Phase 40 does. Not classified C (duplicative); backtesting proceeded.

## 5. Data integrity

`reports/phase40_data_integrity.md`. `research_data_validator` passed. All four MT5 H1 pulls (EURUSD/GBPUSD/AUDUSD/USDCAD, 47,345-47,347 bars each) passed integrity asserts. Volatility calculation verified point-in-time-correct by construction (one-bar-lagged state, TRAIN-fixed thresholds never re-estimated on VALIDATION/OOS).

## 6. Candidate reproduction

`reports/phase40_reproduction.csv`. TRAIN: 4,570 trades, PF 0.890, expectancy -0.0550R. VALIDATION: 2,014 trades, PF 0.790, expectancy -0.1112R. **OOS: 2,228 trades, PF 0.668, expectancy -0.1767R, total -393.71R.** Implementation matched the frozen preregistration exactly (single-bar-lag state, TRAIN-fixed terciles per instrument, no re-fitting) — no STOP condition triggered.

## 7. OOS edge

`reports/phase40_oos_results.csv`. **Gate 1: FAIL.** PF 0.668, well below 1.0, on the largest OOS sample (2,228 trades) tested at any point in this project.

## 8. OOS consistency

`reports/phase40_oos_consistency.csv`. First half (1,113 trades): PF 0.726. Second half (1,115 trades): PF 0.617. **Sign-consistent, verdict PASS** — but consistently negative, not consistently positive.

## 9. Parameter robustness

`reports/phase40_parameter_robustness.csv`. ATR window 11/14/17: PF 0.718/0.668/0.691. Negative at every setting, no sign reversal (nothing to reverse from — already negative throughout).

## 10. Cost stress

`reports/phase40_cost_stress.csv`. PF 0.668 (normal) → 0.600 (1.5x) → 0.539 (2x). Already well below 1.0 before cost stress is even relevant.

## 11. Volatility regime analysis

`reports/phase40_volatility_regimes.csv`. By design this candidate trades *only* the HIGH-volatility state; there is no LOW/NORMAL trading result to report, since the hypothesis structurally excludes those states from participation (per Part 3/4's activation-gate design).

## 12. Volatility transition analysis (diagnostic only, per Part 17)

`reports/phase40_volatility_transitions.csv`. **PERSISTENT_HIGH** (state was already HIGH the bar before): 1,877 trades, expectancy -0.182R. **TRANSITION_NORMAL_TO_HIGH**: 349 trades, expectancy -0.144R. **TRANSITION_LOW_TO_HIGH**: 2 trades (too thin to interpret). No sub-slice of the transition breakdown rescues the candidate — every meaningfully-sized bucket is negative. Per the frozen no-rescue rule (Part 35), this diagnostic is not used to construct a replacement strategy; it is recorded purely as evidence that the failure is broad-based, not concentrated in one transition type.

## 13. Historical regime analysis

`reports/phase40_historical_regimes.csv`. Negative in all five characterized periods: 2019-2020 (PF 0.837), 2021-2022 (PF 0.838), 2023-2024 (PF 0.858), 2025 (PF 0.802), 2026 YTD (PF 0.607, the worst period). Unlike AUDUSD Monday LONG (which strengthened toward the present), this candidate's performance **weakens** toward the present — a genuinely broad-based and, if anything, worsening failure, not a recent-data artifact.

## 14. HIGH-volatility gate

`reports/phase40_high_volatility.csv`. **C. MATERIALLY DETERIORATES IN HIGH VOLATILITY.** This is the single most important negative finding of the phase: the current portfolio's most important known weakness (per Phase 39's Gap 1) is HIGH-volatility behavior, and this candidate — specifically designed to trade only in that state — performs *worse* there than an unconditional strategy typically would, not better. Volatility conditioning as a portfolio-repair mechanism, at least in this momentum-continuation form, does not work.

## 15. Drawdown correlation

`reports/phase40_drawdown_correlation.csv`. Normal-day correlation 0.090, drawdown-day correlation 0.251 (26 overlapping days — well-sampled, well above the 8-day floor). **Classification: CORRELATED.** Would have failed this hard gate (Part 20) even had Gate 1 passed.

## 16. Mechanism diversification

`reports/phase40_mechanism_diversification.csv`. STRONGLY DISTINCT from AMR (mean-reversion vs. momentum, Asian vs. NY, JPY vs. non-JPY). MEANINGFULLY DISTINCT from GBPUSD Monday and from the six-strategy control in aggregate (no live strategy uses volatility-state gating). Structural distinctness is confirmed — but, per the phase's own repeated caution, structural difference alone did not prevent failure on the decisive gates (§14, §15).

## 17. JPY exposure

`reports/phase40_jpy_exposure.csv`. Candidate carries 0% JPY exposure (EURUSD/GBPUSD/AUDUSD/USDCAD), vs. the control's heavy JPY concentration (4 of 6 live strategies are JPY-linked). Per Part 22's own caution, this alone does not constitute diversification — and indeed did not: the drawdown-correlation and portfolio-integration results (§15, §19) show the non-JPY exposure did not translate into useful diversification.

## 18. Session diversification

`reports/phase40_session_diversification.csv`. The first hypothesis in this project's ledger to trade exclusively within the New York session with no Asian/London signal dependency — a genuine structural first. As with JPY exposure, this structural novelty did not translate into a portfolio benefit.

## 19. Portfolio integration

`reports/phase40_portfolio_integration.csv`. At 0.5x weight: control total_R 126.72 → combined -70.13 (a swing of -196.85R); control max_dd -14.53 → combined -110.85. At 1.0x weight: combined total_R -266.99; combined max_dd -279.10. **This is the most severe portfolio-integration failure of any candidate tested across Phases 37-40** — the candidate's own losses are large enough, on a large enough trade count, to overwhelm the control portfolio's own positive return entirely.

## 20. AUDUSD comparison

AUDUSD Monday LONG (Phase 37, unmodified): strong standalone edge (PF 3.070) that failed specifically and only on drawdown diversification. **The Phase 40 candidate does not solve AUDUSD's specific failure — it fails at an earlier, more fundamental gate** (no edge at all, PF 0.668 vs. AUDUSD's 3.070) and is *also* CORRELATED on drawdown days, just like AUDUSD. On the one dimension where a comparison is meaningful (drawdown-day correlation), Phase 40's 0.251 is numerically better than AUDUSD's 0.742 — but this is irrelevant given the candidate has no standalone edge to deploy in the first place.

## 21. Monte Carlo

`reports/phase40_monte_carlo.csv`. **SIMULATED.** Actual max DD -395.05R sits at the 78.6th percentile of the 10,000-draw reshuffled distribution (median -397.93R) — the actual trade sequencing is not an outlier relative to a random reshuffle of the same (already strongly negative) trades. The problem is the underlying edge, not adverse sequencing.

## 22. Sample size

`reports/phase40_sample_size.csv`. OOS trades: 2,228 — STATISTICALLY INFORMATIVE by a wide margin, the best-sampled candidate this project has tested. OOS sub-halves: 1,113/1,115 — ADEQUATE. Drawdown-correlation overlap: 26 days — STATISTICALLY INFORMATIVE (well above the 8-day floor). Historical regime periods: 5 of 5 have ≥10 trades.

## 23. Multiple testing

`reports/phase40_multiple_testing.csv`. Exactly 1 confirmatory hypothesis, with its 1 preregistered ±20% ATR-window perturbation (3 values) and 1 preregistered diagnostic (volatility-transition breakdown, explicitly not used to construct a replacement strategy). Two FUTURE RESEARCH IDEAS recorded (volatility-gated mean-reversion in the opposite direction; volatility-gated defensive exposure-scaling) but explicitly NOT tested in this phase, per the frozen no-rescue rule.

## 24. Final classification

`reports/phase40_candidate_classification.csv`. **B. REJECTED — NO CREDIBLE OOS EDGE.** Per the mechanical gate order (structural independence → edge → ...), the candidate is rejected at Gate 1, the earliest substantive gate — though it would independently have also failed the HIGH-volatility gate (§14) and the hard drawdown-correlation gate (§15) had it somehow cleared Gate 1.

## 25. Limitations

- The HIGH-volatility gate result (§14) is specific to this candidate's momentum-continuation direction; a structurally opposite hypothesis (mean-reversion in HIGH-volatility states) remains untested and is recorded as a FUTURE RESEARCH IDEA, not evidence about volatility-conditioning as a class.
- The volatility-transition diagnostic (§12) has a very thin TRANSITION_LOW_TO_HIGH bucket (n=2) — not interpretable, correctly left unused.
- This candidate's non-JPY, NY-session structural novelty (§16-18) was confirmed but did not translate into portfolio value — a useful negative data point, but a single candidate is not conclusive evidence that volatility-gated momentum-continuation can never work in any instrument/session/direction combination.

## 26. Phase 41 recommendation

**Volatility conditioning as a directional-momentum-continuation mechanism, in this specific form, is now rejected — do not re-test a variant of it in Phase 41** (per the no-rescue rule). The two FUTURE RESEARCH IDEAS flagged in §23/§12 (opposite-direction mean-reversion; defensive exposure-scaling rather than a new directional stream) are the most evidence-grounded starting points if volatility-conditioning is to be revisited, but neither is pre-selected here — Phase 41 needs its own independent decision process. Given that this phase, like Phase 38, again found a structurally novel, well-sampled candidate correlated with the control's drawdowns, **the drawdown-correlation problem itself (not the specific mechanism tried) may deserve direct research attention** — e.g., characterizing exactly what the control portfolio's worst days have in common (already partially done in Phase 31/32) rather than continuing to generate new candidate return streams and testing each against the same recurring failure mode.

## 27. Final verdict

### Answers to the 23 required questions

1. **Structurally distinct from existing FX-technical portfolio?** Yes — B. RELATED BUT MEANINGFULLY DIFFERENT (first volatility-activation-gated hypothesis in the ledger).
2. **Credible OOS edge?** No — PF 0.668 on 2,228 trades, the most decisive rejection sample in this project's history.
3. **Profitable in both OOS halves?** No — negative in both (PF 0.726 and 0.617).
4. **Survives ±20% parameter perturbation?** No sign reversal, but only because it was negative at every setting (PF 0.718/0.668/0.691).
5. **Survives 2x cost stress?** No — PF 0.539 at 2x, already failing before stress.
6. **Works across multiple historical regimes?** No — negative in all 5 characterized periods, and worsening toward the present (opposite of AUDUSD's pattern).
7. **Behaves acceptably in HIGH volatility?** No — C. MATERIALLY DETERIORATES, the specific gate this candidate was built to test.
8. **Behaves differently during portfolio drawdowns?** No — CORRELATED (0.251 vs. 0.090 normal).
9. **Normal correlation?** 0.090.
10. **Drawdown correlation?** 0.251.
11. **Worst-day correlation?** Same as drawdown correlation in this methodology (control's worst-decile days) — 0.251.
12. **Materially reduces correlated losses?** No — it materially *increases* portfolio losses at every weight tested (§19).
13. **Diversifies the mechanism mix?** Structurally yes (first volatility-gate mechanism tested) — but not usefully, given the failures above.
14. **Diversifies session exposure?** Structurally yes (first pure-NY-session hypothesis) — same caveat.
15. **Reduces or worsens JPY concentration?** Structurally reduces it (0% JPY exposure) — but this did not translate into portfolio value.
16. **Does CONTROL + CANDIDATE improve portfolio-level behaviour?** No — catastrophically worse at every tested weight (§19).
17. **Is the sample sufficiently large?** Yes, by a wide margin — 2,228 OOS trades, the best-sampled candidate this project has tested to date.
18. **What killed the candidate?** No standalone edge (Gate 1), on the largest, most decisive sample tested — reinforced independently by HIGH-volatility deterioration and drawdown correlation.
19. **Outperform AUDUSD Monday LONG's diversification profile?** Numerically better drawdown-correlation (0.251 vs. 0.742) but irrelevant without a standalone edge.
20. **Qualify for DEMO FORWARD TEST?** No.
21. **Most important lesson if it fails?** Volatility-state gating alone does not guarantee good HIGH-volatility behavior — the direction/mechanism gated by volatility matters as much as the gating itself; a candidate can be structurally novel (non-JPY, NY-only, first-ever volatility-activation design) and still fail on every dimension it was designed to fix.
22. **If it qualifies, what evidence remains insufficient?** N/A — did not qualify.
23. **What should Phase 41 investigate?** Not a variant of this candidate (excluded per the no-rescue rule). Consider either the two FUTURE RESEARCH IDEAS flagged in §23, or a direct investigation of what specifically characterizes the control portfolio's own worst days (extending Phase 31/32), given that 4 of 4 candidates now tested at the portfolio-integration stage (AUDUSD, Phase38 H1, Phase38 H2, Phase40) have failed on drawdown correlation regardless of mechanism.

---

## Safety check confirmation

No live strategy modified · no live parameter modified · no risk modified · no strategy paused · no 5ers configuration modified · no candidate deployed · AUDUSD Monday LONG untouched · AMR untouched · ARB untouched · Phase 40 preregistration committed (`bea0a31`) before results, unchanged after · structural independence verified (B, not C) · data validator passed · no future leakage (one-bar-lagged state, TRAIN-fixed thresholds) · OOS boundaries respected · no parameter optimization · no strategy variants tested (exactly 1 hypothesis, its 1 preregistered perturbation, and 1 preregistered diagnostic) · cost stress completed · OOS consistency completed · HIGH-volatility analysis completed · drawdown correlation completed · portfolio integration completed · multiple testing documented (`reports/phase40_multiple_testing.csv`) · sample-size limitations documented (`reports/phase40_sample_size.csv`) · raw production 5ers export not committed.

---

*No live trading change authorized. Candidate does not reach classification I or J. No replacement volatility strategy created in this phase, per the frozen no-rescue rule.*
