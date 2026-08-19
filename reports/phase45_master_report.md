# Phase 45 — Portfolio Viability & Evidence Sufficiency Audit (Master Report)

**RESEARCH / DECISION-FRAMEWORK ONLY. No strategy created, modified, paused, or removed. No portfolio control deployed. No live-system change. Strongest allowed recommendation: CONTINUE VALIDATION / INVESTIGATE / RESEARCH REQUIRED / SUFFICIENT EVIDENCE FOR A FUTURE DECISION — never "deploy," "implement," or a risk change.**

---

## 1. Executive summary

The current-6 portfolio has a real, positive historical edge (PF 1.211, +194.1R over 2,712 trades, 2023-08 to 2026-08) but this evidence base has three material weaknesses, none of them previously consolidated in one place: **(a) robustness evidence is WEAK** — none of the six live strategies has ever been subjected to the parameter/cost-stress framework used for every Phase 33+ research candidate; **(b) strategy independence is WEAK** — effective diversification is only ~3.1 of the 6 nominal strategies (average pairwise correlation 0.192, JPY/AMR concentration is structural per Phase 41/42, not a stress artifact); **(c) the live post-demotion sample is small** (19 closed trades, R -4.32) and, per a fresh block-bootstrap against the historical distribution, sits at the **9.4th percentile** — outside the "expected variation" band but not in the range that would constitute a genuine deterioration signal. No individual strategy shows a clear deterioration signal at an adequate sample size (only `AUDJPY_AMR` has ≥5 post-demotion trades, and it is classified CONSISTENT). Phase 39's FX-technical research ceiling stands, unweakened by Phases 40-44. Phase 44 found no validated portfolio control. **Final classification: E. INSUFFICIENT EVIDENCE — CONTINUE OBSERVATION**, paired with one concrete, immediately-actionable, zero-new-data recommendation: run the existing parameter/cost-stress robustness battery against the current-6 strategies for the first time.

## 2. Purpose

Consolidate Phases 30-44's evidence to determine whether the current-6 portfolio is sufficiently evidenced to justify continued unchanged operation, and what evidence would be required before any change.

## 3. Research history

Phases 30-40 produced 71 confirmatory-plus-screen hypotheses (11 confirmatory hypotheses across 11 distinct return-driver families, plus a 60-cell exploratory calendar/drift screen); 0 reached portfolio-qualified status. Phases 41-44 were forensic/counterfactual research on the existing control, not new trading hypotheses: Phase 41 found no single dominant stress factor; Phase 42 decomposed volatility to a non-monotonic, MODERATE-evidence relationship; Phase 43 refined that finding via exposure attribution (with an important counter-finding); Phase 44 tested and rejected 4 frozen portfolio-control counterfactuals.

## 4. Data integrity

`research_data_validator` passed on `data/phase26_all_trades.csv`, `experiments/experiments.csv`, and the local `reports/5ers_trade_export.csv` (the freshest available live production data — 72 rows, 36 CLOSED/36 OPEN, never committed per this project's standing convention). Trade count (2,712) reconciled exactly against every prior phase's use of the control file.

## 5. Master research ledger

`reports/phase45_research_master_ledger.csv` — 71 rows, extending Phase 39's 70-row inventory with Phase 40's candidate. No hypothesis added from memory.

## 6. Research-family coverage

`reports/phase45_research_family_audit.csv` — 11 confirmatory hypotheses, each in its own distinct return-driver family (per Phase 39's structural-duplication finding, 2 of those 11 are near-duplicate variants of 2 broader driver concepts — see Phase 39's own reconciliation), 0 portfolio-qualified.

## 7. Strategy independence

`reports/phase45_strategy_evidence.csv`, reusing Phase 41's already-validated correlation matrix. **Nominal N = 6, effective N ≈ 3.06** (average full-period pairwise correlation 0.192) — the six live strategies behave, in diversification terms, closer to 3 independent strategies than 6. This is not a new finding (Phase 31/41/42 established the JPY/AMR concentration underlying it) but had not previously been expressed as a single effective-N number.

## 8. Historical portfolio edge

`reports/phase45_historical_portfolio.csv`. PF 1.211, expectancy +0.0716R/trade, win rate 67.0%, max drawdown -29.07R over a 226-day drawdown episode with a 91-day recovery, negative daily-return skew (-0.624) — a real, positive, moderately fat-tailed historical edge, not exceptional in magnitude.

## 9. Strategy contribution

`reports/phase45_strategy_contribution.csv`. `GBPJPY_AMR` contributes the largest single share of portfolio R (29.9%, on only 14.9% of trades — the highest per-trade efficiency of the six), followed by `EURJPY_AMR` (20.3%) and `AUDJPY_AMR` (17.2%). `CADJPY_AMR`, despite being the second-largest strategy by trade count (22.1% of trades), contributes the smallest R share (9.1%) — the portfolio's historical edge is not evenly distributed across trade volume.

## 10. Strategy stability

`reports/phase45_strategy_stability.csv`. Sample-size evidence is STRONG for 4 of 6 strategies (≥400 trades) and MODERATE for the other 2 (`CADJPY_ARB` 192, `GBPUSD_MONDAY` 154). **Every strategy's parameter and cost robustness is explicitly marked "NOT SEPARATELY TESTED IN THIS LEDGER"** — a disclosed, real gap, not glossed over: these strategies predate this project's Phase 33+ preregistration/perturbation discipline.

## 11. Live validation

`reports/phase45_live_validation.csv`, using the freshest local production export. Post-demotion (≥2026-07-31), current-6-membership, closed trades: **19 trades, R -4.32, 36.8% win rate.** A 7th strategy present in the live export, `GBPJPY_ARB`, is **not part of the frozen current-6 control** and is reported separately (3 closed trades, R -4.23) — never pooled with the current-6 figures.

## 12. Live sample sufficiency

`reports/phase45_live_sample_sufficiency.csv`. **SIMULATED** contiguous block-bootstrap (10,000 draws, block size = 19, drawn from the historical trade-order R series): the live result sits at the **9.4th percentile** of the bootstrap distribution — classified **UNUSUAL** (outside the 10th-90th "expected variation" band) but not in the 2nd-5th percentile range that would flag it as statistically notable. This is a precise, quantified statement, not "19 trades is too small" — it is specifically "19 trades producing this particular result is somewhat below-median but not an outlier."

## 13. Live vs. historical comparison

`reports/phase45_live_vs_historical.csv`. Only `AUDJPY_AMR` has ≥5 post-demotion trades (9); it is classified **CONSISTENT** with its historical expectancy despite a negative live result (-0.462R avg vs. +0.0512R historical) — within one historical standard deviation. The other 5 strategies have 2-4 post-demotion trades each, explicitly **INSUFFICIENT SAMPLE**, not classified either way.

## 14. Deterioration framework

`reports/phase45_deterioration_framework.csv`. Concrete, evidence-derived thresholds for sustained negative expectancy, PF deterioration, regime failure, and unusual loss sequencing — each sourced from an already-published prior-phase figure, never invented for this phase. Parameter instability and execution deterioration are marked **NOT YET JUSTIFIABLE** — honest, disclosed gaps (no baseline exists to compare against).

## 15. Continued-viability framework

`reports/phase45_viability_framework.csv`. Seven dimensions, each with a specific evidence requirement — explicitly not "it becomes profitable again."

## 16. Strategy scorecard

`reports/phase45_strategy_evidence.csv`. `AUDJPY_AMR`: GREEN — CONTINUE VALIDATION (the only strategy with an adequate live sample, and it is consistent). The other 5: AMBER — WATCH, specifically because their live sample is insufficient to confirm OR refute, not because of any negative signal.

## 17. Portfolio scorecard

`reports/phase45_portfolio_scorecard.csv`. 10 dimensions scored: Historical edge MODERATE, Robustness WEAK, Regime diversity MODERATE, Strategy independence WEAK (effective N 3.1/6), Live validation INSUFFICIENT, Execution integrity UNKNOWN, Risk integrity NOT SEPARATELY AUDITED, Stress behaviour MODERATE, Research breadth MODERATE, overall Evidence sufficiency **WEAK-TO-MODERATE**.

## 18. Research ceiling

Phase 39's C. FX TECHNICAL RESEARCH CEILING REACHED FOR NOW is **CONFIRMED, not weakened**, by Phases 40-44: Phase 40 (volatility-conditioned trend continuation) was rejected on the largest OOS sample this project has tested; Phases 41-43 diagnosed the portfolio's stress structure without producing a new trading hypothesis; Phase 44 found no portfolio-level fix either. Five phases of subsequent work have not identified a path around the ceiling.

## 19. Phase 44 counterfactual findings

Incorporated directly: NO PORTFOLIO CONTROL JUSTIFIED. The strongest-looking control (~30% drawdown reduction) suppressed ~61% historical winners and inverted in the most recent regime — interpreted, per the task's own instruction, as evidence that **lower historical drawdown alone does not establish a genuinely useful portfolio control**, not as proof no control could ever work.

## 20. What we know

- The historical current-6 reconstruction has a real, positive edge (PF 1.211) across a 774-day, 3-regime-period sample.
- Effective portfolio diversification is materially lower than the nominal strategy count (≈3.1 of 6).
- 11 confirmatory hypotheses across 11 distinct FX-technical return-driver families have all failed to reach portfolio-qualified status (Phases 30-40).
- No tested portfolio control improves tail risk without a disqualifying side effect (Phase 44).

## 21. What we probably know

- The current-6 strategies' historical edge is not evenly distributed — `GBPJPY_AMR` and `EURJPY_AMR` account for over half of total historical R.
- HIGH volatility is the most consistent (though only MODERATE) stress-adjacent factor identified across Phases 41-43.
- The FX-technical research ceiling (Phase 39) is a durable, not provisional, finding.

## 22. What we suspect

- The live post-demotion underperformance (9.4th percentile) may reflect ordinary variance rather than deterioration, but the sample is too small to be confident either way.
- The lack of formal robustness testing on the current-6 strategies may be masking latent parameter/cost fragility not yet visible in the aggregate historical PF.

## 23. What we do not know

- Whether any current-6 strategy would survive the Phase 33+ parameter/cost-stress framework if formally applied — never tested.
- Whether the live sample's 9.4th-percentile result would persist, revert, or worsen with more trades.
- Execution quality (slippage, fill integrity) — no baseline exists.

## 24. What we cannot currently test

- Point-in-time macro/event conditioning (Phase 39's confirmed data gap, unchanged).
- Pre-2023 history for the current-6 strategies (not investigated — may or may not exist).
- Genuinely out-of-sample validation of any future portfolio control (Phase 44's findings are explicitly in-sample only).

## 25. Research activities to stop

Another undifferentiated FX-technical strategy search (Phase 39 ceiling, reconfirmed by Phase 40); rescuing or re-parameterizing any rejected Phase 33-44 candidate; optimizing a portfolio control against the same historical sample already used in Phases 41-44.

## 26. Research activities to continue

Continued live validation (zero incremental cost, already running); formal parameter/cost-stress robustness testing of the current-6 strategies (high-value, low-cost, closes a real disclosed gap, uses only existing infrastructure); volatility-conditioned research in a non-directional framing (per Phase 39, not restarted here); Event/macro and Index-based infrastructure consideration (unchanged priority from Phase 39).

## 27. Information gaps

`reports/phase45_information_gaps.csv` — 6 gaps, classified CAN FIX NOW (the robustness-testing gap), REQUIRES TIME (live sample size, independent OOS validation of any future control), REQUIRES NEW DATA SOURCE (macro/event data, execution-quality baseline, possible pre-2023 history).

## 28. Decision tree

`reports/phase45_decision_tree.md` — built from this phase's own evidence, not a template; traces to E. INSUFFICIENT EVIDENCE — CONTINUE OBSERVATION.

## 29. Minimum evidence requirements

`reports/phase45_future_requirements.csv` — for each of the 8 possible portfolio actions (A-H), the minimum evidence required, or an explicit "NO EVIDENCE-BASED THRESHOLD ESTABLISHED" where this project has never studied the relevant dimension (risk sizing, capital scaling).

## 30. Forward validation requirements

Portfolio-level: the live sample needs to grow well beyond 19 trades before its bootstrap percentile could be trusted as more than "somewhat below median" — this project's own convention (Phase 37/38's ≥30-40-trade statistical-informativeness bar) suggests a **rough range of 60-120 additional live trades** (2-4x the current sample) before a portfolio-level live signal could be treated as more than exploratory, though this range is itself an extrapolation from OOS-testing conventions, not a formally derived power calculation, and is disclosed as such. Strategy-level: given 5 of 6 strategies currently have 2-4 post-demotion trades, each would individually need a comparable multiple to reach even the loosest adequacy bar used elsewhere in this project (n≥30).

## 31. Research program decision

**A + B combined**: A. CONTINUE CURRENT VALIDATION (live observation, unchanged) and B. LIMITED TARGETED RESEARCH (the formal robustness-testing gap specifically, per §26) — not C (no new infrastructure investment is urgently indicated beyond what Phase 39 already prioritized), not D (no evidence supports reassessing portfolio design right now), though E remains the honest characterization of the live-evidence dimension specifically.

## 32. Limitations

- The block-bootstrap (§12) draws contiguous historical blocks, which preserves some temporal/regime structure but does not specifically match the live sample's actual calendar period — a disclosed modeling simplification, consistent with every prior phase's Monte Carlo convention.
- Risk/execution integrity (portfolio scorecard dimensions) were not independently audited in this phase — would require live configuration inspection, out of this phase's historical-data-analysis scope.
- The forward-validation trade-count range (§30) is an extrapolation from this project's existing OOS-sample conventions, not a formally derived statistical power calculation.

## 33. Final verdict

### Answers to the 24 required questions

1. **Genuinely robust historically?** Real edge, but formal robustness evidence is WEAK (never tested) — "robust" is not yet established either way.
2. **How much edge survives across regimes?** Positive in all 3 available regime periods per Phase 42/43's regime-robustness checks on the underlying control.
3. **Dependent on one or two strategies?** Partially — `GBPJPY_AMR` + `EURJPY_AMR` together contribute >50% of total R.
4. **How independent are the six strategies really?** Effective N ≈3.1 of 6 — materially less diversified than the nominal count suggests.
5. **What does the current live sample tell us?** 19 trades, 9.4th percentile of the historical bootstrap — below median, not an outlier.
6. **Is the live drawdown statistically unusual?** UNUSUAL by the bootstrap classification, not STATISTICALLY NOTABLE.
7. **Evidence of genuine strategy deterioration?** No — the only strategy with adequate live sample (`AUDJPY_AMR`) is CONSISTENT.
8. **Evidence of portfolio-level deterioration?** No clear signal — UNUSUAL but within a range plausibly explained by ordinary variance on a small sample.
9. **Strongest historical evidence?** `AUDJPY_AMR`, `EURJPY_AMR`, `GBPJPY_AMR` (STRONG sample-size tier, ≥400 trades each).
10. **Weakest current evidence?** `CADJPY_ARB` and `GBPUSD_MONDAY` (MODERATE historical sample) combined with only 2 post-demotion live trades each.
11. **Evidence supporting continuing unchanged?** Positive historical PF, no deterioration signal, no validated alternative (Phase 44).
12. **Evidence arguing for investigation?** The disclosed robustness gap (§10) and the live sample's below-median (though not extreme) result.
13. **Evidence against changing anything?** No strategy or the portfolio as a whole clears any of the minimum-evidence bars in §29 for a change.
14. **Did Phase 44 produce a credible portfolio control?** No.
15. **Is FX-technical research genuinely at a ceiling for now?** Yes, confirmed by Phases 40-44.
16. **Which research families remain rational to investigate?** Volatility-conditioned (non-directional), Index-based, per Phase 39's unchanged priority.
17. **What data infrastructure is missing?** Point-in-time macro/event data (confirmed blocked); execution-quality baseline (never built).
18. **Single most valuable new piece of information?** More live post-demotion trades — nothing else in this audit is as directly decision-relevant.
19. **How much additional live evidence is required?** Roughly 60-120 additional trades portfolio-level (§30), an extrapolation, not a precise figure.
20. **What should trigger a formal strategy review?** Per `phase45_future_requirements.csv`: sustained negative expectancy or PF<0.8 over ≥100 trades for a specific strategy, or a Monte Carlo <5th-percentile loss sequence.
21. **What should trigger a portfolio review?** Portfolio-level live result falling below the 5th percentile of the bootstrap distribution over an adequately-sized sample, or simultaneous strategy-level triggers.
22. **What should NOT trigger a review?** The current 19-trade, 9.4th-percentile result alone — it does not clear the bar in §20-21.
23. **What should Phase 46 investigate?** The formal parameter/cost-stress robustness battery applied to the current-6 strategies for the first time (§26) — the single highest-value, lowest-cost, immediately actionable gap identified in this audit.
24. **Should Phase 46 investigate anything at all?** Yes, but narrowly (§31's "B. LIMITED TARGETED RESEARCH") — not a new strategy search, not a portfolio-control redesign.

### Final classification

## **E. INSUFFICIENT EVIDENCE — CONTINUE OBSERVATION**

Neither A (historically robust, continue unchanged) nor C/D (fragile, reassess) is supported by the assembled evidence. The historical edge is real but under-tested for robustness; the live sample is too small and too recent to confirm or refute anything at the strategy level (only one of six strategies has an evaluable sample); no validated portfolio-level intervention exists. This is not a hedge — it is the specific, disclosed conclusion this phase's own decision tree (§28) traces to.

---

## Safety check confirmation

Preregistration committed (`b41da5d`) before results, unchanged after · research validator passed on all 4 source artifacts · historical ledger reconciled (2,712 trades matching Phase 41-44) · six-strategy control unchanged · live sample separated (A full/B pre-demotion/C post-demotion, plus the non-control 7th strategy GBPJPY_ARB kept fully separate) · no live strategy touched · no live risk touched · no strategy parameters touched · no portfolio controls deployed · no optimization performed · Phase 44 findings incorporated (§19) · Phase 39 FX ceiling reviewed and confirmed (§18) · research-family coverage reconciled (§6) · information gaps identified (§27) · minimum evidence requirements documented (§29) · forward-validation requirements documented (§30, with explicit uncertainty disclosed) · raw production 5ers export not committed.

---

*No live trading change authorized. No strategy modified, paused, or removed. Recommended next step: run the existing parameter/cost-stress robustness framework against the current-6 strategies — a low-cost, evidence-closing action, not a new strategy search.*
