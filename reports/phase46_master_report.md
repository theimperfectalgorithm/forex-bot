# Phase 46 — Current Six-Strategy Robustness Audit (Master Report)

**HISTORICAL ROBUSTNESS / LIVE-VALIDATION RESEARCH ONLY. No strategy code, parameter, indicator, entry, exit, SL/TP, position sizing, or risk setting modified. No pause, removal, filter, or optimization. No repair.**

---

## 1. Executive summary

Applying the same gates used for Phase 33-40 research candidates to the six strategies actually carrying the live portfolio produces a genuinely mixed, honest result. **All six strategies clear Gate 1 (OOS edge, PF 1.14-2.47) and OOS sub-half consistency** — a dramatically better pass rate than the Phase 33-40 candidate pool (27.3% Gate-1 pass rate among 11 confirmatory hypotheses). But **3 of 6 (`AUDJPY_AMR`, `CADJPY_AMR`, `CADJPY_ARB`) fail the HIGH-volatility gate** (PF 0.827-0.875, negative expectancy), and **3 of 6 (`AUDJPY_AMR`, `EURJPY_AMR`, `GBPJPY_AMR`) fail the drawdown-correlation gate** — the exact hard gate that rejected AUDUSD Monday LONG and every Phase 38/40 candidate. Only `CADJPY_ARB` passes drawdown correlation, on the thinnest possible sample (n=8, exactly the preregistered floor); `GBPUSD_MONDAY`'s drawdown-correlation is UNKNOWN (n=7, just below the floor). Formal parameter/cost-stress robustness testing — the gap Phase 45 identified — **could not be performed in this pass**, disclosed before any result was seen, because the historical ledger stores only already-executed trade outcomes, not a re-runnable backtest engine. **No strategy can therefore be classified higher than B (PLAUSIBLE — EVIDENCE STILL INSUFFICIENT)**, per the preregistered ceiling rule. This is an honest, sometimes uncomfortable result, delivered without protecting the live strategies or rescuing them.

## 2. Phase 45 context

Phase 45 identified that the current-6 strategies had never been subjected to the Phase 33+ robustness framework and recommended this audit as the single highest-value, lowest-cost next step.

## 3. Research question

If the six live strategies were presented to the research program today as new candidates, which would survive?

## 4. Preregistration

`reports/phase46_preregistration.md`, committed separately (`6e0875f`) before any result. Disclosed, before any result was seen, that parameter perturbation and cost-stress re-simulation are classified INSUFFICIENT DATA / REQUIRES NEW RE-EXECUTION INFRASTRUCTURE — not silently omitted.

## 5. Data integrity

All source artifacts validated clean; trade count (2,712) reconciled against every prior phase.

## 6. Six-strategy inventory

`reports/phase46_strategy_inventory.csv` — all 6 verified directly against `data/phase26_all_trades.csv`'s own strategy names; the live-feed's `GBPUSD_MON` confirmed to be the same strategy as `GBPUSD_MONDAY` via cross-referencing symbol/strategy_reason fields, not assumed.

## 7. Strategy definitions

`reports/phase46_strategy_definitions.csv`, built by reading the actual committed source (`strategies/asian_hours_reversion.py`, `asian_range_breakout.py`, `monday_drift.py`) and their YAML configs directly. **Two genuine, previously-undocumented discrepancies were discovered**: (a) `CADJPY_ARB`'s live YAML sets `h4_filter: false`, disabling the H4 trend filter its own docstring describes as core logic; (b) `GBPJPY_AMR`'s YAML discloses that a documented breakeven-exit refinement is backtest-only — live breakeven handling still uses a separate, older mechanism. Neither is corrected in this phase.

## 8. Historical methodology

TRAIN 2023-08-2024-08, VALIDATION 2024-09-2025-04, OOS 2025-05-2026-08 — the exact split already used since Phase 35.

## 9. OOS results

`reports/phase46_oos_results.csv`. All six PASS Gate 1: `AUDJPY_AMR` PF 1.200, `CADJPY_AMR` PF 1.140, `EURJPY_AMR` PF 1.313, `GBPJPY_AMR` PF 1.767, `CADJPY_ARB` PF 1.584, `GBPUSD_MONDAY` PF 2.472. All on adequate samples (67-301 OOS trades).

## 10. OOS sub-half consistency

`reports/phase46_oos_subhalf.csv`. All six PASS — sign-consistent in both halves, several strengthening in the second half (`GBPJPY_AMR`, `GBPUSD_MONDAY`).

## 11. Parameter perturbation

`reports/phase46_parameter_perturbation.csv`. **INSUFFICIENT DATA / REQUIRES NEW RE-EXECUTION INFRASTRUCTURE for all six**, disclosed before results per the preregistration. Not fabricated.

## 12. Parameter stability

`reports/phase46_parameter_stability.csv`. Same disclosed limitation. Informal prior evidence exists in each strategy's own docstring/YAML (a 36-variant grid for the AMR pairs reported "parameter-insensitive," an informal SL/TP grid for `GBPUSD_MONDAY` reported "robust, OOS PF 2.9-3.1") — explicitly noted as methodologically different from, and not equivalent to, the frozen ±20% single-perturbation standard.

## 13. Cost stress

`reports/phase46_cost_stress.csv`. **NOT COMPUTABLE from this dataset** for all six — identical disclosed limitation to Phase 44.

## 14. Regime robustness

`reports/phase46_regime_robustness.csv`. All six positive in every available regime period (2023-2024, 2025, 2026 YTD); 2019-2022 UNKNOWN BY DATA ABSENCE.

## 15. Volatility behavior

`reports/phase46_volatility_behavior.csv`. **`AUDJPY_AMR` (PF 0.827), `CADJPY_AMR` (PF 0.833), and `CADJPY_ARB` (PF 0.875) all go net-negative in the HIGH-volatility state** — directly consistent with this project's own Phase 41/42/43 findings that AMR/ARB mechanism concentration is where the portfolio's HIGH-volatility weakness lives. `EURJPY_AMR`, `GBPJPY_AMR`, and `GBPUSD_MONDAY` remain positive in HIGH volatility.

## 16. Drawdown correlation

`reports/phase46_drawdown_correlation.csv`, using the exact OOS-window-matched methodology from Phases 33-40 (each strategy's own contribution excluded from the control before computing correlation, to avoid trivial self-correlation). **`AUDJPY_AMR` (0.635), `EURJPY_AMR` (0.554), and `GBPJPY_AMR` (0.608) are all classified CORRELATED** — the exact hard gate that rejected AUDUSD Monday LONG (Phase 37), Phase 38's H1/H2, and Phase 40's candidate. `CADJPY_AMR` is NEUTRAL, close to the CORRELATED boundary. `CADJPY_ARB` is the only STRONG DIVERSIFIER, but on a thin 8-day drawdown-overlap sample (exactly the preregistered floor). `GBPUSD_MONDAY`'s result is UNKNOWN (7-day overlap, just below the floor). **This is the single most important, most uncomfortable finding of this phase**: half of the live portfolio's strategies would fail, on this specific gate, the exact standard that has rejected every new candidate since Phase 37.

## 17. Portfolio integration (leave-one-out)

`reports/phase46_portfolio_integration.csv`. Removing `GBPJPY_AMR` causes the largest total-R drop (194.11 → 136.10, -30%) and worsens max drawdown (-29.07 → -31.19) — it is both the largest R contributor and, by this leave-one-out measure, a net drawdown-reducer despite failing the direct drawdown-correlation test in §16 (these are different measures — leave-one-out shows portfolio-level effect of removal, not the strategy's own correlation with the control's stress days). Removing `GBPUSD_MONDAY` similarly worsens drawdown (-30.62) despite its modest R contribution.

## 18. Strategy contribution

`reports/phase46_strategy_contribution.csv`. `GBPJPY_AMR` (29.9%) and `EURJPY_AMR` (20.3%) together contribute >50% of total historical R, confirming Phase 45's finding directly from this phase's own independent computation.

## 19. Monte Carlo

`reports/phase46_monte_carlo.csv`. **SIMULATED.** All six strategies' actual OOS drawdowns sit within a reasonable range of their own reshuffled distributions (22nd-88th percentile) — none shows evidence of unusually adverse historical sequencing.

## 20. Live comparison

`reports/phase46_live_comparison.csv`, using the freshest local production export. Live post-demotion samples are tiny for every strategy (2-5 trades each); results are mixed (some positive, some negative) with no strategy showing a dramatic reversal from its historical expectancy given the sample sizes involved.

## 21. Live sample sufficiency

`reports/phase46_live_sample_sufficiency.csv`, reusing Phase 45's exact block-bootstrap methodology per strategy. `AUDJPY_AMR` and `CADJPY_AMR` are classified UNUSUAL BUT NOT DECISIVE (6.1th and 11.2th percentile respectively) — worth watching, not alarming. The other four are CONSISTENT or borderline. **No strategy shows a live result in the extreme-tail "possible deterioration" range.**

## 22. Candidate comparison

`reports/phase46_candidate_comparison.csv`. On the computable Gate-1 measure alone, all six live strategies would pass where only 27.3% of Phase 33-40 candidates did. **This comparison is incomplete without §16's drawdown-correlation finding**: 3 of 6 live strategies would additionally face the same rejection gate (poor drawdown diversification) that has been the single most consistent failure point for every portfolio-integration-stage candidate since Phase 37 (AUDUSD Monday LONG, Phase 38 H1/H2, Phase 40).

## 23. Survivorship audit

`reports/phase46_survivorship_audit.csv`. These six strategies were not selected via the current competitive gate — each was individually validated through an earlier, informal process (documented in their own source/YAML comments) before this project's more rigorous framework existed. This is a real, disclosed consideration: the strategies' presence in the "current-6" reflects when they were built and validated, not a demonstrated ability to survive the modern gate — which is precisely what this phase now tests directly, as honestly as possible given the disclosed scope limitation.

## 24. Evidence matrix

`reports/phase46_evidence_matrix.csv` — full per-strategy synthesis across every gate.

## 25. Strategy classifications

Per the preregistered ceiling rule (no strategy can be classified A without the missing parameter/cost robustness evidence):

- **`AUDJPY_AMR`: B. PLAUSIBLE — EVIDENCE STILL INSUFFICIENT** (fails HIGH-vol and drawdown-correlation)
- **`CADJPY_AMR`: B. PLAUSIBLE — EVIDENCE STILL INSUFFICIENT** (weakest OOS PF; fails HIGH-vol; NEUTRAL/borderline drawdown-correlation)
- **`EURJPY_AMR`: B. PLAUSIBLE — EVIDENCE STILL INSUFFICIENT** (passes HIGH-vol; fails drawdown-correlation)
- **`GBPJPY_AMR`: B. PLAUSIBLE — EVIDENCE STILL INSUFFICIENT** (strongest OOS PF and largest contributor; passes HIGH-vol; fails drawdown-correlation; disclosed live/backtest exit mismatch)
- **`CADJPY_ARB`: B. PLAUSIBLE — EVIDENCE STILL INSUFFICIENT** (only strategy passing drawdown-correlation, on the thinnest sample; fails HIGH-vol; disclosed h4_filter discrepancy)
- **`GBPUSD_MONDAY`: B. PLAUSIBLE — EVIDENCE STILL INSUFFICIENT** (cleanest overall computable profile — passes edge, consistency, regime, and HIGH-vol; drawdown-correlation UNKNOWN, not failed)

No strategy is classified C or D on the computable evidence alone — none shows a research-level Gate-1 or consistency failure. No strategy is classified A, per the preregistered ceiling.

## 26. Portfolio classification

**B. PLAUSIBLE PORTFOLIO — EVIDENCE STILL INSUFFICIENT.** Consistent with Phase 45's E. INSUFFICIENT EVIDENCE finding at the portfolio level, now sharpened at the strategy level: the portfolio's aggregate edge is corroborated by every individual strategy independently clearing Gate 1, but the same structural weakness Phase 41-44 diagnosed (HIGH-volatility sensitivity, drawdown correlation) is now shown to be concentrated in specific, named strategies (`AUDJPY_AMR`/`CADJPY_AMR`/`CADJPY_ARB` for volatility; `AUDJPY_AMR`/`EURJPY_AMR`/`GBPJPY_AMR` for drawdown correlation) rather than being a diffuse, unexplained portfolio-level phenomenon.

## 27. What passed

Gate 1 (OOS edge): all 6. OOS sub-half consistency: all 6. Regime robustness: all 6. Monte Carlo (no adverse sequencing): all 6.

## 28. What failed

HIGH-volatility gate: `AUDJPY_AMR`, `CADJPY_AMR`, `CADJPY_ARB` (3 of 6). Drawdown-correlation gate: `AUDJPY_AMR`, `EURJPY_AMR`, `GBPJPY_AMR` (3 of 6, with `CADJPY_AMR` bordering).

## 29. What remains uncertain

Parameter and cost robustness for all six (the central disclosed gap of this phase). `CADJPY_ARB`'s drawdown-correlation pass (thin 8-day sample). `GBPUSD_MONDAY`'s drawdown-correlation status (below the sample floor). Whether the live sample's UNUSUAL-BUT-NOT-DECISIVE readings for `AUDJPY_AMR`/`CADJPY_AMR` would firm up in either direction with more trades.

## 30. Future validation

`reports/phase46_future_validation.csv` — 7 items, all FUTURE STRATEGY REVIEW CANDIDATES or FUTURE VALIDATION REQUIREMENTS, none implemented: the parameter/cost re-execution infrastructure gap (highest priority), the two disclosed live-config discrepancies (`CADJPY_ARB`'s h4_filter, `GBPJPY_AMR`'s breakeven mismatch), the HIGH-vol and drawdown-correlation findings themselves as review candidates, and continued live-sample accumulation.

## 31. Limitations

- The single largest limitation of this phase is disclosed in the preregistration itself: parameter and cost-stress robustness could not be directly tested, capping every strategy's classification at B regardless of how well it performs on every other gate.
- The drawdown-correlation methodology excludes each strategy's own contribution from the control before computing correlation — a fairer test than a trivial self-inclusion, but a specific methodological choice worth restating.
- `CADJPY_ARB`'s STRONG DIVERSIFIER classification and `GBPUSD_MONDAY`'s UNKNOWN classification both rest on samples at or below this project's own preregistered 8-day floor — genuinely thin evidence, not a confident finding either way.
- Live samples remain small (2-5 trades per strategy) — no strategy-level live conclusion is more than exploratory.

## 32. Final verdict

### Answers to the 30 required questions

1. **Would each pass today's OOS gate?** Yes, all six.
2. **OOS sub-half consistency?** Yes, all six.
3. **±20% parameter perturbation?** Not tested — INSUFFICIENT DATA, disclosed.
4. **2x cost stress?** Not tested — NOT COMPUTABLE, disclosed.
5. **Historical regime testing?** Yes, all six positive in every available period.
6. **Broad parameter plateaus?** Unknown — not tested this phase; informal prior evidence (different methodology) suggests the AMR pairs and Monday were reported stable in their original validation.
7. **Narrow parameter dependence?** Unknown, same reason.
8. **Which fail robustness?** 3 fail HIGH-volatility (`AUDJPY_AMR`, `CADJPY_AMR`, `CADJPY_ARB`); 3 fail drawdown-correlation (`AUDJPY_AMR`, `EURJPY_AMR`, `GBPJPY_AMR`).
9. **Strongest evidence?** `GBPUSD_MONDAY` (cleanest profile across every computable gate) and `GBPJPY_AMR` (strongest OOS PF and largest contributor, though failing drawdown-correlation).
10. **Weakest evidence?** `CADJPY_AMR` (weakest OOS PF, fails HIGH-vol, borderline drawdown-correlation) and `AUDJPY_AMR` (fails both HIGH-vol and drawdown-correlation).
11. **Any live strategy rejected if discovered today?** On the computable gates: `AUDJPY_AMR` would face the most scrutiny (fails 2 of 2 tested secondary gates). None fails Gate 1 itself.
12. **How much of portfolio return depends on GBPJPY_AMR + EURJPY_AMR?** 50.2% combined (29.9% + 20.3%).
13. **Robust or fragile dependence?** Mixed — both individually pass Gate 1/consistency/regime robustness, but both fail (or in EURJPY's case, also fail) drawdown-correlation, so the dependence is on strategies with a shared, disclosed weakness, not an independently-robust pair.
14. **Effective diversification still ~3.1?** Consistent with this phase's findings — 3 of 6 strategies sharing the drawdown-correlation weakness reinforces rather than contradicts Phase 45's effective-N estimate.
15. **What does the live sample say per strategy?** Mixed, small-sample results; no dramatic reversal for any strategy (§21).
16. **Any live result statistically unusual?** `AUDJPY_AMR` and `CADJPY_AMR` are UNUSUAL BUT NOT DECISIVE; none reaches the extreme-tail deterioration range.
17. **Evidence of genuine live deterioration?** No.
18. **Any strategy show a failure mode NOT present in historical research?** No — HIGH-vol and drawdown-correlation weaknesses were already characterized in Phase 41/42/43; this phase confirms they are visible strategy-by-strategy, not a new discovery.
19. **Any strategy show a known historical weakness repeating live?** The live samples are too small to confirm this at the strategy level (§21).
20. **Does the portfolio remain defensible?** Yes, in the same qualified sense as Phase 45 — real edge, real disclosed gaps, no deterioration signal.
21. **What evidence would change that conclusion?** A completed parameter/cost robustness test showing sign reversal or catastrophic cost sensitivity for a major contributor; a live sample large enough to show sustained, statistically notable underperformance.
22. **What should NOT change the conclusion?** The current small live samples alone; the already-known and already-characterized HIGH-vol/drawdown-correlation weaknesses, which are not new information.
23. **Should any strategy enter formal review?** The three HIGH-vol failures and three drawdown-correlation failures are recorded as FUTURE STRATEGY REVIEW CANDIDATES (§30) — not automatically triggering review, per the phase's own no-repair rule, but flagged.
24. **Should any strategy be removed?** No — insufficient evidence and explicitly out of this phase's authority.
25. **Should any strategy receive additional risk?** No evidence-based basis established this phase.
26. **Should any strategy receive reduced risk?** No evidence-based basis established this phase.
27. **Should a new strategy search begin?** No — unchanged from Phase 39/45.
28. **Single biggest remaining evidence gap?** The parameter/cost-stress re-execution infrastructure (§4, §11-13) — the same gap Phase 45 identified, now further specified as requiring an actual re-runnable backtest harness around the live strategy source.
29. **What should Phase 47 investigate?** Building the re-execution infrastructure to finally close the parameter/cost-stress gap for these six strategies — the single most concrete, well-specified next step across two consecutive phases now.
30. **Should Phase 47 investigate anything at all?** Yes, narrowly — the infrastructure gap specifically, not a new strategy search or a portfolio-control redesign.

---

## Safety check confirmation

Preregistration committed (`6e0875f`) before results, unchanged after · research validator passed · six strategies frozen and verified against the repository · historical definitions documented directly from source code (2 genuine discrepancies discovered and disclosed, not corrected) · version changes disclosed where found · OOS/parameter/cost/regime/volatility/drawdown-correlation frameworks unmodified from Phases 31-45 · no parameter optimization · no strategy repair · no live strategy touched · no live risk touched · no deployment · live sample separated (pre-demotion/post-demotion/7th non-control strategy) · candidate comparison completed · survivorship bias assessed and disclosed, not resolved · Monte Carlo methodology consistent with Phases 37-45 · raw production 5ers export not committed.

---

*No live trading change authorized. No strategy modified, repaired, paused, or removed. All six strategies classified B (PLAUSIBLE — EVIDENCE STILL INSUFFICIENT); none reaches A; none reaches C or D. Recommended next step: build the re-execution infrastructure to close the parameter/cost-stress evidence gap.*
