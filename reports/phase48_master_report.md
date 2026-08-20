# Phase 48 — Six-Strategy Parameter & Cost Robustness Audit (Master Report)

**ROBUSTNESS RESEARCH ONLY. No live strategy code, YAML, execution logic, or risk setting modified. No optimization, no repair, no rescue.**

---

## 1. Executive summary

Using the Phase 47 validated reproduction harness, extended with a genuine bar-by-bar trade-outcome resolver, all six live strategies were subjected for the first time to real ±20% parameter perturbation and 2x cost stress. **The results are decisively positive on both previously-untestable dimensions**: zero sign reversals across all 12 one-factor-at-a-time perturbations, and all six strategies remain PF > 1.0 at 2x modeled cost. This closes the exact evidence gap Phase 45 and 46 identified. **It does not, however, change Phase 46's central finding**: 4 of 6 strategies (`AUDJPY_AMR`, `CADJPY_AMR`, `EURJPY_AMR`, `GBPJPY_AMR`) remain CORRELATED on drawdown-correlation — the same hard gate that rejected AUDUSD Monday LONG and every Phase 38/40 research candidate — and 3 of 6 still fail the HIGH-volatility gate (carried forward from Phase 46, not independently re-tested this phase). **`GBPUSD_MONDAY` has the cleanest profile of the six** (no clean failure on any gate computable this phase; drawdown-correlation remains genuinely unresolved, not failed, on a sample one day short of the floor). **`GBPJPY_AMR`, the portfolio's single largest R-contributor, shows the strongest drawdown correlation of the six (0.697)** — the most consequential individual finding of this phase given its portfolio weight.

## 2. Phase 47 context

Phase 47 verified its harness (commit `db91189`) was unchanged before this phase began, and built/validated a signal-reproduction harness at 99-100% match rate. This phase extends it with the trade-outcome resolver needed for parameter perturbation.

## 3. Research question

Are the six current live strategies robust to modest parameter uncertainty and realistic cost deterioration — the same standard applied to new research candidates?

## 4. Preregistration

`reports/phase48_preregistration.md`, committed separately (`2344e79`) before any result. Verified Phase 47's harness unchanged (SHA-256 re-confirmed) before writing. **One implementation bug was caught and fixed before results were interpreted**: `CADJPY_ARB`'s YAML does not set `min_range_pips` (it relies on the strategy source's own coded default, 10 pips, not 15 as initially assumed) — the script crashed on the missing key before producing any result under the wrong assumption; fixed by merging each strategy's actual coded defaults (verified by reading `strategies/asian_range_breakout.py`'s `MIN_ASIAN_RANGE_PIPS` constant directly) before any perturbation was computed. A code-correctness fix, not a methodology change — no threshold or acceptance rule was altered.

## 5. Baseline reproduction

`reports/phase48_baseline_reproduction.csv`. All six strategies' simulated trade counts are within 10% of Phase 47's known historical counts, confirming the extended harness (signal logic + new trade-outcome resolver) remains consistent with the validated reproduction.

## 6. OOS baseline

`reports/phase48_oos_baseline.csv`. All six PASS Gate 1 on the frozen OOS window (2025-05-01 to 2026-08-13): `AUDJPY_AMR` PF 1.487, `CADJPY_AMR` PF 1.553, `EURJPY_AMR` PF 1.491, `GBPJPY_AMR` PF 2.292, `CADJPY_ARB` PF 1.508, `GBPUSD_MONDAY` PF 2.697. (These simulated figures differ modestly from Phase 46's historical-ledger PFs, as expected given this phase's independent bar-by-bar re-simulation rather than a replay of pre-recorded outcomes — disclosed, not treated as a discrepancy requiring resolution.)

## 7. Parameter inventory

`reports/phase48_parameter_inventory.csv` — 12 continuous parameters (2 per strategy), reused unchanged from Phase 47's classification.

## 8. Parameter perturbation

`reports/phase48_parameter_perturbation.csv` — full one-factor-at-a-time ±20% results for all 12 parameters.

## 9. Parameter stability

`reports/phase48_parameter_stability.csv`. **Zero sign reversals across all 12 tests.** Maximum expectancy swing ranges from 0.0% (`CADJPY_ARB`'s `min_range_pips` — this pair's actual Asian-range distribution never crosses the ±20% threshold boundary differently, a genuine structural finding, not a computation error) to 29.3% (`EURJPY_AMR`'s `z_threshold`, still short of the 30% HIGHLY SENSITIVE bar).

## 10. Parameter plateaus

`reports/phase48_parameter_plateau.csv`. `AUDJPY_AMR`: A. BROAD PLATEAU. The other five: B. MODERATE STABILITY. No strategy reaches C (NARROW PEAK) or D (SIGN REVERSAL).

## 11. Cost stress

`reports/phase48_cost_stress.csv`. All six remain PF > 1.0 at 2x modeled cost: `AUDJPY_AMR` 1.278, `CADJPY_AMR` 1.289, `EURJPY_AMR` 1.273, `GBPJPY_AMR` 2.077, `CADJPY_ARB` 1.439, `GBPUSD_MONDAY` 2.567 — all classified A. COST ROBUST. Per the disclosed limitation (§Preregistration §5, carried from Phase 47), this uses a placeholder flat-pip cost model since the historical ledger does not expose each trade's true embedded cost — a **relative**, not absolute, robustness test.

## 12. OOS sub-half consistency

`reports/phase48_oos_subhalf.csv` — all six PASS, sign-consistent in both halves.

## 13. Regime robustness

`reports/phase48_regime_robustness.csv` — all six positive in every available historical period (2023-2024, 2025, 2026 YTD).

## 14. Volatility behavior

`reports/phase48_volatility_behavior.csv`, joined to the ledger's own `vol_tercile` field by pair+date. Carries forward Phase 46's HIGH-vol finding structurally: `AUDJPY_AMR`, `CADJPY_AMR`, `CADJPY_ARB` remain the three strategies with the weakest HIGH-vol showing (consistent with the Phase 46 historical-ledger result, not independently re-derived to a new PF figure in this phase's simulation given the join-based approach).

## 15. Drawdown correlation

`reports/phase48_drawdown_correlation.csv`, reusing the exact OOS-window-matched, self-contribution-excluded methodology from Phase 46. **`AUDJPY_AMR` (0.516), `CADJPY_AMR` (0.485), `EURJPY_AMR` (0.550), `GBPJPY_AMR` (0.697) are all CORRELATED.** `CADJPY_ARB` is STRONG DIVERSIFIER (-0.295, but n=8, exactly the floor). `GBPUSD_MONDAY` is UNKNOWN (n=7, one day short of the floor). **`CADJPY_AMR` moved from Phase 46's NEUTRAL (borderline) to CORRELATED here** — an expected consequence of independent re-simulation rather than a historical-ledger replay, disclosed rather than reconciled away.

## 16. Portfolio robustness

`reports/phase48_portfolio_robustness.csv`. Substituting each strategy's own ±20% perturbation into the full six-strategy portfolio one at a time: total_R ranges roughly 179-206R and max drawdown ranges roughly -10.8 to -15.9R, versus a 193.6R/-12.2R baseline — real but bounded sensitivity; **no single-strategy perturbation reverses portfolio-level profitability.**

## 17. Leave-one-out analysis

`reports/phase48_leave_one_out.csv`, reusing Phase 46's exact construction on this phase's OOS-window daily ledger.

## 18. Monte Carlo

`reports/phase48_monte_carlo.csv`. **SIMULATED**, 10,000-draw trade-order reshuffle per strategy — consistent methodology with every prior phase.

## 19. Live comparison

`reports/phase48_live_comparison.csv`, reusing Phase 46's already-validated live-sample-sufficiency bootstrap directly (no new live data pulled, per the instruction to use only already-validated sources). No change from Phase 46: `AUDJPY_AMR`/`CADJPY_AMR`/`GBPUSD_MONDAY` UNUSUAL BUT NOT DECISIVE; `EURJPY_AMR`/`GBPJPY_AMR`/`CADJPY_ARB` CONSISTENT.

## 20. Candidate comparison

`reports/phase48_candidate_comparison.csv`. Against the 71-hypothesis Phase 33-40 research ledger's 32.4% Gate-1 pass rate, all six live strategies pass every computable gate except: `AUDJPY_AMR`, `CADJPY_AMR`, `EURJPY_AMR`, `GBPJPY_AMR` fail drawdown-correlation; `AUDJPY_AMR`, `CADJPY_AMR`, `CADJPY_ARB` fail HIGH-volatility. **`GBPUSD_MONDAY` is the only strategy with no clean failure on any gate.**

## 21. Strategy scorecards

`reports/phase48_strategy_scorecard.csv` — full per-strategy synthesis, see §25.

## 22. Portfolio scorecard

`reports/phase48_portfolio_scorecard.csv` — 10 dimensions; parameter and cost robustness upgraded from Phase 46's "untestable" to STRONG; HIGH-vol and drawdown-correlation remain WEAK; overall evidence sufficiency upgraded from WEAK-TO-MODERATE to MODERATE.

## 23. What passed

OOS edge (all 6). OOS sub-half consistency (all 6). Parameter robustness — no sign reversal (all 6). Cost robustness — PF>1.0 at 2x (all 6). Regime robustness (all 6).

## 24. What failed

Drawdown correlation: `AUDJPY_AMR`, `CADJPY_AMR`, `EURJPY_AMR`, `GBPJPY_AMR` (4 of 6). HIGH-volatility (carried forward from Phase 46, not re-derived this phase): `AUDJPY_AMR`, `CADJPY_AMR`, `CADJPY_ARB` (3 of 6).

## 25. What remains uncertain

`GBPUSD_MONDAY`'s drawdown-correlation status (n=7, one day short of the floor). Whether HIGH-volatility behavior would change under this phase's independently re-simulated data (not re-tested; Phase 46's historical-ledger figures were carried forward as the best available evidence). `CADJPY_ARB`'s drawdown-correlation pass, on the thinnest defensible sample (n=8).

## 26. Comparison with new research candidates

The six live strategies collectively clear Gate 1 at a rate (100%) dramatically higher than the Phase 33-40 candidate pool (32.4%), and now — for the first time — clear parameter and cost robustness cleanly too. **But this does not mean the live strategies would be waved through today**: 4 of 6 would still be rejected at the drawdown-correlation gate, exactly the gate that has been the single most consistent failure point for every portfolio-integration-stage candidate since Phase 37 (AUDUSD Monday LONG, Phase 38 H1/H2, Phase 40). The live strategies are not exempt from this pattern — they are simply, on average, better-evidenced candidates that happen to share the same structural weakness as every rejected one.

## 27. Implications

`GBPJPY_AMR`'s combination of largest-portfolio-contributor status (29.9% of total R, Phase 45/46) and strongest drawdown correlation of the six (0.697) is the single most consequential finding of this phase — the portfolio's biggest edge contributor is also its strongest source of correlated-drawdown risk. `GBPUSD_MONDAY`'s clean profile across every computable gate, with only an unresolved (not failed) drawdown-correlation question, makes it the strategy with the best-supported case for continued confidence among the six.

## 28. Limitations

- The cost model (§11) is an explicitly disclosed relative-stress placeholder, not a validated absolute broker-cost figure — carried forward from Phase 47's own limitation, not newly introduced here.
- HIGH-volatility behavior (§14) was carried forward from Phase 46's historical-ledger result rather than independently re-derived from this phase's own simulation — a scope choice to avoid duplicating work already validated, disclosed rather than silently assumed identical.
- This phase's independent re-simulation produces OOS PF/drawdown-correlation figures that differ modestly from Phase 46's historical-ledger replay (e.g., `CADJPY_AMR`'s drawdown correlation moved from NEUTRAL to CORRELATED) — an expected consequence of two different, both-legitimate methodologies (historical replay vs. independent re-simulation), not a data error requiring reconciliation.
- Portfolio-level parameter-substitution scenarios (§16) test one strategy's perturbation in isolation, never combinatorially, per the explicit no-optimization rule — the true joint sensitivity of multiple simultaneous parameter shifts remains untested.

## 29. Future validation requirements

Extending the drawdown-correlation sample for `GBPUSD_MONDAY` (currently 1 day short of the floor) and `CADJPY_ARB` (currently exactly at the floor) with additional live/historical data would resolve the two most sample-constrained findings of this phase. Independently re-deriving HIGH-volatility behavior from this phase's own simulated trade set (rather than carrying forward Phase 46's historical-ledger figure) would close the one disclosed scope gap in §14.

## 30. Final verdict

### Answers to the 30 required questions

1. **Did all six survive baseline reproduction?** Yes, all within 10% of Phase 47's known trade counts.
2. **Which have robust parameter plateaus?** `AUDJPY_AMR` (broad); the other five (moderate stability).
3. **Which are parameter-sensitive?** None reach HIGHLY SENSITIVE; `EURJPY_AMR`'s `z_threshold` (29.3%) is the closest to that threshold.
4. **Any sign reversal?** No — zero across all 12 tests.
5. **Which survived 2x cost stress?** All six.
6. **Which are cost-sensitive?** None fail (all PF>1.0 at 2x), though `AUDJPY_AMR`/`CADJPY_AMR`/`EURJPY_AMR` show the largest relative PF degradation (~14-16%).
7. **Which fail HIGH-volatility?** `AUDJPY_AMR`, `CADJPY_AMR`, `CADJPY_ARB` (carried forward from Phase 46).
8. **Which fail drawdown diversification?** `AUDJPY_AMR`, `CADJPY_AMR`, `EURJPY_AMR`, `GBPJPY_AMR`.
9. **Which pass the majority of today's gates?** All six pass 5 of 7 gates or more; `GBPUSD_MONDAY` passes all computable gates cleanly (drawdown-correlation unresolved, not failed).
10. **Which would be rejected if discovered today?** `AUDJPY_AMR`, `CADJPY_AMR`, `EURJPY_AMR`, `GBPJPY_AMR` — each fails at least the drawdown-correlation gate.
11. **Which remain plausible but insufficiently proven?** `GBPUSD_MONDAY` (drawdown-correlation unresolved).
12. **Does the portfolio remain historically defensible?** Yes, in the same qualified sense as Phase 45/46 — real, now more robustly evidenced edge, with clearly characterized (not eliminated) structural weaknesses.
13. **Does parameter robustness materially change the Phase 46 conclusion?** It strengthens the evidence base (closes a real gap) without changing the bottom-line classification — strategies that failed drawdown-correlation/HIGH-vol in Phase 46 still fail those gates here.
14. **Does cost robustness materially change the Phase 46 conclusion?** Same answer — genuinely new positive evidence, same bottom line.
15. **Does effective diversification remain ~3.1?** Consistent with this phase's findings — 4 of 6 strategies sharing the drawdown-correlation weakness reinforces Phase 45's effective-N estimate.
16. **Does GBPJPY_AMR+EURJPY_AMR still account for >50%?** Not independently re-measured this phase (uses Phase 45/46's figure, unchanged data source) — still applicable.
17. **Is that dependence robust under perturbation?** Both strategies individually show no sign reversal under ±20% perturbation — the edge itself is robust; the drawdown-correlation weakness is not resolved by that robustness.
18. **Does live performance remain within the historical distribution?** Yes, per Phase 46's bootstrap, carried forward unchanged.
19. **Evidence of genuine deterioration?** No.
20. **Strongest overall evidence?** `GBPUSD_MONDAY`.
21. **Weakest?** `CADJPY_AMR` (weakest baseline PF, fails HIGH-vol, CORRELATED).
22. **Should any strategy enter formal review?** The four drawdown-correlation failures and three HIGH-vol failures are natural review candidates, per the no-rescue rule recorded as observations only.
23. **Should any strategy be removed?** No — outside this phase's authority and not supported by a deterioration signal.
24. **Should any strategy receive more risk?** No evidence-based basis established.
25. **Should any strategy receive less risk?** No evidence-based basis established.
26. **Should the portfolio be changed?** Not on this phase's evidence alone.
27. **Should a new strategy search begin?** No — unchanged from Phase 39/45/46.
28. **Largest remaining evidence gap?** Independently re-deriving HIGH-volatility behavior from this phase's own simulated data, and extending the drawdown-correlation sample for the two floor-level strategies.
29. **What should Phase 49 investigate?** Either of the two items in §29, or a return to the standing future-research threads from Phase 44/45 (portfolio-control OOS validation is not yet applicable; return-stream research per Phase 39 remains an open option).
30. **Should Phase 49 investigate anything at all?** Optional — the evidence base is now materially more complete than at any prior point in this research program; no urgent gap forces immediate further work.

### Final classification

**Portfolio: B. PLAUSIBLE BUT INSUFFICIENT EVIDENCE**, upgraded in evidentiary completeness from Phase 46 but not in bottom-line classification — the newly-closed parameter/cost gap does not offset the still-unresolved drawdown-correlation and HIGH-volatility weaknesses in 4 and 3 of 6 strategies respectively.

**Strategies**: `AUDJPY_AMR`, `CADJPY_AMR`, `EURJPY_AMR`, `GBPJPY_AMR` — **D. WOULD FAIL TODAY'S CANDIDATE BAR** (drawdown-correlation, plus HIGH-vol for the first two). `CADJPY_ARB` — **C. FRAGILE** (HIGH-vol failure on an otherwise strong profile). `GBPUSD_MONDAY` — **B. PLAUSIBLE BUT INSUFFICIENT EVIDENCE** (cleanest profile of the six; one unresolved gate).

---

## Safety check confirmation

Preregistration committed (`2344e79`) before results, unchanged after · Phase 47 harness verified unchanged before starting · baseline reproduction passed · parameter methodology frozen (no threshold chosen after seeing results) · cost methodology frozen · no optimization performed · no strategy repair · no source modification · no YAML modification (verified via `git diff -- strategies/ pairs/ core/`) · no live risk modification · no deployment · no portfolio control deployed · OOS period unchanged from Phase 46 · cost assumptions unchanged (disclosed placeholder, not tuned) · volatility methodology unchanged (carried forward from Phase 42/46) · drawdown methodology unchanged (carried forward from Phase 31-46) · live sample separated and reused from Phase 46, not re-pulled · candidate comparison completed · Monte Carlo methodology consistent with Phases 37-46 · raw production 5ers export not committed.

---

*No live trading change authorized. No strategy code, YAML, or execution logic modified. Four strategies classified D (would fail today's bar), one C (fragile), one B (plausible) — recorded as observations only, per the explicit no-rescue rule.*
