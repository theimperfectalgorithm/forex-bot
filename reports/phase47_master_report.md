# Phase 47 — Live-Source Reproduction & Robustness Harness (Master Report)

**INFRASTRUCTURE + REPRODUCTION ONLY. No live strategy code, YAML, execution logic, or risk setting modified. No deployment. No robustness conclusions produced — this phase establishes only whether the testing machinery is trustworthy.**

---

## 1. Executive summary

A source-faithful reproduction harness was built and run against the actual live strategy source (`strategies/asian_hours_reversion.py`, `asian_range_breakout.py`, `monday_drift.py`) and real MT5 historical price data for all six current-6 strategies. **Signal reproduction is strong: all six strategies reach 99.0-100% match rate against the known historical trade ledger — well above the preregistered 85% threshold for A. REPRODUCTION PASS.** Both Phase 46's disclosed discrepancies were investigated directly from source: `CADJPY_ARB`'s H4-filter disablement is **partially explained** by the source code's own documented rationale (a per-pair override, evidenced for GBPJPY, not independently justified for CADJPY); `GBPJPY_AMR`'s breakeven-logic comment is **stale documentation, not a live bug** — the actual execution code excludes all `@`-tagged book strategies from generic breakeven handling, correctly matching backtest assumptions. The reproduction harness passed determinism (identical results across two independent runs, all six strategies) and immutability (zero source/config file changes, verified by SHA-256 hash comparison) tests. The parameter and cost harnesses were built and sanity-tested (16/16 checks pass) but, per the preregistered scope, **produce no final robustness conclusions in this phase.** **Final classification: B. PARTIALLY REPRODUCTION READY — LIMITED BLOCKERS** — signal reproduction is strong and the harnesses are technically trustworthy, but execution/fill-price reproduction remains explicitly APPROXIMATE (no broker-level spread/slippage model), which the preregistration itself flagged as a disclosed limitation, not a full EXACT REPRODUCTION.

## 2. Phase 46 context

Phase 46 applied the Phase 33-40 robustness framework to the six live strategies and found all six passed OOS edge/consistency, but flagged parameter/cost-stress testing as impossible without a re-execution harness, and surfaced two documented-vs-live discrepancies.

## 3. Research question

Can a trustworthy reproduction harness be built that executes the actual six live strategy implementations against historical data and reproduces their known behavior, before it is trusted for robustness testing?

## 4. Preregistration

`reports/phase47_preregistration.md`, committed separately (`2edd176`) before any reproduction was run. No amendment required.

## 5. Source inventory

`reports/phase47_source_inventory.csv` — all 3 strategy source files located and SHA-256 hashed (`asian_hours_reversion.py`, `asian_range_breakout.py`, `monday_drift.py`); nothing marked missing.

## 6. Live configuration snapshot

`reports/phase47_live_config_snapshot.csv` — all 6 pair YAML configs read directly and SHA-256 hashed, full config content preserved as JSON in the CSV.

## 7. Documented vs. live discrepancies

`reports/phase47_documented_vs_live.csv`. **`CADJPY_ARB` H4 filter**: the strategy's own `prepare()` method contains an inline comment explaining that `h4_filter: false` is an intentional, supported per-pair override — the source explicitly cites GBPJPY-specific walk-forward evidence for disabling it, but does **not** independently justify CADJPY's disablement — classified **PARTIALLY EXPLAINED, NOT A BUG, but UNRESOLVED** whether CADJPY was deliberately tuned this way or the GBPJPY finding was assumed to generalize. **`GBPJPY_AMR` breakeven logic**: direct inspection of `src/agents/agent_execution.py` found the 25-pip generic breakeven rule is explicitly skipped for every `@`-tagged book strategy (not just `GBPJPY_AMR`), with an inline comment citing a specific live incident (2026-07-10, a CADJPY@arb breakeven scratch) as the reason for the exclusion — this **directly contradicts** the older YAML comment's implication that `GBPJPY_AMR` still receives the 25-pip rule. Classified **DOCUMENTATION IS STALE, NOT A LIVE BUG.** Neither discrepancy was corrected, per the explicit no-fix rule.

## 8. Historical data inventory

`reports/phase47_data_inventory.csv` — M15/H1/H4 bars for all 5 relevant pairs, 2023-08-01 to 2026-08-13, all classified EXACT REPRODUCTION (broker OHLC fully available, tens of thousands of bars per pair/timeframe) — no INSUFFICIENT DATA classification was needed.

## 9. Reproduction targets

`reports/phase47_reproduction_targets.csv` — the already-validated `data/phase26_all_trades.csv` historical ledger (used as this project's control since Phase 31), 2,712 known trades across the six strategies.

## 10. Trade-level reproduction

`reports/phase47_trade_reproduction.csv` — full per-trade match classification for all reconstructed and historical trades, pair/date/direction methodology per the preregistration.

## 11. Reproduction metrics

`reports/phase47_reproduction_metrics.csv`. Match rates: `EURJPY_AMR` 100.0%, `GBPJPY_AMR` 99.8%, `AUDJPY_AMR` 99.7%, `GBPUSD_MONDAY` 99.4%, `CADJPY_AMR` 99.5%, `CADJPY_ARB` 99.0% — **all six clear the 85% A-grade threshold decisively.** False-positive counts (reconstructed signals with no historical counterpart) are modest (10-30 per strategy, likely reflecting the disclosed approximate execution model — e.g., a reconstructed signal that would have been filtered by a spread/slippage check not modeled here); false-negative counts (historical trades not reconstructed) are near-zero (0-3 per strategy).

## 12. Reproduction failures

`reports/phase47_reproduction_failures.csv` — empty; no strategy fell below the 85% threshold, so no failure diagnosis was required.

## 13. Portfolio reproduction

`reports/phase47_portfolio_reproduction.csv`. Known historical: 2,712 trades, total R 194.11. Reconstructed: 2,797 signal-level trades (a 3.1% overcount, consistent with the modest false-positive rates in §11) — **full R-multiple portfolio reconstruction was not attempted**, since this phase's execution model does not compute exact fills (disclosed APPROXIMATE REPRODUCTION scope, not a gap discovered after the fact).

## 14. Parameter inventory

`reports/phase47_parameter_inventory.csv` — 2 continuous perturbable parameters identified per strategy (e.g. `z_threshold`/`sl_multiplier` for AMR, `tp_multiplier`/`min_range_pips` for ARB, `sl_atr_mult`/`tp_atr_mult` for Monday); `h4_filter`, `entry_end_hour`, session windows, and `risk_percent` explicitly excluded as categorical/structural, per the preregistration.

## 15. Parameter-harness readiness

`reports/phase47_parameter_sanity.csv` — 12/12 sanity checks PASS: perturbing one parameter by ±20% changes only that parameter, never an unrelated one, and never mutates the base config in place. `tests/test_phase47_parameter_harness.py` (6 tests) confirms this programmatically, including a negative-control test that verifies the sanity check itself would catch a deliberately broken perturbation function.

## 16. Cost-harness readiness

`reports/phase47_cost_sanity.csv` — 4/4 non-trivial sanity checks (1.5x/2.0x) PASS: cost stress changes only `r_multiple`, never SL/TP/direction/timing fields, and never mutates the original trade dict. `tests/test_phase47_cost_harness.py` (5 tests) confirms this programmatically.

## 17. Determinism

`reports/phase47_determinism.csv` — all six strategies produced byte-for-byte identical reconstructed trade lists across two independent harness runs. No STOP condition triggered.

## 18. Immutability

`reports/phase47_immutability.csv` — SHA-256 hashes of all 3 source files and 6 YAML configs, captured before and after the full harness run: **zero changes across all 9 files.** The harness never wrote to any live strategy or configuration path.

## 19. Software-test results

18 automated tests across 3 files (`tests/test_phase47_reproduction.py`, `test_phase47_parameter_harness.py`, `test_phase47_cost_harness.py`) — all pass. Tests cover hash determinism, trade-matching logic (exact match, false positive, false negative, direction mismatch), parameter perturbation isolation, config non-mutation, cost-stress field isolation, and a negative-control test proving the sanity-check methodology itself is sound (would catch a deliberately introduced bug).

## 20. Readiness matrix

`reports/phase47_readiness_matrix.csv` — full per-strategy and per-harness synthesis.

## 21. Known limitations

- **Execution/fill reproduction is explicitly APPROXIMATE, not EXACT** — no broker-level spread or slippage model exists in this harness (none exists in the live strategy source either, which uses the raw signal-bar close), disclosed before any result was seen.
- The replay's R-multiple for AMR trades assumes the TP distance is always reached at the recorded ratio (a simplification for the reproduction check, not a full trade-lifecycle simulation with intrabar stop/target resolution) — sufficient for pair/date/direction matching but not for independently re-deriving each historical trade's exact R.
- Full portfolio-level R reconstruction was not attempted (§13), consistent with the disclosed scope.
- The `CADJPY_ARB` h4_filter question (§7) remains genuinely unresolved — the source explains the *mechanism* but not definitively whether CADJPY's specific override was deliberate or inherited.

## 22. What Phase 47 established

That the six live strategies' **signal generation logic can be reproduced from source with very high fidelity** (99.0-100% match rate) against real historical price data; that both disclosed Phase 46 discrepancies have concrete, source-verified explanations (one partially resolved, one clearly resolved as stale documentation); that the reproduction harness is deterministic and never modifies live files; that the parameter and cost harnesses are software-correct (sanity-tested, isolated, non-mutating).

## 23. What Phase 47 did NOT establish

Any robustness conclusion whatsoever — no ±20% parameter or 2x-cost result is reported in this phase, per the explicit prohibition. Exact execution-price reproduction (only signal-level reproduction was validated). Whether the CADJPY_ARB h4_filter override was a deliberate, evidenced decision for that specific pair.

## 24. Phase 48 recommendation

**Proceed to Phase 48 for the actual parameter/cost robustness audit, with the execution-model limitation explicitly carried forward and disclosed in that phase's own preregistration** — the signal-reproduction fidelity is strong enough to trust the harness for relative (baseline vs. perturbed) comparisons, even though absolute execution-price reproduction remains approximate. Phase 48 should not claim EXACT reproduction of any single historical trade's fill price.

## 25. Final verdict

### Answers to the 15 required questions

1. **Can the six live strategies be reproduced from source?** Yes — signal logic reproduces at 99.0-100% match rate for all six.
2. **Can their signals be reproduced?** Yes, cleanly.
3. **Can their execution behavior be reproduced?** Only approximately — no broker-level spread/slippage/fill model exists.
4. **Which strategies reproduce cleanly?** All six, with `EURJPY_AMR` the cleanest (100.0%, 0 false negatives) and `CADJPY_ARB` the least clean (99.0%, highest false-positive count, consistent with its disclosed h4_filter uncertainty).
5. **Which have source/config discrepancies?** `CADJPY_ARB` (H4 filter) and `GBPJPY_AMR` (breakeven logic) — both investigated and documented, neither corrected.
6. **Is CADJPY_ARB's H4 filter actually active in production?** No — `h4_filter: false` in the live YAML, confirmed directly.
7. **What does GBPJPY_AMR's live breakeven logic actually do?** Nothing — it is explicitly excluded from the generic 25-pip breakeven rule, per `agent_execution.py`'s own code and comment, confirmed directly.
8. **Are the discrepancies documentation, source, configuration, or unresolved issues?** `CADJPY_ARB`: a configuration question the source explains the mechanism for but doesn't fully justify for this specific pair (partially unresolved). `GBPJPY_AMR`: a documentation issue — the YAML comment is stale relative to the actual, correct execution code.
9. **Can the six-strategy portfolio be reproduced?** Signal-level yes (2,797 reconstructed vs. 2,712 known trades, a 3.1% overcount); full R-level portfolio reconstruction was not attempted, disclosed scope.
10. **Is the reproduction deterministic?** Yes, verified across two independent runs for all six strategies.
11. **Is the parameter harness technically trustworthy?** Yes — 12/12 sanity checks and 6 automated tests pass.
12. **Is the cost-stress harness technically trustworthy?** Yes — 4/4 sanity checks and 5 automated tests pass.
13. **What remains impossible to reproduce exactly?** Exact historical fill prices, spread, and slippage — no broker-level tick data has ever been available to this project.
14. **Is Phase 48 ready to run the actual robustness audit?** Yes, for *relative* (baseline-vs-perturbed) comparisons, with the execution-approximation limitation explicitly carried forward.
15. **If not, what exact blocker remains?** None blocking — the one disclosed limitation (approximate execution) does not prevent relative parameter/cost comparisons, since the same approximation applies equally to baseline and perturbed runs.

### Final classification

## **B. PARTIALLY REPRODUCTION READY — LIMITED BLOCKERS**

Signal reproduction, determinism, immutability, and both harnesses' software correctness are all strongly validated (well above threshold, deterministic, non-mutating). The single disclosed limitation — approximate rather than exact execution/fill reproduction — was flagged before any result was seen and does not block relative robustness comparisons in a future phase, but does mean this phase stops short of an unqualified **A**. Phase 48 may proceed using this harness for parameter/cost robustness testing, explicitly carrying forward the execution-approximation disclosure.

---

## Safety check confirmation

Preregistration committed (`2edd176`) before results, unchanged after · six strategies frozen and verified · source snapshots captured (SHA-256 hashed) · live config captured (SHA-256 hashed) · documentation discrepancies documented, not resolved · no live configuration changed · no strategy code changed · no strategy parameters changed · no risk changed · no deployment · historical data validated (all EXACT REPRODUCTION coverage, no INSUFFICIENT DATA) · reproduction attempted and passed for all six · determinism tested and passed · parameter harness sanity-tested and passed · cost harness sanity-tested and passed · source immutability verified (0/9 files changed) · automated tests added (18 tests, all passing) · no final robustness conclusions produced (explicitly deferred to Phase 48) · raw production 5ers export not committed (not used in this phase).

---

*No live trading change authorized. No strategy code, YAML, or execution logic modified. Classification: B. PARTIALLY REPRODUCTION READY — Phase 48 may proceed with the disclosed execution-approximation limitation carried forward.*
