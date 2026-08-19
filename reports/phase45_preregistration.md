# Phase 45 Preregistration — Portfolio Viability & Evidence Sufficiency Audit

**Frozen before substantive analysis. Committed separately, before any Phase 45 conclusion exists. Not modified after seeing results.**

RESEARCH / DECISION-FRAMEWORK ONLY. No strategy created, modified, paused, or removed. No portfolio control deployed. No live-system change of any kind. The strongest permitted recommendation is "CONTINUE VALIDATION" / "INVESTIGATE" / "RESEARCH REQUIRED" / "SUFFICIENT EVIDENCE FOR A FUTURE DECISION" — never "deploy," "implement," "increase risk," or "decrease risk."

---

## 1. Source-of-truth inventory (frozen before inspection, per Part 4)

**Master research ledger base**: `reports/phase39_fx_research_inventory.csv` (70 rows — Phase 36's 68-row ledger, AUDUSD Monday LONG updated in place with Phase 37's full validation, plus Phase 38's H1/H2) — **extended in this phase with Phase 40's HIGH-volatility-state candidate (1 new row) to form a 71-row ledger.** Phases 41-44 are forensic/counterfactual research, not new trading hypotheses, and are referenced narratively (§Research ceiling, §Phase 44 findings) rather than added as ledger rows.

**Current six-strategy control**: `data/phase26_all_trades.csv` (2,712 trades, unchanged since Phase 31, identical membership used throughout Phases 31-44: `EURJPY_AMR`, `AUDJPY_AMR`, `CADJPY_AMR`, `GBPJPY_AMR`, `CADJPY_ARB`, `GBPUSD_MONDAY`).

**Live production data**: the local, uncommitted `reports/5ers_trade_export.csv` (raw production export — **never committed**, per this project's standing convention, verified unchanged across every phase in this session) is used as the freshest available live source. At the time of this preregistration it contains 72 rows, 36 CLOSED / 36 OPEN, spanning 2026-07-20 to 2026-08-13, and includes a 7th strategy (`GBPJPY_ARB`) not part of the frozen current-6 control — this strategy's trades are reported **separately**, never pooled into the current-6 live sample, since it is not part of the frozen control definition used since Phase 31. The previously-used `reports/5ers_portfolio_update_aug13_trade_level.csv` (19-trade excerpt) is superseded by this fuller export where they overlap and is not separately re-analyzed.

**Experiment ledger**: `experiments/experiments.csv` (142 rows as of this preregistration, EXP-001 through EXP-142).

## 2. Historical windows (identical convention to Phases 41-44)

A = full historical control (2,712 trades). B = pre-demotion (entry_time < 2026-07-31, within the same reconstruction). C = post-demotion live (the CLOSED subset of the live export, current-6 membership only, entry_time ≥ 2026-07-31) — reported with explicit sample size, never pooled with A/B.

## 3. Data integrity (per Part 5)

`research_data_validator` run on `data/phase26_all_trades.csv`, `experiments/experiments.csv`, and `reports/5ers_trade_export.csv` before analysis. Reconciliation against Phase 44's own use of the same control file (trade count, date range, strategy composition) is required before proceeding — if it does not reconcile, STOP per every prior phase's convention.

## 4. Master ledger fields (frozen, per Part 6)

experiment_id, phase, hypothesis_id, strategy_family, instrument, mechanism, session, direction (where available), IS sample, OOS sample, OOS PF, OOS R, robustness result, cost stress result, regime result, drawdown correlation, portfolio integration, final classification, rejection/acceptance reason, deployed (Y/N), currently active (Y/N), evidence confidence (STRONG/MODERATE/WEAK/INSUFFICIENT, assigned per §Evidence hierarchy below). `NOT AVAILABLE` used wherever a field cannot be sourced from a committed artifact — never inferred.

## 5. Research-family taxonomy (reused, not reinvented)

Identical to the taxonomy already established across Phases 36/39/41 (`calendar_drift`, `mean_reversion` [AMR], `range_breakout` [ARB], `trend_momentum_continuation`, `volatility_contraction_expansion_breakout`, `new_york_open_range_breakout`, `new_york_session_momentum`, `london_ny_overlap_continuation`, `multi_timeframe_trend_continuation`, `cross_sectional_relative_momentum`, `session_transition_breakout_continuation`, `volatility_conditioned_trend_continuation` [Phase 40]).

## 6. Strategy independence audit methodology (frozen, per Part 8)

Reuses Phase 41's already-computed, already-validated conditional-correlation matrix (`reports/phase41_conditional_correlation.csv`, all 15 strategy pairs, full-period/normal-day/stress-day correlations) rather than recomputing — no new correlation methodology is invented in this phase. Effective diversification is reported using the simple inverse-Herfindahl-style measure already used in Phase 31 (`effective N`), reusing that prior methodology.

## 7. Historical portfolio edge / contribution methodology (frozen, per Parts 9-10)

Reuses Phase 41's daily portfolio ledger (`reports/phase41_daily_portfolio_ledger.csv`) and Phase 44's baseline metrics (`reports/phase44_baseline.csv`) for aggregate portfolio statistics — not recomputed from scratch, to avoid any risk of a silent methodology drift from the already-validated figures. Per-strategy contribution (total R, trade count, win/loss split) is computed directly from `data/phase26_all_trades.csv` grouped by `strategy` — a simple, direct aggregation, not a modeling exercise.

## 8. Live sample sufficiency methodology (frozen, per Part 13)

For the post-demotion live sample (current-6 membership, CLOSED trades only), a **block-bootstrap** of the historical control (reusing the Monte-Carlo trade-order-reshuffle infrastructure already established in Phases 37-40/44) draws random contiguous blocks of historical trades of the same size as the live sample, repeated 10,000 times, to determine the percentile of the live sample's observed total R and win rate within the historical distribution of same-sized samples. This is the same class of methodology already used project-wide (SIMULATED, clearly labeled), not a new statistical framework invented for this phase.

## 9. Deterioration / continued-viability framework (frozen, per Parts 15-16)

Thresholds are derived **only** from already-published historical distributional evidence (e.g., Phase 41's own worst-1%/5%/10%/20% daily-R stress-window definitions, Phase 37/40's Monte Carlo percentile conventions) — never chosen to match the current live result. Where no historical evidence supports a specific numeric threshold, the framework explicitly states `NOT YET JUSTIFIABLE` rather than inventing one.

## 10. Evidence hierarchy (frozen, per Part 22, used to assign the ledger's evidence-confidence field)

1. Repeated historical evidence across regimes (STRONGEST) → 2. Pre-registered OOS evidence → 3. Robust OOS evidence (survives parameter/cost stress) → 4. Portfolio integration evidence → 5. Live evidence with adequate sample → 6. Historical in-sample evidence → 7. Exploratory associations → 8. Single-sample observations (WEAKEST).

## 11. Multiple-testing / scope discipline

This phase performs zero new backtests and zero new counterfactuals — it is a pure consolidation and audit of already-completed, already-committed research. No new statistical test is introduced beyond the bootstrap described in §8, which reuses established infrastructure. No finding from this phase may be used to silently alter any Phase 30-44 conclusion — corrections, if any are found, are documented as `CORRECTED RESULT` alongside the `ORIGINAL RESULT`, per Part 5's explicit instruction, never a silent replacement.

## 12. Classification framework (frozen, per Part 34)

Final portfolio classification: exactly one of A (HISTORICALLY ROBUST — CONTINUE LIVE VALIDATION) / B (HISTORICALLY PLAUSIBLE — LIVE EVIDENCE INSUFFICIENT) / C (HISTORICALLY FRAGILE — REASSESS PORTFOLIO) / D (LIVE DETERIORATION SIGNAL — FORMAL REVIEW REQUIRED) / E (INSUFFICIENT EVIDENCE — CONTINUE OBSERVATION), selected strictly from the evidence assembled in this phase, not from prior expectation in either direction.

---

*No amendment has been made to this document after any Phase 45 conclusion was produced.*
