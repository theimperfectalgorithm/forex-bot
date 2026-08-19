# Phase 44 Preregistration — Portfolio-Control Counterfactual Validation

**Frozen before any counterfactual is run. Committed separately, before any Phase 44 result exists. Not modified after seeing results.**

HISTORICAL COUNTERFACTUAL RESEARCH ONLY. No live strategy, parameter, risk, or portfolio-weight modified. No control deployed. Exactly 5 controls (A baseline + B/C/D/E interventions) are tested — no additional control may be added after seeing results, no threshold may be tuned after seeing results.

---

## 1. Control portfolio (unchanged from Phases 41-43)

`data/phase26_all_trades.csv`, 2,712 trades, 6 strategies, 2023-08-01 to 2026-08-13. No candidate strategy added.

## 2. Historical windows (identical convention)

A = full historical control. B = pre-demotion. C = post-demotion live sample (`reports/5ers_portfolio_update_aug13_trade_level.csv`, 19 trades) — reported separately if relevant, never pooled.

## 3. Cost/execution assumptions

The counterfactual does not alter entry price, exit price, SL, TP, or holding time of any executed trade — historical `r_multiple`/`pnl` are used exactly as recorded (identical cost model already embedded in the source data, no new cost assumption invented). Cost-sensitivity (§Part 23 of the task) is evaluated by re-scaling each executed trade's R by a fixed multiplier applied to its (unrecoverable, already-embedded) cost component is **not possible from this dataset** (the source `r_multiple` does not separately expose the cost component per trade) — therefore cost sensitivity in this phase is reported as a **disclosed limitation**, not fabricated, and the existing project convention of re-running with 1.5x/2x explicit cost multipliers (used in Phases 33-40 for direct backtests) does not apply here since Phase 44 does not re-simulate trades, it only suppresses/retains historical ones.

## 4. Intervention timing

Every intervention decision is evaluated using the portfolio state **immediately before** a trade's own `entry_time` (identical convention to Phase 43's `open_positions_at`), reusing the exact `open_positions_at()` function already validated in Phase 43. No intervention ever modifies an already-open (accepted) position.

## 5. The five frozen controls (no threshold chosen after seeing results — all thresholds motivated by Phase 42/43's already-published findings, never by Phase 44's own output)

**Control A — NO INTERVENTION.** The historical baseline exactly as recorded.

**Control B — HIGH-volatility exposure reduction.** Using Phase 42's frozen daily volatility-state definition (TRAIN-fixed... actually full-period tercile, per Phase 42 §4/§Part7), on any day classified HIGH, **suppress every 2nd new entry in chronological order within that day** (a deterministic 50% reduction rule — alternating, not random, to keep the rule fully reproducible without inventing a pseudo-random seed). This fraction (50%, alternating) is frozen here, before any Phase 44 result exists, motivated only by wanting a materially large but not total exposure reduction to test against Phase 42's finding.

**Control C — HIGH volatility × high concurrency.** When a new trade's entry-time state is BOTH (a) that day is volatility-state HIGH, AND (b) the open-position count immediately before entry is **≥ 4**, suppress the entry (100% suppression when both conditions hold). The threshold of 4 is taken directly from Phase 43's own evidence (§11, §23 of the Phase 43 master report: the HIGH-vol+4 cell was the single worst cell in Phase 43's position-count table, and the tail-concentration analysis used the 4+ threshold) — not chosen by testing 2/3/4/5/6 in this phase.

**Control D — Volatility-transition exposure control.** On any day classified as a `HIGH_to_NORMAL` transition (Phase 42's frozen state-transition definition — the transition Phase 42 found had the worst average daily R of any transition), suppress 100% of new entries. No other transition window is tested.

**Control E — Exposure-agnostic defensive control.** When the open-position count immediately before a new entry is **≥ 5** (a purely portfolio-state-based signal — independent of volatility, JPY, AMR/ARB, or any specific strategy), suppress the entry (100% suppression). The threshold of 5 is motivated by Phase 42/43's own repeated use of the 5+ concurrency bucket as one of the three predefined thresholds (4+/5+/6+) already established in Phase 42 §Part12 — not chosen here.

## 6. Trade-suppression execution model (frozen, per Part 14)

For each of Controls B-E: iterate the historical trade list in chronological entry order; for each trade, reconstruct portfolio state immediately before its entry (reusing Phase 43's methodology exactly); if the frozen rule is active, **exclude that trade's entire historical R/P&L from the counterfactual portfolio** (labeled COUNTERFACTUAL TRADE SUPPRESSION); if not active, retain the trade exactly as historically recorded. No SL/TP/entry/exit/holding-time is ever modified. No synthetic trade is ever created. Suppressed trades do not change the portfolio state used to evaluate subsequent trades' eligibility (i.e., a suppressed trade is treated as never having existed for the purposes of open-position counting going forward) — this is disclosed as a modeling simplification (§Part 15's limitation).

## 7. Primary evaluation metrics (frozen, per Part 16)

Total R, max drawdown (R), worst daily R, worst 5-day R, worst 10-day R, 95th/99th percentile daily loss, downside deviation, drawdown duration (days), recovery duration (days), profit factor, trade count, % trades suppressed, % gross historical R removed, % historical losses removed, % historical gains removed.

## 8. Trade-off metric (frozen, per Part 17)

For each intervention: % drawdown reduction vs. Control A, % total-R reduction vs. Control A, % trade-count reduction, % worst-tail-loss reduction — reported together, never drawdown-reduction alone.

## 9. Primary success criteria (frozen, per Part 18)

A control is only classified "HISTORICALLY PROMISING" if it clears **all** of: (A) meaningful tail-risk improvement (worst-5%/worst-1% R improves by a materially larger margin than the % of total R sacrificed), (B) limited return sacrifice (total-R reduction does not exceed the drawdown-R reduction in relative terms — i.e., the trade-off ratio favors risk reduction, not merely activity reduction), (C) no catastrophic deterioration in recovery duration, (D) directional consistency across the 3 available historical regime periods, (E) not dependent on removal of only the worst 1-10 days (i.e., the improvement, if any, survives the extreme-day-robustness re-run). Any single failed criterion caps the classification below "HISTORICALLY PROMISING."

## 10. Suppressed-trade attribution (frozen, per Part 25)

For every suppressed trade under each control: strategy, symbol, direction, original R, volatility state, prior concurrency, session — classified historically profitable (R>0) / losing (R<0) / near-zero (|R|<0.05). % of suppressed trades that were historical winners is reported explicitly for every control.

## 11. Regime robustness (identical convention to Phases 41-43)

2019-2020 and 2021-2022 UNKNOWN BY DATA ABSENCE (control starts 2023-08-01); 2023-2024, 2025, 2026 YTD tested where sample permits.

## 12. Extreme-day robustness (identical convention)

Each control's primary trade-off metrics re-run excluding the worst 1/5/10 baseline days.

## 13. Monte Carlo (frozen, per Part 24)

Reuses the exact 10,000-draw trade-order-reshuffle methodology already established in Phases 37-40, applied separately to Control A's full trade set and each intervention's surviving (non-suppressed) trade set — comparing the two reshuffled-drawdown distributions descriptively. Labeled SIMULATED throughout.

## 14. False-positive / in-sample disclosure (frozen, per Part 26)

Every finding in this phase is explicitly labeled **IN-SAMPLE COUNTERFACTUAL EVIDENCE**, never presented as out-of-sample validation or production-ready evidence, per the task's own explicit instruction.

## 15. Multiple-testing policy

Exactly 5 controls (1 baseline + 4 interventions), each fully specified above before any result exists. No additional control, threshold, or fraction may be introduced after this document is committed. Any idea for a further control belongs in `reports/phase44_future_research_ideas.csv` as a FUTURE VALIDATION CANDIDATE for a later phase, never tested here.

## 16. Classification framework (frozen, per Part 30)

Each of Controls B-E receives exactly one of: A. HISTORICALLY PROMISING — REQUIRES OOS VALIDATION / B. MIXED / INSUFFICIENT / C. REJECTED — NO MEANINGFUL BENEFIT / D. REJECTED — EXCESSIVE RETURN SACRIFICE / E. REJECTED — FRAGILE / REGIME-DEPENDENT / F. REJECTED — METHODOLOGICAL LIMITATION. No control may be classified "DEPLOY," "READY," or "PRODUCTION."

---

*No amendment has been made to this document after any Phase 44 result was produced.*
