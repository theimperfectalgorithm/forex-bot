# Phase 48 Preregistration — Six-Strategy Parameter & Cost Robustness Audit

**Frozen before any strategy-specific robustness result is inspected. Committed separately, before any Phase 48 result exists. Not modified after seeing results.**

ROBUSTNESS RESEARCH ONLY. No live strategy code, YAML, execution logic, or risk setting modified. No optimization, no repair, no rescue. Uses the Phase 47 validated reproduction harness as the sole source of truth — verified unchanged (commit `db91189`, SHA-256 hashes re-confirmed identical to Phase 47's own record) before this document was written.

---

## 1. The six frozen strategies

`AUDJPY_AMR`, `CADJPY_AMR`, `EURJPY_AMR`, `GBPJPY_AMR`, `CADJPY_ARB`, `GBPUSD_MONDAY` — unchanged from Phases 46-47.

## 2. Source of truth

The Phase 47 signal-reproduction logic (`src/phase47_reproduction_harness.py`'s `replay_amr`/`replay_arb`/`replay_monday` functions) is extended, not rebuilt, with a **trade-outcome resolver**: a bar-by-bar forward walk from each reconstructed signal's entry to its SL/TP/time-exit, needed because Phase 47 only validated signal timing/direction, not realized R under a changed SL/TP distance (a prerequisite for parameter perturbation, which by definition changes SL/TP distances). This resolver reuses the exact SL/TP/time-exit rules already documented in each strategy's own source (`asian_hours_reversion.py`'s 07:00 UTC time exit; `asian_range_breakout.py`'s Asian-range-derived SL/TP with no explicit time exit documented beyond the same-day breakout window; `monday_drift.py`'s 21:00 UTC Monday time exit) — not invented.

## 3. Historical data and OOS period (unchanged from Phase 46)

MT5 M15/H1/H4, 2023-08-01 to 2026-08-13. TRAIN 2023-08-2024-08, VALIDATION 2024-09-2025-04, **OOS 2025-05-01 to 2026-08-13** — identical to Phase 46's frozen split, not re-chosen.

## 4. Parameter perturbation rules (frozen, per Phase 47's parameter inventory)

One-factor-at-a-time, ±20% on each strategy's continuous parameters only: AMR — `z_threshold`, `sl_multiplier`; ARB — `tp_multiplier`, `min_range_pips`; Monday — `sl_atr_mult`, `tp_atr_mult`. `entry_end_hour`, `h4_filter`, session windows, and `risk_percent` remain non-perturbable (categorical/structural), per Phase 47's frozen classification — not reopened here.

## 5. Cost model and 2x-cost definition

A flat per-trade cost, expressed in pips, subtracted from the raw price move at entry (consistent with this project's project-wide cost convention used since Phase 26/30/37). Baseline cost = 1.0 pip (a conservative, disclosed placeholder given the historical ledger does not expose each trade's actual embedded cost component, per Phase 44/46's own disclosed limitation — carried forward here explicitly, not silently assumed away). 2x cost = 2.0 pips, applied identically to every strategy. This tests the **relative** sensitivity to cost escalation, not an absolute claim about true broker costs, per Phase 47's own disclosed execution-approximation limitation.

## 6. Acceptance criteria (identical framework to Phase 33-40 candidates, reused not reinvented)

Gate 1 OOS edge: PF > 1.0 on ≥30 trades. OOS sub-half: sign-consistent (WARNING tier if total OOS < 40). Parameter perturbation: no sign reversal in expectancy across -20%/baseline/+20% for STABLE; PF/expectancy degradation < 30% relative for MODERATELY SENSITIVE; ≥30% degradation but no sign reversal for HIGHLY SENSITIVE; expectancy sign flip = SIGN REVERSAL. Cost stress: PF remains > 1.0 at 2x cost for COST ROBUST; PF drops but remains > 0.8 for COST SENSITIVE; PF ≤ 0.8 or turns negative for COST FAILURE. Regime/volatility/drawdown-correlation: identical methodology and thresholds already frozen in Phases 41-46 (0.15-point CORRELATED threshold, 8-day minimum overlap, LOW/NORMAL/HIGH terciles reusing the ledger's own `vol_tercile` field joined by pair+date to the freshly-simulated trades).

## 7. Minimum sample requirements

Identical to every prior phase: ≥30 OOS trades for a point estimate to be STATISTICALLY INFORMATIVE; <30 is OBSERVED ONLY; regime/volatility buckets require ≥10 trades or UNKNOWN; drawdown-correlation requires ≥8 overlapping days or UNKNOWN.

## 8. Portfolio-level and leave-one-out methodology

Reuses Phase 46's exact daily-ledger and leave-one-out construction. Each strategy's isolated perturbation is substituted into the six-strategy portfolio one at a time (never combinatorially) to assess portfolio-level sensitivity — per the explicit no-combinatorial-optimization rule.

## 9. Live evidence

Reuses Phase 45/46's already-computed live comparison and sample-sufficiency figures directly — not re-pulled from a new source, per the instruction to use only already-validated live data.

## 10. Final classification framework (frozen, per Parts 29-30)

Each strategy: exactly one of A (ROBUST) / B (PLAUSIBLE BUT INSUFFICIENT EVIDENCE) / C (FRAGILE) / D (WOULD FAIL TODAY'S CANDIDATE BAR) / E (INSUFFICIENT DATA) — a classification must weigh ALL computable gates together; one strong metric never overrides multiple failures, per the explicit anti-cherry-picking rule. Portfolio: one of A/B/C/D.

## 11. No-rescue rule

Any strategy failing any gate is recorded as a failure. No parameter, filter, stop/target, risk weight, or volatility threshold is changed in response, per the explicit prohibition.

---

*No amendment has been made to this document after any Phase 48 result was produced.*
