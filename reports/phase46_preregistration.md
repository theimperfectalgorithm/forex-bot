# Phase 46 Preregistration — Current Six-Strategy Robustness Audit

**Frozen before any strategy-specific robustness result is inspected. Committed separately, before any Phase 46 result exists. Not modified after seeing results.**

HISTORICAL ROBUSTNESS / LIVE-VALIDATION RESEARCH ONLY. No strategy code, parameter, indicator, entry, exit, SL/TP, position sizing, or risk setting modified. No pause, removal, filter, or optimization. No repair of a failing strategy. All output analytical.

---

## 1. The six-strategy audit population (frozen, verified against the repository, per Part 2)

Verified via `data/phase26_all_trades.csv`'s own `strategy` column (unchanged since Phase 31): **`AUDJPY_AMR`, `CADJPY_AMR`, `EURJPY_AMR`, `GBPJPY_AMR`, `CADJPY_ARB`, `GBPUSD_MONDAY`.** No candidate strategy from Phases 33-40 is added. Live-feed naming (`GBPUSD_MON` in `reports/5ers_trade_export.csv`) is normalized to `GBPUSD_MONDAY` for comparison — confirmed the same strategy by cross-referencing `strategy_reason`/`symbol` fields, not assumed.

## 2. Historical data source and period (unchanged from Phases 41-45)

`data/phase26_all_trades.csv`, 2,712 trades, 2023-08-01 to 2026-08-13 — the same reconstruction used as the control throughout Phases 31-45. Live production data: the freshest local, uncommitted `reports/5ers_trade_export.csv` (as used in Phase 45).

## 3. Strategy source-code freeze audit (per Part 7)

The live strategy source (`strategies/asian_hours_reversion.py` = AMR, `strategies/asian_range_breakout.py` = ARB, `strategies/monday_drift.py` = GBPUSD Monday) is inspected directly and documented in `reports/phase46_strategy_definitions.csv` — entry/exit/SL/TP/session/parameters as written in the actual committed source, not inferred. Each source file's own docstring documents its original internal validation (AMR: `src/phase3_session_structure_search.py`/`phase3b_amr_jpy_refine.py`; Monday: `src/phase8_monday_validation.py`) — these are cited as historical record, not re-derived, and their evidence format (informal parameter grids, not the frozen ±20% single-perturbation standard used since Phase 33) is explicitly disclosed as methodologically different from — not equivalent to — the current framework.

## 4. Scope limitation, disclosed before any result exists (per Part 6's explicit "classify as INSUFFICIENT DATA rather than substitute" instruction)

**Parameter perturbation (Part 12) and cost-stress re-simulation (Part 14) as literally specified — re-running the actual strategy source against MT5 price history with a ±20% parameter change or an explicit cost multiplier — are classified `INSUFFICIENT DATA / REQUIRES NEW RE-EXECUTION INFRASTRUCTURE` in this phase.** The committed historical ledger (`data/phase26_all_trades.csv`) contains only the strategies' *already-executed* trade outcomes (entry/exit prices, R-multiples, SL/TP pip distances) — it does not retain a re-runnable backtest engine bound to live price history with adjustable parameters, unlike the Phase 33-40 candidates, which were built with exactly that re-executable design from the start. Re-implementing and validating such an engine for 4 AMR pairs + 1 ARB + 1 calendar-drift strategy, against live MT5 price pulls, is a materially larger engineering undertaking than the rest of this audit and is **not attempted in this pass**, per the explicit instruction not to fabricate a result the data cannot support. This is recorded as the single most concrete, actionable gap for a future phase (§Future validation), not silently omitted.

**What IS fully computable from the existing, already-validated ledger and infrastructure** (and therefore performed in this phase): OOS edge, OOS sub-half consistency, historical regime robustness, volatility-state behavior (reusing the ledger's own `vol_tercile` field, Phase 42's convention), drawdown correlation (reusing Phase 31/41's methodology), portfolio integration / leave-one-out (reusing Phase 41's daily-ledger construction), strategy contribution, Monte Carlo (reusing the Phase 37-44 trade-order-reshuffle methodology), live comparison and live-sample-sufficiency bootstrap (reusing Phase 45's methodology exactly), candidate comparison against the Phase 33-40 evidence bar, and survivorship-bias assessment.

## 5. TRAIN/VALIDATION/OOS split (frozen, per Part 8)

Reuses the exact split already established and used consistently since Phase 35 for this control: **TRAIN = 2023-08-01 to 2024-08-31, VALIDATION = 2024-09-01 to 2025-04-30, OOS = 2025-05-01 to 2026-08-13** (end-date extended 1 day beyond Phase 35's 2026-08-14 to match this control's actual last trade date, a data-availability fact, not a chosen convenience). Chronological only, no random split.

## 6. OOS sub-half / regime / volatility / drawdown-correlation / portfolio-integration / Monte Carlo methodology

Identical, unmodified conventions already frozen and used in Phases 33-45 (sign-consistency rule with the n<40 WARNING tier; five historical regime periods where the 2023-08-2026-08 window permits — 2019-2022 UNKNOWN BY DATA ABSENCE, unchanged from every prior phase; `vol_tercile`-based LOW/NORMAL/HIGH; OOS-window-matched drawdown correlation with the 8-day minimum-overlap floor and 0.15-point CORRELATED threshold; leave-one-out portfolio integration; 10,000-draw trade-order-reshuffle Monte Carlo).

## 7. Live evidence methodology (identical to Phase 45)

A = full historical. B = pre-demotion. C = post-demotion live (current-6 membership only, CLOSED trades, the freshest local export) — never pooled. Live-sample-sufficiency block bootstrap reuses Phase 45's exact methodology per strategy (block size = that strategy's own live trade count, 10,000 draws from that strategy's own historical trade sequence).

## 8. Candidate-comparison methodology (per Part 24)

Each of the six strategies' OOS PF, OOS consistency, regime robustness, volatility behavior, and drawdown correlation are placed alongside the Phase 33-40 candidates' own results (reusing `reports/phase45_research_master_ledger.csv`) in one comparison table — an apples-to-apples read of the *same gates*, with the disclosed exception that parameter/cost robustness cannot be directly compared (§4) and is marked as such in every row, not silently left blank.

## 9. Survivorship-bias framing (per Part 25)

Explicitly stated before any result is seen: these six strategies were not selected via this project's Phase 33+ competitive gate — they predate it and were individually validated (per their own docstrings) via an earlier, less formal process, then later became "the current-6" by virtue of being the strategies live at the time this project's more rigorous framework was built. This is a real, disclosed survivorship consideration, not resolved by this phase, only acknowledged.

## 10. Classification framework (frozen, per Parts 27-28)

Each strategy: exactly one of A (ROBUST — CONTINUE LIVE VALIDATION) / B (PLAUSIBLE — EVIDENCE STILL INSUFFICIENT) / C (FRAGILE — ROBUSTNESS CONCERNS) / D (RESEARCH-LEVEL FAILURE — WOULD NOT PASS CURRENT CANDIDATE BAR) / E (INSUFFICIENT DATA). Given §4's disclosed scope limitation, **no strategy can be classified A outright** without the missing parameter/cost-robustness evidence — the ceiling classification for any strategy passing every *computable* gate is **B (PLAUSIBLE — EVIDENCE STILL INSUFFICIENT)**, explicitly reflecting the real, disclosed gap rather than silently assuming robustness that was never tested. Portfolio: one of A/B/C/D per Part 28, same principle applied at the aggregate level.

## 11. No-repair rule (frozen, per Part 33)

Any strategy that appears to fail a gate is recorded as a **FUTURE STRATEGY REVIEW CANDIDATE**, never modified, re-parameterized, or filtered in this phase.

---

*No amendment has been made to this document after any Phase 46 result was produced.*
