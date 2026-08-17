# Phase 36 Pre-Registration — FROZEN BEFORE SUBSTANTIVE ANALYSIS

**Written and committed before the research ledger, base-rate calculations, or regime analysis are computed. Not changed after seeing results. Any later methodological flaw is documented as a dated amendment in §8, never a silent edit.**

Phase 36 is an audit of the research process itself (Phases 27/29-35), not a new strategy search. No candidate is backtested in this phase.

---

## 1. Research questions (frozen — RQ1-RQ10, verbatim from the task)

RQ1-RQ10 as specified in the task instructions are treated as the fixed question set for this phase; no additional research questions are added after seeing results.

## 2. Datasets used (frozen)

- **Research ledger source**: every committed candidate registry/results CSV from Phase 30 (`reports/non_jpy_candidate_registry.csv`), Phase 33 (`reports/phase33_candidate_registry.csv` + `phase33_candidate_results.csv` + `phase33_robustness_results.csv` + `phase33_final_rankings.csv`), and Phase 35 (`reports/phase35_candidate_registry.csv` + `phase35_candidate_results.csv` + `phase35_parameter_robustness.csv` + `phase35_final_rankings.csv`) — no hypothesis is added to the ledger from memory alone; every row must trace to a committed artifact.
- **Current portfolio reconstruction**: `reports/5ers_trade_export.csv` (the latest committed production export — cutoff explicitly re-verified in §Part 12 of the analysis, not assumed from memory) and `data/phase26_all_trades.csv` (historical frozen-parameter reconstruction, unchanged from Phases 31-35).
- **Broad market-regime characterization**: fresh MT5 D1 price data, pulled explicitly for this phase, for the instruments already used across Phases 30/33/35 (AUDUSD, USDCAD, USDCHF, XAUUSD) plus GBPJPY (a proxy for the live AMR book's own JPY-cross exposure) — **only for characterizing market volatility/trend conditions across calendar periods, not for backtesting any new strategy.**

## 3. Historical regime periods (frozen, adapted to actual data availability — disclosed, not hidden)

The task's suggested periods (2019-2020 / 2021-2022 / 2023-2024 / 2025 / 2026 YTD) are adopted for **market-regime characterization** (D1 price-level volatility/trend statistics, for which MT5 data is confirmed available back to 2015). **They are explicitly NOT used to claim strategy-level backtested evidence outside the window any candidate was actually tested in** — every candidate tested in this project (Phases 30/33/35) was backtested only within 2023-01-01 to 2026-08-14, so this phase's REGIME C/D/E (2023-2024 / 2025 / 2026 YTD) are the only periods for which candidate-level performance evidence exists. REGIME A (2019-2020) and REGIME B (2021-2022) are characterized by market conditions only (a factual description of the environment), with an explicit "NO CANDIDATE-LEVEL EVIDENCE EXISTS FOR THIS PERIOD" label — not filled with invented backtest results.

| Regime | Period | Candidate-level backtest evidence exists? |
|---|---|---|
| A | 2019-01-01 to 2020-12-31 | NO — market-characterization only |
| B | 2021-01-01 to 2022-12-31 | NO — market-characterization only |
| C | 2023-01-01 to 2024-08-31 | YES — this is every candidate's own TRAIN period |
| D | 2024-09-01 to 2025-04-30 | YES — every candidate's own VALIDATION period |
| E | 2025-05-01 to 2026-08-14 | YES — every candidate's own OOS period |

**This mapping is itself a finding of the audit, not an assumption**: the project's entire candidate-testing history (Phase 30 onward) has occurred within a single ~3.5-year window, itself a single macro/rate-cycle regime by most conventional definitions — this phase's own RQ4 ("have we tested enough independent historical regimes") is partly answered by this table alone, before any further calculation.

## 4. Metrics and definitions (frozen, reused from prior phases — not redefined)

- **OOS edge**: OOS PF > 1.0 and positive OOS expectancy (Phase 33/35 convention).
- **OOS consistency**: both OOS sub-halves same sign (Phase 33/35 convention); WARNING if total OOS trades < 40 (Phase 34's explicit recommendation).
- **Parameter robustness**: no sign reversal and <50% expectancy-magnitude degradation across ±20% perturbation (Phase 33/35 convention).
- **Cost robustness**: OOS PF > 1.0 at 1.5x cost (Phase 33 convention).
- **HIGH-vol classification**: STRONG/NEUTRAL/WEAK/UNKNOWN with a 10-trade floor for the HIGH tercile (Phase 34's precondition).
- **Drawdown-correlation classification**: STRONG DIVERSIFIER/NEUTRAL/CORRELATED/UNKNOWN with an 8-overlapping-day floor (Phase 35's precondition).
- **Current six-strategy portfolio**: GBPJPY/EURJPY/AUDJPY/CADJPY AMR, CADJPY ARB, GBPUSD Monday Drift, date-floored at the 2026-07-31 demotion for any "post-demotion" statistic — exactly the correction established in `reports/5ers_portfolio_update_aug13.md` and reused unchanged since.

## 5. Statistical methods (frozen)

- **Confidence intervals**: Wilson score interval for proportions (appropriate for small-n binomial rates, e.g. "X of Y hypotheses passed"), reported alongside every base-rate percentage in §Part 5/14 — not point estimates alone.
- **"Observed research-set frequency," never "population probability"** — per explicit instruction, every base-rate figure in this phase is labeled as a description of this specific tested sample, not an estimate of the true underlying probability any future hypothesis will succeed.
- **No shuffling of any historical time series in this phase** — Phase 36 does not run any new Monte Carlo; it audits already-completed results.

## 6. Treatment of small samples (frozen)

Any category (e.g. HIGH-vol tercile, drawdown-day overlap) with fewer trades/days than its own pre-established floor (§4) is reported **UNKNOWN**, never estimated. Any base rate computed from fewer than 5 underlying hypotheses is explicitly flagged **"too small to support a rate-based conclusion — reported as a raw count only."**

## 7. Treatment of failed hypotheses (frozen)

Every hypothesis in the ledger (§2) is retained regardless of outcome — none is dropped, hidden, or excluded from a denominator because it failed. Rejected hypotheses are never re-tested or re-parameterized in this phase (this phase runs no new backtests at all).

## 8. Amendments

None required as of this commit.

---

*Frozen at the time of this commit. No ledger, base-rate, or regime calculation has been performed yet.*
