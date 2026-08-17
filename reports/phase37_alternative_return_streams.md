# Phase 37 Track B — Alternative Return-Stream Map

**Identification and structured scoring only. No backtesting performed. No entry/exit rule, threshold, or parameter proposed for any class. Extends (does not duplicate) `reports/phase36_alternative_return_streams.md`'s qualitative descriptions with the formal scoring framework frozen in `reports/phase37_preregistration.md` §B1-B4.**

---

## The 10 classes, full detail

`reports/phase37_return_stream_classes.csv` — class name, return driver, underlying market, mechanism, expected correlation/HIGH-vol behavior, session dependency, data quality, complexity, overfitting risk, cost sensitivity, research cost, potential independence, and an explicit reason-to-test / reason-not-to-test pair for each. Every field not backed by prior evidence is HIGH/MEDIUM/LOW/**UNKNOWN** — no fabricated quantitative result appears anywhere in this table.

## Portfolio gap mapping

`reports/phase37_portfolio_gap_mapping.csv` — each class scored 0-3 against the six known portfolio gaps (HIGH-vol weakness, drawdown correlation, mean-reversion concentration, Asian/London session concentration, JPY concentration, lack of a genuinely different driver). **Explicitly labeled RESEARCH-PRIORITY ASSESSMENT ONLY on every row — not a profitability claim, per Part 29's evidence-labeling requirement.** Multi-asset momentum scores highest in total (15 of 18) but carries the highest implementation cost; Event/macro-conditioned systems scores highest specifically on the two heaviest-weighted priority dimensions (drawdown correlation and HIGH-vol compatibility) despite a confirmed data gap.

## Data availability audit

`reports/phase37_data_availability.csv` — historical source, timeframe/bid-ask/spread availability, corporate-action/rollover/contract-change/survivorship concerns, timestamp quality, and an overall readiness verdict per class. **Only two classes are rated READY**: Cross-sectional FX (reuses the exact MT5 feed already validated across Phases 30-37) and, with a minor caveat, Relative-value/spread structures (same data, but methodology needs new stability-testing infrastructure). **Session-specific structures is MOSTLY READY** (price data confirmed, calendar-feed depth not yet audited). **Five classes are confirmed NOT READY** — most critically, Volatility-conditioned systems and Event/macro-conditioned systems, the two classes structurally best-aligned with Phase 32's top two priorities, both lack a confirmed underlying data source in this project's current toolchain.

## Overfitting risk audit

`reports/phase37_overfitting_risk.csv` — degrees of freedom, event-selection risk, instrument/threshold count, event sparsity, discretionary-classification risk, and parameter-tuning-frequency risk per class. **Event/macro-conditioned systems carries the highest overall overfitting risk** (HIGH) — its classifier construction is inherently discretionary and event-sparse. **Multi-asset momentum is second-highest** (MEDIUM-HIGH) purely from combining many instruments and thresholds across asset classes. No class is rejected solely for overfitting risk (per the frozen exclusion criteria) — it is used only for prioritization.

## Weighted priority ranking

`reports/phase37_return_stream_priorities.csv`, using the exact frozen weights (portfolio independence 25%, drawdown diversification 20%, HIGH-vol compatibility 15%, mechanism diversity 15%, data quality 10%, researchability 5%, cost/execution feasibility 5%, overfitting risk 5%):

| Rank | Class | Weighted score |
|---|---|---|
| 1 | Event/macro-conditioned systems | 79.2 |
| 2 | Index-based return streams | 71.7 |
| 3 | Volatility-conditioned systems | 69.2 |
| 4 | Cross-sectional FX | 67.5 |
| 5 | Multi-asset momentum | 66.7 |
| 6 | Commodity-based return streams | 63.3 |
| 7 | Session-specific structures | 63.3 |
| 8 | Relative-value / spread structures | 60.0 |
| 9 | Cross-asset relationships | 58.3 |
| 10 | Other structurally distinct mechanisms | 11.7 |

**This is a RESEARCH-PRIORITY ASSESSMENT, not a profitability forecast** — the top three by theoretical score (Event/macro, Index-based, Volatility-conditioned) are exactly the three classes with a **confirmed data gap** in this project's toolchain (§"Data availability audit"). The score reflects how well each class's *structural characteristics* match the portfolio's known gaps, weighted per the frozen formula — it does not account for whether the class can actually be tested today. **Selection (`§Top three`, below) combines this score with practical data readiness, exactly as Part 23 requires** ("data quality" and "researchability" are explicit selection criteria, not optional context).

---

*No specific strategy, entry rule, or backtest configuration proposed for any class.*
