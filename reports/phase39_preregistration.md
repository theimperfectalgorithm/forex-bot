# Phase 39 Preregistration — Research Program Decision Audit

**Frozen before substantive analysis. Committed separately, before any Phase 39 conclusion, coverage table, or ceiling classification exists. Not modified after seeing results.**

RESEARCH / METHODOLOGY AUDIT ONLY. No new strategy is backtested, no parameter is optimized, no live code/parameter/risk/portfolio-weight is modified. This phase produces a decision, not a strategy.

---

## A. FX research taxonomy methodology

Source of truth: `reports/phase36_research_ledger.csv` (68 rows, itself built entirely from committed artifacts — Phase 30's 60-cell calendar/drift screen + Phase 33's 2 confirmatory candidates + Phase 35's 5 confirmatory candidates + AUDUSD Monday LONG), **extended** with Phase 37's AUDUSD full-validation update (same hypothesis, richer result — not a new row) and Phase 38's H1/H2 (2 new rows) to form `reports/phase39_fx_research_inventory.csv` (70 rows). No hypothesis is added from memory; every row must trace to a committed report, CSV, or `experiments/experiments.csv` entry.

## B. 67/70-hypothesis classification methodology

Each row classified by strategy_family, mechanism, session, instrument, timeframe, exactly as already schematized in the Phase 36 ledger's columns. Ambiguous cases → `UNKNOWN`, never forced.

## C. Definition of "tested territory"

A (mechanism × session × instrument-class) combination is TESTED TERRITORY if at least one preregistered confirmatory hypothesis (not merely an exploratory screen cell) was run against it and reached at least a Gate-1 (OOS edge) determination.

## D. Definition of "genuinely unexplored territory"

A combination is GENUINELY UNEXPLORED if no confirmatory hypothesis in the inventory shares its mechanism AND its session AND its broad instrument class (FX-major/FX-cross/commodity/index/other). Sharing only one or two of these three dimensions does not make it explored — but does inform the "meaningfully related" (Part 8, category B) classification.

## E. Definition of "materially distinct"

A hypothesis is materially distinct from prior work only if its **return driver** (the economic mechanism claimed to produce the edge) differs, not merely its parameter values, instrument, or session. Two hypotheses sharing the same return driver (e.g., "range-fade of Asian extremes") with different pairs or thresholds are NOT materially distinct — they are variants. This mirrors the frozen rule already applied in Phase 38's structural-independence gate.

## F. FX continuation decision criteria

Per Part 10 of the task instructions, exactly four allowed conclusions: A. CONTINUE, B. CONTINUE NARROW, C. CEILING REACHED FOR NOW, D. INSUFFICIENT EVIDENCE. Decision is evidence-based on: research coverage breadth (§Part 4-7), structural diversity (§Part 8), robustness/portfolio outcomes (already in the ledger), and expected information gain of remaining unexplored cells (§Part 11) — never on raw rejection count alone.

## G. Return-stream classes to evaluate (frozen, exactly 3, per Part 13)

Event/Macro-conditioned, Volatility-conditioned, Index-based. No other class is evaluated in this phase (Cross-sectional FX and Session-structure were tested in Phase 38; Cross-asset/Commodity/Relative-value/Multi-asset-momentum/Other were deprioritized in Phase 37 and are not re-opened here).

## H-M. Feasibility/quality/researchability/overfitting/portfolio-relevance criteria

Identical rubric to Phase 37's Track B (HIGH/MEDIUM/LOW/UNKNOWN throughout, A-D data quality grades per Part 17, 0-3 portfolio-relevance scoring per Part 19 against the same six gaps used since Phase 32). No fabricated quantitative figures; `UNKNOWN` used wherever a claim cannot be verified from an actual data check performed in this phase (e.g., an actual `mt5.symbols_get()` query, not an assumption).

## N. Class-priority methodology

Exact weights specified in Part 22: Portfolio independence 25%, Drawdown diversification 20%, HIGH-vol relevance 15%, Mechanism diversity 15%, Data quality 10%, Researchability 5%, Execution feasibility 5%, Overfitting risk 5%. Applied identically to the 3 classes in scope. This is a RESEARCH-PRIORITY score, never a profitability estimate — no PF/return/Sharpe figure is used as an input.

## O. Phase 40 recommendation methodology

A class is READY FOR PREREGISTRATION (Part 23) only if it meets all seven listed conditions (adequate data, credible point-in-time integrity, sufficient historical coverage, modelable execution, freezable research design, structural distinctness, manageable overfitting risk). Otherwise READY AFTER INFRASTRUCTURE, NOT READY, or UNKNOWN. The Phase 40 direction (Part 24) is selected from the five allowed outcomes (A/B/C/D/E) using information gain, data quality, portfolio relevance, structural independence, research cost, and overfitting risk — never a fabricated backtest.

---

## Evidence labeling (frozen, per Part 28)

Every conclusion in this phase's deliverables is labeled OBSERVED / CALCULATED / RESEARCH-PRIORITY ASSESSMENT / DATA-FEASIBILITY ASSESSMENT / UNKNOWN. **SIMULATED is never used in this phase** — no strategy simulation of any kind is performed.

## No-backtest constraint (frozen, per Part 29)

This phase performs zero new strategy backtests, zero parameter searches, zero new OOS trade generation. Any MT5 data queries in this phase are limited to symbol/history existence and metadata checks (`symbols_get`, `symbol_info`, bar-count/date-range checks) — never a trade-generating backtest loop.

---

*No amendment has been made to this document after any Phase 39 conclusion was produced.*
