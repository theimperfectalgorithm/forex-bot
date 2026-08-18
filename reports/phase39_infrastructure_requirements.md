# Phase 39 — Infrastructure Requirements (requirements document only, nothing implemented)

RESEARCH ONLY. No infrastructure is built in this phase — this is a requirements specification to inform a Phase 40 investment decision.

## Event/Macro-conditioned

- **Point-in-time economic-calendar database**: a historical archive of releases with actual/forecast/previous fields AND revision history, distinguishing what was known *at release time* from later-revised values. This is the single hardest requirement — the current `core/news_calendar.py` integration (Forex Factory current-week JSON) provides none of this.
- **Vendor evaluation**: identify and cost at least one licensed provider (e.g., a TradingEconomics/Econoday/FRED-vintage-style API) — not evaluated in this phase; genuinely UNKNOWN cost/terms.
- **Revision handling**: a data model that stores each release's as-first-published value separately from any subsequent revision, and a query interface that returns only the point-in-time-correct value for a given historical backtest date.
- **Timezone normalization**: consistent UTC event timestamps, reconciled with this project's existing server-time convention (`server_utc_offset_hours()` in `src/agents/agent_strategy.py`).
- **Event-importance/currency tagging**: reliable country/currency/impact-tier fields, not currently persisted even in the current-week cache.

## Volatility-conditioned

**Two distinct paths, per this phase's own feasibility finding (`reports/phase39_volatility_data_audit.csv`):**

- **Path A — self-calculated realized volatility (LOW infrastructure cost)**: essentially already built. Extends the existing ATR/rolling-std machinery already used in Phases 19/31/32/36-38. Would need: a shared, reusable "volatility regime" feature-computation module (currently duplicated ad hoc per phase script) and a documented, frozen regime-bucketing convention (terciles vs. a fixed threshold) to avoid re-deriving it per hypothesis.
- **Path B — true implied volatility / dedicated FX-vol index (currently blocked)**: would require a new external data source; no true VIX-equivalent symbol exists on the current broker feed (confirmed this phase). Not recommended as a near-term investment given Path A's much lower cost and confirmed availability.

## Index-based

- **Roll/rollover handling**: needed if any index CFD tested turns out to be a dated-contract product rather than a continuous synthetic index — not yet confirmed either way for US500/US30/DE40/JPN225/UK100 (flagged UNKNOWN in `reports/phase39_index_data_audit.csv`).
- **Session normalization**: equity-index trading hours are exchange-specific and narrower than 24-hour FX; a dedicated session-boundary characterization (analogous to this project's own prior "server-time fix" for FX) would be needed before any session-conditioned index hypothesis could be frozen.
- **Cost model**: the demo feed showed 0 spread for every index CFD checked — almost certainly a demo-account artifact, not a genuine execution assumption; live spread/commission data must be obtained (from a live account statement or broker spec sheet) before any cost-stress test could be trusted.
- **Corporate-action handling**: dividend-adjustment effects on index CFDs were flagged as a concern in Phase37's Track B audit and were not re-verified with live data in this phase — needs a dedicated check.
- **Data-continuity investigation**: UK100 showed a ~3-month gap between its last available bar (2026-05-15) and this phase's pull end date (2026-08-14) — must be investigated (feed issue vs. genuine trading halt vs. symbol renamed) before UK100 is used for anything.

## Cross-cutting

- None of the three classes requires a new backtesting engine, execution simulator, or portfolio-integration methodology — the existing Phase 37/38 battery (edge/OOS-consistency/robustness/cost-stress/regime/drawdown-correlation/portfolio-integration/Monte Carlo) applies unchanged once the underlying data gap for a given class is resolved.
