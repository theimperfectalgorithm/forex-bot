# Phase 36 — Research Direction Decision

**Describes a FAMILY-level direction only, per explicit instruction. No Phase 37 strategy is designed, parameterized, or backtested here.**

---

## The decision

**D. EXPAND INTO A DIFFERENT RETURN-STREAM CLASS**, combined with elements of **A. CONTINUE CURRENT DISCOVERY** (not mutually exclusive — see below), is the recommended direction, in that priority order.

## Why not each alternative, explicitly

- **Not B (change research architecture before continuing)**: `reports/phase36_discovery_validation_audit.md` found the *process* itself (pre-registration discipline, mechanical classification, full disclosure) is sound — the null results reflect genuine tests, not a flawed methodology. There is no evidence the architecture needs to change before more search.
- **Not C alone (pause discovery, deepen portfolio validation)**: the current six-strategy portfolio's live sample (19 post-demotion trades) is already under active monitoring via the existing scorecard framework and the scheduled 2026-08-25 checkpoint (`reports/live_strategy_scorecard.md`) — there is no new evidence in this audit that portfolio validation needs *more* depth beyond what's already running. Pausing discovery entirely would not address the diversification gap Phase 31/32 already established as real.
- **Not E (insufficient evidence, more audit required)**: this phase itself IS the audit Part E would call for, and it produced clear, actionable findings (the taxonomy, the coverage numbers, the regime characterization) — another audit cycle without a new research input would not add information.
- **Why D, specifically, over simply continuing A**: `reports/phase36_search_space_coverage.csv` shows 89.6% of all hypotheses tested to date belong to a single family (calendar_drift, from Phase 30's screen), and every technical-FX mechanism tested confirmatorily (Phase 33/35's 7 candidates) has failed. **RQ7's concern is empirically supported**: the search has been heavily concentrated in simple technical FX rules across a narrow 3.5-year calendar window (§Part 8's regime finding: the entire project's candidate-testing history sits inside REGIME C/D/E only). Continuing to search only within that same space, with the same instrument/mechanism/data limitations, is not obviously the highest-value next step.

## Why not D alone — retain A as a secondary track

`reports/phase36_alternative_return_streams.md` items 4 and 9 (volatility-sensitive systems, macro/event-conditioned systems) — the two directions most directly aligned with Phase 32's top two priorities — both require data this project does not currently have access to. **A pure pivot to D risks stalling on data-access work with no interim research output.** Item 7 (cross-sectional FX) and item 6 (session-specific event structures) are both immediately actionable with existing data and existing project infrastructure (`core/news_calendar.py` for item 6; the already-available multi-pair MT5 feed for item 7) — **these two are the recommended near-term focus**, not a full architectural pivot away from FX.

## Recommended Phase 37 scope (family-level only, per instruction)

1. **Cross-sectional FX momentum/relative-strength** (`phase36_alternative_return_streams.md` item 7) — the single most evidence-backed alternative, directly extending this project's own already-validated CADJPY cross-sectional finding (`PROJECT_REPORT.md` §4, phase 6) to a properly currency-neutral basket construction, rather than the single-pair momentum designs (H2, Phase 33's USDCAD) that have now failed twice.
2. **Session-specific event structures** (item 6) — lowest implementation cost given existing calendar infrastructure, and a genuinely different mechanism (event-conditioned, not generic-session-conditioned) from every FX candidate tested so far.

Both should go through the identical pre-registration discipline established in Phases 33/35 — frozen hypotheses, committed separately before results, mechanical classification, full disclosure regardless of outcome.

**Explicitly NOT recommended for Phase 37**: another single-pair technical breakout/momentum/trend design on the same AUDUSD/USDCAD/USDCHF/XAUUSD universe using the same 2023-2026 data window — Phase 33 and Phase 35 have now tested 7 such candidates confirmatorily with 0 survivors, and `reports/phase36_regime_analysis.csv` shows the specific OOS window used was itself an unusually low-volatility period for the FX majors tested, a further reason not to over-index on more of the same design pattern before diversifying either the mechanism class or (eventually, pending data access) the asset class.

## What would change this recommendation

If Phase 37's cross-sectional-FX or event-structure hypotheses also fail confirmatorily, that would be a meaningfully stronger signal (10 confirmatory hypotheses, 0 survivors, across 4+ distinct mechanism families) than exists today — at that point, a genuine architectural pause (Option C) or a harder pivot requiring new data infrastructure (fully committing to Option D's data-dependent items) would become substantially better-supported than it is now.

---

*Research direction only. No Phase 37 hypothesis designed or backtested. No live trading change authorized.*
