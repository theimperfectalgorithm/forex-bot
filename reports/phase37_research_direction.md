# Phase 37 — Research Direction Decision

**Combines Track A's AUDUSD verdict with Track B's return-stream map. No Phase 38 strategy is designed, parameterized, or backtested here — family/class level only, per instruction.**

---

## Track A implication for the direction decision

AUDUSD Monday LONG — the project's single strongest candidate across every phase to date — **failed the standardized validation battery specifically on drawdown diversification** (Track A §F, `reports/phase37_audusd_drawdown_correlation.csv`: 0.742 correlation on the control's worst days vs. 0.228 on normal days), despite passing every other gate cleanly (exact reproduction, OOS-consistent, parameter-stable, cost-robust to 2x, positive across all five characterized historical regimes back to 2019). **This is a meaningful, negative data point for the direction decision**: the project's best FX-technical candidate, when finally held to the same rigor as every other candidate, still does not solve Phase 32's #2 priority. This strengthens (does not create, but reinforces) the case that continuing to search the same technical-FX space is unlikely to find something Track A's AUDUSD candidate itself couldn't deliver.

## Track B implication

`reports/phase37_return_stream_priorities.csv` shows the classes structurally best-aligned with the portfolio's actual gaps (Event/macro-conditioned, Volatility-conditioned) are also the ones this project currently lacks the data to test. The two classes that are both well-scoring AND immediately testable (Cross-sectional FX, Session-specific structures) were already identified in Phase 36 — this phase's more rigorous scoring **confirms**, rather than changes, that recommendation.

## The decision

**C. Expand into different return-stream classes**, informed by (not replacing) continued disciplined use of the existing architecture — consistent with, and now reinforced by, Phase 36's D verdict. **Not B alone** (validating AUDUSD was necessary and is now complete, not an ongoing direction). **Not A** (continuing to search only technical FX families is weakened further by Track A's result). **Not D** (pause) — there is a clear, evidence-supported next step, not an absence of direction.

### What the evidence supports vs. what remains unknown

**Supported by evidence:**
- AUDUSD Monday LONG does not currently meet the bar for demo-forward-test eligibility (Track A, mechanically applied classification).
- The technical-FX search space has been extensively covered (8 confirmatory candidates now tested across Phases 33/35/37, 0 reaching portfolio-qualified) relative to its own family diversity.
- Cross-sectional FX and session-specific event structures are the two return-stream classes that are simultaneously well-scoring on portfolio relevance AND immediately testable with existing data.

**Remains unknown:**
- Whether cross-sectional FX or session-specific structures will themselves survive a standardized validation battery — this phase does not test them, only scopes them.
- Whether the two highest-theoretical-priority classes (Event/macro-conditioned, Volatility-conditioned) would outperform if their data gaps were resolved — genuinely unknown, not estimated.
- Whether AUDUSD Monday LONG's drawdown-correlation failure is a persistent structural property or partly an artifact of the specific control-window overlap used (9 drawdown-day observations is the minimum viable sample under the frozen preregistration, not a large one) — flagged as a limitation, not resolved here.

## Recommended Phase 38 scope (class-level only)

1. **Cross-sectional FX momentum/relative-strength** — directly extends the project's own validated CADJPY cross-sectional finding, immediately testable, data confirmed READY.
2. **Session-specific event structures** — lowest implementation cost given existing `core/news_calendar.py` infrastructure, data MOSTLY READY pending a quick calendar-feed depth audit.
3. **(Longer-horizon, not immediate) Event/macro-conditioned systems** — the highest theoretical priority score, explicitly recommended as a *data-infrastructure* project first (building and validating a simple risk-on/risk-off classifier), not a backtest to start next. Listed as a third priority precisely because Part 23 requires exactly this kind of explicit "why test / why not test" reasoning, not because it is ready for a Phase 38 pre-registration today.

**Explicitly not recommended**: another single-instrument technical breakout/momentum/trend design on the same FX-majors/XAUUSD universe — this space has now had 8 confirmatory tests (7 from Phase 33/35 plus AUDUSD Monday LONG's fresh validation) with 0 reaching portfolio-qualified status.

---

*Research direction only. No Phase 38 hypothesis designed or backtested. No live trading change authorized.*
