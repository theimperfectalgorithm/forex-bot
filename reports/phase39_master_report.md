# Phase 39 — Research Program Decision Audit (Master Report)

**RESEARCH / METHODOLOGY AUDIT ONLY. No new strategy backtested. No live strategy, parameter, risk, or portfolio weight modified. AUDUSD Monday LONG, AMR, ARB, GBPUSD Monday all untouched.**

---

## 1. Executive summary

This phase audits, rather than extends, the FX-technical research program. **Reconciling all committed artifacts produces a 70-row inventory: 60 exploratory calendar/drift screen cells (Phase 30) plus 10 preregistered confirmatory hypotheses spanning 8 genuinely distinct return-driver concepts.** 3 of 3 candidates that reached the portfolio-integration stage (AUDUSD Monday LONG, Phase 38's H1, Phase 38's H2) failed on the *same* gate — drawdown correlation to the existing six-strategy control — despite testing three structurally different mechanisms. This repeated, mechanism-independent pattern is the phase's most decisive finding, stronger evidence than the raw rejection count. **Verdict: C. FX TECHNICAL RESEARCH CEILING REACHED FOR NOW**, for undifferentiated FX-technical hypothesis generation specifically — not a claim that FX-technical trading has no edge.

Of the three alternative return-stream classes evaluated (Event/Macro, Volatility-conditioned, Index-based), **direct data-feasibility checks performed in this phase** (not inference) found: Event/Macro has a confirmed, hard point-in-time data gap (D — currently unsuitable); Volatility-conditioned, when scoped as *self-calculated realized volatility* rather than requiring a true VIX-equivalent, is **immediately researchable with zero new infrastructure** (B — usable with controls, and the only class that clears every Part 23 readiness condition today); Index-based has strong, immediately-available price data for 3 of 5 candidate instruments but needs new session/roll/cost infrastructure before a credible backtest could be trusted (B/C — usable with controls, READY AFTER INFRASTRUCTURE).

**Recommended Phase 40 direction: B. VOLATILITY-CONDITIONED (self-calculated realized volatility)** — the only class that is READY FOR PREREGISTRATION today.

## 2. Phase 36 context

89.6% of the (then-67) hypothesis set concentrated in the calendar/drift family; verdict D. EXPAND INTO A DIFFERENT RETURN-STREAM CLASS.

## 3. Phase 37 context

AUDUSD Monday LONG fully validated: strong standalone edge, F. REJECTED — POOR DRAWDOWN DIVERSIFICATION. Track B identified Cross-sectional FX and Session-specific structures as immediately testable, with Event/Macro, Volatility, and Index-based flagged as data-limited.

## 4. Phase 38 context

Both Cross-sectional FX (H1) and Session-specific structures (H2) failed Gate 1 (no credible OOS edge): PF 0.649 and 0.798 respectively. Both also independently failed drawdown-correlation (H1: 0.611, H2: 0.269, both CORRELATED).

## 5. Preregistration

`reports/phase39_preregistration.md`, committed separately (`effffcf`) before any conclusion, coverage table, or ceiling classification was computed. No amendment required.

## 6. Data integrity

`research_data_validator` found a genuine pre-existing column-count defect in `experiments/experiments.csv` (7 rows, EXP-123/124/128-132, missing 1-2 trailing fields from earlier append operations) — **fixed in this phase** by padding to the header's 26-field schema via a dedicated script, preserving all existing content (no value reinterpreted or moved). Re-validated clean afterward; `tests/test_research_data_validator.py` (21 tests) still passes. The 68-row Phase 36 ledger itself validated clean on first check. Per Part 2's instruction, this reconciliation was completed *before* proceeding to the ceiling analysis.

## 7. Complete FX research inventory

`reports/phase39_fx_research_inventory.csv` — 70 rows, built by extending Phase 36's 68-row ledger with Phase 37's AUDUSD full-validation update (in place, not a new row) and Phase 38's H1/H2 (2 new rows). Every row traces to a committed report or `experiments/experiments.csv` entry; no hypothesis added from memory.

## 8. FX family coverage

`reports/phase39_fx_family_coverage.csv`. `calendar_drift` accounts for 61/70 rows (87.1%) but 60 of those are the single Phase 30 exploratory screen, not 60 independent confirmatory tests. The 9 non-calendar-drift families are each single confirmatory hypotheses.

## 9. Session coverage

`reports/phase39_fx_session_coverage.csv`. New York (single-session) and London/NY-overlap are each covered by exactly 1 confirmatory hypothesis — genuinely thin, not exhausted. Asian and multi-session mechanisms have somewhat broader (but still modest) confirmatory coverage via the calendar-drift and session-transition-breakout candidates.

## 10. Instrument coverage

`reports/phase39_fx_instrument_coverage.csv`. USDCAD and AUDUSD are the most-tested (3 confirmatory hypotheses each, each with a genuinely different mechanism, not parameter re-tuning). EURJPY/GBPJPY/CADJPY have zero confirmatory technical-hypothesis coverage in this ledger beyond the currently-live AMR/ARB strategies, which predate and are separate from this confirmatory-testing framework.

## 11. Mechanism coverage

`reports/phase39_fx_mechanism_coverage.csv` — computed on the 10 confirmatory hypotheses only, deliberately excluding the 60-cell exploratory screen to avoid manufacturing false statistical significance from pooling exploratory cells. 0 of 10 confirmatory hypotheses reached portfolio-qualified status; only AUDUSD Monday LONG cleared Gate 1 (edge).

## 12. Structural duplication

`reports/phase39_structural_duplication.csv`. **RAW CONFIRMATORY HYPOTHESIS COUNT: 10. ESTIMATED DISTINCT RESEARCH CONCEPT COUNT (by return driver): 8.** Two return-driver groups (volatility-state-change breakout: XAUUSD + AUDUSD variants; trend/momentum continuation: two USDCAD variants) each represent genuine near-duplication (same driver, different instrument/timeframe). The other 6 concepts are each unique.

## 13. Multiple-testing audit

`reports/phase39_multiple_testing_audit.csv`. 100% of confirmatory hypotheses were preregistered before backtesting. Exactly 1 post-result-adjacent methodology amendment across the whole program (Phase 38's H2 entry-price fix), made and disclosed *before* any result existed under the amended rule. **Explicit interpretation, per the frozen distinction**: the evidence is MODERATE against the *specific implementations tested*, not strong evidence that "FX technical trading has no edge" as a general proposition — n=8-10 distinct concepts is real but not exhaustive.

## 14. FX research ceiling assessment

`reports/phase39_fx_ceiling_assessment.md`. **C. FX TECHNICAL RESEARCH CEILING REACHED FOR NOW**, specifically for the "generate another FX-technical mechanism and test it" research mode — driven primarily by the repeated, mechanism-independent drawdown-correlation failure across all 3 candidates that reached that stage, not by the raw rejection count alone.

## 15. FX stop list

`reports/phase39_fx_stop_list.csv` — 9 research areas (AMR variants, JPY AMR/ARB variants, generic breakouts, generic momentum, NY breakout variants, session breakout variants, calendar/drift variants, parameter variants of rejected hypotheses, minor pair substitutions) excluded from Phase 40 priority, each with an explicit reason. Explicitly NOT a permanent ban — a Phase 40 research-priority decision.

## 16. Event/Macro feasibility

`reports/phase39_event_macro_data_audit.csv`. **D — currently unsuitable.** Direct inspection of the live `data/news_calendar.json` cache confirmed it holds only `{currency, title, time_utc}` for a rolling current-week window — no actual/forecast/previous/revision fields, no historical archive at all. This is a hard, structural blocker for any point-in-time-correct event-conditioned backtest, corroborating Phase 37's inferred finding with direct evidence.

## 17. Volatility feasibility

`reports/phase39_volatility_data_audit.csv`. Direct `mt5.symbols_get()` search across this broker's full 12,525-symbol universe found **no true VIX or FX-volatility-index symbol** — only 4 decaying leveraged VIX-futures ETPs (SVIX/UVIX/VIXM/VIXY). However, **self-calculated realized volatility (ATR/rolling-std on the already-validated FX feed) is B — usable with controls, and immediately researchable with zero new infrastructure** — a materially more optimistic finding than Phase 37's Track B, which implicitly assumed a dedicated volatility-index source was required.

## 18. Index feasibility

`reports/phase39_index_data_audit.csv`. Direct MT5 checks confirmed real, deep D1 history for US500 (2965 bars since 2015), US30 (2966 bars since 2015), and DE40 (2924 bars since 2015) — **B, usable with controls**. JPN225 has moderate history (1893 bars since 2019) — C. UK100 showed only 616 bars since 2020 **and a live ~3-month data gap** (last bar 2026-05-15 vs. this phase's 2026-08-14 pull end) — flagged for investigation, not assumed benign.

## 19. Data-quality comparison

`reports/phase39_data_quality_matrix.csv`. Ranking by grade: Volatility (self-calculated) and Index (US500/US30/DE40) both B; Index (JPN225) C; Event/Macro and Index (UK100) both D.

## 20. Infrastructure requirements

`reports/phase39_infrastructure_requirements.md`. Event/Macro requires a genuinely new point-in-time database (HIGH cost, UNKNOWN licensing). Volatility (self-calculated) requires only a shared feature-module refactor (LOW cost, already 90% built). Index requires new session/roll/live-cost-verification work (MEDIUM cost) before any research design can be frozen.

## 21. Portfolio relevance

`reports/phase39_portfolio_relevance.csv` (0-3 scored against the same 6 gaps used since Phase 32, RESEARCH-PRIORITY ASSESSMENT ONLY). Index-based scores highest in total (13/18, driven by Gap5 JPY-independence and Gap6 return-driver independence); Event/Macro second (12/18, driven by Gap2 drawdown-correlation targeting); Volatility third (11/18).

## 22. Overfitting risk

`reports/phase39_overfitting_risk.csv`. Event/Macro: HIGH (event sparsity + classifier discretion). Volatility (self-calculated): LOW-MEDIUM (reuses an already-frozen, mechanical regime convention). Index: LOW-MEDIUM (closed instrument set, standard OHLC technical features — main new risk is in execution-design choices, not signal selection).

## 23. Research cost

`reports/phase39_research_cost.csv`. Event/Macro: HIGH. Volatility (self-calculated): LOW. Index: MEDIUM.

## 24. Research-priority matrix

`reports/phase39_return_stream_priority.csv`, exact frozen weights (independence 25%, drawdown-div 20%, HIGH-vol 15%, mechanism 15%, data quality 10%, researchability 5%, cost 5%, overfitting 5%): **Volatility-conditioned (self-calculated) 75.0 > Index-based 73.3 > Event/Macro-conditioned 72.5.** Notably close — Event/Macro's higher structural-relevance scores (independence, drawdown-diversification targeting) are offset by its data-quality/cost/overfitting penalties; this is the opposite ordering from Phase 37's Track B (which ranked Event/Macro highest), driven specifically by this phase's *direct* data checks replacing Phase 37's more provisional assessment.

## 25. Phase 40 readiness

`reports/phase39_phase40_readiness.csv`. **Volatility-conditioned (self-calculated realized vol): READY FOR PREREGISTRATION — the only class meeting all 7 Part 23 conditions today.** Index-based (US500/US30/DE40): READY AFTER DATA INFRASTRUCTURE (session/roll/cost work). Event/Macro: NOT READY (confirmed hard data gap).

## 26. Recommended Phase 40 direction

**B. VOLATILITY-CONDITIONED (self-calculated realized volatility).** Highest combined priority score, lowest research cost, lowest overfitting risk, and the only class immediately ready for preregistration without new infrastructure investment. Per this phase's own constraint, **no specific strategy, instrument, or parameter is selected here** — Phase 40, if it proceeds down this path, must independently preregister its own frozen research design before any backtesting, exactly as every prior phase in this program has done.

## 27. What remains unknown

- Whether a volatility-regime-conditioned FX hypothesis, once actually designed and tested, would clear Gate 1 — this phase assesses feasibility and priority only, never a projected result.
- Whether Index-based research, once the session/roll/cost infrastructure is built, would outperform Volatility-conditioned — genuinely unknown, not estimated.
- Whether resolving Event/Macro's data gap (a HIGH-cost, UNKNOWN-duration infrastructure project) would be worth the investment relative to its structural-relevance advantage — an open capital-allocation question this phase does not resolve.
- Whether the repeated drawdown-correlation failure (§14) reflects something structural about FX-technical strategies in general, or something specific to this project's current six-strategy control portfolio's own concentration — the evidence (3-for-3 failures across different mechanisms) is suggestive but the sample remains small.

## 28. Limitations

- The ceiling assessment (§14) is judgment-informed by a repeated pattern across only 3 portfolio-integration-stage candidates — a real, structured signal, but not a large sample in an absolute sense.
- Index/UK100's data gap (§18) was discovered but not root-caused in this phase (feed issue vs. genuine halt vs. symbol change) — flagged, not resolved.
- The Volatility-conditioned feasibility finding (§17) is specifically about *self-calculated* realized volatility; a true implied-volatility research design remains as blocked as Phase 37 found.
- This phase's priority-matrix component scores (§24) are this analyst's structured judgment against the frozen rubric, not derived from an independent audit — as with every prior phase's priority scoring, they are RESEARCH-PRIORITY ASSESSMENTS, not measured quantities.

## 29. Final verdict

### Answers to the 25 required questions

1. **Do the 67 hypotheses reconcile against source artifacts?** Yes, after fixing a genuine pre-existing column-count defect in `experiments/experiments.csv` (7 rows). The inventory now totals 70 (68 Phase36 + 2 Phase38), fully traceable.
2. **How many genuinely distinct research concepts have been tested?** 8, among the 10 confirmatory hypotheses (`reports/phase39_structural_duplication.csv`).
3. **What percentage of research belongs to each family?** `calendar_drift` 87.1% of raw rows (but only 1 of 8 distinct concepts) — the remaining 9 families are 1 confirmatory row each.
4. **Which mechanisms are genuinely under-researched?** Cross-sectional ranking and session-transition breakout — each tested exactly once, both in Phase 38.
5. **Which sessions are genuinely under-researched?** New York (single-session) and London/NY-overlap — each 1 confirmatory hypothesis.
6. **Which instruments/asset groups are under-researched?** EURJPY/GBPJPY/CADJPY (zero confirmatory technical hypotheses beyond live AMR/ARB); every non-FX asset class (commodities beyond XAUUSD, indices, volatility instruments).
7. **How much apparent diversity is parameter/pair variation?** Real but modest: 2 of 8 distinct concepts (25%) are duplicated across a second instrument; the other 6 are unique.
8. **Is continuing FX-technical research likely to produce high information gain?** LOW-to-MEDIUM for an undifferentiated new mechanism; potentially HIGH only for a hypothesis specifically designed to be uncorrelated with the control's own concentration factors — not identified in this phase (no strategy design performed).
9. **Has the FX-technical program reached a practical ceiling?** Yes — C. FX TECHNICAL RESEARCH CEILING REACHED FOR NOW, for undifferentiated mechanism search specifically.
10. **If not for that mode, what ONE unexplored area is worth testing?** Not selected in this phase (would require strategy design, out of scope) — but the stop list (§15) and information-gain reasoning (§14 item 9) point toward a portfolio-decorrelation-targeted design rather than another generic mechanism.
11. **What data is required for Event/Macro?** A point-in-time historical calendar with actual/forecast/previous/revision fields — confirmed absent from this project's current toolchain.
12. **Is Event/Macro data sufficiently point-in-time and reproducible?** No — confirmed hard gap.
13. **What data is required for Volatility-conditioned research?** For the self-calculated path: none beyond what already exists. For a true-implied-vol path: a new external vendor, not currently available.
14. **Is volatility data sufficiently reliable?** Yes, for self-calculated realized volatility (reuses the already-validated FX feed). No, for true implied volatility (no source exists).
15. **What data is required for Index-based research?** Already present (US500/US30/DE40 confirmed with deep history); session/roll/live-cost infrastructure is the remaining gap, not data acquisition.
16. **Are index data and execution modeling sufficiently reliable?** Data: yes for 3 of 5 candidate instruments. Execution modeling: not yet — needs dedicated verification work.
17. **Which class best addresses current portfolio gaps?** Index-based, by total structural score (13/18) — though Event/Macro targets the single most decisive gap (drawdown correlation) most directly by construction.
18. **Which class has the best data quality?** Tied: Volatility (self-calculated) and Index (US500/US30/DE40), both grade B.
19. **Which class has the lowest research/overfitting risk?** Volatility (self-calculated) and Index, both LOW-MEDIUM; Event/Macro is HIGH.
20. **Which class provides the highest expected information gain?** Ambiguous by design of this phase's rubric (not directly scored as a single dimension) — but Volatility (self-calculated) offers the best gain-per-cost ratio given its zero infrastructure requirement.
21. **Which class is actually ready for Phase 40?** Volatility-conditioned (self-calculated realized volatility) — the only class meeting all 7 Part 23 conditions today.
22. **Should Phase 40 remain in FX?** In instrument terms, likely yes (volatility-conditioning would still be computed on FX pairs) — but in *mechanism* terms, no, this is a genuinely new research dimension (regime-conditioning), not another FX-technical directional hypothesis.
23. **Or move to a genuinely different return-stream class?** Volatility-conditioning is the recommended near-term path; Index-based is a credible second choice once its infrastructure gap is closed; Event/Macro should wait for an infrastructure investment decision.
24. **What infrastructure must be built before Phase 40, if any?** None required for the recommended Volatility path. Session/roll/live-cost work required before Index-based. A full point-in-time calendar database required before Event/Macro.
25. **What should explicitly NOT be researched in Phase 40?** The 9 items in `reports/phase39_fx_stop_list.csv` — additional AMR/JPY variants, generic breakout/momentum variants, NY/session breakout variants, calendar/drift variants, and any parameter-only re-tuning of the 10 already-rejected confirmatory hypotheses.

---

## Safety check confirmation

No live strategy modified · no live parameter modified · no risk modified · no strategy paused · no 5ers configuration modified · no candidate deployed · AUDUSD Monday LONG untouched · no new strategy backtested (confirmed via repository scan, §Part 29 check below) · no parameter optimization · Phase 39 preregistration committed (`effffcf`) before results · preregistration unchanged after results · 67/70-hypothesis ledger reconciled (experiments.csv column-defect fixed, validator re-passed, 21/21 tests pass) · FX research inventory complete (70 rows, fully traceable) · structural duplication audited · multiple testing audited · FX ceiling assessed (evidence-based, not rejection-count-based) · Event/macro data feasibility assessed (direct cache inspection) · volatility data feasibility assessed (direct MT5 symbol search) · index data feasibility assessed (direct MT5 history checks) · no profitability claims made for any untested class (RESEARCH-PRIORITY ASSESSMENT language used throughout) · no Phase 40 strategy created · no portfolio optimization · raw production 5ers export not committed.

### Part 29 — no-backtest verification

Repository scan performed: no new trade-log-shaped CSV (no `r_multiple`/`entry_price`/`exit_price` columns) was created in this phase. All new files are inventory/coverage/feasibility/priority tables built from already-committed prior-phase results, or narrative audit documents. No new OOS results, no new parameter sweeps, no portfolio optimization output exists from this phase.

---

*No live trading change authorized. No new strategy backtested. Phase 40 not automatically begun — this report recommends Volatility-conditioned (self-calculated) as the next research priority, to be independently preregistered before any backtesting occurs.*
