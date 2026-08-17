# Phase 36 — Research Base-Rate & Portfolio Viability Audit (Master Report)

**Audit only. No new candidate backtested. No live strategy, parameter, risk, or portfolio weight modified. No candidate promoted, rescued, or optimized. AUDUSD Monday LONG untouched.**

---

## 1. Executive summary

Across the project's full candidate-testing history (Phase 30 onward), **89.6% of all tested hypotheses belong to a single strategy family** (calendar/drift, from Phase 30's exploratory screen), and **every one of the 7 confirmatory, pre-registered candidates tested since (Phase 33's 2 + Phase 35's 5) has been rejected** — 2 for robustness failure after showing an initial edge (Phase 33), 5 for showing no edge at all (Phase 35). The entire confirmatory testing history sits inside a single ~3.5-year calendar window (2023-2026), and the specific 15.5-month OOS window used throughout was itself, per this phase's own fresh market-data analysis, an unusually **low**-volatility period for the FX majors tested — a plausible structural headwind for breakout/momentum designs specifically. **The research process itself is sound** (pre-registration discipline verified intact via git history); **the search space has been narrow** (RQ7 confirmed). The current six-strategy portfolio remains **DEFENSIBLE WITH MONITORING** — its live sample (19 post-demotion trades) is too small to draw a confident verdict either way, and its historical evidence base (2,712 trades, PF 1.211) remains intact and unchanged. **Recommended direction: D (expand into a different return-stream class), prioritizing two immediately actionable, evidence-backed families (cross-sectional FX momentum, session-specific event structures) over a further architectural pivot or a full pause.**

---

## 2. Phase 35 context

Phase 35 tested 5 confirmatory candidates addressing the New York session and mechanism gaps Phase 34 identified; all 5 failed Gate 1 (no credible OOS edge), a cleaner and more decisive rejection than Phase 33's near-misses. This phase (36) was commissioned specifically because that outcome changed the research question from "which strategy" to "should we keep searching this way."

## 3. Research questions

RQ1-RQ10 as specified in `reports/phase36_preregistration.md` §1 — answered throughout this report and consolidated in §25 (Final Verdict).

## 4. Preregistration

`reports/phase36_preregistration.md`, committed separately (`b5c3769`) before any ledger, base-rate, or regime calculation was performed. No amendment was required.

## 5. Data integrity

`src/research_data_validator.py` passed on both the consolidated research ledger and the live production export (`reports/5ers_trade_export.csv`) before any analysis proceeded. No malformed rows, no column-shift, no lifecycle-pairing failure. The production export's exact cutoff was independently re-verified (not assumed from memory): **latest entry 2026-08-13 05:00:05 UTC, latest exit 2026-08-13 19:12:09 UTC** — identical to the cutoff already established in `reports/5ers_portfolio_update_aug13.md`, confirming no newer production data has been fabricated or assumed for this phase.

## 6. Research ledger

`reports/phase36_research_ledger.csv` — **68 rows**, consolidated entirely from already-committed artifacts (Phase 30's 60-cell registry, Phase 33's 2-candidate registry+results+robustness+rankings, Phase 35's 5-candidate registry+results+rankings, plus one summary row for the standing AUDUSD Monday LONG candidate) — no hypothesis added from memory. Every failed hypothesis is retained; none excluded from any denominator below.

## 7. Research base rate ("observed research-set frequencies," never population probabilities — per the frozen preregistration)

| Population | n | Metric | Count | Observed rate | Wilson 95% CI |
|---|---|---|---|---|---|
| Confirmatory candidates (Phase 33+35) | 7 | Initial OOS edge (PF>1.0) | 2 | 28.6% | 8.2–64.1% |
| Confirmatory candidates | 7 | OOS sub-half consistency (of the 2 with an edge) | 0 | 0.0% | 0–65.8% |
| Confirmatory candidates | 7 | Parameter robustness pass | 0 | 0.0% | 0–35.4% |
| Confirmatory candidates | 7 | Portfolio-qualified (Cat. H/I) | 0 | 0.0% | 0–35.4% |
| Exploratory screen cells (Phase 30) | 60 | Cleared pre-registered screening bar | 2 | 3.3% | 0.9–11.4% |
| ALL hypotheses (screen + confirmatory) | 67 | Reached PROMISING or better | 3 | 4.5% | — |

**These confidence intervals are wide** (e.g. 8.2–64.1% for the confirmatory edge rate) — explicitly because n=7 is a small sample, exactly the caveat the frozen preregistration required be stated, not glossed over. **The confirmatory-candidate sample is too small to support a precise rate-based conclusion on its own; the screen-stage sample (n=60) supports a tighter estimate but describes a different, less rigorous testing stage (single IS/OOS split, no separate validation fold).**

## 8. Failure taxonomy

`reports/phase36_failure_taxonomy.csv`:

| Failure category | Count | % of 67 rejected/pending hypotheses |
|---|---|---|
| OOS_INSTABILITY (robustness-type failure) | 38 | 56.7% |
| EDGE_ABSENT (no credible OOS edge at all) | 25 | 37.3% |
| PARAMETER_FRAGILITY | 2 | 3.0% |
| PROMISING_NOT_REJECTED | 2 | 3.0% |

**"What is actually killing our candidates?"** — predominantly OOS/robustness instability (56.7%), not a simple absence of any signal (37.3%). But this blends two different testing rigors: **within the confirmatory stage specifically**, Phase 33's 2 candidates were both robustness/parameter-fragility failures (they had a real initial edge), while **all 5 of Phase 35's candidates were edge-absent** — a genuinely different failure shape between the two confirmatory rounds, not a single uniform pattern. Both are real, distinct findings, not conflated here.

## 9. Discovery vs. validation

Full detail: `reports/phase36_discovery_validation_audit.md`. Finding: **the process is sound** — pre-registration is genuinely frozen before results (verified via git log showing exactly one commit per preregistration file), no candidate has been rescued or silently re-parameterized, and every tested hypothesis is disclosed regardless of outcome. The one real caveat: candidates share substantial calendar/instrument overlap, so the "8 of 16 families" tally is not 8 fully independent draws from market history.

## 10. Historical regimes

`reports/phase36_regime_analysis.csv` — real MT5 D1 price data, 2019-2026, for USDCAD/AUDUSD/USDCHF/XAUUSD/GBPJPY. **Key finding: the actual OOS window used throughout this project's confirmatory testing (2025-05 to 2026-08) was a comparatively LOW-volatility period for every FX major tested** (e.g. USDCAD daily volatility 0.282%, the lowest of all five characterized periods; AUDUSD 0.507%, also the lowest) **— while XAUUSD's same window was the HIGHEST-volatility period of the five characterized** (1.612% daily vol vs. 0.87-1.06% in earlier periods). This is a plausible structural headwind specifically for the FX-pair breakout/momentum designs (H1-H4, Phase 33's USDCAD) tested in this window, and a partial explanation (not the whole story — see §16) for why XAUUSD's HIGH-vol classification looked strong in isolation despite ultimately failing on drawdown correlation. **REGIME A (2019-2020) and REGIME B (2021-2022) are characterized by market conditions only — no candidate-level backtested evidence exists for either period, disclosed explicitly rather than estimated.**

## 11. Family-level regime analysis

`reports/phase36_family_regime_analysis.csv`. Of 8 tested families: calendar_drift shows a mixed result (38 rejected, 20 no-edge, 2 reaching PROMISING) — genuinely "partial signal found," not uniformly negative. The other 7 families (each tested with exactly 1 hypothesis) show **"no tested implementation met the evidence bar"** — explicitly NOT phrased as "the family is useless," per the frozen interpretation rule (Part 25 of the task instructions), since a single implementation's failure does not establish the family itself doesn't work.

## 12. Current portfolio reconstruction

| Population | Date range | Trades | Win rate | PF | Expectancy R | Total R | Max DD (R) |
|---|---|---|---|---|---|---|---|
| Historical frozen-parameter reconstruction | 2023-08-01 to 2026-08-13 | 2,712 | 67.0% | 1.211 | +0.072 | +194.11 | −29.53 |
| **Live production, post-demotion current-six** | 2026-08-02 to 2026-08-13 | **19** | 36.8% | **0.292** | **−0.227** | −4.32 | −3.72 |

Correctly date-scoped at the 2026-07-31 demotion boundary (per the established correction from `reports/5ers_portfolio_update_aug13.md`) — no pre-demotion trades included. **These two figures describe genuinely different things and are not blended into a single number**: the historical reconstruction is the accumulated frozen-parameter evidence base (large sample, positive); the live figure is the actual current account's very recent, very small sample (negative point estimate, statistically uninformative on its own).

## 13. Portfolio viability

`reports/phase36_portfolio_viability.csv`. Per the explicit reframing (not "is it profitable enough" but "does evidence justify continuing unchanged while research continues"):

| Criterion | Assessment |
|---|---|
| Historical edge | SUPPORTIVE (PF 1.211, 2,712 trades) |
| Live edge | Negative point estimate, but statistically uninformative (n=19, well below any confident threshold) |
| Concentration | Pre-existing concern, already monitored (effective N 2.67/6, Phase 31/32, unchanged) |
| Sample size | Insufficient for a confident live verdict either way |
| Known pre-existing weaknesses | AUDJPY/CADJPY AMR HIGH-vol weakness — known since Phase 20/21, under active monitoring via the pre-existing 2026-08-25 checkpoint |

**Classification: B. DEFENSIBLE WITH MONITORING.** Not A (defensible without qualification) — the live sample's negative point estimate and the known AMR weaknesses warrant continued active monitoring, not silence. Not C (insufficient evidence) — there is enough historical and structural evidence (2,712 trades, an established concentration diagnosis, a specific monitored weakness with a scheduled checkpoint) to form a view, not merely "we don't know." Not D (evidence supports change) — nothing in this audit's fresh evidence (the live 19-trade sample, still too small; the historical reconstruction, unchanged and positive) crosses the threshold this project has consistently required before recommending a live change.

## 14. Live evidence

Per Part 12's explicit reminder: **the entry-price logging defect (Phase 27) was a logging defect, not an execution failure** — SL/TP/PnL were never affected, only the recorded entry price for pre-2026-08-08 trades. This does not affect any figure in §12/§13, which are computed from `R`/`profit` fields that were never corrupted by that defect.

## 15. AUDUSD Monday LONG

Not modified, not re-backtested. Reviewed against the complete cross-phase evidence trail:

- **Why it survived while other candidates failed:** it is the only candidate in the project's history to show a genuinely strong, cost-robust OOS result (Phase 30: OOS PF 3.07 at 1x cost, 2.647 at 2x) — none of Phase 33's or Phase 35's 7 confirmatory candidates came close to this magnitude of edge.
- **Is its evidence genuinely independent?** Partially — it shares the calendar-drift family and the 2023-2026 data window with the other 59 Phase 30 screen cells, so it is not a fully independent draw from a different search process; but its specific (instrument, day, direction) combination has not been re-tested or re-parameterized since discovery.
- **Is its apparent edge robust?** Not formally tested against the ±20% perturbation standard Phase 33/35 established — a genuine, disclosed gap (it predates that standard).
- **Is it merely surviving because it has been tested less?** Partially plausible — it has not been subjected to the OOS-sub-half or parameter-perturbation checks that rejected all 7 of Phase 33/35's candidates, so its survival reflects a lighter validation history, not necessarily stronger underlying evidence than a candidate that failed those checks would have shown if it hadn't been tested that hard.
- **Portfolio correlation:** 0.29 to the control (Phase 30/32) — above the control's own 0.192 internal average, a real limitation.
- **Target profile fit:** satisfies Priority 1 (HIGH-vol, its best-evidenced characteristic) and Priority 4 (non-JPY); fails Priority 2 (correlation above target) and Priority 5 (still Monday-only, no session-gap fill).

**Classification: PROMISING BUT UNDER-SAMPLED / UNDER-VALIDATED.** Not rescued, not promoted, not modified.

## 16. Sample-size audit

`reports/phase36_sample_size_audit.csv`. Every confirmatory candidate's OOS trade count (57-532) is **statistically informative for an aggregate PF/expectancy point estimate** (all ≥30); most are also adequate for a 2-way sub-split. **The current live portfolio's 19-trade post-demotion sample is explicitly OBSERVED ONLY, not statistically informative** — consistent with every prior phase (27-35) reaching the same conclusion independently, not a new finding of this phase but a reconfirmation.

## 17. Multiple-testing audit

`reports/phase36_multiple_testing_audit.csv`. 67 total hypotheses/cells across 6 instruments, 8 strategy families, 12 sessions. 7 confirmatory (pre-registered, held-out OOS) candidates; 60 exploratory screen cells. 65 rejected, 3 surviving to PROMISING (2 calendar-drift cells including AUDUSD Monday LONG, plus 1 residual). **Per instruction, no retroactive statistical correction is applied merely to make results look worse** — the honest limitation stated is that the confirmatory sample (n=7) remains small and the calendar/instrument overlap across hypotheses (§9) means these are not fully independent tests.

## 18. Search-space coverage

`reports/phase36_search_space_coverage.csv`. **89.6% of all tested hypotheses (60 of 67) belong to the calendar_drift family** — a stark, unambiguous concentration. By instrument: USDCAD (19.4%) and AUDUSD (17.9%) are the most-tested; by session, the calendar screen's five weekdays are each tested at similar (~18%) rates. **This single number is the clearest quantitative answer to RQ7.**

## 19. Alternative return streams

Full detail: `reports/phase36_alternative_return_streams.md` — 10 categories mapped (cross-asset relationships, index CFDs, commodities, volatility-sensitive systems, relative-value spreads, session-specific events, cross-sectional FX, multi-asset momentum, macro/event-conditioned systems, and a catch-all). Ranked by evidence backing, data availability, and implementation cost — no specific strategy proposed.

## 20. Is the search space wrong?

Per Part 18's explicit multi-answer format:

- **B. Too narrow — supported.** 12.5% family coverage after Phase 34, now 50% after Phase 35, but 89.6% of all raw hypothesis-count concentrated in one family (§18).
- **C. Too biased toward simple technical FX strategies — supported.** Every confirmatory candidate (7 of 7) has been a single-instrument technical price-action rule on a major FX pair or gold; zero candidates from any other asset class or data source.
- **D. Too heavily dependent on current market regime — partially supported.** All confirmatory testing sits inside one ~3.5-year window, and this phase's own fresh regime analysis (§10) found the specific OOS window was an unusually low-volatility period for the FX pairs tested — a genuine, disclosed regime-dependency concern, though not proven to be the *sole* cause of the rejections (the robustness failures in Phase 33 and the uniform edge-absence in Phase 35 are independently diagnosable, per Phase 34's own analysis, as implementation-specific issues too).
- **A. Broad enough but unlucky — not well supported** given the 89.6% concentration figure.
- **E. Methodologically over-filtered — not supported.** The gates (OOS consistency, parameter perturbation) each independently caught real, severe problems (Phase 34's own finding, reconfirmed here) — they are not rejecting viable candidates on technicalities.
- **F. Insufficient evidence — not the primary answer**, though the confirmatory sample (n=7) remains genuinely small (§7's wide confidence intervals).

## 21. Discovery efficiency

67 hypotheses tested → 3 reaching PROMISING (1 in 22.3) → 0 reaching portfolio-qualified (0 in 67, so far). **"Observed research frequency suggests"** (not "there is an X% chance") that finding a portfolio-qualified candidate within the current technical-FX search space, at the current pace and mechanism diversity, would require substantially more testing at the current narrow concentration — which is exactly why §22/`phase36_research_direction.md` recommends diversifying the search space rather than simply running more of the same.

## 22. Research direction options

Full detail: `reports/phase36_research_direction.md`. All five options (A-E) evaluated explicitly against this phase's evidence.

## 23. Recommended Phase 37 direction

**D (expand into a different return-stream class), prioritizing cross-sectional FX momentum and session-specific event structures** — both immediately actionable with existing data/infrastructure, both genuinely distinct from every technical-FX design tested so far. Full reasoning: `reports/phase36_research_direction.md`.

## 24. Limitations

- The confirmatory-candidate base rate (n=7) is genuinely small — every percentage in §7 carries a wide confidence interval, stated explicitly, not hidden behind a single point estimate.
- All confirmatory testing shares a single ~3.5-year calendar window and overlapping instruments — not fully independent evidence, per §9/§17.
- REGIME A/B (2019-2022) are characterized by market conditions only; no candidate-level evidence exists for those periods, and none is estimated.
- The live portfolio's 19-trade post-demotion sample remains too small for a confident verdict on its own — this phase's portfolio-viability classification rests primarily on the much larger historical reconstruction plus the existing monitoring framework, not on the small live sample in isolation.

## 25. Final verdict

### Answers to Part 24's fourteen questions

1. **% of tested hypotheses with a credible OOS edge?** Confirmatory: 28.6% (2 of 7, wide CI 8.2-64.1%). Screen-stage: not directly comparable (single-split methodology).
2. **% surviving robustness?** 0% of confirmatory candidates (0 of 7).
3. **% surviving cost stress?** Both Phase 33 candidates that had an edge survived cost stress (2 of 2 with an edge); Phase 35's candidates were never tested (no edge to stress).
4. **% with acceptable HIGH-vol behaviour?** 1 of 7 confirmatory candidates (XAUUSD, STRONG) plus AUDUSD Monday LONG (STRONG) — the rest WEAK or UNKNOWN.
5. **% with useful drawdown diversification?** 0 of the candidates where this was actually tested (XAUUSD failed it; most others weren't reached).
6. **% reaching portfolio-qualified?** 0% (0 of 7).
7. **Dominant failure modes?** OOS/robustness instability (56.7% of all rejections) and edge-absence (37.3%) — a real split between Phase 33's near-misses and Phase 35's clean rejections.
8. **Are failures concentrated in the current regime?** Partially — the OOS window was measurably low-volatility for the FX pairs tested, a genuine contributing factor, though not the sole diagnosed cause.
9. **Is the current six-strategy portfolio still defensible?** Yes — B. DEFENSIBLE WITH MONITORING.
10. **Is the research universe sufficiently broad?** No — 89.6% concentration in one family.
11. **Are we over-searching simple technical FX strategies?** Yes, confirmed.
12. **Is another technical FX strategy still the rational next search?** Not as the primary focus — a different mechanism class (cross-sectional FX, event structures) is better supported by the evidence than another single-instrument technical rule.
13. **What alternative return-stream families deserve consideration?** Cross-sectional FX momentum and session-specific event structures, immediately; volatility-sensitive and macro/event-conditioned systems, pending data access (`phase36_alternative_return_streams.md`).
14. **Should Phase 37 continue current discovery, change architecture, or expand into a different return-stream class?** **Expand (D)**, informed by (not replacing) the existing disciplined architecture.

### Overall classification

**D. EXPAND INTO A DIFFERENT RETURN-STREAM CLASS** (primary), retaining elements of **A. CONTINUE CURRENT DISCOVERY** for the two immediately-actionable families identified — not B, C, or E, per the reasoning in `reports/phase36_research_direction.md`.

---

## Safety check confirmation

No live strategy/parameter/risk modified · no strategy paused · no 5ers configuration modified · no candidate deployed · AUDUSD Monday LONG untouched · Phase 36 preregistration committed (`b5c3769`) before substantive analysis and not altered since · no existing research report overwritten · all 67 rejected/pending hypotheses retained in the ledger · no candidate rescued · no new strategy backtested in this phase · no portfolio optimization · no live decision made · research validator passed · current six-strategy population correctly date-scoped at 2026-07-31 · latest production cutoff explicitly re-verified (2026-08-13 19:12:09 UTC) · multiple testing tracked · sample limitations documented throughout · raw production export not committed.

---

*No live trading change authorized. Phase 37 will scope (not yet backtest) the recommended return-stream families.*
