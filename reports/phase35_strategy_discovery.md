# Phase 35 — Expanded Target-Profile Strategy Discovery

**Research only. No live strategy, parameter, risk, or portfolio weight modified. No candidate deployed. AUDUSD Monday LONG not modified.**

**Pre-registration:** `reports/phase35_preregistration.md`, committed (`7821cd7`) before any candidate was backtested. Not altered after seeing results.

---

## 1. Executive summary

All five pre-registered hypotheses (H1-H5) were backtested with real MT5 data, a strict chronological TRAIN/VALIDATION/OOS split, and the exact frozen mechanics from the pre-registration. **All five failed Gate 1 (no credible OOS edge)** — every candidate's OOS profit factor sits below 1.0, and four of the five show negative expectancy in *both* OOS sub-halves (the fifth, H4, is negative in aggregate and in its first half, with only a mild positive second half that does not change its overall rejection). The result is stable across ±20% parameter perturbation for every candidate — none flip to a positive edge at any tested parameter value. **This is a clean, decisive, non-borderline "no edge" outcome across the entire expanded search, not a set of near-misses.**

**Research outcome: A. NO CANDIDATE.**

---

## 2. Phase 34 target profile (recap, unchanged)

Priority order used to design H1-H5: (1) HIGH-volatility compatibility, (2) low drawdown-specific correlation, (3) genuinely different mechanism, (4) preferably non-JPY, (5) preferably London/NY exposure. Not modified in this phase.

## 3. Preregistration

Full document: `reports/phase35_preregistration.md`, committed separately (`7821cd7`) before any backtest ran. No amendment was required.

## 4. Data sources

MT5 MetaQuotes-Demo feed (unchanged limitation from every prior phase). USDCAD: 22,457 H1 bars + 5,624 H4 bars + 940 D1 bars. AUDUSD: 22,458 H1 bars. USDCHF: 22,457 H1 bars. All 2023-01-01 to 2026-08-14, validated for monotonic timestamps, zero duplicates, positive/consistent OHLC before any backtest ran — no data-integrity STOP was triggered.

## 5. Candidate registry

Full detail: `reports/phase35_candidate_registry.csv` — all five hypotheses, instruments, and rationale recorded before any result existed.

---

## 6. H1 — NY Open Range Breakout (USDCAD)

**Result: REJECTED — NO EDGE.** OOS PF 0.890 (306 trades), expectancy −0.071R. Both OOS sub-halves negative and nearly identical (−0.072R / −0.070R) — a stable, non-borderline negative result, not noise. Parameter perturbation (TP multiplier 1.2x/1.5x/1.8x) shows PF monotonically *declining* as the target is loosened (0.951 → 0.890 → 0.824) — the tighter target actually performs relatively better, but never clears 1.0 at any setting.

## 7. H2 — NY Session Momentum (AUDUSD)

**Result: REJECTED — NO EDGE.** The most decisive rejection of the five: 532 OOS trades (the largest sample), OOS PF 0.639, total OOS loss −159.58R. Negative in both sub-halves (−0.343R / −0.257R). PF stays in a narrow 0.639-0.666 band across all three perturbations — the momentum-continuation hypothesis, as specifically implemented (3-hour move exceeding its own 20-day rolling average), shows no exploitable signal in AUDUSD's NY session.

## 8. H3 — London/NY Overlap Continuation (USDCHF)

**Result: REJECTED — NO EDGE.** Weakest profit factor of the five candidates: OOS PF 0.540 (172 trades), expectancy −0.342R. Negative in both sub-halves (−0.404R / −0.280R). The hypothesis that a high-efficiency-ratio London session predicts continuation through the NY overlap is not supported by this implementation.

## 9. H4 — Multi-Timeframe Trend Continuation (USDCAD)

**Result: REJECTED — NO EDGE**, but the least decisively rejected of the five and the only candidate showing an interesting internal pattern worth carrying forward as a lesson (not a rescue): OOS PF 0.795 (63 trades, expectancy −0.151R), first OOS half strongly negative (−0.378R) but **second OOS half mildly positive (+0.068R)**. This is the only candidate in this phase whose OOS sub-halves disagree in sign — flagged per the pre-registered rule as a WARNING-tier observation on a sample that, at 31-32 trades per half, is genuinely thin (`reports/phase34_validation_bar_audit.csv`'s own recommended treatment for exactly this situation). **This does not change the classification** (aggregate OOS PF remains below 1.0 at every perturbation, 0.771-0.836), but it is the one result in this phase that doesn't look like flat, uniform noise, and is worth noting for any future revisit. Adding the D1 trend filter (the specific design change motivated by Phase 33/34's USDCAD lesson) did not produce a credible edge, but it also did not reproduce the earlier candidate's *catastrophic* parameter fragility (a full sign reversal) — this newer design's perturbation range (0.771 to 0.836) is comparatively narrow and stable, even though the whole range sits below the required bar.

## 10. H5 — ATR-Scaled Volatility Expansion (AUDUSD)

**Result: REJECTED — NO EDGE.** OOS PF 0.653 (123 trades), expectancy −0.314R, negative in both sub-halves (−0.483R / −0.148R). Interesting internal pattern: PF *improves* monotonically as the TP-ATR multiplier loosens (0.553 at 2.0x → 0.653 at 2.5x → 0.763 at 3.0x) — the same directional pattern seen for H1 in reverse, suggesting these breakout-family designs may generally benefit from wider targets, though even the loosest tested value (3.0x) doesn't clear 1.0. **This candidate directly tests Phase 34's XAUUSD diagnosis (that a fixed, non-adaptive target caused the earlier instability) by scaling the target to realized ATR instead — and the ATR-scaled version still shows no edge on AUDUSD.** This is informative: it suggests the earlier XAUUSD problem may not have been solely about the exit-scaling mechanism, or that the volatility-contraction-to-expansion precondition itself doesn't generalize cleanly from gold to AUDUSD — both are now open questions for any future work in this family, not resolved by this phase.

---

## 11. OOS results

Full detail: `reports/phase35_candidate_results.csv`. All five OOS profit factors: 0.540 to 0.890 — every one below the 1.0 breakeven line.

## 12. OOS consistency

Full detail: `reports/phase35_oos_consistency.csv`. Four of five candidates negative in both OOS sub-halves. H4 is the sole exception (§9).

## 13. Parameter robustness

Full detail: `reports/phase35_parameter_robustness.csv`. **No candidate flips to a positive edge at any of the three tested parameter values (−20%/baseline/+20%).** This is itself a meaningful finding distinct from Phase 33's: Phase 33's candidates were unstable (sign-flipping) around a positive baseline; Phase 35's candidates are *stable* around a negative baseline. Neither pattern indicates a usable edge, but they are different failure shapes, worth distinguishing for future methodology (`reports/phase34_validation_bar_audit.csv`'s framework correctly classifies both as failures without conflating them).

## 14. Cost stress

Not run in full, per the frozen preregistration's rule ordering: Gate 1 (OOS edge) is evaluated first, and none of the five candidates has a credible edge to stress-test against costs. `reports/phase35_cost_stress.csv` records this explicitly as N/A per gate, not omitted or hidden.

## 15. HIGH-volatility behaviour

Full detail: `reports/phase35_regime_analysis.csv`. Where classifiable: H2's HIGH-vol tercile is WEAK (expectancy −0.242R, 118 trades); H3's HIGH-vol tercile is WEAK (−0.496R, 46 trades); H5's HIGH-vol tercile is WEAK (−0.117R, 27 trades). H1 could not be classified (insufficient TRAIN+VAL sample to fix ATR terciles — this candidate's backtest function did not persist an `atr_at_entry` value, a design limitation noted for any future revisit). H4's HIGH-vol tercile has only 7 trades — UNKNOWN per the pre-registered 10-trade floor. **None of the five candidates shows STRONG HIGH-volatility compatibility** — every classifiable candidate is WEAK in exactly the regime the target profile most wants a candidate to succeed in.

## 16. Drawdown correlation

Not run in full, per the same Gate-1-first rule ordering as §14 (`reports/phase35_drawdown_correlation.csv`, N/A per gate). Per the pre-registered rule ("Do not rescue [Gate-1 failures] using portfolio correlation"), a low or favorable drawdown correlation could not have changed any of these five classifications, so it was not computed in full for a candidate that already fails on the more fundamental question of whether an edge exists.

## 17. Mechanism diversification

All five candidates are, by design, genuinely different mechanisms from the current AMR-heavy portfolio (breakout, momentum, session continuation, multi-timeframe trend, volatility expansion — none are mean-reversion). **This criterion is satisfied by construction for every candidate — it is not the reason any of them was rejected.**

## 18. Session diversification

H1/H2 (New York), H3 (London/NY overlap), H5 (London) would all have added exposure outside the current book's 94.7%-Asian-concentrated design; H4 (D1/H4, session-unrestricted) is the exception by design. **All five candidates would have addressed at least part of the session gap identified in Phase 31/34 — this too was not the rejection reason for any candidate.**

## 19. Currency diversification

All five are non-JPY (USDCAD ×2, AUDUSD ×2, USDCHF ×1) — satisfying Phase 32's Priority 4. **Also not the rejection reason for any candidate.**

## 20. Portfolio integration

Not run, per the same Gate-1-first rule ordering (§14/§16). `reports/phase35_portfolio_integration.csv` records this as N/A per gate.

## 21. Monte Carlo

Not run in full for the same reason (§14/§16/§20). `reports/phase35_monte_carlo.csv` records this as N/A per gate.

---

## 22. Multiple-testing assessment

- **5 pre-registered hypotheses, 1 parameter set each** — 15 total parameter evaluations (5 baseline + 10 perturbations), all disclosed in `reports/phase35_parameter_robustness.csv`, none omitted for looking weak.
- **0 candidates added after seeing results.**
- Cumulative project total: Phase 30 (60 exploratory cells, one family) + Phase 33 (2 confirmatory candidates, two families) + Phase 35 (5 confirmatory candidates, four families, since H1/H2 share the "New York" family bucket at a coarse level but are mechanistically distinct — see `reports/phase34_strategy_family_taxonomy.csv`) = **8 of 16 taxonomized families now tested (50%)**, exactly matching the projection made in `reports/phase35_preregistration.md` §11 before this phase began.
- This entire OOS window (2025-05-01 to 2026-08-14) has now been inspected once per H1-H5. Any future revisit of the same window for the same hypotheses is EXPLORATORY, not confirmatory.

## 23. AUDUSD Monday LONG assessment (prior candidate, not re-run)

Not modified, not re-backtested. Its status is unchanged from Phase 34: **PROMISING / PARTIAL MATCH.** It remains the only candidate in the project's history with a genuinely positive, cost-robust OOS result (Phase 30) and a demonstrated HIGH-volatility strength (Phase 32) — a materially stronger evidence base than any of H1-H5 produced this phase. It still fails on drawdown correlation (0.29, above target) and session coverage (still Monday-only). **It remains the strongest single candidate in the project's cumulative research to date, despite not being promoted.**

## 24. Candidate classifications

| Candidate | Classification |
|---|---|
| H1 USDCAD NY ORB | A. REJECTED — NO EDGE |
| H2 AUDUSD NY Momentum | A. REJECTED — NO EDGE |
| H3 USDCHF Overlap Continuation | A. REJECTED — NO EDGE |
| H4 USDCAD Multi-Timeframe Trend | A. REJECTED — NO EDGE |
| H5 AUDUSD ATR-Scaled Vol Expansion | A. REJECTED — NO EDGE |

Full detail: `reports/phase35_final_rankings.csv`.

## 25. Rejected candidates

All five, §24. Full evidence trail retained in every supporting CSV.

## 26. Promising candidates

None from this phase's new hypotheses. AUDUSD Monday LONG (a prior candidate, §23) remains the project's one standing PROMISING/PARTIAL MATCH result.

## 27. Demo-qualified candidates

None.

## 28. Limitations

- **Data source**: MT5 MetaQuotes-Demo feed, not the 5ers production broker — unchanged limitation across every phase.
- **H1's backtest function did not persist an `atr_at_entry` value**, so its HIGH-volatility gate could not be classified at all (§15) — a design limitation for any future revisit of this specific implementation, not a finding about the strategy itself.
- **Gates 14/16/20/21 (cost stress, drawdown correlation, portfolio integration, Monte Carlo) were not run in full**, per the frozen rule that a Gate-1 failure cannot be rescued by downstream evidence — this is a deliberate, pre-registered efficiency choice, not an omission; all four CSVs are still produced and explicitly marked N/A rather than left missing.
- **H4's positive second-OOS-half (§9) rests on only 32 trades** — explicitly flagged as thin-sample, not treated as evidence of a real, recoverable edge.
- **Five hypotheses is still a small sample of the 16-family taxonomy** (8/16 or 50% cumulative coverage after this phase) — this phase's uniformly negative result narrows, but does not close, the search space Phase 34 identified as too narrow.

## 29. Final verdict

### Answers to Part 29's ten questions

1. **Did any hypothesis produce a credible independent OOS edge?** No — all five OOS PFs are below 1.0.
2. **Did any candidate survive parameter perturbation?** Not applicable in the affirmative sense — none had a positive baseline to test surviving; all five remained negative (i.e., "stable" in the sense of consistently showing no edge) across all three perturbations.
3. **Did any candidate survive 2x cost stress?** Not tested, per the Gate-1-first rule — moot given no candidate had an edge to stress.
4. **Did any candidate demonstrate HIGH-volatility compatibility?** No — every classifiable candidate was WEAK in its HIGH-vol tercile.
5. **Did any candidate demonstrate low drawdown correlation?** Not tested, per the Gate-1-first rule.
6. **Did any candidate provide genuine mechanism diversification?** Yes, all five by design (§17) — this was never the limiting factor.
7. **Did any candidate provide useful New York/session diversification?** By design, yes (§18) — but this is moot without an underlying edge.
8. **Did any candidate reduce JPY concentration?** All five are non-JPY by design (§19) — also moot without an edge.
9. **Did any candidate improve portfolio-level behaviour?** Not tested — no candidate reached the portfolio-integration stage.
10. **Did any candidate qualify for DEMO FORWARD TEST?** No.

### Research outcome: **A. NO CANDIDATE**

Reported plainly, per instruction: this expanded, deliberately-designed five-hypothesis search — addressing the exact gap (New York session, non-mean-reversion mechanism) that Phase 34 identified as under-tested — produced a uniformly negative result. This is a different, and in some ways more informative, outcome than Phase 33's near-misses: **these five specific implementations show no edge at all, not a fragile one.** The bar was not lowered anywhere in this phase, and no candidate was rescued or advanced past its earned classification.

---

## Safety check confirmation

Six live strategies unchanged · no parameters/risk/5ers configuration changed · no candidate deployed or optimized · AUDUSD Monday LONG untouched · Phase 35 preregistration committed (`7821cd7`) before any candidate result existed and not modified since · all five hypotheses tested exactly as pre-registered, no substitutions · every candidate's full result recorded regardless of outcome · no candidate rescued · OOS boundaries respected (2025-05-01 cutoff, never inspected before backtesting) · no future leakage · parameter perturbation performed for all five · OOS consistency performed for all five · HIGH-volatility analysis performed where sample allowed · multiple-testing tracked (§22) · research validator passed · raw production 5ers export not committed.

---

*No live trading change authorized. Reproducible via `python src/phase35_strategy_discovery.py`.*
