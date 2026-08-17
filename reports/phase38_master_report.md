# Phase 38 — Cross-Sectional FX (H1) + Session-Specific Structures (H2) Master Report

**Research only. No live strategy, parameter, risk, or portfolio weight modified. AUDUSD Monday LONG untouched. No candidate deployed.**

---

## 1. Executive summary

**Both H1 (cross-sectional FX relative-momentum) and H2 (Asian-range breakout continuation) failed at Gate 1 — no credible OOS edge.** H1: OOS PF 0.649 (84 trades). H2: OOS PF 0.798 (458 trades, the largest and most decisive sample in the phase). Neither reaches the drawdown-diversification question with a standalone edge to actually integrate — but both were carried through the full battery anyway (per the preregistered "do not stop early" discipline), and **both would also have failed the drawdown-correlation gate** had they passed Gate 1 (H1: 0.611 drawdown-day corr vs 0.136 normal; H2: 0.269 vs -0.085). This is a genuinely negative, informative result: two structurally distinct return-stream classes, entering with real theoretical rationale, both failed to clear even the first bar.

## 2. Phase 36 context

Phase 36 concluded D. EXPAND INTO A DIFFERENT RETURN-STREAM CLASS after finding 89.6% of 67 hypotheses concentrated in the calendar/drift family.

## 3. Phase 37 context

Phase 37 validated AUDUSD Monday LONG to F. REJECTED — POOR DRAWDOWN DIVERSIFICATION (strong standalone edge, PF 3.070, but 0.742 drawdown-day correlation) and recommended Cross-sectional FX and Session-specific structures as the two immediately testable classes — exactly H1 and H2 here.

## 4. Preregistration

`reports/phase38_preregistration.md`, committed separately (`af03e04`) before any backtest ran. **One amendment** (`111e09d`, dated 2026-08-18, committed before any H2 result under the amended rule existed): the literal open-price breakout entry was non-executable by construction (1 EURUSD trade in ~1,900 days — the London-open bar's open price is definitionally almost always inside the range that produced it), amended to an intrabar high/low breakout with entry at the breakout level. Not a parameter search.

## 5. Data integrity

`research_data_validator` passed on the control input before every run. MT5 D1/H1 pulls for all 10 instruments (7 for H1, 3 for H2) passed monotonicity/duplicate/positive-OHLC asserts. Session hours used raw MT5 server-hour convention (ASIAN=[0,7), LONDON=[7,16), NY≈21-22 close), matching the project's established post-server-time-fix convention (see `src/phase19_london_ny_volatility_persistence.py`), not a naive UTC assumption.

## 6. Structural independence

`reports/phase38_structural_independence.csv`. **H1: A. GENUINELY DISTINCT** — no prior hypothesis in the 68-item ledger uses a multi-instrument relative-ranking construction. **H2: B. RELATED BUT MEANINGFULLY DIFFERENT** — distinguished from AMR (mean-reversion vs. this candidate's breakout-continuation), and from Phase 35's three NY-session hypotheses (this candidate triggers at the Asian→London transition and holds through NY close, spanning three sessions, rather than triggering/exiting within a single NY-local window). Neither classified C (duplicative); both proceeded to backtesting.

## 7. H1 cross-sectional FX

`reports/phase38_h1_cross_sectional_oos.csv`. 8-currency universe (USD/EUR/GBP/JPY/AUD/CAD/CHF/NZD via 7 USD-pair legs), 20-trading-day relative-momentum ranking, weekly Friday-close rebalance, long-strongest/short-weakest, held Monday open to Friday close.

## 8. H1 OOS

IS (2023-2025): 297 trades, PF 0.830, expectancy -0.0783R. **OOS (2025-2026): 84 trades, PF 0.649, expectancy -0.1648R, total -13.84R.** Negative in both IS and OOS. **Gate 1: FAIL.**

## 9. H1 robustness

`reports/phase38_h1_parameter_robustness.csv`. Lookback 16d/20d/24d: PF 0.582/0.649/0.517 — negative at every setting, no sign reversal (already negative, nothing to reverse from).

## 10. H1 cost stress

`reports/phase38_h1_cost_stress.csv`. PF 0.649 (normal) → 0.628 (1.5x) → 0.608 (2x). Already below 1.0 before cost stress is even relevant.

## 11. H1 regime behaviour

`reports/phase38_h1_regime_analysis.csv`. HIGH-vol bucket: 15 trades, PF 0.491, expectancy -0.6319R — classification **WEAK**, its worst regime.

## 12. H1 drawdown correlation

`reports/phase38_h1_drawdown_correlation.csv`. Normal-day corr 0.136, drawdown-day corr 0.611 (8 overlapping days — at the preregistered floor). **Classification: CORRELATED.** Would have failed this gate even had Gate 1 passed.

## 13. H1 portfolio integration

`reports/phase38_h1_portfolio_integration.csv`. At 1.0x weight: control total_R 142.31 → combined 128.47 (worse); control max_dd -14.53 → combined -14.74 (worse). Adding H1 makes the portfolio strictly worse on both dimensions.

## 14. H1 Monte Carlo

`reports/phase38_h1_monte_carlo.csv`. **SIMULATED.** Actual max DD -14.61R sits at the 86.3rd percentile of 10,000 reshuffles (median -17.07R) — the actual sequencing is somewhat better than a typical random reshuffle of its own (already-losing) trades, not a meaningful mitigant given the underlying negative edge.

## 15. H2 session structure

`reports/phase38_h2_session_oos.csv`. EURUSD/GBPUSD/AUDUSD, Asian range [0,7h) UTC-server-hour, London-open (hour 7) intrabar breakout of that range, stop at the opposite range edge, exit at NY close (hour 22) or stop, whichever first.

## 16. H2 OOS

IS (2023-2025): 517 trades, PF 0.961, expectancy -0.0239R. **OOS (2025-2026): 458 trades, PF 0.798, expectancy -0.119R, total -54.5R.** The largest, most statistically decisive sample of the phase. **Gate 1: FAIL.**

## 17. H2 robustness

`reports/phase38_h2_parameter_robustness.csv`. Asian-window 5h/7h(baseline)/8h: PF 1.050 (positive) / 0.798 / **degenerate — 0 trades at 8h** (the perturbed window overlaps the entry hour itself, making a bar's breakout of its own contributing range structurally impossible — a disclosed limitation of that specific perturbed value, not a hidden result). **Sign reversal: YES** (positive at -20%, negative at baseline) — an independent robustness failure, on top of the baseline edge failure.

## 18. H2 cost stress

`reports/phase38_h2_cost_stress.csv`. PF 0.798 → 0.746 (1.5x) → 0.697 (2x). Already below 1.0.

## 19. H2 regime behaviour

`reports/phase38_h2_regime_analysis.csv`. HIGH-vol bucket: 198 trades (large, reliable sample), PF 1.006, expectancy +0.0025R. Classification **STRONG** by the mechanical rule (expectancy > 0), but this is economically negligible — a coin-flip result, not a real edge.

## 20. H2 drawdown correlation

`reports/phase38_h2_drawdown_correlation.csv`. Normal-day corr -0.085, drawdown-day corr 0.269 (28 overlapping days — well-sampled). **Classification: CORRELATED.**

## 21. H2 portfolio integration

`reports/phase38_h2_portfolio_integration.csv`. At 1.0x weight: control total_R 142.31 → combined 87.81 (materially worse); control max_dd -14.53 → combined -22.78 (materially deeper). The largest portfolio degradation of any candidate tested across Phases 37-38.

## 22. H2 Monte Carlo

`reports/phase38_h2_monte_carlo.csv`. **SIMULATED.** Actual max DD -57.62R sits at the 85.1st percentile of the reshuffled distribution — again, sequencing is not the problem; the underlying edge is negative.

## 23. Sample size

`reports/phase38_sample_size.csv`. H1's OOS sample (84 trades) and drawdown-overlap sample (8 days, at the floor) are both adequate-but-thin. **H2's OOS sample (458 trades) and drawdown-overlap sample (28 days) are both large and statistically decisive** — this is the more confidently rejected of the two hypotheses.

## 24. Multiple testing

`reports/phase38_multiple_testing.csv`. Exactly 2 confirmatory hypotheses tested, each with exactly 1 preregistered ±20% perturbation (3 values each). One necessary, disclosed, pre-results methodology amendment for H2's entry-price operationalization. No alternative signal families or session mechanisms were explored (explicitly excluded by preregistration). No hidden variants, no cherry-picking.

## 25. Portfolio gap assessment

`reports/phase38_portfolio_gap_assessment.csv`. Neither candidate reaches a meaningful gap-solving assessment because neither has a standalone edge — per Part 9's own instruction, a candidate that only reduces JPY concentration or adds session diversity without solving the drawdown-correlation and edge questions is not a successful solution regardless.

## 26. AUDUSD comparison

AUDUSD Monday LONG (Phase 37, unmodified, not re-tested here): strong standalone edge (OOS PF 3.070) + strong robustness + poor drawdown diversification (0.742). **Neither H1 nor H2 solves the specific failure that rejected AUDUSD** — both fail even earlier, at the standalone-edge gate, and both would independently fail the same drawdown-diversification gate AUDUSD failed (H1: 0.611, H2: 0.269, vs. AUDUSD's 0.742 — better than AUDUSD on this one dimension, but moot without an edge). AUDUSD Monday LONG remains, by a wide margin, this project's strongest-ever standalone FX candidate; H1 and H2 do not approach it.

## 27. Candidate classifications

`reports/phase38_candidate_classifications.csv`. **H1: B. REJECTED — NO CREDIBLE OOS EDGE. H2: B. REJECTED — NO CREDIBLE OOS EDGE.** Neither reaches classification I or J.

## 28. Limitations

- H1's synthetic cross-sectional trade construction (combining USD-leg log-returns rather than trading a direct cross pair) is an approximation of a tradeable position, not an executed synthetic spread — real execution would add basis/rollover cost not modeled here, which would only make the (already negative) edge worse.
- H2's entry-price operationalization required a disclosed, pre-results amendment (§4) — the original literal rule was non-executable, not merely suboptimal; this is a genuine implementation-definition gap in the original hypothesis specification, not evidence about the underlying market phenomenon.
- H2's +20% Asian-window perturbation produced a structurally degenerate (0-trade) result because that specific window overlaps the entry hour — a known artifact of this particular perturbation value, disclosed rather than hidden, but it means the ±20% robustness test for H2 is less informative on the high side than the low side.
- Both hypotheses' drawdown-correlation classifications (H1: 8-day overlap, right at the preregistered floor) carry real sampling uncertainty despite passing the minimum-sample bar.

## 29. Final verdict

### Answers to the sixteen required questions

1. **Was H1 genuinely structurally different?** Yes — A. GENUINELY DISTINCT.
2. **Was H2 genuinely structurally different?** Yes — B. RELATED BUT MEANINGFULLY DIFFERENT.
3. **Did H1 demonstrate credible OOS edge?** No (PF 0.649).
4. **Did H2 demonstrate credible OOS edge?** No (PF 0.798, larger and more decisive sample).
5. **Did either survive OOS consistency?** Both were sign-consistent — but consistently negative, not consistently positive.
6. **Did either survive ±20% parameter perturbation?** H1: no sign reversal (already negative throughout). H2: sign reversal (positive at -20%, negative/degenerate elsewhere) — an independent failure.
7. **Did either survive 2x cost stress?** No — both were already below PF 1.0 before cost stress.
8. **Did either demonstrate acceptable HIGH-volatility behaviour?** H1: no (WEAK, expectancy -0.63R). H2: nominally yes (STRONG by the mechanical rule) but economically negligible (+0.0025R).
9. **Did either show low correlation specifically during portfolio drawdowns?** No — both CORRELATED (H1: 0.611 vs 0.136 normal; H2: 0.269 vs -0.085 normal).
10. **Did either improve portfolio-level drawdown behaviour?** No — both worsened it (H1 modestly, H2 materially).
11. **Did either meaningfully diversify the mechanism mix?** Structurally yes for both (relative-ranking for H1; session-transition breakout for H2) — but moot without a standalone edge.
12. **Did either meaningfully diversify session exposure?** H2 structurally yes (London-open trigger, non-JPY universe) — moot without an edge. H1 not session-specific by design.
13. **Did either reduce JPY concentration without creating correlated losses?** Neither reaches this question meaningfully — both fail the edge gate first, and both would fail the drawdown-correlation gate independently.
14. **Did either outperform AUDUSD Monday LONG's portfolio-diversification profile?** On the single dimension of drawdown-day correlation, both numerically beat AUDUSD's 0.742 (H1: 0.611, H2: 0.269) — but this is irrelevant without a standalone edge to actually deploy.
15. **Did either qualify for DEMO FORWARD TEST?** No.
16. **If neither qualifies, what specifically killed them?** Both died at Gate 1 (no credible OOS edge) — the most fundamental gate, before drawdown diversification, cost, or regime behavior become decision-relevant. H2 additionally failed independently on parameter robustness (sign reversal).
17. **Does the evidence support continuing research within FX?** Weakly — two more FX-based hypothesis classes (cross-sectional relative-value and session-transition breakout) have now failed at the most basic level, on top of the 9 single-instrument technical hypotheses already rejected across Phases 33/35/37. The base rate of FX-technical hypotheses clearing even Gate 1 in this project's research history is now low.
18. **Or should Phase 39 move to a different asset/return-stream class?** Yes — this is the more defensible reading of the cumulative evidence. Both of Phase 37's practically-actionable, data-ready FX classes have now failed. The remaining higher-theoretical-priority classes from Phase 37's map (Event/macro-conditioned, Index-based, Volatility-conditioned) all require new data infrastructure not yet built — Phase 39 should either invest in that infrastructure or explicitly reconsider the overall research strategy, rather than generating a third FX-technical variant.

### Final classification table

| Hypothesis | Structural independence | Gate 1 edge | Final classification |
|---|---|---|---|
| H1 — Cross-sectional FX | A. GENUINELY DISTINCT | FAIL (PF 0.649) | **B. REJECTED — NO CREDIBLE OOS EDGE** |
| H2 — Session-specific structures | B. RELATED BUT MEANINGFULLY DIFFERENT | FAIL (PF 0.798) | **B. REJECTED — NO CREDIBLE OOS EDGE** |

## 30. Phase 39 recommendation

**Both H1 and H2 failing is itself the successful, informative research result this phase set out to obtain — it is not a research failure.** Combined with Phase 37's AUDUSD rejection and the 9 prior FX-technical rejections (Phases 33/35), the cumulative evidence base now weighs toward: **Phase 39 should NOT design a third FX-technical hypothesis.** Two credible options, not decided here: (a) invest in the data infrastructure Phase 37 already flagged as the blocker for Event/macro-conditioned or Volatility-conditioned systems (the two classes structurally best-aligned with the portfolio's actual gaps), or (b) conduct a dedicated audit of whether this project's FX-technical research program has reached a base-rate ceiling (extending Phase 36's audit) before committing further research effort to any single-instrument or cross-instrument FX design. **Per the phase's own final instruction: do not automatically begin Phase 39. This report stops here and reports the evidence.**

---

## Safety check confirmation

No live strategy/parameter/risk modified · no strategy paused · no 5ers configuration modified · no candidate deployed · AUDUSD Monday LONG untouched, no optimization performed · Phase 38 preregistration committed (`af03e04`) before substantive backtesting, one disclosed pre-results amendment (`111e09d`) · H1 and H2 frozen before results (amendment applied before any result under the new rule existed) · structural independence tested for both, neither duplicative · no duplicate hypothesis rescued · no unregistered candidate promoted · OOS boundaries respected, no future leakage (Friday-close-anchored ranking for H1, session-boundary-only signals for H2) · cost stress performed for both · parameter robustness performed for both (H2's sign reversal disclosed, not hidden) · OOS consistency performed for both · HIGH-volatility analysis performed for both · drawdown correlation performed for both · portfolio integration performed for both · multiple testing tracked (`reports/phase38_multiple_testing.csv`) · sample limitations documented (`reports/phase38_sample_size.csv`) · raw production 5ers export not committed.

---

*No live trading change authorized. Neither H1 nor H2 reaches J. PORTFOLIO QUALIFIED. Phase 39 not automatically begun.*
