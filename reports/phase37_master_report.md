# Phase 37 — AUDUSD Validation + Return-Stream Diversification Map (Master Report)

**Research only. No live strategy, parameter, risk, or portfolio weight modified. No candidate deployed, optimized, or promoted. AUDUSD Monday LONG parameters unchanged.**

---

## 1. Executive summary

**Track A: AUDUSD Monday LONG reproduces exactly (84 OOS trades, PF 3.070, an exact match to the original Phase 30 result) and passes every gate in the standardized battery — OOS consistency, parameter stability, 2x cost stress, and positive performance across all five historical regimes back to 2019 — except one: it fails the drawdown-diversification gate decisively (0.742 correlation to the control on the control's worst days, vs. 0.228 on normal days, a swing far exceeding the pre-registered 0.15 threshold).** Per the frozen classification rules, this produces **F. REJECTED — POOR DRAWDOWN DIVERSIFICATION**, applied mechanically despite the candidate's genuine strength on every other dimension.

**Track B: of 10 return-stream classes scored against the portfolio's known gaps, the two classes that are simultaneously well-aligned with those gaps AND immediately testable with existing data are Cross-sectional FX and Session-specific event structures.** The two theoretically highest-priority classes (Event/macro-conditioned, Volatility-conditioned systems) both have a confirmed data gap in this project's current toolchain.

**Overall direction: continue expanding into different return-stream classes (per Phase 36's D verdict), now reinforced by Track A's result** — the project's strongest FX-technical candidate, held to full rigor, still does not solve the portfolio's #2 priority (drawdown diversification).

---

## 2. Phase 36 findings (recap)

89.6% of prior hypotheses concentrated in one family; 0/7 confirmatory candidates portfolio-qualified; AUDUSD Monday LONG flagged PROMISING BUT UNDER-VALIDATED, specifically noting it had never been tested against the ±20% robustness standard applied to every other candidate — the exact gap this phase's Track A closes.

## 3. Preregistration

`reports/phase37_preregistration.md`, committed separately (`5da2552`) before any Track A re-run or Track B score was computed. No amendment required.

## 4. Data integrity

`src/research_data_validator.py` passed on the historical control input before analysis. All three newly-authored Track B CSVs were validated for column-count consistency after an initial hand-typing error was caught and corrected by the validator itself — a live demonstration of the tool doing exactly what it was built for.

## 5. AUDUSD reproduction

`reports/phase37_audusd_reproduction.csv`. Re-running `drift_cell()` **verbatim** (not from memory — read directly from `src/phase30_nonjpy_calendar_screen.py`) against a fresh MT5 pull: **OOS trade count 84 (exact match), OOS PF 3.070 (exact match to 3 decimal places).** Reproduction tolerance met with no discrepancy to investigate. New figures not in the original registry, now established: OOS expectancy +0.2548R, win rate 65.5%, total OOS R +21.40, max OOS drawdown −2.87R, max losing streak 3.

## 6. AUDUSD OOS consistency

`reports/phase37_audusd_oos_consistency.csv`. First half (42 trades): expectancy +0.2323R, PF 2.678. Second half (42 trades): expectancy +0.2773R, PF 3.572. **Sign-consistent, verdict PASS.**

## 7. AUDUSD parameter robustness

`reports/phase37_audusd_parameter_robustness.csv`. **Important, disclosed limitation**: this candidate has no trade-selection parameter (entry/exit are fully determined by calendar day and D1 open/close) — the only perturbable value is the ATR(14) normalization window, tested at 11/14/17 bars. PF ranges narrowly from 3.051 to 3.152 across the three settings — no sign reversal, a stable result, but **a structurally weaker robustness test than Phase 33/35's threshold-gated candidates**, since perturbing this parameter only rescales R-multiples, not trade selection.

## 8. AUDUSD cost stress

`reports/phase37_audusd_cost_stress.csv`. PF 3.070 (normal) → 2.851 (1.5x) → 2.647 (2x). Remains comfortably above 1.0 at every tested cost level — matching the figure already on record from Phase 30.

## 9. AUDUSD regime behaviour

`reports/phase37_audusd_regime_analysis.csv`. LOW-vol: 48 trades, PF 2.02. NORMAL: 16 trades, PF 6.13. **HIGH: 20 trades, PF 6.25, classification STRONG** — the candidate's best regime bucket, not its worst, directly confirming Phase 32's earlier finding that this is the project's one candidate with genuinely positive HIGH-volatility behavior.

## 10. AUDUSD historical regime behaviour (2019-2026, real data, unmodified mechanics)

| Period | Trades | Win rate | PF | Expectancy R | Max DD (R) |
|---|---|---|---|---|---|
| 2019-2020 | 102 | 47.1% | 1.170 | +0.034 | −5.53 |
| 2021-2022 | 104 | 54.8% | 1.235 | +0.049 | −5.30 |
| 2023-2024 | 104 | 55.8% | 1.505 | +0.079 | −3.41 |
| 2025 | 52 | 67.3% | 2.573 | +0.215 | −2.87 |
| 2026 YTD | 32 | 62.5% | 4.162 | +0.319 | −1.15 |

**Positive in all five characterized periods, all with adequate sample (≥10 trades)** — genuinely broad, not regime-specific or a recent artifact. **Disclosed caveat**: PF strengthens monotonically toward the present (1.17 → 4.16), and the 2025/2026 figures overlap with the candidate's own original discovery OOS window — this is not a fully independent replication for those two periods, though 2019-2024 (306 trades, PF 1.17-1.51) is genuinely prior, out-of-original-sample evidence.

## 11. AUDUSD drawdown correlation

`reports/phase37_audusd_drawdown_correlation.csv`. Control = `data/phase26_all_trades.csv`, restricted to the candidate's own OOS window (2025-01-01 to 2026-08-14) for a fair comparison. **Normal-day correlation: 0.228. Drawdown-day correlation (9 overlapping worst-decile control days — just above the 8-day viability floor): 0.742.** A swing of 0.514, far exceeding the 0.15 threshold. **Classification: CORRELATED.** This is the decisive finding of Track A: on an average day, AUDUSD Monday LONG looks like a modest diversifier; on the portfolio's actual worst days, it has historically moved *with* the portfolio, not against it.

## 12. AUDUSD portfolio integration

`reports/phase37_audusd_portfolio_integration.csv`. CONTROL (OOS-window-matched, 2025-01-01 onward): total R 142.31, max DD −14.53R. CONTROL+CANDIDATE at 1.0x weight: total R 163.72 (higher), max DD −15.24R (**deeper**, consistent with the CORRELATED finding above — adding a positively-correlated return stream increases return but does not shrink, and modestly deepens, portfolio drawdown).

## 13. AUDUSD Monte Carlo

`reports/phase37_audusd_monte_carlo.csv`. **SIMULATED** (10,000-draw reshuffle of the candidate's own 84 OOS trades): actual max DD −2.87R sits at the 8.9th percentile of the reshuffled distribution (median −1.95R) — the candidate's actual drawdown sequencing is somewhat worse than a typical random reshuffle of its own trades, though not an extreme outlier. Actual max losing streak (3) is below the simulated 95th percentile (6).

## 14. AUDUSD sample size

`reports/phase37_audusd_sample_size.csv`. OOS trade count (84) is statistically informative for a point estimate. OOS sub-halves (42 each) are adequate. HIGH-vol bucket (20 trades) clears the 10-trade floor. **Drawdown-correlation overlap (9 days) clears the 8-day floor, but only barely** — this is the single most sample-constrained input to the final classification, disclosed explicitly rather than treated as unambiguous.

## 15. AUDUSD final classification

Applying `reports/phase37_preregistration.md` §A12's rules mechanically, in order: Gate A (edge) PASS → Gate B (OOS instability) PASS → Gate C (parameter fragility) PASS (with the §A5 limitation noted) → Gate D (cost fragility) PASS → Gate E (regime failure) PASS → **Gate F (poor drawdown diversification): FAIL.**

### **Final classification: F. REJECTED — POOR DRAWDOWN DIVERSIFICATION.**

Not rescued. Not modified. Not re-parameterized to try to fix the correlation. The candidate remains exactly as originally defined; only its evidence status changes.

---

## 16. Return-stream classes

`reports/phase37_return_stream_classes.csv` — 10 classes, full structural profile each (§Track B of `reports/phase37_alternative_return_streams.md`).

## 17. Portfolio gap mapping

`reports/phase37_portfolio_gap_mapping.csv` — 0-3 scored against 6 known gaps, RESEARCH-PRIORITY ASSESSMENT ONLY throughout.

## 18. Data availability

`reports/phase37_data_availability.csv` — only Cross-sectional FX is unambiguously READY; Session-specific structures is MOSTLY READY; five classes (including the two theoretically highest-scoring) are NOT READY due to confirmed data gaps.

## 19. Overfitting risk

`reports/phase37_overfitting_risk.csv` — Event/macro-conditioned systems carries the HIGHEST overall risk (inherently discretionary, event-sparse classifier construction); Multi-asset momentum is second (most combined degrees of freedom).

## 20. Research-priority ranking

`reports/phase37_return_stream_priorities.csv`, fixed weights (independence 25%, drawdown-diversification 20%, HIGH-vol 15%, mechanism 15%, data quality 10%, researchability 5%, cost 5%, overfitting 5%): **Event/macro-conditioned (79.2) > Index-based (71.7) > Volatility-conditioned (69.2) > Cross-sectional FX (67.5) > Multi-asset momentum (66.7)** > the remaining five. **The top three by score are NOT the top three by practical readiness** — an important, honestly-reported tension, resolved in §21.

## 21. Top three return-stream classes (selected per Part 23's combined criteria: score AND data quality AND researchability AND overfitting risk)

1. **Cross-sectional FX** — WHY TEST: directly extends the project's own validated CADJPY cross-sectional finding; data confirmed READY. WHY NOT TEST: basket construction adds real degrees of freedom (MEDIUM overfitting risk) not present in a single-pair test. WHAT WOULD FALSIFY IT: a properly currency-neutral basket showing no relative-strength persistence once transaction costs and rebalancing are modeled. DATA REQUIRED: the already-available multi-pair MT5 feed — no new sourcing needed. DISTINCT FROM CURRENT BOOK: yes — the current book has zero cross-instrument ranking logic of any kind.
2. **Session-specific event structures** — WHY TEST: lowest implementation cost of any data-ready class, reusing `core/news_calendar.py`. WHY NOT TEST: calendar effects are numerous and easy to data-mine without tight pre-registration. WHAT WOULD FALSIFY IT: a pre-registered, small set of specific recurring events showing no OOS edge once the same battery Track A just applied is run. DATA REQUIRED: existing price feed + a quick depth audit of the calendar feed (not yet performed). DISTINCT FROM CURRENT BOOK: yes — event-conditioned, not generic-session-conditioned, unlike every strategy tested in Phases 30/33/35/37 so far.
3. **Event/macro-conditioned systems** (longer-horizon, infrastructure-first) — WHY TEST: highest theoretical priority score, directly targeting Phase 32's #2 priority (the exact dimension AUDUSD Monday LONG just failed on). WHY NOT TEST (yet): no confirmed risk-on/risk-off classifier data source exists in this project — Phase 31 already found this gap independently. WHAT WOULD FALSIFY IT: even a simple, pre-registered classifier failing to show any differential drawdown behavior once built. DATA REQUIRED: a macro/sentiment data source not currently in this project's toolchain — the explicit prerequisite. DISTINCT FROM CURRENT BOOK: yes, maximally — no current strategy has any macro-regime awareness.

## 22. Continue vs. change research architecture

The architecture itself (frozen pre-registration, mechanical classification, full disclosure) remains sound, per Phase 36's own audit, reconfirmed by this phase's clean AUDUSD reproduction and mechanical F-classification. **No architecture change is recommended** — only a continued shift in *what* is searched, consistent with Phase 36.

## 23. Recommended Phase 38 direction

**Cross-sectional FX and Session-specific event structures**, both immediately pre-registerable with existing data; Event/macro-conditioned systems flagged as a longer-horizon infrastructure project. Full reasoning: `reports/phase37_research_direction.md`.

## 24. Limitations

- The parameter-robustness test on AUDUSD Monday LONG (§7) is structurally weaker than the threshold-gated tests applied to Phase 33/35's candidates, since this candidate has no trade-selection parameter — disclosed, not glossed over.
- The drawdown-correlation classification (§11) rests on only 9 overlapping observation days — just above the pre-registered 8-day floor, not a large sample.
- The 2025/2026 historical-regime figures (§10) overlap with the candidate's own original discovery OOS window and are not fully independent replication for those two sub-periods.
- Track B's priority scores (§20) are a structural/qualitative assessment, not a profitability forecast — repeated here as the phase's own governing evidence-labeling rule, not merely a footnote.

## 25. Final verdict

### Answers to Part 28's sixteen questions

1. **Does AUDUSD Monday LONG reproduce?** Yes, exactly (84 trades, PF 3.070).
2. **OOS sub-period consistency?** Yes, PASS.
3. **±20% parameter perturbation?** Yes, stable (with the disclosed §7 limitation).
4. **2x cost stress?** Yes, PF 2.647.
5. **Multiple historical regimes?** Yes, positive across all five (2019-2026).
6. **HIGH volatility?** Yes, STRONG — its best regime.
7. **Drawdown diversification?** **No — CORRELATED (0.742 vs. 0.228), the decisive failure.**
8. **Improves portfolio-level behaviour?** Increases return; modestly deepens drawdown — consistent with #7.
9. **Sufficiently sampled?** Mostly yes; the drawdown-correlation sample (9 days) is the thinnest input.
10. **Final classification?** **F. REJECTED — POOR DRAWDOWN DIVERSIFICATION.**
11. **Classes most directly addressing portfolio gaps?** Event/macro-conditioned and Volatility-conditioned systems (by score).
12. **Best data quality?** Cross-sectional FX (READY), Session-specific structures (MOSTLY READY).
13. **Lowest overfitting risk?** Commodity-based, Session-specific, Cross-sectional FX (all MEDIUM or better).
14. **Three classes for future research?** Cross-sectional FX, Session-specific structures, Event/macro-conditioned systems (infrastructure-first).
15. **Should Phase 38 remain in FX technical strategies?** No — as a primary focus.
16. **Move into a different return-stream class?** Yes, per §21/§23.

### Overall outcome

Track A closes AUDUSD Monday LONG's outstanding validation gap with a definitive, mechanically-applied **F** classification — the project's strongest FX-technical candidate does not clear the bar. Track B confirms and sharpens Phase 36's direction: **expand into Cross-sectional FX and Session-specific event structures next**, with Event/macro-conditioned systems marked for future infrastructure investment.

---

## Safety check confirmation

No live strategy/parameter/risk modified · no strategy paused · no 5ers configuration modified · no candidate deployed · AUDUSD Monday LONG untouched, parameters unchanged, no optimization performed · Phase 37 preregistration committed (`5da2552`) before substantive analysis and not altered since · AUDUSD reproduction completed and matched · the identical validation framework from Phases 33/35 applied to AUDUSD · no new strategy backtested · no Track B strategy designed · no portfolio optimization · no live decision made · research validator passed (and caught + fixed a real CSV-quoting error during Track B authoring) · historical boundaries respected · multiple-testing and sample-size limitations documented throughout · raw production export not committed.

---

*No live trading change authorized. Phase 38 will pre-register (not yet backtest) Cross-sectional FX and Session-specific event structures.*
