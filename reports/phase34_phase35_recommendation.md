# Phase 34 — Phase 35 Recommendation

**Research map only. No candidate backtested in this document. Full detail: `reports/phase34_phase35_search_map.csv`, ranked via `reports/phase34_research_gaps.csv`.**

---

## The bar is unchanged

Per explicit instruction, this document does **not** propose loosening any Phase 33 gate. Every hypothesis below will be subject to the identical protocol: frozen pre-registration before backtesting, chronological TRAIN/VALIDATION/OOS split, OOS sub-half consistency check, ±20% parameter-perturbation check, cost stress to 2x, HIGH-volatility classification (with an explicit minimum-trade-count precondition, per `reports/phase34_validation_bar_audit.csv`'s recommendation), drawdown-correlation gate, and mechanism/portfolio-integration review. **Only the hypotheses themselves are new — the validation bar is not.**

---

## Recommended scope: 4 families, 5 hypotheses total

Deliberately constrained, per instruction (3-5 families, 2-3 hypotheses each — this recommendation uses fewer hypotheses per family than the maximum where a second implementation wasn't independently justified):

### Priority 1 — New York session breakout/momentum (2 hypotheses)
1. **NY-open range breakout** (USDCAD, AUDUSD) — structurally analogous to CADJPY ARB's Asian-range breakout, but timed to the NY open. Fills the single largest, most unambiguous coverage gap this phase identified (`phase34_phase33_coverage.csv`: 0 of Phase33's 2 candidates specifically targeted NY).
2. **US-data-driven momentum** (USDCAD, AUDUSD) — a structurally distinct NY-session mechanism, reducing the risk that a single implementation choice determines this family's fate the way it appears to have for both Phase 33 candidates.

### Priority 2 — London/NY overlap cross-session continuation (1 hypothesis)
3. **London-move continuation into the NY open** (USDCAD, AUDUSD, USDCHF) — the one candidate that could resolve two target-profile priorities (mechanism diversity AND session gap) in a single test.

### Priority 3 — Multi-timeframe trend, informed by the USDCAD lesson (1 hypothesis)
4. **H4-filtered H1 trend entry** (USDCAD direct retest, AUDUSD) — a specific, falsifiable response to `reports/phase34_failure_analysis.md`'s diagnosis that USDCAD's single-timeframe design asked one threshold to carry all the discrimination burden. This is not "rescuing" the rejected candidate — it is a structurally different implementation, pre-registered fresh, that happens to test the same underlying economic hypothesis with a design change motivated by a specific, documented weakness.

### Priority 4 — Non-JPY volatility expansion, informed by the XAUUSD lesson (1 hypothesis)
5. **ATR-scaled-target volatility-expansion breakout** (AUDUSD, USDCHF — deliberately avoiding gold, given the diagnosed macro/hedge correlation issue) — directly targets Phase 32's #1-ranked factor (HIGH-vol compatibility) with a design fix for the specific weakness identified in XAUUSD's fixed-multiple exit.

---

## What was deliberately excluded, and why

- **Market structure continuation**: ranked lowest in `phase34_research_gaps.csv` given the highest overfitting risk in the entire taxonomy (structure definitions are the most subjective) relative to its novelty over the mechanisms already prioritized above.
- **Pullback continuation**: a real gap, but lower-ranked than the four selected — a reasonable Phase 36+ candidate if Priority 3's multi-timeframe trend hypothesis shows promise and a complementary entry-timing variant becomes worth exploring.
- **Further calendar/statistical effects**: already substantially covered by Phase 30's 60-cell screen — a further scan of the same weekday × pair grid would not be new evidence.
- **Mean reversion on a new instrument**: explicitly deprioritized per Phase 33's own preregistration logic and Phase 32's factor-importance evidence (mechanism diversity is a top-3 factor; another mean-reversion system would not provide it).

---

## Explicit reminder for whoever runs Phase 35

- Freeze the pre-registration **before** pulling any candidate data, exactly as Phase 33 did, in a separate git commit.
- Apply the exact same sample-size floors this phase recommended (`phase34_validation_bar_audit.csv`): don't attempt a HIGH-vol classification below 10 trades in that tercile; treat an OOS sub-half disagreement in a thin sample (<40 total OOS trades) as a warning requiring corroboration, not an automatic B classification, unless the parameter-sensitivity check independently agrees.
- The two "informed by the lesson" hypotheses (Priorities 3 and 4) are pre-registered as genuinely new, differently-designed candidates — **not** as re-tests of the exact rejected XAUUSD/USDCAD implementations. If either produces a result, it must be evaluated fresh against Phase 33's full gate sequence, with no credit carried over from the rejected predecessor.

---

*No live trading change authorized. Phase 35 will perform the actual backtesting.*
