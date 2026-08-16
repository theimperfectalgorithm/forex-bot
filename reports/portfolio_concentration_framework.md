# Portfolio Concentration Framework — JPY Exposure & Risk-Source Analysis

**Purpose:** determine whether the six-strategy portfolio is genuinely diversified or is "six strategy names expressing a small number of underlying risk factors," and whether that concentration is materially contributing to the current drawdown. Diagnostic only — no pairs added, no strategies modified.

**Data:** `reports/5ers_trade_export.csv` (fresh production export, verified 72 rows/36 tickets), post-demotion window (entry ≥ 2026-07-31, 19 trades, current six strategies only), analysis via `src/phase29_live_scorecard.py`.

---

## 1. Are six strategy names actually six independent risk sources?

**No — not by underlying currency factor.** Of the six current strategies:

| Strategy | Pair | Underlying currency exposure |
|---|---|---|
| GBPJPY AMR | GBPJPY | JPY |
| EURJPY AMR | EURJPY | JPY |
| AUDJPY AMR | AUDJPY | JPY |
| CADJPY AMR | CADJPY | JPY |
| CADJPY ARB | CADJPY | JPY |
| GBPUSD Monday | GBPUSD | Non-JPY |

**5 of 6 strategies carry JPY exposure; only GBPUSD Monday (1 of 6) is JPY-free.** Two strategies (CADJPY AMR and CADJPY ARB) trade the *same pair* with different mechanics — genuinely different signal logic (mean-reversion vs. breakout) and different sessions, but both still fully exposed to any CADJPY-wide move.

**Session/time diversification is also thin:** all four AMR strategies trade the identical 00:00–07:00 UTC Asian session window. Only CADJPY ARB (07:00-09:00 breakout) and GBPUSD Monday (Monday-specific) trade outside that window.

**Quantified, post-demotion window (19 trades):**

| Metric | Value |
|---|---|
| % of trades JPY-linked | **78.9%** |
| % of risk JPY-linked | **71.5%** |
| Days with 2+ current-six strategies active | **6 of 8 (75.0%)** |
| Days with 2+ current-six strategies losing together | **4 of 8 (50.0%)** |

---

## 2. Is JPY concentration actually contributing to the observed drawdown?

**Evidence supports "observed clustering," not a formally significant statistical claim** (n=8 trading days is too small for that) — worded exactly this way per the project's established convention.

**Day-by-day, post-demotion:**

| Date | Strategies active | Strategies losing | Total R that day |
|---|---|---|---|
| 2026-08-02 | 3 | 3 | −1.96 |
| 2026-08-03 | 3 | 2 | −0.12 |
| 2026-08-05 | 1 | 1 | −0.34 |
| 2026-08-06 | 1 | 0 | +0.33 |
| 2026-08-09 | 3 | 2 | −0.84 |
| 2026-08-11 | 4 | 3 | −1.20 |
| 2026-08-12 | 2 | 0 | +0.10 |
| 2026-08-13 | 2 | 1 | −0.29 |

**6 of 8 days had 2 or more strategies trading simultaneously; on 4 of those 6, at least 2 strategies lost on the same day.** The two worst single days (08-02 at −1.96R, 08-11 at −1.20R) are both multi-strategy, multi-loss days — together they account for roughly 73% of the entire post-demotion drawdown (−3.16R of −4.32R total). **This is the single strongest piece of evidence that concentration, not any one strategy's individual edge, explains the shape (though not necessarily the total magnitude) of this drawdown.**

---

## 3. Are multiple JPY strategies effectively expressing the same factor?

**Partially yes, with an important mechanical distinction.** All four AMR strategies (GBPJPY/EURJPY/AUDJPY/CADJPY) share:
- The same underlying mechanic (M15 z-score mean-reversion vs. SMA20)
- The same trading session (00:00–07:00 UTC)
- The same force-flat exit time (07:00 UTC)
- No higher-timeframe trend filter (by design, per `PROJECT_REPORT.md` §5's AMR root-cause note)

This means a genuine multi-day JPY-wide trending move (as already documented in `PROJECT_REPORT.md` §5's "early-Aug AMR trending-JPY losing cluster") can plausibly push **several AMR pairs into losing SELL or BUY signals simultaneously**, since they're all reacting to correlated JPY-cross price action with the same blind spot (no trend awareness) at the same time of day. CADJPY ARB, despite also being CADJPY-linked, uses a different session (07:00-09:00 breakout, not the Asian mean-reversion window) and different mechanics — so it is JPY-correlated but not *mechanically* redundant with the AMR pairs the way the four AMR pairs are with each other.

**This is a real, previously-flagged mechanism** (not a new discovery in this document) — the four AMR pairs are closer to "one mean-reversion strategy applied to four correlated instruments in the same session" than to four independent edges.

---

## 4. Would non-JPY strategies genuinely diversify the portfolio?

**Only GBPUSD Monday currently tests this**, and its evidence is too limited (n=2 post-demotion, per the scorecard) to draw a conclusion about whether non-JPY diversification actually helps in live conditions. What can be said from the historical record:

- GBPUSD Monday's historical correlation to the JPY-cross strategies has not been directly computed in this or any prior phase (flagged as **NOT AVAILABLE** — would require a dedicated cross-strategy correlation study using the full historical trade-level data, not attempted here since it would constitute new research, out of scope for this diagnostic task).
- Structurally, GBPUSD Monday differs from every JPY strategy on every dimension that plausibly drives the JPY cluster's correlated losses: different underlying currency pair (no JPY leg at all), different session (Monday-specific, not the Asian 00:00-07:00 window), different mechanic (drift/momentum, not mean-reversion), and a different force-flat time (21:00 UTC Monday vs. 07:00 UTC daily). **This is suggestive of genuine diversification potential, not proof of it** — the mechanism for correlation among the JPY strategies (§3) simply doesn't apply to GBPUSD Monday by construction, but this has not been empirically confirmed via a correlation statistic.

---

## 5. What characteristics would a useful non-JPY strategy need?

Derived from what §1-4 identified as the source of the current concentration risk, not from optimizing for the current losses:

1. **A different underlying currency pair with no JPY leg** — this is the most basic and important criterion; even a second GBPUSD or EURUSD strategy would only help if it doesn't share GBPUSD Monday's own session/mechanic (see #2).
2. **A different session window than 00:00-07:00 UTC** — to avoid the same-session correlation mechanism identified in §3.
3. **A mechanic that isn't pure mean-reversion without a trend filter** — since that specific combination (§3) is the identified mechanical driver of the AMR cluster's correlated losses; a trend-following or breakout mechanic, or a mean-reversion mechanic with regime-awareness, would structurally differ from the failure mode already observed.
4. **Already-existing candidates in the research pipeline that meet these criteria** (per `PROJECT_REPORT.md` §4's "Candidates awaiting a build decision" and "Research directions" lists — cited for reference only, not endorsed or recommended for action here): the NZDJPY cross-asset-momentum candidate (phase 10/10b) is itself still JPY-linked and would not address the JPY concentration problem; the commodity-bloc crosses (AUDCAD, NZDCAD, AUDNZD) and gold London/NY-session research direction are the two listed candidates that would actually meet criterion #1.

**This document does not recommend building or researching any specific non-JPY strategy now** — it only specifies what evidence and characteristics would make one useful, per the task's explicit instruction not to recommend adding random non-JPY pairs.

---

## 6. Portfolio-level scorecard summary (post-demotion, 19 trades)

| Metric | Value |
|---|---|
| Total portfolio R | −4.32 |
| Portfolio drawdown (R) | −3.60 (current = max; no recovery yet) |
| Portfolio PF | 0.245 |
| Portfolio max losing streak | 4 |
| JPY exposure (trades / risk) | 78.9% / 71.5% |
| Multi-strategy losing days | 4 of 8 (50.0%) |
| Contribution by strategy family | AMR: −5.29R combined (4 strategies) vs. ARB+Monday: −1.69R combined (2 strategies) — AMR carries the larger share of both trade count and loss, consistent with its larger allocation (4 of 6 slots) |
| Contribution by currency factor | JPY strategies: the large majority of both trade count and losing R (§1); GBPUSD Monday: a small minority |

---

*No strategies added, modified, or researched further in this document. This is a diagnostic characterization of existing concentration, produced to inform (not replace) future research decisions.*
