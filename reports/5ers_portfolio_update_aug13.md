# LIVE PORTFOLIO UPDATE — FINAL VERDICT

**Previous classification:** LEVEL 2 — ELEVATED BUT PLAUSIBLE

**Current classification:** **LEVEL 2 — ELEVATED, WITH ONE CORRECTED FINDING THAT MAKES THE PICTURE MODESTLY WORSE AND ONE THAT MAKES IT LESS EXTREME** (see §10; net effect does not change the decision level)

**Previous cutoff:** 2026-08-13 07:00 UTC (actual: 07:00:05 UTC, last trade in that snapshot)

**New latest trade:** entry 2026-08-13 05:00:05 UTC, exit **2026-08-13 19:12:09 UTC** (CADJPY ARB SELL, ticket 588709831)

**New trades added:** 1 new closed trade (ticket 588709831); 36 unique tickets now vs. 35 previously

**New P&L since previous cutoff:** −$15.72 (the one new trade)

**Current post-demotion P&L (properly date-scoped, entry ≥ 2026-07-31, all available data — "Period B"):** −$66.07 over 19 trades

**Current post-demotion R (Period B):** −4.32R

**Current losing streak (Period B):** 1 (the most recent trade); **max losing streak within Period B: 4**

**JPY exposure (Period B):** 78.9% of trades, 71.5% of risk, 70.5% of losing R

---

## 0. Data integrity — independently reproduced

| Check | Claimed | Reproduced |
|---|---|---|
| Row count | 72 | **72** ✓ |
| Unique tickets | 36 | **36** ✓ |
| OPEN / CLOSED | 36 / 36 | **36 / 36** ✓ |
| Missing strategy | 0 | **0** ✓ |
| New ticket 588709831 present | yes | **yes** — CADJPY ARB SELL, entry 114.204, exit 114.383, profit −$15.72, R −0.66, exit_reason SL ✓ (exact match to your stated values) |
| Latest exit | 2026-08-13 19:12:09 UTC | **2026-08-13 19:12:09 UTC** ✓ |
| R recomputation (`profit/initial_risk` vs. exported R) | — | **0 mismatches** across all 36 CLOSED trades |

All checks pass. Proceeding.

---

## 1. Reproduction of the previous forensic report (Period A repro check)

Using the **exact same population definition** as the previous report ("current six, non-PRE_DEMOTION-labeled, no additional date floor," restricted to trades closed before the previous snapshot's cutoff):

| Metric | Previous report | Reproduced from fresh export | Match |
|---|---|---|---|
| Trades | 32 | 32 | ✓ |
| Wins / Losses | 12 / 20 | 12 / 20 | ✓ |
| Win rate | 37.5% | 37.5% | ✓ |
| Total P&L | −$80.20 | −$80.20 | ✓ |
| Total R | −6.43 | −6.43 | ✓ |
| Expectancy R | −0.201 | −0.201 | ✓ |
| Profit factor | 0.513 | 0.513 | ✓ |
| Max losing streak | 9 | 9 | ✓ |
| Max drawdown (R) | −8.71 | −8.71 | ✓ |

**Exact reproduction confirmed on every metric.** The fresh production export is consistent with the previous snapshot for every trade that was already closed at that time — no historical trade's recorded outcome changed between the two pulls. Proceeding to interpret the new data with confidence in the underlying dataset's stability.

---

## 2. A structural correction found during reproduction (important — read before §3)

While reproducing the previous report's population, a scoping issue was found: **the previous report's "current six-strategy" population was never actually date-floored at the 2026-07-31 demotion.** Its `demotion_status` field only labels `GBPJPY_ARB`/`XAUUSD_ARB` trades as `PRE_DEMOTION` (they're the only strategies in the export tool's `DEMOTED_STRATEGIES` set); every other current-six strategy's trades get `N/A (not a demoted strategy)` **regardless of whether they occurred before or after 2026-07-31**. Since the current six strategies were part of the original 8-slot book from day one (only GBPJPY ARB/XAUUSD ARB were removed on 07-31), **14 of the previous report's 32 "current six" trades actually predate the demotion** — meaning they traded under the OLD `risk_scale: 1.0` regime (8-slot book), not the current `risk_scale: 0.5` (6-slot) regime the report's own narrative described.

This matters because one of those 14 pre-demotion trades is **CADJPY ARB's +$36.83 win (ticket 579709124, entered 2026-07-23)** — nearly the entire gross-win side of the previous report's CADJPY ARB assessment ("near-breakeven, PF 0.907"). Once properly restricted to trades that actually occurred under the current 6-slot/risk_scale-0.5 regime, the picture changes materially (§4).

**Your request's own period definitions (§2 of your instructions: "Period B: 2026-07-31 through latest") already anticipated exactly this correction** — this report uses your literal, properly date-floored definitions from here on, labeled Periods A/B/C, distinct from the "reproduction" population in §1 above.

- **Period A (strict):** entry_time ≥ 2026-07-31, closed before 2026-08-13 07:01 UTC — the properly-scoped equivalent of the previous snapshot.
- **Period B (strict):** entry_time ≥ 2026-07-31, all data through 2026-08-13 19:12 UTC.
- **Period C:** entry_time ≥ 2026-08-09 (the recent-deterioration window), through latest.

---

## 3. Recalculated account performance, properly scoped

| Metric | Period A (strict, ~old cutoff) | Period B (strict, full current) | Period C (Aug 9 onward) |
|---|---|---|---|
| Trades | 18 | **19** | 11 |
| Wins / Losses | 7 / 11 | 7 / 12 | 5 / 6 |
| Win rate | 38.9% | 36.8% | 45.5% |
| Total P&L | −$50.35 | **−$66.07** | −$40.68 |
| Total R | −3.66 | **−4.32** | −2.23 |
| Expectancy R | −0.203 | −0.227 | −0.203 |
| Profit factor | **0.299** | **0.245** | 0.215 |
| Max losing streak | 4 | **4** | 4 |
| Current losing streak | 0 | **1** | 1 |
| Max drawdown (R) | −3.41 | **−3.60** | −2.34 |
| Current drawdown (R) | −2.94 | **−3.60** (at trough — no recovery yet) | −2.34 |
| Largest single loss | −$12.01 | −$15.72 (the new trade) | −$15.72 |
| Largest single win | $6.27 | $6.27 | $4.40 |
| % SL / TP / Scheduled exit | 44.4 / 16.7 / 38.9 | 47.4 / 15.8 / 36.8 | 54.5 / 18.2 / 27.3 |
| Avg holding hours | 9.2 | 9.46 | 9.92 |

**The properly-scoped picture is worse on profit factor than the previous report's headline (0.513) — Period A/B PF is 0.299/0.245, not 0.513.** This is not new deterioration; it is the same historical trades, correctly excluding the ones that predate the current risk regime. The 14 excluded pre-07-31 trades were net −$29.85 themselves, but their gross-win composition (dominated by the one CADJPY ARB +$36.83 trade) had propped up the previous report's profit factor specifically. **Confirmed finding, not an estimate**: the true post-demotion, current-6-strategy-regime performance has been PF < 0.3 essentially continuously since the demotion, not the 0.513 the previous headline conveyed.

**Recovery requirement:** to return to breakeven from the current −$66.07 / −4.32R (Period B), the portfolio needs +4.32R of net gain from here, with no further drawdown extension. The account is currently AT its trough — the most recent closed trade was a loss, and current drawdown equals max drawdown (−3.60R) for Period B.

---

## 4. Strategy-level breakdown, Periods A/B/C

| Strategy | Period | Trades | Wins | Losses | WR | PF | Expectancy R | Total R | Max streak | Total P&L |
|---|---|---|---|---|---|---|---|---|---|---|
| AUDJPY AMR | B | 5 | 0 | 5 | 0.0% | 0.0 | −0.462 | −2.31 | 5 | −$27.94 |
| CADJPY AMR | B | 4 | 1 | 3 | 25.0% | 0.016 | −0.360 | −1.44 | 3 | −$17.41 |
| **CADJPY ARB** | B | **2** | **0** | **2** | **0.0%** | **0.0** | **−0.580** | **−1.16** | **2** | **−$27.73** |
| EURJPY AMR | B | 4 | 3 | 1 | 75.0% | 1.462 | +0.062 | +0.25 | 1 | +$2.95 |
| GBPJPY AMR | B | 2 | 2 | 0 | 100.0% | INF | +0.435 | +0.87 | 0 | +$10.53 |
| GBPUSD Monday | B | 2 | 1 | 1 | 50.0% | 0.166 | −0.265 | −0.53 | 1 | −$6.47 |

**CADJPY ARB — the most materially changed strategy in this update.** Properly scoped to the current risk regime, it is now **0-for-2, both trades SL losses (−0.50R on 08-11, −0.66R on 08-13)**, PF 0.0, not the previous report's "3 trades, 1 win, PF 0.907, near-breakeven" characterization — that characterization depended entirely on a pre-demotion trade. **This is a downgrade in classification confidence for CADJPY ARB** (see §11).

AUDJPY AMR remains the largest dollar/R loss contributor (−$27.94/−2.31R over 5 trades, 0% win rate) — unchanged in direction from the previous report, though the trade count within the strict post-demotion window (5) is smaller than the previous report's blended count (9).

EURJPY AMR and GBPJPY AMR remain the two healthiest current-six strategies in this window — both net positive P&L and R, consistent with the previous report.

Full per-period breakdown (including Period A and Period C rows): `reports/5ers_portfolio_update_aug13_strategy_by_period.csv`.

---

## 5. Directional analysis (Period B)

| Strategy | Direction | Trades | Wins | WR | PF | Total R | Expectancy R |
|---|---|---|---|---|---|---|---|
| AUDJPY AMR | BUY | 3 | 0 | 0.0% | 0.0 | −1.46 | −0.487 |
| AUDJPY AMR | SELL | 2 | 0 | 0.0% | 0.0 | −0.85 | −0.425 |
| CADJPY AMR | BUY | 2 | 1 | 50.0% | 0.032 | −0.70 | −0.350 |
| CADJPY AMR | SELL | 2 | 0 | 0.0% | 0.0 | −0.74 | −0.370 |
| EURJPY AMR | BUY | 2 | 1 | 50.0% | 0.689 | −0.16 | −0.080 |
| EURJPY AMR | SELL | 2 | 2 | 100.0% | INF | +0.41 | +0.205 |
| GBPJPY AMR | BUY | 2 | 2 | 100.0% | INF | +0.87 | +0.435 |
| CADJPY ARB | BUY | 1 | 0 | 0.0% | 0.0 | −0.50 | −0.500 |
| CADJPY ARB | SELL | 1 | 0 | 0.0% | 0.0 | −0.66 | −0.660 |
| GBPUSD Monday | BUY (design) | 2 | 1 | 50.0% | 0.166 | −0.53 | −0.265 |

**AUDJPY AMR: BOTH sides are losing in this properly-scoped window (BUY 0/3, SELL 0/2)** — a genuinely different picture from the previous report's finding that BUY was specifically AUDJPY's worst live bucket (there, n=4 BUY/5 SELL, both from a larger, differently-scoped population). With this small a sample (2-3 trades per side), **the evidence does not support attributing AUDJPY's weakness to either direction specifically — it currently looks uniformly weak across both, which is itself informative: it argues against a simple directional-filter narrative (BUY-only or SELL-only) explaining or fixing AUDJPY's live results.**

CADJPY AMR's SELL side remains weaker than BUY (0/2 vs 1/2) — directionally consistent with the pre-existing historical SELL-weakness finding, but n=2 per side.

EURJPY AMR's SELL side (2/2 wins) outperforming BUY (1/2) continues to be the **opposite** of the historical directional-asymmetry premise (SELL flagged as historically weaker for EURJPY) — same finding as the previous report, now with one more data point supporting it. **This remains a live-data pattern that does not match the historical directional research, reported honestly rather than forced to fit.**

---

## 6. Regime analysis (ATR tercile, Period B)

| Strategy | Regime | Trades | WR | Avg R | Total R |
|---|---|---|---|---|---|
| AUDJPY AMR | HIGH | 2 | 0.0% | −0.510 | −1.02 |
| AUDJPY AMR | LOW | 2 | 0.0% | −0.430 | −0.86 |
| AUDJPY AMR | NORMAL | 1 | 0.0% | −0.430 | −0.43 |
| CADJPY AMR | HIGH | 3 | 0.0% | −0.487 | −1.46 |
| CADJPY AMR | NORMAL | 1 | 100.0% | +0.020 | +0.02 |
| EURJPY AMR | LOW | 2 | 100.0% | +0.225 | +0.45 |
| EURJPY AMR | NORMAL | 2 | 50.0% | −0.100 | −0.20 |
| GBPJPY AMR | HIGH | 1 | 100.0% | +0.520 | +0.52 |
| GBPJPY AMR | NORMAL | 1 | 100.0% | +0.350 | +0.35 |
| GBPUSD Monday | NORMAL | 2 | 50.0% | −0.265 | −0.53 |

**Important correction to the previous report's HIGH-volatility-specific framing:** the previous report found AUDJPY AMR and CADJPY AMR losing 100% of the time specifically in HIGH-ATR trades (with LOW/NORMAL milder). In this properly-scoped, smaller window, **AUDJPY AMR is now losing uniformly across ALL three regime buckets (0% win rate in HIGH, LOW, and NORMAL alike)** — the regime-specificity of AUDJPY's weakness is **not confirmed** in this window; it looks like a broader weakness, not one confined to HIGH volatility. **CADJPY AMR's pattern is more consistent with the prior finding** — still 0% in its 3 HIGH-ATR trades, its one NORMAL trade a small win. Given n=1-3 per bucket throughout this table, none of these regime splits individually carry statistical weight — reported as **PLAUSIBLE, not CONFIRMED**, per this task's evidence-grading requirement (§10).

---

## 7. JPY concentration and clustering (Period B)

| Metric | Previous report (old population) | This update (Period B, properly scoped) |
|---|---|---|
| % trades JPY | 81.2% | 78.9% |
| % risk JPY | 74.2% | 71.5% |
| % losing R from JPY | 76.8% | 70.5% |
| Days with 2+ JPY strategies active | 9 of 14 (64.3%) | 5 of 8 (62.5%) |
| Days with 2+ JPY strategies losing together | 6 of 14 (42.9%) | 4 of 8 (50.0%) |

**JPY concentration is stable, not worsening** — all four headline ratios sit within a few points of the previous report's figures. The proportion of multi-JPY-active days where 2+ strategies lose together has, if anything, ticked slightly higher (50.0% vs 42.9%), consistent with — not contradicting — the previous finding that JPY correlation is a real contributor to this drawdown's shape. Day-by-day detail (including the new 08-13 CADJPY ARB loss day) in `reports/5ers_portfolio_update_aug13_jpy_correlation.csv`.

---

## 8. Execution quality — using only trustworthy (POST-FIX) entries where entry-price matters

Per `reports/entry_price_logging_audit.md`: the historical `entry_price` logging defect (fixed in commit `0b64c02`, 2026-08-07 19:09 UTC) affected only the **recorded** entry price — never execution, SL/TP, or P&L/R. In Period B (19 trades): **8 PRE_FIX, 11 POST_FIX, 0 UNKNOWN.** The new ticket (588709831) is **POST_FIX** (entry_price 114.204 is genuine, confirmed via `positions_get()`), so it is fully trustworthy for any entry-price-anchored analysis, not just the R/PnL/exit-reason fields that were always reliable.

**Spread/implied-SL-distance analysis (uses `spread_pips` and an implied SL distance — both independent of the entry_price bug, per the entry-price audit's §10/§11 finding — so this table is valid regardless of PRE/POST-FIX status):**

| Bucket | Trades | Win rate | Avg R |
|---|---|---|---|
| <10% | 18 | 38.9% | −0.221 |
| 10–20% | 1 | 0.0% | −0.340 |

No material execution/cost signature — consistent with the previous report's finding. The single 10-20% trade is too small a sample (n=1) to draw any conclusion from.

---

## 9. Exit reason and holding time (Period B)

| Strategy | Exit reason | Count | Win rate | Avg R | Avg hold (h) |
|---|---|---|---|---|---|
| AUDJPY AMR | SL | 4 | 0.0% | −0.492 | 5.89 |
| AUDJPY AMR | SCHEDULED_STRATEGY_EXIT | 1 | 0.0% | −0.340 | 9.00 |
| CADJPY AMR | SL | 2 | 0.0% | −0.620 | 6.55 |
| CADJPY AMR | SCHEDULED_STRATEGY_EXIT | 2 | 50.0% | −0.100 | 8.12 |
| CADJPY ARB | SL | 2 | 0.0% | −0.580 | 22.30 |
| EURJPY AMR | TP | 2 | 100.0% | +0.350 | 4.28 |
| EURJPY AMR | SL | 1 | 0.0% | −0.530 | 3.71 |
| EURJPY AMR | SCHEDULED_STRATEGY_EXIT | 1 | 100.0% | +0.080 | 4.25 |
| GBPJPY AMR | TP | 1 | 100.0% | +0.350 | 4.46 |
| GBPJPY AMR | SCHEDULED_STRATEGY_EXIT | 1 | 100.0% | +0.520 | 6.25 |
| GBPUSD Monday | SCHEDULED_STRATEGY_EXIT | 2 | 50.0% | −0.265 | 23.00 |

**CADJPY ARB's exit reason is 100% SL in this window (both losses)** — notably, this strategy's design intends a 2:1 reward-to-risk breakout with TP as the primary win mechanism (per its historical validation), so two consecutive SL exits (no TPs) is the specific pattern behind its downgraded classification (§11), not a scheduled-exit artifact. No other strategy shows a scheduled-exit-driven pattern materially different from the previous report.

---

## 10. Monte Carlo — updated, same methodology, properly-scoped population

Same two methods as the previous report (pooled / strategy-aware, 20,000 simulations each, drawn from `data/phase26_all_trades.csv`'s historical R-multiple pools), now applied to **Period B's 19-trade properly-scoped population**:

| Method | Metric | p1 | p25 | p50 | p75 | p99 | **Observed** | **Observed percentile** |
|---|---|---|---|---|---|---|---|---|
| Pooled | PF | 0.40 | 0.88 | 1.22 | 1.77 | 5.32 | **0.245** | **0.075th** |
| Pooled | Win rate % | 42.1 | 57.9 | 68.4 | 73.7 | 89.5 | **36.8** | **0.14th** |
| Pooled | Max DD (R) | −8.20 | −3.94 | −2.77 | −2.12 | −1.06 | **−3.60** | **31.2th** |
| Strategy-aware | PF | 0.39 | 0.88 | 1.25 | 1.80 | 5.52 | **0.245** | **0.070th** |
| Strategy-aware | Win rate % | 42.1 | 57.9 | 68.4 | 73.7 | 89.5 | **36.8** | **0.22th** |
| Strategy-aware | Max DD (R) | −8.08 | −3.88 | −2.73 | −2.11 | −1.06 | **−3.60** | **30.2th** |
| Strategy-aware | Max losing streak | 1 | 2 | 2 | 3 | 6 | **4** | **86.6th** |

(Full CSV: `reports/5ers_portfolio_update_aug13_monte_carlo.csv`.)

**Two findings, pulling in opposite directions — reported as found, not smoothed into a single narrative:**

1. **PF and win rate are now MORE extreme than the previous report's figures** (0.075th/0.14th percentile here vs. ~1.1th/0.02-0.03th before) — properly scoping the population to only post-demotion trades removed the pre-demotion win that had been propping up the profit factor, so the tail-rareness of the observed PF is genuinely worse under correct scoping, not an artifact of adding the new trade specifically.
2. **The max losing streak is LESS extreme than before** (4 trades, ~87th percentile, vs. the previous report's 9-trade streak at the 99.9th percentile) — because that earlier 9-trade streak count was itself an artifact of blending pre- and post-demotion trades into one continuous streak; once properly scoped to only the post-demotion regime, the true max streak is 4, a much more ordinary result.

**Methodology limitation restated explicitly, per instruction, not silently carried over:** both Monte Carlo methods resample trades **independently (i.i.d.)**. They have no mechanism to reproduce the same-day cross-strategy correlation documented in §7 (JPY strategies losing together on the same days). This means the *streak* statistic in particular should be read cautiously in either direction — a real portfolio where 2+ correlated JPY strategies tend to lose on the same calendar day will show *shorter* nominal trade-count streaks than an equivalent independent-draw process would, precisely because correlated losses "bunch" on fewer calendar days rather than stretching across more consecutive trades. **This methodology was not changed to produce a more favorable result — it is identical to the previous report's, and this caveat cuts both ways (it affects the streak metric in both directions depending on how you frame it), not in whichever direction happens to look better.**

---

## 11. Change-point check

**Is the recent period (Aug 9 onward) statistically or structurally unusual relative to what was already expected — not "can parameters explain it," per your explicit framing?**

- **Relative to each strategy's validated historical distribution:** already tested via the Monte Carlo above (§10) — PF/win-rate sit in the extreme tail (<1st percentile) of the historical-pool-derived distribution. **Strong evidence** that the aggregate portfolio result is unusual relative to what the validated historical strategies would typically produce, even accounting for strategy-mix weighting.
- **Relative to the pre-demotion live period:** the pre-demotion period (before 2026-07-31, not part of Periods A/B/C) is not directly comparable on a P&L basis because `risk_scale` doubled between periods (1.0 pre-demotion vs. 0.5 post) — R-multiples are risk-normalized and thus more comparable, but the pre-demotion period includes GBPJPY ARB's severe -1.65R/-1.38R/-1.20R losses (already investigated and actioned via the 07-31 demotion) which are not part of the current six-strategy book at all. **No clean apples-to-apples comparison is possible here without conflating a strategy that's since been removed** — this specific comparison is **INSUFFICIENT EVIDENCE**, not attempted further.
- **Relative to the immediately-preceding post-demotion period (Period A vs. Period C):** Period A (18 trades, PF 0.299, expectancy −0.203R) and Period C (11 trades, PF 0.215, expectancy −0.203R) are **very similar in expectancy and nearly identical in magnitude of severity** — Period C is not a sharp new escalation relative to Period A; it reads as a continuation of the same regime-level pattern already underway before 08-09, not a fresh deterioration. **Confirmed finding** (both periods are drawn from the same underlying computed metrics, directly comparable): the "recent deterioration window" the task asked about does not show a materially different expectancy than the weeks immediately preceding it.

---

## 12. Distinguishing the problem type (explicit, per instruction)

| Candidate explanation | Evidence found | Confidence |
|---|---|---|
| **Strategy deterioration** | CADJPY ARB is 0-for-2 post-demotion with two real SL losses (no TP hits) — a genuine pattern shift from its historical 2:1-breakout design, though n=2. AUDJPY AMR remains the largest loss contributor, now losing across all regime buckets and both directions in this window (n=5). | **PLAUSIBLE for CADJPY ARB specifically (elevated concern, not confirmed given n=2). INSUFFICIENT EVIDENCE for AUDJPY AMR beyond what was already known — its weakness is confirmed present but not newly proven to be edge failure vs. a bad stretch.** |
| **Portfolio concentration** | JPY exposure stable at ~71-79% of trades/risk (§7); no new evidence of worsening concentration since the previous report. | **CONFIRMED as a standing structural feature of the current 6-slot book (not new), unchanged severity.** |
| **Regime mismatch** | AUDJPY's HIGH-vol-specific pattern from the previous report did NOT cleanly replicate in this smaller window (losses now spread across all regime buckets); CADJPY AMR's HIGH-vol pattern did replicate. | **PLAUSIBLE, mixed — confirmed for CADJPY AMR, not confirmed (though not contradicted either, given tiny n) for AUDJPY AMR in this specific window.** |
| **Trade correlation** | 50% of multi-JPY-active days in Period B saw 2+ JPY strategies lose together (§7) — slightly higher than the previous report's 42.9%. | **STRONG EVIDENCE of observed clustering (not formally "statistically significant" — n=8 days is too small for that claim), consistent with and slightly reinforcing the previous finding.** |
| **Normal sampling variance** | The corrected Monte Carlo (§10) puts PF/win-rate in the extreme tail (<1st percentile), which argues against "purely normal variance" as a complete explanation — but the *streak* result is now much less extreme (87th percentile) than previously reported, which argues the account is not in as unprecedented a position as the prior streak-based framing suggested. | **MIXED — genuinely both directions of evidence exist; this is the single most nuanced finding in this update, not force-resolved into either "it's fine" or "it's broken."** |
| **Execution problems** | No material cost/spread signature found (§8); PRE/POST-FIX distinction confirmed the entry-price bug never affected execution itself (per the separate entry-price audit). | **CONFIRMED ABSENT as a material contributor.** |

**Do not attribute the portfolio-level loss to individual strategy failure without evidence — honored here**: only CADJPY ARB shows a strategy-specific pattern change strong enough to flag by name (§4/§9), and even that is explicitly graded PLAUSIBLE, not CONFIRMED, given n=2.

---

## 13. Does the previous LEVEL 2 conclusion still hold?

Per your options:

**Answer: B. Previous conclusion weakened but still plausible** — closest fit, with an important asterisk.

- The **profit-factor picture is genuinely worse than previously portrayed** once properly scoped (§3/§10) — this weakens confidence that the account's poor showing is "just a rough patch," since even the corrected, more-defensible population sits deeper in the historical distribution's tail than the previous headline suggested.
- **But the losing-streak evidence — the single most dramatic statistic in the previous report (9 trades, 99.9th percentile) — turns out to have been partly a scoping artifact.** Properly isolated to the current risk regime, the real streak is 4 trades at the ~87th percentile: elevated, but not the near-unprecedented event the previous framing implied.
- **JPY correlation and CADJPY AMR's regime-specific weakness both replicate cleanly in the new data — reinforcing, not weakening, those parts of the previous explanation.**
- **AUDJPY's regime-specificity did not cleanly replicate** in this smaller window (losses now spread across all buckets, not concentrated in HIGH-vol) — this specific secondary explanation from the previous report is weaker than before, though AUDJPY's overall weakness remains real.
- **Net effect: still LEVEL 2, not LEVEL 3.** No single strategy has accumulated the kind of unambiguous, sample-size-adjusted evidence of edge failure that would justify escalating past "elevated but plausible" — but CADJPY ARB has moved from "healthy, monitor" to "specifically flagged, closest to the evidence threshold of any current strategy" (§11 below).

---

## 14. Final decision

### **C. FURTHER VALIDATION REQUIRED BEFORE ANY CHANGE**

Not A (continue unchanged) — the corrected Monte Carlo tail result (§10) and CADJPY ARB's 0-for-2 post-demotion pattern (§4) are real enough to warrant continued active tracking, not silence.

Not B alone (enhanced monitoring, implying no elevated concern) — undersells CADJPY ARB's specific pattern change, which is closer to a concrete evidence threshold than any strategy was in the previous report.

Not D (pause a specific strategy) — CADJPY ARB's n=2 is far too small to justify a pause recommendation; both of its losses are individually unremarkable (−0.50R, −0.66R, both within its own design's normal loss range) and the pattern (0 TPs so far) could easily reverse with the very next trade. Pausing now would be reacting to two data points.

Not E (portfolio-level risk reduction) — the portfolio-construction observation (JPY concentration, §7/§12) is unchanged in severity from the previous report, which already reached C, not E, on the same evidence; nothing here raises it further.

**Evidence threshold that would move CADJPY ARB specifically toward D:** a third consecutive post-demotion SL loss with no intervening TP hit would put it at 0-for-3, a considerably stronger (though still small-sample) signal than the current 0-for-2 — this is offered as an explicit trigger point for the *next* check, not an action taken now.

**This does not override or supersede the existing 2026-08-25 AMR trend-regime checkpoint** (PROJECT_REPORT.md §6) for AUDJPY/CADJPY AMR — that remains the designated decision point for those two strategies specifically.

---

## Answers to your seven questions (plain English)

**1. Are we still looking at a difficult but plausible sample?** Yes, but "plausible" now needs a caveat: the corrected profit-factor/win-rate percentiles (§10) are more extreme than previously reported, so "difficult but plausible" is accurate on the streak dimension and less comfortably accurate on the PF/win-rate dimension. Both are true at once — this isn't a contradiction, it's what happens when you correctly separate two different statistics that had been blended together before.

**2. Are the recent losses primarily JPY-correlated?** Substantially yes — JPY concentration (78.9% of trades) and same-day multi-JPY-losing clustering (50% of active days) are stable-to-slightly-higher than the previous report, and remain the single most consistent structural explanation across both reports.

**3. Are AUDJPY/CADJPY still failing specifically in known weak regimes?** CADJPY AMR — yes, cleanly replicated (0% win rate in its 3 HIGH-ATR trades this window). AUDJPY AMR — not cleanly replicated this window; its losses now span all three regime buckets rather than concentrating in HIGH-vol, so the regime-specific explanation is weaker for AUDJPY specifically than the previous report suggested, even though AUDJPY's overall weakness is unchanged.

**4. Is there evidence that EURJPY/GBPJPY/CADJPY ARB/GBPUSD Monday Drift are deteriorating?** EURJPY AMR and GBPJPY AMR remain net positive in this window — no deterioration evidence for either. GBPUSD Monday is unchanged (1W/1L, too few trades to say anything). **CADJPY ARB is the one exception** — its properly-scoped picture (0-for-2, both SL, no TPs) is a genuine downgrade from the previous report's characterization, though still far too small a sample (n=2) to call it deterioration with confidence.

**5. Has the probability of strategy failure increased materially?** For the portfolio overall: modestly, via the corrected PF/win-rate Monte Carlo tail (§10) — a genuine, non-manufactured finding. For any single strategy: only CADJPY ARB moved meaningfully, and even there "increased probability of concern" is more accurate than "increased probability of failure" given n=2.

**6. Has the previous LEVEL 2 conclusion changed?** No — it remains LEVEL 2, but with reduced confidence on the PF/win-rate dimension (worse than portrayed) and increased confidence on the losing-streak dimension (less extreme than portrayed). Net: still ELEVATED BUT PLAUSIBLE, per §13's "B" classification.

**7. What should we do next?** Nothing implemented from this report. Recommended (not implemented): (a) track CADJPY ARB specifically toward the 0-for-3 threshold noted in §14, (b) carry the AUDJPY/CADJPY AMR regime-weakness question into the already-scheduled 2026-08-25 checkpoint as planned, (c) if a future evidence-update is run again, re-verify the demotion-date scoping correction from §2 is preserved (it is now built into `src/phase28_5ers_portfolio_update_aug13.py`'s period definitions) rather than reverting to the unscoped population that produced the previous report's optimistic PF figure.

---

*No strategy, parameter, risk setting, filter, pair, or configuration changed. All recommendations in §14 and above are analytical only and were not implemented. Reproducible via `python src/phase28_5ers_portfolio_update_aug13.py` (imports `src/phase27_5ers_current_portfolio_forensic.py` for shared methodology).*
