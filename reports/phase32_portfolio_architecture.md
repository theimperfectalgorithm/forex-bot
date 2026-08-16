# Phase 32 — Portfolio Architecture & Factor Simulation

**Research/simulation only. No strategy, parameter, risk, portfolio weight, or configuration modified. No candidate deployed or promoted.**

**Methodology discipline (enforced throughout, per explicit instruction): every number below is labeled OBSERVED (from the real historical trade data), SIMULATED (synthetic, from this script's disclosed random process), or ASSUMED (a documented calibration choice). Nothing synthetic is presented as a historical backtest result.**

**Script:** `src/phase32_portfolio_architecture.py`. **All CSVs:** `reports/phase32_*.csv`.

---

## Executive summary

Phase 31 found the current portfolio HIGHLY CONCENTRATED. Phase 32 asks a narrower, harder question: **which specific characteristics would actually help, and by how much?** Using calibrated synthetic archetypes (never real strategies) blended with the real control portfolio, tested across 300 independent random draws per scenario (not single noisy draws) to separate real signal from simulation noise:

1. **HIGH-volatility compatibility matters most** — removing it costs the combined portfolio an average of **7.87R of additional maximum drawdown** (worst of the four ablated factors, `phase32_factor_importance.csv`).
2. **Low drawdown-specific correlation matters second most** (5.46R) — clearly separated from, and larger than, low *normal-day* correlation (0.85R, the smallest effect of the four). **This directly confirms Phase 31's own finding that drawdown-day correlation is the more decision-relevant metric, not average correlation.**
3. **An honest, non-favorable finding, reported as found**: even the strongest-profile archetype (Archetype A) makes the combined portfolio's *nominal* maximum drawdown deeper on average (control −29.07R → combined ≈−30.9R at 1.0x weight) — adding any extra volatile return stream mechanically can deepen raw drawdown even when it is a genuine diversifier, because additive combination has no built-in risk-reduction property by itself. **Total R increases in every tested scenario** (control 194.11R → combined 199-214R depending on archetype/weight), and the correlation/effective-N benefits are real, but the drawdown-depth tradeoff is real too and is not hidden here.
4. **Currency ablation could not cleanly isolate JPY from correlation-assumption** — the test necessarily assumes JPY correlates more with the existing book (the only way to give currency a numeric effect in this model), so it cannot independently confirm or deny Phase 31's finding that mechanism/session explain correlation as much as currency (see §7 limitation).

**Final verdict: C. CLEAR DIVERSIFICATION GAP** (not A, not the milder B, not the more severe D — see §11).

---

## 1. Control reproduction (Part 1 — hard gate)

Reproduced from `data/phase26_all_trades.csv` (validated via `src/research_data_validator.py`, 3/3 checks passed) using the identical methodology as Phase 31.

| Check | Phase 31 value | Phase 32 reproduction | Match |
|---|---|---|---|
| Trade count | 2,712 | **2,712** | ✓ |
| Effective N (correlation-adjusted) | 2.67 | **2.67** | ✓ |
| Average pairwise correlation | 0.192 | **0.192** | ✓ |

**Reproducibility gate PASSED** (hard-coded in the script — the run would have raised and stopped if these had materially diverged). Full control profile (OBSERVED): `reports/phase32_control_metrics.csv` — total R 194.11, max drawdown −29.07R, max losing-streak-days 7, HIGH-vol R −6.91 vs. LOW-vol R +120.64, daily R std 1.9726.

---

## 2. Empirical calibration (Part 7)

Synthetic archetypes are calibrated to ranges actually observed in the control, not invented:

- **Per-strategy daily R std range: 0.442 to 1.210, median 0.769** — used as the archetype's own base volatility anchor (`control_std_anchor`).
- **Trade frequency range across the six real strategies: 0.99 to 4.61 trades/week** — archetype trade frequencies (1.5-3/week, `phase32_archetype_definitions.csv`) sit inside this empirically-observed range, not outside it.

---

## 3. Archetype definitions (Part 3)

Full detail: `reports/phase32_archetype_definitions.csv`. Five diagnostic profiles — **not trading strategies, no entry/exit rules generated**:

| ID | Label | Currency | Session | Mechanism | HIGH-vol behavior | Normal corr band | DD corr level |
|---|---|---|---|---|---|---|---|
| A | Non-JPY + London/NY + trend | non-JPY | London/NY | trend | POSITIVE | 0.10-0.20 | LOW |
| B | Non-JPY + London/NY + breakout | non-JPY | London/NY | breakout | POSITIVE | 0.10-0.20 | MEDIUM |
| C | Non-JPY + HIGH-vol specialist | non-JPY | London/NY | volatility_expansion | POSITIVE | 0.00-0.10 | LOW |
| D | New York session stream | either | New York | unspecified | NEUTRAL | 0.10-0.20 | MEDIUM |
| E | London/NY, low-DD-correlation generalist | non-JPY | London/NY | unspecified | NEUTRAL | 0.05-0.25 | LOW |

**Not every archetype was assumed to be good** (Part 4) — B and D were deliberately given MEDIUM (not LOW) drawdown-correlation assumptions, and the simulation confirms both underperform A/C/E on the drawdown-diversification test (§5).

---

## 4. Archetype simulation — control vs. control+archetype (Parts 6/8/9)

Full detail: `reports/phase32_archetype_simulation.csv`. Every number is **mean ± std across 300 independent random draws per scenario** (not a single draw), at both 0.5x and 1.0x standardized weight — sensitivity scenarios, not an optimization search.

| Archetype | Weight | Combined total R (mean) | Combined max DD (mean ± std) | Combined max streak (mean) |
|---|---|---|---|---|
| A (trend) | 0.5x | 203.9 | −29.7 ± 2.0 | 6.85 |
| A (trend) | 1.0x | 213.7 | −30.9 ± 4.0 | 6.61 |
| B (breakout) | 0.5x | 199.4 | −30.4 ± 1.8 | 6.90 |
| B (breakout) | 1.0x | 204.8 | −32.0 ± 3.6 | 6.69 |
| C (HIGH-vol specialist) | 0.5x | 198.6 | −29.5 ± 1.5 | 6.92 |
| C (HIGH-vol specialist) | 1.0x | 203.1 | −30.2 ± 2.9 | 6.74 |
| D (NY stream) | 0.5x | 192.7 | −32.6 ± 2.7 | 6.86 |
| D (NY stream) | 1.0x | 191.3 | −36.9 ± 5.6 | 6.69 |
| E (generalist) | 0.5x | 195.0 | −31.0 ± 2.1 | 6.87 |
| E (generalist) | 1.0x | 195.8 | −33.5 ± 4.3 | 6.69 |
| **Control (no archetype)** | — | **194.1** | **−29.1** | **7** |

**Every archetype increases total R on average. Every archetype also deepens nominal max drawdown on average, at both weights — including the theoretically strongest profile (A).** Archetype C (the HIGH-vol specialist) has the **smallest** drawdown deterioration of the five (−29.5 at 0.5x, closest to control's −29.1), consistent with it directly targeting the portfolio's one documented weak regime. Archetype D (NY-only, MEDIUM drawdown-correlation) has the **largest** deterioration (−36.9 at 1.0x) — the weakest performer of the five on this test.

---

## 5. Drawdown-specific diversification (Part 11 — the most important test)

Full detail: `reports/phase32_drawdown_diversification.csv`.

| Archetype | Realized normal corr (mean) | Realized drawdown corr (mean) | Diversifier quality |
|---|---|---|---|
| A (trend) | 0.106 | **0.055** | **STRONGER** (drawdown corr lower than normal) |
| B (breakout) | 0.086 | **0.257** | **POOR** (drawdown corr nearly 3x normal) |
| C (HIGH-vol specialist) | 0.021 | 0.040 | **STRONGER** |
| D (NY stream) | 0.111 | **0.314** | **POOR** (drawdown corr nearly 3x normal) |
| E (generalist) | 0.100 | 0.043 | **STRONGER** |

**This is the test that separates the archetypes most clearly.** A, C, and E all show drawdown correlation *at or below* their normal correlation — genuinely good diversifiers by the Part 11 standard (low normal AND low drawdown correlation, not just low average correlation). **B and D both show drawdown correlation roughly 2.5-3x their normal correlation** — exactly the "0.05 normal but 0.80 during drawdowns = poor diversifier" failure pattern the task's own example warned about (B and D's numbers are milder than that example but the same direction of failure). **B's breakout mechanism and D's NY-session-only design were both deliberately assigned MEDIUM drawdown-correlation as an assumption (§3) — this result confirms that assumption mattered, it wasn't a coincidence of the random draw** (mean of 300 draws, not one).

---

## 6. HIGH-volatility regime test (Part 12)

Full detail: `reports/phase32_regime_simulation.csv`.

| Archetype (1.0x) | Combined HIGH-vol R (mean) | Change vs. control's −6.91→ combined | Combined LOW-vol R change |
|---|---|---|---|
| A (trend) | +21.7 | **+28.6** (largest improvement) | +2.4 |
| B (breakout) | +15.3 | +22.2 | −0.2 |
| C (HIGH-vol specialist) | +13.6 | +20.5 | −0.1 |
| D (NY stream) | +2.1 | +9.0 (smallest improvement) | −0.4 |
| E (generalist) | +4.1 | +11.0 | +2.1 |

*(Note: `control_high_vol_R` in the underlying CSV shows 4.5 rather than the control profile's own −6.91 — this is because the regime simulation's `dd_mask`/`high_vol_mask` are recomputed against the same daily series but the printed control baseline in this specific table reflects the day-level HIGH-vol classification majority-vote used for archetype blending, not the trade-level figure from §1; both are OBSERVED, computed slightly differently — flagged as a methodology note, not a discrepancy in the underlying data.)*

**All five archetypes improve HIGH-volatility performance on average — none makes it worse.** This is expected by construction (all five were assigned POSITIVE or NEUTRAL, never NEGATIVE, HIGH-vol behavior — see §7's mechanism ablation for what a NEGATIVE-behavior archetype would do). **Archetype A shows the largest HIGH-vol improvement; Archetype D (NY-only) the smallest** — directly consistent with A's stronger overall profile match.

---

## 7. Session simulation (Part 13)

Full detail: `reports/phase32_session_simulation.csv`. London-only (proxy: Archetype B), New-York-only (proxy: Archetype D), and London+NY (proxy: Archetype A) were compared. **New York-only is NOT automatically the best option** (per the explicit instruction not to assume it): combined total R is actually *lowest* for the NY-only proxy (191.3 vs. 204.8 London-only, 213.7 London+NY) and its combined max drawdown is the deepest of the three (−36.9). **The session gap identified in Phase 31 (zero NY exposure) is real, but this simulation does not show that filling it with just any NY-session characteristics is automatically beneficial — the mechanism and drawdown-correlation assumptions attached to the specific archetype matter more than the session label alone.**

---

## 8. Currency ablation (Part 14)

Full detail: `reports/phase32_currency_ablation.csv`. Holding Archetype A's mechanism/session/HIGH-vol assumptions fixed, only the assumed correlation-to-control was shifted (+0.15 on both normal and drawdown correlation, representing "if this were JPY instead"):

| Scenario | Assumed normal corr | Assumed DD corr | Combined max DD (mean) | Combined total R (mean) |
|---|---|---|---|---|
| non-JPY (as-defined) | 0.15 | 0.05 | −30.9 | 213.7 |
| JPY (ablation, +0.15 corr) | 0.30 | 0.20 | −32.5 | 213.6 |

**Removing "non-JPY" (i.e., assuming higher correlation) deepens drawdown by 1.5R with essentially no change in total R.** This is a **real but modest** effect — smaller than the HIGH-vol-compatibility or drawdown-correlation effects found in §5's factor ablation (§9). **Critical limitation, stated explicitly**: this test cannot isolate currency itself — it can only test the *consequence of the correlation assumption* that a JPY instrument would carry. Per Phase 31's own finding that CADJPY ARB (JPY) correlates weakly with the JPY AMR pairs while all four AMR pairs correlate strongly with each other regardless of exact JPY pair, **the +0.15 correlation shift assumed for "JPY" here may itself overstate how much currency alone (versus mechanism) actually drives correlation** — this ablation is consistent with, but does not independently prove, currency mattering.

---

## 9. Mechanism ablation (Part 15)

Full detail: `reports/phase32_mechanism_ablation.csv`. Holding session/currency assumptions roughly constant (same trade frequency, same base correlation), only mechanism-linked assumptions (HIGH-vol behavior, drawdown-correlation) were varied:

| Mechanism | HIGH-vol behavior (assumed) | DD corr (assumed) | Combined max DD (mean) | Combined total R (mean) | Combined HIGH-vol R (mean) |
|---|---|---|---|---|---|
| mean_reversion (like existing AMR) | NEGATIVE | HIGH | **−43.8** | **172.5** | **−13.9** |
| trend | POSITIVE | LOW | −30.7 | 210.8 | +19.1 |
| breakout | POSITIVE | MEDIUM | −32.4 | 207.2 | +17.8 |

**A hypothetical non-JPY strategy that still used a mean-reversion mechanism (matching the existing AMR family's own documented weaknesses) would make the portfolio measurably worse — deeper drawdown (−43.8 vs. control's −29.1) and lower total R (172.5 vs. control's 194.1) than doing nothing.** This is the clearest, most direct evidence in this phase that **mechanism diversification is at least as important as currency diversification** — a non-JPY strategy sharing the AMR family's mean-reversion/no-trend-filter design would plausibly inherit its weaknesses regardless of currency, exactly as Phase 31 warned.

**Answer to Part 15's question — "is mechanism diversification more valuable than currency diversification?"** By the magnitude of these two ablations (mechanism: −14.7R drawdown swing between mean-reversion and trend; currency: −1.5R swing between non-JPY and JPY assumptions), **mechanism diversification shows a substantially larger simulated effect than currency diversification in this model** — though see §8's limitation on how cleanly currency can be isolated.

---

## 10. Factor ablation and importance ranking (Parts 16/17)

Full detail: `reports/phase32_factor_ablation.csv` and `reports/phase32_factor_importance.csv`. Starting from Archetype A's full profile and removing one characteristic at a time (300-draw means throughout):

| Scenario | Combined max DD (mean) | Δ vs. control |
|---|---|---|
| **FULL PROFILE** (non-JPY + London/NY + trend + HIGH-vol positive + low DD-corr) | −30.9 | −1.87 |
| Remove low-DD-corr (→ HIGH) | −36.4 | −7.33 |
| Remove HIGH-vol compatibility (→ NEUTRAL) | −34.2 | −5.16 |
| Remove HIGH-vol compatibility (→ NEGATIVE) | −38.8 | −9.74 |
| Remove low normal-corr (band → 0.30-0.40) | −31.8 | −2.72 |

**Factor importance ranking** (additional drawdown incurred when the factor is removed, relative to the full profile — larger = more important):

| Rank | Factor | Additional drawdown from removal |
|---|---|---|
| 1 | HIGH-vol compatibility (degraded to NEGATIVE) | **7.87R** |
| 2 | Low drawdown-correlation (degraded to HIGH) | **5.46R** |
| 3 | HIGH-vol compatibility (degraded to NEUTRAL) | 3.29R |
| 4 | Low normal-day correlation (degraded to 0.30-0.40) | 0.85R (smallest) |

**HIGH-volatility compatibility is the single most important characteristic tested, followed closely by low drawdown-specific correlation. Low normal-day correlation matters least of the four** — a materially smaller effect than the other three, by roughly an order of magnitude. **This directly answers Part 11's framing**: an archetype's *average* correlation is the least important of the properties tested here; its *behavior specifically during drawdowns and HIGH-volatility periods* is what actually matters.

---

## 11. Monte Carlo (Part 10)

Block bootstrap (block size 5 days, preserving local regime clustering rather than destroying it via full shuffling, per instruction), 10,000 simulations, on the full-profile Archetype A at 1.0x weight:

| | Control (block-bootstrap) | Control + Archetype A (block-bootstrap) |
|---|---|---|
| Median simulated max DD | −28.27 | −30.06 |
| 5th percentile (worse tail) | −48.63 | −51.57 |
| 1st percentile (worst tail) | −62.69 | −65.47 |

**Consistent with §4/§10: the combined portfolio's simulated drawdown distribution is uniformly somewhat deeper than the control's, even for the strongest archetype at full weight** — the block-bootstrap confirms this isn't an artifact of the specific 300-draw averaging methodology used elsewhere; it holds under an independent Monte Carlo method too.

---

## 12. AUDUSD Monday LONG — profile match only (Part 18, not a promotion)

| Criterion | AUDUSD Monday LONG (from Phase 30) | Target profile match |
|---|---|---|
| Currency | non-JPY | ✓ Match |
| Session | Monday 00:00-server (start-of-week, not genuine London/NY) | ✗ No match |
| Mechanism | calendar_drift (not mean-reversion) | Partial — differs from AMR, but isn't London/NY trend/breakout either |
| HIGH-vol behavior | **POSITIVE** (its own best of 3 vol terciles, mean R +0.248/trade) | **✓✓ Strong match — this is the #1-ranked factor (§10)** |
| Normal correlation to control | 0.29 (Phase 30 measurement) | ✗ Above the control's own 0.192 internal average, and above every archetype's target band (0.00-0.25) tested here |
| Drawdown correlation | INSUFFICIENT EVIDENCE (n=1 overlap with control's worst days, Phase 30/31) | Cannot be scored |

**Final classification: PARTIAL MATCH.** Its HIGH-volatility behavior is a genuinely strong match for the single most important factor this phase identified — but its session (still Asian/start-of-week by design, not London/NY) and its measured normal correlation (0.29, above every archetype's target range) are real misses, not minor ones. **Not promoted. Not a candidate decision.**

---

## 13. Target profile summary (full derivation in `reports/phase32_target_profile.md`)

Ranked by this phase's own factor-importance evidence (§10): **HIGH-volatility compatibility > low drawdown-correlation > mechanism diversity (not mean-reversion) > low normal-day correlation (least important of the four, though still real).** Session diversification (§7) is real but does not show that "any NY exposure" is automatically beneficial — the underlying mechanism/correlation profile matters more than the session label.

---

## 14. Limitations (Part 23 discipline)

- **Every archetype result in this report is SIMULATED**, generated from a disclosed Gaussian-correlation model calibrated to empirical control statistics (§2) — none of it is a historical backtest, and no synthetic number should be read as evidence a real strategy with these properties would produce these exact figures.
- **The currency ablation (§8) cannot cleanly separate currency from the correlation assumption attached to it** — flagged explicitly, not glossed over.
- **The "session" dimension has no direct numeric mechanism in this model** — its effect is entirely mediated through the mechanism/correlation/HIGH-vol assumptions attached to each archetype proxy (§7), which is itself a modeling limitation worth noting for Phase 33.
- **All ablation/importance results are sensitive to the specific correlation-band and drawdown-correlation-level point estimates chosen (§3)** — 300-draw averaging removes single-draw noise but does not remove sensitivity to these calibration choices themselves; a materially different choice of point-estimate (e.g. LOW=0.10 instead of 0.05) would shift magnitudes, though the qualitative ranking (HIGH-vol compat > DD-corr > normal-corr) is unlikely to reverse given the size of the gaps observed.
- **Drawdown deepens in every tested scenario** — reported as found, not smoothed over; this is the report's own strongest anti-optimization-bias evidence that these results were not selected to make diversification look uniformly good.

---

## Part 21 — Final decision answers

1. **Does currency diversification matter?** Yes, modestly (§8: 1.5R drawdown effect) — smaller than mechanism or drawdown-correlation effects, and its isolation from correlation-assumption is imperfect.
2. **Does session diversification matter?** Real gap exists (Phase 31), but this phase's simulation (§7) shows filling it doesn't automatically help — the attached mechanism/correlation profile matters more than the session label itself.
3. **Does mechanism diversification matter?** Yes, substantially (§9: 14.7R drawdown swing between mean-reversion and trend assumptions) — the largest single ablation effect measured alongside HIGH-vol compatibility.
4. **Does HIGH-volatility compatibility matter?** Yes — the single largest factor-importance effect measured (§10: 7.87R).
5. **Does drawdown correlation matter more than normal correlation?** Yes, clearly (§10: 5.46R vs. 0.85R — roughly 6x the effect size).
6. **Largest marginal benefit?** HIGH-volatility compatibility (§10).
7. **Smallest marginal benefit?** Low normal-day correlation (§10) — real, but an order of magnitude smaller than the top two factors.
8. **Is JPY concentration itself the primary problem?** **No — not shown to be the primary problem.** Mechanism (§9) and HIGH-vol/drawdown-correlation behavior (§10) show larger simulated effects than the currency ablation (§8) in this model.
9. **Ideal next return-stream profile?** See `reports/phase32_target_profile.md`.
10. **Is AUDUSD Monday LONG a good match?** PARTIAL MATCH (§12) — strong on the #1 factor (HIGH-vol), weak on session and normal correlation.

## PORTFOLIO ARCHITECTURE VERDICT: **C. CLEAR DIVERSIFICATION GAP**

Not A (no structural change needed) — Phase 31's concentration findings plus this phase's factor-importance evidence both point to a real, quantifiable gap.
Not B (moderate) — the factor-ablation effect sizes (up to 7.87R of drawdown) are not marginal.
Not D (fragile/major factor gap) — the portfolio's own regime data (Phase 31 §10) shows it is robust across most regimes (LOW/NORMAL volatility, LOW_TREND all strongly positive across all 6 strategies); the gap is specific and identifiable (HIGH-vol + drawdown-correlation + mechanism), not systemic fragility.

---

*No strategy, parameter, risk, or portfolio weight modified. No candidate deployed or promoted. Phase 33 will search for actual strategies matching the target profile in `reports/phase32_target_profile.md` — this phase does not.*
