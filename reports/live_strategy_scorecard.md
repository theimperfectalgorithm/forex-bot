# Live Strategy Scorecard — AEGIS / 5ers Live Portfolio

**Purpose:** a formal live-validation decision framework, not an optimization exercise. Answers, per strategy: what evidence exists, what's missing, and exactly what future evidence would move it to CONTINUE, REVIEW, REDUCE, or PAUSE. Companion documents: `reports/live_strategy_decision_rules.md` (threshold derivation), `reports/portfolio_concentration_framework.md` (JPY/factor concentration).

**No strategy, parameter, risk, pair, or configuration was modified. No deployment performed.**

---

## 0. Safety check — completed before analysis

- [x] Fresh production export used (`reports/5ers_trade_export.csv`)
- [x] 72 rows verified
- [x] 36 unique tickets verified
- [x] 36 CLOSED verified
- [x] 36 OPEN verified
- [x] Current window properly date-scoped (entry ≥ 2026-07-31 00:00 UTC for "post-demotion")
- [x] Pre-demotion trades excluded from current baseline (kept as a separate window, not mixed in — see §1C)
- [x] No strategy modified · [x] No parameters modified · [x] No risk modified · [x] No deployment performed
- [x] No raw production data committed (only derived CSVs/reports below)
- [x] No arbitrary thresholds introduced without historical justification — every threshold in `live_strategy_decision_rules.md` traces to a number already in `reports/current_6_strategy_revalidation.csv` or `PROJECT_REPORT.md`, written before this task

All checks pass.

---

## 1A. Pre-live evidence (reconstructed from the repository, not memory)

Source: `reports/current_6_strategy_revalidation.csv` (EXP-096..111), `pairs/*.yaml` config comments, `PROJECT_REPORT.md` §3-4, git commit `e1f3ec9` (2026-07-05, "Configs: deploy Book B+").

| Strategy | Discovery phase | OOS trades | OOS PF | OOS expectancy R | Historical max DD (R) | Historical max losing streak | Known regime weakness | Known cost sensitivity | Pre-live classification |
|---|---|---|---|---|---|---|---|---|---|
| GBPJPY AMR | phase 3/3b (2026-07-05) | 127 | 2.101 | 0.287 | −7.28 | 5 | None — positive across all vol/trend regimes | ROBUST (PF 1.21 @ 2x spread) | **A. STRONG REVALIDATION** |
| EURJPY AMR | phase 3/3b | 236 | 1.343 | 0.104 | −17.23 | 4 | MIXED — SELL leg independently net-losing (PF 0.836) | FAIL — cost-fragile (PF 0.89 @ 2x, near-negative @ 1.5x) | **C. PROMISING BUT INSUFFICIENT** |
| AUDJPY AMR | phase 3/3b | 205 | 1.144 | 0.050 | −10.25 | 5 | FAIL — HIGH-vol regime net-losing (PF 0.826); SELL leg net-losing (PF 0.706); weakening trend into 2026 | FAIL — cost-fragile (PF 0.84 @ 2x, already net-losing @ 1.5x) | **C. PROMISING BUT INSUFFICIENT** (weakest OOS PF of the AMR family at deployment) |
| CADJPY AMR | phase 6 (2026-07-05) | 189 | 1.305 | 0.092 | −15.42 | 6 | FAIL — HIGH-vol regime net-losing (PF 0.831); SELL leg net-losing (PF 0.763); vol×trend interaction | FAIL — worst cost-fragility in book (net-losing @ 1.5x) | **D. WEAK / PROVISIONAL** (weakest in book, no falsification of edge) |
| CADJPY ARB | phase 6 (2026-07-05) | 64 | 1.519 | 0.248 | −14.87 | 10 | FAIL — HIGH-vol/HIGH-trend regime net-losing | ROBUST (PF 1.16 @ 2x) | **B. ACCEPTABLE BUT MONITOR** |
| GBPUSD Monday | phase 7/8 (2026-07-05) | 53 | 2.929 | 0.177 | −2.23 | 4 | None — positive across all vol/trend regimes tested | ROBUST — best in book (PF 1.98 @ 2x) | **A. STRONG REVALIDATION** (smallest total sample: 154 trades, ~52/yr — sample-size caveat, not a performance flag) |

**Deployment provenance:** all six strategies deployed together as "Book B+" (commit `e1f3ec9`, 2026-07-05); AMR pairs carry an explicit YAML comment ("DEMO FORWARD-TEST candidate... forward-test 2-3 months on demo before any challenge use") that was overridden by the later decision to run the identical book on both demo and the 5ers challenge account from day one (`PROJECT_REPORT.md` §5) — this override is a standing project decision, not something re-litigated here.

---

## 1B/1C. Current live evidence, properly windowed (fresh export, 33 current-six trades total)

Population: `strategy_norm` ∈ current six, from `reports/5ers_trade_export.csv`, deduplicated to unique CLOSED trade_id (33 CLOSED trades for the current six across all time — the fresh export's 36 CLOSED total minus 3 pre-demotion GBPJPY ARB trades, which are excluded entirely as they belong to a demoted strategy not part of this scorecard).

| Window | Definition | Trades |
|---|---|---|
| Entire live history | All current-six trades, any date | 33 |
| **Pre-demotion** | entry_time < 2026-07-31 | 14 |
| **Post-demotion** (the current live-validation baseline) | entry_time ≥ 2026-07-31 | **19** |
| Recent (Aug 9-13) | entry_time ≥ 2026-08-09 | 11 |

**Pre-demotion and post-demotion are never mixed in this document's per-strategy verdicts** — every "current status" and "live n" figure below uses the post-demotion window only, consistent with `reports/5ers_portfolio_update_aug13.md`'s finding that blending the two produced a materially misleading picture in an earlier phase of this project.

### Per-strategy, post-demotion window:

| Strategy | Trades | Wins | Losses | WR | Gross win | Gross loss | Net P&L | Total R | Expectancy R | PF | Avg win | Avg loss | Max loss | Max streak | Avg hold (h) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GBPJPY AMR | 2 | 2 | 0 | 100.0% | $10.53 | $0 | +$10.53 | +0.87 | +0.435 | INF | $5.26 | n/a | n/a | 0 | 5.36 |
| EURJPY AMR | 4 | 3 | 1 | 75.0% | $13.42 | −$9.18† | +$2.95 | +0.25 | +0.062 | 1.462 | $4.47 | −$9.18 | −$9.18 | 1 | ~5.5 |
| AUDJPY AMR | 5 | 0 | 5 | 0.0% | $0 | $27.94 | −$27.94 | −2.31 | −0.462 | 0.0 | n/a | −$5.59 | largest in set | 5 | ~7.5 |
| CADJPY AMR | 4 | 1 | 3 | 25.0% | $0.28 | $17.69 | −$17.41 | −1.44 | −0.360 | 0.016 | $0.28 | −$5.90 | — | 3 | ~7.5 |
| CADJPY ARB | 2 | 0 | 2 | 0.0% | $0 | $27.73 | −$27.73 | −1.16 | −0.580 | 0.0 | n/a | −$13.87 | −$15.72 | 2 | ~14.5 |
| GBPUSD Monday | 2 | 1 | 1 | 50.0% | $1.29 | $7.76 | −$6.47 | −0.53 | −0.265 | 0.166 | $1.29 | −$7.76 | −$7.76 | 1 | 23.0 |

†gross/avg figures derived from the per-strategy total P&L and win/loss counts in `reports/live_validation_baseline.csv`; individual trade-level detail is in that CSV.

**Open trades:** every strategy currently has exactly 1 OPEN trade in the fresh export (part of the standard OPEN+CLOSED lifecycle pairing already documented in `reports/entry_price_logging_audit.md`) — these are not double-counted as additional closed trades anywhere in this document, per the explicit instruction.

Full per-strategy × per-window breakdown (all four windows, every metric): `reports/live_validation_baseline.csv`.

---

## 2. Live vs. pre-live expectations — divergence classification

| Strategy | Historical OOS PF | Live PF (post-demotion) | Historical OOS expectancy R | Live expectancy R | Divergence class |
|---|---|---|---|---|---|
| GBPJPY AMR | 2.101 | INF (n=2, both wins) | 0.287 | +0.435 | **A. Within expected variance** (live is directionally consistent, trivially small sample) |
| EURJPY AMR | 1.343 | 1.462 | 0.104 | +0.062 | **A. Within expected variance** (live PF exceeds historical) |
| AUDJPY AMR | 1.144 | 0.0 | 0.050 | −0.462 | **C. Material negative divergence** (n=5; see §3 for why this is not yet "D") |
| CADJPY AMR | 1.305 | 0.016 | 0.092 | −0.360 | **C. Material negative divergence** (n=4) |
| CADJPY ARB | 1.519 | 0.0 | 0.248 | −0.580 | **B. Mild negative divergence** (n=2 — too small to call "material" with confidence; see §3) |
| GBPUSD Monday | 2.929 | 0.166 | 0.177 | −0.265 | **B. Mild negative divergence** (n=2) |

**No strategy is classified "D. Potential strategy deterioration."** Per the explicit instruction not to call a strategy broken solely because live PF < 1 with a tiny sample, AUDJPY AMR and CADJPY AMR — the two largest negative divergences — are held at "C. Material negative divergence," not escalated further, because §3's sample-size analysis shows neither yet has enough trades for that stronger claim to be statistically supportable.

---

## 3. Sample-size analysis

**Minimum sample for a meaningful directional conclusion**, using a rough normal-approximation planning heuristic (n ≈ 16·σ²/δ², σ≈1 in R-units, δ = the strategy's own historical OOS expectancy — see `src/phase29_live_scorecard.py::min_sample_for_direction()` for the exact formula; **this is a planning approximation, not a formal power calculation on the actual R-distribution**, stated explicitly as a limitation):

| Strategy | Live n (post-demotion) | Approx. min n needed to detect its own historical-magnitude expectancy shift | Verdict |
|---|---|---|---|
| GBPJPY AMR | 2 | 773 | **Grossly insufficient** — expected, given only 2 live trades |
| EURJPY AMR | 4 | 5,233 | **Grossly insufficient** |
| AUDJPY AMR | 5 | 6,104 | **Grossly insufficient** |
| CADJPY AMR | 4 | 18,262 | **Grossly insufficient** (its own historical expectancy is smallest in the book, so an even larger sample would be needed to distinguish a real shift from noise) |
| CADJPY ARB | 2 | 910 | **Grossly insufficient** |
| GBPUSD Monday | 2 | 936 | **Grossly insufficient** |

**Every strategy is, by this measure, orders of magnitude short of the sample needed for a statistically confident directional verdict.** This is not a surprising result — it is the expected outcome of a book that has only been live 2 weeks post-demotion — and it is the primary reason this document builds a *threshold framework* rather than a *verdict*.

**Bootstrap 90% CI on live expectancy (10,000 resamples, post-demotion window):**

| Strategy | Live n | Live expectancy R | 90% CI | % of bootstrap draws positive |
|---|---|---|---|---|
| GBPJPY AMR | 2 | +0.435 | **INSUFFICIENT (n<3)** | INSUFFICIENT |
| EURJPY AMR | 4 | +0.062 | [−0.305, +0.340] | 63.1% |
| AUDJPY AMR | 5 | −0.462 | [−0.532, −0.394] | **0.0%** |
| CADJPY AMR | 4 | −0.360 | [−0.595, −0.115] | 0.3% |
| CADJPY ARB | 2 | −0.580 | **INSUFFICIENT (n<3)** | INSUFFICIENT |
| GBPUSD Monday | 2 | −0.265 | **INSUFFICIENT (n<3)** | INSUFFICIENT |

**Read carefully — this is a bootstrap of the *live sample itself*, not a comparison to the historical distribution.** AUDJPY AMR's CI excluding zero (0.0% of resamples positive) means: *given only these 5 live trades, resampling them can't produce a positive-expectancy scenario* — it does **not** mean AUDJPY's true edge is proven negative; it means the 5 trades themselves are consistently negative, which is a different and much weaker claim. **Explicitly: "Insufficient evidence" for a deterioration verdict on any strategy** — the bootstrap here characterizes the live sample's own internal consistency, not statistical significance against history (that comparison is what §2's Monte Carlo work in the parent forensic reports already attempted, with the same "extreme tail but small sample" conclusion).

---

## 9. Live evidence vs. known AMR weaknesses

| Weakness (pre-live, documented) | AUDJPY AMR live check | CADJPY AMR live check | EURJPY AMR live check |
|---|---|---|---|
| SELL leg net-losing historically | Live SELL: 0/2, −0.425R expectancy — **consistent** with the historical flag, but n=2 | Live SELL: 0/2, −0.370R expectancy — **consistent**, n=2 | Live SELL: 2/2, +0.205R expectancy — **inconsistent** (opposite of the historical flag), n=2 |
| BUY leg (no historical weakness flagged for any of the three) | Live BUY: 0/3, −0.487R — **also losing**, i.e. AUDJPY is not sparing its "safe" side live | Live BUY: 1/2, −0.350R — mixed | Live BUY: 1/2, −0.080R — mixed, weaker than SELL |
| HIGH-volatility regime net-losing | Live HIGH bucket: 0/2, −0.510R avg — **consistent**, but LOW (0/2) and NORMAL (0/1) buckets are equally bad — **regime-specificity not cleanly confirmed live** (see `reports/5ers_portfolio_update_aug13.md` §6 for the same finding) | Live HIGH bucket: 0/3, −0.487R avg — **cleanly consistent**; NORMAL bucket (1/1) is a small win | Insufficient regime-bucketed data in this window to test meaningfully (n=4 spread across LOW/NORMAL only) |
| Cost-fragile (net/near-negative at 1.5-2x spread in backtest) | Live avg spread 1.63-6.67 pips depending on PRE/POST-FIX status (§10 below) — no direct 1.5-2x-spread-equivalent test possible from live data alone; historical cost-stress result stands as the operative pre-live evidence | Same — no live-data cost-stress test possible; historical result stands | Same |
| Scheduled-exit impact | 1 of 5 live exits (20%) is scheduled — not the dominant exit type | 2 of 4 live exits (50%) scheduled | 1 of 4 live exits (25%) scheduled, and it was a small win — no evidence scheduled exits are converting winners to losers |
| Holding time | ~7.5h avg — broadly consistent with the strategy's ≤4-6h design window (force-flat 07:00 UTC) | ~7.5h avg — same | ~5.5h avg — same |

**Do not modify any AMR strategy based on this table** — consistent with the task instruction, this section only tests whether live behavior matches already-known weaknesses. It does, for CADJPY AMR's HIGH-vol pattern and both pairs' SELL-leg weakness (partially); it does not cleanly for AUDJPY AMR's regime-specificity or EURJPY AMR's SELL-leg (which is currently the *stronger* side live, opposite the historical flag).

---

## 10. Execution quality (PRE-FIX vs. POST-FIX, per the entry-price logging audit)

Per `reports/entry_price_logging_audit.md`: the historical entry-price defect (`order_send().result.price` returning 0.0, fixed via `_confirm_fill_price()`/`positions_get(ticket).price_open` in commit `0b64c02`) was **logging-only** — never an execution failure. Applied here to the post-demotion window (19 trades):

| entry_fix_status | Trades | Win rate | Avg R | Avg spread (pips) | Avg spread/SL ratio |
|---|---|---|---|---|---|
| PRE_FIX | 8 | 25.0% | −0.261 | 6.67 | 5.9% |
| POST_FIX | 11 | 45.5% | −0.203 | 1.63 | 4.1% |

**POST_FIX trades show a modestly better win rate and expectancy than PRE_FIX trades in this window** — but this is confounded with time (POST_FIX trades are simply the more recent ones, from 2026-08-09 onward per the audit's established boundary) and should not be read as "the fix improved trading outcomes," since the fix never touched execution (per the audit). The most useful reading: **POST_FIX trades' entry prices, spreads, and implied slippage are now fully trustworthy for execution-quality analysis**, and that analysis (avg spread/SL ratio 4.1%, well under any concerning threshold) shows **no evidence that execution quality is contributing to the current drawdown**, consistent with every prior phase's finding on this question.

**Implied slippage:** not independently measurable from the current export schema (it lacks a `signal_price` column, per the entry-price audit's own finding) — reported as **NOT AVAILABLE**, not estimated.

---

## 11. Cost-stress → live threshold influence

Per the pre-existing revalidation (`reports/current_6_strategy_revalidation.csv`), cost-fragility classifications were folded directly into the review/pause thresholds in `reports/live_strategy_decision_rules.md` (not re-derived or re-tested here — no optimization performed):

- **EURJPY/AUDJPY/CADJPY AMR (all cost-fragile)**: their REVIEW/PAUSE thresholds explicitly include a cost-adjusted expectancy condition (spread-inflated, matching their own documented 1.5-2x-spread cost-stress test) — a strategy that was already known to be sensitive to elevated spread conditions should have that sensitivity reflected in its threshold, not just its raw PF.
- **GBPJPY AMR, CADJPY ARB, GBPUSD Monday (all cost-robust)**: their thresholds rely more heavily on streak-length and rolling-PF conditions alone, since cost-fragility is not an independently-flagged risk for these three.

---

## 12. Monitoring dashboard specification (not implemented — specification only)

**Per-strategy row:**

| Field | Source |
|---|---|
| STATUS | GREEN/YELLOW/ORANGE/RED per `reports/live_strategy_decision_rules.md` |
| LIVE TRADES | Post-demotion closed-trade count |
| WIN RATE | Post-demotion win rate |
| PF | Post-demotion profit factor |
| EXPECTANCY | Post-demotion expectancy R |
| TOTAL R | Post-demotion total R |
| CURRENT DD | Current drawdown in R (post-demotion equity curve) |
| LOSING STREAK | Current trailing losing streak |
| BUY R | Post-demotion BUY-side total R |
| SELL R | Post-demotion SELL-side total R (N/A for GBPUSD Monday, BUY-only by design) |
| REGIME STATUS | ATR-tercile win-rate/R breakdown, flagged if HIGH-bucket matches a pre-known weakness |
| COST STATUS | ROBUST/FRAGILE tag from `current_6_strategy_revalidation.csv`, static |
| SAMPLE SIZE | Post-demotion trade count vs. this document's §3 minimum-sample estimate (always shown as "insufficient" until that threshold is reached) |
| NEXT REVIEW CONDITION | The specific §3 (decision-rules doc) trigger text for that strategy |

**Portfolio row:**

| Field | Source |
|---|---|
| PORTFOLIO STATUS | Aggregate of the six strategy statuses, worst-case (e.g., if any is ORANGE, portfolio shows ORANGE) |
| TOTAL R | Post-demotion portfolio total R |
| DD | Post-demotion portfolio drawdown (R) |
| PF | Post-demotion portfolio PF |
| JPY EXPOSURE | % trades / % risk JPY-linked, per `reports/portfolio_concentration_framework.md` |
| CORRELATED LOSS DAYS | Days with 2+ strategies losing simultaneously, post-demotion |
| ACTIVE STRATEGIES | 6 (static, unless one is ever paused) |
| STRATEGIES UNDER REVIEW | Count currently at YELLOW/ORANGE/RED |

**This specification is not implemented in this phase**, per instruction — it describes what the existing `mcp/server.py` dashboard (already documented in prior phases) could be extended to show.

---

## 13. Decision tree (strategy-specific application)

```
LIVE TRADE
  ↓
Does performance remain within historical distribution?
  (§2's divergence class A or B → YES; C or D → NO)
  ↓
  YES (GBPJPY AMR, EURJPY AMR, CADJPY ARB, GBPUSD Monday)
    → CONTINUE (GREEN, except CADJPY ARB → YELLOW per its 0-for-2 pattern, §6 decision-rules doc)
  NO (AUDJPY AMR, CADJPY AMR)
    ↓
    Is sample size sufficient? (§3: NO for all six, by a wide margin)
    ↓
    NO → CONTINUE VALIDATION, but since these two ALSO show...
    ↓
    Is deterioration explained by known regime/cost/factor? (§9: YES for CADJPY AMR's
    HIGH-vol pattern and both pairs' pre-flagged cost-fragility; PARTIALLY for AUDJPY AMR,
    whose regime-specificity did not cleanly replicate)
    ↓
    YES → REVIEW / MONITOR (ORANGE — already in force via the pre-existing
           2026-08-25 AMR checkpoint, not a new rule created here)
```

**No strategy in this book currently reaches the "PAUSE/REDUCE RECOMMENDATION" leaf of this tree.**

---

## 14. Final scorecard

| Strategy | Pre-live evidence | Live n | Live PF | Live R | Current DD | Main weakness | Evidence status | Current status | Review trigger | Pause trigger |
|---|---|---|---|---|---|---|---|---|---|---|
| GBPJPY AMR | A. STRONG REVALIDATION | 2 | INF | +0.87 | 0.0 | None identified | INSUFFICIENT (n=2) | **GREEN** | Rolling 20-trade PF < 1.0 OR 2 consecutive losses | See decision-rules doc |
| EURJPY AMR | C. PROMISING BUT INSUFFICIENT | 4 | 1.462 | +0.25 | −1.03 | Cost-fragile; historical SELL weakness (not replicated live) | INSUFFICIENT (n=4) | **GREEN** | 4 consecutive losses OR rolling 10-trade expectancy < −0.30R | See decision-rules doc |
| AUDJPY AMR | C. PROMISING BUT INSUFFICIENT | 5 | 0.0 | −2.31 | −2.31 | 0-for-5, losses span all regimes/directions; largest live loss contributor | PLAUSIBLE CONCERN, still INSUFFICIENT for deterioration | **ORANGE** | Already triggered — heightened validation to 2026-08-25 checkpoint | See decision-rules doc |
| CADJPY AMR | D. WEAK / PROVISIONAL | 4 | 0.016 | −1.44 | −1.44 | 3-for-4 losses; HIGH-vol bucket 0-for-3 (matches historical flag) | PLAUSIBLE CONCERN | **ORANGE** | Already triggered — weakest evidence base in book | See decision-rules doc |
| CADJPY ARB | B. ACCEPTABLE BUT MONITOR | 2 | 0.0 | −1.16 | −1.16 | 0-for-2, both SL (no TP) — pattern worth tracking | INSUFFICIENT (n=2 — explicitly not pause-worthy) | **YELLOW** | 0-for-3 (pattern watch, not statistical proof) | Losing streak > 10 (own historical max) — see §6 decision-rules doc |
| GBPUSD Monday | A. STRONG REVALIDATION | 2 | 0.166 | −0.53 | −0.53 | 1W/1L, too few to say anything; both trades within normal design range | INSUFFICIENT (n=2) | **GREEN** | 4 consecutive losses (own historical max) | See decision-rules doc |

**What we know:** every strategy's live losing streak is currently inside (AUDJPY: at the edge of) its own historical maximum; JPY correlation and CADJPY AMR's HIGH-vol weakness both replicate cleanly live; execution quality shows no material issue; no strategy has anywhere near enough live trades for a confident directional verdict.

**What we don't know:** whether AUDJPY AMR's and CADJPY AMR's live weakness will persist past the 2026-08-25 checkpoint or fade as an expected trending-regime dip (the pre-existing, unresolved question this book has been tracking since before this task); whether CADJPY ARB's 0-for-2 is the start of a genuine pattern or ordinary noise inside its own 10-trade historical streak envelope; true slippage for any strategy (schema limitation, not fixable from existing data).

**What we are waiting for:** the 2026-08-25 AMR checkpoint (pre-existing rule); any strategy accumulating enough post-demotion trades to approach the (still very large) sample sizes in §3; a third CADJPY ARB SL loss, purely as a pattern-tracking signal, not a pause trigger.

---

*No strategy modified. No parameters modified. No risk modified. No deployment performed. All statuses and triggers in this document are analytical only.*
