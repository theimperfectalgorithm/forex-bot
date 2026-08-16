# Live Strategy Decision Rules — Predefined Review/Kill Criteria

**Purpose:** a decision framework, not an optimization exercise. Every threshold below is derived from evidence that existed **before** this analysis was run (pre-live acceptance criteria, historical distributions, cost-stress results) — never from the current losses themselves. Written to be defensible regardless of whether the next 10 live trades are all winners, all losers, or mixed (see §7, anti-bias check).

**Source data:** `reports/current_6_strategy_revalidation.csv` (pre-live acceptance criteria, EXP-096..104/105..111), `data/phase26_all_trades.csv` (2,712-trade historical population), `reports/5ers_trade_export.csv` (fresh production export, 72 rows/36 tickets, verified — see `reports/live_strategy_scorecard.md` §0), `src/phase29_live_scorecard.py` (analysis engine, reused/imports `src/phase27_5ers_current_portfolio_forensic.py`).

**No strategy, parameter, risk, pair, or configuration was modified in producing this document.**

---

## 1. Why thresholds are strategy-specific, not uniform

A single "N consecutive losses = pause" rule applied uniformly to all six strategies would be indefensible, because the six strategies have very different **historical loss-streak distributions**, **pre-live evidence quality**, and **cost-fragility**:

| Strategy | Historical max losing streak | Pre-live classification | Cost-stress result |
|---|---|---|---|
| GBPJPY AMR | 5 | A. STRONG REVALIDATION | ROBUST (PF 1.21 @ 2x spread) |
| EURJPY AMR | 4 | C. PROMISING BUT INSUFFICIENT | FAIL — cost-fragile (PF 0.89 @ 2x spread) |
| AUDJPY AMR | 5 | C. PROMISING BUT INSUFFICIENT (weakest OOS PF of AMR family) | FAIL — cost-fragile (PF 0.84 @ 2x, already net-losing @ 1.5x) |
| CADJPY AMR | 6 | D. WEAK / PROVISIONAL (weakest in book) | FAIL — worst cost-fragility in book (net-losing @ 1.5x) |
| CADJPY ARB | 10 | B. ACCEPTABLE BUT MONITOR | ROBUST (PF 1.16 @ 2x), but HIGH-vol/HIGH-trend regime net-losing |
| GBPUSD Monday | 4 | A. STRONG REVALIDATION | ROBUST — best cost-robustness in book (PF 1.98 @ 2x) |

A strategy whose **own historical worst stretch** was a 10-trade losing streak (CADJPY ARB) should not be treated the same way as one whose historical worst was 4 (GBPUSD Monday) when both show 2 live losses. This is the core design principle behind every threshold below.

---

## 2. NORMAL VARIANCE — what is still expected, per strategy

Defined as: live losing-streak length, live PF, and live expectancy still comfortably inside the strategy's own historical distribution, given the current tiny live sample size.

| Strategy | Normal-variance losing streak | Normal-variance PF range (given n<10) | Basis |
|---|---|---|---|
| GBPJPY AMR | Up to 5 | Any PF value is uninformative below ~10 trades given historical PF variance across its 90.9%-of-windows-profitable walk-forward record | Historical max streak = 5; walk-forward PASS |
| EURJPY AMR | Up to 4 | Same reasoning | Historical max streak = 4 |
| AUDJPY AMR | Up to 5 | Same reasoning, but starting confidence is lower (weakest OOS PF, cost-fragile) | Historical max streak = 5 |
| CADJPY AMR | Up to 6 | Same reasoning, weakest starting confidence of the six | Historical max streak = 6 |
| CADJPY ARB | Up to 10 | Same reasoning; this strategy's own historical worst stretch (10) is more than double any AMR pair's | Historical max streak = 10; walk-forward INSUFFICIENT (2 consecutive negative windows in 2024 — already knew this strategy has rough stretches) |
| GBPUSD Monday | Up to 4 | Same reasoning; smallest total historical sample (154 trades) means even its own historical streak figure carries more uncertainty than the others' | Historical max streak = 4 |

**None of the six strategies' current live losing streaks (0-5, see the scorecard) exceed their own historical maximum.** This is the first and most important test any of them must pass before further scrutiny is warranted — currently, all six pass it.

---

## 3. REVIEW TRIGGER — what escalates to deeper investigation

A review trigger is **not** a decision to change anything — it means "look closer, using the tools already built" (this exact document, `src/phase29_live_scorecard.py`, and the existing `core/health_monitor.py`).

| Strategy | Review trigger | Source of the threshold |
|---|---|---|
| GBPJPY AMR | Rolling 20-trade PF falls below ~1.0 (breakeven), OR 2 consecutive live losses (would be this strategy's *first* live losing streak of any length) | Its historical walk-forward record shows profitability in 90.9% of rolling windows — 2 straight losses would be a genuinely new event for this specific strategy, not just "a loss" |
| EURJPY AMR | 4 consecutive losses (= its own historical max streak, i.e. the review point is "at the edge of normal," not "past normal") OR rolling 10-trade expectancy < −0.30R | Historical max streak 4; −0.30R is roughly 5x its historical OOS expectancy (0.104R) in the adverse direction — a rough "well outside typical variance" heuristic, not an exact statistical bound given the small live sample |
| AUDJPY AMR | **Already triggered** — its documented HIGH-vol-regime weakness (PF 0.826 in backtest) was flagged in the pre-live evidence itself, and live data (§9, main scorecard doc) shows it losing across all regime buckets, not just HIGH — this alone is sufficient basis for heightened validation under the existing 2026-08-25 AMR checkpoint rule (PROJECT_REPORT.md §6) | Already in force from a prior phase of this project, not newly created here |
| CADJPY AMR | **Already triggered**, same reasoning as AUDJPY — weakest pre-live evidence base in the book plus HIGH-vol weakness reappearing live | Weakest historical PF (1.084) + worst cost-stress result in the book |
| CADJPY ARB | 3 consecutive SL losses with no intervening TP (i.e., 0-for-3) | See §6 below for why this specific number was chosen and why it is a REVIEW trigger, not a PAUSE trigger |
| GBPUSD Monday | 4 consecutive losses (= its own historical max streak) OR rolling 10-trade PF < 1.0 | Historical max streak 4; smallest total sample (154 trades) means its own historical streak number is itself less certain, so treating it as a review (not pause) threshold is appropriate |

---

## 4. REDUCE-RISK TRIGGER — when risk reduction would be *considered* (not implemented)

This tier sits between REVIEW and PAUSE. It is reserved for evidence that is concerning but not yet strong enough to justify full removal from the live book.

**General principle, derived from the prop-firm risk constraint (5ers $5K account, daily/overall drawdown limits documented in `PROJECT_REPORT.md`):** a strategy should be considered for a reduce-risk (not pause) recommendation when it has **both** (a) crossed its own REVIEW trigger, **and** (b) its cumulative post-demotion drawdown contribution exceeds roughly 15% of the account's total allowable drawdown headroom, **and** (c) at least one independent explanatory factor (cost-fragility, regime weakness, or directional asymmetry) that was already flagged pre-live has reappeared in the live data.

| Strategy | Currently meets reduce-risk criteria? | Basis |
|---|---|---|
| AUDJPY AMR | **Closest of the six** — meets (a) and (c) (cost-fragile + HIGH-vol regime weakness, both pre-flagged, reappearing live); (b) is not independently confirmable from account-level drawdown-limit data not available in this analysis | Largest live dollar/R loss contributor in the post-demotion window |
| CADJPY AMR | Meets (a) and (c) on the same logic as AUDJPY, with a weaker starting evidence base | Weakest pre-live PF in the book |
| All others | Not currently meeting the (a) threshold | See §3 |

**This is a recommendation-only tier — nothing has been reduced.** Per the pre-existing 2026-08-25 AMR checkpoint rule, AUDJPY/CADJPY AMR's status here is being tracked toward that date, not acted on now.

---

## 5. PAUSE TRIGGER — minimum evidence required

**Deliberately conservative, derived from sample-size analysis (Phase 3 of the parent task), not from wanting an easy-to-trigger rule.**

General principle: a PAUSE recommendation requires the live losing streak or drawdown to **exceed the strategy's own historical worst-case envelope**, not merely approach it, **and** the live sample to be large enough that a plausible historical explanation (variance, regime, correlation) has been actively ruled out using the same tools in this document (bootstrap CI excluding zero in the adverse direction with reasonable confidence, not just a point estimate).

| Strategy | Pause trigger (both conditions required) |
|---|---|
| GBPJPY AMR | Losing streak > 5 (exceeds historical max) AND bootstrap 90% CI on rolling expectancy excludes zero on the negative side with n ≥ 15 |
| EURJPY AMR | Losing streak > 4 AND cost-adjusted (spread-inflated, matching its documented 2x-spread cost-stress test) rolling expectancy negative over n ≥ 10, with bootstrap CI excluding zero |
| AUDJPY AMR | Losing streak > 5 (already at 5 as of this report — see §6/scorecard) sustained past the 2026-08-25 checkpoint with no recovery toward backtested expectation, per the pre-existing decision rule (not newly created here) |
| CADJPY AMR | Losing streak > 6 sustained past the 2026-08-25 checkpoint, same logic as AUDJPY given its shared HIGH-vol weakness flag |
| CADJPY ARB | Losing streak > 10 (its own historical max) — given CADJPY ARB trades far less frequently than the AMR pairs, this could take months to accumulate; the practical pause consideration point is a **sustained** PF < 0.6 over at least 15 live trades (roughly its historical trade frequency over several months), not streak length alone |
| GBPUSD Monday | Losing streak > 8 (2x its historical max, given the smallest total historical sample and the correspondingly wider uncertainty on its own streak figure) AND rolling 15-trade PF < 1.0 |

**None of the six strategies currently meet their PAUSE trigger.** This is stated plainly, not softened: even AUDJPY AMR, the weakest performer live, is at a 5-trade streak against its own historical max of 5 — at the edge, not past it.

---

## 6. The "third loss = pause" fallacy — CADJPY ARB specifically investigated

**Current CADJPY ARB state:** 2 post-demotion trades, both SL, 0 wins, latest −0.66R.

**Is "a third consecutive SL" a statistically defensible PAUSE trigger? No.** Reasoning:

1. **CADJPY ARB's own historical maximum losing streak is 10 trades** (`reports/current_6_strategy_revalidation.csv`). A streak of 3 is well inside that envelope — treating 3 as a pause point would be applying a threshold roughly **3x stricter** than the strategy's own documented normal behavior, with no evidentiary basis for the tighter number beyond "it feels uncomfortable."
2. **Sample-size math:** with only 2 trades so far (and CADJPY ARB trading far less frequently than the AMR pairs — ~192 trades over roughly 2 years historically, versus 400-700+ for the AMR pairs), a third loss brings the live sample to n=3. The bootstrap-CI approach used elsewhere in this framework (§4 above, and Phase 3 of the parent analysis) could not produce a meaningful confidence interval at n=2 (insufficient data — see `reports/live_strategy_scorecard.csv`), and n=3 would not materially change that. **A rule that fires at the exact sample size where no statistical test can yet distinguish signal from noise is not defensible.**
3. **What a third loss WOULD mean:** it is worth a closer look (a REVIEW trigger, §3), specifically because CADJPY ARB's design is TP-driven (2:1 reward:risk breakout) and two-then-three consecutive SL exits with zero TP hits would be a genuinely unusual *pattern* for this strategy's mechanics even at small n — but pattern-worth-investigating is different from evidence-sufficient-to-pause.

**Replacement rule, derived from the strategy's own historical distribution rather than an arbitrary round number:** CADJPY ARB's PAUSE trigger is set at **losing streak > 10** (its own historical max) **or** a sustained PF < 0.6 over 15+ live trades (§5) — not a fixed small trade count. Its REVIEW trigger (0-for-3, §3) exists precisely so the *pattern* gets tracked without prematurely treating a 3-trade sample as decisive.

---

## 7. Anti-bias check — would these rules survive the next 10 trades going either way?

**All-winners scenario:** every REVIEW/REDUCE/PAUSE trigger in this document is defined by adverse outcomes (losing streaks, negative rolling PF/expectancy) — a run of winners simply resets losing-streak counters toward zero and does not retroactively change any threshold. No threshold in this document depends on the current loss total.

**All-losers scenario:** each strategy's PAUSE trigger requires exceeding its OWN pre-existing historical maximum streak (not a number chosen to match the current tally) plus a statistical condition (bootstrap CI excluding zero, or a sustained-PF condition over a stated minimum trade count) — both conditions were derived from data that existed before this task began (the `current_6_strategy_revalidation.csv` figures are all pre-live/pre-this-analysis). A run of losses would trigger REVIEW and then REDUCE-RISK tiers in a way fully explainable by these pre-set numbers, not by moving goalposts to fit whatever the losses happen to total.

**Mixed scenario:** thresholds are all rolling-window or streak-based (not cumulative-since-inception), so they respond to genuinely recent evidence without being either desensitized by an early good patch or oversensitized by a single bad patch.

**Verification, not just assertion:** every numeric threshold above (streak lengths in §2/§3/§5, the cost-stress conditions, the 2026-08-25 checkpoint reference) traces to a specific number already present in `reports/current_6_strategy_revalidation.csv` or `PROJECT_REPORT.md` §6, both written before this task and before today's trades were known. None of the six strategies' thresholds were adjusted to make the current 19-trade post-demotion sample look better or worse than it is.

---

## 8. Reinstatement criteria (for any strategy paused in the future)

Not currently applicable — no strategy is paused. Defined here in advance, per instruction, so it cannot later be set reactively:

A paused strategy should only return to live trading when:
1. The root cause identified at pause time has either resolved (e.g., a regime condition passed) or been specifically addressed through the project's normal validation pipeline (full IS/OOS discipline, per `PROJECT_REPORT.md` §4's methodology) — **never** a parameter tweak made in response to the live losses themselves without that full discipline.
2. A fresh forward-test period (demo, not live capital) of at least the strategy's own historical minimum meaningful sample (roughly 15-20 trades, consistent with the sample sizes treated as minimally informative throughout this document) shows performance back within its historical distribution.
3. The reinstatement decision is made by the same evidence-based process as the original deployment decision (`PROJECT_REPORT.md` §4's IS/OOS/walk-forward/cost-stress/regime discipline) — not a unilateral "it's been a while, let's try again."

---

*No strategy, parameter, risk, pair, or configuration was modified in producing this document. All triggers are recommendations for future review, not actions taken now.*
