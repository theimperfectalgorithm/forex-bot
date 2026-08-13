# 5ers Current Drawdown — Forensic Analysis

**DIAGNOSTIC ONLY. No strategy, entry/exit logic, parameter, risk, or
live configuration was modified. Nothing was paused or deployed. The
AUDJPY BUY-only candidate was not implemented.**

**Experiments:** EXP-112 through EXP-116, `experiments/experiments.csv`.
**Scripts:** ad hoc MT5 queries (this session) + `data/phase26_all_trades.csv`
(EXP-105-111, `src/phase26_current6_revalidation.py`, commit 6fd93a3).

## 0. Critical data-availability finding — stated first, not buried

**This session's MetaTrader5 connection was verified directly (not
assumed) before any analysis began:**

```
login=5052472770  server=MetaQuotes-Demo  balance=101,365.14
```

**This is the DEMO account. This session has no access to the 5ers
account (FivePercentOnline-Real, per project memory) — no local MT5
terminal binding, no `C:\MT5-5ers\` path on this machine, no
`data/journal/events.jsonl`, and no dashboard API credentials were
supplied in this conversation.** Per the explicit instruction not to
infer from screenshots when machine-readable source data would
otherwise be required, and not to guess: **Steps 2-14 and most of
16-19, which require real 5ers trade-level data (trade ID, per-trade
strategy tag, entry/exit price, spread, ATR, exit reason), are NOT
AVAILABLE in this session.** This is reported as a finding, not an
excuse — the honest answer to "reconstruct every 5ers trade" is "cannot
be done from here without that data."

**What this report does instead, and why it is still substantive:**
Step 15's Monte Carlo/resampling test does **not** require real 5ers
trade data — it tests whether the account-level *summary* statistics
you supplied are statistically compatible with the strategies' own
validated historical trade distributions. That test was run rigorously
(20,000 draws) and produced a decisive, important result (§14 below).
Sections 11-13, 16-19, and 20 draw on the structural findings already
established in `reports/current_6_strategy_revalidation.md`
(EXP-105-111) — cost fragility, JPY correlation, regime dependence —
explicitly labeled as **candidate contributing factors consistent with
prior research, not confirmed attribution**, since confirming which
factor actually fired in this specific 33-trade window requires the
real trade log this session does not have.

---

## 1. Executive summary

The current 5ers account (~$5,000 → ~$4,797, ~33 trades, PF~0.30, win
rate~30%, expectancy~-0.33R) **cannot be fully forensically attributed
in this session** because no 5ers trade-level data was accessible. What
**can** be established, rigorously: **a resampling test drawing 33-trade
samples from the current 6 strategies' own combined historical trade
pool (20,000 draws, weighted by each strategy's actual trading
frequency) never once produced a PF as low as 0.30 or a win rate as low
as 30.3%** — the 1st percentile of the simulated distribution is
already a PF of 0.48 and a win rate of 48.5%. **This is strong evidence
that the current period's aggregate weakness is not well-explained by
ordinary trade-sequence variance from the validated historical
population alone**, and raises — without being able to confirm from
here — the possibility that either (a) the "33 trades" figure includes
trades from before the 2026-07-31 demotion (GBPJPY ARB/XAUUSD ARB, not
part of the current 6-strategy set), (b) current live execution
materially differs from the backtested population, or (c) something
about the live regime is currently unfavorable in a way the 33-trade
sample happens to concentrate. **This report does not and cannot
distinguish between these explanations without real 5ers trade data.**

## 2. Exact current losing period

**NOT AVAILABLE.** Determining the exact start of the current losing
episode requires the account's real equity/balance sequence, which was
not supplied to or accessible from this session. The only anchor points
available are the account-level snapshot you provided (start $5,000,
current ~$4,797, ~33 trades total) — this does not distinguish "one
continuous drawdown since account inception" from "a recovery-then-new-drawdown
pattern," and this report does not guess which.

**One specific, important ambiguity flagged rather than resolved:** the
5ers account went live between 2026-07-15 and 2026-08-11
(`reports/live_portfolio_validation_audit.md` §0, EXP-096-104), and
GBPJPY ARB + XAUUSD ARB were demoted from it on 2026-07-31 — **roughly
midway through the account's life so far.** If "~33 trades" is the
account's full history, a meaningful fraction of those trades likely
predate the demotion and belong to strategies **not in the current
6-strategy set this investigation is scoped to.** This session cannot
determine how many. **This materially affects every downstream question
in this report** — see §20.

## 3. Trade-by-trade drawdown reconstruction

**NOT AVAILABLE.** Requires real 5ers trade-level data. See §0.

## 4. Strategy attribution

**NOT AVAILABLE for the current period specifically.** See §14 for the
one thing that could be tested (aggregate statistical compatibility)
and §20 for structural (not confirmed) candidate explanations.

## 5. Pair attribution

**NOT AVAILABLE.** Same constraint.

## 6. Directional attribution (BUY vs SELL)

**NOT AVAILABLE for the current period.** What **is** available: the
historical SELL-leg weakness for 3 of 4 AMR pairs, independently
reconfirmed this session (`reports/current_6_strategy_revalidation.md`
§9, EXP-105-111): EURJPY AMR SELL PF 0.836, AUDJPY AMR SELL PF 0.706,
CADJPY AMR SELL PF 0.763 (all net-losing), vs. BUY-side PF 1.43-1.65 for
the same three pairs. **Whether the current losses are disproportionately
SELL trades cannot be tested without the real trade log** — this report
does not assume it, per instruction.

## 7. Exit-reason attribution

**NOT AVAILABLE** for the current period. Per instruction, restated for
the record: any dashboard row showing `MANUAL/OTHER` should be read as
**SCHEDULED_EXIT** (the bot's own scheduled London-open or time-based
closure), never as discretionary intervention — this classification
rule is available and correct, but applying it requires the actual
exit-reason column, which this session does not have.

## 8. Holding-time analysis

**NOT AVAILABLE** for the current period. Historical reference (EXP-105-111):
median holding time ranges from 1.25h (CADJPY/AUDJPY AMR) to 21h
(Monday Drift, by design — force-flat 21:00 Monday) to 13h (CADJPY ARB).

## 9. Cost/execution analysis

**NOT AVAILABLE for real current-trade spread/slippage data.** What
**is** available and directly relevant: `reports/current_6_strategy_revalidation.md`
§6 (EXP-105-111) found **3 of the 4 currently-active AMR pairs
(EURJPY, AUDJPY, CADJPY) are COST-FRAGILE** — they flip net-losing
under 1.5x-2x realistic spread stress in the historical backtest, using
frozen parameters. This is a structural vulnerability that makes the
strategies *more susceptible* to a bad run if live spreads have been
wider than the 2.0-pip backtest assumption at any point — but **this
report cannot confirm whether that is what happened in the current
period**, only that the strategies are known to be structurally
sensitive to it.

**On slippage calculation, per the explicit instruction not to trust it
blindly:** this session has no visibility into the dashboard's slippage
formula or its current output — `mcp/server.py`'s `/api/slippage`
endpoint exists per this project's own architecture, but was not queried
in this session (no API access). **No slippage number is being
reported, trusted, or "fixed" here** — there is nothing to evaluate.

## 10. Volatility / trend regime analysis (current trades)

**NOT AVAILABLE** for the current period — classifying real trades into
LOW/NORMAL/HIGH volatility or trend terciles requires their actual
entry timestamps, which requires the real trade log. Historical
reference (EXP-105-111): AUDJPY AMR and CADJPY AMR both show clear
HIGH-volatility-regime net-losing behavior (PF 0.826 and 0.831
respectively in the HIGH tercile); EURJPY AMR shows no strong volatility
dependency but is trend-tercile-sensitive. **Whether the current period
happens to fall in an unfavorable regime cannot be tested without
knowing when the current trades actually occurred.**

## 11. AMR mechanism check

**NOT AVAILABLE** to test directly against current trades. The
mechanism this report would be checking for — high-volatility/trending
conditions coinciding with AMR SELL-side losses — is documented in
`reports/amr_regime_mechanism.md` (EXP-076-081, e10d189) and
reconfirmed in `reports/current_6_strategy_revalidation.md` (EXP-105-111).
**Whether the current losing period resembles this mechanism cannot be
confirmed without the real trade data.**

## 12-13. Historical comparison / classification

Per Step 14's own instruction, this requires per-strategy live data,
which is **NOT AVAILABLE**. The only comparison this report can
legitimately make is **portfolio-aggregate**, done rigorously in §14
below.

## 14. Monte Carlo / resampling compatibility test — the core rigorous result of this report

**Methodology:** 20,000 simulated 33-trade portfolios were built by
drawing trades **with replacement** from the pooled historical trade
set of exactly the current 6 strategies
(`data/phase26_all_trades.csv`, EXP-105-111), weighting each draw by
that strategy's **actual historical trading frequency** (trades/year),
so the simulated mix approximates how the 6 strategies would naturally
interleave in a random 33-trade stretch:

| Strategy | Implied mix weight |
|---|---|
| EURJPY AMR | 26.3% |
| AUDJPY AMR | 24.0% |
| CADJPY AMR | 22.1% |
| GBPJPY AMR | 14.9% |
| CADJPY ARB | 7.1% |
| GBPUSD Monday | 5.7% |

**Per the explicit instruction: this is order-randomization of
drawdown/streak-style statistics, not a shuffled-P&L confidence
interval** — each simulated portfolio is an independently-drawn *new*
33-trade sample (not a reordering of one fixed set), specifically
constructed to test compatibility with the observed *aggregate rate*
statistics (PF, win rate, expectancy), which is a different and valid
question from the drawdown-path question Monte Carlo order-shuffling
answers elsewhere in this project.

**Results:**

| Statistic | Simulated 1st pctile | Simulated 5th pctile | Simulated 10th pctile | Simulated 25th pctile | Simulated median | **Observed 5ers value** | **Percentile of observed** |
|---|---|---|---|---|---|---|---|
| Profit factor | 0.477 | 0.618 | 0.713 | 0.914 | 1.216 | **~0.30** | **0.00%** (below the entire 20,000-draw distribution) |
| Win rate | 48.5% | 54.5% | 57.6% | 60.6% | 66.7% | **~30.3%** | **0.00%** (below the entire distribution) |
| Expectancy (scaled to same $100k reference capital as the historical pool) | -$82.63 | -$53.36 | -$36.99 | -$9.61 | +$20.74 | **-$123.03 equivalent** | **0.04%** |

**Interpretation, stated carefully:** in 20,000 independent 33-trade
draws from the current 6 strategies' own validated historical
population, **not one simulation produced a profit factor or win rate
as low as what the 5ers account is currently showing.** This is a
materially different result from the earlier drawdown/streak Monte
Carlo comparisons in this project (`reports/portfolio_drawdown_distribution_audit.md`,
EXP-092-095; `reports/current_6_strategy_revalidation.md` §10,
EXP-105-111), which found the current *drawdown magnitude and losing
streak length* to be elevated but plausible. **This resampling test
asks a different, arguably more direct question — is a ~30% win rate
and 0.30 profit factor plausible from this population at all? — and the
answer is: essentially no, not from ordinary sampling variance alone.**

**What this does NOT prove:** it does not prove the strategies are
broken. Three real, unconfirmable-from-here possibilities could each
independently explain this gap: (1) the 33-trade sample includes
retired strategies (GBPJPY ARB / XAUUSD ARB, not modeled in this
resampling pool at all) from before the 2026-07-31 demotion — see the
critical ambiguity flagged in §2; (2) current live execution
(spread/slippage/fills) differs materially from the 2.0-pip backtest
assumption, which the cost-fragility finding (§9) shows would be
sufficient to explain a large swing for 3 of the 6 strategies; (3) a
genuine, currently-in-progress unfavorable regime stretch that this
sample happens to concentrate. **This report explicitly does not rank
these without real data to discriminate between them** — see §20 for
the most honest ranking this session's evidence supports.

## 15. Loss clustering

**NOT AVAILABLE** for the current period specifically. Historical
structural reference (EXP-105-111): the current 6-strategy portfolio
sees 2+ JPY strategies losing on the same day 29.3% of the time, 3+ on
14.2% of days, with all pairwise JPY daily-correlation positive
(0.02-0.36). **Whether the current drawdown exhibits this clustering
pattern cannot be confirmed without dated, per-strategy trade data.**

## 16. JPY factor analysis

Structurally, **94.3% of the current 6-strategy portfolio's trades and
94.7% of its risk-weight are JPY-exposed** (EXP-105-111) — only Monday
Drift (GBPUSD, 0.25% of 1.75% total risk) is non-JPY. **This means, by
construction, any common JPY-factor move would be very likely to affect
most of the book simultaneously** — this is a structural fact about
portfolio concentration, not a claim about what actually happened in
the current period, which cannot be tested without a daily JPY-return
series matched against dated trade data neither of which is available
here.

## 17. Demo vs. 5ers comparison

**Partial data available, with an important caveat.** This session
pulled a fresh snapshot of the DEMO account's closed trade history (55
trades, via direct MT5 query, `login=5052472770`) and classified them
by their comment tags (e.g. `5ers_asian_BUY_AMR`, `5ers_london_BUY`),
which map cleanly to strategy + direction:

| Symbol | Comment tag | Implied strategy |
|---|---|---|
| GBPJPY/EURJPY/AUDJPY/CADJPY | `5ers_asian_{BUY,SELL}_AMR` | AMR family |
| GBPJPY/CADJPY/XAUUSD | `5ers_london_{BUY,SELL}` | ARB family |
| GBPUSD | `5ers_monday_BUY_MON` | Monday Drift |
| EURUSD | `5ers_ny_BUY_EMA` | **Not in the current 8-strategy book at all** — a stale/retired tag from an earlier strategy generation (consistent with `PROJECT_REPORT.md`'s note that an EURUSD sma_ema_combined book was retired) |

**This confirms the demo account still runs the full 8-slot book**
(including GBPJPY ARB and XAUUSD ARB, which are 5ers-excluded) — exactly
as `reports/live_portfolio_validation_audit.md` §3 documented. **No
signal-level match (same strategy, same pair, same direction, same
entry timestamp) between demo and 5ers could be performed** — this
session has demo data but no 5ers data to match it against. Classified
per the brief's own options: **D. INSUFFICIENT DATA** for the
signal-matching comparison specifically.

## 18. Live vs. backtest trade characteristics

**NOT AVAILABLE for 5ers.** For demo (the only live account this
session can query), a full ATR/spread/holding-time/regime distribution
comparison was not built in this pass — the demo pull returned symbol,
direction, entry/exit price, and P&L only (no ATR/spread/regime tags
were requested or computed for this fresh pull). This is noted as a
further limitation, not filled with an estimate.

## 19. Ranked candidate causes

Per instruction, evidence for/against/confidence for each candidate,
without claiming causality beyond what the evidence supports:

| Cause | Evidence FOR | Evidence AGAINST | Confidence |
|---|---|---|---|
| **9. Data/attribution ambiguity (pre- vs. post-demotion trades mixed)** | Timing overlap is real and unresolved (§2); the account's full history spans the 2026-07-31 demotion boundary | None available to rule it in or out | **MEDIUM** — plausible and unconfirmed, the single most important open question |
| **10. Insufficient sample size** | Only ~33 trades; even a real underlying edge produces wide short-run PF variance | The resampling test (§14) shows the observed result is still far outside normal sampling variance even accounting for small-N noise | **MEDIUM** — sample size alone does not fully explain the resampling result, but combined with cause #9 it could |
| **6. Spread/cost** | 3 of 6 strategies independently shown COST-FRAGILE in backtest (EXP-105-111); a $5,000 account with 0.25-0.50% risk has less room for spread/lot-size friction than the $100k backtest reference (the same class of issue that caused XAUUSD ARB's demotion) | No current spread data available to confirm live spreads were actually elevated | **LOW-MEDIUM** — structurally plausible, not confirmed |
| **8. JPY correlation / portfolio concentration** | 94.3-94.7% JPY exposure by construction; historical clustering (29.3% of days see 2+ JPY strategies lose together) | Cannot confirm clustering occurred in the specific current window | **LOW-MEDIUM** — structurally plausible, not confirmed |
| **3. AMR SELL-side weakness** | Well-documented historical weakness (3 of 4 pairs' SELL leg net-losing) | Cannot confirm the current losses are disproportionately SELL trades without the trade log | **LOW** — a real historical pattern, unconfirmed as the current cause |
| **4. High-volatility regime** | AUDJPY/CADJPY AMR historically weak in HIGH-vol regime | Cannot date the current trades to check regime | **LOW** — plausible, unconfirmed |
| **1. Normal variance** | Would be the default explanation for a 33-trade losing stretch in isolation | **Directly contradicted by §14's resampling test** — the observed PF/win-rate combination essentially never occurs by chance from this exact population, IF the 33 trades are genuinely a clean sample of the current 6 strategies | **LOW as the sole/primary explanation**, though it remains a partial contributor to any short sample's noise |
| **2. Strategy edge weakness / deterioration** | None of the individual strategies showed evidence of a broken edge in the full revalidation (EXP-105-111) — all showed real historical/OOS/walk-forward positive results | The resampling gap (§14) is large enough that "normal variance" alone is a poor fit, but that gap is better explained by causes #9/#6 than by an actual edge reversal, given the revalidation found no walk-forward or year-by-year evidence of deterioration | **LOW** |
| **7. Execution differences (general)** | The known BE-exit-logic gap (AMR: live uses a different rule than researched) is a real, documented live/backtest difference | Affects in-trade management, not entry signal quality; not large enough on its own to explain a PF-0.30 stretch | **LOW** |
| **5. Trend/mean-reversion conflict** | CADJPY AMR's known volatility×trend interaction (EXP-077) | No current data to test | **LOW** |

**Ranked, highest to lowest confidence:** (9) data/attribution
ambiguity ≈ (10) insufficient sample size > (6) spread/cost ≈ (8) JPY
concentration > (3) SELL-side weakness ≈ (4) volatility regime > (1)
normal variance alone > (2) edge deterioration ≈ (7) execution ≈ (5)
trend conflict.

## 20. Distinguish account problem from strategy problem

**F. COMBINATION OF ABOVE, with LOW-MEDIUM overall confidence** — most
specifically a combination of (9) unresolved data attribution and (10)
small sample size, with (6) cost/spread and (8) JPY concentration as
plausible amplifiers. **This report does not have enough evidence to
assign a percentage confidence** — per instruction, using HIGH/MEDIUM/LOW
qualitative confidence instead, and the honest answer here is that no
single cause reaches HIGH confidence given the data gap.

## 21. Strategy status (analytical only — nothing paused or changed)

| Strategy | Status | Reasoning |
|---|---|---|
| CADJPY ARB | **YELLOW** | No current-period data to assess directly; historical revalidation shows real regime dependence (HIGH-vol/trend losing) and a genuine 2024 walk-forward weak stretch — concerning but not disqualifying |
| GBPJPY AMR | **GREEN** | Strongest historical/OOS/walk-forward/cost-robustness/bootstrap-significance record of the 6 (EXP-105-111); no current-period data available to contradict it |
| EURJPY AMR | **YELLOW** | Cost-fragile in backtest, bootstrap CI crosses zero, but no evidence of current-period-specific failure (no data to assess) |
| AUDJPY AMR | **YELLOW** | Same profile as EURJPY, plus a weakening 2026 YTD trend and the deepest post-live research scrutiny of any strategy in the book; still no falsifying evidence |
| CADJPY AMR | **ORANGE** | Weakest evidence base of the 6 on nearly every dimension (lowest historical PF, worst cost-stress result, HIGH-regime losing) — "requires further validation" is the accurate read even before considering the current period, independent of any current-account attribution |
| GBPUSD Monday Drift | **GREEN** | Strongest single-strategy record in the project; small sample remains a standing caveat but nothing here contradicts it |

**No strategy is classified RED.** Per instruction, this classification
is not based on current 5ers performance (which cannot be attributed to
any specific strategy in this session) — it reflects each strategy's
own standalone historical evidence quality from
`reports/current_6_strategy_revalidation.md`.

## 22. Final portfolio verdict — direct answers

1. **What caused the current drawdown?** **Cannot be determined in this
   session** — no 5ers trade-level data was accessible. The resampling
   test shows the aggregate result is not well-explained by ordinary
   variance from the current 6 strategies' own history, which points
   toward either a data-attribution issue (pre-demotion trades mixed
   in), a small-sample effect, or a currently-adverse condition
   (cost/regime/clustering) — not distinguishable from here.
2. **Which strategy contributed most?** **NOT AVAILABLE.**
3. **Which contributed least?** **NOT AVAILABLE.**
4. **Concentrated in AMR?** **NOT AVAILABLE to confirm**, though AMR is
   87.5% of the book by strategy-count (5 of the 6 non-ARB/Monday risk
   units) and would statistically dominate any random sample by sheer
   representation, independent of any AMR-specific weakness.
5. **Concentrated in SELL trades?** **NOT AVAILABLE to confirm.**
6. **Concentrated in high volatility?** **NOT AVAILABLE to confirm.**
7. **Concentrated in trending conditions?** **NOT AVAILABLE to confirm.**
8. **Is spread/cost materially contributing?** **Plausible, not
   confirmed** — 3 of 6 strategies are structurally cost-fragile.
9. **Is execution materially different from backtest?** **One confirmed,
   documented gap** (AMR live BE-exit logic vs. researched refinement)
   — real but not obviously large enough alone to explain a PF-0.30 result.
10. **Is JPY concentration causing simultaneous losses?** **Structurally
    plausible** (94%+ JPY exposure, historically-positive correlations)
    — **not confirmed for the current period.**
11. **Is the current drawdown (~4%) statistically unusual?** Per the
    prior corrected Monte Carlo (`reports/current_6_strategy_revalidation.md`
    §14, EXP-105-111): **no**, within/near the historically-expected range.
12. **Is the current losing streak statistically unusual?** Per the same
    source: **elevated but plausible**, not extreme.
13. **Is current 5ers performance consistent with historical backtests?**
    **On the aggregate PF/win-rate dimension specifically: no** — the
    resampling test in §14 is the clearest evidence in this report, and
    it says the observed rate statistics are not well-explained by the
    validated historical population alone.
14. **Is demo behaving differently?** **Cannot be determined** — no
    signal-level match was possible (§17); demo's own aggregate
    performance was not separately computed in this pass.
15. **Is there evidence live implementation differs from validated
    implementation?** **One confirmed, minor gap** (AMR BE-exit logic);
    no other divergence found or newly discovered in this pass.
16. **Which strategy deserves the most confidence right now?** **GBPJPY
    AMR and GBPUSD Monday Drift** — per the standalone revalidation
    (unrelated to current-period attribution, which is unavailable).
17. **Which deserves the least?** **CADJPY AMR** — same basis.
18. **Has the portfolio entered an unfavorable regime?** **Cannot be
    confirmed** without dated current trades; structurally plausible
    given known regime sensitivities, not verified.
19. **Is there evidence of an actual broken edge?** **No** — the full
    revalidation (EXP-105-111) found no strategy with evidence
    contradicting its underlying edge; §14's resampling gap is better
    explained by data-attribution ambiguity or sample size than by edge
    failure, given the absence of any other corroborating evidence.
20. **Single most important finding of this investigation:** **The
    account-level aggregate performance (PF~0.30, win rate~30%) is
    statistically incompatible with ordinary sampling variance from the
    current 6 strategies' own validated historical trade population
    (0.00th percentile of 20,000 resampled draws) — but this session
    cannot determine why, because no 5ers trade-level data was
    accessible, and a specific, unresolved ambiguity (whether the ~33
    trades include pre-2026-07-31 trades from the now-demoted GBPJPY
    ARB / XAUUSD ARB) could fully or partially explain the gap without
    implicating any of the current 6 strategies at all.**

---

## Final recommendation

# **C. FURTHER VALIDATION REQUIRED BEFORE ANY CHANGE**

Not A (continue unchanged without comment) — the resampling result in
§14 is too large a gap to treat as routine noise without further
investigation. Not B (monitor) alone — monitoring without first
resolving the pre/post-demotion trade-attribution ambiguity in §2 risks
drawing conclusions from contaminated data. Not D (strategy review
required) — there is no evidence of a broken edge in any individual
strategy's own standalone record, so a full strategy review is not yet
justified by what this session was able to establish.

**The specific "further validation" this recommends (not implemented
here): obtain the actual 5ers trade-level history** (via the dashboard
`/api/trades` and `/api/journal` endpoints, or direct MT5 query on the
machine where the 5ers terminal runs) **and re-run this exact forensic
protocol with real data** — every section marked NOT AVAILABLE in this
report becomes answerable once that data exists. This is the single
most valuable, concrete next step this investigation identified.

---

## What I did NOT do (per instructions)

- Did not modify any strategy, entry/exit logic, parameter, or risk.
- Did not pause, deploy, or change any account or configuration.
- Did not implement the AUDJPY BUY-only candidate.
- Did not optimize anything or search for replacement strategies.
- Did not start non-JPY diversification research.
- Did not fabricate trade-level 5ers data to fill the deliverable CSVs
  — see the accompanying CSVs for exactly what is and is not populated,
  and why.
- Did not try to make the account, the strategies, or this project look
  better or worse than the evidence supports.
