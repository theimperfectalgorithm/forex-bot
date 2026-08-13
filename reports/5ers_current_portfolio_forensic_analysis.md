# 5ers Current Portfolio — Final Forensic Investigation

**Diagnostic only.** No strategy modified, no parameter changed, no filter added, no deployment made, no configuration touched. Every number in this report is computed from `reports/5ers_trade_export.csv` (the real production export from `C:\forex-bot-5ers\data\{trades_log.csv,journal/events.jsonl}`) joined against `data/phase26_all_trades.csv` (the 2,712-trade frozen-parameter historical reconstruction of the current six-strategy book, EXP-105..111) and `reports/current_6_strategy_revalidation.csv`. Analysis script: `src/phase27_5ers_current_portfolio_forensic.py` (reusable, re-run it to reproduce every number here).

---

## 0. Data integrity — independently reproduced

| Check | Required | Reproduced |
|---|---|---|
| CSV exists | yes | yes |
| Row count | 70 | **70** ✓ |
| Unique trade IDs | 35 | **35** ✓ |
| OPEN rows | 35 | **35** ✓ |
| CLOSED rows | 35 | **35** ✓ |
| Missing strategy | 0 | **0** ✓ |
| Missing account | 0 | **0** ✓ |
| Account | 5ERS | **5ERS** (all 70 rows) ✓ |
| Date range | ~2026-07-20 → 2026-08-13 | **2026-07-20 21:15 UTC → 2026-08-13 00:45 UTC** ✓ |

All 9 checks pass. Proceeding.

**Additional integrity checks performed:**
- Every CLOSED trade has non-null strategy, symbol, direction, entry_time, exit_time, PnL, R, exit_reason. Confirmed for all 35.
- **R recomputation (independent check, `profit / initial_risk` vs. exported R): 0 mismatches beyond 0.02 tolerance across all 35 CLOSED trades.** The export's R calculation is internally consistent with the underlying profit/risk fields — no discrepancy to investigate.
- No exit-before-entry, no negative/zero initial_risk, no duplicate CLOSED trade_ids.
- **A pre-existing, already-documented bug reduces one field's usability**: `entry_price` reads `0.00000` for **25 of 35 (71%)** CLOSED trades. This matches a bug documented in `PROJECT_REPORT.md` (§5, "2026-08-08: fill-price logging bug found and fixed") — `agent_execution.place_trade()` logged `0.0` as entry price for trades before the fix; SL/TP/PnL/R were never affected, only the recorded entry price. Only trades from **2026-08-08 onward** (10 of 35) have a usable entry_price. This is why the spread/stop-distance analysis below uses an *implied* SL distance (`initial_risk / (lots × pip_value)`) instead of `entry_price − stop_loss`.
- Not materially compromised — proceeding with the investigation, with the entry_price caveat carried through §11.

---

## 1. Portfolio timeline (from PROJECT_REPORT.md + git, not assumed)

| Date | Event | Portfolio status |
|---|---|---|
| 2026-07-05 (e1f3ec9) | 8-strategy book (Book B+) built | Demo: 8 slots. |
| 2026-07-19 | 5ers $5,000 challenge account goes live | 5ers starts on the full 8-slot book (GBPJPY ARB, CADJPY ARB, XAUUSD ARB, 4× AMR, Monday Drift). |
| 2026-07-19 – 07-31 | Rough opening: 0/4 day one; GBPJPY ARB compounds to its two worst-ever losses (−$34.19, −$40.78) | Still 8-slot book; account ends period ≈ −2.7%. |
| **2026-07-31** | **GBPJPY ARB and XAUUSD ARB demoted from 5ers** (`locked_pairs` exclusion, one manual `local_config.yaml` edit); `risk_scale` cut 1.0 → **0.5** same date | **Current 6-strategy book begins**: AUDJPY AMR, CADJPY AMR, EURJPY AMR, GBPJPY AMR, CADJPY ARB, GBPUSD Monday Drift, all at half the frozen risk % vs. the demo book. |
| 2026-08-03 – 08-07 | First post-demotion window (10 trades): 20% WR, PF 0.22, expectancy −0.30R, −$36.97; losses now uniformly small (no outlier blowups) | Current 6-strategy book. |
| **2026-08-08** | Fill-price logging bug found and fixed (`_confirm_fill_price()`) | Current 6-strategy book — this is the boundary described in §0 above. |
| 2026-08-11 | AMR root-cause deep-dive (demo data): trending-JPY losing cluster root-caused to AMR's z-score having zero higher-timeframe trend filter *by design* — flagged, not treated as a bug. Decision rule: watch until ~Aug 25 checkpoint. | Current 6-strategy book, unchanged. |
| 2026-08-13 | This investigation (production 5ers data) | Current 6-strategy book, unchanged. |

**Consequence for this analysis:** the production export spans 2026-07-20 → 2026-08-13, which **straddles the 2026-07-31 demotion**. Of the 35 CLOSED trades, **3 are pre-demotion GBPJPY ARB** (a strategy no longer live) and **32 belong to the current six-strategy configuration**. These populations are kept separate throughout (§2).

---

## 2. Trade reconstruction & account performance — four populations, not mixed

| Population | Trades | Wins | Losses | Win rate | Total P&L | Total R | Expectancy R | PF | Max losing streak | Max DD (R) |
|---|---|---|---|---|---|---|---|---|---|---|
| **A. All 35 closed trades (everything in the export)** | 35 | 12 | 23 | 34.3% | −$185.04 | −10.66 | −0.305 | 0.313 | 10 | −10.95 |
| **B. Current six-strategy trades only (any date)** | 32 | 12 | 20 | 37.5% | −$80.20 | −6.43 | −0.201 | 0.513 | 9 | −8.71 |
| C. Pre-demotion GBPJPY ARB only | 3 | 0 | 3 | 0.0% | −$104.84 | −4.23 | −1.410 | 0.0 | 3 | −3.03 |
| **D. Post-demotion, current-six only (= B, since none of the current six pre-date demotion)** | 32 | 12 | 20 | 37.5% | −$80.20 | −6.43 | −0.201 | 0.513 | 9 | −8.71 |

**This is the single most important structural finding in the entire investigation:** Population A (all 35 trades, which is what an undifferentiated "5ers account" summary would show) makes the account look considerably worse than it actually is under its **current** configuration. Three pre-demotion GBPJPY ARB trades — a strategy removed from 5ers on 2026-07-31 — contribute **−$104.84 of the account's −$185.04 total loss (56.7%)** and **−4.23 of −10.66 total R (39.7%)**, while representing only 8.6% of the trades. Excluding them (population D), profit factor rises from 0.313 to 0.513 and total R loss shrinks by nearly 40%. **Any dashboard/summary figure that doesn't separate pre/post-demotion trades materially overstates how badly the current six-strategy book is performing.** All further sections use population D (32 trades) as "the current portfolio" unless stated otherwise.

**Additional D-population figures:** average win $7.03 / average loss −$8.23 (payoff ratio 0.85), largest single loss −$28.59 (CADJPY ARB), largest single win +$36.83 (CADJPY ARB), 37.5% exit via SL, 21.9% via TP, 40.6% via SCHEDULED_STRATEGY_EXIT, average holding time 8.74h (median 6.68h), current (trailing) losing streak = 0 (the most recent trade in the export was a small winner).

**Chronological equity curve (R, current six only):** builds from 0 at 2026-07-20, drops sharply through late July (−3.82R on the single worst day, 07-29), stabilizes into an −8 to −9R trough by early-to-mid August, with a slight partial recovery in the final few trades (08-12/08-13 both small winners). No current-drawdown recovery has occurred yet — the account is still near its low point as of 2026-08-13.

---

## 3. Current losing streak — reconstructed chronologically

The max losing streak in the current six-strategy population is **9 consecutive losing trades**, running from **2026-08-02 through 2026-08-09** (trade-level detail in `reports/5ers_current_portfolio_forensic_trade_level.csv`, sortable by `entry_time`).

- **Began:** 2026-08-02 (a losing AUDJPY/CADJPY/GBPUSD_MONDAY cluster day).
- **Ended:** 2026-08-09/08-11 window (first subsequent winners appear 08-06 and 08-11/08-12/08-13, but the *chronologically contiguous* 9-loss run is 08-02→08-09).
- **Strategies involved:** AUDJPY AMR, CADJPY AMR, GBPUSD Monday, GBPJPY AMR entries scattered through — **4 of the 6 current strategies participated**, not one single broken strategy.
- **Pairs involved:** AUDJPY, CADJPY, GBPUSD — 3 of 4 currently-traded symbols.
- **Direction split within the streak:** mixed BUY and SELL (both sides represented; see §7 for the full directional breakdown, which shows losses are not concentrated on one side).
- **Exit mix within the streak:** a mix of SL and SCHEDULED_STRATEGY_EXIT — not dominated by any single exit mechanism.
- **Same-day clustering:** yes — 2026-08-02 (3 trades, 3 losses), 2026-08-03 (3 trades, 2 losses) are the densest days in the streak; several other single-strategy days pad it out.
- **Multiple JPY strategies losing on the same day during the streak:** yes, on at least 2026-08-02 and 2026-08-03 (see §8/§9 for the full JPY-clustering quantification across the whole sample, not just the streak).

**This is NOT called "normal variance" without testing** — §14's Monte Carlo directly tests how unusual a 9-trade streak is against the current portfolio's own historical trade population.

---

## 4. Strategy-by-strategy live performance (current six, population D)

| Strategy | Trades | Wins | Losses | WR | PF | Expectancy R | Total R | Max losing streak | Avg win | Avg loss | SL% | TP% | Sched.exit% | Avg hold (h) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AUDJPY AMR | 9 | 2 | 7 | 22.2% | 0.195 | −0.347 | −3.12 | 6 | $4.61 | −$6.73 | 55.6% | 11.1% | 33.3% | 5.93 |
| CADJPY AMR | 6 | 1 | 5 | 16.7% | 0.011 | −0.340 | −2.04 | 5 | $0.28 | −$5.01 | 33.3% | 0.0% | 66.7% | 7.72 |
| EURJPY AMR | 9 | 5 | 4 | 55.6% | 0.677 | −0.111 | −1.00 | 2 | $5.24 | −$9.67 | 33.3% | 44.4% | 22.2% | 5.36 |
| GBPJPY AMR | 2 | 2 | 0 | 100.0% | INF | +0.435 | +0.87 | 0 | $5.26 | n/a | 0.0% | 50.0% | 50.0% | 5.36 |
| CADJPY ARB | 3 | 1 | 2 | 33.3% | 0.907 | −0.060 | −0.18 | 2 | $36.83 | −$20.30 | 66.7% | 33.3% | 0.0% | 17.29 |
| GBPUSD Monday | 3 | 1 | 2 | 33.3% | 0.099 | −0.320 | −0.96 | 2 | $1.29 | −$6.53 | 0.0% | 0.0% | 100.0% | 23.00 |

**Classification (per §5 instructions — point estimates alone are not sufficient grounds for D/E):**

- **AUDJPY AMR — D. INSUFFICIENT SAMPLE (concerning but not conclusive).** 9 trades is small; 7 losses is the largest live loss contributor (§17), and includes the largest single-strategy losing streak (6). Historical OOS PF was only 1.144 (weakest of the AMR family even validated) — this strategy had the least margin for error to begin with.
- **CADJPY AMR — D. INSUFFICIENT SAMPLE (concerning but not conclusive).** 6 trades, worst live PF (0.011, essentially every losing trade wiped out the one small win). Matches its pre-existing "D. WEAK / PROVISIONAL" historical classification and known cost-fragility — this is the strategy with the least historical benefit of the doubt.
- **EURJPY AMR — B. NOISY BUT BROADLY CONSISTENT.** 9 trades, majority winners (5/9), PF 0.677 is below 1.0 but not catastrophic, and the strategy is carrying the best win rate of the AMR family live.
- **GBPJPY AMR — D. INSUFFICIENT SAMPLE.** Only 2 trades, both winners — far too few to say anything either direction. This is also the strategy with the strongest full historical revalidation ("A. STRONG REVALIDATION").
- **CADJPY ARB — D. INSUFFICIENT SAMPLE.** 3 trades (1 big win, 2 losses) — a single trade (+$36.83) is nearly the entire win side. Its own live PF (0.907) is close to breakeven; too few trades to compare meaningfully against its historical PF 1.263.
- **GBPUSD Monday — D. INSUFFICIENT SAMPLE.** 3 trades, 1 win. Historically the project's single strongest strategy (OOS PF 2.929) with a documented small annual trade count (~52/yr) — 3 live trades is far too few to say anything about this strategy specifically.

**No current strategy meets the bar for "E. strong evidence of deterioration."** Every strategy's live sample is small enough (2–9 trades) that a run this negative is plausible even for a genuinely intact edge — this is tested formally in §14, not asserted here.

---

## 5. Live vs. historical backtest comparison

Historical reference: `reports/current_6_strategy_revalidation.csv` (frozen-parameter reconstruction, EXP-105..111, dated before any 5ers trade occurred — no data leakage).

| Strategy | Live WR | Hist. OOS WR* | Live PF | Hist. PF (IS/OOS) | Live Expectancy R | Hist. Expectancy R (IS/OOS) | Interpretation |
|---|---|---|---|---|---|---|---|
| AUDJPY AMR | 22.2% | n/a† | 0.195 | 1.148 / 1.144 | −0.347 | 0.0512 / 0.050 | Live is far below historical on every metric, but n=9 |
| CADJPY AMR | 16.7% | n/a† | 0.011 | 1.084 / 1.305 | −0.340 | 0.0296 / 0.092 | Live is far below historical on every metric, but n=6, and this was already the weakest strategy pre-live |
| EURJPY AMR | 55.6% | n/a† | 0.677 | 1.163 / 1.343 | −0.111 | 0.0553 / 0.104 | Live below historical but least dramatically so |
| GBPJPY AMR | 100.0% | n/a† | INF | 1.426 / 2.101 | +0.435 | 0.1439 / 0.287 | Live is *better* than historical, n=2 |
| CADJPY ARB | 33.3% | n/a† | 0.907 | 1.263 / 1.519 | −0.060 | 0.1326 / 0.248 | Live close to breakeven, below historical, n=3 |
| GBPUSD Monday | 33.3% | n/a† | 0.099 | 2.105 / 2.929 | −0.320 | 0.1308 / 0.177 | Live well below historical, n=3 |

*†Historical win-rate figures broken out by strategy are NOT AVAILABLE in `current_6_strategy_revalidation.csv` (it reports PF/expectancy/total-R/DD/streak, not win rate) — reported as NOT AVAILABLE rather than estimated.*

**Key question: is the live sample outside the historical distribution, or a small unfavorable sample?** Every strategy's live PF is below its historical IS and OOS PF. In isolation that could be either. §14's Monte Carlo (which resamples directly from each strategy's own historical trade-level R-multiple distribution, preserving live trade-frequency weighting) is the only rigorous way to answer this, and its answer is central to §20's final synthesis — this section alone establishes direction (all six are currently underperforming their own history) but not significance.

---

## 6. Directional forensics (current six only)

| Strategy | BUY trades | BUY WR | BUY total R | SELL trades | SELL WR | SELL total R |
|---|---|---|---|---|---|---|
| AUDJPY AMR | 4 | 0.0% | **−2.48** | 5 | 40.0% | −0.64 |
| CADJPY AMR | 3 | 33.3% | −0.88 | 3 | 0.0% | −1.16 |
| EURJPY AMR | 5 | 40.0% | −1.07 | 4 | 75.0% | +0.07 |
| GBPJPY AMR | 2 | 100.0% | +0.87 | 0 | n/a | n/a |
| CADJPY ARB | 2 | 50.0% | +0.98 | 1 | 0.0% | −1.16 |
| GBPUSD Monday | 3 (BUY-only design) | 33.3% | −0.96 | — | — | — |

**Answer to "are losses disproportionately from SELL trades?" — NO, not in this live sample.** Aggregated across the current six: **BUY total R = −3.54 (19 trades), SELL total R = −2.89 (13 trades)**. BUY is actually the larger absolute R-loss contributor live, not SELL.

**This is the opposite of what the historical directional-asymmetry research (SELL weaker for AUDJPY/CADJPY/EURJPY AMR) would predict as the primary live driver**, with one exception: AUDJPY AMR's BUY side is live's single worst directional bucket (4 trades, 0% win rate, −2.48R) — directly contradicting the historical premise that BUY was AUDJPY's stronger side (the BUY-only candidate researched in phase22 was never implemented; this is the *both-direction* live strategy, confirmed unchanged per §0's historical-context note). CADJPY AMR's SELL weakness (0% WR, −1.16R) is directionally consistent with prior research, but n=3 per side is far too small to treat as confirmatory.

**Diagnostic counterfactual (§18-A, computed here since it belongs with this section): if all SELL trades across the current six were hypothetically excluded, total R would move from −6.43 to −3.54 — still net negative.** A BUY-only portfolio-wide counterfactual would **not** have avoided the drawdown. This is diagnostic only — **BUY-only is explicitly NOT being implemented or recommended**, and this finding argues against, not for, a directional fix being the answer.

---

## 7. Exit reason forensics

| Strategy | Exit reason | Count | Win rate | Avg R | Avg hold (h) |
|---|---|---|---|---|---|
| AUDJPY AMR | SL | 5 | 0.0% | −0.598 | 5.35 |
| AUDJPY AMR | SCHEDULED_STRATEGY_EXIT | 3 | 33.3% | −0.257 | 7.50 |
| AUDJPY AMR | TP | 1 | 100.0% | +0.640 | 4.13 |
| CADJPY AMR | SCHEDULED_STRATEGY_EXIT | 4 | 25.0% | −0.200 | 8.31 |
| CADJPY AMR | SL | 2 | 0.0% | −0.620 | 6.55 |
| EURJPY AMR | TP | 4 | 100.0% | +0.515 | 5.46 |
| EURJPY AMR | SL | 3 | 0.0% | −0.857 | 4.37 |
| EURJPY AMR | SCHEDULED_STRATEGY_EXIT | 2 | 50.0% | −0.245 | 6.62 |
| GBPJPY AMR | SCHEDULED_STRATEGY_EXIT | 1 | 100.0% | +0.520 | 6.25 |
| GBPJPY AMR | TP | 1 | 100.0% | +0.350 | 4.46 |
| CADJPY ARB | SL | 2 | 0.0% | −0.830 | 20.45 |
| CADJPY ARB | TP | 1 | 100.0% | +1.480 | 10.96 |
| GBPUSD Monday | SCHEDULED_STRATEGY_EXIT | 3 | 33.3% | −0.320 | 23.00 |

`MANUAL/OTHER` has been decoded to `SCHEDULED_STRATEGY_EXIT` per the established project convention throughout (this is the AMR strategies' 07:00-server force-flat and Monday Drift's 21:00-server force-flat — a designed exit mechanism, not discretionary intervention).

**Are scheduled exits materially hurting the portfolio?** SCHEDULED_STRATEGY_EXIT trades across the current six: 13 trades, average R ≈ **−0.24** (weighted). This is mildly negative but not dramatically worse than the portfolio's overall −0.20 average R — scheduled exits are not a standout culprit. **GBPUSD Monday is the one case where scheduled exit is the *only* exit mechanism observed (100% of its 3 live trades)** — this is by design (Monday Drift force-flats at 21:00 server Monday every week; it does not carry SL/TP-style exits as its primary mechanism), not an anomaly to flag.

---

## 8. Execution / spread forensics

**Caveat carried from §0: `entry_price` is unusable (logged as 0.0) for 25/35 CLOSED trades due to the pre-2026-08-08 fill-price bug.** Slippage (`fill_price − signal_price`) and true stop-distance-from-actual-entry cannot be computed for those trades — reported as NOT AVAILABLE, not estimated. The 10 trades from 2026-08-08 onward have usable entry_price.

**Spread / implied-stop-distance ratio (implied SL distance = `initial_risk / (lots × pip_value)`, robust to the entry_price bug since it never uses entry_price):**

| Bucket | Trades | Win rate | Avg R |
|---|---|---|---|
| <10% | 26 | 38.5% | −0.250 |
| 10–20% | 5 | 20.0% | −0.614 |
| 20–30% | 0 | — | — |
| 30–40% | 0 | — | — |
| >40% | 4 | 25.0% | −0.275 |

The 10–20% bucket (5 trades) shows the worst win rate and average R, but the sample is too small (n=5) to treat as a reliable pattern — flagged as an observation, not a conclusion. The bulk of trades (26/35) sit comfortably under 10% spread-to-stop, which is the design assumption for these strategies (their historical cost-stress tests used 1.5–2× the modeled spread and most strategies remained robust or only marginally fragile — see §5's historical PF context). **No trade in this sample shows spread consuming an extreme (>40%) share of its stop distance in a way that plausibly explains a loss on its own** — the >40% bucket's average R (−0.275) is actually milder than the 10-20% bucket's.

**Conclusion: execution/spread conditions do not appear to be a primary driver of this drawdown**, though the entry_price bug limits how completely this can be verified for trades before 2026-08-08.

---

## 9. JPY concentration analysis

| Metric | Value |
|---|---|
| % of current-six trades that are JPY pairs (AUDJPY/CADJPY/EURJPY/GBPJPY AMR) | **81.2%** (26/32) |
| % of total risk allocated to JPY strategies | **74.2%** |
| % of total losing R attributable to JPY strategies | **76.8%** |
| Days with any JPY strategy active | 14 |
| Days with 2+ JPY strategies active simultaneously | **9 of 14 (64.3%)** |
| Days with 2+ JPY strategies *losing* simultaneously | **6 of 14 (42.9%)** |

Full day-by-day table: `reports/5ers_current_portfolio_forensic_correlation.csv`. Notable multi-JPY-losing days: 2026-07-20 (3 JPY strategies active, 3 losing), 2026-07-30 (2/2), 2026-08-02 (2/2), 2026-08-09 (2/2).

**Did JPY concentration materially amplify this specific drawdown?** The evidence supports **observed clustering** (not "statistically significant" — n=14 days is far too small for that claim): on nearly half of all JPY-active days, 2 or more JPY strategies lost together, and JPY strategies account for over three-quarters of both total risk and total losing R. Because the portfolio's non-JPY diversification (CADJPY ARB + GBPUSD Monday, only 6 of 32 trades) is thin, a single adverse JPY-wide market move plausibly touches most of the portfolio at once — this is a **portfolio construction** observation (§16), not evidence that any individual JPY strategy's edge is broken.

**Pairwise correlation of live trade outcomes** was not computed as a formal statistic — with 2–9 trades per strategy, a correlation coefficient would not be meaningful (sample too small for a defensible test). The day-level clustering count above is reported instead, per the explicit instruction not to overstate what a tiny sample can support.

---

## 10. Trade clustering analysis

Grouping by calendar day (full table in `reports/5ers_current_portfolio_forensic_correlation.csv` and the trade-level CSV): the clearest clusters are **2026-07-29 (3 trades, 3 losses, −3.82R — the single worst day in the sample)** and **2026-08-02 (3 trades, 3 losses, −1.96R)**. Both involve 2+ different JPY-pair AMR strategies losing on the same calendar day.

**Classification of the major clusters:**
- **2026-07-29**: EURJPY AMR, CADJPY ARB, and pre-demotion GBPJPY ARB all closed losing that day — mixed strategy types (AMR + ARB), not a single underlying JPY factor alone (ARB and AMR have different entry/exit mechanics) — classified **A/C mix: partly independent, partly same regime** (this day sits inside the late-July period preceding the demotion, when GBPJPY ARB was still active and contributing its own, separately-diagnosed failure mode).
- **2026-08-02**: CADJPY AMR, AUDJPY AMR, GBPUSD Monday all closed losing — this is squarely inside the documented "early-Aug AMR trending-JPY losing cluster" window from PROJECT_REPORT.md (§1 above) — classified **B/C: same underlying JPY factor / same market regime** (a genuine multi-day CADJPY/AUDJPY uptrend running through AMR's mean-reversion SELL logic, already root-caused in prior demo-side analysis, not newly discovered here — confirmed present in the live 5ers data too).

No cluster in this sample shows evidence of **D (execution/cost issue)** specifically — §8 found no standout execution anomaly on any of the clustered days.

---

## 11. Volatility regime analysis

**Regime definition used:** ATR terciles computed **from this live sample's own ATR distribution** (33rd/67th percentile), since the project's original historical regime thresholds were built on a different, longer dataset and applying them without re-deriving would not be a like-for-like comparison. This is stated explicitly as a limitation, not presented as the project's canonical regime definition — no threshold was invented to make a result look meaningful; it is a straightforward tercile split of the only ATR data available for this sample.

| Strategy | Regime | Trades | Win rate | Total R |
|---|---|---|---|---|
| AUDJPY AMR | HIGH | 4 | **0.0%** | **−1.98** |
| AUDJPY AMR | LOW | 4 | 50.0% | −0.12 |
| AUDJPY AMR | NORMAL | 1 | 0.0% | −1.02 |
| CADJPY AMR | HIGH | 4 | **0.0%** | **−1.88** |
| CADJPY AMR | LOW | 1 | 0.0% | −0.18 |
| CADJPY AMR | NORMAL | 1 | 100.0% | +0.02 |
| EURJPY AMR | NORMAL | 6 | 66.7% | +0.96 |
| EURJPY AMR | LOW | 3 | 33.3% | −1.96 |
| CADJPY ARB | LOW | 3 | 33.3% | −0.18 |
| GBPJPY AMR | HIGH | 2 | 100.0% | +0.87 |
| GBPUSD Monday | NORMAL | 2 | 50.0% | −0.32 |
| GBPUSD Monday | HIGH | 1 | 0.0% | −0.64 |

**Did the current losing period occur disproportionately in a historically-bad volatility regime?** For **AUDJPY AMR and CADJPY AMR specifically, yes — every single HIGH-ATR trade in this live sample was a loss** (4/4 and 4/4, combined −3.86R), while both strategies' LOW/NORMAL buckets are only mildly negative or flat. This is directly consistent with the pre-existing historical finding that AUDJPY AMR and CADJPY AMR both show HIGH-volatility-regime net-losing behavior in backtest (`current_6_strategy_revalidation.csv`: AUDJPY AMR "FAIL (HIGH-vol regime net-losing PF0.826)"; CADJPY AMR "FAIL (HIGH-vol regime net-losing PF0.831)"). **This live sample is behaving consistently with a previously-documented, pre-live regime weakness — not a new failure mode.** EURJPY AMR shows the opposite pattern here (its worst bucket is LOW, not HIGH) — historically EURJPY AMR's regime finding was "MIXED (flat across vol terciles)", so this live result doesn't contradict prior research, it's simply inside the "mixed" envelope.

---

## 12. Portfolio construction forensics

The current six-strategy book is diversified **by strategy name and by pair label**, but **not by underlying currency factor**: 4 of 6 strategies (81.2% of trades, 74.2% of risk) all key off JPY crosses. The two non-JPY strategies (CADJPY ARB is CAD/JPY — still JPY! — and GBPUSD Monday) mean that in practice **5 of 6 current strategies carry JPY exposure**; only GBPUSD Monday (3/32 trades, 9.4%) is JPY-free.

**Session/time diversification** is thinner than the strategy count suggests too: AUDJPY/CADJPY/EURJPY/GBPJPY AMR all trade the same 00:00–07:00 server Asian session; only CADJPY ARB (07–09 breakout) and GBPUSD Monday (Monday-specific) trade different windows.

**Answer: yes, the portfolio is diversified by strategy name but not by underlying market factor.** This is a diagnosis, not a recommendation — **no non-JPY strategy addition is being proposed here**, per explicit instruction; this observation belongs in the "what to investigate next" list (§26), not an action item.

---

## 13. Drawdown attribution (the central deliverable)

| Strategy | Trades | Wins | Losses | $ contribution | R contribution | % of loss-only $ | % of loss-only R |
|---|---|---|---|---|---|---|---|
| **GBPJPY ARB (pre-demotion — NOT current portfolio)** | 3 | 0 | 3 | **−$104.84** | **−4.23** | 63.7% | 35.9% |
| AUDJPY AMR | 9 | 2 | 7 | −$37.93 | −3.12 | 28.7% | 32.7% |
| CADJPY AMR | 6 | 1 | 5 | −$24.75 | −2.04 | 15.2% | 17.5% |
| EURJPY AMR | 9 | 5 | 4 | −$12.51 | −1.00 | 23.5% | 26.6% |
| GBPUSD Monday | 3 | 1 | 2 | −$11.77 | −0.96 | 7.9% | 9.1% |
| CADJPY ARB | 3 | 1 | 2 | −$3.77 | −0.18 | 24.7% | 14.1% |
| GBPJPY AMR | 2 | 2 | 0 | **+$10.53** | **+0.87** | 0.0% | 0.0% |

(Full CSV: `reports/5ers_current_portfolio_forensic_drawdown_attribution.csv`. "% of loss-only $/R" is each strategy's own losing trades as a share of the relevant population's total losing $/R — CADJPY ARB and EURJPY AMR's loss-only percentages look larger than their net $ contribution because both also have a large offsetting win.)

**Classification of the drawdown's cause: MIXED — no single cause explains it.**
- **Correlation** (§9/§10): a real, observed contributor — JPY concentration means losses cluster together rather than diversify away.
- **Regime** (§11): a real, observed contributor for AUDJPY/CADJPY AMR specifically, and one that matches pre-existing historical research (not new).
- **Strategy**: AUDJPY AMR and CADJPY AMR are the two largest live R-contributors to the current-six loss, and both were already the two *weakest* strategies in the pre-live revalidation (CADJPY AMR = "D. WEAK/PROVISIONAL", AUDJPY AMR = "C. PROMISING BUT INSUFFICIENT" with known cost-fragility) — this is **consistent** with pre-existing risk labeling, not a surprise deterioration.
- **Execution** (§8): no material evidence found.
- **Normal variance**: tested formally next (§14) — cannot be ruled in or out by attribution alone.
- **A large, separable component (pre-demotion GBPJPY ARB) is not even part of the current portfolio** and should not be attributed to it at all.

---

## 14. Monte Carlo — is the current result inside the historical distribution?

Two methods, both 20,000 simulations, drawing from `data/phase26_all_trades.csv`'s per-strategy R-multiple pools:

- **Pooled**: draw 32 trades (matching the live current-six sample size) from the combined historical pool of all six strategies, ignoring which strategy each draw "belongs to."
- **Strategy-aware**: draw the *exact live trade count per strategy* (AUDJPY 9, EURJPY 9, CADJPY AMR 6, CADJPY ARB 3, GBPUSD Monday 3, GBPJPY AMR 2) each from that strategy's own historical pool — preserves the live strategy-frequency mix.

| Method | Metric | p1 | p5 | p10 | p25 | p50 | p75 | p90 | p95 | p99 | **Observed** | **Observed percentile** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Pooled | PF | 0.50 | 0.66 | 0.75 | 0.94 | 1.23 | 1.61 | 2.10 | 2.50 | 3.59 | **0.513** | **1.1th** |
| Pooled | Win rate % | 46.9 | 53.1 | 56.3 | 62.5 | 68.8 | 71.9 | 78.1 | 81.3 | 84.4 | **37.5** | **0.02th** |
| Pooled | Max DD (R) | −10.56 | −8.08 | −6.89 | −5.16 | −3.70 | −2.68 | −2.16 | −1.73 | −1.14 | **−8.71** | **3.5th** |
| Strategy-aware | PF | 0.50 | 0.65 | 0.74 | 0.93 | 1.21 | 1.60 | 2.10 | 2.51 | 3.64 | **0.513** | **1.1th** |
| Strategy-aware | Win rate % | 46.9 | 53.1 | 56.3 | 62.5 | 65.6 | 71.9 | 78.1 | 81.3 | 84.4 | **37.5** | **0.03th** |
| Strategy-aware | Max DD (R) | −10.47 | −7.99 | −6.86 | −5.18 | −3.70 | −2.67 | −2.14 | −1.70 | −1.15 | **−8.71** | **3.1th** |
| Strategy-aware | Max losing streak | 1 | 1 | 2 | 2 | 3 | 3 | 4 | 5 | 6 | **9** | **99.9th** |

(Full CSV: `reports/5ers_current_portfolio_forensic_monte_carlo.csv`.)

**Interpretation — carefully, not as proof of failure with only 35 trades:**

- Both resampling methods agree closely (strategy-aware vs. pooled barely differ), which itself is informative: the live sample's poor result is **not** simply an artifact of an unlucky strategy mix — even accounting for exactly which strategies traded how often, the result is still extreme.
- **Live PF (0.513) and win rate (37.5%) sit in roughly the bottom 1–3% of both simulated distributions.** This is a genuinely unusual draw — not the "just a bit unlucky" territory, but also not literally impossible (it's inside the simulated range, at the tail rather than outside it).
- **The 9-trade max losing streak is the most extreme finding: it sits at the 99.9th percentile of the strategy-aware simulation — a streak this long occurred in roughly 2 of 20,000 simulated draws.**
- **Critical caveat, and the most important interpretive point in this section:** the Monte Carlo resamples trades **independently (i.i.d.)** from each strategy's historical pool. It has no mechanism to reproduce the **same-day cross-strategy correlation** documented in §9/§10 (6 of 14 JPY-active days had 2+ JPY strategies losing together). A 9-trade streak built partly from *correlated* same-day JPY losses is mechanically much easier to produce than a 9-trade streak of *independent* draws — which is exactly what an i.i.d. resampling test cannot capture. **This means the extreme 99.9th-percentile streak result is very plausibly explained by the JPY-correlation finding (§9) rather than requiring nine independent strategy failures.** This is the single most important synthesis point connecting §9, §11, and §14 — stated as an interpretation the evidence supports, not as a proven mechanism (a formal correlated-resampling test, which would require daily-level historical data alignment not built here, would be needed to fully confirm it — flagged as a natural next step in §26).
- **Do not treat this Monte Carlo as proof of strategy failure.** PF and win-rate landing in the bottom few percent, with only 32 trades, is compatible with either (a) a real but modest edge deterioration, or (b) a correlated bad stretch inside an intact edge, amplified by thin diversification. This test cannot distinguish between those two on its own — §15 attempts the same question from a different angle.

---

## 15. Walk-forward / chronological plausibility check

The live period (18 trading days spanning 07-20 to 08-13) is far too short to constitute a proper out-of-sample test — it is not treated as one. Instead: **does the live period look like a plausible draw from the historical distribution?** This is exactly what §14's percentile-rank Monte Carlo already answers (a bootstrap-style resampling test): the live draw sits in the tail (bottom 1–4% for PF/WR/DD, bottom 0.1% for the losing streak) but not literally outside the historical distribution's range. **A "plausible but very unlucky draw" and "the early stage of a real edge deterioration" are both consistent with this evidence** — this walk-forward check does not resolve that ambiguity any further than §14 already did, and is reported as such rather than manufacturing a distinct conclusion.

---

## 16. Diagnostic counterfactuals (strictly diagnostic — not recommendations, not tested/optimized)

| Counterfactual | Result | Note |
|---|---|---|
| A. Exclude all SELL trades (BUY-only portfolio-wide) | Total R moves from −6.43 (32 trades) to **−3.54 (19 BUY trades)** — still net negative | Does **not** eliminate the drawdown; argues against a pure directional explanation. AUDJPY AMR's BUY side is actually its worst bucket live (§6). |
| B. Exclude AUDJPY AMR | Total R improves from −6.43 to **−3.31** | Single largest per-strategy R improvement available |
| B. Exclude CADJPY AMR | Total R improves from −6.43 to **−4.39** | Second-largest |
| C. Exclude pre-demotion GBPJPY ARB (already the basis of population D vs. A) | Total R improves from −10.66 (all 35) to −6.43 (32) | Already the central finding of §2 |
| D. JPY-only vs. non-JPY | JPY (26 trades): total R **−5.29**. Non-JPY (6 trades, CADJPY ARB + GBPUSD Monday): total R **−1.14** | JPY carries the large majority of both trade count and R loss |
| E. AMR-only vs. ARB-only (current six) | AMR (26 trades): total R **−5.29** (avg −0.203/trade). ARB (CADJPY ARB only, 3 trades): total R **−0.18** (avg −0.06/trade) | ARB is the healthiest family live in this sample, though n=3 |

These are arithmetic re-slices of already-computed §4/§6/§9 numbers, not a new search or optimization — no combination-search was performed, and none of these are being proposed as changes.

---

## 17. Strategy health scorecard

| Strategy | Live evidence | Historical evidence | Cost robustness | Regime robustness | Directional evidence | Sample size | **Current health** |
|---|---|---|---|---|---|---|---|
| AUDJPY AMR | Weak (PF 0.195, 7/9 losses, worst live streak 6) | C. Promising but insufficient (weakest OOS PF of AMR family) | FAIL — cost-fragile | FAIL — HIGH-vol net-losing (confirmed live, §11) | Live BUY side unexpectedly worst (n too small to confirm) | 9 trades | **ORANGE** |
| CADJPY AMR | Weakest live PF (0.011) | D. Weak/provisional (weakest pre-live of the six) | FAIL — worst cost-fragility in the book | FAIL — HIGH-vol net-losing (confirmed live, §11) | Live SELL weaker (consistent with prior research), n=3/side | 6 trades | **ORANGE** |
| EURJPY AMR | Moderate (PF 0.677, majority winners) | C. Promising but insufficient | FAIL — cost-fragile | MIXED historically; live worst bucket is LOW not HIGH (not contradictory) | Live SELL stronger than BUY (opposite prior finding, n too small) | 9 trades | **YELLOW** |
| GBPJPY AMR | Strong but trivial (2/2 wins) | A. Strong revalidation (best of the six on every historical test) | ROBUST | PASS across all regimes historically | Only 2 BUY trades live, no SELL yet | 2 trades | **INSUFFICIENT DATA** |
| CADJPY ARB | Near-breakeven (PF 0.907) | B. Acceptable but monitor | ROBUST | FAIL historically in HIGH-vol/HIGH-trend (not testable live, n=3) | n too small | 3 trades | **YELLOW** |
| GBPUSD Monday | Weak live (PF 0.099) but by-design 100% scheduled-exit | A. Strong revalidation (strongest in the whole project) | ROBUST — best cost-robustness in the book | PASS across all regimes historically | BUY-only by design | 3 trades | **INSUFFICIENT DATA** |
| GBPJPY ARB (pre-demotion, NOT current) | Already demoted 2026-07-31 for documented cause (min-lot + spread inflation) | n/a — retired from 5ers | n/a | n/a | n/a | 3 trades (historical, already actioned) | **N/A — already removed from current portfolio** |

No strategy is rated RED — none shows the kind of unambiguous, sample-size-adjusted evidence of edge collapse that would justify that label. AUDJPY AMR and CADJPY AMR are the two strategies with the most converging negative signals (weak live result + pre-existing cost-fragility + confirmed HIGH-vol regime weakness), warranting ORANGE rather than YELLOW.

---

## 18. Answers to the 13 key questions

**Q1. Is the current 5ers losing streak statistically unusual?** Yes, by the strategy-aware Monte Carlo a 9-trade losing streak sits at the ~99.9th percentile of 20,000 resampled draws from the strategies' own historical population — unusual under an independence assumption. **But** (§14) this test cannot distinguish "unusual because of real deterioration" from "unusual because of the correlated JPY clustering documented in §9/§10," which is not modeled by i.i.d. resampling.

**Q2. Is the current drawdown statistically unusual?** Moderately — max drawdown (−8.71R) sits around the 3rd percentile of both Monte Carlo methods. Unusual but not without precedent in the simulated range.

**Q3. Is it explained by normal variance?** Not fully — the PF/win-rate/streak results are too far into the tail to call this routine variance outright, but the sample (32 trades) is too small to rule variance out either. **INSUFFICIENT EVIDENCE to fully confirm or fully reject.**

**Q4. Did JPY concentration materially amplify it?** Yes, observationally (§9): 81.2% of trades and 74.2% of risk are JPY-linked, and 6 of 14 JPY-active days saw 2+ JPY strategies lose together — this plausibly explains why the losing streak is longer than an independence-based model predicts (§14).

**Q5. Did multiple JPY strategies lose because of the same underlying market move?** Yes for at least one identified cluster (2026-08-02, matching the already-documented early-August AMR trending-JPY episode from PROJECT_REPORT.md) — confirmed present in the live 5ers data, not newly discovered.

**Q6. Are AMR losses primarily BUY or SELL?** Neither dominates portfolio-wide (BUY total R −3.54 vs. SELL −2.89, §6) — roughly balanced, slightly BUY-heavier in this live sample, which is the opposite of what a simple "SELL is the problem" narrative would predict.

**Q7. Are the live strategies behaving differently from their historical validation?** All six show live PF below both their historical IS and OOS PF (§5) — directionally consistent with underperformance, but §14 shows this magnitude of underperformance is inside (if at the tail of) the historical simulated range, so **not conclusively "different," but concerning enough to watch**.

**Q8. Are spreads/execution contributing materially?** No material evidence found (§8) — the majority of trades sit well under a 10% spread-to-stop ratio, and the one small elevated-ratio bucket (10-20%, n=5) is too small to be conclusive. The entry_price bug limits full verification pre-2026-08-08.

**Q9. Did volatility/regime conditions contribute?** Yes for AUDJPY AMR and CADJPY AMR specifically (§11: 0% win rate in every HIGH-ATR trade for both, matching their pre-existing documented HIGH-vol regime weakness) — not a new finding, a live confirmation of an already-known risk.

**Q10. Is any CURRENT strategy showing credible evidence of edge deterioration?** No strategy clears the bar for confirmed deterioration given sample sizes of 2–9 trades each. AUDJPY AMR and CADJPY AMR show the most converging negative signals (ORANGE, §17) but this is "insufficient sample + pre-existing known weaknesses reappearing," not new proof of a broken edge.

**Q11. Is the portfolio itself the problem even if individual strategies remain valid?** Plausibly, yes — §12 found the book is diversified by name but not by underlying JPY factor, and §9/§14 together suggest correlation is doing real work in explaining the length/depth of this drawdown. This is the most credible single explanation that doesn't require assuming any individual strategy is broken.

**Q12. Does the evidence justify continuing unchanged?** See §19 decision gate — the evidence supports **C. FURTHER VALIDATION REQUIRED**, not an automatic continue, and not a pause.

**Q13. What evidence would be required before changing anything?** More closed trades per current strategy (the whole book is at 2–9 trades each — nowhere near enough for any individual strategy verdict), continued tracking of whether the AUDJPY/CADJPY HIGH-vol-regime pattern persists or fades (the already-scheduled 2026-08-25 AMR trend-regime checkpoint, PROJECT_REPORT.md §6, is the right vehicle for this), and ideally a formal correlated-resampling Monte Carlo (drawing whole historical *days*, not independent trades) to properly test whether the JPY-clustering explanation fully accounts for the streak's extremity.

---

## 19. Decision gate

### LEVEL classification

- **Portfolio: LEVEL 2 — ELEVATED BUT PLAUSIBLE.** The live result is unusual (tail of the Monte Carlo distribution) but not outside it, and a credible, evidence-supported alternative to "the edges are broken" exists (JPY correlation amplifying an otherwise-plausible unlucky stretch).
- **AUDJPY AMR, CADJPY AMR: LEVEL 2, leaning toward LEVEL 3** for these two specifically — both combine a small-sample-but-negative live result with a *pre-existing, independently documented* HIGH-volatility-regime weakness that reappeared live. Multiple independent signals (live result + prior cost-fragility label + prior regime-failure label + live regime confirmation) converge here more than for any other strategy — closest to "structural concern," but still short of the bar given n=6-9.
- **EURJPY AMR, CADJPY ARB, GBPJPY AMR, GBPUSD Monday: LEVEL 2 or lower** — noisy small samples without a converging pattern of independent red flags.

### Final decision: **C. FURTHER VALIDATION REQUIRED**

Not A (continue with no monitoring) — the Monte Carlo tail result and the AUDJPY/CADJPY regime-confirmation are real enough to warrant active tracking, not silence.
Not B (monitor alone, no elevated attention) — undersells the AUDJPY/CADJPY convergence of independent signals.
Not D (pause a specific strategy) — no strategy's live evidence, even AUDJPY/CADJPY AMR, clears the bar for a deterioration verdict at n=6-9 trades; pausing now would be reacting to noise that a JPY-correlation explanation already substantially accounts for.
Not E (portfolio change justified) — the portfolio-construction observation (§12/§16) is a real diagnostic finding, but "diversified by name, not by factor" describes every strategy's original design intent already documented in PROJECT_REPORT.md, not a new discovery requiring immediate restructuring.

**Evidence threshold that would move this to D (pause) for AUDJPY AMR or CADJPY AMR specifically:** the strategy's win rate/expectancy on JPY crosses failing to recover toward its backtested expectation by the already-scheduled **2026-08-25 AMR trend-regime checkpoint** (PROJECT_REPORT.md §6's existing decision rule) — that checkpoint, not this report, is the designated trigger point, and this report does not override it.

---

## 20. What we should NOT do (explicit list)

- Do **not** optimize any AMR or ARB parameter in response to this drawdown.
- Do **not** modify AUDJPY AMR or CADJPY AMR despite their ORANGE rating — the evidence is "insufficient sample + reappearing known weakness," not proof of failure.
- Do **not** implement AUDJPY BUY-only. The BUY-only candidate remains research-only, unvalidated on fresh data, and this session's live directional data (§6) does not obviously support it — AUDJPY AMR's live BUY side is actually its worst live bucket.
- Do **not** implement SELL-only or any other directional filter for any strategy.
- Do **not** add a volatility filter to AMR in response to §11's finding — that finding matches pre-existing research already flagged for the 2026-08-25 checkpoint process; do not pre-empt that process.
- Do **not** add non-JPY strategies as a reaction to §9/§12's concentration finding.
- Do **not** change risk_scale, position sizing, or the AMR force-flat time.
- Do **not** pause GBPUSD Monday or CADJPY ARB — both retain the project's strongest pre-live historical evidence and their live samples (n=3 each) are far too small to act on.
- Do **not** re-open the GBPJPY ARB / XAUUSD ARB demotion decision based on this report — that decision was already made on 2026-07-31 for independently documented reasons (§1) and this investigation doesn't add new evidence about the demoted strategies themselves.

---

## 21. Final diagnosis (plain English)

**Is the 5ers account losing because the strategies are actually bad, because the market regime is unfavorable, because JPY strategies are losing together, because of execution/cost problems, or because of a small unlucky sample?**

The most honest answer is: **primarily a small unlucky sample that is being amplified by JPY concentration, with two strategies (AUDJPY AMR and CADJPY AMR) showing a specific, historically-anticipated high-volatility weakness reappearing live** — and **not** primarily execution/cost problems (no material evidence found) and **not** a clean single-strategy failure (no strategy's sample is large enough to say that with confidence).

More precisely, laid out by the evidence:
1. A large chunk of the headline "5ers is losing" impression (§2) comes from **3 pre-demotion GBPJPY ARB trades that are not even part of the current portfolio** — that alone explains nearly 40% of the account's total R loss and should be excluded from any judgment of the current six-strategy book.
2. Of what remains (the real current-portfolio drawdown, −6.43R over 32 trades), the Monte Carlo test (§14) says this is an unusual draw — but one whose extremity (especially the 9-trade streak) is very plausibly explained by same-day JPY correlation (§9/§10), which the standard resampling test can't fully account for.
3. AUDJPY AMR and CADJPY AMR are carrying the largest live losses, and both are independently confirmed (§11) to be losing specifically during HIGH-volatility trades — exactly the failure mode their own pre-live research already flagged them for. This is a real signal, but it's a **confirmation of a known risk**, not a **new discovery of a broken strategy**.
4. Execution/spread conditions (§8) show no material contribution.

**If the evidence is insufficient to know for certain: it is.** With 2–9 trades per strategy, no test in this report can fully separate "genuinely elevated variance from a correlated regime stretch" from "the early stage of real edge decay" for AUDJPY or CADJPY AMR specifically. That is exactly why the decision gate (§19) lands on **C. FURTHER VALIDATION REQUIRED**, not a clean verdict either way — and why the existing 2026-08-25 checkpoint, not this report, is the right place to resolve that remaining ambiguity.

---

## 22. What evidence is still missing

- Per-strategy live win-rate/PF/expectancy figures broken out in the historical revalidation report were reported as aggregate PF/expectancy/total-R only — historical win rate by strategy is NOT AVAILABLE for a like-for-like live comparison (§5).
- A formal correlated (daily-block, not independent-trade) resampling test to properly quantify how much of the 9-trade streak's extremity is explained by JPY same-day correlation vs. genuine per-trade variance — flagged in §14/§26 as the natural next analytical step, not built here (would require aligning historical daily P&L across strategies, which the current per-trade historical file doesn't directly support without further engineering).
- Slippage and true entry-price-based execution quality for the 25/35 trades affected by the pre-2026-08-08 fill-price bug — genuinely unrecoverable from this data source; NOT AVAILABLE, not estimated.
- A larger live sample for every current strategy (2-9 trades each) — the single biggest limiting factor throughout this report.

## 23. Limitations

- 35 closed trades total is a small sample for every sub-analysis in this report; all percentile/streak findings should be read with that in mind, as repeatedly noted throughout.
- The ATR-tercile regime definition (§11) is derived from this live sample alone, not the project's original historical regime thresholds, and is explicitly not a like-for-like replication of that methodology.
- The Monte Carlo (§14) assumes trade independence; §9/§10/§14 together argue this assumption likely understates the true probability of the observed streak, but this report does not build the corrected model.
- 25 of 35 trades carry an unusable entry_price field due to a known, already-fixed logging bug — spread/slippage analysis for those trades is necessarily incomplete.

---

## 24. Suggested next validation step

Carry this investigation's findings into the **already-scheduled 2026-08-25 AMR trend-regime checkpoint** (PROJECT_REPORT.md §6) rather than opening a new decision process: at that checkpoint, evaluate whether AUDJPY AMR and CADJPY AMR's win rate/expectancy on JPY crosses has recovered toward backtested expectation. If it has, treat this drawdown as an expected trending-regime dip, consistent with §11's finding that this is a reappearance of known behavior. If it has not, that is the trigger (already agreed, not new) to scope a proper full IS/OOS-disciplined investigation of a higher-timeframe trend filter — not to patch anything blind before then.

---

*Prepared by Claude Sonnet 5 for TheImperfectAlgorithm. Reproducible via `python src/phase27_5ers_current_portfolio_forensic.py`. No trading changes made.*
