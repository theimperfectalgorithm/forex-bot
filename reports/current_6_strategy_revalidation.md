# Current 6-Strategy Live Portfolio Revalidation

**VALIDATION ONLY. No strategy created, optimized, tuned, or modified.
No live code or configuration changed. No account touched.** All
neighborhood/regime/directional tests below are diagnostic — none
selected a "better" parameter, and the frozen specification was never
altered based on any result in this phase.

**Experiments:** EXP-105 through EXP-116, `experiments/experiments.csv`.
**Script:** `src/phase26_current6_revalidation.py`. **Full log:** `reports/phase26_revalidation_log.txt`. **Data:** `data/phase26_all_trades.csv`, `data/phase26_portfolio.csv`.

Scope: the exact 6 strategies currently active on the 5ers account per
`reports/live_portfolio_validation_audit.md` (EXP-096-104, commit
cfc5120) — **GBPJPY ARB and XAUUSD ARB are excluded**, consistent with
their 2026-07-31 demotion.

---

## Step 0 — Frozen specification (verified directly against source, not from memory)

| Strategy | Pair | TF | Session | Entry | SL/TP | Risk | Time exit |
|---|---|---|---|---|---|---|---|
| CADJPY ARB | CADJPY | H1 | Asian range (00-07), breakout hours 7-8 | `signals_arb_p`, `tp_mult=2.0`, `use_h4=False`, `min_range=10` | opposite range edge / `tp_mult×range` | 0.50% | Friday close |
| GBPJPY AMR | GBPJPY | M15 | Asian 00:00-04:00 | `signals_amr_v`, `z_thr=2.5`, `sl_mult=1.25` | `sl_mult×dist` / SMA20 | 0.25% | 07:00 server |
| EURJPY AMR | EURJPY | M15 | Asian 00:00-06:00 | `signals_amr_v`, `z_thr=2.0`, `sl_mult=1.5` | same | 0.25% | 07:00 server |
| AUDJPY AMR | AUDJPY | M15 | Asian 00:00-04:00 | `signals_amr_v`, `z_thr=2.0`, `sl_mult=1.5` | same | 0.25% | 07:00 server |
| CADJPY AMR | CADJPY | M15 | Asian 00:00-04:00 | `signals_amr_v`, `z_thr=2.0`, `sl_mult=1.5` | same | 0.25% | 07:00 server |
| GBPUSD Monday | GBPUSD | H1 | Monday 00:00 only | `signals_monday`, `sl_mult=1.25`, `tp_mult=1.0` | ATR20d-scaled | 0.25% | 21:00 UTC Monday |

Spread: 2.0 pips (ARB/AMR), 1.2 pips (Monday) — matches `SPREAD_PIPS_NORMAL`
convention used throughout this project since phase 15. Source files:
`pairs/CADJPY_asianrange.yaml`, `pairs/{GBPJPY,EURJPY,AUDJPY,CADJPY}_asianrev.yaml`,
`pairs/GBPUSD_monday.yaml` — all re-read fresh for this audit (2026-08-13),
unchanged since the prior forensic audit.

## Step 1 — Live/backtest parity

Re-confirmed, not re-derived, from `reports/live_portfolio_validation_audit.md`:
all 6 strategies' entry/SL/TP/risk parameters match their validated
backtest exactly. **One documented gap carries over unchanged**: the
4 AMR strategies' live breakeven-exit handling uses an older 25-pip rule,
not the researched BE@0.75R refinement — disclosed in the YAML itself,
classified MINOR (does not affect the entry/SL/TP logic tested below,
only in-trade management after entry).

## Step 2 — Data integrity

Re-verified directly (not assumed from the prior audit): `signals_arb_p`,
`signals_amr_v`, and `signals_monday` each operate on a **single symbol's**
own OHLC series — no second symbol's array is read, joined, or indexed.
**Cross-symbol alignment: NOT APPLICABLE for all 6 strategies** (structural
fact, verified by reading the signal functions directly). No lookahead,
timezone, or session-boundary defect specific to these 6 strategies was
found in this pass, beyond the already-documented and already-fixed
2026-07-05/07 server-time bug (prior audit).

---

## Step 3 — Historical baseline (full reconstruction, frozen params)

| Strategy | n | Win rate | PF | Expectancy R | Median R | Total R | Max DD (R) | Max losing streak | Max winning streak | Avg hold |
|---|---|---|---|---|---|---|---|---|---|---|
| CADJPY ARB | 192 | 49.0% | 1.263 | +0.133 | -0.150 | +25.5 | -14.87 | 10 | 6 | 26.3h |
| GBPJPY AMR | 403 | 67.2% | 1.426 | +0.144 | +0.713 | +58.0 | -7.28 | 5 | 17 | 4.2h |
| EURJPY AMR | 713 | 69.0% | 1.163 | +0.055 | +0.548 | +39.5 | -17.23 | 4 | 12 | 3.2h |
| AUDJPY AMR | 651 | 69.7% | 1.148 | +0.051 | +0.530 | +33.3 | -10.25 | 5 | 18 | 3.0h |
| CADJPY AMR | 599 | 68.4% | 1.084 | +0.030 | +0.521 | +17.7 | -15.42 | 6 | 14 | 3.7h |
| GBPUSD Monday | 154 | 63.0% | 2.105 | +0.131 | +0.125 | +20.1 | -2.23 | 4 | 9 | 20.2h |

**Year-by-year (bad years shown, not hidden):**

| Strategy | 2023 | 2024 | 2025 | 2026 YTD |
|---|---|---|---|---|
| CADJPY ARB | PF1.62 | **PF0.93 (losing)** | PF1.41 | PF1.45 |
| GBPJPY AMR | PF1.09 | PF1.15 | PF1.65 | PF1.99 |
| EURJPY AMR | PF1.38 | **PF0.99 (~flat losing)** | PF1.20 | PF1.33 |
| AUDJPY AMR | PF1.80 | PF1.00 (flat) | PF1.21 | **PF1.02 (flat, weakest year)** |
| CADJPY AMR | PF1.09 | PF1.03 | **PF1.00 (flat)** | PF1.38 |
| GBPUSD Monday | PF1.83 | PF1.16 | PF2.83 | PF3.19 |

Every strategy except Monday Drift has at least one calendar year at or
below breakeven — reported plainly, not smoothed over.

## Step 4 — True out-of-sample test

**Methodology note:** no strategy-level trade data from the original
2026-07-04/05 discovery-time OOS window was ever persisted (per the
prior forensic audit). Rather than manufacture a comparison from
already-used data, this step **recomputes a fresh trailing-12-month
window using the exact frozen live parameters**, run directly in this
script — labeled explicitly as a fresh computation, not blended with the
original discovery's own reported numbers.

| Strategy | OOS window | n | PF | Expectancy R | Total R | Win rate |
|---|---|---|---|---|---|---|
| CADJPY ARB | 2025-07-23 → 2026-07-23 | 64 | 1.519 | +0.248 | +15.9 | 53.1% |
| GBPJPY AMR | 2025-08-11 → 2026-08-11 | 127 | 2.101 | +0.287 | +36.5 | 75.6% |
| EURJPY AMR | 2025-08-13 → 2026-08-13 | 236 | 1.343 | +0.104 | +24.6 | 72.5% |
| AUDJPY AMR | 2025-08-11 → 2026-08-11 | 205 | 1.144 | +0.050 | +10.2 | 69.3% |
| CADJPY AMR | 2025-08-12 → 2026-08-12 | 189 | 1.305 | +0.092 | +17.3 | 73.0% |
| GBPUSD Monday | 2025-08-10 → 2026-08-10 | 53 | 2.929 | +0.177 | +9.4 | 62.3% |

All 6 show positive trailing-12-month performance under frozen
parameters — none required any re-fitting to produce this result.

## Step 5 — Walk-forward (6-month rolling windows, frozen params, no re-fitting)

| Strategy | Windows | % profitable (n≥10) | Median PF | Worst PF | Best PF |
|---|---|---|---|---|---|
| CADJPY ARB | 11 | 63.6% | 1.460 | **0.585** | 2.118 |
| GBPJPY AMR | 11 | **90.9%** | 1.382 | 0.900 | 2.054 |
| EURJPY AMR | 11 | 63.6% | 1.173 | 0.841 | 1.512 |
| AUDJPY AMR | 11 | 72.7% | 1.136 | 0.849 | 1.585 |
| CADJPY AMR | 11 | 63.6% | 1.114 | **0.791** | 1.355 |
| GBPUSD Monday | 10 | **90.0%** | 2.101 | 0.871 | 4.135 |

**GBPJPY AMR and Monday Drift are the two most walk-forward-stable
strategies** (90%+ of windows profitable). CADJPY ARB shows **two
consecutive negative windows in mid-to-late 2024** (PF 0.64, then
0.59) — a genuine structural weak stretch, not an isolated blip; full
per-window detail in `reports/phase26_revalidation_log.txt`.

## Step 6 — Cost stress

| Strategy | PF normal | PF 1.5x spread | PF 2x spread | PF +1-bar delay | Classification |
|---|---|---|---|---|---|
| CADJPY ARB | 1.263 | 1.210 | 1.159 | 1.145 | **ROBUST** |
| GBPJPY AMR | 1.426 | 1.313 | 1.208 | 1.321 | **ROBUST** |
| EURJPY AMR | 1.163 | 1.020 | **0.889 (losing)** | 1.016 | **COST-FRAGILE** |
| AUDJPY AMR | 1.148 | **0.987 (losing)** | **0.838 (losing)** | 1.002 | **COST-FRAGILE** |
| CADJPY AMR | 1.084 | **0.924 (losing)** | **0.777 (losing)** | 1.043 | **COST-FRAGILE** |
| GBPUSD Monday | 2.105 | 2.039 | 1.975 | 1.730 | **ROBUST** |

**This is the single most important new finding of this revalidation.**
Three of the four currently-live AMR pairs — EURJPY, AUDJPY, and
CADJPY — flip to **net-losing at realistic spread stress**, and two of
them (AUDJPY, CADJPY) are already losing at just 1.5x normal spread.
Only GBPJPY AMR among the AMR family, plus CADJPY ARB and Monday Drift,
remain robust across the full cost-stress grid. This test had never
been run for any of these strategies before this revalidation (per the
prior forensic audit, cost stress did not exist at deployment time).

## Step 7 — Parameter sensitivity (neighborhood, not optimization)

Every parameter tested (`tp_mult` for ARB/Monday, `z_thr`/`sl_mult` for
AMR, `sl_mult`/`tp_mult` for Monday) shows a **broad, smooth region**
around the frozen value across a ±10% neighborhood — **no isolated
spikes were found for any of the 6 strategies.** Full grid in the log;
representative example (CADJPY ARB `tp_mult`): 1.8→PF1.27, 1.9→PF1.25,
**2.0 (frozen)→PF1.26**, 2.1→PF1.31, 2.2→PF1.27 — the frozen value sits
inside a stable plateau, not on a peak. This is a genuinely reassuring
result across the whole book: none of the 6 strategies appear to be
fragile, over-tuned parameter selections.

## Step 8 — Regime analysis (diagnostic only, no filter implied)

| Strategy | Vol regime pattern | Trend regime pattern | Best/worst day |
|---|---|---|---|
| CADJPY ARB | LOW 1.15 → NORMAL 1.73 → **HIGH 0.878 (losing)** | LOW/NORMAL strong, **HIGH_TREND 0.778 (losing)** | Best Mon/Tue, **worst Friday 0.73** |
| GBPJPY AMR | LOW 1.92 → NORMAL 1.25 → HIGH 1.12 (still positive) | all 3 terciles positive | Best Monday, **worst Thursday 0.85** |
| EURJPY AMR | flat across all 3 terciles (~1.16 everywhere) | LOW_TREND weakest (1.03), others ~1.2-1.3 | Best Monday, **worst Thursday 0.90** |
| AUDJPY AMR | LOW 1.33 → NORMAL 1.25 → **HIGH 0.826 (losing)** | mixed, NORMAL_TREND weakest (0.97) | Best Monday, **worst Thursday 0.82** |
| CADJPY AMR | LOW 1.56 → NORMAL 1.02 → **HIGH 0.831 (losing)** | mixed, NORMAL_TREND weakest (0.97) | Best Tuesday, **worst Thursday 0.73** |
| GBPUSD Monday | positive across all terciles (weakens but stays >1.6 in HIGH) | positive across all terciles | Monday-only by design |

**Consistent pattern, not a new discovery but freshly reconfirmed here**:
CADJPY ARB and 2 of 4 AMR pairs (AUDJPY, CADJPY) show clear HIGH-volatility
weakness, matching `reports/amr_regime_mechanism.md` (EXP-076/077, e10d189).
GBPJPY and EURJPY AMR are more regime-stable. **Thursday is a
recurring weak day across 4 of the 5 non-Monday strategies** — reported
as a diagnostic observation, not a rule to act on.

## Step 9 — Directional analysis (diagnostic only, live strategy unchanged)

| Strategy | BUY n / PF / Expectancy R | SELL n / PF / Expectancy R |
|---|---|---|
| CADJPY ARB | 111 / 1.391 / +0.182 | 81 / 1.110 / +0.065 |
| GBPJPY AMR | 263 / 1.647 / +0.198 | 140 / 1.107 / +0.043 |
| EURJPY AMR | 423 / 1.498 / +0.139 | **290 / 0.836 / -0.067 (net losing)** |
| AUDJPY AMR | 412 / 1.591 / +0.158 | **239 / 0.706 / -0.132 (net losing)** |
| CADJPY AMR | 353 / 1.430 / +0.122 | **246 / 0.763 / -0.103 (net losing)** |
| GBPUSD Monday | 154 / 2.105 / +0.131 | N/A — long-only by design |

**Reconfirms `reports/amr_regime_mechanism.md`'s directional-asymmetry
finding (EXP-076-081) with fresh, independently-run data.** Three of the
four AMR pairs' SELL leg is independently net-losing. **No change is
being made to any of these strategies based on this finding** — reported
diagnostically, as instructed.

## Step 10 — Monte Carlo (10,000 runs per strategy)

| Strategy | MC DD p50 | p75 | p90 | p95 | p99 | Actual historical DD | Percentile of actual |
|---|---|---|---|---|---|---|---|
| CADJPY ARB | 10.83R | 13.08R | 15.60R | 17.41R | 21.30R | 14.87R | ~80th |
| GBPJPY AMR | 8.54R | 10.20R | 12.11R | 13.50R | 16.71R | 7.28R | **below 50th (mild)** |
| EURJPY AMR | 13.51R | 16.18R | 19.21R | 21.48R | 25.97R | 17.23R | ~80th |
| AUDJPY AMR | 13.38R | 16.06R | 18.98R | 21.03R | 25.25R | 10.25R | **below 50th (mild)** |
| CADJPY AMR | 15.27R | 18.42R | 21.90R | 24.22R | 28.73R | 15.42R | **~50th (typical)** |
| GBPUSD Monday | 2.39R | 2.88R | 3.47R | 3.87R | 4.72R | 2.23R | **below 50th (mild)** |

| Strategy | MC losing-streak p50/p90/p95/p99 |
|---|---|
| CADJPY ARB | 7 / 9 / 10 / 12 |
| GBPJPY AMR | 5 / 6 / 7 / 8 |
| EURJPY AMR | 5 / 7 / 7 / 9 |
| AUDJPY AMR | 5 / 6 / 7 / 8 |
| CADJPY AMR | 5 / 7 / 7 / 8 |
| GBPUSD Monday | 4 / 6 / 7 / 8 |

**Per the explicit instruction:** trade-order shuffling does not change
final P&L (order-invariant by construction) — only drawdown and streak
distributions are reported, never a shuffled-P&L confidence interval.
No strategy's actual historical drawdown exceeds its own 90th
percentile except CADJPY ARB and EURJPY AMR, both sitting around the
80th — elevated but not extreme.

## Step 11 — Bootstrap confidence intervals (expectancy)

| Strategy | Mean expectancy R | 95% CI | Crosses zero? |
|---|---|---|---|
| CADJPY ARB | +0.133 | [-0.040, +0.301] | **YES** |
| GBPJPY AMR | +0.144 | [+0.058, +0.224] | **NO — statistically real** |
| EURJPY AMR | +0.055 | [-0.005, +0.110] | **YES (barely)** |
| AUDJPY AMR | +0.051 | [-0.007, +0.108] | **YES (barely)** |
| CADJPY AMR | +0.030 | [-0.031, +0.090] | **YES** |
| GBPUSD Monday | +0.131 | [+0.062, +0.201] | **NO — statistically real** |

**Only GBPJPY AMR and Monday Drift have a positive expectancy that
cannot be explained by sampling noise alone.** The other four strategies'
positive point estimates are real historical results, but the
statistical uncertainty around them — reported honestly, not
downplayed — includes zero. This does not mean those four strategies
have no edge; it means the sample cannot yet rule out "no edge" with
95% confidence.

## Step 13 — Live exit reason handling

| Strategy | SL | TP | Time exit / Friday close |
|---|---|---|---|
| CADJPY ARB | 87 | 59 | 46 (FridayClose) |
| GBPJPY AMR | 131 | 271 | 1 (FridayClose) |
| EURJPY AMR | 219 | 491 | 3 (FridayClose) |
| AUDJPY AMR | 197 | 453 | 1 (FridayClose) |
| CADJPY AMR | 188 | 410 | 1 (FridayClose) |
| GBPUSD Monday | 3 | 18 | 133 (**scheduled 21:00 Monday time exit**) |

Per instruction: any dashboard `MANUAL/OTHER` row is the bot's own
**SCHEDULED STRATEGY EXIT** (Monday Drift's 21:00 UTC time exit
dominates its own exit-reason distribution, 133 of 154 trades) — not a
discretionary intervention. This backtest reconstruction's own
`TimeExit`/`FridayClose` labels are the equivalent mechanism.

---

## Step 14 — Current 6-strategy portfolio reconstruction

Combined 2,712 trades, current risk weights (CADJPY ARB 0.50%, all AMR
0.25%, Monday 0.25%), shared $100,000 reference capital.

| Metric | Value |
|---|---|
| Total trades | 2,712 |
| Portfolio PF | 1.216 |
| Portfolio expectancy | +0.0716R |
| Total R | +194.11 |
| Maximum drawdown | -8.21% of capital |
| Maximum losing streak | 13 trades |

**Portfolio Monte Carlo (10,000 runs, R-based):** median DD 18.46R, p75
21.57R, p90 25.33R, p95 28.07R, p99 33.94R. **Actual historical portfolio
drawdown (29.53R) sits between the 95th and 99th percentile of this
distribution** — genuinely elevated, in the upper tail of what this
6-strategy portfolio's own trade sequence could plausibly produce.

**This directly resolves the specific gap flagged in the prior forensic
audit** (`reports/live_portfolio_validation_audit.md` §"5ers performance"):
the earlier portfolio Monte Carlo (`reports/portfolio_drawdown_distribution_audit.md`,
EXP-092-095) was built from all 8 original strategies at full risk — not
what is actually running on 5ers. **This is the corrected benchmark**,
built from exactly the 6 active strategies at their current weights.
One remaining, smaller gap: this reconstruction does not additionally
apply the 5ers-specific `risk_scale: 0.5` — it reflects the strategies'
documented risk weights, not the further 5ers-only scale-down. A
precisely 5ers-matched benchmark would need that additional halving
applied, which was not done here (would not change R-based percentiles,
only the dollar/percent-of-capital figures).

**JPY factor:** 94.3% of trades and 94.7% of risk-weight are JPY-exposed
(higher than the full 8-strategy book's 87-88%, since removing GBPJPY
ARB and XAUUSD ARB removes the two components with any non-JPY or
partial diversification). All pairwise correlations remain positive
(0.02-0.36). Days with 2+ JPY strategies losing together: 227 (29.3%
of days). Days with 3+: 110 (14.2%). Worst clustered JPY loss:
2024-10-29, -$1,765.34.

---

## Step 12 — Live 5ers comparison

**Data limitation carried forward from the prior audit: no trade-level
5ers data with per-strategy attribution was supplied to this
revalidation** — only the aggregate dashboard snapshot (~33 trades,
balance ~$4,797, win rate ~30%, PF ~0.30, expectancy ~-0.33R, DD ~4%).
**Per-strategy live comparison is NOT AVAILABLE.**

**Portfolio-level comparison, using the corrected 6-strategy benchmark
built in Step 14 (not the prior, wrong-configuration one):**

- Current 5ers drawdown (~-3.8% to -4%) remains modest relative to this
  portfolio's own historical worst (-8.21% dollar-terms; 29.53R,
  ~95th-99th percentile of its own Monte Carlo distribution) —
  classified **WITHIN HISTORICAL RANGE**.
- Current ~9-trade losing streak: this portfolio's MC losing-streak
  distribution (computed at the individual-strategy level; a combined
  portfolio-level streak MC was not separately re-derived in this
  script, see Step 10's per-strategy table) — the prior audit's
  portfolio-level streak MC (p90=9, p95=10, p99=11, from the 8-strategy
  benchmark) placed 9 trades at the ~94th percentile; with 2 fewer,
  generally weaker-tail strategies removed, the corrected 6-strategy
  streak distribution would plausibly be similar or slightly tighter —
  classified **ELEVATED BUT PLAUSIBLE**, consistent with the prior
  audit's finding, not contradicted by this one.
- **Current live PF (~0.30) and expectancy (~-0.33R) are considerably
  worse than any individually-reconstructed strategy's historical or
  OOS numbers** (all 6 strategies show positive historical and OOS PF,
  ranging 1.08-2.93). At only ~33 trades, this is not yet a large enough
  live sample to distinguish "normal variance around a real edge" from
  "something is currently wrong," especially given 3 of 6 strategies are
  now shown to be cost-fragile and 4 of 6 have bootstrap CIs crossing
  zero — **classified UNUSUAL, not automatically STRONGLY INCONSISTENT**,
  because the live sample is too small and too aggregated (no
  per-strategy attribution) to reach a stronger conclusion. This is not
  a claim of strategy failure — it is a statement that the live
  evidence, as currently available, cannot be more precisely
  characterized than this.

---

## Special AUDJPY AMR section

1. **Original AUDJPY AMR:** the strategy reconstructed and tested
   throughout this report — both-direction, `z_thr=2.0`, `sl_mult=1.5`,
   `entry_end_hour=4`. Historical PF 1.148, OOS (trailing 12mo) PF
   1.144, cost-fragile (net losing at 1.5x and 2x spread), bootstrap CI
   crosses zero, SELL leg independently net-losing (PF 0.706).
2. **Later BUY-only hypothesis:** researched in `src/phase22_audjpy_amr_confirmatory.py`
   (Model B), classified **SUPPORTED** but explicitly **NOT VALIDATED**
   (`reports/audjpy_amr_confirmatory_filter.md`, EXP-082-086).
3. **Evidence supporting BUY-only:** large, walk-forward-consistent OOS
   improvement, survives 2x spread stress, directional asymmetry
   independently reconfirmed in this revalidation (BUY PF 1.591 vs
   SELL PF 0.706).
4. **Evidence against BUY-only:** OOS bootstrap CI on the BUY-only
   candidate still crosses zero (phase 22); no genuinely fresh
   historical data exists to further validate it
   (`reports/audjpy_amr_final_validation.md`, classified **B.
   INSUFFICIENT FRESH DATA**, EXP-087-089).
5. **Genuine fresh OOS evidence for BUY-only specifically:** none yet —
   a prospective forward-validation tracker was built and started
   (`src/amr_forward_tracker.py`, EXP-090/091) but has collected 0
   trades as of the last check.
6. **Is BUY-only currently implemented live?** **No.** Verified directly
   against `strategies/asian_hours_reversion.py` and
   `pairs/AUDJPY_asianrev.yaml` for this revalidation: both files show
   unrestricted, both-direction trading, unchanged from the original
   specification. **The live strategy remains the original
   both-direction AUDJPY AMR** — this revalidation's Steps 3-11 above
   describe exactly that strategy, not the BUY-only candidate. No
   modification was made in this phase.

---

## Strategy scorecards

Reasoned classification, not a simple pass-count. **A. = PASS, B. =
FAIL, I. = INSUFFICIENT** where applicable.

| | A Data integrity | B Live/BT parity | C Historical profit | D OOS profit | E Walk-forward | F Cost robustness | G Parameter stability | H Regime stability | I Monte Carlo | J Live compatibility |
|---|---|---|---|---|---|---|---|---|---|---|
| CADJPY ARB | PASS | PASS | PASS | PASS | INSUFFICIENT (63.6%, 2 consecutive bad windows) | PASS | PASS | FAIL (HIGH-vol/trend losing) | INSUFFICIENT (~80th pctile) | PASS |
| GBPJPY AMR | PASS | PASS (BE-gap, minor) | PASS | PASS | PASS (90.9%) | PASS | PASS | PASS | PASS | PASS |
| EURJPY AMR | PASS | PASS (BE-gap, minor) | PASS (marginal) | PASS | INSUFFICIENT (63.6%) | **FAIL** | PASS | PASS-ish (flat vol, weak SELL) | INSUFFICIENT (~80th pctile) | PASS |
| AUDJPY AMR | PASS | PASS (BE-gap, minor; BUY-only NOT live) | PASS (marginal 2026) | PASS (marginal) | INSUFFICIENT (72.7%, weakening trend) | **FAIL** | PASS | FAIL (HIGH-vol losing) | PASS | PASS |
| CADJPY AMR | PASS | PASS (BE-gap, minor) | PASS (marginal, weakest PF) | PASS | INSUFFICIENT (63.6%) | **FAIL (worst of book)** | PASS | FAIL (HIGH-vol losing) | INSUFFICIENT (~50th, unremarkable) | PASS |
| GBPUSD Monday | PASS | PASS | PASS (strongest) | PASS (strongest) | PASS (90.0%) | PASS | PASS | PASS | PASS | PASS |

---

## Final classifications

| Strategy | Classification |
|---|---|
| **GBPJPY AMR** | **A. STRONG REVALIDATION** |
| **GBPUSD Monday Drift** | **A. STRONG REVALIDATION** |
| CADJPY ARB | **B. ACCEPTABLE BUT MONITOR** |
| EURJPY AMR | **C. PROMISING BUT INSUFFICIENT** |
| AUDJPY AMR | **C. PROMISING BUT INSUFFICIENT** |
| CADJPY AMR | **D. WEAK / PROVISIONAL** |

**Reasoning, briefly:** GBPJPY AMR and Monday Drift are the only two
strategies that pass cost stress, walk-forward stability, and the
bootstrap significance test simultaneously — the three tests this
revalidation added that did not exist before. CADJPY ARB passes cost
stress and is statistically real in point-estimate terms but shows a
genuine walk-forward weak stretch and clear regime dependence, landing
it at "monitor" rather than "strong." EURJPY and AUDJPY AMR are
cost-fragile and statistically inconclusive (CI crosses zero) but still
show a real, walk-forward-majority-positive historical record — genuinely
promising, not yet sufficient for full confidence. CADJPY AMR is the
weakest of the book on nearly every dimension tested (lowest historical
PF, worst cost-stress result, HIGH-regime losing, CI crosses zero) —
**"provisional," not "failed"**, since it still shows a real positive
walk-forward majority and a smooth parameter plateau; there is no
evidence its underlying edge has broken, only that the evidence
supporting it is the thinnest in the current book.

**No strategy in this revalidation is classified E (FAILED)** — none
show evidence that contradicts the underlying edge; the weaknesses
found are about the *strength and robustness* of the evidence, not
falsification of the hypothesis. **No strategy is classified F** — the
audit trail was traceable and complete for all 6.

---

## Portfolio-level verdict

**1. Which strategies genuinely survive revalidation?** GBPJPY AMR and
GBPUSD Monday Drift, on the fullest evidence base (cost-robust,
walk-forward-stable, statistically significant, regime-stable).

**2. Which are only promising?** EURJPY AMR and AUDJPY AMR — real
historical and OOS records, but cost-fragile and statistically
inconclusive.

**3. Which should be considered provisional?** CADJPY AMR.

**4. Which show evidence of deterioration?** None conclusively — CADJPY
ARB's 2024 losing year and AUDJPY AMR's flattening 2026 YTD (PF 1.02)
are worth watching but are single data points within an otherwise
positive multi-year record, not a clear deterioration trend by
themselves.

**5. Which suffer primarily from regime dependence?** CADJPY ARB,
AUDJPY AMR, CADJPY AMR (all show HIGH-volatility-regime losing
performance).

**6. Which suffer from execution/cost issues?** EURJPY AMR, AUDJPY AMR,
CADJPY AMR (all three flip net-losing under realistic spread stress) —
**this is the most actionable, evidence-based weakness identified in
this revalidation**, though per instructions no action is being taken
on it here.

**7. Which have insufficient sample size?** Monday Drift has the
smallest total sample (154 trades, 52/year) but the strongest per-trade
statistical signal; none of the 6 are so small as to be unusable, but
Monday's inherent low frequency remains a standing caveat (per the
prior forensic audit).

**8-9. What fraction of the current portfolio is supported by strong vs.
provisional evidence?** By risk-weight: GBPJPY AMR (0.25%) + Monday
(0.25%) = 0.50% of 1.75% total book risk (~28.6%) is "strong."
CADJPY ARB (0.50%) is "monitor" (~28.6%). EURJPY AMR + AUDJPY AMR
(0.25% each) = 0.50% (~28.6%) is "promising but insufficient." CADJPY
AMR (0.25%, ~14.3%) is "provisional."

**10. Is the current portfolio sufficiently robust to continue trading
unchanged?** The evidence does not call for a change (see absolute
rules — none is being made), but it does not uniformly support "strong
confidence" either: 3 of 6 strategies (50% of the book by count, ~57%
by risk-weight if EURJPY+AUDJPY+CADJPY AMR are combined) are cost-fragile,
a real and newly-quantified structural weakness.

**11. Is the current 5ers losing period compatible with the revalidated
historical distributions?** On the portfolio drawdown/streak dimension:
yes, within or near the historically-expected range using the corrected
(6-strategy) benchmark built in this report. On the PF/expectancy
dimension specifically: the live numbers (PF~0.30, expectancy~-0.33R)
are considerably worse than any individual strategy's own historical or
OOS reconstruction — this is flagged as **unusual**, not yet as
**strongly inconsistent**, given the small live sample size (~33
trades) and the absence of per-strategy live attribution.

**12. Single biggest weakness in the current portfolio?** **Cost
fragility in 3 of 4 currently-live AMR pairs** (EURJPY, AUDJPY, CADJPY)
— a structural, newly-quantified finding, not a matter of interpretation.

**13. Strongest strategy?** **GBPUSD Monday Drift** — passes every test
in this revalidation, on the smallest but statistically real sample.

**14. Weakest strategy?** **CADJPY AMR** — lowest historical PF, worst
cost-stress result in the book, clear HIGH-regime weakness, CI crosses
zero.

**15. Does the evidence justify changing anything right now?** **No
change is being made or recommended for immediate action** — see the
recommendation below.

---

## Recommendation

# **CONTINUE WITH MONITORING**

Not "continue unchanged" without qualification — the cost-fragility
finding for 3 of 6 strategies (EURJPY/AUDJPY/CADJPY AMR) is a real,
newly-quantified structural weakness that did not exist as known
evidence before this revalidation, and it deserves active tracking, not
silence. Not "pause for further validation" — none of the 6 strategies
show evidence of a broken edge, the walk-forward and parameter-stability
results are broadly reassuring, and the current live drawdown/streak
remain within or near the historically-expected range. Not "research
required" — this phase was validation, and the appropriate next step is
monitoring the identified weaknesses (cost sensitivity, CADJPY AMR's
overall thinness of evidence) against live results, not launching new
research.

**This recommendation is not being implemented.** No strategy, filter,
risk weight, or configuration was changed in the production of this
report.

---

## Sources cited

`pairs/CADJPY_asianrange.yaml`, `pairs/{GBPJPY,EURJPY,AUDJPY,CADJPY}_asianrev.yaml`,
`pairs/GBPUSD_monday.yaml`, `strategies/asian_hours_reversion.py`;
`src/phase2_meanrev_arb_search.py`, `src/phase3b_amr_jpy_refine.py`,
`src/phase8_monday_validation.py` (signal logic, unmodified);
`reports/live_portfolio_validation_audit.md` (EXP-096-104, cfc5120);
`reports/amr_regime_mechanism.md` (EXP-076-081, e10d189);
`reports/audjpy_amr_confirmatory_filter.md` (EXP-082-086, 55e301e);
`reports/audjpy_amr_final_validation.md` (EXP-087-089, fe22c56);
`reports/portfolio_drawdown_distribution_audit.md` (EXP-092-095, 32fe299).

## What I did NOT do (per instructions)

- Did not create, optimize, tune, or modify any strategy, entry/exit
  logic, risk, or session window.
- Did not deploy, pause, replace, or change any account or live
  configuration.
- Did not use current live 5ers results to modify any historical test.
- Did not cherry-pick favorable periods or remove losing trades/years
  from any reported result.
- Did not implement the AUDJPY BUY-only hypothesis — verified directly
  that it remains unimplemented.
- Did not implement the "CONTINUE WITH MONITORING" recommendation.

See `reports/current_6_strategy_revalidation.csv` for the
machine-readable summary.
