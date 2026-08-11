# AUDJPY AMR Confirmatory Filter Experiment

**Experiments:** EXP-082 through EXP-088, `experiments/experiments.csv`.
**Script:** `src/phase22_audjpy_amr_confirmatory.py`. **Full log:** `reports/phase22_confirmatory_log.txt`. **Data:** `data/phase22_audjpy_trades.csv`.

**Research only. AUDJPY AMR (and CADJPY/GBPJPY/EURJPY AMR, ARB, Monday
Drift, XAUUSD ARB) were NOT modified anywhere in this work. The demo
account was not touched.** Even the strongest classification below does
not authorize a live change — it only identifies whether a model
deserves a separate, final validation gate.

## 1. Frozen baseline definition

Existing live AUDJPY AMR, unchanged: `signals_amr_v(z_thr=2.0,
sl_mult=1.5, end_hour=4)`, spread 2.0 pips, risk 0.25% — the exact
reconstruction used in phases 20/21. No entry, exit, stop, target,
session, or execution assumption was altered.

## 2. Frozen filter definitions (pre-registered before any OOS result was examined)

- **Model A (volatility filter):** trade only when entry-time ATR
  percentile < 0.75 — the same HIGH-regime boundary already established
  in phase 20/21, not searched here.
- **Model B (BUY-only):** exclude SELL trades; BUY entry logic unchanged.
- **Model C (secondary/exploratory only):** A AND B combined. Per
  instructions, not used to select a winner even if it performs best.

## 3. Exact train/validation/OOS dates

652 baseline trades reconstructed, 2023-07-31 to 2026-08-11. Strict
chronological thirds, frozen before any result was examined:

| Period | Dates | n trades |
|---|---|---|
| TRAIN/IS | 2023-07-31 → 2024-08-03 | 223 |
| VALIDATION | 2024-08-03 → 2025-08-07 | 221 |
| **FINAL OOS** | **2025-08-07 → 2026-08-11** | **208** |

## 4. Baseline results

| Period | n | win rate | PF | expectancy | total R |
|---|---|---|---|---|---|
| TRAIN | 223 | 73.5% | 1.29 | +22.82 | 20.0 |
| VALIDATION | 221 | 66.1% | 1.02 | +2.22 | 2.0 |
| **OOS** | 208 | 69.2% | **1.14** | **+12.86** | 10.2 |

## 5. Model A results (volatility filter)

| Period | n | win rate | PF | expectancy | total R |
|---|---|---|---|---|---|
| TRAIN | 209 | 74.6% | 1.35 | +26.28 | 21.6 |
| VALIDATION | 159 | 68.6% | 1.11 | +10.16 | 6.3 |
| **OOS** | 170 | 71.2% | **1.21** | **+18.34** | 11.8 |

Real, consistent improvement over baseline in every period, but modest
in magnitude (PF +0.07 in OOS, expectancy +5.48/trade).

## 6. Model B results (BUY-only)

| Period | n | win rate | PF | expectancy | total R |
|---|---|---|---|---|---|
| TRAIN | 144 | 79.2% | 1.79 | +48.52 | 27.1 |
| VALIDATION | 128 | 71.1% | 1.29 | +23.60 | 11.6 |
| **OOS** | 140 | 77.9% | **1.74** | **+49.79** | 26.2 |

**A large, consistent improvement over baseline in every period** — OOS
PF is 53% higher than baseline (1.74 vs 1.14), OOS expectancy is nearly
4x baseline (+49.79 vs +12.86 per trade).

## 7. Model C results (secondary/exploratory)

| Period | n | win rate | PF | expectancy |
|---|---|---|---|---|
| TRAIN | 137 | 79.6% | 1.81 | +49.01 |
| VALIDATION | 97 | 73.2% | 1.38 | +29.13 |
| **OOS** | 120 | 79.2% | **1.83** | **+52.89** |

Slightly better than Model B alone on every metric — but per
instructions, this is **not** being used to pick a winner and would need
its own separately-scoped confirmatory experiment if pursued.

## 8. Walk-forward (6-month rolling windows, full log has all 11 windows)

**Model A** tracks baseline closely — better in most windows, worse in
a couple, no dramatic separation; both baseline and Model A share the
same weak stretch (Oct 2024–Apr 2025, both near/below breakeven).

**Model B** is better than baseline in **10 of 11 windows**, often by a
wide margin (e.g. Jul 2025–Jan 2026: Model B PF 2.77 vs baseline's
corresponding window PF ~1.28; Apr–Oct 2025: Model B PF 1.90 vs
baseline's ~1.18). The one shared weak window (Oct 2024–Apr 2025) is
still present for Model B too (PF 0.87, also losing) — **the filter
does not manufacture a fake win in every period, it genuinely
underperforms in the same historically-difficult window as the
baseline**, which is a good sign for honesty of the test rather than
data-mining.

## 9. Cost stress

| Scenario | BASELINE PF | MODEL A PF | MODEL B PF |
|---|---|---|---|
| Normal spread | 1.14 | 1.20 | 1.48 |
| 1.5x spread | 0.98 | 1.01 | 1.27 |
| **2x spread** | **0.83 (losing)** | **0.84 (losing)** | **1.08 (still profitable)** |
| 1-bar delay | 1.00 | 1.00 | 1.24 |

**Model B is the only one of the three that remains profitable under
the most severe cost-stress scenario tested (2x spread).** Both the
baseline and Model A turn net-losing at 2x spread; Model B does not.

## 10. Statistical comparison (bootstrap, FINAL OOS only — the decisive test)

| | mean expectancy diff vs. baseline | 95% CI | P(model > baseline) |
|---|---|---|---|
| Model A | +5.70 | [-37.08, +45.19] | 61.2% |
| Model B | +37.49 | **[-5.95, +81.31]** | **95.4%** |

**Model A's confidence interval comfortably includes zero — not
statistically distinguishable from no improvement in this single test.**
Model B's interval also technically includes zero, but only barely (the
lower bound is -5.95, close to excluding it entirely), and P(B >
baseline) = 95.4% is very close to a clean one-sided significance
threshold. Per instructions, this alone is not being treated as proof —
but combined with the walk-forward and cost-stress evidence, it is
meaningfully stronger corroboration for Model B than for Model A.

## 11. Trade-count / opportunity impact

| | trades retained | total-R retained | gross-profit retained | opportunity reduction |
|---|---|---|---|---|
| Model A | 82.5% | 123.1% | 83.6% | 17.5% |
| Model B | 63.2% | **201.5%** | 69.3% | 36.8% |
| Model C | 54.3% | 188.6% | 59.6% | 45.7% |

**Model B retains 201.5% of baseline's total R while trading only 63.2%
of baseline's trade count** — this passes the trade-count-penalty check
explicitly required by the brief: it is not simply a smaller, similar
sample, it is a smaller sample that is meaningfully *more* profitable in
aggregate, not just per-trade. Model A's retained-R (123.1%) also
exceeds its retained-trade-count (82.5%), a smaller but real version of
the same pattern.

## 12. Year consistency

| Year | BASELINE expectancy | MODEL A expectancy | MODEL B expectancy |
|---|---|---|---|
| 2024 | +0.13 | +8.64 | **+39.25** |
| 2025 | +18.16 | +21.43 | **+44.64** |
| 2026 YTD | +1.89 | +6.68 | **+32.95** |

**Model B improves over baseline in all 3 years, by a large margin every
time.** Model A also improves in all 3 years, consistently but by a much
smaller margin.

## 13. Drawdown comparison

| | max DD | max losing streak | worst 10-trade seq | worst month | worst quarter |
|---|---|---|---|---|---|
| BASELINE | -2,687 | 5 | -1,344 | -1,262 | -1,581 |
| MODEL A | -2,745 | 5 | -1,596 | -1,213 | -1,367 |
| MODEL B | **-2,250** | **4** | -1,758 | **-979** | **-452** |

Model B improves max drawdown, losing streak, worst month, and worst
quarter — the one exception is worst-10-trade-sequence, which is
slightly worse than baseline (-1,758 vs -1,344), a reminder that no
metric universally improves and this should not be glossed over.

## 14. Regime robustness (reporting only, no re-optimization)

Model A's filtered population sanity-checks correctly (ATR percentile
range 0.003–0.748, confirming the filter is applied as specified).

**Model B: BUY (retained) vs. SELL (excluded), full history:**

| | n | win rate | PF | expectancy | total R |
|---|---|---|---|---|---|
| BUY (retained) | 412 | 76.2% | **1.59** | +41.21 | +65.0 |
| SELL (excluded) | 240 | 58.3% | **0.70** | **-36.36** | **-32.7** |

**The excluded SELL population is a clear historical net loser on its
own** (PF 0.70, total R -32.7) — this is not cherry-picking a marginal
subgroup; it is removing a segment that has been actively losing money,
which is exactly the mechanism phase 21's directional-asymmetry finding
predicted.

## 15. Strongest evidence FOR

Model B: improves every period (TRAIN/VALIDATION/OOS), 10 of 11
walk-forward windows, all 3 years, survives cost stress up to 2x spread
where baseline and Model A both fail, retains 201.5% of total R on 63.2%
of the trades, and the excluded SELL population is independently shown
to be a historical net loser rather than an arbitrarily chosen subgroup.

## 16. Strongest evidence AGAINST

The single formal statistical test (bootstrap CI on OOS expectancy
difference) technically still includes zero for both models, including
Model B — the improvement, while large and consistent, is not proven
beyond the bar a strict significance test would require. Model B's
worst-10-trade-sequence is slightly worse than baseline's, not
uniformly better on every risk metric. Model A's evidence, while
directionally consistent, is weak enough (P=61.2% in the OOS bootstrap)
that it does not clear a meaningful evidentiary bar on its own.

## 17. Final OOS conclusion

**Model A: improvement is real in direction but not statistically
convincing — modest magnitude, weak OOS bootstrap result (P=61.2%).**

**Model B: improvement is large, directionally consistent across every
period/window/year tested, uniquely survives severe cost stress, and
its OOS bootstrap result (P=95.4%, CI lower bound barely below zero) is
close to — though not formally past — a clean significance threshold.**

## 18. Model classifications

- **MODEL A — VOLATILITY FILTER: PROMISING.** Real, consistent, small
  improvement; does not clear a statistical bar strong enough for
  SUPPORTED.
- **MODEL B — BUY-ONLY: SUPPORTED.** Improvement is large, consistent
  across every cut tested (periods, walk-forward windows, years, cost
  stress), and the underlying mechanism (SELL trades are an independent
  historical net loser) is corroborated, not just correlated. The
  bootstrap CI is marginal by the strictest reading, but the weight of
  the walk-forward and cost-stress evidence is the primary basis for
  this classification, as instructed.

**Per explicit instruction: SUPPORTED does not authorize changing the
live/demo AUDJPY AMR strategy.** It means Model B has earned a separate,
final validation gate — not that it should be deployed.

## 19. Overall decision

# **ONE FILTER JUSTIFIED FOR FURTHER VALIDATION (Model B, BUY-only)**

Not "no change justified" — Model B's evidence is too consistent and
too large to close the branch. Not "strong evidence for change" — the
formal statistical test is marginal, not decisive, and per instructions
even a SUPPORTED classification is not itself authorization to change
anything live.

## 20. Exact next step

**Recommend exactly one next validation phase, not implemented here:** a
final, pre-registered validation gate for AUDJPY AMR BUY-only, run on
genuinely fresh data as it accumulates (not the same OOS window reused
again), following this project's standard validation-gate template
(the same one used for the original NZDJPY and Monday-drift gates) —
IS/OOS discipline, Monte Carlo, permutation test, and a single final
VALIDATED / PROMISING BUT INSUFFICIENT / FAILED classification. **Do not
optimize Model B further, do not test Model C as a standalone candidate
without its own separate registration, and do not touch the live
strategy before that gate completes.** This should be brought to the
2026-08-25 AMR checkpoint alongside the mechanism-research findings from
phase 21 as the two pieces of evidence to weigh together.

---

## What I did NOT do (per instructions)

- Did not modify AUDJPY AMR, CADJPY AMR, GBPJPY AMR, EURJPY AMR, ARB,
  Monday Drift, or XAUUSD ARB.
- Did not change the demo account.
- Did not search any threshold — Model A's 75th percentile and Model
  B's BUY-only rule were both frozen before any OOS result was examined.
- Did not select Model C as a winner despite its slightly better
  numbers, per explicit instruction that it is secondary/exploratory only.
- Did not treat "SUPPORTED" as authorization for a live change.
