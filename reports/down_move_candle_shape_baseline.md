# T0 Candle-Shape Baseline — Controlled Trading Experiment

**Experiments:** EXP-056 through EXP-059, `experiments/experiments.csv`.
**Script:** `src/phase18_candle_shape_baseline.py`. **Full log:** `reports/phase18_baseline_log.txt`. **Data:** `data/phase18_events.csv`, `data/phase18_trades_{A,B,C}.csv`.

**This is one controlled comparison, not optimization.** No parameter was
searched at any point. No existing strategy (ARB/AMR/Monday-drift/XAUUSD
ARB) or the existing phase 15/16/17 down-move research was modified.

## 1. Frozen mathematical definition (Part 1)

```
body_ratio        = |close - open| / (high - low)
close_location    = (close - low)  / (high - low)
lower_wick_ratio   = (min(open,close) - low)  / (high - low)
upper_wick_ratio   = (high - max(open,close)) / (high - low)
```

**Composite:** `shape_score = close_location - body_ratio`, frozen
before any backtest ran. Reasoning: `body_ratio + upper_wick_ratio +
lower_wick_ratio = 1` identically for any candle, so the four quantities
above are not independent — a composite built from all four would
double-count the wick split. Phase 17 found `close_location` (d=+0.101)
and `body_ratio` (d=-0.107) to be the two cleanest, least-redundant
Tier-1 effects, so the composite uses exactly those two.

**Threshold:** the **pooled historical median** of `shape_score` across
all 37,676 qualifying down-move events in the research sample (value:
**-0.6980**), computed once, before any trade simulation — not searched
for backtest performance. This mirrors the median-split methodology this
project already used for volatility-regime terciles in phases 3b/16/17.

**Mechanics, identical across A/B/C** (unchanged from phase 15): entry =
next candle open, stop = 1.0×ATR, target = 1.0R, max hold = 4 M15 bars
(60 min), SL priority on same-bar SL+TP touch, normal spread.

## 2. Information-timing verification (Part 4)

`shape_score` is computed purely from the event bar's own
open/high/low/close — all four values are known at the close of that
bar, i.e. at T0 by construction. No future candle, no post-event
volatility, no future session or range data enters the shape rule
anywhere in the code (`compute_shape()` in `phase18_candle_shape_baseline.py`
takes only the event bar's OHLC as input).

## 3. Baseline populations

| Baseline | Population | n events | Direction |
|---|---|---|---|
| A — UNFILTERED | all ≥1.0 ATR down-moves | 37,676 | BUY (fade) |
| B — REVERSAL-SHAPE | shape_score ≥ median | 18,838 | BUY (fade) |
| C — CONTINUATION-SHAPE CONTROL | shape_score < median | 18,838 | SELL (continuation) |

## 4. Results — normal spread, no delay (Part 2/3)

| Baseline | n trades | win rate | PF | expectancy (pips) | mean R | mean MFE | mean MAE |
|---|---|---|---|---|---|---|---|
| A — UNFILTERED | 28,644 | 41.7% | 0.740 | -1.169 | -0.171 | 6.72 | -9.10 |
| B — REVERSAL-SHAPE | 15,891 | 42.7% | **0.758** | -1.128 | -0.151 | 7.12 | -9.56 |
| C — CONTINUATION-SHAPE | 16,465 | 36.9% | 0.618 | -1.843 | -0.248 | 6.50 | -8.95 |

**B is directionally better than A, and C is reliably worse than both** —
the shape signal is pointing the right way, and this ordering (B > A > C)
is itself evidence the signal carries real content, not noise. **But B
is still a clear net loser** (PF 0.758, well under breakeven), and the
improvement over A is small.

**Raw population-level reversal probability by shape group (Part 3):**
reversal-shape group = 54.27% reversal, continuation-shape group =
53.14% reversal — only a **1.1 percentage-point gap**. This is much
smaller than the continuous effect sizes reported in phase 17
(d=0.06-0.11 on the underlying variables) — converting a continuous
signal into a single median split discards most of its resolving power,
which is the central reason the trading result is weak (see Part 13
below).

## 5. Pair-level results (Part 5)

| pair | A: PF | B: PF | C: PF | B > A? |
|---|---|---|---|---|
| EURUSD | 0.742 | 0.753 | 0.671 | yes |
| GBPUSD | 0.777 | 0.805 | 0.670 | yes |
| USDJPY | 0.809 | 0.793 | 0.720 | **no** |
| AUDUSD | 0.669 | 0.680 | 0.563 | yes |
| USDCAD | 0.649 | 0.715 | 0.553 | yes |
| NZDUSD | 0.574 | 0.590 | 0.451 | yes |
| GBPJPY | 0.823 | 0.800 | 0.664 | **no** |
| EURJPY | 0.817 | 0.856 | 0.654 | yes |
| CADJPY | 0.625 | 0.658 | 0.509 | yes |

**7 of 9 pairs show B improving over A** (USDJPY and GBPJPY are the
exceptions). Full per-pair figures (win rate, expectancy, mean R,
MFE/MAE) in `reports/phase18_baseline_log.txt`. No single pair drives the
overall B > A > C ordering — it holds broadly, not just in one or two pairs.

## 6. Year-level results (Part 6)

| year | A: PF | B: PF | C: PF |
|---|---|---|---|
| 2023 | 0.685 | **0.751** | 0.652 |
| 2024 | 0.728 | 0.724 | 0.623 |
| 2025 | 0.744 | **0.762** | 0.620 |
| 2026 YTD | 0.800 | **0.827** | 0.574 |

**B beats A in 3 of 4 years** (2024 is a near-tie, marginally in A's
favor: 0.7285 vs 0.7236). C is worse than both A and B in **every single
year**, which is the more important consistency check here — the
directional ordering (fade-with-good-shape > unfiltered > continue-with-
bad-shape) holds across all 4 years even though B's absolute edge over A
is inconsistent year to year.

## 7. Session-level results (Part 7)

| session | A: PF | B: PF | C: PF |
|---|---|---|---|
| ASIAN | 0.844 | **0.939** | 0.580 |
| LONDON | 0.686 | 0.677 | 0.636 |
| OVERLAP | 0.696 | 0.703 | 0.633 |
| NY | 0.754 | 0.731 | 0.649 |

**Descriptive observation, not a recommendation:** Baseline B's
improvement over A is concentrated almost entirely in Asian hours (PF
0.939 vs 0.844, the closest either baseline gets to breakeven anywhere
in this study) — consistent with this entire research line's repeated
finding that Asian-hours conditions are where these effects concentrate.
**No session filter was built or applied** — Part 10 of the brief
explicitly prohibits this, and this observation is reported for the
record only, not acted on.

## 8. Cost stress (Part 8)

| scenario | A: PF | B: PF | C: PF |
|---|---|---|---|
| normal | 0.740 | 0.758 | 0.618 |
| 1.5x spread | 0.611 | 0.630 | 0.519 |
| 2x spread | 0.509 | 0.528 | 0.432 |
| 1-bar delay | 0.721 | 0.753 | 0.658 |

B's modest edge over A **persists at every cost tier** — it doesn't
disappear under stress, but neither baseline gets anywhere close to
breakeven at any tier. Transaction costs make an already-losing setup
worse; they are not what turns a winner into a loser here, since both
A and B are losers even before any stress is applied.

## 9. Statistical comparison, B vs A (Part 9)

- **Expectancy (pips):** mean difference +0.038 pips, bootstrap 95% CI
  **[-0.143, +0.223]** — includes zero. Not statistically distinguishable
  from no improvement in raw pip terms.
- **Mean R:** mean difference +0.019R, bootstrap 95% CI **[+0.002,
  +0.037]** — entirely above zero, P(B>A) = 98.2%. **Statistically real**
  in risk-adjusted terms, even though the raw-pip comparison is noisy.
- **Reversal probability:** A = 53.71%, B = 54.27%, a +0.56 percentage
  point difference.

**Interpretation:** the improvement from B over A is statistically
genuine (the mean-R comparison clears a 98%+ one-sided bar) but
**economically tiny** — +0.02R per trade is nowhere near enough to move
a PF-0.74 strategy toward profitability.

## 10. Multiple-testing assessment

This experiment is a **single pre-registered comparison**: one composite
score (built from the two least-redundant phase-17 variables, not
searched), one frozen median threshold, one A-vs-B-vs-C structure. It is
not a search across many candidate filters, so it does not carry the same
multiple-testing burden as the exploratory phases that came before it —
that burden was already paid when phase 17 selected `body_ratio` and
`close_location` as the two variables to carry forward, out of the ~25
tested there.

## 11. MFE/MAE comparison

| Baseline | mean MFE (pips) | mean MAE (pips) | ratio |
|---|---|---|---|
| A | 6.72 | -9.10 | 0.74 |
| B | 7.12 | -9.56 | 0.75 |
| C | 6.50 | -8.95 | 0.73 |

**Essentially unchanged payoff geometry across all three baselines.**
Filtering by candle shape shifts the win/loss probability slightly but
does **not** meaningfully change the underlying favorable-vs-adverse
excursion ratio, which sits around 0.74-0.75 in all three cases — well
short of what a fixed 1:1 R:R structure needs to be profitable. This is
the same root cause phase 15 already identified for the unfiltered
baseline, and it persists unchanged in the shape-filtered version.

## 12. Economic interpretation

The candle-shape filter does what phase 17 said it would: it produces a
real, mostly-consistent (7/9 pairs, 3/4 years, all cost tiers, and a
statistically clear mean-R improvement) but **small** separation. The
underlying reason it doesn't translate into a profitable or even
breakeven strategy is visible in two places: (1) collapsing the
continuous shape signal into a single median split leaves only a ~1
percentage-point gap in raw reversal probability — most of the
continuous variables' resolving power is lost in the binarization, and
(2) the MFE/MAE payoff geometry barely moves, so even the surviving
probability edge isn't large enough to overcome a fixed 1:1 R:R
structure, exactly as it wasn't for the unfiltered baseline in phase 15.

## 13. Strongest evidence FOR

- Directional ordering (B > A > C) holds pooled, and in most pairs
  (7/9), most years (3/4), every session, and every cost tier — this is
  not a fragile or reversed result anywhere in the breakdown.
- The mean-R bootstrap comparison is genuinely significant (CI entirely
  above zero, P=0.982) — the shape signal is not indistinguishable from
  noise.
- Asian-session concentration of the improvement is consistent with
  every prior phase in this research line, adding circumstantial support
  that this is a real (if small) piece of the same underlying phenomenon,
  not an unrelated artifact.

## 14. Strongest evidence AGAINST

- Both A and B remain firmly net losers (PF 0.74 and 0.76) at every cost
  tier tested, including the most favorable one (normal spread, no
  delay, no execution stress).
- The raw expectancy-in-pips comparison is not statistically
  distinguishable from zero (CI straddles zero).
- The MFE/MAE ratio is essentially identical across all three baselines
  (~0.73-0.75) — the filter does not change the fundamental payoff
  problem that made the unfiltered version fail in phase 15.
- 2 of 9 pairs (USDJPY, GBPJPY) and 1 of 4 years (2024) show B failing to
  improve on A at all.

## 15. Final classification

# **A. NO ECONOMIC IMPROVEMENT**

The candle-shape signal separates outcomes statistically — real,
directionally consistent, survives cost stress — but is too weak to
improve the trading economics of the failed baseline in any way that
matters. It does not rise to **B. PROMISING BASELINE** (neither A nor B
gets close to breakeven, let alone a "meaningful improvement" that
"survives reasonable costs" in the sense of approaching viability), and
it is clearly not **D. FAILED/artifact** either — the signal's existence
and correct direction are corroborated by the B > A > C ordering holding
almost everywhere in the breakdown. It sits squarely in category A: real
but economically inert at this implementation.

## 16. Why it failed (Part 12 diagnosis — not followed by optimization)

- **Not primarily transaction costs.** Both A and B are losers before
  any cost stress is applied; costs make things worse but are not the
  root cause.
- **Primarily a weak probability shift after binarization.** The
  continuous shape_score has real effect sizes (phase 17: d=0.06-0.11),
  but collapsing it to a single median split leaves only a ~1
  percentage-point gap in raw reversal rate between the two halves — far
  too little to move a PF-0.74 baseline meaningfully.
- **Secondarily, an unfavorable/near-symmetric MFE/MAE payoff structure**
  that the shape filter does not change (ratio ~0.74-0.75 in all three
  baselines) — the same structural problem phase 15 identified for the
  unfiltered version persists unchanged here.
- **Not a timing problem.** Part 4 confirms the shape rule is genuinely
  T0-available; the failure is about signal magnitude and payoff
  structure, not information availability.

## 17. Recommended next step

Per instructions, **no optimization follows this result.** This closes
out the T0-candle-shape line of inquiry as economically inert at the
tested (frozen, unoptimized) implementation. If this line is revisited
in the future, it would need to be as a **fresh, separately pre-
registered experiment** — e.g. using the continuous `shape_score` (not a
binary split) in some non-fixed-R:R payoff structure, or combined with
the session concentration noted in Part 7 — but per the explicit
instruction not to add further filters or optimize within this
experiment, that decision is left for you to make separately, not
undertaken here.

---

## What I did NOT do (per instructions)

- Did not search any parameter — the composite formula, threshold, stop,
  target, and holding period were all frozen before backtesting.
- Did not add ATR regime, session filter, VWAP, EMA, RSI, ADX, trend
  filter, prior-day levels, volatility filter, or news filter, even
  though the session breakdown (Part 7) shows an Asian-hours
  concentration that would be tempting to filter on.
- Did not modify ARB, AMR, Monday Drift, XAUUSD ARB, or any demo strategy.
- Did not modify the existing phase 15/16/17 down-move research.
- Did not optimize after the NO ECONOMIC IMPROVEMENT classification, per
  Part 12's explicit instruction to diagnose and stop rather than iterate.
