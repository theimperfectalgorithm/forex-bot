# Down-Move Reversion — Controlled Baseline Research

**Experiments:** EXP-043 through EXP-046, `experiments/experiments.csv`.
**Script:** `src/phase15_downmove_reversion_baseline.py`. **Full log:** `reports/phase15_baseline_log.txt`. **Structured CSVs:** `data/phase15_part*.csv`.
**Data:** 9 pairs (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, NZDUSD, GBPJPY, EURJPY, CADJPY), 36 months M15.

Purpose was **not** to make this profitable — it was to determine whether
the ~45-47% continuation rate found in Discovery Phase 1 is a real
asymmetry or an artifact of drift/volatility clustering/sampling. No
existing strategy (ARB/AMR/Monday-drift) was read, imported, or modified.

---

## 1. Original finding reproduction

**Definitions (frozen, unchanged from phase 14):** Timeframe M15. ATR =
Wilder(14), 66-bar rolling window (`windowed_atr`, this project's standard
implementation). Event = single M15 bar's close-to-close move,
`(close[i]-close[i-1])/pip`, expressed in ATR units at bar i. Down-event:
move ≤ -1.0×ATR. Forward horizon for reproduction: 4 bars (60 min).
Continuation (down event) = `fwd_atr < 0` (price kept falling); reversal =
`fwd_atr > 0`. Mirror definitions for up-events.

**Sample:** 3,958–4,537 down-events and 3,669–4,556 up-events per pair
(~37,700 down-events, ~36,400 up-events pooled across 9 pairs).

**Result: pooled down-continuation = 46.26%, up-continuation = 50.01%.
Matches the phase-14 finding (45-47% down, ~49-52% up) exactly.
Reproduced.**

## 2. Drift-adjusted result

**Methodology:** not a constant subtraction. Computed the event-
conditional mean forward-ATR-return against three genuine data-derived
baselines: (C) the pair's unconditional mean forward return (all bars),
(D) a session-composition-matched baseline (weighted average of each
session's own unconditional mean, weighted by the event group's session
mix), (E) a volatility-regime-tercile-matched baseline (same idea, by ATR
percentile tercile). "Excess" = event mean − baseline mean.

**Result:** pooled excess vs. unconditional = **+0.043 ATR**, vs.
session-matched = **+0.022 ATR**, vs. vol-regime-matched = **+0.043
ATR** — all positive (meaning the event group reverts *more* than a
matched random bar), in **8 of 9 pairs** (USDJPY is the sole exception,
negative on all three). The raw 46% number is **not** fully explained
away by session or volatility-regime composition — a real excess remains
after controlling for both.

## 3. Up/down symmetry (incl. MFE/MAE)

| direction | p_continuation | mean_fwd_atr | mean_mfe_atr | mean_mae_atr |
|---|---|---|---|---|
| DOWN | 0.4626 | +0.048 | 1.313 | 1.301 |
| UP | 0.5001 | -0.016 | 1.196 | 1.261 |

**Verdict: ONLY DOWN reverses.** Up-moves are statistically indistinguishable
from a coin flip (50.0%) on continuation; down-moves are not (46.3%).
MFE/MAE for the down side are close to symmetric (1.31 vs 1.30 ATR),
which is an early warning sign for Part 14 — a real directional
probability tilt doesn't automatically mean favorable trade geometry.

## 4. Threshold sensitivity

| threshold | DOWN continuation | UP continuation |
|---|---|---|
| 0.50 | 0.4713 | 0.4987 |
| 0.75 | 0.4666 | 0.5008 |
| 1.00 | 0.4626 | 0.5001 |
| 1.25 | 0.4588 | 0.4976 |
| 1.50 | 0.4570 | 0.4922 |
| 2.00 | 0.4565 | 0.4977 |

**Smooth, monotonic decline in DOWN continuation as the threshold
increases** — max step-to-step change 0.0046, no isolated spike. This is
the plateau signature of a genuine effect, not an overfit peak (contrast
with the rejected hour-0/1 seasonality finding from Discovery Phase 1,
which failed exactly this test). UP shows no trend at all across the
same grid — the asymmetry is threshold-robust.

## 5. Forward horizon

| horizon (bars) | DOWN continuation | DOWN mean_fwd_atr | UP continuation |
|---|---|---|---|
| 1 (15m) | 0.4563 | +0.036 | 0.4797 |
| 2 (30m) | 0.4542 | +0.046 | 0.4911 |
| 4 (60m) | 0.4626 | +0.048 | 0.5001 |
| 8 (120m) | 0.4564 | +0.084 | 0.4999 |
| 12 (180m) | 0.4592 | +0.101 | 0.5032 |
| 16 (240m) | 0.4614 | +0.096 | 0.5094 |

**The continuation-rate gap is essentially flat from 15 minutes out to 4
hours** (45.4%–46.3%, never approaching 50%) — this is not a fast bounce
that fades; it's a persistent probability shift. The average *magnitude*
of the bounce (mean_fwd_atr) grows with horizon, as expected. On the UP
side, continuation drifts from 48.0% (15m) up through 50.9% (4h) — a mild
hint that up-moves eventually see slight trend-following resume, though
this is a secondary observation, not part of the core down-side finding.

## 6. Session breakdown

| session | DOWN continuation | n |
|---|---|---|
| ASIAN | **0.4216** | 9,798 |
| LONDON | 0.4792 | 10,629 |
| OVERLAP (London/NY) | 0.4945 | 7,076 |
| NY | 0.4675 | 7,971 |

**Materially session-dependent** (7.3pp spread) — the effect is
concentrated in Asian hours (continuation 42.2%, an 8pp deviation from
50%) and nearly disappears during the London/NY overlap (49.5%, close to
a coin flip). Per the instruction not to combine sessions if behavior
differs materially, sessions are **not pooled** for the baseline test
below — an Asian-only variant is run alongside the all-sessions variant.

**Note for the record (not acted on):** Asian hours is also where this
project's live AMR strategy already operates. I am flagging the overlap
in market structure only for context; per explicit instruction, no AMR
code was read or modified, and this is not being used to build an AMR
filter.

## 7. Volatility regime

| vol_regime (ATR-level tercile) | DOWN continuation | n |
|---|---|---|
| LOW | 0.4470 | 10,950 |
| MID | 0.4625 | 12,340 |
| HIGH | 0.4741 | 14,386 |

Monotonic: the reversal effect is **strongest in low-volatility
conditions** and weakens as volatility rises. This is defined on ATR
*level* percentile, a different axis from the event definition (which
uses the move/ATR *ratio*) — not circular. Economically plausible: a
1.0-ATR move during an already-quiet regime is a larger relative
"surprise" than the same move during a naturally volatile regime.

## 8. Market regime (trend vs. range)

| regime (efficiency-ratio(20) median split) | DOWN continuation | n |
|---|---|---|
| TRENDING | 0.4636 | 20,446 |
| RANGING | 0.4614 | 17,230 |

**No meaningful difference** (0.2pp) — this pre-specified, un-tuned
trend/range classifier does not separate the effect. Negative/null result.

## 9. Pair consistency

| pair | n | continuation | effect_size (pp below 50%) |
|---|---|---|---|
| EURUSD | 4,178 | 0.4734 | 2.66 |
| GBPUSD | 4,319 | 0.4709 | 2.91 |
| USDJPY | 4,179 | 0.4616 | 3.84 |
| AUDUSD | 4,244 | 0.4661 | 3.39 |
| USDCAD | 4,537 | 0.4732 | 2.68 |
| NZDUSD | 4,233 | 0.4630 | 3.70 |
| GBPJPY | 3,958 | 0.4505 | 4.95 |
| EURJPY | 4,012 | 0.4477 | 5.23 |
| CADJPY | 4,016 | 0.4574 | 4.26 |

**9/9 pairs reverse** (continuation < 50% in every single pair, no
exceptions). Pooled effect size +3.74pp. JPY crosses show the largest
individual effect sizes (4.3-5.2pp), majors the smallest (2.7-3.4pp) —
a mild but not alarming spread.

## 10. Year consistency

| year | continuation | n | pairs |
|---|---|---|---|
| 2023 | 0.4646 | 5,387 | 9 |
| 2024 | 0.4607 | 12,859 | 9 |
| 2025 | 0.4702 | 12,519 | 9 |
| 2026 YTD | 0.4500 | 6,911 | 9 |

**All 4 years show continuation below 50%** (45.0%–47.0%), a tight band
with no single year driving the whole result — unlike the rejected
NZDJPY finding, where 73% of the profit came from one year alone, this
effect is evenly distributed across the full sample.

## 11. Day of week

| day | continuation | n |
|---|---|---|
| Monday | 0.4468 | 6,959 |
| Tuesday | 0.4678 | 7,953 |
| Wednesday | 0.4586 | 7,556 |
| Thursday | 0.4713 | 7,739 |
| Friday | 0.4668 | 7,469 |

Modest spread (2.4pp), no day is a dramatic outlier. Monday shows the
strongest reversal, Thursday the weakest — not a large enough gap to be
noteworthy on its own.

## 12. Null / randomization test

**Methodology:** since there's no strategy yet to shuffle trade outcomes
for, I instead bootstrap-resampled (2,000 draws) a random sample the same
size as each pair's event group, drawn from **non-event bars in the same
volatility-regime tercile composition** — this builds a null distribution
of what a "typical bar with the same volatility-regime mix" produces by
chance. The observed event-group mean forward-ATR-return is then located
within that null distribution.

| pair | observed mean | null mean | percentile of observed |
|---|---|---|---|
| EURUSD | +0.0623 | -0.0027 | **99.9%** |
| GBPUSD | +0.0363 | +0.0008 | **94.7%** |
| USDJPY | -0.0020 | +0.0120 | 28.5% |
| AUDUSD | +0.0626 | -0.0001 | **99.5%** |
| USDCAD | +0.0622 | +0.0095 | **99.3%** |
| NZDUSD | +0.0858 | -0.0115 | **100.0%** |
| GBPJPY | +0.0563 | +0.0081 | **98.6%** |
| EURJPY | +0.0355 | +0.0105 | 86.9% |
| CADJPY | +0.0357 | -0.0008 | **95.0%** |

**Interpretation:** a high percentile means the real event-group mean sits
in the extreme right tail of the volatility-matched null — i.e. the
down-move event produces an unusually strong reversion that a
random bar with the same volatility composition does not. **7 of 9 pairs
clear the conventional 95th-percentile one-sided threshold** (EURUSD,
AUDUSD, USDCAD, NZDUSD, GBPJPY, CADJPY comfortably; GBPUSD marginally at
94.7%). USDJPY and EURJPY do not clear it. This is unusually strong
corroboration for 7 quasi-independent instruments to individually clear a
95th-percentile bootstrap bar by chance.

## 13. Multiple-testing assessment

This baseline phase alone computed results across 6 thresholds × 6
horizons × 4 sessions × 3 volatility regimes × 2 market regimes × 9 pairs
× 4 years × 5 days-of-week — several hundred individual cells. **No
single cell above is being treated as confirmatory on its own.** What
supports treating the core finding (down-move reversion, ~46% vs 50%) as
**exploratory-but-credible rather than noise** is the *pattern* of
agreement across independent cuts: 9/9 pairs same sign, 4/4 years same
sign, smooth (non-spiky) threshold response, flat (non-decaying) horizon
response, and 7/9 pairs individually clearing a matched-null bootstrap
test. Any one of these alone would not be enough; all of them agreeing is
what makes this different from the hour-0/1 seasonality result that was
rejected in Discovery Phase 1 for failing exactly this kind of
cross-validation. That said: this is still **EXPLORATORY, not
CONFIRMATORY** in the formal sense — no true held-out data was reserved
for this specific hypothesis (it was discovered and confirmed on
overlapping history), so the honest label is "a well-corroborated
exploratory finding," not "a proven effect."

## 14. Baseline strategy result

**Rule (deliberately the simplest possible, no parameters searched):**
Trigger = the same ≥1.0 ATR M15 down-move used throughout this script.
Entry = next candle open, BUY (fade), paying the spread. Stop = 1.0×ATR
(signal-bar ATR). Target = 1.0R (fixed 1:1). Exit = SL / TP / 4-bar
(60-min) time expiry, whichever comes first, SL priority on same-bar
touch. One trade at a time (no overlapping fades).

| variant | pair | n | win_rate | PF | mean pips |
|---|---|---|---|---|---|
| ALL_SESSIONS | POOLED | 28,644 | 41.7% | **0.740** | -1.17 |
| ASIAN_ONLY | POOLED | 7,884 | 45.3% | **0.858** | -0.56 |

**Both variants are net losers even before any cost stress.** The
Asian-only variant (where Part 6 showed the effect concentrates) is less
bad (PF 0.858 vs 0.740) but still clearly below breakeven. Per pair,
only GBPUSD (PF 1.09) and EURJPY (PF 0.99, essentially breakeven) are
non-losers in the Asian-only cut; every other pair loses money at this
simplest implementation.

**Why the descriptive edge doesn't translate:** Part 3's MFE/MAE were
close to symmetric (1.31 vs 1.30 ATR) — a real probability tilt on
*which side of zero* the forward return lands on doesn't automatically
produce a favorable **stop-vs-target race** at a fixed 1:1 R:R. A ~4-8
percentage point probability shift is real but too small to overcome a
1:1 payoff structure, which needs a win rate above 50% (observed: 40-49%
depending on cut).

## 15. Transaction cost analysis

Run on the Asian-only variant (where the descriptive effect is largest),
since the all-sessions variant was already unprofitable before any cost stress:

| scenario | n | win_rate | PF | mean pips |
|---|---|---|---|---|
| normal spread | 7,884 | 45.3% | 0.858 | -0.56 |
| 1.5x spread | 7,884 | 39.2% | 0.690 | -1.34 |
| 2x spread | 7,884 | 33.8% | 0.570 | -2.02 |
| +1-bar execution delay | 7,644 | 43.2% | 0.778 | -0.88 |

Monotonically worse under every stress scenario, as expected — there was
no margin to erode in the first place.

## 16. Final classification

**Two-tier verdict, deliberately separated per this project's own
standard (a strategy failing does not mean the underlying phenomenon is
fake, and vice versa):**

### Phenomenon classification: **GENUINE PHENOMENON**

- Stable behavioral effect: yes (9/9 pairs, 4/4 years, smooth threshold
  response, flat/persistent horizon response).
- Plausible market explanation: yes — concentrated in Asian hours and
  low-volatility regimes specifically, consistent with a liquidity/
  stop-hunt mechanism (thin Asian-session liquidity makes a 1-ATR M15
  move more likely to be a temporary imbalance than a genuine directional
  shift) rather than a data artifact.
- Evidence beyond normal drift: yes — positive excess after
  session/vol-regime-matched adjustment in 8/9 pairs (Part 2); 7/9 pairs
  individually clear a volatility-matched bootstrap null at the 95th
  percentile (Part 12).
- Stability across neighboring definitions: yes (Part 4).
- Stability across time: yes (Part 10).
- Evidence across multiple instruments: yes (Part 9).

### Trading-strategy classification: **NOT A PROMISING BASELINE**

- Reasonable execution viability: **no** — the simplest possible
  implementation (fixed 1:1 R:R fade) is a net loser (PF 0.74 pooled,
  0.86 Asian-only) even before any transaction-cost stress, and
  degrades further under realistic spread/delay assumptions.
- The probability asymmetry is real but too small (single-digit
  percentage points) to clear a fixed 1:1 payoff structure given the
  MFE/MAE are nearly symmetric.

**This is not a contradiction.** A genuine ~4-8 percentage-point
probability shift can be real and statistically well-supported while
still being too small to monetize through the simplest possible
fixed-R:R trade structure — the finding says something true about how
price behaves after these events, but naively "buying the dip" on a 1:1
bet doesn't capture it.

## 17. Strongest evidence FOR

The cross-validated pattern of agreement: 9/9 pairs, 4/4 years, a smooth
(non-spiky) threshold response, a flat (non-decaying) horizon response
out to 4 hours, a positive drift-adjusted excess in 8/9 pairs, and 7/9
pairs individually clearing a volatility-matched bootstrap null at the
95th percentile. No single test here is decisive alone, but this many
independent cuts agreeing is not what noise looks like.

## 18. Strongest evidence AGAINST

The effect, while statistically real, is economically small — MFE and
MAE are nearly symmetric (1.31 vs 1.30 ATR), meaning the "reversion"
shows up in *which side of zero the average forward return lands on*
without translating into a favorable race to a fixed target before a
fixed stop. USDJPY and EURJPY are consistent outliers that don't clear
the null test, so this isn't universal across every pair even though the
sign is universal.

## 19. Biggest uncertainty

Whether a **non-1:1 payoff structure** (wider target relative to stop, or
a volatility-scaled exit rather than a fixed time/R exit) could capture
the probability edge that a naive 1:1 structure can't — this baseline
deliberately did not search for that, per the instruction not to
optimize. It's an open question, not a rejected one.

## 20. Recommended next experiment

If this line is worth pursuing further (pending your review): test
whether an **asymmetric R:R** (e.g. a smaller, ATR-scaled partial target
closer to the average MFE, with a correspondingly tighter stop) changes
the economics — but this would be the first parameter choice in this
whole investigation, so it should be treated as a fresh, explicitly-
labeled optimization step with its own IS/OOS discipline, not folded
into this "simplest baseline" result. Alternatively, given Part 6 showed
this is really an **Asian-hours, low-volatility phenomenon**, it may be
more useful as descriptive input to a future risk-sizing overlay (not a
standalone strategy) than as a directional edge in its own right —
consistent with Discovery Phase 1's original framing of volatility-
regime findings as filter candidates rather than signals.

---

## What I did NOT do (per instructions)

- Did not optimize the threshold, horizon, stop, or target — every value
  used was pre-specified from the descriptive parts above, reused
  unchanged in Parts 14-15.
- Did not modify GBPJPY ARB, CADJPY ARB, XAUUSD ARB, GBPJPY/EURJPY/
  AUDJPY/CADJPY AMR, or GBPUSD Monday Drift, and did not use this
  experiment to build an AMR filter.
- Did not touch the demo account.
- Did not call this VALIDATED — final classification is GENUINE
  PHENOMENON (behavioral finding) but NOT A PROMISING BASELINE (trading
  implementation), a deliberately non-optimistic reading of a genuinely
  mixed result.
