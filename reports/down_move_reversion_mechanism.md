# Down-Move Reversion — MECHANISM Research

**Experiments:** EXP-046 through EXP-050, `experiments/experiments.csv`.
**Script:** `src/phase16_downmove_mechanism.py`. **Full log:** `reports/phase16_mechanism_log.txt`. **Full per-event table:** `data/phase16_events.csv` (37,676 rows, for any follow-up slicing without re-fetching).

**This phase does not touch the trading implementation.** No parameter
was searched, no filter was combined, no strategy was built or modified,
and AMR/ARB/Monday-drift were not read or touched anywhere in this work.
The question is strictly: *what distinguishes down-moves that reverse
from down-moves that continue?*

## 1. Event definition (frozen, unchanged from phase 15)

M15 close-to-close move ≤ -1.0×ATR(Wilder14, 66-bar window). Outcome
split (frozen, unchanged, 4-bar/60-min horizon): **REVERSAL** = forward
ATR-normalized return ≥ 0 (net higher 60 min later); **CONTINUATION** =
forward ATR-normalized return < 0.

## 2. Reversal vs. continuation population

**37,676 events pooled across 9 pairs: 20,234 REVERSAL (53.7%) / 17,442
CONTINUATION (46.3%).** (This 53.7% figure is the same phenomenon as
phase 15's "46.3% continuation" reported from the other side — consistent.)
Per-pair counts range 3,958–4,537, all comfortably powered for the
comparisons below.

---

## 3. Hypothesis 1 — Broader market location

| variable | REVERSAL mean | CONTINUATION mean | effect size (Cohen's d) | cross-pair agreement | 2025 vs 2026 |
|---|---|---|---|---|---|
| dist_from_sess_high_atr | 2.758 | 2.863 | -0.074 | 9/9 | SAME |
| dist_from_sess_low_atr | 0.807 | 0.790 | +0.016 | 6/9 | DIFFERENT |
| pos_in_day_range | 0.335 | 0.327 | +0.026 | 8/9 | SAME |
| pos_in_prevday_range* | 0.407 (median) | 0.401 (median) | ~0 | 6/9 | SAME |
| dist_from_prevday_high_atr | 6.539 | 6.534 | +0.001 | 6/9 | SAME |
| dist_from_prevday_low_atr | 4.688 | 4.667 | +0.004 | 5/9 | SAME |
| dist_from_recent_h1_high_atr | 3.290 | 3.323 | -0.021 | 5/9 | DIFFERENT |
| dist_from_recent_h1_low_atr | 1.028 | 1.028 | ~0 | 6/9 | DIFFERENT |
| dist_from_recent_h4_high_atr | 6.533 | 6.547 | -0.003 | 5/9 | DIFFERENT |
| dist_from_recent_h4_low_atr | 4.400 | 4.391 | +0.002 | 5/9 | SAME |

*`pos_in_prevday_range`'s mean is corrupted by 19/37,676 outlier rows
where the previous day's range was near-zero (division blowup); the
median (reported above) is the reliable statistic and shows essentially
no separation.

**Only one variable in this hypothesis is both non-trivial in size and
fully consistent: `dist_from_sess_high_atr`** (effect -0.074, 9/9 pairs
agree, both years agree). Reversal events occur, on average, **closer to
the session high** than continuation events (2.76 vs 2.86 ATR away). The
rest of Hypothesis 1's variables are small (|d| < 0.03) and/or
inconsistent across years — recent-H1/H4-extreme distances flip sign
between 2025 and 2026, which is a red flag against trusting them.

**Interpretation:** a down-move that hasn't traveled far from the
session's high yet (still "close to the top") is modestly more likely to
be a pullback that snaps back, versus one occurring deep into an already
extended decline. This is a small effect, not a strong discriminator on
its own.

---

## 4. Hypothesis 2 — Prior directional pressure

| variable | effect size | cross-pair agreement | 2025 vs 2026 |
|---|---|---|---|
| ret_1h_atr | +0.012 | 7/9 | SAME |
| ret_4h_atr | +0.005 | 6/9 | DIFFERENT |
| ret_8h_atr | +0.010 | 5/9 | DIFFERENT |
| persistence_1h | -0.006 | 5/9 | DIFFERENT |
| persistence_4h | ~0 | 6/9 | SAME |
| persistence_8h | -0.002 | 5/9 | SAME |
| prev_session_ret_pips | -0.019 | 8/9 | DIFFERENT |
| prev_day_ret_pips | +0.025 | 6/9 | SAME |

**Essentially a null result.** Every effect size is negligible (|d| <
0.025, an order of magnitude smaller than Hypothesis 3's strongest
finding below), and roughly half the variables flip sign between 2025
and 2026 — the signature of noise, not signal. **Whether the market was
already under bearish pressure before the event, over any of the 5
pre-specified windows, does not meaningfully distinguish reversal from
continuation.** This directly answers the hypothesis's question: no,
reversal does not occur mainly in "neutral/bullish" preceding conditions
as opposed to already-bearish ones — prior direction doesn't discriminate
either way.

---

## 5. Hypothesis 3 — Volatility transition

| variable | REVERSAL mean | CONTINUATION mean | effect size | cross-pair agreement | 2025 vs 2026 |
|---|---|---|---|---|---|
| atr_pctile_pre | 0.530 | 0.541 | -0.040 | 7/9 | SAME |
| event_range_atr | 1.848 | 1.848 | ~0 | 6/9 | DIFFERENT |
| realized_vol_pre | 5.576 | 5.676 | -0.020 | 8/9 | SAME |
| realized_vol_post | 6.260 | 7.156 | **-0.157** | **9/9** | SAME |
| vol_expansion_ratio | 1.402 | 1.537 | **-0.118** | **9/9** | SAME |
| vol_pctile_change | 0.037 | 0.071 | **-0.233** | **9/9** | SAME |

**This is the strongest and most consistent hypothesis by a wide margin.**
`vol_pctile_change` has the largest effect size found anywhere in this
research (-0.233), and all three post-event volatility measures agree in
both sign and near-perfect cross-pair/cross-year consistency (9/9 pairs,
both years, every time).

**The direction is the OPPOSITE of what the hypothesis speculated.** The
hypothesis asked whether "LOW VOLATILITY → SUDDEN EXPANSION → REVERSAL"
is the real mechanism, as distinct from "LARGE MOVE → REVERSAL" alone.
The data says: **continuation events are the ones followed by MORE
volatility expansion, not reversal events.** Reversal events show
comparatively muted follow-through volatility (mean expansion ratio 1.40x
vs 1.54x for continuation; ATR percentile rises +0.037 after reversal
events vs +0.071 after continuation events). In plain terms: **a
down-move that keeps triggering more volatility after it happens tends
to keep going; a down-move that is NOT followed by a real volatility
follow-through tends to snap back.** This is economically coherent — a
move accompanied by genuine expanding participation looks like a real
repricing event, while a move that doesn't recruit further volatility
looks more like a temporary liquidity dislocation that reverts once the
imbalance clears.

---

## 6. Hypothesis 4 — Session location

| session | reversal rate |
|---|---|
| ASIAN | 57.8% |
| LONDON | 52.0% |
| OVERLAP | 50.5% |
| NY | 53.1% |

Confirms phase 15's session concentration from the other direction: Asian
hours show a materially higher reversal rate.

| Asian third (pre-defined) | reversal rate | n |
|---|---|---|
| EARLY_ASIAN (hours 0-2) | **67.3%** | 3,965 |
| MID_ASIAN (hours 3-4) | 50.6% | 4,026 |
| LATE_ASIAN (hours 5-6) | 53.1% | 1,807 |

**Important caution, not a clean confirmation.** The entire Asian-session
effect is concentrated almost exclusively in the EARLY_ASIAN window
(hours 0-2 server time) — MID and LATE Asian are close to a coin flip.
**Hours 0-2 overlap directly with the hour-0/1 "seasonality spike" that
Discovery Phase 1 explicitly investigated and REJECTED as a likely
rollover/bar-boundary artifact** (isolated single-hour spikes, not a
smooth region, on nearly every pair). I am flagging this prominently
rather than presenting EARLY_ASIAN's 67.3% reversal rate as clean
evidence — it may be measuring the same data-construction artifact rather
than a genuine behavioral effect. This needs to be resolved before
trusting the session-location finding at all, let alone using it.

---

## 7. Hypothesis 5 — Range break vs. internal move

| | reversal rate | n |
|---|---|---|
| Internal move (no break) | 53.3% | 20,955 |
| Range break (20-bar low breached) | 54.2% | 16,721 |

**Null result** — 0.9 percentage point difference, well within noise.
**No support for the hypothesis** that internal/liquidity-sweep moves
reverse while genuine range breaks continue. Whether the down-move
breaches an established 20-bar range boundary does not meaningfully
predict the outcome.

---

## 8. Hypothesis 6 — Distance from session VWAP

| | REVERSAL mean | CONTINUATION mean | effect size | cross-pair | years |
|---|---|---|---|---|---|
| dist_from_vwap_atr | -1.151 | -1.220 | +0.063 | 9/9 | SAME |

Small but **fully consistent** (9/9 pairs, both years): reversal events
occur modestly *closer* to the session VWAP than continuation events (the
event bar's close is less far below VWAP). This is directionally
coherent with a mean-reversion-toward-a-fair-value-anchor story, though
the effect size (0.063) is smaller than Hypothesis 3's findings.

---

## 9. Cross-pair consistency (summary across all hypotheses)

Only variables clearing 9/9 (or 8/9) cross-pair sign agreement:
`dist_from_sess_high_atr` (9/9), `realized_vol_post` (9/9),
`vol_expansion_ratio` (9/9), `vol_pctile_change` (9/9),
`dist_from_vwap_atr` (9/9), `realized_vol_pre` (8/9), `pos_in_day_range`
(8/9), `prev_session_ret_pips` (8/9, but flips year sign — see below).
Everything else in Hypotheses 1, 2, and the raw session-location
sub-split falls short of this bar or shows contradictory year behavior.

## 10. Cross-year consistency (2025 vs. 2026)

All of Hypothesis 3's headline variables (`realized_vol_post`,
`vol_expansion_ratio`, `vol_pctile_change`) and Hypothesis 6
(`dist_from_vwap_atr`) and the strongest Hypothesis-1 variable
(`dist_from_sess_high_atr`) hold the **same sign in both 2025 and
2026**. Several Hypothesis-1 and Hypothesis-2 sub-variables flip sign
between years (`dist_from_recent_h1_high_atr`, `ret_4h_atr`, `ret_8h_atr`,
`persistence_1h`, `prev_session_ret_pips`) — these are explicitly **not**
trusted as real, regardless of their pooled effect size, since a genuine
mechanism shouldn't reverse direction year to year.

---

## 11. Multiple-testing assessment

24 explanatory variables were tested across 6 hypotheses. **No single
p-value or subgroup result is being treated as confirmatory.** The filter
applied throughout this report is the same one used in prior phases:
prefer variables with (a) meaningful effect size, (b) 9/9 or near-9/9
cross-pair sign agreement, and (c) same-sign results in both 2025 and
2026. Only 4 variables clear all three bars cleanly:
`dist_from_sess_high_atr`, `realized_vol_post` /
`vol_expansion_ratio` / `vol_pctile_change` (functionally one signal, not
three independent ones), and `dist_from_vwap_atr`. Everything else —
including the superficially striking EARLY_ASIAN 67.3% reversal rate —
is being explicitly flagged as **not confirmatory**, either for weak
cross-pair/year agreement (Hypothesis 1's distance-to-recent-extreme
variables, all of Hypothesis 2) or for a plausible confound with an
already-identified data artifact (EARLY_ASIAN).

---

## 12. Strongest explanatory variables (ranked)

| Rank | Variable | Effect size | Stability | Economic plausibility | Simplicity |
|---|---|---|---|---|---|
| 1 | `vol_pctile_change` (post-event volatility percentile shift) | -0.233 (largest found) | 9/9 pairs, both years | High — distinguishes genuine repricing from liquidity dislocation | Simple |
| 2 | `vol_expansion_ratio` / `realized_vol_post` (same underlying signal) | -0.118 / -0.157 | 9/9 pairs, both years | Same as #1 | Simple |
| 3 | `dist_from_vwap_atr` | +0.063 | 9/9 pairs, both years | Moderate — proximity to a fair-value anchor | Simple |
| 4 | `dist_from_sess_high_atr` | -0.074 | 9/9 pairs, both years | Moderate — "how far into the decline" as an exhaustion proxy | Simple |
| 5 | Session location (Asian > other sessions) | large (57.8% vs ~52%) | Driven almost entirely by EARLY_ASIAN sub-window | **Confounded** — overlaps a previously-rejected data artifact | Simple, but untrustworthy as-is |
| 6 (null) | Prior directional pressure (all H2 variables) | negligible | Weak/inconsistent | Plausible in theory, not supported by data | — |
| 6 (null) | Range break vs. internal move | negligible (0.9pp) | N/A (no effect to test) | Plausible in theory, not supported by data | — |

---

## 13. Evidence AGAINST each explanation

- **Volatility transition (#1-2):** none found against the direction of
  the effect itself, but the *magnitude* is still modest (Cohen's d ~0.1-0.2
  is a small-to-medium effect by conventional standards) — this explains
  part of the reversal/continuation split, not most of it. `atr_pctile_pre`
  (the pre-event level) is much weaker (-0.040) than the post-event change,
  meaning the *starting* volatility regime is a weaker signal than the
  *subsequent* volatility trajectory — which is a subtlety worth noting:
  it's harder to act on a "what happens after" variable in a live signal.
- **VWAP distance (#3) / session-high distance (#4):** both are real but
  small (Cohen's d 0.06-0.07) — neither, alone, would separate the two
  populations with any practical reliability.
- **Session location (#5):** the EARLY_ASIAN result is the single most
  eye-catching number in this whole report (67.3% vs a 50-53% baseline
  elsewhere) and is exactly the kind of result that should NOT be trusted
  at face value, given this project's own prior finding that the same
  hour window produces artifact-signature results in raw seasonality data.
- **Prior directional pressure (#6) and range break/internal (#6):** no
  evidence found for either hypothesized mechanism at all — both are
  clean negative results, not "weak positive" ones.

---

## 14. Is a coherent mechanism emerging?

**Partially, yes — with an important caveat.** The most defensible
picture, using only the variables that survived cross-pair and cross-year
scrutiny, is:

> Down-moves that are **not** followed by genuine volatility/participation
> expansion, and that occur **relatively close to** a recent fair-value
> anchor (session VWAP) or **haven't traveled far from** the session's
> high yet, are somewhat more likely to be temporary liquidity
> dislocations that snap back. Down-moves that trigger real follow-through
> volatility, or that occur deep into an already-extended decline, are
> somewhat more likely to be genuine repricing events that continue.

This is economically sensible and not contradicted by any of the tested
variables. **However**, all four surviving explanatory variables are
individually small effects (Cohen's d 0.06-0.23) — this is a partial,
modest mechanism, not a strong, dominant one. And the single largest
*apparent* signal in the whole study (EARLY_ASIAN session timing) is
under a specific, credible suspicion of being a data artifact rather than
part of this mechanism, which means the true picture may be even more
modest than the ranked table suggests once/if that's resolved.

## 15. Does any variable deserve future confirmatory testing?

**Yes: the volatility-transition variables (`vol_pctile_change` /
`vol_expansion_ratio`), and the VWAP-distance variable**, given they are
the only ones clearing effect-size, cross-pair, AND cross-year bars
simultaneously. They deserve a genuine held-out confirmatory test (fresh
data, ideally from a period not used in this exploratory phase) before
being treated as more than "exploratory but well-corroborated," per the
statistical-discipline instruction not to promote exploratory findings to
confirmed status within the same study that discovered them.

**The EARLY_ASIAN session finding specifically needs the artifact
question resolved BEFORE any further confirmatory work** — it would be a
waste of research budget to build confirmatory tests on top of a result
that may simply be re-measuring the previously-rejected hour-0/1 artifact
from a different angle.

## 16. Recommended next experiment

1. **Resolve the EARLY_ASIAN vs. artifact question directly**: rerun the
   Hypothesis 4 session breakdown using a data source/method less exposed
   to bar-boundary/rollover effects (e.g., exclude the literal first bar
   of each trading day, or use a different broker/feed for cross-
   verification) before trusting or discarding the Asian-session finding.
2. **A genuine held-out confirmatory test** of `vol_pctile_change` and
   `dist_from_vwap_atr` on data outside this study's exploratory window
   (e.g., the next several months going forward, or a pre-2023 sample if
   available) — this is the only way to move these two variables from
   "well-corroborated exploratory" to "confirmatory" per Part 13's own
   discipline.
3. Do **not** yet combine these variables into a filter or build a
   strategy on them — per instructions, that step (if ever taken) is
   explicitly out of scope for this phase.

## Note on AMR relevance (documented only, not implemented)

Per instructions, this is documentation only — nothing here was
implemented or used to modify AMR. The volatility-transition finding
(down-moves NOT followed by expansion tend to revert) and the earlier
low-ATR-percentile association are both, in principle, the *kind* of
signal that could eventually inform a mean-reversion-friendly-conditions
filter for AMR, since AMR is itself a mean-reversion strategy operating
in the same Asian-hours window this research keeps surfacing. This is
noted for the record as a possible future research thread, explicitly
**not** acted on here, and still subject to the same EARLY_ASIAN artifact
caution above before it would be worth pursuing.

---

## What I did NOT do (per instructions)

- Did not optimize the failed 1:1 baseline trading implementation.
- Did not search for the best stop, target, holding period, entry delay,
  ATR threshold, or indicator combination.
- Did not combine multiple filters to maximize PF, and did not build a
  final strategy.
- Did not modify AMR, ARB, or Monday Drift, and did not implement an AMR
  filter — the AMR-relevance note above is documentation only.
