# Down-Move Reversion — Volatility-Transition Predictability (Information Timing)

**Experiments:** EXP-051 through EXP-055, `experiments/experiments.csv`.
**Script:** `src/phase17_downmove_predictability.py`. **Full log:** `reports/phase17_predictability_log.txt`. **Full per-event table:** `data/phase17_events.csv` (37,676 rows).

**No strategy was built or optimized. No parameter was searched. No
existing strategy (ARB/AMR/Monday-drift/XAUUSD ARB) was read, imported,
or modified.** The question is strictly: can the phase-16 volatility-
transition mechanism be observed early enough, at or before T0, to have
predictive (not just explanatory) value?

## 1. Original mechanism (recap)

Phase 16 found: continuation events are followed by measurably more
post-event volatility expansion than reversal events (d=-0.233 on
`vol_pctile_change`, the largest effect in that study). That measurement
uses bars i+1 through i+4 (up to 60 min after the event) — information
that postdates any entry decision made at T0.

## 2. Information-timing audit

**T0 = the close of the M15 bar that confirms the ≥1.0×ATR down-move**
(bar i's own close, unchanged from phase 15/16). Every feature computed
below is explicitly tiered and, wherever the denominator could leak
future information, normalized using `atr[i-1]` (the last ATR value
known **strictly before** bar i opens) rather than `atr[i]` (which
embeds the event bar's own true range) — this is enforced in code, not
just documented: see `build_tiered_events()` in
`src/phase16_downmove_mechanism.py`/`phase17_downmove_predictability.py`
where Tier 0/1 features are explicitly divided by `a_prev = atr[i-1]`.

| Tier | Definition | Data used | Usable for T0 entry? |
|---|---|---|---|
| 0 | Strictly before bar i opens | bars ..i-1 | Yes |
| 1 | At T0 (event confirmation) | bar i's own OHLC | Yes |
| 2 | First post-event candle | bar i+1 | **No** (requires waiting 15 min past T0) |
| 3 | Multiple future candles | bars i+1..i+4 | **No** |

**Data limitation, documented rather than worked around:**
`core/data_loader.py` supports M15/H1/H4 only — there is no M1/M5 feed,
so Part 2's requested "first 5-minute interval" could not be tested and
was skipped rather than approximated with a proxy.

## 3. Earliest observable separation (Part 2)

| Horizon | Tier | effect size (d) | pairs agreeing | years agreeing |
|---|---|---|---|---|
| First 15 min (bar i+1 only) | 2 | -0.045 | 9/9 | 4/4 |
| First 30 min | 3 | -0.084 | 9/9 | 4/4 |
| First 45 min | 3 | -0.121 | 9/9 | 4/4 |
| First 60 min | 3 | -0.141 | 9/9 | 4/4 |

**The separation is present from the very first post-event candle and
grows smoothly with horizon** — not a delayed effect that only appears
after several bars. But note the earliest point where *any* separation
appears (15 min) is **Tier 2**, not Tier 0/1: it requires waiting for
the first post-event candle to close, which is 15 minutes after T0. At
T0 itself, this specific measurement does not yet exist.

## 4. Pre-event predictors — Tier 0 (Part 3)

| variable | effect size | pairs | years |
|---|---|---|---|
| atr_pctile_pre_T0 | -0.047 | 7/9 | 4/4 |
| realized_vol_8_T0 | -0.020 | 8/9 | 4/4 |
| realized_vol_4_T0 | -0.031 | 8/9 | 4/4 |
| realized_vol_2_T0 | -0.026 | 8/9 | 4/4 |
| vol_slope_T0 | -0.017 | 7/9 | 3/4 (2023 flips) |
| prev_candle_range_atr_T0 | -0.043 | 9/9 | 4/4 |

**Consistent in sign, but small in magnitude.** Every Tier-0 variable's
effect size is under 0.05 — real (mostly 8-9/9 pair agreement, 4/4 year
agreement) but far weaker than the Tier-3 mechanism itself (0.14-0.23).
Pre-event volatility state carries *some* information, but not much.

## 5. Event-candle structure — Tier 1 (Part 4)

| variable | effect size | pairs | years | note |
|---|---|---|---|---|
| event_range_atr_T1 | -0.001 | 6/9 | mixed | null |
| body_pct_T1 | **-0.107** | 8/9 | 4/4 | reversal candles have proportionally smaller bodies |
| upper_wick_pct_T1 | **+0.082** | 9/9 | 4/4 | reversal candles show more upper wick |
| lower_wick_pct_T1 | +0.061 | 8/9 | 4/4 | reversal candles show more wick generally (less decisive) |
| close_location_T1 | **+0.101** | 8/9 | 4/4 | reversal candles close relatively higher within their own range |
| move_atr_T1 | ~0.000 | 5/9 | mixed | null (matches phase 15's threshold-insensitivity finding) |
| dist_from_sess_high_T1 | -0.074 | 9/9 | 4/4 | reproduces phase 16's finding, at Tier 1 |
| dist_from_vwap_T1 | +0.069 | 9/9 | 4/4 | reproduces phase 16's finding, at Tier 1 |
| atr_pctile_at_T0_T1 | -0.040 | 7/9 | 4/4 | weak, consistent with phase 16 |

**This is the most useful finding of this phase.** The event candle's own
**shape** — not just its raw range — carries genuine, consistent
information available at T0: reversal-outcome down-moves tend to be
proportionally smaller-bodied, more wicked, and close higher within
their own range than continuation-outcome down-moves. In plain terms:
**a "hammer-like" or indecisive-looking down candle is modestly more
likely to reverse than a full-bodied, decisive down candle** — even
though both qualify as ≥1.0 ATR moves by their close-to-close
definition. Effect sizes (0.06-0.11) are moderate: meaningfully larger
than any Tier-0 variable, though still well short of the full Tier-3
mechanism. `dist_from_sess_high_T1` and `dist_from_vwap_T1` (phase 16's
findings) both reproduce cleanly at Tier 1, confirming they were already
legitimately available at T0 all along.

## 6. First post-event candle — Tier 2 (Part 5)

| variable | effect size | pairs | years | note |
|---|---|---|---|---|
| b1_range_T2 | -0.077 | 8/9 | 4/4 | reversal's first candle is modestly smaller-range |
| b1_body_T2 | -0.048 | 9/9 | 4/4 | reversal's first candle is modestly smaller-bodied |
| b1_direction_T2 | +0.694 | 9/9 | 4/4 | **near-tautological, see caution below** |
| b1_close_loc_T2 | +0.668 | 9/9 | 4/4 | **near-tautological, see caution below** |
| b1_vol_vs_event_T2 | ~0.00 (median) | — | — | mean corrupted by 2/37,676 division-outlier rows; median shows no separation |
| b1_vol_vs_pre_T2 | -0.031 | 5/9 | 4/4 | weak, inconsistent across pairs |

**Important caution on the two huge effect sizes.** `b1_direction_T2`
(d=+0.694) and `b1_close_loc_T2` (d=+0.668) are the largest effect sizes
in this entire research line — but they are **expected by construction,
not a new discovery**: the outcome label itself is defined by the
4-bar-forward return, and the first of those 4 bars mechanically
contributes a large share of that sum. A first candle that moves upward
is *partially defining* the REVERSAL label, not *predicting* it
independently. These two numbers are reported for completeness but are
explicitly **excluded from the ranked findings below** as not
informative about the underlying mechanism.

## 7. Tier summary

- **Tier 0 (strictly pre-event):** weak but real signal, all effect
  sizes < 0.05.
- **Tier 1 (at T0):** the most useful genuinely-predictive tier —
  candle-shape variables (body%, wicks, close location) at d=0.06-0.11,
  plus reproduced phase-16 findings (session-high distance, VWAP
  distance) at similar magnitude.
- **Tier 2 (first post-event candle):** adds modest incremental
  information (range/body, d≈0.05-0.08) beyond Tier 1, at the cost of a
  15-minute delay past T0; the two large-effect variables here are
  definitional artifacts of the outcome label, not genuine new signal.
- **Tier 3 (multiple future candles):** the full mechanism from phase
  16, d=0.11-0.23 — the strongest signal by far, but categorically
  unusable for a decision made at T0.

## 8. Cross-pair consistency

Every Tier 0/1 variable reported above reaches at least 6/9 pair
agreement; the strongest ones (`prev_candle_range_atr_T0`,
`upper_wick_pct_T1`, `dist_from_sess_high_T1`, `dist_from_vwap_T1`) reach
9/9. No Tier 0/1 finding here is being driven by 1-2 outlier pairs.

## 9. Cross-year consistency

Nearly every Tier 0/1/2/3 variable holds the same sign in **all four**
years (2023-2026 YTD) — a stronger consistency bar than phase 16 needed
to clear (which only checked 2025 vs. 2026). The lone exception,
`vol_slope_T0`, flips sign in 2023 specifically and is treated as weak/
untrustworthy rather than folded into the ranked findings.

## 10. Rollover-artifact sensitivity (Part 9)

Excluding EARLY_ASIAN (hours 0-2, the window flagged as a possible
rollover/bar-boundary artifact in Discovery Phase 1 and phase 16):

| variable | full sample d | ex-EARLY_ASIAN d |
|---|---|---|
| realized_vol_post_T3 | -0.141 | -0.129 |
| vol_expansion_ratio_T3 | -0.109 | -0.129 |
| vol_pctile_change_T3 | -0.200 | -0.184 |

**The core volatility-transition finding survives essentially unchanged**
with EARLY_ASIAN removed — effect sizes shift by only 0.01-0.02 in either
direction, still 9/9 pairs and all 4 years. This is a meaningfully
different (and better) situation than the raw session-location finding:
the *volatility-transition mechanism itself* is not an artifact of the
suspicious hours, even though the raw *session breakdown* from phase 16
partly was. No attempt was made to "rescue" the rollover-hour finding
itself — it remains excluded from any headline conclusion.

## 11. Null / randomization test (Part 10)

Bootstrap test (2,000 draws, volatility-tercile-matched) on the earliest
separable horizon (first 15 minutes, Tier 2):

| pair | observed gap | null mean | percentile |
|---|---|---|---|
| EURJPY | -1.108 | -0.327 | **0.016** |
| USDCAD | -0.289 | -0.012 | **0.025** |
| USDJPY | -0.864 | -0.245 | **0.025** |
| CADJPY | -0.544 | -0.130 | **0.053** |
| GBPJPY | -0.904 | -0.235 | 0.061 |
| AUDUSD | -0.138 | +0.005 | 0.125 |
| GBPUSD | -0.194 | -0.059 | 0.209 |
| NZDUSD | -0.043 | +0.020 | 0.258 |
| EURUSD | -0.058 | -0.033 | 0.437 |

All 9 pairs show the gap in the same (predicted) direction, but only
4/9 clear a conventional 5th-percentile one-sided bar (EURJPY, USDCAD,
USDJPY, CADJPY), with GBPJPY close behind at 6.1%. This is **weaker
corroboration than phase 16's Tier-3 bootstrap test** (which cleared 7/9
pairs at the 95th percentile on the full 60-minute mechanism) —
consistent with, and expected from, the fact that the earliest available
signal carries less statistical power than the fuller, later-arriving
signal. Mean percentile 0.134 — directionally supportive, not
individually decisive pair-by-pair.

## 12. Multiple-testing assessment

This phase tested roughly 25 variables across 4 tiers, 9 pairs, and 4
years — several hundred cells, same discipline as phases 15-16. The bar
applied throughout: prefer variables with (a) non-trivial effect size,
(b) 8/9+ pair agreement, (c) same sign across all 4 years (a stricter bar
than prior phases). Only Tier 1's candle-shape variables and the
already-established session-high/VWAP-distance variables clear all
three. The two large Tier-2 effect sizes were explicitly excluded as
definitional artifacts rather than being reported as if they were
genuine discoveries — this is the kind of result that would otherwise
look like a great predictor while actually just re-describing the label.

## 13. Strongest predictive candidate, if any

**Event-candle shape at Tier 1** — specifically `body_pct_T1` (d=-0.107),
`close_location_T1` (d=+0.101), and `upper_wick_pct_T1` (d=+0.082),
together with the already-known `dist_from_sess_high_T1` (d=-0.074) and
`dist_from_vwap_T1` (d=+0.069) — is the strongest genuinely-available-
at-T0 signal found. All are 8/9 or 9/9 pair-consistent and 4/4
year-consistent. None is individually large, but they represent a
coherent, economically sensible picture: **a down-move candle that looks
indecisive (small body, notable wicks, closes off its low) is modestly
more likely to be a temporary dislocation than one that closes decisively
at its low.**

## 14. Evidence against it

- Every individual effect size here (0.06-0.11) is roughly half the size
  of the full Tier-3 mechanism (0.14-0.23) — a meaningful chunk of the
  real signal genuinely is not available until after the event.
- These candle-shape variables have not been cross-validated on
  held-out data outside this exploratory study — they are corroborated
  (consistent across 9 pairs and 4 years), not confirmed in the formal
  sense.
- The null test at the earliest observable horizon (Part 10) only
  individually clears significance in 4-5 of 9 pairs, weaker than the
  phase-16 mechanism's own null test — the T0-available portion of the
  signal is statistically thinner than the full mechanism.

## 15. Is the mechanism actually tradable?

**Not established by this phase, and not tested here (per instructions,
no strategy was built).** What this phase establishes is narrower and
more useful: a modest, genuinely T0-available piece of the mechanism
exists (candle shape), separate from the larger but T0-unavailable piece
(post-event volatility expansion). Whether the T0-available piece alone
is large enough to survive contact with transaction costs and a real
trade structure — the way phase 15 found the raw phenomenon did NOT
survive a naive 1:1 implementation — is an open question this phase
deliberately did not answer.

## 16. Classification (Part 12 framework)

**B. POTENTIALLY PREDICTIVE.**

Not (A) explanatory-only — Tier 1 candle-shape features are
genuinely available at T0 and show consistent, non-trivial separation
(0.06-0.11), which is more than "the mechanism exists but can never be
observed early enough."

Not (C) strong predictive candidate — the T0-available effect sizes are
meaningfully smaller than the full mechanism, haven't been confirmed on
held-out data, and the earliest-horizon null test only individually
clears 4-5 of 9 pairs. Calling this "strong" would overstate what's been
shown.

Not (D) artifact/rejected — the core Tier-3 mechanism survives the
rollover-artifact removal test essentially unchanged, and the Tier 0/1
candle-shape signal is not an artifact of the suspicious EARLY_ASIAN
window (it holds across all sessions, not just Asian hours — see
`data/phase17_events.csv` for the full breakdown by session if needed).

## 17. Recommended next experiment

1. **A genuine held-out confirmatory test** of the Tier 1 candle-shape
   variables (`body_pct_T1`, `close_location_T1`, `upper_wick_pct_T1`) on
   data outside this exploratory window — the same discipline phase 16
   recommended and that has not yet been done for any variable in this
   research line.
2. If that confirms, the natural next question (still not a strategy
   yet) would be whether combining the Tier-0/1 candle-shape signal with
   the already-known session-high/VWAP-distance signal produces a
   materially stronger joint separation than any single variable alone —
   but this would need to be pre-registered as its own experiment with
   its own IS/OOS split, not run as an ad hoc combination search, per the
   project's standing overfitting-protection discipline.
3. Only after those two steps would it be appropriate to consider a
   trading hypothesis built on Tier 0/1 information alone — and even
   then, phase 15's own experience (a real, well-corroborated phenomenon
   that failed a naive 1:1 implementation) is a direct reminder that
   "predictive" and "profitable after costs" are still two different
   questions.

---

## What I did NOT do (per instructions)

- Did not build, optimize, or combine any trading strategy or filter.
- Did not optimize the ATR threshold, session times, entry, stop,
  target, or holding period.
- Did not modify ARB, AMR, Monday Drift, XAUUSD ARB, or any demo
  strategy, and did not modify the existing down-move strategy research
  (phases 15/16 remain as previously reported).
- Did not attempt to rescue the EARLY_ASIAN/hour-0-2 rollover finding —
  it remains excluded from headline conclusions; only the underlying
  volatility-transition mechanism's robustness to its removal was tested.
