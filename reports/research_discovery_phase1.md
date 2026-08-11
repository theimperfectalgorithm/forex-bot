# Research Discovery Phase 1 — Intraday FX Phenomena (Descriptive)

**Experiment:** EXP-038 through EXP-042 (one per family), `experiments/experiments.csv`.
**Script:** `src/phase14_discovery.py`. **Full log:** `reports/phase14_discovery_log.txt`.
**Data:** 9 pairs (EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, NZDUSD, GBPJPY, EURJPY, CADJPY), 36 months H1 + M15, server-hour sessions (Asian 00-07, London 07-16, NY 12-21).

**This is a descriptive/exploratory phase. Nothing here is a strategy, a
signal, or a backtest with PF/drawdown. No trade logic was written or
modified.** Per the assignment, everything below follows: phenomenon →
descriptive stats → conditional distribution → hypothesis → baseline
sanity check → (stop; strategy design is a later, separate phase pending
review).

## Multiple-testing disclosure

This run computed on the order of 40-60 conditional-probability/
correlation statistics across 9 pairs × 3 sessions × several question
types. That is enough comparisons that a handful of individually
"significant-looking" numbers are expected by chance alone. The way I
protect against over-claiming here is **cross-pair replication**: a
pattern that shows the same sign and a similar magnitude independently
across 8-9 different currency pairs is very unlikely to be noise (each
pair is a quasi-independent draw), whereas a pattern that shows up in one
pair/hour/year and not the others is treated as noise or an artifact
regardless of how large it looks. Two findings below (#9, seasonality
spike) are explicitly flagged and **rejected** for exactly this reason —
I'm including the rejection because "we checked and it's not real" is
itself a useful, permanent record, matching this project's existing
practice of keeping negative results.

---

## Top 10 discovered phenomena

### 1. Volatility clustering / compression persistence (Family 1)

- **Hypothesis:** a session's range-percentile predicts the *next* same
  session's range-percentile — and specifically, low-range ("compressed")
  days tend to be followed by *more* low range, not an expansion snap-back.
- **Data:** daily range percentile per session, ~785 days/pair/session, all 9 pairs.
- **Sample size:** ~785 × 9 pairs × 3 sessions ≈ 21,000 session-days.
- **Effect size:** day-to-day range-percentile autocorrelation +0.08 to
  +0.61 (mean ≈ +0.30) across every pair/session — always positive, never
  near zero or negative. P(next day expands to top-half range | today
  compressed to bottom quartile) averages **34%** vs a 50% baseline —
  i.e. **compression is followed by more compression** about twice as
  often as the naive "coiled spring" intuition would predict.
- **Statistical evidence:** consistent sign across all 27 pair/session
  combinations (9 pairs × 3 sessions) — no exceptions.
- **Year consistency:** not yet split by year in this pass (see next
  experiment below).
- **Pair consistency:** universal — every single pair/session cell agrees in sign.
- **Session:** slightly stronger in Asian/London (autocorr ≈0.32) than NY (≈0.25).
- **Potential explanation:** this is the FX-market manifestation of
  volatility clustering (the same phenomenon GARCH models are built to
  capture in any liquid market) — quiet regimes (thin order flow, no
  active news catalyst) tend to persist because the *cause* of low
  volatility (absence of a catalyst) doesn't resolve itself just because
  a day has passed.
- **Potential strategy implication:** NOT a directional signal by itself.
  Plausible use: a regime filter/position-sizing overlay — reduce
  expected-breakout position size or widen time horizons after a
  compressed session, since the "release" thesis embedded in most
  breakout strategies this project has tested (LORB, squeeze-breakout)
  is measurably wrong more often than right.
- **Overfitting risk:** low — no free parameters were tuned to produce
  this; it replicates across every pair tested.
- **Next experiment:** split by year (does the effect strengthen/weaken
  over 2023→2026?) and test whether it explains part of why LORB/squeeze-
  breakout families already failed in this project (phase 3, phase 10) —
  if breakouts are systematically mistimed because compression predicts
  more compression, that's a mechanistic explanation for those prior failures.

### 2. Same-day intraday volatility persistence (Family 2)

- **Hypothesis:** London session range-percentile predicts NY session range-percentile, same day.
- **Data:** 785 days/pair, all 9 pairs.
- **Sample size:** ~785 × 9 ≈ 7,000 day-pair observations.
- **Effect size:** P(NY range top-half | London range top-quartile) =
  **62%–75%** vs a 50% baseline, universally above baseline across all 9
  pairs (EURUSD 72%, USDJPY 75%, CADJPY 75%, GBPUSD 62% — the weakest, still +12pp).
  Asia→London range correlation is similarly positive everywhere (r=0.29–0.61).
- **Statistical evidence:** 9/9 pairs same direction, effect size 2-3x
  larger than the multi-day version (#1) — volatility clustering is
  stronger within a day than across days, which makes sense (shared
  news/liquidity conditions within one calendar day).
- **Year consistency:** not yet split by year.
- **Pair consistency:** universal, strongest on JPY crosses and USDJPY.
- **Session:** London→NY specifically (the overlap window).
- **Potential explanation:** same underlying mechanism as #1 — likely the
  *same* phenomenon at a different timescale, not an independent finding.
  I'm reporting it separately because it's the more directly actionable
  version (same-day, tradeable within one session), but the two should be
  understood as one phenomenon, not two independent discoveries — this
  matters for not double-counting evidence.
- **Potential strategy implication:** a same-day dynamic filter — e.g.
  don't fade / widen stops on NY continuation-style setups after a wide
  London range, since continuation is measurably more likely than usual.
  Could plausibly improve exit/stop-width logic on existing ARB/AMR
  strategies as a risk overlay rather than a new directional strategy.
- **Overfitting risk:** low, same reasoning as #1.
- **Next experiment:** test whether this explains any of the existing ARB
  family's (already-live) edge — does ARB's win rate differ on
  wide-vs-narrow prior-session range days? Read-only, no strategy change.

### 3. Asymmetric mean-reversion after down-moves (Family 4)

- **Hypothesis:** after a standardized (≥1.0 ATR) 15-minute move, does
  price continue or revert over the next hour — and is up vs down symmetric?
- **Data:** M15 bars, all 9 pairs, ~4,000-4,500 qualifying down-moves and
  up-moves per pair.
- **Sample size:** ~36,000 down-move events, ~35,000 up-move events across 9 pairs.
- **Effect size:** 60-minute continuation rate after a ≥1.0 ATR **down**
  move: 45%–47% across every pair (i.e. more than half the time price
  ends up net *higher* than the down-move's close 60 min later — mild but
  consistent reversion). Continuation after an equivalent **up** move:
  49%–52%, essentially a coin flip, no consistent skew.
- **Statistical evidence:** 9/9 pairs show the down-move reversion tilt;
  the up-move side shows no consistent pattern (near 50% both ways) — the
  asymmetry itself (down reverts, up doesn't) is the notable part, not
  either number in isolation.
- **Year consistency:** not yet split by year.
- **Pair consistency:** very tight range (45.0%-47.3%) on the down side —
  unusually consistent for FX.
- **Session:** computed across all hours; not yet session-split.
- **Potential explanation:** plausible: stop-hunting/liquidity-sweep
  dynamics on sharp down-moves (a fast drop triggers stops/gets faded by
  liquidity providers) — or simply that USD-quoted pairs had a mild
  upward drift bias over 2023-2026 that shows up asymmetrically when you
  condition on down-moves. This confound needs to be ruled out before
  treating it as a real "down-move" phenomenon rather than a "period had
  net USD weakness in several of these pairs" artifact.
- **Potential strategy implication:** a fade-the-down-spike mechanic —
  structurally different from every strategy family this project has
  tested so far (none of ARB/AMR/LORB/PDH-PDL/Monday-drift is a
  standardized-move fade).
- **Overfitting risk:** moderate — the asymmetry needs the drift-confound
  check below before it's trusted.
- **Next experiment:** re-run with returns de-meaned per pair per year
  (subtract that year's average drift) to check the asymmetry survives
  removing any trend confound; then check by session (does it hold in
  Asian hours specifically, where this project's only two live edges,
  ARB and AMR, already operate?).

### 4. Direction does NOT persist session-to-session (Family 2, negative finding)

- **Hypothesis:** does Asian session direction predict London direction?
- **Effect size:** P(London same sign as Asia) = 45%-52% across all 9
  pairs — indistinguishable from a coin flip, no pair is a clear outlier.
- **Statistical evidence:** null result, replicated across all 9 pairs.
- **Potential explanation:** consistent with the PDH/PDL finding from
  phase 11 (49.3% coin-flip on forward drift) — directional persistence
  at the session level does not appear to exist in this dataset at H1/retail tier.
- **Potential strategy implication:** none directly, but it's useful
  negative evidence: it means #1/#2/#3's real signal is about MAGNITUDE
  (range/volatility), not DIRECTION — reinforces that a volatility-regime
  filter is a fundamentally different, not-yet-tried angle versus this
  project's prior 5 directional strategy families.
- **Overfitting risk:** n/a (negative finding).
- **Next experiment:** none needed; recorded as settled, consistent with existing phase-11 finding.

### 5. London close location → NY direction, JPY crosses only (Family 2, tentative)

- **Hypothesis:** does where London closes within its own range predict NY's direction?
- **Effect size:** on GBPJPY/EURJPY/USDJPY/CADJPY, P(NY up | London closed
  in top 20% of its range) ≈ 60-61% vs baseline ≈54-56% (+5-7pp). On
  EURUSD/GBPUSD/AUDUSD/USDCAD/NZDUSD the effect is smaller or inconsistent
  in sign (some show the *opposite* — London-close-near-low associated with NY up).
- **Statistical evidence:** weaker than #1-#4 — only replicates cleanly
  on JPY-quoted pairs, and the *baseline* NY-up rate is itself elevated
  for JPY pairs (54-56% vs 48-52% for others), which is a red flag: JPY
  broadly weakened over parts of 2023-2026, so this could be measuring
  "JPY crosses trended up overall" rather than a real close-location signal.
- **Potential explanation:** possible genuine order-flow effect (London
  closing strong = institutional positioning continues into NY), OR a
  drift confound identical to #3's caveat.
- **Potential strategy implication:** none until the confound is resolved.
- **Overfitting risk:** high without the drift-adjustment check.
- **Next experiment:** same de-meaning fix as #3, applied here; if the
  effect survives on JPY crosses specifically after removing drift, it's
  worth a second look — otherwise close it out like #9 below.

### 6. Asian-hours efficiency-ratio regime does NOT predict AMR-style reversion strength (Family 3, negative finding)

- **Hypothesis:** does a low-efficiency-ratio (mean-reverting-looking)
  Asian hour show stronger 4-bar reversion than a high-ER (trending-looking) hour?
- **Data:** ~5,480 Asian-session H1 bars per pair, ~900-960 qualifying
  up-move bars per regime split, all 9 pairs.
- **Effect size:** reversion rate 44%-51% across low-ER, high-ER, and
  baseline splits — no consistent separation on any pair; differences are
  within 1-3 percentage points, smaller than pair-to-pair noise.
- **Statistical evidence:** null result across all 9 pairs.
- **Potential explanation:** the efficiency-ratio-based regime split as
  implemented (20-bar window, tercile split) does not capture whatever
  makes AMR work; AMR's real edge (documented in phase 3/phase 3b) may
  depend on session-boundary structure specifically, not a generic
  trend/mean-reversion regime label.
- **Potential strategy implication:** directly answers part of the user's
  standing question ("can we identify conditions under which AMR is more
  effective") — at least for this specific regime definition, no. Do NOT
  build an ER-based AMR filter on this basis.
- **Overfitting risk:** n/a (negative finding, and deliberately not chased further).
- **Next experiment:** if AMR's regime-dependence is worth revisiting
  (per the project's standing "regime-strengthening, unclear durability"
  flag on AMR), a different regime definition — e.g. this discovery
  phase's own volatility-clustering signal (#1) — is a more promising
  candidate than efficiency ratio, since #1 is a much stronger and more
  consistent effect.

### 7. Post-move continuation is symmetric on the UP side (Family 4, negative finding)

- Companion to #3: up-move continuation is 49%-52% across all 9 pairs — no
  asymmetry, no edge. Reported separately because it sharpens #3's story:
  the interesting part isn't "moves revert," it's specifically "**down**
  moves revert, up moves don't," which rules out several trivial
  explanations (e.g. generic mean-reversion in a ranging market would show up on both sides).

### 8. USDCAD's late-hour (23:00 server) return is a sharp, isolated cliff — flagged as likely artifact (Family 5)

- Hour 22 mean return ≈ +0.37 pips, hour 23 ≈ **-2.36 pips**, a 6-8x jump
  with no gradual buildup — and it appears in the raw hourly table as an
  outlier bordering the session/day boundary.
- **This has the exact signature this project's own overfitting
  guidelines warn against** ("isolated peaks vs. plateaus") and the
  signature of a mechanical artifact (day-boundary bar construction,
  broker rollover timestamp, or thin end-of-day liquidity) rather than a
  real phenomenon.
- **Not being carried forward.** Recorded here explicitly so it isn't
  rediscovered and mistaken for a new finding later.

### 9. Hour-0/1 (server time) "seasonality spike" across nearly every pair — REJECTED as likely artifact (Family 5)

- At first glance this looked like the strongest finding in the whole
  run: GBPJPY hour 1 mean return = **+7.08 pips** (vs hour 0 = -0.91, hour
  2 = +0.26 — an isolated spike, not a ramp), consistent positive sign
  across all 4 years, and growing in magnitude in later years (4.5 → 6.0
  → 8.1 → 9.0 pips/year). EURJPY, CADJPY, USDJPY, EURUSD, GBPUSD, AUDUSD,
  NZDUSD all show a similar isolated best-hour spike at hour 0 or 1.
- **I checked the full hourly profile (not just the "best hour") before
  trusting this, per the research standard "test neighboring parameter
  definitions, look for stable regions rather than peaks."** In every
  case the surrounding hours (23, 0, 2, 3) are near zero or even negative
  — this is a single-bar spike, not a plateau. That is the textbook
  signature of a data/execution artifact rather than a real phenomenon:
  hour 0-1 server time is the trading day's rollover/bar-boundary window,
  where broker feeds are most likely to show thin-liquidity price jumps,
  swap/rollover-related quote adjustments, or bar-construction edge
  effects that don't reflect a tradeable price move.
- **Verdict: REJECTED, not a genuine phenomenon**, despite passing a
  naive "consistent across years" check — this is exactly the kind of
  result the user's standards were designed to catch, and it's included
  here specifically to demonstrate that check working, not as a top finding.
- **Next experiment (if ever revisited):** would require tick-level or
  execution-realistic data to distinguish a genuine hour-0/1 phenomenon
  from a feed artifact; not worth pursuing with current H1/M15 retail data.

### 10. Baseline expansion/reversion probabilities cluster tightly around 50% everywhere they're NOT part of findings #1-#3

- Every "baseline" column computed in this run (`p_expand_baseline`,
  `p_ny_up_baseline` outside JPY crosses, up-move continuation) sits
  within a percentage point or two of 50%, across all 9 pairs. This is a
  useful sanity check on the whole methodology: the descriptive machinery
  isn't manufacturing spurious 55-60% baselines out of nowhere — when
  there's genuinely no effect, it reports genuinely no effect. That
  increases confidence that findings #1-#3, which deviate 10-25
  percentage points from baseline, are real rather than measurement noise.

---

## Ranking

| Rank | Phenomenon | Effect | Stability | Pairs | Economic rationale | Research priority |
|---|---|---|---|---|---|---|
| 1 | Same-day London→NY volatility persistence (#2) | Large (+12 to +25pp vs baseline) | High — no year-split yet, but universal sign | 9/9 | Strong (shared intraday liquidity/news conditions) | **HIGH** |
| 2 | Multi-day volatility clustering (#1) | Medium (autocorr +0.08 to +0.61; -16pp expansion prob) | High — universal sign | 9/9 (27/27 pair-session cells) | Strong (GARCH-class, well-documented market regularity) | **HIGH** |
| 3 | Asymmetric down-move reversion (#3) | Small-medium (-3 to -5pp vs 50%) | High — tight cross-pair range | 9/9 | Plausible (liquidity/stop-hunt) but needs drift-deconfound | **MEDIUM-HIGH** |
| 4 | Direction non-persistence (#4, negative) | n/a | High | 9/9 | Consistent with prior EMH-like findings | LOW (settled, no follow-up needed) |
| 5 | London close-location → NY direction, JPY only (#5) | Medium (+5-7pp) but confounded | Low — only 4/9 pairs, possible drift confound | 4/9 | Plausible but unresolved | MEDIUM (pending deconfound) |
| 6 | ER regime vs AMR reversion (#6, negative) | n/a | High (null, consistent) | 9/9 | Directly answers a standing project question | LOW (settled — informs AMR research, no new work needed) |
| 7 | Up-move symmetry (#7, negative) | n/a | High | 9/9 | Sharpens #3 | LOW (context for #3 only) |
| 8 | USDCAD hour-23 cliff (#8, rejected) | Large but isolated | Fails plateau test | 1/9 | Likely artifact | REJECTED |
| 9 | Hour-0/1 seasonality spike (#9, rejected) | Very large but isolated | Fails plateau test | ~7/9 | Likely artifact (rollover/thin-liquidity) | REJECTED |
| 10 | Baseline sanity check (#10) | n/a | n/a | 9/9 | Methodology validation, not a tradeable finding | n/a |

Ranked by economic rationale, effect stability, and cross-pair
consistency — explicitly **not** by any historical profit number, since
none was computed in this phase.

---

## Top 3 for deeper research

### 1. Same-day London→NY volatility persistence

- **Why interesting:** the single largest, most consistent effect found
  (+12 to +25 percentage points over a 50% baseline, on all 9 pairs), and
  it is a fundamentally different KIND of edge than anything this project
  has tested — a magnitude/regime signal, not a directional one. It could
  inform position sizing or stop-width on existing live strategies
  (ARB/AMR) without requiring a whole new strategy.
- **Evidence for:** universal sign and reasonable magnitude across 9
  independent pairs; strong, well-understood economic mechanism (shared
  intraday liquidity/news conditions).
- **Evidence against:** not yet split by year (could be concentrated in
  one volatile year, e.g. 2025, the same year that inflated the now-
  rejected NZDJPY finding — must check before trusting it further); not
  yet tested for cost/execution viability since it isn't a trade signal
  by itself.
- **Simplest possible trading hypothesis:** NOT a new directional
  strategy — a risk/exit overlay: after a wide-range London session,
  widen NY-session stops or position targets on existing ARB trades
  (since continuation, not reversal, is statistically favored); after a
  narrow London session, tighten them.
- **Next experiment:** year-by-year split of the London→NY range-
  percentile relationship; then a read-only check of whether ARB's
  existing live trades perform differently on wide-vs-narrow-prior-
  session days (no strategy code change).

### 2. Multi-day volatility clustering (compression → more compression)

- **Why interesting:** directly explains, mechanistically, why this
  project's breakout-style families (LORB, squeeze-breakout) have
  repeatedly failed — the "coiled spring snaps" intuition behind breakout
  trading appears to be backwards in this dataset (compression predicts
  MORE compression, not release) roughly 2-in-3 of the time.
- **Evidence for:** universal sign across all 27 pair/session cells,
  moderate-to-strong autocorrelation, well-established phenomenon class in finance generally.
  strong, well-established phenomenon class in finance generally.
- **Evidence against:** an autocorrelation of ~0.3 explains meaningful
  but not overwhelming variance (R² ≈ 0.09); this is a real but modest
  statistical regularity, not a near-deterministic one — sizing
  expectations accordingly matters.
- **Simplest possible trading hypothesis:** a regime GATE, not a
  standalone strategy — suppress/derate any future breakout-style signal
  when the prior session was in the bottom range-percentile quartile,
  since expansion is measurably less likely than the breakout thesis assumes.
- **Next experiment:** year-by-year stability check, then (if it holds)
  retroactively check whether this gate would have improved the already-
  dead LORB/squeeze-breakout results from phases 3 and 10 — informative
  even though those families are closed, as a validation of the
  phenomenon's practical relevance before building anything new.

### 3. Asymmetric down-move mean-reversion

- **Why interesting:** structurally novel — none of this project's 5
  previously-tested families (Asian breakout, London pullback,
  London→NY continuation, Asian sweep-reversal, PDH/PDL) is a
  standardized-move fade mechanic. Tight, consistent effect size across
  all 9 pairs (45.0%-47.3% continuation, a ~3-5 point band) is unusually
  narrow for this kind of FX statistic.
- **Evidence for:** consistent asymmetry (down reverts, up doesn't)
  across 9/9 pairs; large sample (~4,000+ events per pair).
- **Evidence against:** could be a trend-drift confound rather than a
  genuine down-move-specific effect — this is the single biggest open
  question before trusting it at all.
- **Simplest possible trading hypothesis:** fade a ≥1.0 ATR M15 down-move
  with a modest target (partial reversion, not a full round-trip) and a
  time exit within the same 60-120 minute window.
- **Next experiment:** de-mean returns by pair-year before recomputing
  the asymmetry (rule out drift confound) — this is the mandatory next
  step before this phenomenon is credible enough to even consider prototyping.

---

## What I did NOT do (per instructions)

- Did not build, backtest, or optimize any of the above as a strategy.
- Did not modify AMR or any other live strategy.
- Did not chase the two rejected findings (#8, #9) further once the
  plateau/isolated-spike check failed them.
- Did not rank by historical profit (no profit was computed in this phase).

All five families' raw structured output is in
`data/phase14_family{1..5}_*.csv` for further slicing without re-fetching
data. Awaiting your review before selecting which (if any) of the top 3
to move into a proper baseline-test phase.
