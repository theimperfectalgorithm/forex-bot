# AMR Regime Mechanism Research — Volatility Cause vs. Trend Proxy

**Experiments:** EXP-076 through EXP-085, `experiments/experiments.csv`.
**Script:** `src/phase21_amr_regime_mechanism.py`. **Full log:** `reports/phase21_mechanism_log.txt`. **Data:** `data/phase21_amr_trades.csv` (2,365 AMR trades, all 4 pairs).

**Mechanism research only. AMR was not modified anywhere in this work —
no filter, no threshold search, no entry/exit/stop/target/risk change.
The 2026-08-25 AMR checkpoint is unaffected; this report supplies input
to it, not a decision.**

## 1. Methodology

All regime/explanatory variables are computed from data strictly at or
before each trade's own entry bar. `run_sim` enters at the signal bar's
own close, so that bar's own state (ATR, efficiency ratio, persistence,
recent range) is legitimate entry-time information — this is the same
no-lookahead convention used since phase 16/17/20, re-verified here, not
re-derived. MFE/MAE (Part 11) is the only place post-entry data is used,
strictly for path/outcome analysis, never for regime classification.

## 2. Entry-time information validation

Reconstructed 2,365 trades (GBPJPY 403, EURJPY 712, AUDJPY 652, CADJPY
598) with, per trade: pair, direction, entry price/time, outcome, R
multiple, ATR percentile, directional persistence (20-bar), efficiency
ratio (20-bar, Kaufman, unchanged from phase 16), ATR-normalized
20-bar and 8-bar returns, distance from recent 20-bar high/low, position
within the 20-bar range, and a purely backward-looking volatility-change
ratio (std of the 4 bars immediately before entry vs. the 4 bars before
that). All windows (20-bar, 8-bar, 4-bar) reuse this project's existing
precedent (phase 16/17), none were searched here.

## 3. Volatility reproduction (Part 2)

| Pair | LOW PF | NORMAL-LOW PF | NORMAL-HIGH PF | HIGH PF |
|---|---|---|---|---|
| GBPJPY | 1.92 | 1.40 | 1.22 | 1.19 |
| EURJPY | 1.04 | 1.35 | 1.22 | 1.01 |
| AUDJPY | 1.35 | 1.33 | 1.00 | 0.85 |
| CADJPY | 1.85 | 1.02 | 1.07 | 0.76 |

**Exact reproduction of phase 20's numbers** — confirms the same
reconstruction pipeline and volatility definition. No changes.

## 4. Trend/persistence analysis and the causal test (Parts 3-4)

The central question: does volatility remain predictive after
conditioning on trend, and does trend remain predictive after
conditioning on volatility? This is the key test for distinguishing
"volatility causes the effect" from "volatility is a proxy for trend."

### AUDJPY — volatility survives conditioning cleanly

| Trend tercile | vol_LOW expectancy | vol_HIGH expectancy | diff |
|---|---|---|---|
| LOW TREND | +17.82 | -8.15 | **-25.97** |
| NORMAL TREND | +12.74 | -17.17 | **-29.91** |
| HIGH TREND | +47.60 | -33.94 | **-81.54** |

**The volatility penalty is negative in all three trend terciles, and
gets stronger, not weaker, as trend rises.** Trend does not explain away
the volatility effect for AUDJPY.

| Vol tercile | trend_LOW expectancy | trend_HIGH expectancy | diff |
|---|---|---|---|
| LOW | +17.82 | +47.60 | +29.78 |
| NORMAL | +56.96 | +25.28 | -31.68 |
| HIGH | -8.15 | -33.94 | -25.79 |

The reverse conditioning (trend within volatility) has **no stable
sign** — it flips from positive to negative across the three volatility
terciles. This asymmetry (volatility: consistently negative in all 3
cuts; trend: sign-flipping across all 3 cuts) is the clearest evidence
in this study that **volatility, not trend, is the more fundamental
variable for AUDJPY.**

### CADJPY — an interaction, not a clean independent effect

| Trend tercile | vol_LOW expectancy | vol_HIGH expectancy | diff |
|---|---|---|---|
| LOW TREND | +20.15 | +21.50 | **+1.36 (no effect)** |
| NORMAL TREND | +43.35 | -50.02 | **-93.37** |
| HIGH TREND | +44.47 | -18.29 | **-62.75** |

**The volatility effect essentially disappears within the LOW TREND
tercile** (near-zero difference) but is large within NORMAL/HIGH TREND —
volatility matters *conditionally on* trend being present, not
independently of it. This is a genuinely different, more nuanced
pattern than AUDJPY's — CADJPY's mechanism looks like an **interaction**
between volatility and trend, not a pure volatility effect.

### GBPJPY — the volatility effect reverses sign under high trend

| Trend tercile | vol_LOW expectancy | vol_HIGH expectancy | diff |
|---|---|---|---|
| LOW TREND | +90.94 | +6.97 | -83.97 |
| NORMAL TREND | +74.56 | -28.87 | -103.43 |
| HIGH TREND | +34.30 | +54.19 | **+19.89 (reversed!)** |

**In the HIGH TREND tercile, high volatility is actually slightly
*better*, not worse** — the pooled "stable" relationship phase 20
reported does not survive this conditioning check. This is an important
revision, made with the same transparency this project applies
elsewhere: the deeper analysis here shows GBPJPY AMR's apparent
volatility dependence is less robust than the pooled 4-bin table alone
suggested.

### EURJPY — also sign-unstable across trend terciles

| Trend tercile | vol_LOW expectancy | vol_HIGH expectancy | diff |
|---|---|---|---|
| LOW TREND | +3.33 | +35.11 | **+31.78 (reversed!)** |
| NORMAL TREND | +14.20 | +19.81 | +5.61 |
| HIGH TREND | +25.90 | -8.92 | -34.82 |

Consistent with phase 20's original "weak/mixed" read — the
conditioning analysis confirms the instability rather than resolving it.

## 5. 2D volatility × trend regime matrix (Part 5)

Full 3×3 matrices for all 4 pairs are in `reports/phase21_mechanism_log.txt`.
The clearest cell-level story is AUDJPY: expectancy is positive in 8 of
9 cells except specifically where volatility is HIGH (all three trend
columns negative: -8.15, -17.17, -33.94) — **AMR fails specifically when
volatility is high, largely independent of which trend tercile
accompanies it.** CADJPY's matrix instead shows its worst cells
concentrated in NORMAL/HIGH-TREND × NORMAL/HIGH-VOL jointly, consistent
with the interaction read above.

## 6. Volatility transition (Part 6) — a genuinely new nuance

**Question: does AMR fail because volatility is already high, or because
it's rapidly expanding into the trade?** Using a purely backward-looking
4-bar-vs-prior-4-bar volatility-change ratio:

| Pair | EXPANDING expectancy | STABLE/CONTRACTING expectancy | Within HIGH-ATR: EXPANDING | Within HIGH-ATR: STABLE |
|---|---|---|---|---|
| GBPJPY | +32.5 | +44.5 | **+27.8** | **-2.0** |
| EURJPY | +16.1 | +12.3 | **+28.3** | **-6.1** |
| AUDJPY | +29.1 | -3.8 | +4.7 | **-47.8** |
| CADJPY | -2.1 | +16.5 | -19.0 | -17.5 (similar) |

**3 of 4 pairs (GBPJPY, EURJPY, AUDJPY) show that, within the
already-high-ATR regime specifically, recently-EXPANDING volatility
performs noticeably better than STABLE/plateaued high volatility** —
the worst sub-condition is not "volatility is rising into the trade,"
it's "volatility has been elevated and stayed elevated." CADJPY is the
exception, showing no such distinction (both sub-groups weak within
HIGH-ATR). This directly answers Part 6: for most pairs, **already-
sustained high volatility, not fresh volatility expansion, is the more
damaging condition.**

## 7. Market location (Part 7) — the most consistent finding in this study

| Pair | NEAR_LOW expectancy | MID_RANGE expectancy | NEAR_HIGH expectancy |
|---|---|---|---|
| GBPJPY | **+54.9** | +50.5 | +11.5 |
| EURJPY | **+36.9** | +27.7 | **-17.4 (losing)** |
| AUDJPY | **+40.4** | +42.0 | **-34.6 (losing)** |
| CADJPY | **+29.0** | +49.6 | **-26.3 (losing)** |

**All 4 pairs — including GBPJPY and EURJPY, which showed weak/unstable
volatility relationships — show the same clean pattern here**: AMR
performs best when price is near the low of its own recent 20-bar range,
and worst (net losing in 3 of 4 pairs) when price is near the recent
high. This is expected structurally, given AMR's z-score construction
(BUY when price is stretched below its SMA, i.e. typically near recent
lows; SELL when stretched above, near recent highs) — but the size of
the gap, and that it holds even for the two pairs where volatility alone
is unstable, makes this the single most consistent finding in the study.

## 8. Directional asymmetry (Part 8) — likely the real underlying structure

| Pair | BUY @ LOW vol PF | SELL @ LOW vol PF | BUY @ HIGH vol PF | SELL @ HIGH vol PF |
|---|---|---|---|---|
| GBPJPY | **2.60** | 0.86 (losing) | 1.08 | 1.18 |
| EURJPY | 1.71 | 0.64 (losing) | 1.41 | 0.88 (losing) |
| AUDJPY | 1.76 | 0.72 (losing) | 1.12 | **0.58 (losing badly)** |
| CADJPY | 1.66 | 1.36 | 1.05 | **0.66 (losing badly)** |

**This is likely the real underlying structure the pooled volatility
tables were partially picking up.** AMR's edge is overwhelmingly carried
by BUY-side (dip-buying) trades, especially in LOW-to-NORMAL volatility
— every pair's BUY-LOW cell is strong (PF 1.66-2.60). SELL-side
(rally-fading) trades are structurally much weaker across the board, and
collapse hardest as volatility rises (AUDJPY and CADJPY SELL-HIGH are
both clearly net-losing, PF 0.58-0.66). This connects directly to the
market-location finding above (BUY trades occur near recent lows, SELL
trades near recent highs, by construction) — **the volatility-regime
degradation is not symmetric across AMR's own two trade types, and the
SELL leg is doing most of the damage in high-volatility conditions.**

## 9. Year consistency (Part 9, using vol terciles)

| Pair | Years confirming (HIGH worse than LOW) | Years contradicting |
|---|---|---|
| AUDJPY | 2024, 2025 (both clear) | 2026 (near-zero both sides, not a real contradiction) |
| CADJPY | 2024, 2025, 2026 (all three, 2026 strongest) | none |
| GBPJPY | 2024, 2025 (LOW beats HIGH both years) | none at the tercile level (differs from phase 20's quartile-based read) |
| EURJPY | 2024 (both negative, HIGH worse) | 2025 (HIGH actually better) |

**Note:** using terciles instead of phase 20's quartiles shifts some
readings — CADJPY's 2026 becomes usable and confirms strongly; GBPJPY's
tercile-level year check looks more consistent than its quartile-level
one did. This is a real methodological sensitivity worth flagging
honestly rather than picking whichever binning looks best: the
core AUDJPY/CADJPY story is robust to this choice, GBPJPY's is
bin-sensitive, and EURJPY remains mixed either way.

## 10. Pair consistency (Part 10)

**The mechanism is not the same across all 4 pairs.** AUDJPY shows a
clean, conditioning-robust, purely volatility-driven effect. CADJPY
shows a real but volatility×trend interaction effect. GBPJPY's
apparent volatility dependence reverses sign under high trend and is
better explained as trend-contingent, not purely volatility-driven.
EURJPY's is unstable under conditioning either way. **Do not generalize
the AUDJPY/CADJPY finding to GBPJPY/EURJPY** — the deeper analysis here
specifically does not support that.

## 11. MFE/MAE path analysis (Part 11)

| Pair | MFE:MAE ratio, LOW vol | MFE:MAE ratio, HIGH vol |
|---|---|---|
| GBPJPY | 1.55 | 1.33 |
| EURJPY | 1.21 | 1.16 |
| AUDJPY | 1.27 | **0.97 (MAE > MFE)** |
| CADJPY | 1.45 | **0.98 (MAE > MFE)** |

For AUDJPY and CADJPY specifically, the favorable/adverse excursion
ratio **crosses below 1.0** in the HIGH volatility regime — trades in
that regime experience more adverse than favorable movement on average,
not just a lower win rate by coincidence. This points toward **regime
failure** (the market genuinely moves against these trades harder in
high volatility) rather than pure **entry-timing failure** (a fine entry
that just doesn't work out) — consistent with, and reinforcing, the
"volatility is real, not just win-rate noise" reading of Part 4.

## 12. Comparison with the previous efficiency-ratio result

Discovery Phase 1's Family 3 test (a single-bar-reversion check within
Asian hours, using a generic ER tercile split unrelated to any specific
strategy's own trades) found no relationship and was explicitly a
different, narrower test. **This phase's evidence is genuinely
different, not a rerun**: here, efficiency ratio is used specifically as
a *conditioning variable against AMR's own actual trade outcomes*, not
as a standalone reversion predictor. The two tests answer different
questions, and this one finds real structure the earlier one could not
have found (an interaction effect for CADJPY, a sign-reversal for
GBPJPY) — this is not a contradiction of the earlier null result, since
that result was never about AMR's own trades to begin with. Where trend
measures again find nothing informative (EURJPY, and GBPJPY's own trend
axis in isolation) that is *itself* useful: it means, for those two
pairs specifically, AMR's regime-dependence question remains genuinely
open rather than answered by either variable tested so far.

## 13. Multiple-testing assessment

This phase tested 4 pairs × multiple regime cuts × 2 conditioning
directions × 3×3 matrices × 2 directions × 4 years — a large number of
cells. No single favorable/unfavorable cell is treated as a finding on
its own. The classifications below rest specifically on: (a) whether the
volatility effect's *sign* is stable across all three trend terciles
(the sharpest test in this report), (b) cross-year replication, and (c)
whether independent variables (market location, directional split, MFE/
MAE) point the same direction. AUDJPY is the only pair that clears all
three; the others are graded accordingly, not elevated on pooled numbers
alone.

## 14. Strongest mechanism

**AUDJPY AMR: volatility is a genuine, conditioning-robust causal
variable**, independently corroborated by directional asymmetry (SELL
side collapses hardest) and path data (MAE exceeds MFE in HIGH regime).

## 15. Evidence against the mechanism (per pair)

- **AUDJPY**: none found against — every independent cut (conditioning,
  years, direction, MFE/MAE) agrees.
- **CADJPY**: the effect is trend-conditional (absent in LOW TREND),
  which complicates a pure "volatility causes it" story even though the
  pooled and year-level evidence is strong.
- **GBPJPY**: the volatility effect reverses sign under HIGH TREND —
  the strongest evidence against treating this as a stable, standalone
  volatility mechanism.
- **EURJPY**: sign-unstable across both conditioning directions; no
  variable tested here provides a coherent explanation.

## 16. Pair-specific conclusions

- **AUDJPY AMR**: volatility-dependent, robust, multi-angle-corroborated.
- **CADJPY AMR**: volatility-and-trend interaction; real, but more
  complex than a simple volatility filter would capture.
- **GBPJPY AMR**: the pooled "stable" relationship from phase 20 does
  not survive conditioning on trend — closer to trend-contingent or
  inconclusive than a genuine volatility mechanism.
- **EURJPY AMR**: inconclusive; no variable tested (volatility or trend)
  provides a stable explanation.

## 17. Overall AMR conclusion

Across all four pairs, the single most consistent structural finding —
more consistent than volatility itself — is **market location /
directional asymmetry**: AMR's BUY (dip-buying) trades are consistently
strong across all four pairs, while SELL (rally-fading) trades are
structurally weaker and specifically responsible for most of the
high-volatility-regime degradation. The volatility-regime story, where
it holds (AUDJPY clearly, CADJPY with an interaction caveat), appears to
be substantially — though perhaps not entirely — a reflection of this
deeper directional asymmetry rather than a wholly separate phenomenon.

## 18. Final classifications (Part 15)

| Pair | Classification |
|---|---|
| AUDJPY AMR | **B. VOLATILITY-DEPENDENT** |
| CADJPY AMR | **D. VOLATILITY + TREND INTERACTION** |
| GBPJPY AMR | **E. OTHER / INCONCLUSIVE** (revises phase 20's "C. STABLE" downward given it fails the trend-conditioning test) |
| EURJPY AMR | **E. OTHER / INCONCLUSIVE** (consistent with phase 20's "B. WEAK/INCONSISTENT") |

## 19. Is a confirmatory filter experiment justified?

**Yes, but only for AUDJPY AMR, and only as a separately-scoped
experiment — not implemented here.** AUDJPY is the only pair where the
volatility mechanism survives every conditioning check applied in this
report. CADJPY's interaction structure means a simple ATR-percentile
filter would likely be mis-specified (it would penalize NORMAL/HIGH-TREND
high-vol trades appropriately but wrongly penalize LOW-TREND high-vol
trades, which showed no effect). GBPJPY and EURJPY do not currently
justify a filter experiment at all — their apparent regime relationships
did not survive this deeper analysis.

## 20. Exact recommended next experiment

1. **A single, pre-registered confirmatory filter experiment on AUDJPY
   AMR only**: freeze one ATR-percentile threshold (using the existing
   HIGH-regime boundary already defined, [75,100), not a newly searched
   one) as an exclusion filter, and test it against the AUDJPY AMR
   baseline with proper IS/OOS discipline — this is the next experiment,
   not something to run now.
2. **Present this report, not a code change, at the 2026-08-25 AMR
   checkpoint** — the checkpoint should weigh AUDJPY's robust finding
   against CADJPY's more complex interaction, GBPJPY's downgraded
   status, and EURJPY's continued inconclusiveness, and decide whether
   any filter work is worth prioritizing at all given the directional
   asymmetry finding (Part 8) may be the more fundamental lever.
3. **If a filter experiment for AUDJPY is undertaken, it should account
   for the directional asymmetry finding** — a plain ATR-percentile
   filter might simply be indirectly filtering out SELL-side trades in
   high volatility, which Part 8 suggests may be the more precise (and
   more defensible) mechanism than volatility alone.

---

## What I did NOT do (per instructions)

- Did not add an ATR filter, trend filter, or any filter to AMR.
- Did not remove trades, modify entries, exits, stops, targets, or risk.
- Did not pause AMR or change the demo account.
- Did not change the 2026-08-25 AMR checkpoint date or process.
- Did not implement the recommended confirmatory filter experiment —
  it is a recommendation for a future, separately-scoped phase.
