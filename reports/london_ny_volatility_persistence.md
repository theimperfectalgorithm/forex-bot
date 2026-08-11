# London → NY Volatility Persistence — Regime Research

**Experiments:** EXP-060 through EXP-069, `experiments/experiments.csv`.
**Scripts:** `src/phase19_london_ny_volatility_persistence.py`,
`src/phase19b_existing_strategy_regime_check.py`. **Full logs:**
`reports/phase19_london_ny_log.txt`, `reports/phase19b_regime_check_log.txt`.

**This is volatility-regime research, not a trading strategy.** No entry,
exit, stop, or target was built or optimized. No existing strategy
(ARB/AMR/Monday-drift/XAUUSD ARB) was modified, and the demo account was
not touched anywhere in this work.

## 0. A methodology bug found and fixed while reproducing Part 1

Before anything else: reproducing the original finding surfaced a real
issue in **my own reproduction code**, not the underlying market data —
worth reporting with the same transparency this project has applied to
prior data-integrity findings (see `reports/data_integrity_audit.md`).

Discovery Phase 1's `add_session_col()` assigned session labels via
**sequential boolean-mask overwrites** on the same array (ASIAN, then
LONDON, then NY, in that order). Because Python executes assignments in
order, hours 12-15 — nominally inside both `LONDON=[7,16)` and
`NY=[12,21)` — were claimed LAST by the NY assignment and silently
overwritten out of LONDON. The original finding's **effective** London
window was therefore disjoint: `[7,12)`, not `[7,16)`.

My first reproduction attempt used the two windows as independent,
overlapping masks (`[7,16)` and `[12,21)` in parallel) and got
**79-86%** — visibly stronger than the claimed 62-75%, because 4 of the
9 "NY" hours were also being counted inside "London," mechanically
inflating the correlation with itself. Recomputing with the correct,
disjoint `LONDON=[7,12)` / `NY=[12,21)` definition reproduced **62.4%-
74.6%**, matching the original range exactly. All results below use the
corrected, disjoint definition, consistent with what the original
finding actually measured.

## 1. Original finding reproduction (Part 1)

| pair | n days | n top-quartile London | P(NY top-half \| London top-Q) | baseline |
|---|---|---|---|---|
| EURUSD | 785 | 197 | 71.6% | 50.1% |
| GBPUSD | 785 | 197 | 62.4% | 50.1% |
| USDJPY | 785 | 197 | 74.6% | 50.1% |
| AUDUSD | 785 | 198 | 66.2% | 50.1% |
| USDCAD | 785 | 198 | 66.7% | 50.1% |
| NZDUSD | 785 | 196 | 63.3% | 50.1% |
| GBPJPY | 785 | 197 | 73.1% | 50.1% |
| EURJPY | 785 | 197 | 72.6% | 50.1% |
| CADJPY | 785 | 197 | 74.6% | 49.9% |

**Reproduced: 62.4%-74.6%, matching the original 62-75% claim.**

## 2. Full conditional distribution (Part 2)

| London range bin | n | mean NY range | mean NY pctile | P(NY top-half) | P(NY top-quartile) |
|---|---|---|---|---|---|
| 0-20% | 1,400 | 51.4 | 0.358 | 30.0% | 13.0% |
| 20-40% | 1,419 | 58.6 | 0.451 | 42.1% | 18.4% |
| 40-60% | 1,413 | 61.6 | 0.497 | 49.2% | 21.8% |
| 60-80% | 1,413 | 66.8 | 0.543 | 57.3% | 28.3% |
| 80-100% | 1,420 | 84.6 | 0.651 | 71.4% | 43.3% |

**Monotonic increasing across all 5 bins.** This is a genuine smooth
dose-response relationship — London volatility ↑ → NY volatility ↑ — not
an artifact of one arbitrary threshold. P(NY top-quartile) more than
triples from the bottom to top London bin (13.0% → 43.3%).

## 3. Earliest useful prediction point (Part 3)

| checkpoint | hours elapsed | corr with NY | P(NY top-half \| checkpoint top-Q) |
|---|---|---|---|
| 25% of London | 1 | 0.324 | 67.3% |
| 50% of London | 2 | 0.317 | 66.5% |
| 75% of London | 4 | 0.336 | 69.6% |
| 100% (full) | 5 | 0.340 | 69.4% |

**The relationship is nearly fully formed after just the first hour of
London** (corr 0.324 at 1 hour vs. 0.340 at the full 5-hour session) —
waiting for the complete London session adds almost nothing beyond what
the first checkpoint already shows. This is a genuinely early and
actionable timing property, if the signal is used at all.

## 4. Predictive value vs. simple persistence baselines (Part 4)

| predictor | pooled correlation with NY range percentile |
|---|---|
| **Recent H1 ATR percentile at NY session start** | **0.453** |
| London session range percentile | 0.340 |
| Previous full day's range percentile | 0.281 |
| Asian session range percentile | 0.274 |

**This is the most important, and most tempering, finding in this
phase.** A simple, already-established, generic measure — the
rolling-window ATR percentile this project already uses everywhere,
evaluated right at NY's own start — predicts NY volatility **better**
than the specific London-session-range construct (0.453 vs. 0.340). The
question Part 4 asked directly ("does London contain information beyond
what a simple recent-volatility measure already provides?") has an
honest answer: **not clearly.** London range is a real, valid, but
somewhat *coarser* summary of the same underlying volatility-persistence
phenomenon (Discovery Phase 1's finding #1) that a continuously-updated
ATR percentile already captures more efficiently. London does still beat
the Asian-session and previous-day baselines, so it isn't redundant with
*those* — but it is not shown to add value over the simplest available
generic persistence measure.

## 5. News confound (Part 5) — limited by data availability

**Limitation, stated plainly:** this project's `core/news_calendar.py`
only caches the *current week's* ForexFactory feed — there is no
reliable historical (2023-2026) economic calendar in this repo. Rather
than fabricate historical news data, this used a deterministic proxy for
one major recurring event: the first Friday of each month (US Non-Farm
Payrolls). With the corrected (smaller, disjoint) London window, too few
top-quartile-London days coincided with NFP-proxy days to report a
reliable split for most pairs (fewer than 10 qualifying days per pair)
— **the data does not support a robust news-confound conclusion**, and
none is claimed. The only reliably-sized group (non-NFP days) shows
P(NY top-half | London top-Q) of 62-75% per pair — consistent with the
full-sample result, meaning the relationship is not obviously an
NFP-day-only artifact, but this is a weak, partial check, not a
complete news-confound test.

## 6. NY intrasession timing (Part 6)

| NY quarter | P(quarter top-half \| London top-Q) |
|---|---|
| Q1 (12-14) | 76.5% |
| Q2 (14-16) | 69.2% |
| Q3 (16-18) | 68.3% |
| Q4 (18-21) | 68.7% |

**The elevated-volatility state persists through the entire NY session**,
not just an early spillover that dies immediately — strongest right at
the London-NY handoff (Q1) but still meaningfully elevated (68-69% vs.
a ~50% baseline) all the way through the session's last quarter. This
matters for the earlier stated caveat: a purely front-loaded effect
would be far less useful for anything happening later in NY; this one
is not front-loaded only.

## 7. Directional independence (Part 7)

| | P(NY same direction as London) |
|---|---|
| Pooled across 9 pairs | **50.48%** |

**Clean null, as expected.** London range says nothing about NY
direction — the volatility relationship (Parts 1-2, 6) was tested with
zero shared logic against direction, and no directional filter was
combined with the volatility study at any point.

## 8. Pair consistency (Part 8)

Every one of the 9 pairs shows P(NY top-half | London top-Q) above the
~50% baseline in the pooled reproduction (Part 1: 62.4%-74.6%). Pooled
correlation (Part 4) ranges 0.24 (NZDUSD) to 0.46 (USDJPY) — broadly
cross-pair, with JPY crosses and USDJPY somewhat stronger than the
non-JPY majors, but no pair shows a null or reversed relationship at the
full-sample level.

## 9. Year consistency (Part 9)

| pair | 2023 | 2024 | 2025 | 2026 YTD |
|---|---|---|---|---|
| AUDUSD | 76.9% | 60.0% | 53.0% | 81.3% |
| CADJPY | 82.4% | 76.6% | 81.8% | **35.0%** |
| EURJPY | 76.9% | 80.7% | 70.3% | **45.8%** |
| EURUSD | 88.2% | 58.3% | 76.8% | 67.6% |
| GBPJPY | 57.1% | 78.7% | 76.7% | 59.3% |
| GBPUSD | 61.5% | 50.0% | 70.6% | 61.1% |
| NZDUSD | 76.5% | 56.1% | 58.9% | 67.4% |
| USDCAD | 76.5% | 68.0% | 65.1% | 58.1% |
| USDJPY | 64.7% | 80.9% | 73.6% | 57.9% |

**Broadly persistent, but genuinely noisier at the pair-year level than
the pooled result suggests — flagged honestly, not smoothed over.** Two
cells (CADJPY and EURJPY, both 2026 YTD) actually fall *below* the 50%
baseline. 2026 YTD covers roughly 7 months and has the smallest sample
of the four years (~50 top-quartile-London days per pair), so this is
plausibly a small-sample effect rather than a genuine regime break, but
it cannot be ruled out with the current data, and it means the relationship
should not be treated as uniformly reliable at the individual pair-year level.

## 10. Session-boundary robustness (Part 12)

| definition | P(NY top-half \| London top-Q) |
|---|---|
| Original (L=7-12, NY=12-21) | 69.4% |
| Shift earlier (L=6-11, NY=11-20) | 70.0% |
| Shift later (L=8-13, NY=13-22) | 68.6% |

**Stable across all 3 pre-specified neighboring boundary definitions**
(68.6%-70.0%, a 1.4 percentage-point range) — the relationship does not
depend on the exact hour chosen for the session split.

## 11. Null / randomization test (Part 11)

**Methodology:** within each (pair, year) group, the *pairing* between a
given London session and its same-day NY session was permuted 1,000
times, while preserving each session's own marginal range/percentile
distribution and the pair/year grouping — only the same-day linkage was
shuffled (not the raw values, which would have destroyed the already-
established volatility-persistence structure and invalidated the test).

| pair | observed P | null mean | null std | percentile |
|---|---|---|---|---|
| All 9 pairs | 62.4%-74.6% | ~51-54% | ~0.03 | **1.0000 (all 9)** |

**Every single pair's observed relationship beat all 1,000 shuffles** —
the same-day linkage between London and NY carries real information
beyond what each session's own independent volatility distribution would
produce by chance. This is about as decisive as this kind of test gets.

## 12. Existing-strategy observational analysis (Part 14)

**Critical timing caveat, stated first:** ARB enters at server hours 7-8
(the very start of London — before London's own range is known), AMR
enters at server hours 0-4 (before London begins at all), and Monday
Drift enters at server hour 0 Monday (also before London). **None of
these 8 strategies' entry decisions could use "completed London session
range" as an input — that information does not exist yet at their entry
time.** What follows is a purely retrospective/observational correlation
between each strategy's already-completed trade P&L and that day's
later-revealed regime label. It is not a usable real-time filter and is
not being proposed as one.

| Strategy | n topQ-London days | mean P&L (topQ) | mean P&L (other) | win rate (topQ) | win rate (other) |
|---|---|---|---|---|---|
| GBPJPY ARB | 68 | +76.96 | +117.81 | 47.1% | 49.6% |
| CADJPY ARB | 66 | -15.20 | +112.55 | 43.9% | 51.6% |
| XAUUSD ARB | 62 | **+114.40** | +62.23 | **58.1%** | 47.9% |
| GBPJPY AMR | 103 | +12.17 | +47.03 | 61.2% | 69.2% |
| EURJPY AMR | 181 | -11.67 | +23.16 | 61.9% | 71.4% |
| AUDJPY AMR | 173 | -22.95 | +25.24 | 60.7% | 72.8% |
| CADJPY AMR | 164 | -18.75 | +16.43 | 61.0% | 71.1% |
| GBPUSD Monday Drift | 39 | **+53.10** | +26.78 | 61.5% | 63.2% |

**A striking, consistent pattern across the mean-reversion family:** all
4 AMR variants perform meaningfully worse — lower mean P&L *and* lower
win rate — on days when that same day's London session later turns out
to be top-quartile-range. 2 of 3 ARB strategies (GBPJPY, CADJPY) show
the same pattern; XAUUSD ARB and GBPUSD Monday Drift are the exceptions,
both performing *better* on those days.

This is genuinely interesting circumstantial context for AMR's own
already-documented open question (project memory: AMR's edge is
"real but regime-strengthening, unclear durability") — a day that turns
out to be a broad high-volatility/trending day appears to coincide with
AMR's Asian-hours mean-reversion setups performing worse, which is
economically sensible for a mean-reversion strategy. **This is
documented for the record only. No filter was created, no AMR logic was
read for modification purposes, and nothing here is implemented or
acted upon**, per the explicit instruction that this analysis is
observational only.

## 13. Strongest evidence FOR

- The null/randomization test result (Part 11) — every one of 9 pairs
  beats 1,000 shuffles decisively (100th percentile every time).
- A genuinely smooth, monotonic dose-response relationship (Part 2), not
  a threshold artifact.
- Most of the predictive content is available after just the first hour
  of London (Part 3) — an early, not late-arriving, signal.
- The elevated state persists through the entire NY session, not just
  an early spillover (Part 6).
- Robust to reasonable session-boundary shifts (Part 12) and clean on
  the directional-independence check (Part 7).

## 14. Strongest evidence AGAINST

- **A simpler, already-available, generic volatility measure (recent
  ATR percentile) predicts NY volatility better than London's specific
  session-range construct** (Part 4) — the central complication for
  calling this a genuinely new, incremental piece of information rather
  than a coarser restatement of already-known volatility persistence.
- Pair-year-level consistency is real but noisier than the pooled
  numbers suggest — 2 of 36 pair-year cells fall below the 50% baseline
  entirely (Part 9), both in the smallest-sample year.
- The news-confound check (Part 5) could not be completed reliably with
  available data — an unresolved gap, not a clean pass.
- My own initial reproduction attempt got this wrong before the
  session-boundary issue was found and corrected (Part 0) — a reminder
  that even a "reproduce the original exactly" step needs active
  verification, not just re-running old code.

## 15. Final classification

# **B. PREDICTIVE VOLATILITY SIGNAL**

Not (A) descriptive only — the null test (Part 11), monotonic
conditional distribution (Part 2), and early-checkpoint availability
(Part 3) together establish this genuinely forecasts something not yet
known, not merely describes a pattern after the fact.

Not (C) strong predictive volatility signal — a simpler, generic,
already-available baseline (recent ATR percentile) outperforms the
specific London construct (Part 4), and pair-year consistency shows real
cracks in the smallest-sample year (Part 9). Calling this "strong" would
overstate what London specifically contributes beyond ordinary
volatility persistence.

Not (D) artifact/rejected — the relationship survives the null test at
the most decisive level this kind of test can show, and is stable across
reasonable session-boundary redefinitions (Part 12). The one artifact
actually found in this phase was in my own initial reproduction code
(Part 0), not in the underlying phenomenon, and was corrected before any
downstream analysis ran.

## 16. Portfolio relevance

# **POTENTIAL REGIME INFORMATION**

Not "no useful relationship" — the observational cross-check (Part 14)
shows a consistent, non-trivial pattern across all 4 AMR variants and 2
of 3 ARB variants (worse performance on later-revealed high-volatility
days), which is circumstantially relevant to AMR's own already-flagged
regime-dependence question.

Not "strong regime information" — none of the 8 existing strategies can
observe the London signal before their own entry decisions (Part 14's
central timing caveat), so however consistent the retrospective
correlation is, it cannot currently inform any of these strategies'
actual trading logic without a fundamentally different entry-timing
design that does not exist today. **This is not being called a trading
edge.**

## 17. Recommended next experiment

1. **Resolve the year-level instability (Part 9)** with a larger sample
   once more of 2026 has accumulated, before treating the pair-year
   relationship as fully settled.
2. **A genuine head-to-head test of London range vs. recent-ATR-
   percentile as competing predictors** (e.g. which one, or whether
   both together, best explains NY range in a proper regression) — Part
   4 only compared them individually; a joint test would clarify whether
   London adds anything incremental once ATR percentile is already
   accounted for. This should be pre-registered as its own experiment,
   not folded ad hoc into this one.
3. If a future research phase explores an actual news calendar with
   historical coverage (not the current-week-only feed this project has
   today), Part 5's news-confound question could be answered properly
   rather than with the first-Friday NFP proxy used here.
4. The AMR regime-dependence observation from Part 14 remains exactly
   what it was called: a documented, non-implemented observation. If the
   project wants to pursue it, that would need to be a fresh,
   separately-scoped experiment — not a filter derived from this report.

---

## What I did NOT do (per instructions)

- Did not build entries, exits, stops, or targets.
- Did not add a directional filter alongside the volatility study.
- Did not modify GBPJPY ARB, CADJPY ARB, XAUUSD ARB, GBPJPY/EURJPY/
  AUDJPY/CADJPY AMR, or GBPUSD Monday Drift.
- Did not create an AMR filter from the Part 14 observation, despite it
  being the most consistent finding in that section.
- Did not change the demo account or any live configuration.
