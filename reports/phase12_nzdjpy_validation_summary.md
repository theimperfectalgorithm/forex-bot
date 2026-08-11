# Validation Gate Summary — NZDJPY Cross-Asset Momentum (FROZEN)

**Experiments:** EXP-030 (frozen spec record), EXP-031 (final summary) —
full detail in `experiments/experiments.csv`. Raw console log:
`reports/phase12_nzdjpy_validation_report.md`.

**Frozen spec:** BUY/SELL NZDJPY at server-hour 15 when USDJPY's
session move exceeds 1.25× its own ATR(14); SL 1.5×ATR / TP 2.5×ATR
(NZDJPY's own ATR); flat by server-hour 21; 0.5% risk/trade; 2.2 pip
spread; no slippage, no news filter modeled. See EXP-030 for the
complete record. **Nothing below altered this spec.**

## Automated scorecard (mechanical threshold)

| Test | Result | Pass/Fail |
|---|---|---|
| In-sample (24mo) | PF 1.49, DD 5.86%, 84% profitable months | PASS |
| Out-of-sample (12mo, held out) | PF 1.20, +$7,731 | PASS |
| Walk-forward (4 folds) | 3/4 profitable (75%) | PASS |
| Monte Carlo risk of ruin | 0.03% | PASS |
| **Cost stress (2× spread)** | **IS PF 1.19, OOS PF 0.93 (losing)** | **FAIL** |
| Parameter sensitivity | smooth plateau, no isolated spike | PASS |
| **Year consistency** | **73% of total P&L from 2025 alone; 2026 YTD is negative (PF 0.90)** | **FAIL** |
| Direction vs. random (permutation) | beats 100% of 1,000 random-direction shuffles | PASS |
| Drawdown | -6.85% historical | PASS |

**7/9 mechanical passes → the script's own threshold logic says
VALIDATED.**

## Why I am overriding that to PROMISING BUT INSUFFICIENT

You explicitly asked me not to call something VALIDATED simply because
a scorecard tally clears a threshold — it has to demonstrate a robust
edge. Two things the mechanical count under-weights:

1. **The edge's margin over cost is thin.** At 1.5× spread (3.3
   pips — not an exotic scenario, just a somewhat wider real-world
   spread) OOS profit factor is 1.06, barely above breakeven. At 2×
   spread it's already losing OOS. A 1-bar execution delay alone
   — modeling nothing more dramatic than "the live order doesn't fill
   at the exact instant of the signal bar's close" — also flips OOS to
   losing (PF 0.95). Two independent, realistic stresses each
   individually erase the out-of-sample edge.
2. **The most recent complete-ish year is a loser.** 2023: PF 1.08
   (near breakeven). 2024: PF 1.31. 2025: PF 1.77 (73% of all-time
   profit). **2026 year-to-date: PF 0.90, -$2,368.** The edge is not
   spread evenly across time — it's concentrated in one strong year,
   and the most recent stretch of data we have is negative.

These two facts point at the same underlying risk from two different
angles: **the edge may be real but currently fading, or was
regime-dependent to a period that's ending.** That is not a reason to
reject the hypothesis outright — see the evidence in favor, below —
but it is a real reason not to call it robustly validated yet.

## Strongest evidence FOR the hypothesis

- **The permutation test is genuinely strong evidence of structural
  signal, not spread-harvesting.** Same entry bars, same SL/TP
  construction, only the direction randomized: the null model's mean
  outcome was a LOSING strategy (mean PF 0.815, mean P&L -$18,706).
  The real strategy's PF (1.32 over the full period) beat literally
  all 1,000 random-direction shuffles. This means the USDJPY-move
  direction signal itself is doing real work — this is not just "any
  ATR-scaled trade at 15:00 on NZDJPY happens to work."
- **The parameter plateau is clean, not a spike.** Perturbing every
  parameter ±10%/±20% individually — check_hour, threshold, SL
  multiple, TP multiple — produces smoothly varying, consistently
  positive profit factors in every single neighboring cell (IS PF
  never drops below 1.25 across 18 neighbor tests). This is the
  opposite signature of an overfit, isolated peak.
- Walk-forward: 3 of 4 rolling 6-month test windows were profitable,
  including two of the three most recent.
- MFE/MAE sanity-checked correctly (losers' median adverse excursion
  ≈1.04R, exactly what you'd expect if the stop-loss is the genuinely
  binding constraint — no logic error in the simulation).

## Strongest evidence AGAINST

- Cost/execution fragility (above) — the edge doesn't have much room
  to spare against realistic friction.
- Year concentration (above) — most of the historical profit is one
  year, and the newest data is a loser.
- Notably, **this is the same open question already flagged for
  AMR** (this project's other JPY-cross candidate): a real,
  statistically-supported signal that may be regime-dependent rather
  than durable. Two independent findings sharing the identical
  uncertainty profile is itself informative — it may say something
  about JPY-cross edges as a class right now, not just about this one
  candidate.

## Answers to your 7 questions

1. **Does the frozen candidate survive?** Conditionally. It survives
   as a genuine, statistically-supported hypothesis — it does NOT yet
   survive as a ready-to-deploy strategy. Status: **PROMISING BUT
   INSUFFICIENT** (overriding the mechanical scorecard's VALIDATED for
   the reasons above).
2. **Strongest evidence for:** the permutation test (100th percentile
   vs. 1,000 random-direction shuffles) and the clean parameter
   plateau — both point at a real, structural directional signal, not
   noise or curve-fitting.
3. **Strongest evidence against:** thin margin over realistic cost/
   execution stress, and profit concentrated in one year with the
   most recent year negative.
4. **Statistically/structurally credible?** Statistically, yes — the
   permutation test is a real hypothesis test and it's decisive.
   Structurally, plausible (USDJPY as a broad JPY-strength proxy has
   an honest economic story) but unconfirmed as *durable* — 3 years
   and ~500 trades cannot yet distinguish "a real, persistent edge
   having a bad year" from "a temporary regime effect that already
   ended."
5. **Biggest remaining uncertainty:** whether 2025 was the true
   regime and 2026 is normal variance, or whether 2025 was the anomaly
   and the edge is already fading. Not resolvable with current data —
   only more time (or a different validation angle) resolves it.
6. **Improve, reject, or move on?** **Neither improve nor reject yet
   — extend observation.** Per your own instruction not to modify a
   frozen candidate based on this pass/fail, and given the failures
   are about *durability and cost margin* rather than *the signal
   being fake* (the permutation test rules that out), the correct
   action is more data, not a parameter change.
7. **Exact next experiment recommended:** **Do not touch the strategy.
   Log it as PROMISING BUT INSUFFICIENT and set a dated re-check** —
   re-run this identical frozen validation gate again once
   meaningfully more of 2026 has accumulated (e.g. at the existing
   2026-08-25 AMR checkpoint, or ~60-90 days out), asking only: has
   2026 continued negative, stabilized, or recovered? That single new
   data point is worth more than any parameter search right now. In
   parallel, if you want a second, independent angle on the same
   underlying idea, the descriptive-first discipline that worked so
   well in phase 11 suggests testing whether the USDJPY-proxy
   mechanism generalizes to a DIFFERENT JPY cross entirely (not a
   parameter tweak — a genuinely separate market) as an out-of-family
   replication check, which is exactly the kind of evidence that made
   CADJPY convincing for the ARB/AMR families earlier in this project.
