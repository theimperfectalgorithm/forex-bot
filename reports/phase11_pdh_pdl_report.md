# Research Report — Previous Day High/Low Reactions (London/NY)

**Experiments:** EXP-002 through EXP-029 (28 baseline) — see `experiments/experiments.csv`
**Family:** `pdh_pdl_breakout`, `pdh_pdl_rejection`
**Date:** 2026-08-11

## Hypothesis

Price reacts predictably around the previous complete trading day's
high and low during London and New York sessions — either continuing
through the level (breakout) or reversing back into range after a
false break (rejection).

## Step 1–2: descriptive analysis (measured before any trading rule)

Across all 7 pairs × 2 sessions, the previous day's high/low is
touched in **18–31%** of session bars — a common, liquid phenomenon,
not a rare event. But the forward-8-hour price drift after a touch —
whether the bar closed beyond the level (candidate breakout) or
snapped back inside it (candidate rejection) — was **positive between
38.2% and 59.1% of the time, averaging 49.3%** across all 56
pair/session/side/type combinations. **49.3% is statistically
indistinguishable from a coin flip.** This descriptive step alone was
close to sufficient to predict the outcome below — exactly the value
the "measure before you trade" discipline is supposed to deliver.

## Step 3: baseline backtest

28 configurations (7 pairs × London/NY × breakout/rejection variant),
36 months H1, spread-paid, IS = first 24mo / OOS = last 12mo, standard
project criteria (PF > 1.3, DD < 8%, ≥60% profitable months IS,
positive OOS).

**Result: 0/28 passed. 0/28 even passed in-sample.**

- **Breakout variant: uniformly dead.** Every one of the 14 IS profit
  factors fell between 0.72 and 0.99 — not "no edge," but
  *consistently losing to spread and false breaks*, the same failure
  signature as this project's already-dead LORB/NY-continuation
  families (see below).
- **Rejection variant: weak, and — on closer inspection —
  incoherent rather than marginal.** IS PF ranged 0.51–1.20 with no
  pattern tying the better cells together (best: NZDUSD-NY at 1.20;
  worst: USDCHF-NY at 0.51, right next to it in the same table).
  Compare this to the phase-10 NZDJPY finding, where 9 neighboring
  parameter cells were ALL between 1.25 and 1.53 with drawdown
  improving monotonically as one parameter moved — a coherent signal.
  Here, adjacent cells swing wildly with no explanatory story. That's
  the noise signature, not a real edge hiding just below the bar.
  **Conclusion: reject, do not chase with further parameter
  refinement** — consistent with the project's own overfitting
  red flags ("isolated peaks vs. plateaus") and the master-prompt's own
  instruction to prefer discovering a new phenomenon over optimizing a
  weak one.

## Robustness / walk-forward / Monte Carlo

Not run. Per this project's established process (and the master
prompt's own instruction), deep validation is reserved for candidates
that clear in-sample criteria first. None did.

## Relationship to prior project research

This closes the loop on a broader research prompt that named 5
strategy families for EURUSD/GBPUSD-class intraday trading. The other
4 (Asian-range breakout, London trend-pullback, London→NY continuation,
Asian sweep-reversal) were already tested to failure in this project's
earlier phases (1, 3, 4, 5) — this was the one previously-untested
angle. With this result, **all 5 originally-proposed families are now
closed as dead ground** on this pair universe at H1/retail-data tier.

## Problems discovered

None in the tooling (walk_forward.py and monte_carlo.py were smoke-
tested against the known-good ARB-GBPJPY strategy before use — see
git history — and one real bug was caught and fixed there: reporting
"confidence intervals" on trade-order-shuffled final P&L, which is
mathematically always a single point since summation is order-
invariant). The finding itself is clean and not an artifact of a
measurement error.

## Decision

**REJECT** both variants, all 7 pairs, both sessions. Logged
individually per pair/session in `experiments/experiments.csv`
(EXP-002–EXP-029, status `FAILED`).

## What this adds to settled dead ground

Previous-day-high/low breakout and rejection reactions, London/NY,
H1, on EURUSD/GBPUSD/USDJPY/AUDUSD/USDCAD/USDCHF/NZDUSD — never
re-test this family on these pairs without new data (tick-level,
order-flow) or a genuinely different mechanic.

## Next experiment

No natural follow-up within this family — the descriptive statistics
themselves (49.3% average forward-drift positivity) already say there
is nothing here to refine. Candidates for the next research cycle,
in the order this project's evidence currently favors:
1. Continue the JPY-cross session-structure line (where every
   validated edge in this project has come from) — e.g. deep-validate
   the pending NZDJPY cross-asset-momentum candidate from phase 10/10b
   through this same walk-forward + Monte Carlo pipeline before any
   build decision.
2. The AMR trend-filter research already queued for the 2026-08-25
   checkpoint.
3. If EURUSD/GBPUSD-class pairs are to be revisited at all, it should
   be with a structurally different data source (tick/order-flow) or
   angle (calendar/seasonal, per GBPUSD Monday-drift's precedent) —
   not another session/level-reaction geometry at H1.
