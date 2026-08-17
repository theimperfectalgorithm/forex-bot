# Phase 33 Pre-Registration — FROZEN BEFORE ANY CANDIDATE RESULT IS EXAMINED

**This document is written and committed before any candidate backtest is run. Per the explicit instruction, it will not be changed based on results. If a methodological flaw is found later, it will be documented as a dated amendment in §12, never silently edited.**

---

## 1. Research universe (frozen)

**Instruments:** AUDUSD, USDCAD, USDCHF, XAUUSD. **EURUSD and GBPUSD are explicitly excluded from this discovery pass** — both are already-settled dead ground in this project's own research record (`PROJECT_REPORT.md` §4: phase 1's VRT/MDS/RFMC/ARB matrix and phase 4's dedicated pro-style EU/GU screen both returned 0 passes across 38 combined tests; "London breakouts lose chased AND faded" is an explicit prior finding). Re-testing the same instruments with a similarly-shaped mechanism would not be new evidence. GBPUSD is additionally already the base of a live strategy (Monday Drift) — adding a second GBPUSD strategy would not address the diversification gap Phase 32 identified.

**No JPY cross is included in this universe** — Phase 32 priority 4 (non-JPY preferred) and priority 3 (genuinely different mechanism) both argue against it, and no JPY instrument presents an unusual enough mechanism case to justify the Part 2 exception.

## 2. Strategy families and hypotheses (frozen — exactly two candidates, no more)

Per the explicit instruction not to brute-force a large universe, **exactly two pre-registered hypotheses**, each with a small number of economically-motivated parameters (no grid search):

### Candidate 1: XAUUSD_LONDON_VOL_EXPANSION
- **Family:** volatility contraction → expansion breakout (session breakout, Priority 5: London/NY).
- **Hypothesis:** gold's real directional moves cluster around London/NY hours on real-yield/macro-data repricing, following periods of low realized volatility. This is not a new idea invented for this task — it is the exact "Gold, London/NY session-specific" direction already listed as untested in `PROJECT_REPORT.md` §4's own research backlog ("gold's real directional action tends to cluster around US data/real-yield moves in London/NY hours... never been tested").
- **Mechanism:** genuinely different from AMR (mean-reversion) and CADJPY ARB (Asian-range breakout) — a volatility-contraction-to-expansion breakout timed to the London open, distinct session and distinct trigger condition from the existing ARB.
- **Entry concept:** at the London-session open (07:00 UTC), if the preceding 4-hour realized range (ATR-normalized) is below its own 30-day 33rd percentile (a volatility-contraction precondition), enter in the direction of the first subsequent H1 close beyond the pre-London 4-hour range.
- **Exit concept:** SL at the opposite side of the pre-London range; TP at 2.0x the range width (matching this project's existing ARB convention for a like-for-like cost/robustness comparison, not tuned).
- **Risk model:** R-multiple normalized to SL distance, matching every other strategy in this project.
- **Expected regime:** HIGH-volatility compatible by construction (it explicitly requires an antecedent low-vol regime and profits from the subsequent expansion) — this is the direct test of Phase 32 Priority 1.
- **Expected diversification:** London-session entry (vs. the book's 94.7% Asian-session concentration) and a different trigger mechanic from the existing Asian-range ARB.

### Candidate 2: USDCAD_MOMENTUM_CONTINUATION
- **Family:** trend/momentum continuation (Priority 3: genuinely different mechanism).
- **Hypothesis:** this project's own prior research (`PROJECT_REPORT.md` §4, phase 6) found a genuine cross-sectional momentum edge on CADJPY ("CADJPY new edge (both families)") — this candidate tests whether a directly analogous H4 momentum-continuation mechanic, using the same economic logic (persistent multi-day directional continuation once an established move is confirmed), generalizes to USDCAD, a non-JPY CAD pair not currently in the book. This is an extension of an already-validated finding to an adjacent, non-JPY instrument, not a speculative new idea.
- **Mechanism:** trend continuation — genuinely different from mean-reversion (AMR) and from breakout-of-a-fixed-range (ARB, Candidate 1).
- **Entry concept:** H4 bars; if price closes beyond its own 20-bar (≈ prior ~3.3 trading days) high/low AND the 20-bar directional efficiency ratio (net displacement / sum of absolute bar-to-bar moves, already used as a project convention in `data/phase26_all_trades.csv`'s `efficiency_ratio` field) exceeds 0.35, enter in the breakout direction at the next H4 open.
- **Exit concept:** SL at 1.5x the 20-bar ATR; TP at 3.0x the 20-bar ATR (a 2:1 reward:risk trend-following convention, not tuned to this specific instrument).
- **Risk model:** R-multiple normalized to SL distance.
- **Expected regime:** trend-following mechanics are typically HIGH-volatility-compatible or neutral (they need a real move to exist) — tested, not assumed.
- **Expected diversification:** non-JPY, and a trend mechanism entirely absent from the current book (which has zero trend-following strategies).

**No candidate may be added retroactively.** If both fail, that is a reportable outcome (§Part 26 of the task instructions), not a reason to add a third.

## 3. Timeframes and data period (frozen)

- Candidate 1: entries evaluated on H1 bars (needs to detect the London-open breakout at bar resolution); the antecedent volatility-contraction condition uses a rolling ATR computed from H1 bars.
- Candidate 2: entries evaluated on H4 bars.
- **Data period:** 2023-01-01 to 2026-08-14 (matching Phase 30's pull, MetaQuotes-Demo broker feed — this session has no 5ers broker data access, disclosed as a limitation, not hidden).

## 4. Train / validation / OOS split (frozen, chronological, no shuffling)

- **TRAIN (discovery/parameter-freeze window):** 2023-01-01 to 2024-08-31 (~20 months).
- **VALIDATION (in-sample confirmation, still used for any final parameter check before freezing — none is planned, since parameters are fixed by economic reasoning in §2, not fit):** 2024-09-01 to 2025-04-30 (~8 months).
- **OOS (final, held out, never inspected before this point):** 2025-05-01 to 2026-08-14 (~15.5 months).
- Parameters (§2) are fixed by economic/project-precedent reasoning before any window is inspected — there is no parameter-fitting step, so VALIDATION here serves only as an intermediate integrity check (does the hypothesis show any signal at all before the final OOS look), not a tuning fold.

## 5. Cost assumptions (frozen)

| Instrument | Normal spread (round-trip, price units) | 1.5x | 2x |
|---|---|---|---|
| AUDUSD | 0.00018 | 0.00027 | 0.00036 |
| USDCAD | 0.00020 | 0.00030 | 0.00040 |
| USDCHF | 0.00020 | 0.00030 | 0.00040 |
| XAUUSD | $0.35 | $0.53 | $0.70 |

(Identical to the conservative assumptions already used and disclosed in Phase 30's calendar screen — reused for consistency, not re-derived.)

## 6. Parameter ranges (frozen — no grid search)

Exactly the single economically-motivated parameter set per candidate stated in §2. **No sweep, no optimization.** A sensitivity check (§11) perturbs each candidate's key threshold by ±20% as a robustness test only, not a search for a better value — the ORIGINAL parameter set remains the one carried into every gate; the sensitivity check's role is solely to disqualify a candidate whose result is fragile to small perturbations, never to select a better-performing variant.

## 7. Robustness tests required (frozen, Gate 2)

1. Parameter sensitivity (±20% perturbation on the single key threshold — volatility-contraction percentile for Candidate 1, efficiency-ratio threshold for Candidate 2).
2. OOS stability (OOS split further into two sub-halves, checked for consistent sign).
3. Cost stress (1.0x / 1.5x / 2.0x, §5).
4. Monte Carlo trade-order reshuffling (10,000 draws, OOS trades only).
5. Drawdown distribution (OOS).
6. Losing-streak distribution (OOS).
7. Regime stability (HIGH/NORMAL/LOW volatility tercile, §8).
8. Directional stability (BUY vs. SELL, where the candidate is not inherently one-sided).
9. Session stability — not separately applicable (both candidates are single-session by design; documented as N/A, not silently skipped).

## 8. HIGH-volatility gate (frozen, Part 12)

Classified using the same tercile methodology as every prior phase (own-instrument ATR terciles, not portfolio-borrowed terciles — since the candidate is a genuinely new instrument): a candidate is:
- **STRONG HIGH-VOL COMPATIBILITY** if OOS HIGH-vol-tercile expectancy is positive and not the candidate's weakest tercile.
- **NEUTRAL** if HIGH-vol-tercile expectancy is not clearly positive or negative (within noise of zero given the tercile's own trade count).
- **WEAK** if HIGH-vol-tercile expectancy is clearly negative and/or the candidate's worst tercile.
- **UNKNOWN** if the HIGH-vol tercile has too few OOS trades (<10) for any classification.

## 9. Portfolio correlation / drawdown-correlation gate (frozen, Part 13)

Control = the frozen Phase 31/32 historical portfolio (`data/phase26_all_trades.csv`, validated). Correlation computed on daily aggregated R (trade-level correlation is not meaningful for this book, per Phase 31's own documented finding — reused, not re-litigated). **Drawdown correlation (computed on the control's own worst-decile drawdown days, exact same definition as Phase 31/32) is the primary metric — a candidate with drawdown-day correlation exceeding its normal-day correlation by more than 0.15 is downgraded regardless of its average correlation**, per Phase 32's own finding that this exact pattern (low normal, high drawdown correlation) marks a poor diversifier.

## 10. Portfolio integration test (frozen, Parts 17-20)

Candidate blended into the control at a fixed standardized weight (0.5x and 1.0x of the control's own median single-strategy daily-R-std, identical methodology to Phase 32 — reused, not reinvented) — **no weight optimization**. Metrics: total R, expectancy, PF, max drawdown, drawdown duration, max losing streak, HIGH-vol drawdown, correlated-loss days, JPY exposure change, session-concentration change, effective diversification, Monte Carlo p95/p99 drawdown — all computed as CONTROL vs. CONTROL+CANDIDATE, using the actual candidate OOS trade stream (not a synthetic proxy, since this candidate has real backtested trades, unlike Phase 32's diagnostic archetypes).

## 11. Multiple-testing controls (frozen)

- **Exactly 2 pre-registered hypotheses, each with exactly 1 parameter set** (no grid) — total "tests" = 2 primary + the ±20% sensitivity check (2 additional per candidate) = 6 total parameter evaluations, all disclosed in the final registry regardless of outcome.
- Every one of these 6 evaluations will appear in `reports/phase33_candidate_registry.csv` and `reports/phase33_candidate_results.csv`, pass or fail — **no candidate or parameter variant is omitted from the report for looking weak.**
- **EXPLORATORY vs. CONFIRMATORY:** the entire OOS window (2025-05-01 to 2026-08-14) is used exactly once per candidate, inspected only after TRAIN+VALIDATION integrity checks pass — this is as close to a genuine confirmatory test as this project's data allows. If a candidate is revisited in any future phase, that later look must be treated as exploratory relative to this OOS window, since it will no longer be blind.

## 12. Candidate classification rules (frozen — Part 22's 8 categories, applied mechanically)

- **A. REJECTED — NO EDGE**: fails Gate 1 (Part 10 — negative or statistically indistinguishable-from-zero OOS expectancy, OOS PF ≤ 1.0, OOS trade count too small to evaluate, or evidence of a single-trade-dependent result).
- **B. REJECTED — ROBUSTNESS FAILURE**: passes Gate 1 but fails any of the §7 robustness tests materially (sign flips under ±20% parameter perturbation, OOS sub-halves disagree in sign, Monte Carlo shows the observed result within normal noise only by chance at an implausible percentile in the wrong direction).
- **C. REJECTED — COST FRAGILE**: OOS PF falls below 1.0 at 1.5x cost.
- **D. REJECTED — HIGH-VOLATILITY FAILURE**: HIGH-vol tercile classified WEAK (§8).
- **E. REJECTED — POOR DRAWDOWN DIVERSIFICATION**: drawdown-day correlation exceeds normal-day correlation by >0.15 (§9), regardless of other results.
- **F. REJECTED — POOR PORTFOLIO FIT**: passes all of A-E's gates individually but CONTROL+CANDIDATE shows a materially worse combined max drawdown or combined max losing streak than CONTROL alone at 1.0x weight, with no offsetting HIGH-vol or drawdown-correlation benefit large enough to justify it (a qualitative judgment call, made transparently with the numbers shown, not hidden behind a single threshold).
- **G. PROMISING — REQUIRES MORE VALIDATION**: passes Gates 1-2 and the HIGH-vol/drawdown-correlation/portfolio-fit gates, but sample size or another explicitly named limitation (e.g. thin session-stability evidence) falls short of §13's full bar.
- **H. PORTFOLIO QUALIFIED — DEMO FORWARD TEST**: satisfies every item in Part 23's 10-point list with no exceptions.

**These rules will be applied exactly as written above, in the order written, regardless of which candidate they favor or disfavor.**

---

*Frozen at the time of this commit. No candidate has been backtested yet. Any change to this document after candidate results exist will be logged as a dated, explicit amendment below this line — never a silent edit.*
