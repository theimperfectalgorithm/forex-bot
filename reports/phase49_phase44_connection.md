# Phase 49 — Connection to Phase 44

**Diagnostic only. This document does not re-test, re-optimize, or attempt to overturn Phase 44's finding.**

## What Phase 44 established

Testing 4 frozen, pre-declared portfolio-control counterfactuals against the historical control produced **NO PORTFOLIO CONTROL JUSTIFIED**. The most attractive-looking control (a 50% HIGH-volatility new-entry suppression rule) showed a real ~30% drawdown reduction — but suppressed ~61% historical winners and inverted in the most recent regime, disqualifying it. A control targeting Phase 43's own "worst cell" (HIGH-vol + concurrency≥4) made aggregate drawdown *worse*, not better.

## What Phase 49 adds

Phase 49 is a deeper diagnostic pass, not a retest of Phase 44's specific controls. Its most relevant finding for any *future* control design is the **descriptive counterfactual** in `reports/phase49_descriptive_counterfactuals.csv`: restricting the worst-10%-of-days population to only concurrency<4 days reduces that population's total loss from -290.4R to -25.9R — a ~91% reduction **within the already-selected worst-day population**. This is a fundamentally different, more targeted measurement than Phase 44's controls, which suppressed *entries* prospectively across the *entire* dataset (not just on already-identified bad days) and were therefore evaluated on their effect on the *whole* portfolio, including many good days whose winning trades happened to coincide with the same conditions.

## What a future control would need to address, if one were ever tested

1. **It would need to be selective, not broad.** Phase 44's HIGH-vol suppression rule failed specifically because it removed a majority of winning trades along with losers. Phase 49's finding that worst-10%-day losses concentrate heavily in the concurrency≥4 subset does **not** imply that concurrency≥4 trades are bad in general — Phase 49's own concurrency-threshold marginal analysis (`reports/phase49_concurrency_analysis.csv`) shows concurrency alone has no clean monotonic relationship with daily R across the *whole* dataset (mean R is not monotonically worse at higher concurrency thresholds). The concentration effect is specific to the *already-stressed* population, not a general property of high-concurrency days.

2. **It would need to address a genuinely multi-factor, low-explanatory-power mechanism, not a single lever.** Phase 49's explanatory OLS model (§Multi-factor model) explains only ~3.8% of daily-R variance (R²=0.038) even using seven candidate predictors simultaneously — consistent with Phase 41's original H. NO SINGLE DOMINANT FACTOR verdict, now reconfirmed with a different methodology.

3. **It would need to account for GBPJPY_AMR's role carefully.** Phase 49's GBPJPY_AMR-specific analysis (§15) finds GBPJPY_AMR's own daily R is highly correlated with total portfolio R on the days it's active (0.727) — but its *presence* does not by itself predict worse days (active-day mean R is actually lower than inactive-day mean R, a modest, not dramatic, difference) — consistent with interpretation #2/#3 from this phase's own framing (GBPJPY_AMR co-occurs with and amplifies portfolio-wide moves, more than it independently causes them). A future control naively targeting GBPJPY_AMR specifically would likely repeat Phase 44's CADJPY_ARB-h4-filter-style mistake of solving a symptom, not the underlying multi-strategy correlated-loss pattern.

4. **It would need to survive the same regime-robustness and winner-suppression scrutiny Phase 44 already applied.** Nothing in this phase's evidence base suggests any of the specific patterns found here (concurrency concentration within the stress population, the JPY-controlling-for-volatility pattern in §JPY analysis) would behave differently than Phase 44's tested controls when actually implemented and tested against regime robustness and winner-composition — this remains genuinely untested, and this phase does not test it.

## If no sufficiently stable mechanism is found

Per the explicit instruction, this must be stated plainly if true: **no single, sufficiently stable, temporally-validated mechanism was found in this phase that would, by itself, justify designing a new intervention with confidence.** The concurrency-concentration-within-stress finding is the most striking single number produced by this phase, but it describes the *stress population itself*, not a rule that could be applied prospectively without repeating Phase 44's core problem — most high-concurrency days are not bad days, so any prospective rule built on concurrency alone would still face the same broad-suppression / winner-removal problem Phase 44 already documented.
