# Phase 39 — FX-Technical Research Ceiling Assessment

RESEARCH/METHODOLOGY AUDIT ONLY. No new backtest performed to produce this assessment — built entirely from `reports/phase39_fx_research_inventory.csv` (OBSERVED) and the coverage/duplication CSVs derived from it (CALCULATED).

## 1. Research coverage (CALCULATED)

10 confirmatory hypotheses across 8 distinct return-driver concepts (`reports/phase39_structural_duplication.csv`), plus a 60-cell exploratory calendar/drift screen. 4 session buckets represented (Asian/London/multi-session/session-independent are covered; New York and London/NY-overlap each tested once). 8+ distinct instruments/instrument-groups tested, spanning FX majors, JPY crosses, one commodity (XAUUSD), and one synthetic cross-sectional basket.

## 2. Structural diversity (CALCULATED)

Of the 8 distinct return-driver concepts, 2 (volatility-state-change breakout, trend/momentum continuation) were each tested twice with a different instrument — genuine but modest duplication, not a single concept tested 10 times. The remaining 6 concepts (calendar drift, NY range breakout, NY momentum, overlap continuation, cross-sectional relative ranking, session-transition breakout) are each unique single tests.

## 3. Number of genuinely distinct mechanisms (CALCULATED)

8, per §2. This is a modest but non-trivial number — comparable in scale to a typical single-researcher quant screening pass, not an exhaustive search of the FX-technical hypothesis space.

## 4. Session coverage (CALCULATED, `reports/phase39_fx_session_coverage.csv`)

New York (single-session) and London/NY-overlap were each tested with exactly 1 confirmatory hypothesis — thin coverage per session, not a saturated search. London and Asian sessions have been tested via session-transition and calendar-drift mechanisms respectively, but not via momentum/breakout mechanisms specific to those sessions in isolation.

## 5. Instrument coverage (CALCULATED, `reports/phase39_fx_instrument_coverage.csv`)

USDCAD and AUDUSD are the most-tested single instruments (3 confirmatory hypotheses each, each with a different mechanism — not parameter variants of the same test). EURJPY, GBPJPY, CADJPY have zero confirmatory (non-AMR/ARB) technical hypotheses tested in this ledger; their coverage is limited to the currently-live AMR/ARB strategies (which predate this ledger's confirmatory-testing discipline and are out of scope for backtesting in this phase).

## 6. Robustness outcomes (OBSERVED)

0 of 10 confirmatory hypotheses reached even a WEAK pass at Gate 1 (OOS PF > 1.0) with the exception of AUDUSD Monday LONG, which passed every robustness gate and failed only on drawdown diversification. This is a genuinely low pass rate, but the sample (10) is small enough that a single-digit failure count is not, by itself, decisive evidence of a ceiling — see §9 (information gain) for the fuller reasoning.

## 7. Portfolio integration outcomes (OBSERVED)

Every candidate that reached the portfolio-integration stage (AUDUSD Monday LONG, H1, H2) either failed or worsened portfolio drawdown behavior. All three failed specifically on drawdown-day correlation to the existing six-strategy control — this is the more informative, repeated pattern (3 of 3 candidates that got this far failed the same gate), stronger evidence than the raw edge-failure count alone.

## 8. Remaining unexplored territory (CALCULATED)

Per `reports/phase39_fx_session_coverage.csv`, New York-session and London/NY-overlap mechanisms remain thinly tested (1 hypothesis each). Per `reports/phase39_fx_instrument_coverage.csv`, JPY crosses beyond the currently-live AMR/ARB set have no confirmatory technical-hypothesis coverage. These are genuinely unexplored, not exhausted.

## 9. Expected information gain from further FX-technical research (RESEARCH-PRIORITY ASSESSMENT)

**LOW-to-MEDIUM for another FX-technical hypothesis in general.** The three candidates that reached the portfolio-integration stage (AUDUSD, H1, H2) all failed on the *same* gate (drawdown correlation to the existing control), despite testing three structurally different mechanisms (calendar drift, cross-sectional ranking, session-transition breakout). This repeated pattern across genuinely different mechanisms is more informative than the raw rejection count — it suggests the constraint may be less about "which FX mechanism" and more about the existing control portfolio's own concentration (JPY-linked, Asian/London-session, mean-reversion-heavy — see Phase 31/32's own diagnosis), which any FX-technical candidate correlated to broad USD/JPY/risk-sentiment moves is likely to share. A hypothesis specifically designed to be uncorrelated with the control's known concentration factors (rather than merely a new technical mechanism) might still have HIGH information gain — but that is a narrower, more targeted search than "test another FX mechanism," and Phase 39 does not identify a specific such candidate (per the "no strategy design" constraint of this phase).

## 10. Research cost (RESEARCH-PRIORITY ASSESSMENT)

LOW-to-MEDIUM per additional FX-technical hypothesis (existing data pipeline, existing validation battery, existing control portfolio — no new infrastructure required). This is a real point in favor of continuing IF a genuinely novel, portfolio-uncorrelated candidate can be identified; it is not, by itself, a reason to continue if no such candidate is in view.

---

## Conclusion

### **C. FX TECHNICAL RESEARCH CEILING REACHED FOR NOW — for undifferentiated FX-technical hypothesis generation.**

This is NOT "FX technical trading has no edge" (see `reports/phase39_multiple_testing_audit.csv` for the explicit distinction). It is: 10 confirmatory hypotheses across 8 distinct mechanisms, and — more decisively — 3 of 3 candidates that reached the portfolio-integration stage failing on the *same* drawdown-correlation gate regardless of mechanism, together constitute evidence that another undifferentiated FX-technical mechanism search is unlikely to solve the portfolio's actual constraint. The ceiling is reached **for the "generate another FX mechanism and test it" research mode specifically**, not for FX-technical trading as an asset class in the abstract. This conclusion is evidence-based (§1-9), not merely a function of the raw rejection count (per the frozen decision rule in `reports/phase39_preregistration.md` §F).
