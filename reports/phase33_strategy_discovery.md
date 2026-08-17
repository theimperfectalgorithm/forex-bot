# Phase 33 — Target-Profile Strategy Discovery & Pre-Registered Validation

**Research only. No live strategy, parameter, risk, or portfolio weight modified. No candidate deployed. AUDUSD Monday LONG not modified.**

**Pre-registration:** `reports/phase33_preregistration.md`, committed (`8bcd30e`) before any candidate was backtested. **Not altered after seeing results** — every gate/threshold in this report is copied verbatim from that frozen document.

---

## 1. Executive summary

Two pre-registered candidates were tested end-to-end (XAUUSD London volatility-expansion breakout; USDCAD H4 momentum continuation), both with real MT5-pulled H1/H4 data, a strict chronological TRAIN/VALIDATION/OOS split, cost stress, Monte Carlo, HIGH-volatility classification, and portfolio-integration testing against the real Phase 31/32 control. **Both candidates showed a positive, cost-surviving OOS edge in aggregate — and both failed the pre-registered robustness gate**: each showed a sign-inconsistent OOS split (first-half negative, second-half positive expectancy) and a sign-inconsistent (for USDCAD, fully sign-*reversing*) result under a ±20% parameter perturbation. Per the frozen classification rules, both are **B. REJECTED — ROBUSTNESS FAILURE**.

**Research outcome: A. NO CANDIDATE.** This is reported as a successful research result, per the explicit instruction, not a shortfall.

---

## 2. Phase 32 target profile (recap, not re-derived)

Priority order established in Phase 32 and used to design this phase's candidates: (1) HIGH-volatility compatibility, (2) low drawdown-specific correlation, (3) genuinely different mechanism, (4) preferably non-JPY, (5) preferably London/NY exposure.

## 3. Preregistration

Full document: `reports/phase33_preregistration.md`. No amendments were required — no methodological flaw was discovered during execution that required a documented change.

## 4. Research universe

AUDUSD, USDCAD, USDCHF, XAUUSD (EURUSD/GBPUSD excluded as already-settled dead ground, §1 of the preregistration). **Only 2 of these 4 instruments were actually used** — the two pre-registered hypotheses target XAUUSD and USDCAD specifically; AUDUSD and USDCHF were part of the *eligible* universe but no hypothesis was written against them in this pass (consistent with "exactly two candidates, no more").

## 5. Data sources

MT5 `MetaQuotes-Demo` broker feed (this session has no 5ers broker data access — disclosed limitation, unchanged from every prior phase). XAUUSD: 21,272 H1 bars, 2023-01-01 to 2026-08-14. USDCAD: 5,624 H4 bars, same range. Both validated for monotonic timestamps, zero duplicate candles, and positive/consistent OHLC before any backtest ran (`src/phase33_strategy_discovery.py::pull()`) — no data-integrity STOP was triggered.

## 6. Hypothesis registry

Full detail: `reports/phase33_candidate_registry.csv`. Both hypotheses were written and committed to the registry **before** either backtest was run, with explicit economic justification tracing to prior project research (`PROJECT_REPORT.md` §4's research backlog for XAUUSD; the phase-6 CADJPY cross-sectional momentum finding for USDCAD) — neither is a speculative new idea invented for this task.

## 7. Discovery methodology

Exactly one parameter set per candidate, fixed by economic reasoning before any data window was inspected (§2/§6 of the preregistration) — **no grid search, no optimization, no rule modification after seeing OOS results.** The only parameter variation performed (±20%) was a pre-declared robustness *check*, not a search for a better value — the original parameter set is the one carried into every gate regardless of what the ±20% check showed.

---

## 8. Candidate results (TRAIN / VALIDATION / OOS)

Full detail: `reports/phase33_candidate_results.csv`.

| Candidate | TRAIN (n / PF / expectancy R) | VAL (n / PF / expectancy R) | **OOS (n / PF / expectancy R)** |
|---|---|---|---|
| EXP-125 XAUUSD London Vol-Expansion | 151 / 1.168 / +0.109 | 52 / 1.182 / +0.116 | **114 / 1.185 / +0.117** |
| EXP-126 USDCAD Momentum Continuation | 57 / 1.006 / +0.004 | 26 / 0.687 / −0.239 | **57 / 1.246 / +0.155** |

**Both candidates show a positive OOS profit factor and expectancy in aggregate.** USDCAD's own VALIDATION fold was actually negative (PF 0.687) despite an OOS recovery — an early signal, confirmed more decisively in §10, that this candidate's edge is not stable across sub-periods.

## 9. OOS results

Both candidates clear the pre-registered Gate 1 bar (positive OOS expectancy, OOS PF > 1.0, OOS trade count 57-114, no single-trade dependency observed in either trade log). **Gate 1: PASS for both.**

---

## 10. Robustness (Gate 2 — the decisive test)

Full detail: `reports/phase33_robustness_results.csv`.

| Candidate | OOS 1st-half expectancy R | OOS 2nd-half expectancy R | Sub-half sign consistent? | −20% expectancy R | Baseline | +20% expectancy R | Sign consistent across perturbation? |
|---|---|---|---|---|---|---|---|
| EXP-125 XAUUSD | −0.071 | +0.305 | **NO** | +0.045 | +0.117 | **−0.022** | **NO** |
| EXP-126 USDCAD | −0.091 | +0.393 | **NO** | +0.242 | +0.155 | **−0.260** | **NO** (full sign reversal) |

**Both candidates fail Gate 2 on both of the two tests applied.** XAUUSD's OOS split shows a losing first half and a winning second half — the aggregate positive result is not evenly distributed across the OOS window. USDCAD's parameter sensitivity is the more severe failure: a +20% perturbation of the efficiency-ratio threshold **fully reverses the sign** of OOS expectancy (+0.155R → −0.260R), the single most fragile result found in this phase.

**Per the frozen classification rules (§12 of the preregistration), a robustness failure is disqualifying regardless of downstream results.** Sections 11-19 below are still reported in full — per the explicit instruction not to hide or truncate a candidate's evidence for looking weak — but neither candidate can advance past B given this finding.

---

## 11. Cost stress

Full detail: `reports/phase33_cost_stress.csv`.

| Candidate | Normal | 1.5x | 2.0x |
|---|---|---|---|
| XAUUSD PF | 1.185 | 1.171 | 1.159 |
| USDCAD PF | 1.246 | 1.197 | 1.150 |

**Both candidates remain above PF 1.0 through 2x cost stress** — neither is classified C (cost-fragile). This is a genuine point in both candidates' favor, reported honestly alongside the Gate 2 failure, not omitted.

## 12. HIGH-volatility behaviour

Full detail: `reports/phase33_high_volatility_analysis.csv`.

- **XAUUSD: STRONG HIGH-VOL COMPATIBILITY** — 100% of its 114 OOS trades fell in the HIGH-ATR tercile (terciles fixed from TRAIN+VAL data only, no leakage), with positive aggregate expectancy (+0.117R). **Important limitation, disclosed rather than hidden**: because every OOS trade fell in the HIGH bucket, there is no LOW/NORMAL OOS comparison for this candidate — this may reflect a genuine structural rise in gold volatility over the OOS window (2025-05 to 2026-08) rather than proof the strategy specifically thrives in HIGH-vol *relative to* its own LOW/NORMAL performance. Reported as STRONG per the mechanical rule, with this caveat attached.
- **USDCAD: UNKNOWN** — only 5 of 57 OOS trades fell in the HIGH-ATR tercile, below the pre-registered 10-trade minimum for any classification (§8 of the preregistration).

## 13. Correlation

See §14 (drawdown correlation is the primary, pre-registered metric — reported together for coherence).

## 14. Drawdown correlation

Full detail: `reports/phase33_drawdown_correlation.csv`. Computed using a control window matched to each candidate's actual OOS trading period (2025-05-01 to 2026-08-14) — a fair like-for-like comparison, not the full 3-year control history (an earlier full-history version of this comparison found zero overlapping drawdown days, since the candidates only trade in the most recent ~15.5 months of the control's 38-month history; corrected before being trusted).

| Candidate | Normal-day corr | Drawdown-day corr | Gate result |
|---|---|---|---|
| XAUUSD | −0.151 | **+0.111** | **FAIL — E. POOR DRAWDOWN DIVERSIFICATION** (dd_corr exceeds normal_corr by 0.262, over the 0.15 threshold) |
| USDCAD | −0.112 | INSUFFICIENT (n=5 overlapping drawdown days) | INSUFFICIENT EVIDENCE |

**XAUUSD independently fails the drawdown-diversification gate too** — its correlation to the control actually *rises* during the control's own worst-decile days, exactly the "looks uncorrelated normally but becomes correlated during drawdowns" failure pattern both Phase 32 and this phase's preregistration explicitly warned against. This is a second, independent reason (beyond Gate 2) that XAUUSD cannot advance.

## 15. Mechanism diversification

Both candidates are **GENUINELY DIFFERENT MECHANISM** from the current book: XAUUSD's volatility-contraction-to-expansion breakout differs from both AMR (mean-reversion) and CADJPY ARB (fixed Asian-range breakout); USDCAD's H4 trend/momentum continuation is a mechanism entirely absent from the current six strategies. **This criterion is satisfied by both — it is not the reason either was rejected.**

## 16. Session diversification

XAUUSD enters at the London open (07:00 UTC) through the close of that day's session — a genuine departure from the book's 94.7% Asian-session concentration. USDCAD's H4 momentum entries are not restricted to a single session by design, so they occur across all sessions including London/NY hours. **Both would have filled the Phase 31-identified session gap to some degree — again, not the rejection reason.**

## 17. Currency diversification

Both are non-JPY (XAU/USD and USD/CAD) — both satisfy Phase 32's Priority 4. **Also not the rejection reason.**

## 18. Portfolio integration

Full detail: `reports/phase33_portfolio_integration.csv` (OOS-window-matched control, fair comparison).

| Candidate | Weight | Control total R | Combined total R | Control max DD | Combined max DD |
|---|---|---|---|---|---|
| XAUUSD | 0.5x | 126.72 | 133.37 | −14.53 | **−14.04** (improved) |
| XAUUSD | 1.0x | 126.72 | 140.03 | −14.53 | −14.53 (unchanged) |
| USDCAD | 0.5x | 126.72 | 131.13 | −14.53 | −15.05 (slightly worse) |
| USDCAD | 1.0x | 126.72 | 135.55 | −14.53 | −15.58 (slightly worse) |

**Both candidates would have modestly increased total R over this window.** XAUUSD's combined drawdown was slightly *improved* at 0.5x weight — a genuinely positive portfolio-fit signal that, on its own, would not have triggered an F (poor portfolio fit) classification. **This positive portfolio-integration result does not override the Gate 2 robustness failure or (for XAUUSD) the Gate-E drawdown-diversification failure** — per the frozen classification order, both gates were evaluated and failed before portfolio integration was even reached.

## 19. Monte Carlo

Full detail: `reports/phase33_monte_carlo.csv`. 10,000-simulation trade-order reshuffle of each candidate's own OOS trades.

| Candidate | Actual OOS max DD | MC p5 | MC p50 | MC p95 | Actual DD's percentile in MC |
|---|---|---|---|---|---|
| XAUUSD | −15.14R | −18.55 | −11.61 | −7.47 | 19.0th |
| USDCAD | −7.39R | −13.13 | −8.23 | −5.29 | 65.2th |

Neither candidate's actual drawdown sequencing is a statistical outlier relative to a random reshuffle of its own trades — both sit within an unremarkable percentile range. This test does not itself disqualify either candidate; it is reported for completeness alongside the gates that did.

---

## 20. Multiple-testing assessment

- **2 pre-registered hypotheses, 1 parameter set each** (the frozen preregistration's entire scope).
- **6 total parameter evaluations disclosed**: 2 baseline + 2×2 sensitivity perturbations — every one appears in `reports/phase33_robustness_results.csv`, none omitted for looking weak.
- **0 candidates added retroactively.** No third candidate was tested after seeing these two results.
- **This entire report is EXPLORATORY relative to any future look at this exact OOS window** — the 2025-05-01 to 2026-08-14 period has now been inspected once per candidate; any future revisit of the same window for the same hypothesis would no longer be a blind confirmatory test.

## 21. Candidate classifications

Full detail: `reports/phase33_final_rankings.csv`.

| Candidate | Final classification | Primary rejection reason |
|---|---|---|
| EXP-125 XAUUSD London Vol-Expansion | **B. REJECTED — ROBUSTNESS FAILURE** | Fails both the OOS-sub-half stability check and the ±20% parameter-perturbation check; independently also fails the E. drawdown-diversification gate |
| EXP-126 USDCAD Momentum Continuation | **B. REJECTED — ROBUSTNESS FAILURE** | Fails both the OOS-sub-half stability check and the ±20% parameter-perturbation check, the latter showing a full sign reversal (+0.155R → −0.260R) |

## 22. AUDUSD Monday LONG assessment (prior candidate, not re-run)

Not modified, not re-backtested, per instruction. Evaluated against this phase's preregistered framework using its already-existing Phase 30/32 evidence:

| Preregistered gate | AUDUSD Monday LONG status | Basis |
|---|---|---|
| Gate 1 (OOS edge) | PASS | Phase 30: OOS PF 3.07, t=4.15 |
| Gate 2 (robustness) | **Not independently tested against this phase's specific sub-half/perturbation protocol** — Phase 30 found its IS t-stat (1.65) did not clear its own pre-registered bar, a related but distinct robustness concern already on record | Phase 30 |
| Cost stress | PASS | Cost-robust to 2x spread (Phase 30) |
| HIGH-vol gate | STRONG (its own best of 3 vol terciles) | Phase 32 |
| Drawdown-correlation gate | Correlation 0.29 to control — **above** the target range this phase's two new candidates were held to, and above the control's own 0.192 internal average | Phase 30/32 |
| Session | Monday-00:00-server only — does not fill the London/NY gap | Phase 32 |

**Classification: remains PROMISING / PARTIAL MATCH — not upgraded, not rescued.** Its correlation profile is weaker than what this phase's own drawdown-correlation gate would require of a new candidate, and it does not address the session gap. Consistent with instruction, it is not promoted here.

## 23. Demo-forward candidates

**None.** No candidate reached H. PORTFOLIO QUALIFIED. No demo-forward specification is produced, per the frozen rule that Part 24 only applies to candidates that reach H.

## 24. Rejected candidates

Both EXP-125 and EXP-126, classification B, per §21. Full evidence trail retained in all 9 supporting CSVs — nothing about either candidate's testing was hidden or abbreviated because the final result was a rejection.

## 25. Limitations

- **Data source**: MT5 MetaQuotes-Demo feed, not the 5ers production broker — disclosed throughout, unchanged from every prior phase's constraint.
- **XAUUSD's HIGH-vol classification (§12) rests on 100% of OOS trades falling in one tercile** — a genuine structural feature of the OOS window's gold volatility, not a fully independent within-OOS regime comparison.
- **USDCAD's HIGH-vol tercile has only 5 trades** — genuinely too few to classify, reported as UNKNOWN rather than guessed.
- **The drawdown-correlation gate for USDCAD has only 5 overlapping control-drawdown days within the short OOS window** — reported as INSUFFICIENT EVIDENCE, not estimated.
- **Only 2 candidates were tested** — a deliberately small, pre-registered set per instruction; this phase does not claim to have exhausted the target-profile search space, only this specific, disclosed pair of hypotheses.
- **±20% parameter perturbation is a coarse robustness check**, not a full sensitivity surface — a candidate could in principle be robust to some perturbations and not others; both candidates here failed even this single-axis check, which is sufficient for rejection under the frozen rules but does not fully characterize the parameter-response surface.

## 26. Final verdict

### Answers to Part 29's eight questions

1. **Did we find a credible independent edge?** Yes, in aggregate OOS terms, for both candidates (Gate 1 passed by both).
2. **Did we find a HIGH-volatility-compatible candidate?** XAUUSD showed STRONG HIGH-vol compatibility (with the caveat in §12); USDCAD's evidence was UNKNOWN (too few trades).
3. **Did we find a candidate with low drawdown correlation?** No — XAUUSD explicitly failed this gate; USDCAD's result was inconclusive, not confirmed.
4. **Did we find a genuinely different mechanism?** Yes, both candidates.
5. **Did we find a useful session diversifier?** Both would have added London/NY exposure, but neither survived the robustness gate to make this relevant to a promotion decision.
6. **Did we reduce JPY concentration meaningfully?** Not tested at the promotion stage — both candidates are non-JPY, but neither passed the gates required to evaluate this as a real portfolio change.
7. **Did the candidate improve portfolio-level behaviour?** In the limited OOS-window-matched integration test, both modestly increased total R; XAUUSD modestly improved combined drawdown at 0.5x weight. **This was not sufficient to overcome the robustness failure.**
8. **Did any candidate earn DEMO FORWARD TEST status?** No.

### Research outcome classification: **A. NO CANDIDATE**

Neither pre-registered hypothesis survived its own frozen robustness gate. This is reported as the actual result, not softened — **both candidates showed a real, cost-surviving, mechanism-diversifying OOS edge in aggregate, but neither showed a *stable* edge under either of the two independence checks this phase pre-committed to.** Per the explicit instruction, this is a successful research outcome, not a failure of the phase: it tells us the specific hypotheses tested here do not yet clear this project's own bar, and prevents a fragile candidate from being carried forward into demo testing on the strength of an aggregate number alone.

---

*No strategy, parameter, risk, or portfolio weight modified. No candidate deployed. AUDUSD Monday LONG not modified or promoted. Reproducible via `python src/phase33_strategy_discovery.py` then `python src/phase33_sensitivity_and_portfolio.py`.*
