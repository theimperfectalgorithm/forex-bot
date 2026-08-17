# Phase 34 — Strategy Research Universe Audit & Failure Analysis

**Research/synthesis only. No strategy, parameter, risk, or portfolio weight modified. No candidate deployed, rescued, or optimized. AUDUSD Monday LONG and the Phase 33 preregistration untouched.**

---

## 1. Executive summary

Phase 33 produced **A. NO CANDIDATE** because both pre-registered hypotheses failed a robustness gate that was doing exactly its job — not because the research universe had been exhausted. This phase confirms that conclusion with evidence: Phase 33 tested only **2 of 16 plausible strategy families (12.5% coverage)**, and **neither candidate specifically targeted the New York session** (the largest single coverage gap this phase identified) despite it being Phase 32's Priority 5 characteristic. Both failures trace to a diagnosable, implementation-specific design weakness — a fixed exit target for XAUUSD, a single narrow-peak threshold for USDCAD — not to a structural rejection of either instrument or mechanism family. **Verdict: B. PHASE 33 SEARCH WAS TOO NARROW** (ranked above, not instead of, a secondary C-adjacent finding about gate measurability — see §20).

---

## 2. Phase history

Full detail: `reports/phase34_research_timeline.csv`. Five phases (29-33) built, in order: a live-validation decision framework (29); a first non-JPY screen using one mechanism family (30); a diagnostic factor/regime map of the existing portfolio (31); a synthetic factor-importance ranking (32); and the first real, pre-registered strategy discovery pass (33). Each phase's stated limitations were carried forward honestly into the next, not silently dropped.

## 3. Current portfolio factor map

Full detail: `reports/phase34_portfolio_factor_map.csv`.

**OVERREPRESENTED FACTORS:** mean-reversion mechanism (4 of 6 strategies, 81.5% risk-weighted); Asian-session entry (94.7% risk-weighted, including GBPUSD Monday's session-label artifact); JPY exposure (94.7% risk-weighted).

**UNDERREPRESENTED FACTORS:** breakout mechanism (1 of 6 — CADJPY ARB); calendar/drift mechanism (1 of 6 — GBPUSD Monday); non-JPY exposure (1 of 6).

**MISSING FACTORS ENTIRELY:** New York session (0 of 6 strategies); trend-following (0 of 6); momentum (0 of 6); any HIGH-vol-by-construction mechanism (the closest is GBPUSD Monday, whose HIGH-vol strength is incidental to its calendar design, not a deliberate volatility-targeting mechanic).

**Per instruction, missing is not assumed to be valuable by default** — §10 and §14 test this explicitly rather than asserting it.

## 4. Phase 33 coverage

Full detail: `reports/phase34_phase33_coverage.csv`. Headline: **2/16 strategy families (12.5%)**, **2/4 eligible instruments (50%)**, and — the most consequential gap — **0 candidates specifically targeted New York or the London/NY overlap**, despite Priority 5 of the target profile.

## 5. Strategy-family taxonomy

Full detail: `reports/phase34_strategy_family_taxonomy.csv` — 16 families classified by mechanism, expected regime/session, likely portfolio correlation, HIGH-vol behavior, parameter-sensitivity risk, data requirements, complexity, and diversification rationale. Built as a reference taxonomy, **not tested** in this phase.

## 6. XAUUSD failure analysis

Full detail: `reports/phase34_failure_analysis.md` §XAUUSD. Summary: failed on OOS sub-half instability (large, well-powered evidence) and independently on the drawdown-correlation gate (gold's plausible macro/hedge co-movement with the JPY-heavy book's own risk-off drawdowns). Most likely mechanical cause of the instability: a fixed 2.0x TP multiplier not adapting to the actual size of each volatility expansion. **Diagnosed as implementation-specific, not a structural rejection of gold or the volatility-expansion family.**

## 7. USDCAD failure analysis

Full detail: `reports/phase34_failure_analysis.md` §USDCAD. Summary: the ±20% efficiency-ratio threshold perturbation produced a **full sign reversal** (+0.155R baseline → −0.260R at +20%) — the single most severe finding in either candidate. The baseline sits on a narrow peak, not the "broad plateau" this project's own six live strategies were all validated to have before being trusted. **Diagnosed as a single-timeframe design asking one threshold to carry all the discrimination burden — not a structural rejection of trend/momentum continuation.**

## 8. Common failure modes

Full detail: `reports/phase34_failure_modes.csv`. Both candidates share the identical pattern: **Gate 1 (OOS edge) passes comfortably, Gate 2 (robustness) fails on both of its independent checks.** This 2-for-2 pattern is itself informative (§9/§13 assess how much weight it deserves at n=2) but is not, on its own, proof that the broader strategy universe lacks a qualified candidate.

## 9. Target-profile audit

Full detail: `reports/phase34_target_profile_audit.csv`. Four of five requirements classified **WELL-SUPPORTED** (HIGH-vol compatibility, mechanism diversification, non-JPY-as-preference-not-rule) or **USEFUL BUT LIMITED** (drawdown correlation — conceptually sound, but its measurement is constrained by thin OOS-window/control-drawdown-day overlap). **London/NY is classified NOT YET TESTABLE at the candidate-outcome level** — not because the requirement is wrong, but because no Phase 33 candidate actually tested it. **No requirement was found to be poorly measured to the point of needing revision — the profile itself survives this audit intact.**

## 10. Validation-bar audit

Full detail: `reports/phase34_validation_bar_audit.csv`. The parameter-robustness and OOS-sub-half gates are assessed as the two highest-value checks in the entire protocol — both directly caught real, severe problems in both candidates. The HIGH-volatility and drawdown-correlation gates are the most sample-sensitive, correctly by design (they test rare/thin-sample conditions) but their measurement reliability should be explicitly preconditioned on a minimum trade count going forward (a process recommendation, not a bar reduction).

## 11. Sample-size analysis

Full detail: `reports/phase34_sample_size_analysis.csv`. XAUUSD's core failures (OOS sub-half, parameter sensitivity) are **CONCLUSIVE** — large swings on an adequately-powered 114-trade OOS base. USDCAD's parameter-sensitivity failure is also **CONCLUSIVE** (a full sign reversal is unambiguous regardless of sample size), but its OOS sub-half check (only ~28-29 trades per half) is flagged **LOW-POWER/WARNING ONLY** on its own — though it agrees directionally with the conclusive parameter-sensitivity finding, so the overall Gate 2 rejection for USDCAD remains well-supported by at least one conclusive independent check.

## 12. Multiple-testing assessment

Across the full project: Phase 30 tested 60 cells (one mechanism family); Phase 33 tested 2 hypotheses (two different families). Total distinct strategy-family attempts across the entire non-JPY research program to date: **3** (calendar/drift, volatility-contraction-expansion, trend/momentum) of 16 taxonomized families. **This is not yet enough accumulated exploration to require a stronger confirmation bar than Phase 33 already applies** — the project is still in an early, low-density exploration phase of this specific research direction, not a late-stage regime where selection bias from extensive prior search is a material concern. Phase 30's 60-cell calendar screen is **EXPLORATORY** evidence (a screen, not a confirmatory test); Phase 33's two candidates were run through a genuine, pre-registered, held-out OOS confirmatory protocol — **CONFIRMATORY** in structure, even though both ultimately failed.

## 13. Research universe gaps

Full detail: `reports/phase34_research_gaps.csv`, ranked 1-8. Top four: New York session breakout/momentum, London/NY overlap continuation, multi-timeframe trend (informed by the USDCAD lesson), non-JPY volatility expansion (informed by the XAUUSD lesson) — these four became the Phase 35 search map (§18).

## 14. New York session analysis

Per Part 15's explicit questions: **historical NY opportunity** is plausible but untested by this project directly (NY hosts major US data releases and the equity open, structurally distinct from Asian/London mechanics already tested). **Strategy families that naturally fit NY**: opening-range breakout, momentum, cross-session continuation. **Overlap risk with existing AMR**: low — the current AMR family is entirely force-flat by 07:00 server, hours before NY opens, so there is no mechanical time-of-day overlap. **Data sufficiency**: good (same MT5 feed already validated in Phase 33). **Conclusion: YES, this should be a HIGH priority for Phase 35** — not because NY is assumed valuable merely by being absent (per the explicit instruction against that reasoning), but because it is the largest, most direct, and most feasible untested gap this audit could identify with concrete supporting reasoning (session-mechanics distinctness, zero AMR overlap, adequate data).

## 15. Mechanism gap analysis

Per Part 16's ranking (evidence + feasibility, not expected profitability): **most under-represented and most researchable**, in order: (1) trend-following/multi-timeframe trend, (2) breakout/opening-range breakout, (3) momentum, (4) volatility expansion. **Least researchable given current evidence**: market structure continuation (highest overfitting risk in the taxonomy, per §5) and pullback continuation (moderate-high implementation complexity without a specific diagnosed motivation the way trend and volatility-expansion now have).

## 16. HIGH-volatility gap analysis

Per Part 17's question — **the answer is D: some combination, weighted toward A/C.** Requiring *positive* HIGH-vol performance (A) is achievable (demonstrated by XAUUSD's candidate and by GBPUSD Monday's live behavior) and should remain the primary aim for any Priority-1-motivated candidate (volatility expansion, momentum+regime families). But given how sample-constrained the HIGH-vol gate proved to be for USDCAD (only 5 trades, UNKNOWN), **candidates from families not explicitly volatility-targeting (e.g. the NY-session and cross-session-continuation hypotheses) should be permitted to qualify on strong drawdown-correlation evidence (C) even if their HIGH-vol classification remains UNKNOWN or merely NEUTRAL (B)** — this is a search-strategy recommendation about how Phase 35 allocates its hypotheses across families, not a change to any individual gate's own bar.

## 17. AUDUSD Monday LONG assessment

Not modified, not re-backtested. It remains **PROMISING / PARTIAL MATCH**: it satisfies Priority 1 (HIGH-vol compatibility — its own best-evidenced characteristic, per Phase 32) and Priority 4 (non-JPY) and Priority 3 (calendar-drift is a genuinely different mechanism from AMR), but fails Priority 2 (0.29 correlation to control, above both the control's own 0.192 internal average and above the drawdown-correlation standard this phase's audit reaffirms as important) and Priority 5 (Monday-00:00-server entry does not fill the session gap). **Additional evidence that would be required to move it further**: an independent confirmatory OOS test on data outside its already-inspected window (per Phase 30's own multiple-testing note), and — given this phase's finding that drawdown correlation is hard to measure on thin overlap — a longer observation period specifically to accumulate more control-drawdown-day overlap. **It should remain on the watchlist, unchanged, pending that evidence — not promoted and not discarded.**

## 18. Phase 35 search map

Full detail: `reports/phase34_phase35_search_map.csv` and `reports/phase34_phase35_recommendation.md`. Four families, five hypotheses total: (1) NY-open range breakout, (2) NY-session momentum, (3) London/NY overlap cross-session continuation, (4) multi-timeframe trend (direct, differently-designed retest motivated by the USDCAD lesson), (5) ATR-scaled volatility expansion (direct, differently-designed retest motivated by the XAUUSD lesson, deliberately avoiding gold).

## 19. Limitations

- This audit is a synthesis of prior phases' own already-disclosed limitations (data source, sample sizes, correlation measurement constraints) — it does not independently re-verify every underlying number beyond the reproducibility checks in §Part 1 (all passed).
- The strategy-family taxonomy (§5) is a reference classification, not empirically validated — its "likely correlation" and "likely HIGH-vol behaviour" columns are qualitative, evidence-informed judgments, not backtested figures, and are labeled as such.
- The multiple-testing assessment (§12) concludes the project has not yet over-searched this specific direction, but this is itself a judgment call made with only 3 data points (family-level attempts) — it should be revisited again after Phase 35 adds up to 5 more.

## 20. Final verdict

### Answers (Part 24)

1. **Did Phase 33 fail because the candidates were weak?** Partially — both had genuine, diagnosable implementation weaknesses (§6/§7), not evidence of "no edge exists."
2. **Did Phase 33 fail because the research universe was too narrow?** **Yes, primarily** — 12.5% family coverage, 0% direct NY-session coverage, is not enough to conclude the universe lacks a candidate.
3. **Is there evidence the Phase 32 target profile is wrong?** No (§9) — every requirement remains WELL-SUPPORTED or USEFUL BUT LIMITED; none is contradicted by Phase 33's results.
4. **Which Phase 33 gates were genuinely useful?** OOS sub-half consistency and parameter-perturbation robustness — both directly caught real, severe problems (§10).
5. **Which gates are currently sample-sensitive?** HIGH-volatility classification and drawdown-correlation — both correctly designed, but their reliability should be explicitly preconditioned on minimum trade counts going forward (§10).
6. **Which mechanisms are under-researched?** Trend-following, breakout/opening-range, momentum, volatility expansion — ranked in that order by researchability (§15).
7. **Which sessions are under-researched?** New York, and the London/NY overlap specifically (§14) — the largest gap of any dimension audited.
8. **Is HIGH-volatility compatibility still the correct top priority?** Yes — unchanged and unchallenged by this phase's evidence.
9. **Is drawdown correlation still the correct second priority?** Yes in principle, though its *measurement* needs a longer observation window or a supplementary methodology to overcome thin-overlap sample constraints (§10/§16).
10. **What should Phase 35 test?** The five hypotheses in `reports/phase34_phase35_search_map.csv` (§18).

### Classification: **B. PHASE 33 SEARCH WAS TOO NARROW** (primary), with a secondary, narrower finding under **D. METHODOLOGY NEEDS REVISION BEFORE MORE SEARCH** limited specifically to the HIGH-vol/drawdown-correlation gates' sample-size preconditions (§10) — **not** a revision of the acceptance bar itself, and **not** classification A or C, which the evidence does not support.

---

## Safety check confirmation

Six live strategies unchanged · no parameters/risk/5ers configuration changed · no candidate deployed or optimized · AUDUSD Monday LONG untouched · Phase 33 preregistration untouched (single commit, `8bcd30e`, never edited) · Phase 33 acceptance bar not lowered anywhere in this document · no new live strategy created · no production files modified · research validator passed on both control inputs · Phase 31/32 control reproduction confirmed exact · Phase 33 results reproduced from its own committed CSVs, not re-derived · every failure recorded in full (`phase34_failure_modes.csv`) · no failed candidate rescued · no future leakage (all figures reused from already-completed, chronologically-disciplined prior phases) · exploratory (Phase 30) vs. confirmatory (Phase 33) evidence explicitly separated (§12).

---

*No live trading change authorized. Phase 35 will perform the next preregistered strategy search using this phase's research map.*
