# Phase 34 — Failure Analysis: XAUUSD and USDCAD (Phase 33 Candidates)

**Diagnostic only. Neither candidate is rescued, modified, or re-tested with different parameters in this document.** Full failure-mode matrix: `reports/phase34_failure_modes.csv`. Full sample-size context: `reports/phase34_sample_size_analysis.csv`.

---

## XAUUSD_LONDON_VOL_EXPANSION

**Was XAUUSD fundamentally unsuitable, or did this specific implementation fail? — This specific implementation failed; the evidence does not support ruling out the volatility-contraction-to-expansion family or XAUUSD as an instrument.**

Working through Part 6's checklist:

- **A. Poor hypothesis?** No — the underlying hypothesis (gold's real directional moves cluster around London/NY on macro repricing following low-vol periods) is well-motivated, pre-existing in this project's own research backlog (`PROJECT_REPORT.md` §4), and the candidate *did* clear Gate 1 (OOS PF 1.185, cost-robust to 2x) — the hypothesis produced a real, if unstable, edge.
- **B. Poor parameter robustness?** Partially — the ±20% threshold perturbation moved OOS expectancy from +0.117R (baseline) to +0.045R (−20%) to −0.022R (+20%): a real degradation, though a smaller, less dramatic swing than USDCAD's full reversal.
- **C. Regime instability?** Cannot be fully separated from D given this data — 100% of the OOS sample fell in the HIGH-ATR tercile, so there's no within-OOS LOW/NORMAL comparison to test whether the edge is *regime*-unstable specifically, as opposed to simply time-unstable (D).
- **D. OOS instability?** **Yes — the clearest, best-evidenced failure.** OOS first-half expectancy −0.071R, second-half +0.305R. This is a large, unambiguous swing on an adequately-powered 114-trade OOS base (§Sample Size).
- **E. Portfolio correlation?** **Yes — an independent, second failure.** Normal-day correlation to the control was −0.151 (genuinely low), but drawdown-day correlation was +0.111 — a 0.262 swing, exceeding the pre-registered 0.15 threshold. **This is the more structurally interesting failure**: gold is a well-known macro hedge/risk-off instrument, so it plausibly co-moves with a JPY-heavy portfolio's own risk-off drawdown episodes for reasons that have nothing to do with either instrument's currency composition — a mechanism distinct from, and not resolved by, XAUUSD being non-JPY.
- **F. Insufficient sample?** Only for the drawdown-correlation gate specifically (11 overlapping days) and the regime comparison (no LOW/NORMAL trades in OOS) — the OOS-edge and parameter-sensitivity findings themselves rest on an adequate 114-trade sample.
- **G. Structural characteristic of XAUUSD?** **Plausible for the drawdown-correlation failure specifically** (gold's macro role), but not established for the OOS-instability finding, which is more consistent with an implementation choice (see below).
- **H. Unknown?** Ruled out — enough evidence exists to assign D and E with HIGH confidence.

**Most likely mechanical explanation for the OOS instability (D):** the candidate uses a **fixed 2.0x TP multiplier** regardless of how large the actual post-contraction volatility expansion turns out to be. If the OOS window's second half happened to contain more genuinely large expansions (which a fixed 2.0x target captures efficiently) while the first half contained smaller, choppier ones (which a fixed target handles poorly), that alone would produce exactly this kind of split without requiring any change in the underlying hypothesis's validity. **This is a design lesson, not a rejection of gold or the family** — see `reports/phase34_phase35_search_map.csv` Priority 4 for the specific, falsifiable follow-up this suggests (an ATR-scaled rather than fixed target).

---

## USDCAD_MOMENTUM_CONTINUATION

**Was USDCAD fundamentally unsuitable, or did this specific implementation fail? — This specific implementation failed on parameter design, and the evidence points to a specific, identifiable weakness rather than a structural rejection of trend/momentum continuation as a family.**

**Central finding, per Part 7's explicit focus: the ±20% efficiency-ratio threshold perturbation.**

| Perturbation | Threshold | OOS trades | OOS expectancy R |
|---|---|---|---|
| −20% | 0.28 | 67 | **+0.242** |
| Baseline | 0.35 | 57 | +0.155 |
| +20% | 0.42 | 49 | **−0.260** |

**Which parameter caused the instability?** The single efficiency-ratio threshold (0.35) that gates entry — this is the *only* parameter perturbed, and it alone produced a full sign reversal.

**Does the edge exist across a plateau, or does the baseline sit on a narrow peak?** **A narrow peak, not a plateau.** Both the −20% and the baseline directions are positive (though the *magnitude* actually improves going lower, from +0.155R to +0.242R as the threshold loosens), while the +20% direction reverses entirely. This asymmetric, non-monotonic-in-a-favorable-way pattern — a small increase in selectivity (fewer, "cleaner" signals by the threshold's own logic) makes performance dramatically *worse*, not better — is the opposite of what a genuinely robust momentum filter should show (a well-behaved momentum filter should generally perform *better or flat*, not worse, as its confirmation threshold tightens). **This specific behavior is itself informative: it suggests the 0.35 threshold is not cleanly separating "genuine trend" from "noise" in the way the hypothesis assumes** — trades admitted at 0.35-0.42 efficiency ratio appear to be a meaningfully different, worse population than trades admitted at 0.28-0.35, which is not what the underlying economic hypothesis (efficiency ratio as a trend-confirmation signal) predicts.

**Is the strategy structurally fragile, or could a different implementation of the same family be viable?** **The fragility is diagnosed as implementation-specific — a single H4 timeframe asking one threshold to do all the regime-discrimination work.** This project's own currently-live strategies (GBPJPY/EURJPY/AUDJPY/CADJPY AMR, CADJPY ARB, GBPUSD Monday) were all explicitly validated for a **"broad plateau"** in their key parameters (`reports/current_6_strategy_revalidation.csv`'s `parameter_stability_status` column, every one of the six) before being trusted — USDCAD's momentum candidate was never tested against that same bar until this ±20% check, and it clearly fails it. **A multi-timeframe design (a slower H4/D1 trend filter combined with a less consequential H1 entry trigger, so no single parameter carries this much weight) is a specific, falsifiable next step** — see `reports/phase34_phase35_search_map.csv` Priority 3.

---

## Cross-candidate comparison

| Dimension | XAUUSD | USDCAD |
|---|---|---|
| OOS sub-half swing | −0.071R → +0.305R | −0.091R → +0.393R |
| Parameter perturbation swing | +0.045R → +0.117R → −0.022R (mild) | +0.242R → +0.155R → −0.260R (severe, full reversal) |
| Drawdown correlation | FAILED (independently) | Insufficient evidence (n=5) |
| Most likely root cause | Fixed exit target not adapting to expansion size | Single narrow threshold carrying all the discrimination burden |
| Family-level verdict | Not rejected — a specific design fix is identified | Not rejected — a specific design fix is identified |

**Both failures are best characterized as implementation-specific design choices that a differently-constructed candidate in the same family could plausibly avoid — not as evidence that gold, USDCAD, volatility expansion, or trend/momentum continuation are unsuitable per se.** This conclusion is stated explicitly because it is the load-bearing finding for `reports/phase34_phase35_search_map.csv`'s recommendation to retest both families with targeted design changes, rather than abandoning them.

---

*No candidate rescued, modified, or re-tested with different parameters in this document. Design implications carried forward only as pre-registration input for a future Phase 35, never implemented here.*
