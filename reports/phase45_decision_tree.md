# Phase 45 — Decision Tree

Built from the evidence assembled in this phase (`reports/phase45_*.csv`), not a template.

```
CURRENT STATE
Portfolio: PF 1.211, total R 194.1 over 2,712 historical trades (2023-08 to 2026-08)
Live post-demotion (current-6 only): 19 closed trades, R -4.32
Effective diversification: ~3.1 of 6 nominal strategies (avg pairwise corr 0.192)
Phase39: FX-technical ceiling reached for undifferentiated search (unchanged by 40-44)
Phase44: NO PORTFOLIO CONTROL JUSTIFIED (4 tested, all rejected)
        |
        v
Is the live post-demotion sample sufficient to draw a conclusion?
        |
       NO (n=19, bootstrap-classified UNUSUAL but not extreme -- 9.4th percentile
           of a 10,000-draw block bootstrap of the historical trade-order R;
           outside the 10th-90th "expected variation" band but not below the
           2nd-5th percentile that would flag a statistically notable event)
        |
        v
Does ANY individual strategy show a clear deterioration signal at the
strategy level?
        |
       NO for the 2 strategies with n>=5 post-demotion trades (AUDJPY_AMR:
       CONSISTENT; the other 4 strategies have n=2-4 post-demotion trades,
       explicitly INSUFFICIENT SAMPLE -- not classified either way)
        |
        v
Does the historical reconstruction itself show strong, regime-robust
support for the current-6 portfolio?
        |
       PARTIALLY: historical edge is real (PF 1.211, positive in aggregate)
       but ROBUSTNESS evidence is WEAK (none of the 6 live strategies has ever
       been subjected to the +/-20% parameter or cost-stress framework used
       for every Phase33+ candidate) and STRATEGY INDEPENDENCE is WEAK
       (effective N ~3.1 of 6, JPY/AMR concentration is structural per Phase41/42)
        |
        v
Has Phase 44 (or any other phase) identified a validated portfolio-level
fix for the diversification/robustness weaknesses?
        |
       NO -- Phase44 tested 4 frozen controls, all REJECTED (the most
       attractive one was disqualified for suppressing 61% historical
       winners and inverting in the most recent regime)
        |
        v
==================================================================
CONCLUSION: Evidence does not support either "the portfolio is proven
robust" (historical edge is real but formal robustness/independence
evidence is weak, and live sample is too small) OR "the portfolio has
deteriorated" (no strategy shows a clear deterioration signal at an
adequate sample size; the one strategy with an adequate post-demotion
sample, AUDJPY_AMR, is CONSISTENT with its historical expectancy).
==================================================================
        |
        v
RECOMMENDED PATH: E. INSUFFICIENT EVIDENCE -- CONTINUE OBSERVATION,
paired with two concrete, low-cost, high-value actions that do NOT
require waiting:
  1. Continue live validation (already running, zero incremental cost)
  2. Run the formal parameter/cost-stress robustness battery (already
     built, used for every Phase33+ candidate) against the current-6
     strategies for the FIRST time -- closes a real, disclosed,
     immediately-actionable evidence gap without waiting for more live data
        |
        v
DO NOT: restart undifferentiated FX-technical strategy search (Phase39
ceiling, reconfirmed by Phase40); rescue any rejected candidate;
optimize a portfolio control against the same historical sample already
used in Phase41-44; change live risk, pause, or remove any strategy
without the minimum evidence specified in phase45_future_requirements.csv
```

## Reading the tree

Every branch above traces to a specific committed artifact from this phase or a prior one — no branch was hard-coded from a template. The two "NO" branches at the top (live sample insufficient; no clear per-strategy deterioration) combined with the two "PARTIALLY/WEAK" findings in the middle (historical edge real but under-tested; no validated fix available) are what produce the **E. INSUFFICIENT EVIDENCE — CONTINUE OBSERVATION** classification, not a default or a hedge — it is the specific evidence-supported endpoint of this specific tree.
