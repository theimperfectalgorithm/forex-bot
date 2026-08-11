# Volatility Regime × Existing-Strategy Performance — Diagnostics

**Experiments:** EXP-066 through EXP-076, `experiments/experiments.csv`.
**Script:** `src/phase20_volatility_regime_diagnostics.py`. **Full log:** `reports/phase20_diagnostics_log.txt`. **Data:** `data/phase20_trades.csv` (2,993 reconstructed trades, all 8 strategies).

**This is diagnostic/observational research only. No strategy was
created, optimized, or modified. No filter was added. The demo account
was not touched.** All 8 strategies are analyzed independently, with a
frozen, pre-established volatility measure and frozen regime bins.

## 1. Data and methodology

Each of the 8 live demo strategies was reconstructed from its exact
frozen live parameters (unchanged reconstruction methodology from prior
phases): `signals_arb_p` for ARB, `signals_amr_v` for AMR, `signals_monday`
for Monday Drift, run through this project's standard `run_sim` engine.
2,993 trades reconstructed: GBPJPY ARB (193), CADJPY ARB (192), XAUUSD
ARB (256), GBPJPY AMR (403), EURJPY AMR (712), AUDJPY AMR (652), CADJPY
AMR (598), GBPUSD Monday Drift (154).

## 2. Entry-time information validation

For every trade, ATR percentile is computed from `windowed_atr(14, 66)`
(this project's standard Wilder-ATR implementation, unchanged since
phase 14) evaluated at the trade's own entry bar. `windowed_atr`'s own
contract guarantees this is exactly the value the live strategy class
would compute if that bar were the most recent closed bar — genuinely
available at entry, no future candle involved. MFE/MAE are computed
post-hoc from the bars strictly between each trade's own entry and exit
timestamps (diagnostic only; does not affect any trade decision).

## 3. Primary volatility definition (frozen, Part 2)

ATR percentile = `rank_pct(windowed_atr(14, 66))`, reused unchanged from
phase 14/16/17/19 — this is the same measure phase 19 found to be more
predictive of NY volatility than the London-range construct. No ATR
lookback or percentile threshold was searched here.

**Fixed regimes (frozen before any results were examined):** LOW
[0,25), NORMAL-LOW [25,50), NORMAL-HIGH [50,75), HIGH [75,100].
**Minimum sample per judged cell: 20 trades** — smaller cells are
explicitly flagged insufficient rather than interpreted.

---

## 4-6. ARB family

### GBPJPY ARB

| regime | n | win rate | PF | expectancy | max DD |
|---|---|---|---|---|---|
| LOW | 59 | 42.4% | 1.10 | +33.0 | -5,719 |
| NORMAL-LOW | 44 | 56.8% | 1.93 | +209.1 | -2,609 |
| NORMAL-HIGH | 45 | 60.0% | **2.25** | **+244.5** | -1,490 |
| HIGH | 44 | 38.6% | **0.87** | **-38.2** | -5,043 |

**Inverted-U shape**: performance peaks in the middle regimes and is
worst at *both* extremes, with HIGH volatility the clear loser (only
losing regime). Quintile bins confirm this (Q4 best at +325, Q5 worst
at -33) — not a monotonic decline, but a real, non-trivial pattern.
Year consistency: only 2024 has enough trades in both HIGH and
LOW+NORMAL-LOW to compare — that year confirms the direction (HIGH
expectancy -131.7 vs LOW+NORMAL-LOW +247.0). 2023/2025/2026 are each
individually flagged insufficient. Session: single-window (LONDON), as
expected — not a filter, just how ARB trades.

### CADJPY ARB

| regime | n | win rate | PF | expectancy | max DD |
|---|---|---|---|---|---|
| LOW | 48 | 43.8% | 1.10 | +30.4 | -4,107 |
| NORMAL-LOW | 48 | 54.2% | 1.50 | +121.2 | -2,457 |
| NORMAL-HIGH | 58 | 58.6% | **1.79** | **+166.8** | -2,731 |
| HIGH | 38 | 34.2% | **0.67** | **-99.3** | -6,234 |

**Same inverted-U pattern as GBPJPY ARB** (same underlying signal
function/parameters family) — HIGH is the clear loser. 2024 is again the
only year with sufficient sample in both buckets, and again confirms the
direction (HIGH -176.3 vs LOW+NORMAL-LOW +19.5).

### XAUUSD ARB

| regime | n | win rate | PF | expectancy | max DD |
|---|---|---|---|---|---|
| LOW | 72 | 52.8% | 1.46 | +113.5 | -2,532 |
| NORMAL-LOW | 70 | 40.0% | **0.86** | **-40.3** | -5,524 |
| NORMAL-HIGH | 57 | 52.6% | 1.50 | +114.6 | -2,913 |
| HIGH | 56 | 57.1% | 1.52 | +116.8 | -4,907 |

**A different, contradictory pattern** from its own ARB siblings — HIGH
volatility is XAUUSD ARB's *best* regime, not its worst, and NORMAL-LOW
(not HIGH) is the loser. Year splits are entirely unusable for XAUUSD:
every year is individually flagged insufficient, and 2026 specifically
has **zero** LOW/NORMAL-LOW trades at all — meaning XAUUSD's own realized
volatility has been persistently elevated this year, an incidental but
notable observation. No coherent gradient here.

---

## 7-10. AMR family

### GBPJPY AMR

| regime | n | win rate | PF | expectancy |
|---|---|---|---|---|
| LOW | 111 | **74.8%** | **1.92** | +66.8 |
| NORMAL-LOW | 122 | 67.2% | 1.40 | +36.4 |
| NORMAL-HIGH | 105 | 62.9% | 1.22 | +22.5 |
| HIGH | 65 | 61.5% | **1.19** | +19.9 |

**Clean monotonic decline** in both win rate and PF from LOW to HIGH.
Year check: 2024 confirms (HIGH -27.5 vs LOW+NORMAL-LOW +28.9), but
**2025 does not** (HIGH +81.0 vs LOW+NORMAL-LOW +73.3, essentially the
same, slightly favoring HIGH) — mixed year evidence despite a clean
pooled gradient.

### EURJPY AMR

| regime | n | win rate | PF | expectancy |
|---|---|---|---|---|
| LOW | 235 | 68.9% | 1.04 | +3.7 |
| NORMAL-LOW | 224 | **71.9%** | **1.35** | **+27.8** |
| NORMAL-HIGH | 156 | 68.6% | 1.22 | +18.7 |
| HIGH | 97 | 62.9% | 1.01 | +1.4 |

**Inverted-U, not monotonic** — LOW itself is anomalously weak (PF 1.04,
barely profitable), NORMAL-LOW is the peak, and HIGH is again the
weakest tail. Year check: 2024 confirms (HIGH -55.7 losing vs
LOW+NORMAL-LOW -1.0 near breakeven), **2025 does not** (HIGH +31.2 vs
LOW+NORMAL-LOW +23.9, HIGH slightly better) — mixed, like GBPJPY AMR.

### AUDJPY AMR

| regime | n | win rate | PF | expectancy |
|---|---|---|---|---|
| LOW | 229 | **75.1%** | **1.35** | +26.9 |
| NORMAL-LOW | 159 | 73.0% | 1.33 | +25.8 |
| NORMAL-HIGH | 150 | 65.3% | 1.00 | -0.3 |
| HIGH | 114 | 59.6% | **0.85** | **-17.3** |

**Clean monotonic decline, and the only strategy in this entire study
where the pooled regime relationship is confirmed in every single
testable year:** 2024 (HIGH -47.4 vs LOW+NORMAL-LOW +24.7), 2025
(HIGH +5.6 vs LOW+NORMAL-LOW +28.3), 2026 (HIGH -11.6 vs LOW+NORMAL-LOW
+4.6) — **3 of 3 available years agree.** HIGH-volatility-regime AMR
trades are net losing (expectancy -17.3, PF 0.85).

### CADJPY AMR

| regime | n | win rate | PF | expectancy |
|---|---|---|---|---|
| LOW | 133 | **81.2%** | **1.85** | +47.3 |
| NORMAL-LOW | 177 | 67.8% | 1.02 | +1.9 |
| NORMAL-HIGH | 165 | 67.3% | 1.07 | +5.9 |
| HIGH | 123 | 56.9% | **0.76** | **-26.9** |

**Clean monotonic decline, large swing** — LOW win rate (81.2%) is the
highest of any regime cell in the whole study, HIGH is net losing.
2024 and 2025 both confirm the direction (2024: HIGH -20.5 vs LOW+
NORMAL-LOW +2.6; 2025: HIGH -18.1 vs LOW+NORMAL-LOW +14.5) — the two
testable years agree, though only 2 of 4 years have sufficient sample
(2023/2026 flagged insufficient).

---

## 11. GBPUSD Monday Drift

| regime | n | win rate | PF | expectancy |
|---|---|---|---|---|
| LOW | 19 | — | — | **insufficient (n=19, need ≥20)** |
| NORMAL-LOW | 39 | 61.5% | 2.46 | +40.4 |
| NORMAL-HIGH | 42 | 71.4% | **3.00** | **+50.4** |
| HIGH | 54 | 53.7% | 1.34 | +14.8 |

Different mechanism (weekly frequency), analyzed on its own terms as
instructed — volatility should not necessarily matter the same way here,
and the data agrees: the pattern is not the ARB/AMR-style extremes-are-
worse or gradient story. NORMAL-HIGH, not an extreme, is the best
regime. **Every year-level split is individually flagged insufficient**
— at ~1 trade/week, this strategy simply does not generate enough
volume for a reliable year-by-regime read, and no conclusion is forced
from it.

---

## 12. Year consistency (summary across strategies)

Only AUDJPY AMR achieves full cross-year confirmation (3/3 testable
years agree). CADJPY AMR achieves 2/2. GBPJPY ARB and CADJPY ARB each
have only 1 testable year (2024), which confirms the pooled direction
but cannot establish cross-year stability. GBPJPY AMR and EURJPY AMR
have 2 testable years each, and **disagree** between them (2024 confirms,
2025 contradicts) — a real inconsistency, not glossed over. XAUUSD ARB
and Monday Drift have no usable year-level splits at all.

## 13. Session consistency

Each strategy trades in exactly one session window by construction (ARB:
LONDON, AMR: ASIAN, Monday: ASIAN at Monday 00:00) — this is not a filter
being imposed, it's simply how these strategies already operate, so a
within-strategy session comparison isn't applicable. Noted as requested,
not treated as a gap.

## 14. Portfolio regime analysis (Part 11)

| regime | n days | mean daily P&L | worst day | max DD |
|---|---|---|---|---|
| LOW | 200 | +$117.77 | -$2,773 | -$6,561 |
| NORMAL-LOW | 270 | **+$219.27** | -$2,306 | -$6,085 |
| NORMAL-HIGH | 210 | **+$16.21** | -$2,715 | **-$18,640** |
| HIGH | 99 | +$95.88 | -$2,944 | -$6,731 |

Portfolio-day regime is the mean entry-time ATR percentile across all
trades opened that day, across all 8 strategies (not future-looking —
each day's regime label uses only that day's own entries). **The
striking result is NOT a simple "risk rises with volatility" story**:
mean daily P&L is worst in NORMAL-HIGH, not HIGH, and the portfolio's
single largest drawdown (-$18,640, roughly 3x every other regime's
worst) occurs in the NORMAL-HIGH bucket specifically. This is consistent
with the inverted-U pattern seen in individual ARB strategies (their own
weak regime is often NORMAL-LOW-to-NORMAL-HIGH transition, not purely
HIGH) compounding with AMR's decline into a portfolio-level worst zone
that isn't simply "the most volatile days."

## 15. Clustered-loss analysis (Part 12)

| regime | mean simultaneous losing strategies | % days with 2+ losers |
|---|---|---|
| LOW | 1.79 | 46.8% |
| NORMAL-LOW | 1.92 | 53.1% |
| NORMAL-HIGH | **2.38** | **63.9%** |
| HIGH | 2.11 | 56.2% |

Concentration risk rises with volatility regime but, again, **peaks at
NORMAL-HIGH rather than HIGH** — consistent with the portfolio drawdown
finding above. High volatility does appear to increase the chance of
multiple strategies losing simultaneously, but the relationship is not
a clean linear "the worse the regime, the worse the clustering" story.

## 16. Statistical / multiple-testing assessment

This diagnostic examined 8 strategies × 4 regimes × (year splits +
quintile splits) — a large number of cells. No single favorable or
unfavorable cell is being treated as a discovered edge. The classification
below explicitly weighs: pooled-bin coherence, cross-year replication
(the strictest test applied here), and whether a strategy's own siblings
(same family, same signal function) show the same pattern. AUDJPY AMR and
CADJPY AMR earn their "strong"/"stable" labels specifically because they
pass the cross-year test, not because their pooled numbers alone look
large — several other cells (e.g. XAUUSD ARB's HIGH-regime profitability)
look attractive in isolation but do not replicate across the same
family or across years, and are explicitly NOT elevated to a finding on
that basis.

## 17. Strongest relationships

1. **AUDJPY AMR** — monotonic 4-bin decline, confirmed in 3/3 testable
   years, HIGH regime is net losing (PF 0.85, expectancy -17.3).
2. **CADJPY AMR** — monotonic 4-bin decline, confirmed in 2/2 testable
   years, HIGH regime is net losing (PF 0.76, expectancy -26.9), largest
   LOW-vs-HIGH swing of any strategy studied.
3. **GBPJPY ARB / CADJPY ARB** — consistent inverted-U pattern between
   the two sibling pairs (same signal function), HIGH regime the clear
   loser in both, though cross-year confirmation is limited to a single
   testable year (2024) for each.

## 18. Weakest / contradictory relationships

1. **XAUUSD ARB** — contradicts its own ARB siblings; HIGH is its *best*
   regime, not worst. No usable year-level confirmation at all.
2. **GBPJPY AMR / EURJPY AMR** — real pooled gradients, but each
   contradicted by one of their two testable years (2025 favors HIGH in
   both cases) — real but not stable evidence.
3. **GBPUSD Monday Drift** — no coherent gradient (best regime is
   NORMAL-HIGH, not an extreme), and no year-level confirmation possible
   given trade frequency.

## 19. AMR implications

Two of the four AMR pairs (AUDJPY, CADJPY) show a genuinely strong,
cross-year-confirmed relationship between entry-time volatility and
AMR's own performance: **higher entry-time ATR percentile is associated
with worse AMR expectancy, culminating in net-losing trades in the HIGH
regime.** This is directly relevant to, and consistent with, this
project's already-standing open question about AMR (documented
elsewhere in project history: "real edge, but regime-strengthening,
unclear durability"). **Per instructions, nothing is being implemented
from this** — no filter, no threshold, no code change. This finding is
explicitly flagged as evidence that **should be considered at the
existing AMR checkpoint (~2026-08-25)**, not acted on now. The other two
AMR pairs (GBPJPY, EURJPY) show the same *pooled* direction but weaker,
year-inconsistent evidence, so the checkpoint discussion should
distinguish "AUDJPY/CADJPY: fairly solid evidence" from "GBPJPY/EURJPY:
suggestive but not confirmed."

## 20. Portfolio implications

The portfolio-level regime relationship is real but **not a simple
"volatility = risk" story** — the worst drawdown and the highest
loss-clustering both occur in the NORMAL-HIGH bucket, not HIGH. This
matters for how any future risk-sizing conversation should be framed:
naively scaling down exposure only on "HIGH" days would miss the
regime bucket that has actually produced the worst historical portfolio
outcomes in this reconstruction. No sizing or filter change is being
proposed here — this is recorded as a diagnostic observation only.

## 21. Final classifications

| Strategy | Classification |
|---|---|
| GBPJPY ARB | **C. STABLE REGIME RELATIONSHIP** |
| CADJPY ARB | **C. STABLE REGIME RELATIONSHIP** |
| XAUUSD ARB | **B. WEAK / INCONSISTENT RELATIONSHIP** |
| GBPJPY AMR | **C. STABLE REGIME RELATIONSHIP** |
| EURJPY AMR | **B. WEAK / INCONSISTENT RELATIONSHIP** |
| AUDJPY AMR | **D. STRONG REGIME RELATIONSHIP** |
| CADJPY AMR | **C. STABLE REGIME RELATIONSHIP** (borderline D — strong pooled effect, but only 2/4 years testable vs. AUDJPY's 3/4) |
| GBPUSD Monday Drift | **B. WEAK / INCONSISTENT RELATIONSHIP** |

### Portfolio classification

# **POTENTIAL REGIME DEPENDENCE**

Not "no meaningful regime effect" — the portfolio-level max-drawdown and
clustering differences across regimes are large and consistent enough to
matter. Not "clear regime dependence" — the relationship is not a clean
monotonic story (it peaks at NORMAL-HIGH, not HIGH), and it is built from
strategies with mixed individual classifications (2 strong/stable-to-strong
AMR pairs, 2 stable ARB pairs, and 3 weak/inconsistent components).
**This is descriptive only and is not being called a trading edge.**

## 22. Recommended next experiment

1. **Bring the AUDJPY AMR and CADJPY AMR findings to the existing AMR
   checkpoint (~2026-08-25)** as evidence for that scheduled discussion —
   not before, and not as an automatic trigger for a code change.
2. **Do not extend this finding to GBPJPY/EURJPY AMR or to a portfolio-
   wide volatility filter** without their own year-level confirmation —
   they do not currently meet the same bar.
3. If the AMR checkpoint decides to investigate further, the natural
   next step would be a **held-out confirmatory test** on AUDJPY/CADJPY
   AMR specifically (not a threshold search) — but that is a decision
   for the checkpoint, not something to pre-empt here.
4. The portfolio-level NORMAL-HIGH drawdown concentration (Part 14/15)
   is worth keeping in mind for any future risk-sizing discussion, but
   no sizing change is recommended from this diagnostic alone.

---

## What I did NOT do (per instructions)

- Did not add ATR filters, remove high-volatility trades, or change
  risk, stops, targets, entries, session windows, or any strategy
  parameter.
- Did not modify GBPJPY ARB, CADJPY ARB, XAUUSD ARB, GBPJPY/EURJPY/
  AUDJPY/CADJPY AMR, or GBPUSD Monday Drift.
- Did not change the demo account or any live configuration.
- Did not treat any single favorable or unfavorable cell as a discovered
  trading edge.
- Did not change the existing AMR checkpoint date or process — the
  AUDJPY/CADJPY findings are flagged as input to that checkpoint, not a
  substitute for it.
