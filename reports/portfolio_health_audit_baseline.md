# Portfolio Health Audit — Historical Baseline (8 Live Strategies)

**Purpose:** establish the historical/backtest baseline for GBPJPY ARB,
CADJPY ARB, XAUUSD ARB, GBPJPY AMR, EURJPY AMR, AUDJPY AMR, CADJPY AMR,
and GBPUSD Monday Drift, against which live demo/prop performance can be
compared. **No strategy logic was modified, no optimization was run, no
new research was conducted.** Every number below is pulled from an
existing artifact and cited; where no artifact exists, the field is
marked **NOT AVAILABLE**, not estimated.

## Sources used

| Source | What it provides | Covers |
|---|---|---|
| `PROJECT_REPORT.md` §3 (table, lines 374-383) | Original discovery-time IS/OOS PF, some DD/profitable-months | All 8, IS=Jul23-Jun25 / OOS=Jul25-Jun26 |
| `reports/volatility_regime_strategy_diagnostics.md` + `data/phase20_trades.csv` (EXP-066 to EXP-075) | Full-history (2023-2026) reconstructed trade-level backtest using the exact frozen live parameters: trade count, win rate, PF, expectancy, R, max drawdown, losing streaks, year-by-year, BUY/SELL split, regime dependence | All 8, pooled full history (NOT the same window as the original IS/OOS split above) |
| `reports/amr_regime_mechanism.md` + `data/phase21_amr_trades.csv` (EXP-076 to EXP-081) | Deeper AMR-only mechanism analysis, BUY/SELL split | 4 AMR pairs only |
| `reports/audjpy_amr_confirmatory_filter.md` + `reports/audjpy_amr_final_validation.md` (EXP-082 to EXP-089) | Genuine chronological TRAIN/VALIDATION/OOS split with walk-forward and cost stress | AUDJPY AMR (baseline/original) only |

**Important distinction, stated once here rather than repeated 8 times:**
the "original discovery OOS" (PROJECT_REPORT §3) and the "full-history
reconstruction" (phase 20/21) are **not the same period or the same
computation** — the original used a fixed Jul23–Jun25/Jul25–Jun26 IS/OOS
split with no reported trade count, win rate, or R-multiples; the
phase-20/21 reconstruction runs the identical frozen signal logic across
the full available history (pooled, not IS/OOS split) and does report
those figures. Both are reported below, clearly labeled, never blended
into one number.

---

## 1. GBPJPY ARB

- **Total historical trades (full-history reconstruction, 2023-08-01 to 2026-07-29):** 193 — EXP-066
- **Original discovery IS/OOS:** IS PF 1.45 / DD 4.7% / 62.5% profitable months; **OOS PF 1.19, +$3.7k** — PROJECT_REPORT.md §3, row 1
- **IS/OOS trade counts, IS win rate, OOS win rate, OOS expectancy, OOS total R (original split):** NOT AVAILABLE (not persisted as separate artifacts)
- **Full-history win rate:** 48.7% — EXP-066 / `data/phase20_trades.csv`
- **Full-history PF:** 1.390 — EXP-066
- **Full-history expectancy:** +$103.42/trade — EXP-066
- **Full-history total R:** +37.14 — EXP-066
- **Maximum drawdown (full-history):** **-$6,732.75** — EXP-066
- **Maximum consecutive losing trades:** **8** — EXP-066
- **Average consecutive losing trades:** 2.48 — EXP-066
- **Worst historical losing streak:** 8 consecutive losses (same as max above; no separate "worst" distinct from max is defined in the artifact) — EXP-066
- **Year-by-year (full-history reconstruction):**

| Year | n | Win rate | PF | Expectancy |
|---|---|---|---|---|
| 2023 | 26 | 46.2% | 1.181 | +$50.21 |
| 2024 | 62 | 41.9% | 1.166 | +$41.34 |
| 2025 | 68 | 60.3% | 2.223 | +$266.11 |
| 2026 | 37 | 40.5% | 0.854 | -$54.19 |

  — EXP-066 / `data/phase20_trades.csv`

- **Pair-specific:** N/A — single-pair strategy.
- **BUY vs SELL (full-history):** BUY n=109, win rate 51.4%, PF 1.626, expectancy +$155.89; SELL n=84, win rate 45.2%, PF 1.124, expectancy +$35.32 — EXP-066
- **Known regime dependency:** Inverted-U — best in NORMAL-LOW/NORMAL-HIGH ATR-percentile regimes, HIGH regime is the only losing regime (PF 0.87, expectancy -$38.2). Classification: **C. STABLE REGIME RELATIONSHIP**, but cross-year confirmation limited to 2024 only (2023/2025/2026 individually insufficient sample) — EXP-066, `reports/volatility_regime_strategy_diagnostics.md` §4, §17, §21
- **Cost-stress results (1.5x/2x spread, slippage):** NOT AVAILABLE
- **Walk-forward results (rolling multi-fold):** NOT AVAILABLE
- **Current research classification/status:** C. STABLE REGIME RELATIONSHIP (volatility diagnostic only; not re-validated, not optimized) — EXP-066, `reports/volatility_regime_strategy_diagnostics.md` §21

---

## 2. CADJPY ARB

- **Total historical trades (full-history, 2023-08-02 to 2026-07-23):** 192 — EXP-067
- **Original discovery IS/OOS:** IS PF 1.15; **OOS PF 1.38, +$6.4k** — PROJECT_REPORT.md §3, row 2
- **IS/OOS trade counts, IS win rate, OOS win rate, OOS expectancy, OOS total R (original split):** NOT AVAILABLE
- **Full-history win rate:** 49.0% — EXP-067
- **Full-history PF:** 1.263 — EXP-067
- **Full-history expectancy:** +$68.63/trade — EXP-067
- **Full-history total R:** +25.46 — EXP-067
- **Maximum drawdown (full-history):** **-$7,933.62** — EXP-067
- **Maximum consecutive losing trades:** **10** — EXP-067
- **Average consecutive losing trades:** 2.18 — EXP-067
- **Worst historical losing streak:** 10 consecutive losses — EXP-067
- **Year-by-year:**

| Year | n | Win rate | PF | Expectancy |
|---|---|---|---|---|
| 2023 | 25 | 56.0% | 1.622 | +$148.93 |
| 2024 | 64 | 40.6% | 0.931 | -$20.15 |
| 2025 | 75 | 52.0% | 1.408 | +$97.53 |
| 2026 | 28 | 53.6% | 1.449 | +$122.47 |

  — EXP-067

- **Pair-specific:** N/A — single-pair.
- **BUY vs SELL:** BUY n=111, win rate 51.4%, PF 1.391, expectancy +$95.98; SELL n=81, win rate 45.7%, PF 1.110, expectancy +$31.16 — EXP-067
- **Known regime dependency:** Same inverted-U pattern as GBPJPY ARB (shared signal function) — HIGH regime is the clear loser (PF 0.67, expectancy -$99.3). Classification: **C. STABLE REGIME RELATIONSHIP**, again only 2024 individually testable for year-consistency — EXP-067, `reports/volatility_regime_strategy_diagnostics.md` §4, §17, §21
- **Cost-stress results:** NOT AVAILABLE
- **Walk-forward results:** NOT AVAILABLE
- **Current research classification/status:** C. STABLE REGIME RELATIONSHIP — EXP-067

---

## 3. XAUUSD ARB

- **Total historical trades (full-history, 2023-08-01 to 2026-08-07):** 256 — EXP-068
- **Original discovery IS/OOS:** **PROVISIONAL** — IS PF 1.45 / DD 2.9%; **OOS flat (PF 1.05)** — PROJECT_REPORT.md §3, row 3 (explicitly flagged provisional in that source)
- **IS/OOS trade counts, IS win rate, OOS win rate, OOS expectancy, OOS total R (original split):** NOT AVAILABLE
- **Full-history win rate:** 50.4% — EXP-068
- **Full-history PF:** 1.300 — EXP-068
- **Full-history expectancy:** +$74.87/trade — EXP-068
- **Full-history total R:** +35.86 — EXP-068
- **Maximum drawdown (full-history):** **-$6,894.40** — EXP-068
- **Maximum consecutive losing trades:** **7** — EXP-068
- **Average consecutive losing trades:** 1.92 — EXP-068
- **Worst historical losing streak:** 7 consecutive losses — EXP-068
- **Year-by-year:**

| Year | n | Win rate | PF | Expectancy |
|---|---|---|---|---|
| 2023 | 27 | 48.1% | 1.212 | +$56.06 |
| 2024 | 98 | 46.9% | 1.169 | +$43.57 |
| 2025 | 89 | 48.3% | 1.160 | +$43.21 |
| 2026 | 42 | 64.3% | 2.265 | +$227.06 |

  — EXP-068

- **Pair-specific:** N/A — single-pair.
- **BUY vs SELL:** BUY n=154, win rate 52.6%, PF 1.427, expectancy +$102.10; SELL n=102, win rate 47.1%, PF 1.127, expectancy +$33.75 — EXP-068
- **Known regime dependency:** **Contradicts its own ARB siblings** — HIGH volatility regime is XAUUSD ARB's *best* regime (PF 1.52), not its worst; NORMAL-LOW is the loser (PF 0.86). No usable year-level confirmation (every year individually flagged insufficient; 2026 has zero LOW/NORMAL-LOW trades at all — persistently elevated realized volatility this year). Classification: **B. WEAK / INCONSISTENT RELATIONSHIP** — EXP-068, `reports/volatility_regime_strategy_diagnostics.md` §4, §18, §21
- **Cost-stress results:** NOT AVAILABLE
- **Walk-forward results:** NOT AVAILABLE
- **Current research classification/status:** B. WEAK / INCONSISTENT (volatility diagnostic); original discovery itself was already flagged PROVISIONAL — EXP-068, PROJECT_REPORT.md §3

---

## 4. GBPJPY AMR

- **Total historical trades (full-history, 2023-08-01 to 2026-08-11):** 403 — EXP-069
- **Original discovery IS/OOS:** IS PF 1.16 / 68% profitable months; **OOS PF 2.03** — PROJECT_REPORT.md §3, row 4
- **IS/OOS trade counts, IS win rate, OOS win rate, OOS expectancy, OOS total R (original split):** NOT AVAILABLE
- **Full-history win rate:** 67.2% — EXP-069
- **Full-history PF:** 1.426 — EXP-069
- **Full-history expectancy:** +$38.49/trade — EXP-069
- **Full-history total R:** +58.00 — EXP-069
- **Maximum drawdown (full-history):** **-$1,842.70** (smallest of any of the 8 strategies) — EXP-069
- **Maximum consecutive losing trades:** **5** — EXP-069
- **Average consecutive losing trades:** 1.50 — EXP-069
- **Worst historical losing streak:** 5 consecutive losses — EXP-069
- **Year-by-year:**

| Year | n | Win rate | PF | Expectancy |
|---|---|---|---|---|
| 2023 | 45 | 62.2% | 1.093 | +$9.51 |
| 2024 | 148 | 62.8% | 1.153 | +$15.46 |
| 2025 | 139 | 69.8% | 1.651 | +$53.59 |
| 2026 | 71 | 74.6% | 1.991 | +$75.32 |

  — EXP-069

- **Pair-specific:** N/A — single-pair.
- **BUY vs SELL (from AMR mechanism study):** BUY n=263, win rate 70.3%, PF 1.647, expectancy +$52.89, total R +51.95; SELL n=140, win rate 61.4%, PF 1.107, expectancy +$11.46, total R +6.05 — EXP-078, `data/phase21_amr_trades.csv`
- **Known regime dependency:** Pooled 4-bin table shows a clean monotonic decline (LOW PF 1.92 → HIGH PF 1.19), **but this does NOT survive conditioning on trend** — the volatility effect reverses sign in the HIGH-TREND tercile (+19.89, i.e. high vol becomes slightly *better* there). Phase 20's original "C. STABLE" read was **explicitly revised downward** after this deeper check. **Current classification: E. OTHER / INCONCLUSIVE** — EXP-069 (phase20), EXP-078 (phase21 revision), `reports/amr_regime_mechanism.md` §4 ("GBPJPY — the volatility effect reverses sign under high trend"), §18
- **Cost-stress results:** NOT AVAILABLE
- **Walk-forward results:** NOT AVAILABLE
- **Current research classification/status:** **E. OTHER / INCONCLUSIVE** (downgraded from phase-20's C. STABLE by phase-21's conditioning analysis) — EXP-078, EXP-081

---

## 5. EURJPY AMR

- **Total historical trades (full-history, 2023-07-31 to 2026-08-11):** 712 — EXP-070
- **Original discovery IS/OOS:** IS PF 1.10 / 60% profitable months; **OOS PF 1.47** — PROJECT_REPORT.md §3, row 5
- **IS/OOS trade counts, IS win rate, OOS win rate, OOS expectancy, OOS total R (original split):** NOT AVAILABLE
- **Full-history win rate:** 69.0% — EXP-070
- **Full-history PF:** 1.161 — EXP-070
- **Full-history expectancy:** +$14.23/trade — EXP-070
- **Full-history total R:** +39.06 — EXP-070
- **Maximum drawdown (full-history):** **-$4,472.01** — EXP-070
- **Maximum consecutive losing trades:** **4** — EXP-070
- **Average consecutive losing trades:** 1.45 — EXP-070
- **Worst historical losing streak:** 4 consecutive losses — EXP-070
- **Year-by-year:**

| Year | n | Win rate | PF | Expectancy |
|---|---|---|---|---|
| 2023 | 94 | 73.4% | 1.404 | +$30.36 |
| 2024 | 244 | 65.6% | 0.989 | -$1.09 |
| 2025 | 232 | 69.0% | 1.196 | +$16.96 |
| 2026 | 142 | 71.8% | 1.302 | +$25.40 |

  — EXP-070

- **Pair-specific:** N/A — single-pair.
- **BUY vs SELL:** BUY n=423, win rate 74.2%, PF 1.498, expectancy +$36.58, total R +59.07; SELL n=289, win rate 61.2%, PF 0.832, expectancy -$18.49, total R -20.01 — EXP-079, `data/phase21_amr_trades.csv`
- **Known regime dependency:** Inverted-U, not monotonic — LOW itself anomalously weak (PF 1.04), NORMAL-LOW is the peak, HIGH is the weakest tail. Volatility effect is **sign-unstable across trend terciles** (reverses to +31.78 in LOW-TREND, -34.82 in HIGH-TREND) — same instability signature as GBPJPY. **Classification: E. OTHER / INCONCLUSIVE**, consistent between phase 20 and phase 21 — EXP-070, EXP-079, `reports/amr_regime_mechanism.md` §4, §18
- **Cost-stress results:** NOT AVAILABLE
- **Walk-forward results:** NOT AVAILABLE
- **Current research classification/status:** E. OTHER / INCONCLUSIVE — EXP-079, EXP-081

---

## 6. AUDJPY AMR

- **Total historical trades (full-history, 2023-07-31 to 2026-08-11):** 652 — EXP-071
- **Original discovery IS/OOS:** IS PF 1.17 / 60% profitable months; **OOS PF 1.23** — PROJECT_REPORT.md §3, row 6
- **Full-history win rate:** 69.6% — EXP-071
- **Full-history PF:** 1.143 — EXP-071
- **Full-history expectancy:** +$12.66/trade — EXP-071
- **Full-history total R:** +32.21 — EXP-071
- **Maximum drawdown (full-history):** **-$2,687.24** — EXP-071
- **Maximum consecutive losing trades:** **5** — EXP-071
- **Average consecutive losing trades:** 1.43 — EXP-071
- **Worst historical losing streak:** 5 consecutive losses — EXP-071

**Chronological TRAIN/VALIDATION/OOS split (genuine, not pooled — this is the one strategy with a real held-out test), from the phase-22 confirmatory experiment's baseline/control row:**

| Period | Dates | n | Win rate | PF | Expectancy |
|---|---|---|---|---|---|
| TRAIN | 2023-07-31 → 2024-08-03 | 223 | 73.5% | 1.29 | +$22.82 |
| VALIDATION | 2024-08-03 → 2025-08-07 | 221 | 66.1% | 1.02 | +$2.22 |
| **FINAL OOS** | **2025-08-07 → 2026-08-11** | **208** | **69.2%** | **1.14** | **+$12.86** |

— EXP-082, `reports/audjpy_amr_confirmatory_filter.md` §4 (this is the original/unfiltered AUDJPY AMR baseline, not the BUY-only candidate)

- **Year-by-year (full-history):**

| Year | n | Win rate | PF | Expectancy |
|---|---|---|---|---|
| 2023 | 93 | 78.5% | 1.713 | +$45.06 |
| 2024 | 224 | 67.4% | 1.001 | +$0.13 |
| 2025 | 209 | 69.9% | 1.211 | +$18.16 |
| 2026 | 126 | 66.7% | 1.019 | +$1.89 |

  — EXP-071

- **Pair-specific:** N/A — single-pair.
- **BUY vs SELL:** BUY n=412, win rate 76.2%, PF 1.591, expectancy +$41.21, total R +64.98; **SELL n=240, win rate 58.3%, PF 0.699 (net losing), expectancy -$36.36, total R -32.75** — EXP-076, `data/phase21_amr_trades.csv`
- **Known regime dependency:** **The strongest, most robust finding across all 8 strategies.** Clean monotonic decline (LOW PF 1.35 → HIGH PF 0.85, net losing), and this relationship **survives conditioning on trend in all 3 trend terciles** (stays negative, strengthens under high trend) while trend's own effect (conditioned on volatility) is sign-unstable — the clearest evidence in this project that volatility is causally, not just coincidentally, associated with AUDJPY AMR's performance. Confirmed in **3 of 3 testable years** (2024, 2025, 2026). MAE exceeds MFE in the HIGH regime (0.97 ratio) — trades genuinely move against the position harder, not just lose more by chance. **Classification: D. STRONG REGIME RELATIONSHIP** (phase 20) / **B. VOLATILITY-DEPENDENT** (phase 21's causal-mechanism language) — EXP-071, EXP-076, `reports/volatility_regime_strategy_diagnostics.md` §21, `reports/amr_regime_mechanism.md` §18
- **Cost-stress results (baseline/original AMR, from phase 22):**

| Scenario | PF |
|---|---|
| Normal spread | 1.14 |
| 1.5x spread | 0.98 |
| **2x spread** | **0.83 (losing)** |
| 1-bar delay | 1.00 |

— EXP-082, `reports/audjpy_amr_confirmatory_filter.md` §9 (full-history, not OOS-restricted)

- **Walk-forward results (baseline, 6-month rolling windows, 11 windows, full history):** ranged from PF 1.53 (best window, Jul23-Jan24) down to PF 0.85 (worst window, Oct24-Apr25, the strategy's own historically hardest stretch); 2 of 11 windows net-losing — EXP-082, `reports/phase22_confirmatory_log.txt` Part 7 (full per-window table there)
- **Current research classification/status:** D. STRONG REGIME RELATIONSHIP / B. VOLATILITY-DEPENDENT. A frozen **BUY-only candidate** (not live) was separately tested and classified **SUPPORTED** but explicitly **NOT VALIDATED and NOT approved for deployment** (OOS bootstrap CI still crosses zero) — see EXP-082 to EXP-091 for the full candidate-evaluation chain; **the currently live AUDJPY AMR is the original, unmodified strategy**, not the BUY-only candidate.

---

## 7. CADJPY AMR

- **Total historical trades (full-history, 2023-08-01 to 2026-08-11):** 598 — EXP-072
- **Original discovery IS/OOS:** IS PF 1.10; **OOS PF 1.35** — PROJECT_REPORT.md §3, row 7
- **Full-history win rate:** 68.4% — EXP-072
- **Full-history PF:** 1.082 — EXP-072
- **Full-history expectancy:** +$7.19/trade — EXP-072
- **Full-history total R:** +17.27 — EXP-072
- **Maximum drawdown (full-history):** **-$3,945.24** — EXP-072
- **Maximum consecutive losing trades:** **6** — EXP-072
- **Average consecutive losing trades:** 1.49 — EXP-072
- **Worst historical losing streak:** 6 consecutive losses — EXP-072
- **Year-by-year:**

| Year | n | Win rate | PF | Expectancy |
|---|---|---|---|---|
| 2023 | 75 | 69.3% | 1.092 | +$7.96 |
| 2024 | 210 | 67.6% | 1.033 | +$2.99 |
| 2025 | 201 | 65.7% | 1.002 | +$0.19 |
| 2026 | 112 | 74.1% | 1.368 | +$27.13 |

  — EXP-072

- **Pair-specific:** N/A — single-pair.
- **BUY vs SELL:** BUY n=352, win rate 74.1%, PF 1.425, expectancy +$30.61, total R +42.62; **SELL n=246, win rate 60.2%, PF 0.763 (net losing), expectancy -$26.32, total R -25.35** — EXP-077, `data/phase21_amr_trades.csv`
- **Known regime dependency:** Clean monotonic decline in the pooled 4-bin table (LOW PF 1.85, highest win rate of any cell in the whole study at 81.2% → HIGH PF 0.76, net losing), confirmed in its 2 testable years (2024, 2025; 2023/2026 individually insufficient). **However, the volatility effect is trend-CONDITIONAL** — near-zero within LOW-TREND trades, strong within NORMAL/HIGH-TREND trades — an interaction, not an independent effect like AUDJPY's. **Classification: C. STABLE REGIME RELATIONSHIP** (phase 20) / **D. VOLATILITY + TREND INTERACTION** (phase 21's causal-mechanism language, more precise than "stable") — EXP-072, EXP-077, `reports/volatility_regime_strategy_diagnostics.md` §21, `reports/amr_regime_mechanism.md` §18
- **Cost-stress results:** NOT AVAILABLE
- **Walk-forward results:** NOT AVAILABLE
- **Current research classification/status:** C. STABLE / D. VOLATILITY+TREND INTERACTION — EXP-077, EXP-081. Per `reports/amr_regime_mechanism.md` §19-20, a confirmatory filter experiment was recommended for AUDJPY only, **not** CADJPY, because CADJPY's interaction structure means a simple threshold filter would be mis-specified — no confirmatory experiment has been run for CADJPY.

---

## 8. GBPUSD Monday Drift

- **Total historical trades (full-history, 2023-08-28 to 2026-08-10):** 154 — EXP-073
- **Original discovery IS/OOS:** **Strongest pass of the whole project** — IS PF 1.97 / DD 0.66% / 66.7% profitable months; **OOS PF 3.08 / DD 0.42%** — PROJECT_REPORT.md §3, row 8
- **IS/OOS trade counts, IS win rate, OOS win rate, OOS expectancy, OOS total R (original split):** NOT AVAILABLE
- **Full-history win rate:** 63.0% — EXP-073
- **Full-history PF:** 2.105 — EXP-073
- **Full-history expectancy:** +$33.46/trade — EXP-073
- **Full-history total R:** +20.14 — EXP-073
- **Maximum drawdown (full-history):** **-$559.70** (by far the smallest of all 8 strategies) — EXP-073
- **Maximum consecutive losing trades:** **4** — EXP-073
- **Average consecutive losing trades:** 1.50 — EXP-073
- **Worst historical losing streak:** 4 consecutive losses — EXP-073
- **Year-by-year:**

| Year | n | Win rate | PF | Expectancy |
|---|---|---|---|---|
| 2023 | 18 | 72.2% | 1.826 | +$25.23 |
| 2024 | 52 | 55.8% | 1.164 | +$5.95 |
| 2025 | 52 | 61.5% | 2.830 | +$47.86 |
| 2026 | 32 | 71.9% | 3.188 | +$59.39 |

  — EXP-073

- **Pair-specific:** N/A — single-pair.
- **BUY vs SELL:** BUY n=154 (100% of trades) — **long-only by design** (per `PROJECT_REPORT.md` §3: "the measured anomaly is a positive Monday drift"), SELL = N/A, not missing data — EXP-073
- **Known regime dependency:** Different mechanism (weekly frequency, ~1 trade/week) analyzed on its own terms. No coherent gradient — best regime is NORMAL-HIGH (PF 3.00), not an extreme, unlike the ARB/AMR extremes-are-worse pattern. Every year-level regime split is individually flagged insufficient sample at this trade frequency. **Classification: B. WEAK / INCONSISTENT RELATIONSHIP** — EXP-073, `reports/volatility_regime_strategy_diagnostics.md` §11, §18, §21
- **Cost-stress results:** NOT AVAILABLE
- **Walk-forward results:** NOT AVAILABLE
- **Current research classification/status:** B. WEAK / INCONSISTENT (volatility diagnostic only); original discovery was the strongest pass in the project's history and has not been contradicted by any later research — EXP-073, PROJECT_REPORT.md §3

---

## Cross-cutting notes

- **No cost-stress or rolling walk-forward evidence exists for 7 of the 8
  strategies** (everything except AUDJPY AMR's baseline, tested
  incidentally as the control in the phase-22 confirmatory experiment).
  This is a genuine gap in the historical record, not an oversight in
  this report — it reflects what was actually run at each strategy's
  original discovery time, before this project's later phases (20-22)
  introduced that level of rigor.
- **"Full-history reconstruction" (phase 20/21) figures are pooled
  across the entire 2023-2026 window**, not split IS/OOS. They should
  not be read as "the OOS result" — they answer a related but distinct
  question ("how has this exact frozen logic performed across all
  available history") using this session's own trade-level
  reconstruction, which is more granular (win rate, R, drawdown, losing
  streaks) than what the original discovery-time console output
  preserved.
- **GBPJPY AMR and EURJPY AMR were downgraded** from phase 20's initial
  "C. STABLE" regime classification to "E. OTHER / INCONCLUSIVE" after
  phase 21's deeper conditioning analysis. This report uses the
  **revised, more recent classification** as current status, consistent
  with this project's standing practice of not treating a superseded
  finding as current.
- **AUDJPY AMR's SELL leg (PF 0.699, net losing across 240 trades) and
  CADJPY AMR's SELL leg (PF 0.763, net losing across 246 trades) are
  independently, structurally net losers** across the full available
  history — this is not an artifact of the volatility-regime split, it
  shows up in the unconditional BUY-vs-SELL breakdown too.

---

## Machine-readable summary (CSV)

See `reports/portfolio_health_audit_baseline.csv` for the exact same
figures in the requested column format. To keep every row's `oos_pf`
column measuring the *same thing* (the original discovery-time IS/OOS
split, PROJECT_REPORT.md §3), `oos_trades`, `oos_win_rate`,
`oos_expectancy`, and `oos_total_R` are marked `NOT AVAILABLE` for **all
8 rows** — no strategy has a persisted trade-level breakdown of that
specific original OOS window. AUDJPY AMR's genuine chronological
TRAIN/VALIDATION/OOS trade-level split (n=208, PF 1.14, expectancy
+$12.86, win rate 69.2%, total R +10.18, from phase 22 — a *different,
more recent* OOS window than the original discovery split) is reported
in Section 6 above, not blended into this CSV's `oos_pf` column, to
avoid mixing two different measurement periods in one number.
`historical_trades` and `max_drawdown` / `max_losing_streak` use the
full-history reconstruction (the most granular data available, cited
per-strategy above with exact experiment IDs in the CSV's
`source_exp_ids` column).
