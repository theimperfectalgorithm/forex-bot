# Phase 37 Pre-Registration — FROZEN BEFORE ANY SUBSTANTIVE ANALYSIS

**Written and committed before AUDUSD Monday LONG is re-run through the standardized battery, and before the Track B classification scores are computed. Not changed after seeing results. Any later methodological flaw is documented as a dated amendment, never a silent edit.**

---

## TRACK A — AUDUSD MONDAY LONG (frozen)

### A1. Exact candidate definition (reproduced from source, not memory)

Source: `src/phase30_nonjpy_calendar_screen.py::drift_cell()`, the exact function that originally produced the AUDUSD Monday LONG result. **No parameter, threshold, or rule in this definition is altered.**

- **Instrument:** AUDUSD. **Timeframe:** D1.
- **Signal/entry:** every Monday's D1 bar. Entry at that bar's **open**.
- **Exit:** that same bar's **close** (a pure open-to-close hold — no intrabar SL/TP, no multi-day hold; `HOLD_DAYS = 1` in the source).
- **Direction:** LONG only (`raw_move = close - open`).
- **Cost:** a flat 0.00018 (round-trip, price units) subtracted from the raw move — identical to the value in `SPREAD_COST['AUDUSD']`.
- **R-normalization:** `R = (raw_move - cost) / ATR14`, where ATR14 is a 14-day rolling true-range average computed on the **full daily series** (not Monday-only) and reindexed onto each Monday. This affects only how PnL is *expressed* in R-units, not which days are traded or the raw dollar/pip outcome of any trade.
- **Original data period:** 2023-01-01 to 2026-08-14 (`DATA_START`/`DATA_END` in source).
- **Original split (two-way, not three-way — reproduced exactly, not force-fit into Phase 33/35's three-way convention):** IS/TRAIN = 2023-01-01 to 2025-01-01; OOS = 2025-01-01 to 2026-08-14. The original screen had **no separate validation fold** — this is stated explicitly in its own registry ("N/A — single IS/OOS split, no separate validation fold (small-universe exploratory screen)") and is reproduced as such, not retrofitted with an invented validation period.
- **Original evidence (from `reports/non_jpy_diversification_research.md` / `phase36_research_ledger.csv`):** OOS PF 3.070, OOS t-stat 4.15, IS t-stat 1.65 (did not clear the pre-registered IS+OOS t≥2.0 bar), OOS PF at 2x cost stress 2.647 — classification "E. PROMISING — requires more validation," unchanged since.

### A2. No optimization (frozen constraint)

Every one of §A1's values is fixed. Track A tests only whether this **exact, unmodified** definition survives the standardized battery — it does not search for a better AUDUSD variant.

### A3. Reproduction methodology (frozen)

Re-run `drift_cell()` verbatim (Monday, LONG, AUDUSD, identical cost) against a fresh MT5 pull over the identical date range, and compare OOS trade count/PF/t-stat/expectancy against the values in §A1. **Tolerance: within 5% on PF and expectancy, exact match on trade count** (both pulls use the same broker feed and date range, so an exact trade-count match is expected; a small PF/expectancy tolerance accounts for potential feed-refresh differences in the trailing few days of data).

### A4. OOS sub-period consistency (frozen — identical methodology to Phases 33/35)

Split the OOS population (2025-01-01 to 2026-08-14) into two halves by trade-date median. Same classification rule as Phase 35: **FAIL** if the two halves disagree in sign and total OOS trades ≥ 40; **WARNING** (not automatic FAIL) if disagreement occurs with < 40 total OOS trades. AUDUSD Monday LONG's OOS trade count (84, per prior phases) is **below** the 40-trade "conclusive" floor only if halved (≈42 per half) — noted explicitly, not glossed over.

### A5. Parameter robustness (frozen — identical ±20% framework, with an explicit disclosed limitation)

**This candidate has no trade-selection parameter** — entry/exit are fully determined by calendar day and D1 open/close, unlike every Phase 33/35 candidate (which had a threshold or lookback gating entry). The **only** perturbable numeric parameter is the ATR(14) normalization window, tested at 11 and 17 bars (±20% of 14, rounded to the nearest integer). **This perturbation changes only the R-multiple scaling per trade, not which days are traded or the raw dollar outcome — disclosed explicitly as a materially weaker robustness test than Phase 33/35's threshold perturbations, not presented as equivalent.**

### A6. Cost stress (frozen — identical model)

1.0x / 1.5x / 2.0x of the 0.00018 AUDUSD cost assumption, identical to Phase 30/32/33's convention.

### A7. Regime analysis (frozen — identical tercile methodology)

ATR terciles fixed from the TRAIN/IS period only (2023-01-01 to 2025-01-01), applied to OOS Monday trades. UNKNOWN if any tercile bucket has < 10 OOS trades (Phase 34's precondition, reused unchanged).

### A8. Historical regime analysis (frozen — Phase 36's calendar periods, applied to this specific candidate)

Compute Monday LONG's own performance (trade count, PF, mean R) within each of Phase 36's five characterized periods (2019-2020, 2021-2022, 2023-2024, 2025, 2026 YTD), using the identical `drift_cell()` mechanics, extended back to 2019 where AUDUSD D1 data is confirmed available (1,982 bars, verified before this document was written). **No regime is excluded or reweighted based on its result.**

### A9. Drawdown correlation (frozen — identical fair-window-matched methodology to Phase 33/35)

Control = `data/phase26_all_trades.csv`, restricted to AUDUSD Monday LONG's own OOS date range (2025-01-01 to 2026-08-14) for a fair comparison, exactly as corrected in Phase 33. Drawdown days = control's own worst-decile days within that window. Classification: STRONG DIVERSIFIER (drawdown-day corr ≤ normal-day corr), USEFUL DIVERSIFIER (drawdown-day corr modestly below normal, within a small margin), NEUTRAL (difference within 0.15), CORRELATED (drawdown-day corr exceeds normal-day corr by >0.15), UNKNOWN (< 8 overlapping drawdown days, Phase 35's precondition).

### A10. Portfolio integration (frozen — identical methodology to Phase 33/35)

CONTROL vs. CONTROL+CANDIDATE, AUDUSD Monday LONG's actual OOS trade stream blended at 0.5x and 1.0x of the control's own median single-strategy daily-R-std — no weight optimization.

### A11. Monte Carlo (frozen)

10,000-draw trade-order reshuffle of AUDUSD Monday LONG's own 84 OOS trades (preserves each trade's own date/regime context by not touching the underlying series, only trade order — consistent with Phase 33/35's convention).

### A12. Classification rules (frozen — Part 15's 8 categories, applied mechanically and in this order)

- **A. REJECTED — NO CREDIBLE EDGE**: reproduction (§A3) shows OOS PF ≤ 1.0.
- **B. REJECTED — OOS INSTABILITY**: §A4 = FAIL.
- **C. REJECTED — PARAMETER FRAGILITY**: §A5 shows a sign reversal (with the §A5 limitation noted, a sign reversal here would still be meaningful evidence of fragility, even though the reverse — stability — is weaker evidence of robustness than for a true trade-selection parameter).
- **D. REJECTED — COST FRAGILITY**: §A6 OOS PF < 1.0 at 1.5x cost.
- **E. REJECTED — REGIME FAILURE**: §A8 shows the edge is confined to a single characterized period with no signal (not just weaker magnitude) in every other period with adequate sample.
- **F. REJECTED — POOR DRAWDOWN DIVERSIFICATION**: §A9 = CORRELATED.
- **G. PROMISING BUT UNDER-SAMPLED**: passes A-F, but at least one required category has UNKNOWN or borderline evidence (e.g. the OOS-half WARNING at n<40, or a regime bucket UNKNOWN).
- **H. VALIDATION-PASSED — DEMO FORWARD TEST ELIGIBLE**: passes every category with no UNKNOWN and no exception. **Per the task's explicit instruction, reaching H does not authorize any deployment or demo account creation — it stops the analysis, pending independent review.**

---

## TRACK B — RETURN-STREAM DIVERSIFICATION MAP (frozen)

### B1. Classification framework

The 10 return-stream classes named in the task (cross-asset relationships, commodity-based, index-based, cross-sectional FX, relative-value/spread, volatility-conditioned, multi-asset momentum, event/macro-conditioned, session-specific structures, and any additional structurally distinct class this project's own history surfaces) are scored on the fields specified in Part 18, using **HIGH/MEDIUM/LOW/UNKNOWN** wherever quantitative evidence does not exist — never a fabricated number.

### B2. Portfolio gap mapping (frozen)

Six gaps as specified (HIGH-vol weakness, drawdown correlation, mean-reversion concentration, Asian/London session concentration, JPY concentration, lack of a genuinely different return driver), each class scored 0-3 as a **research-priority signal only**, explicitly not a profitability claim.

### B3. Priority score weights (frozen, exactly as specified in the task)

Portfolio independence 25%, drawdown diversification 20%, HIGH-vol compatibility 15%, mechanism diversity 15%, data quality 10%, researchability 5%, cost/execution feasibility 5%, overfitting risk 5%. **These weights are fixed before any class is scored.**

### B4. Exclusion criteria (frozen)

A class is marked LOW research priority (not excluded from the report, but ranked low) if: (a) data availability is UNKNOWN or confirmed absent from this project's current MT5 toolchain, or (b) overfitting risk is HIGH with no mitigating factor, or (c) it does not structurally differ from at least one dimension already covered by the current book or by Track A's AUDUSD candidate.

### B5. No backtesting (frozen constraint)

Track B produces no entry rule, no exit rule, no parameter, no backtest configuration. It ends at the return-stream-class level, per explicit instruction.

---

*Frozen at the time of this commit. No Track A re-run and no Track B score has been computed yet.*
