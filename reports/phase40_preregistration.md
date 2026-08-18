# Phase 40 Preregistration — Volatility-Conditioned HIGH-Vol-State Trend Continuation

**Frozen before any substantive backtesting. Committed separately, before any Phase 40 result exists. Not modified after seeing results.**

RESEARCH ONLY. No live strategy, parameter, risk, or portfolio weight modified. AMR/ARB/GBPUSD Monday/AUDUSD Monday LONG untouched. Exactly ONE volatility-conditioned hypothesis is tested — no tournament, no variant selection.

---

## 1. The hypothesis (ONE coherent concept, per Part 4)

**HIGH-volatility-state trend continuation.** Economic rationale: realized-volatility regimes exhibit clustering (already established in this project's own Phase 19/31/32/36-38 work); during HIGH-volatility states, directional order flow is more likely to persist over the following bar than during LOW/NORMAL states (a volatility-clustering-plus-continuation mechanism, distinct from calendar drift, cross-sectional relative ranking, and session-transition breakout — the three mechanisms already tested in this project). Volatility is a genuine **conditioning/activation** variable, not a post-hoc label: the strategy takes **no position at all** outside a HIGH-volatility state.

## 2. Volatility definition (frozen)

**Normalized ATR**: `ATR(14) / close`, computed on H1 bars using the standard true-range formula (`max(high-low, |high-prev_close|, |low-prev_close|)`, 14-period rolling mean). Chosen over raw rolling-std-of-returns because it is already the convention used throughout this project (Phase19/31/36/37/38's ATR-tercile regime methodology) — reusing an established, already-validated measure rather than inventing a new one.

## 3. Volatility state classification (frozen, IS-fixed, no leakage)

Terciles computed **once, on the TRAIN period only** (see §9): `q1` = 33rd percentile, `q2` = 66th percentile of normalized ATR over TRAIN. These two threshold **values** (not percentile ranks) are frozen and applied identically, unchanged, to VALIDATION and OOS. **HIGH state** = normalized ATR (computed using data through the bar in question's own close) `> q2`. A bar's volatility state is available only as of that bar's close — the entry decision at bar *t+1*'s open uses bar *t*'s state (one-bar lag, no leakage).

## 4. Instrument universe (frozen, non-JPY, per Part 5)

**EURUSD, GBPUSD, AUDUSD, USDCAD.** Selected for: (a) non-JPY, directly addressing Phase 39/Gap5's JPY-concentration finding; (b) long, continuous, already-validated H1 history in this project's data pipeline (Phases 30-38); (c) high liquidity/reliable spread assumption under the project's existing flat-cost convention; (d) structural independence from the currently-live AMR/ARB (all JPY-cross) and GBPUSD Monday (calendar-drift) strategies.

## 5. Session (frozen, per Part 6)

**New York session, 13:00–21:00 UTC-server-hour** (matching this project's established server-hour convention — `src/phase19_london_ny_volatility_persistence.py`'s `NY_START, NY_END = 12, 21`; this phase uses 13:00 start to align with the more commonly cited NY-open time and avoid the London/NY overlap ambiguity documented in that same file). **Justification**: Phase 39's session-coverage audit (`reports/phase39_fx_session_coverage.csv`) found New York the most thinly-tested confirmatory session (exactly 1 prior hypothesis, Phase 35's H2, which tested NY-session *momentum* without any volatility conditioning) — a genuinely different, better-justified test of the same session, not a re-run. No daylight-saving adjustment is applied, consistent with every prior session-based hypothesis in this project (a disclosed, project-wide limitation, not unique to Phase 40).

## 6. Timeframe (frozen, per Part 7)

**H1.** Matches Phase 38's H2 precedent for session-based signal timing; sufficient signal stability (24 observations/day) without the execution-unrealism risk of finer timeframes; consistent with the project's existing validated H1 data pipeline.

## 7. Entry/exit (frozen, per Part 8)

- **Entry condition**: at each H1 bar's open during the NY session window (13:00–20:00 UTC-server-hour, i.e., bars whose open falls in that range, leaving room for the position to close by 21:00), IF the immediately preceding H1 bar's volatility state (§3) is HIGH, THEN enter in the direction of that preceding bar's own price change (LONG if prior close > prior open, SHORT if prior close < prior open). If the preceding bar's state is not HIGH, **no trade** — volatility is a genuine activation gate, not a post-hoc label.
- **Stop**: 1.0× normalized-ATR-in-price-units (`ATR(14)` in price terms, computed at entry) against the trade direction.
- **Exit**: NY session close (21:00 UTC-server-hour) or stop, whichever occurs first. No separate profit target — a single, minimal exit mechanism (matching Phase 38 H2's precedent), avoiding a second tunable exit parameter.
- **Maximum holding period**: same trading day only, NY session close.
- **Cost**: flat 0.00018 (identical convention used project-wide since Phase 26).

## 8. Perturbable parameter (±20%, frozen in advance)

**ATR window**: 14 → 11 (-20%) / 17 (+20%). The volatility-state terciles are recomputed on TRAIN for each perturbed window (not merely rescaled), since the window choice affects both the volatility measure and the state classification. No other parameter is perturbed (entry, session, stop, exit are structural choices, per the "no optimization" constraint).

## 9. Train / Validation / OOS (frozen, chronological, per Part 9)

Reuses the exact three-way split already established in Phase 35 (not re-invented): **TRAIN = 2023-01-01 to 2024-08-31, VALIDATION = 2024-09-01 to 2025-04-30, OOS = 2025-05-01 to 2026-08-14.** Volatility-state thresholds (§3) are computed on TRAIN only and never re-estimated on VALIDATION or OOS data.

## 10. Data integrity (per Part 10)

`research_data_validator` run on all source CSVs before analysis. MT5 H1 pulls integrity-asserted (monotonic, no duplicates, positive OHLC) exactly as in every prior phase. Volatility-specific checks: (a) each bar's ATR/state uses only data through that bar's own close; (b) the entry decision at bar *t+1* uses bar *t*'s already-realized state (verified by construction — the code computes state as of the bar strictly before the entry bar); (c) TRAIN-derived thresholds are applied unchanged to VALIDATION/OOS, never re-fit.

## 11. Robustness/cost/regime/drawdown/portfolio/Monte Carlo methodology (identical to Phases 37/38)

OOS sub-half consistency (sign-consistent required; WARNING tier if OOS n<40, per the established convention). Cost stress at 1.0x/1.5x/2.0x. Historical-regime characterization using the five frozen calendar periods (2019-2020/2021-2022/2023-2024/2025/2026 YTD) where extended data permits. Drawdown correlation against the OOS-window-matched six-strategy control (`data/phase26_all_trades.csv` via `phase31_factor_regime_map.load_hist()`), STRONG DIVERSIFIER / NEUTRAL / CORRELATED classification using the same 0.15 threshold rule established in Phase 33 and reused every phase since. Portfolio integration at 0.5x/1.0x fixed weight, no optimization. Monte Carlo: 10,000-draw trade-order reshuffle, clearly labeled SIMULATED.

## 12. Minimum sample requirements

OOS trades ≥ 30 for a point estimate to be STATISTICALLY INFORMATIVE; OOS sub-half ≥ 40 total OOS trades for the sign-consistency check to carry FAIL weight; regime/historical-period buckets require ≥ 10 trades or UNKNOWN; drawdown-correlation overlap requires ≥ 8 days or UNKNOWN.

## 13. Acceptance / rejection criteria

Exactly the 10-category list in Part 29 (A-J), applied in mechanical order: A (structural duplication) → B (no edge) → C (OOS instability) → D (parameter fragility) → E (cost fragility) → F (HIGH-volatility failure) → G (poor drawdown diversification) → H (poor portfolio fit) → I (promising) → J (portfolio qualified). Per Part 30, drawdown correlation (Gate G) is a **hard gate**: a candidate passing every other gate is still rejected at G if it materially correlates with the control during the control's own drawdowns. J requires ALL 14 of Part 30's conditions; any critical UNKNOWN forces I, never J. Reaching J does not authorize deployment.

## 14. No-rescue rule (frozen, per Part 35)

If this candidate fails any gate, no parameter, session, instrument, exit, stop, target, or filter is altered to rescue it. Any interesting diagnostic finding is recorded as a **FUTURE RESEARCH IDEA** in `reports/phase40_multiple_testing.csv`, not tested in this phase.

---

*No amendment has been made to this document after any Phase 40 result was produced.*
