# Phase 38 Preregistration — Cross-Sectional FX (H1) + Session-Specific Structures (H2)

**Frozen before any substantive backtesting. Committed separately, before any Phase 38 result CSV exists. This document is not modified after seeing results — any methodology change requires an explicit STOP and a dated amendment section appended below, never a silent edit.**

RESEARCH ONLY. No live strategy, parameter, risk, or portfolio weight is modified by this phase. AUDUSD Monday LONG is not touched or optimized.

---

## 0. Data source and cost model (shared)

MT5 D1/H1 bars, `mt5.copy_rates_range`, same integrity asserts as every prior phase (monotonic time, no duplicate candles, positive OHLC). Cost model: flat 0.00018 (identical convention used project-wide since Phase 26/30/37) subtracted from the raw price move at 1.0x; stressed at 1.5x/2.0x. Train/OOS split (two-way, matching the convention already used for AUDUSD Monday LONG in Phase 37, since neither H1 nor H2 has enough post-2023 sample for a three-way split at daily/session granularity): **TRAIN = 2023-01-01 to 2025-01-01, OOS = 2025-01-01 to 2026-08-14.** Historical-regime characterization (where data permits) reuses Phase 36/37's five frozen periods: 2019-2020, 2021-2022, 2023-2024, 2025, 2026 YTD.

Control portfolio for drawdown-correlation and portfolio-integration tests: `data/phase26_all_trades.csv` via `phase31_factor_regime_map.load_hist()`, identical to Phase 37's methodology — OOS-window-matched (restricted to the H1/H2 OOS window, not the control's full history), worst-decile control days = drawdown days, UNKNOWN if fewer than 8 overlapping drawdown-day observations. Divergence classification: STRONG DIVERSIFIER if drawdown-day corr ≤ normal-day corr; NEUTRAL if drawdown-day corr ≤ normal-day corr + 0.15; CORRELATED otherwise.

## 1. Structural independence gate (Part 2)

Applied to H1 and H2 before any backtest result is trusted. Comparison set = the 68-hypothesis Phase 36 ledger (calendar/drift family, AMR, ARB, GBPUSD Monday, Phase 33 XAUUSD/USDCAD, Phase 35's 5 NY-session hypotheses) plus AUDUSD Monday LONG. Classified A/B/C per the frozen rule: C (duplicative) → rejected before backtesting.

---

## H1 — CROSS-SECTIONAL FX (frozen definition)

**Universe** (frozen, no addition/removal after results): 8 currencies — USD, EUR, GBP, JPY, AUD, CAD, CHF, NZD. Constructed from 7 USD-based majors already used throughout this project's prior phases: EURUSD, GBPUSD, AUDUSD, NZDUSD, USDJPY, USDCAD, USDCHF.

**Signal construction (ONE method, frozen)**: relative currency momentum. For each currency, compute its **trailing 4-week (20 trading day) log return** by averaging the signed log-return of every USD-pair it appears in (sign-flipped for currencies quoted as the base vs. quote of USD, so a positive score always means "this currency strengthened"). Rank the 8 currencies by this score at each weekly rebalance point (Friday close, decision uses only information available through that close — no look-ahead).

**Portfolio construction (ONE method, frozen)**: long the single strongest-ranked currency, short the single weakest-ranked currency, expressed via the most liquid direct or synthetic pair available in the 7-pair universe (e.g., strongest=AUD, weakest=JPY → trade AUDJPY if directly constructible from AUDUSD/USDJPY, else compose a synthetic cross from the two USD legs). Position entered at Monday's D1 open, following the Friday-close ranking.

**Holding period (frozen)**: exactly 1 week — Monday open to Friday close, no exceptions, no discretionary early exit.

**Rebalance frequency (frozen)**: weekly.

**Risk model**: fixed 1R per trade (standardized, no position-sizing optimization).

**Perturbable parameter (±20%, frozen in advance)**: momentum lookback window, tested at 16/20/24 trading days. No other parameter is perturbed (ranking method, holding period, and long/short construction are structural choices, not tunable parameters, per Part "H1 Portfolio Construction": "Do not optimize N after seeing results").

**Acceptance criteria**: Gate 1 (credible OOS edge: OOS PF > 1.0 with sufue statistical adequacy) → OOS consistency (sign-consistent sub-halves, same n<40 WARNING rule as Phase 37) → ±20% robustness (no sign reversal) → cost stress (PF > 1.0 at 2x cost) → HIGH-vol regime (not WEAK) → drawdown correlation (not CORRELATED) → portfolio integration (no unacceptable max-DD deterioration) → sample size adequacy. Classification per Part 6's mechanical A→J order.

---

## H2 — SESSION-SPECIFIC STRUCTURES (frozen definition)

**Structural-difference justification (required before backtesting, per Part "H2 Structural Difference Requirement")**:
1. *Not AMR*: AMR is Asian-hours **mean reversion** (fades range extremes within the Asian session). H2 is a **breakout-continuation** mechanism triggered at the Asian→London session **transition**, held through the NY close — opposite directional logic (follows the break, does not fade it) and spans three sessions, not one.
2. *Not Phase 35 NY breakout* (`phase35_h1`-style): that hypothesis triggered and exited within the NY session on an NY-local range break. H2's trigger is the **London open** breaking the **prior Asian session's range**, and the mechanism is validated purely on that transition — NY session is only the exit boundary, not the signal source.
3. *Not Phase 35 NY momentum* (`phase35_h2_ny_momentum`): that hypothesis used an NY-session momentum-continuation filter on NY price action; H2 uses no NY-session signal at all.
4. *Not Phase 35 overlap continuation* (`phase35_h3_overlap_continuation`): that hypothesis traded the London/NY overlap window itself; H2's entry is at London's *open*, well before the NY overlap begins, and the hold spans into and through NY close, not merely across the overlap.
5. *Not calendar/drift*: no weekday-specific mechanic — H2 is tested on every trading day, not Mondays only.

**Mechanism (ONE coherent structure, frozen)**: Asian-range breakout continuation. Define the Asian session range (00:00–07:00 UTC, broker/server-time-normalized as documented in §"Session boundaries" below) high/low for each trading day. At London open (07:00 UTC), if price has broken above the Asian high, enter LONG; if broken below the Asian low, enter SHORT. No trade if price is still inside the Asian range at London open.

**Universe (frozen, non-JPY preferred)**: EURUSD, GBPUSD, AUDUSD (3 liquid, non-JPY majors already validated in this project's data pipeline).

**Timeframe**: H1 bars for session-range construction and entry timing; D1 for the trading-day calendar.

**Entry**: at the London-open H1 bar's open price, contingent on the breakout condition above.

**Stop**: fixed at the opposite side of the Asian range (i.e., stop = Asian low for a LONG, Asian high for a SHORT) — a structural, non-optimized stop tied to the same range that generates the signal.

**Exit / target**: NY session close (22:00 UTC), or the structural stop, whichever comes first — no profit target search, a single frozen maximum holding period (from London open to NY close, ~15 hours).

**Session boundaries and timezone handling (frozen, verified in Part 3 data-integrity step)**: all session windows defined in UTC using MT5 server-time-to-UTC conversion already established and fixed via the server-time correction (see prior project memory on the server-time fix, e5..d680) — Asian 00:00–07:00 UTC, London open 07:00 UTC, NY close 22:00 UTC, applied identically year-round (DST effects on the *broker's* local session clock are a known limitation, disclosed in the master report, not corrected for in this phase since the same UTC-fixed convention has been used for every session-based hypothesis in this project to date).

**Perturbable parameter (±20%, frozen in advance)**: Asian-session window length, tested at 5.6h/7h/8.4h (i.e., 00:00–05:36, 00:00–07:00, 00:00–08:24 UTC). No other parameter perturbed (entry/exit/stop timing are structural, per the "no optimization" constraint).

**Acceptance criteria**: identical mechanical gate sequence to H1, above.

---

## 2. Multiple-testing accounting (Part 4)

This phase pre-registers exactly 2 confirmatory hypotheses (H1, H2), each with exactly 1 frozen parameter perturbed 3 ways (±20%) for robustness — not treated as separate hypotheses. No hypothesis is added after seeing results. Any additional exploration performed ad hoc during the phase is explicitly labeled EXPLORATORY and is not used for confirmatory acceptance, tracked in `reports/phase38_multiple_testing.csv`.

## 3. Minimum sample requirements

OOS trades ≥ 30 for a point estimate to be treated as more than OBSERVED; OOS sub-half ≥ 40 total OOS trades for the sign-consistency check to carry FAIL weight (else WARNING tier, per the Phase 37 convention); regime/historical-period buckets require ≥ 10 trades or UNKNOWN; drawdown-correlation overlap requires ≥ 8 days or UNKNOWN.

## 4. Rejection/acceptance criteria

Exactly the 10-category list in Part 6 of the task instructions (A through J), applied in mechanical order: A (structural duplication) → B (no edge) → C (OOS instability) → D (parameter fragility) → E (cost fragility) → F (HIGH-vol failure) → G (poor drawdown diversification) → H (poor portfolio fit) → I (promising, more validation required) → J (portfolio qualified). J requires ALL of Part 7's 11 mandatory gates to pass; any critical UNKNOWN forces I, never J.

## 5. Portfolio integration methodology

Fixed hypothetical weight (0.5x, 1.0x of a standardized 1R-per-trade unit), matching Phase 37's convention exactly. No weight optimization at any stage.

---

## Amendment 1 (2026-08-18) — H2 entry-price operationalization

**STOP triggered during implementation, before any usable H2 result existed.** The literal entry rule as originally frozen ("entry at the London-open H1 bar's open price, contingent on price having broken the Asian high/low") is definitionally near-unsatisfiable: the London-open bar's open price is, by market-data construction, essentially identical to the immediately preceding (Asian-session) bar's close — which is itself always inside the Asian range that same bar helped define. Under the literal rule, EURUSD 2019-2026 produced exactly 1 qualifying trade in ~1,900 trading days — not a rejection of the hypothesis, but a non-executable operationalization (confirmed by inspection before any OOS PF/edge number was computed or interpreted).

**Amendment**: the breakout condition is evaluated on the London-open H1 bar's **high/low** (intrabar), not its open — i.e., LONG if that bar's high exceeds the Asian high, SHORT if that bar's low is below the Asian low; entry price = the breakout level itself (Asian high for LONG, Asian low for SHORT), not the bar's open. If both sides are breached within the same bar, direction is resolved by the bar's close position (close > open → LONG, else SHORT) — a fixed, non-optimized tie-break rule, frozen here before any result is computed under the amended rule. All other H2 terms (session boundaries, stop, exit, universe, ±20% parameter perturbation, cost model) are unchanged. This is a necessary operationalization fix, not a threshold or parameter search, and is disclosed as a limitation in the master report.

*No other amendment has been made to this document after any Phase 38 result was produced.*
