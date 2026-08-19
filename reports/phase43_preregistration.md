# Phase 43 Preregistration — Exposure × Volatility Stress Attribution

**Frozen before any substantive analysis. Committed separately, before any Phase 43 result exists. Not modified after seeing results.**

FORENSIC/OBSERVATIONAL ANALYSIS ONLY. No new strategy, no backtest, no intervention, no portfolio change. Narrows Phase 42's C. MODERATE/PROMISING volatility finding by testing whether it is actually an exposure phenomenon.

---

## 1. Control portfolio (unchanged from Phases 41/42)

`data/phase26_all_trades.csv`, 2,712 trades, 6 strategies, 2023-08-01 to 2026-08-13. No candidate strategy added.

## 2. Historical windows (identical convention to Phases 41/42)

A = full historical control. B = pre-demotion (entry_time < 2026-07-31). C = post-demotion live sample (`reports/5ers_portfolio_update_aug13_trade_level.csv`, 19 trades) — reported separately, never pooled, labeled INSUFFICIENT LIVE SAMPLE.

## 3. Core exposure definitions (frozen, per Part 7)

**Open-position count** at any timestamp *t*: number of control trades with `entry_time ≤ t < exit_time`. **Total open risk**: since every trade in this control is R-normalized to a fixed fractional-risk-per-trade convention (the project's standard sizing model used throughout every prior phase's R-multiple methodology — confirmed by inspecting the ledger's own construction, where `r_multiple` is defined as P&L divided by that trade's own initial risk unit), **each open trade contributes exactly 1R of initial risk by construction**. Therefore, **for this dataset, total open risk in R equals open-position count** — this is a disclosed, methodologically important equivalence, not an independent second variable, and is treated explicitly as such in H2 (§Part 12) rather than silently presented as two different measures. A dollar/percentage-of-account risk figure is NOT computable from this dataset (no live account-equity-at-risk series is stored per trade) and is reported as UNKNOWN, not estimated. **Risk-weighted JPY / long / short / AMR / ARB exposure**: since risk-per-trade is constant (1R), these reduce to the respective **open-position sub-counts** (e.g., JPY open risk = count of concurrently open JPY-linked trades) — again disclosed, not fabricated as independent.

## 4. Correlated open risk (frozen, per Part 8)

For each timestamp, the set of currencies (base+quote) represented among open positions. **Shared-currency risk** = count of open positions sharing at least one currency with at least one other open position. **Maximum single-factor concentration** = the largest count of open positions sharing one specific currency, divided by total open count. **Currency-factor count** = number of distinct currencies represented. This is factor-overlap attribution (a direct counting exercise), not a covariance/correlation-matrix calculation — the preregistration explicitly does not claim portfolio covariance, per the task instruction's own caution.

## 5. Entry-state definition (frozen, per Part 9)

For every trade, all exposure/volatility measures are captured using the state that exists in the **instant immediately before** that trade's own `entry_time` (i.e., excluding the trade itself from its own "prior exposure" count).

## 6. Volatility state / transition definitions (unchanged from Phase 42)

Daily volatility level = mean `atr_pctile` of that day's entries; percentile = re-ranked against the full period; state = LOW/NORMAL/HIGH terciles. Volatility expansion event (§Part 10) = any day classified HIGH that was preceded by a non-HIGH day (i.e., `NORMAL_to_HIGH` or `LOW_to_HIGH` in Phase 42's transition-matrix vocabulary) — reusing Phase 42's already-frozen transition classification exactly, no new threshold invented.

## 7. Directional / mechanism / JPY exposure definitions

Identical to Phases 41/42 (`dir` BUY/SELL; mechanism parsed from strategy-name suffix; JPY = instrument base or quote is JPY).

## 8. Stress definition (identical to Phases 41/42)

Worst 1%/5%/10%/20% of daily portfolio R, fixed thresholds computed once on the full period.

## 9. Eight primary hypotheses (frozen, per Parts 11-18)

H1 (volatility × position count) · H2 (position count vs. total open risk — disclosed as near-equivalent per §3) · H3 (correlated open risk vs. total open risk) · H4 (exposure before vs. after volatility expansion) · H5 (entries during volatility transitions) · H6 (directional exposure, controlling for total exposure) · H7 (JPY × volatility × exposure interaction) · H8 (mechanism × volatility × exposure interaction). Each tested exactly as specified in the corresponding task-instruction part.

## 10. Exposure build-up / decay methodology (frozen, per Parts 19-20, with a disclosed data-granularity limitation)

The control dataset is **trade-level** (entry/exit timestamps only), not a continuous intraday tick/bar series. The task's requested T-24h/T-12h/.../T+8h timeline (§Part 19) is therefore reconstructed using the **set of open positions at each of those offsets relative to each stress episode's start**, computed from entry/exit timestamps directly (fully computable, no continuous market data required for *this* specific measure) — this is NOT a data-granularity limitation the way Phase 42's intraday volatility lead-lag was; open-position reconstruction at arbitrary timestamps is directly derivable from entry/exit times. Exposure decay (§Part 20) is measured the same way, extending forward from the stress episode's peak.

## 11. Statistical approach (identical to Phase 42, per Part 32)

Effect size (absolute/relative difference) as the primary reported quantity, qualitative CONFIRMED/STRONG/MODERATE/WEAK/NO CLEAR RELATIONSHIP/INSUFFICIENT scale, minimum 20 observations for a non-INSUFFICIENT verdict, consistent with Phase 41/42's convention.

## 12. Multiple-testing policy (frozen, per Part 31)

H1-H8 are primary preregistered hypotheses. The build-up/decay timeline (§Part 19-20), the two-dimensional count-vs-risk matrix (§Part 22), and the correlated-exposure matrix (§Part 23) are **descriptive extensions of H1-H3**, not new hypotheses. Anything not directly tied to H1-H8 is EXPLORATORY, labeled as such.

## 13. Missing-data handling

Identical to Phase 42: rows with null `atr_pctile` excluded from volatility-conditioned calculations only, retained elsewhere.

## 14. No-intervention rule (frozen, per Part 36)

Any finding — however strong — is recorded as an observation and, where relevant, a FUTURE PORTFOLIO-CONTROL RESEARCH HYPOTHESIS only. No position-count cap, risk cap, or exposure filter is implemented, tested, or optimized in this phase.

---

*No amendment has been made to this document after any Phase 43 result was produced.*
