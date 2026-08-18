# Phase 41 Preregistration — Portfolio Stress Anatomy & Common-Factor Attribution

**Frozen before any substantive analysis. Committed separately, before any Phase 41 result exists. Not modified after seeing results.**

FORENSIC/OBSERVATIONAL ANALYSIS ONLY. No new strategy created or backtested. No live strategy, parameter, risk, or portfolio weight modified. AMR/ARB/GBPUSD Monday/AUDUSD Monday LONG/Phase 40 candidate untouched. No intervention implemented, per the frozen no-intervention rule.

---

## 1. Control portfolio definition (frozen, per Part 2)

**Source**: `data/phase26_all_trades.csv` — the exact, already-validated 2,712-trade historical reconstruction used as the control in every phase since Phase 31. **Strategy membership (frozen, unchanged)**: `CADJPY_ARB`, `GBPJPY_AMR`, `EURJPY_AMR`, `AUDJPY_AMR`, `CADJPY_AMR`, `GBPUSD_MONDAY` — exactly the current-6 live strategy set. No candidate strategy (AUDUSD Monday LONG, Phase 38 H1/H2, Phase 40) is added to the control. Per-trade fields used: `entry_time`, `exit_time`, `dir`, `r_multiple`, `pnl`, `session`, `vol_tercile`, `hold_hours`, `strategy` (mechanism/instrument encoded in the name).

## 2. Historical window (frozen, per Part 4)

**A. Full historical control period**: the entire reconstruction, 2023-08-01 to 2026-08-13 (2,712 trades) — this IS the historical reconstruction methodology, not a live-tracked account.
**B. Pre-demotion historical period**: entry_time < 2026-07-31, within the same reconstruction (the vast majority of the 2,712 trades).
**C. Post-demotion live period**: `reports/5ers_portfolio_update_aug13_trade_level.csv` — 19 real, live-tracked production trades, entry_time ≥ 2026-08-02. This is a **separate, much smaller, genuinely live sample**, not a continuation of the backtested reconstruction. Per the frozen labeling rule, C is never presented as equivalent in weight or reliability to A/B, and any Anti-Bias check (§Part 25) that reruns findings "post-demotion" uses **C only**, explicitly flagged UNKNOWN/INSUFFICIENT SAMPLE given n=19.

## 3. Daily aggregation methodology (frozen, per Part 5)

One row per calendar trading day (`entry_time.dt.date`, UTC). A trade's P&L/R is attributed to its **entry date** (not exit date) for daily aggregation — chosen for consistency with every prior phase's `daily_control` construction (`phase31_factor_regime_map.load_hist()` groups by entry date). Concurrent-position count for a given day = the count of trades whose `[entry_time, exit_time]` interval overlaps any point in that day.

## 4. Drawdown / stress definitions (frozen, per Part 6)

Daily portfolio R is the unit of analysis. Stress quantiles: **worst 1%, 5%, 10%, 20%** of trading days by daily R (fixed thresholds computed once on the full period, per §2A, never re-estimated per stress-bucket). Additionally: worst single day, worst 3/5/10-day rolling-sum windows, longest drawdown (consecutive days with cumulative R below its running peak), largest peak-to-trough drawdown in R, and the single largest clustered-loss episode (the worst rolling-N-day window by total R, N chosen as the window minimizing total R among {3,5,10}).

## 5. Factor definitions (frozen, per Parts 8-14)

- **JPY exposure**: a trade is JPY-exposed if its instrument (derived from the strategy name, e.g. `EURJPY_AMR` → EURJPY) contains JPY as base or quote. All four AMR strategies and the one ARB strategy in this control are JPY-crosses; only GBPUSD_MONDAY is non-JPY.
- **Mechanism**: parsed from the strategy-name suffix (`_AMR`, `_ARB`, `_MONDAY`).
- **Volatility state**: the already-validated, per-trade `vol_tercile` column in the source data (LOW/NORMAL/HIGH), computed by the original AMR/ARB regime-diagnostic pipeline (Phase 20-24) — reused as-is, not recomputed, since re-deriving it would risk a methodology mismatch with the strategies' own live logic.
- **Session**: the source data's `session` column (values observed: ASIAN, LONDON only — the control has zero recorded NY-session trades, consistent with Phase 31's finding).
- **Direction**: the `dir` column (BUY/SELL).
- **Currency**: parsed base/quote from the instrument implied by each strategy name.

## 6. Correlation / clustering methodology (frozen, per Parts 15-21)

**Simultaneous loss**: 2+ distinct strategies each producing a net-negative daily R contribution within the same calendar trading day (§3's daily-aggregation convention). **Loss cluster sizes**: 2+, 3+, 4+, 5+, 6 (all strategies) simultaneously negative on the same day. **Conditional correlation**: for each strategy pair, Pearson correlation of daily R series computed separately on (a) the full period, (b) non-stress days, (c) worst-20%/10%/5% days — using each pair's own overlapping-day population per bucket (not a single fixed population), consistent with the OOS-window-matched convention established in Phase 33/37/38/40. Minimum 8 overlapping days required for a stress-bucket correlation to be reported; below that, UNKNOWN.

## 7. Statistical tests / minimum sample requirements

Effect sizes reported as simple differences/ratios between stress-bucket and normal-day factor exposure (e.g., JPY trade-share on worst-5% days vs. normal days). A factor comparison bucket with < 8 days is UNKNOWN. A factor comparison bucket with 8-20 days is flagged THIN SAMPLE alongside its result. No formal hypothesis-testing p-value machinery is introduced beyond what the project has already used (this is consistent with every prior phase's practice) — evidence strength is classified qualitatively (CONFIRMED/STRONG/MODERATE/WEAK/INSUFFICIENT/UNKNOWN, per Part 24), not via a fabricated significance threshold.

## 8. Missing-data handling

No trade row in the control has missing `entry_time`/`exit_time`/`r_multiple`/`strategy` (verified in §Data Integrity). The single row with missing `vol_tercile`/`atr_pctile` is excluded from volatility-factor calculations only, retained everywhere else, and explicitly disclosed.

## 9. Attribution methodology (frozen, per Parts 22-23)

**Marginal stress contribution**: for each strategy, its own R total during each stress bucket, and its % of that bucket's total negative R (not a regression-based attribution — a direct accounting identity, avoiding any modeling assumption). **Counterfactual attribution**: for each strategy individually, recompute each stress-window's total daily R with that strategy's trades removed from the daily ledger — explicitly labeled COUNTERFACTUAL, never OPTIMIZATION, and never combined across strategies (single-strategy removal only, per the frozen "not portfolio optimization" constraint).

## 10. Multiple-testing handling (frozen, per Part 26)

**Primary preregistered factors** (tested with the full battery, Parts 8-14): JPY, mechanism (AMR/ARB/Monday), volatility state, session, direction, currency concentration. **Exploratory factors** (Parts 17-21, labeled EXPLORATORY throughout): entry/exit clustering, temporal sequencing, factor interactions (a small, predeclared interaction set only — no unrestricted search, per Part 20's explicit instruction). Exploratory findings are never used to claim confirmed evidence strength above WEAK/MODERATE without independent corroboration from a primary factor.

## 11. Evidence classification (frozen, per Part 34)

Every conclusion labeled OBSERVED / CALCULATED / STATISTICAL ASSOCIATION / EXPLORATORY / COUNTERFACTUAL / UNKNOWN. Never "caused by" — only "associated with" / "coincides with" / "preceded" (never "caused"), per Part 27's causality warning.

## 12. Decision framework (frozen, per Parts 28-29)

Hidden common factor (Part 28): one of A (JPY) / B (volatility) / C (mechanism) / D (session) / E (direction) / F (concurrent exposure) / G (multi-factor interaction) / H (no single dominant factor) / I (insufficient evidence) — selected by which factor(s) show the largest, most robust (surviving the anti-bias re-runs in §Part 25) effect size across the stress buckets, not forced to a single answer if the evidence does not support one. Portfolio failure mode (Part 29): one or more of A-J, evidence-based.

## 13. No-intervention rule (frozen, per Part 35)

Any finding that a specific strategy/factor is disproportionately responsible for stress is recorded as an observation only. No strategy is paused, modified, filtered, hedged, or resized in this phase, regardless of finding strength.

---

*No amendment has been made to this document after any Phase 41 result was produced.*
