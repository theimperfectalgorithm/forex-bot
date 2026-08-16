# Non-JPY Diversification Research — Phase 30B

**Research question (per explicit instruction, not the naive version):** *Can we find one or more genuinely diversifying non-JPY strategies that improve the risk-adjusted behaviour of the existing 6-strategy portfolio without sacrificing robustness?* — not "which pair backtests highest."

**Status: RESEARCH ONLY. No candidate is deployed, no candidate is validated to live-ready. No existing strategy modified.**

**Script:** `src/phase30_nonjpy_calendar_screen.py`. **Registry:** `reports/non_jpy_candidate_registry.csv` (60 cells). **Portfolio comparison:** `reports/non_jpy_portfolio_comparison.csv`.

---

## 1. Current portfolio (the control)

Reconstructed from the frozen live parameters, two populations used for different purposes (kept explicitly separate, per Part A's data-integrity discipline):

- **Live post-demotion population** (`reports/5ers_trade_export.csv`, 19 trades since 2026-07-31): PF 0.245, expectancy −0.227R, total R −4.32, max losing streak 4, JPY exposure 78.9% of trades / 71.5% of risk. Full detail: `reports/live_strategy_scorecard.md`.
- **Full historical population** (`data/phase26_all_trades.csv`, 2,712 trades, 2023-08-01 to 2026-08-13): the frozen-parameter reconstruction used for the current-6-strategy revalidation (EXP-105..111). Used in this phase **only** as the daily-return series for the candidate-correlation test (§5) — its own portfolio-level PF/drawdown/Monte Carlo statistics were already computed in `reports/current_6_strategy_revalidation.md` and `reports/portfolio_health_audit_baseline.md` and are not recomputed here to avoid duplicating existing work.

**Neither control population was modified in this phase.**

---

## 2. Why JPY concentration matters (brief recap, full detail in `reports/portfolio_concentration_framework.md`)

5 of 6 current strategies carry JPY exposure; the four AMR pairs share session, mechanic, and a documented no-trend-filter design, making them closer to one correlated risk source than four independent edges. Two multi-strategy loss days account for ~73% of the entire post-demotion drawdown. This is the concrete motivation for this phase — not a generic "diversification is good" assumption.

---

## 3. Candidate universe and why

Per the explicit instruction (B3): EURUSD, GBPUSD, AUDUSD, USDCAD, USDCHF, XAUUSD. No pair outside this list was researched.

**Important pre-existing evidence reused, not re-tested** (consistent with this project's own "do not re-plow settled dead ground" principle, `PROJECT_REPORT.md` §4): EURUSD and GBPUSD price-signal strategies (breakout, mean-reversion, pro-style fade, mechanical ICT — 0/22 + 0/16 across two prior dedicated phases, ~470+ total failures across the project) are **already extensively rejected ground** for those mechanisms. This phase does **not** re-run that search. It tests a **different, not-yet-swept mechanism family** on the full candidate universe instead (§4).

---

## 4. Strategy family and hypothesis (declared before evaluating any result — B5)

**Family: calendar/drift** (per the B4 list — explicitly not a copy of AMR's mean-reversion onto a new pair).

**Hypothesis, stated in advance:** GBPUSD Monday Drift (already live, `PROJECT_REPORT.md` §3, discovered via phase 7's calendar screen, OOS PF 2.929) proves a day-of-week open-to-close drift effect exists and is tradeable on at least one major. This phase asks whether the **same mechanism class** — not a copy of the exact parameters — generalizes to other (pair, weekday) combinations that have never been swept. This is a narrower, better-justified hypothesis than an unconstrained search: it reuses a mechanism already proven once in this project, rather than inventing a new untested idea.

**Market mechanism (hypothesized):** a weekly-cycle positioning/flow effect around session open, of the kind already empirically confirmed for GBPUSD Mondays (t=+3.3 IS / +4.0 OOS in the original phase-7 discovery).

**Timeframe/session:** D1 (daily open-to-close), full weekday session, no intraday entry filter — matching Monday Drift's own basic mechanic before its ATR-scaled SL/TP refinement.

**Entry logic:** long (or, separately tested, short) at that weekday's daily-bar open.
**Exit logic:** at that same day's close (1-day hold, matching Monday Drift's own single-week-slot design).
**Risk model:** R-multiple relative to a rolling 14-day ATR (same normalization convention used throughout this project's research).
**Expected trade frequency:** ~1/week per (pair, day) cell — same order of magnitude as GBPUSD Monday's own live ~52/year.
**Expected holding time:** 1 trading day.
**Reason it might diversify:** none of EURUSD/AUDUSD/USDCAD/USDCHF/XAUUSD carry JPY exposure; a confirmed effect on any of them would mechanically reduce portfolio JPY concentration by construction, and (if the underlying weekly-flow mechanism is currency-pair-specific rather than JPY-specific) plausibly wouldn't share the JPY strategies' correlated-loss-day pattern.

---

## 5. Data methodology

- **Source:** this session's MT5 connection (MetaQuotes-Demo broker feed). **Explicitly documented limitation:** this is *not* the 5ers broker's own historical price data — this session has no access to a 5ers-connected terminal (consistent with every prior phase's finding). Spread/cost assumptions (§6) are therefore generic conservative estimates, not broker-specific.
- **Range:** 2023-01-01 to 2026-08-14 (~940 daily bars per pair — comparable in length to, though not identical to, this project's usual 36-month convention).
- **Split:** chronological, no shuffling. **IS (discovery/train): 2023-01-01 to 2025-01-01** (~24 months, matching this project's standard IS window). **OOS: 2025-01-01 to 2026-08-14** (~19 months). No separate validation fold was used (a single IS/OOS split, appropriate for a small, well-defined 60-cell exploratory screen rather than a full parameter-search pipeline).
- **Cells tested:** 6 pairs × 5 weekdays × 2 directions (LONG/SHORT) = **60 cells**, each evaluated once, with no per-cell parameter tuning.

---

## 6. Cost assumptions

Flat, conservative, per-pair round-trip spread cost, deducted from every simulated trade's raw open-to-close move (not tuned per cell):

| Pair | Assumed cost |
|---|---|
| EURUSD | 0.00015 (1.5 pips) |
| GBPUSD | 0.00020 (2.0 pips) |
| AUDUSD | 0.00018 (1.8 pips) |
| USDCAD | 0.00020 (2.0 pips) |
| USDCHF | 0.00020 (2.0 pips) |
| XAUUSD | $0.35 |

Cost-stress (1.5x, 2x these values) was run on the two candidates that reached the "PROMISING" tier (§8) — not on all 60 cells, since 58 of them were rejected before that stage.

---

## 7. Robustness methodology and screening bar (declared before results)

**Pre-registered screening bar** (mirroring the bar phase 7 used to first flag GBPUSD Monday before its dedicated phase-8 validation): a cell only survives exploratory screening if **both** IS |t-stat| ≥ 2.0 **and** OOS |t-stat| ≥ 2.0, **same sign**, **and** OOS PF > 1.0. This bar was fixed before any cell's result was computed, and was not loosened after seeing the results.

---

## 8. Candidate results

**60 cells tested. 0 cells clear the pre-registered screening bar.** Full detail: `reports/non_jpy_candidate_registry.csv`.

| Classification | Count |
|---|---|
| A. REJECTED — no edge (both IS and OOS t-stat < 1.0) | 20 |
| B. REJECTED — insufficient robustness (IS/OOS sign-inconsistent, or IS below bar) | 38 |
| E. PROMISING — requires more validation | 2 |
| F. PORTFOLIO QUALIFIED | 0 |

**The two "E" cells:**

| Instrument | Cell | IS t | OOS t | IS PF | OOS PF |
|---|---|---|---|---|---|
| GBPUSD | Monday LONG | 1.89 | 4.36 | 1.621 | 3.201 |
| **AUDUSD** | **Monday LONG** | **1.65** | **4.15** | **1.507** | **3.070** |

**GBPUSD Monday LONG is not a new candidate** — it is this screen's D1-open-to-close proxy re-discovering the already-live Monday Drift strategy. Its exact numbers here differ from the live strategy's own validated figures (OOS PF 2.929, `reports/current_6_strategy_revalidation.csv`) because this screen uses a cruder proxy mechanic (plain D1 open-close, no ATR-scaled SL/TP) — this is a useful **sanity check that the screen methodology recovers a known signal**, not a new finding. The live strategy's own already-validated numbers remain authoritative; this screen does not supersede them.

**AUDUSD Monday LONG is the one genuinely new finding.** Both IS and OOS t-stats are directionally strong and same-signed, but **the IS t-stat (1.65) does not clear the pre-registered 2.0 bar** — per this project's own standing discipline ("selection stays IS-only," `PROJECT_REPORT.md` §4), a strong OOS number cannot retroactively justify a weak IS signal. **This candidate is therefore held at "E. PROMISING," not promoted to "F. PORTFOLIO QUALIFIED," strictly because of this rule** — not because the OOS evidence looks weak (it doesn't).

**Cost stress on AUDUSD Monday LONG** (§6): OOS PF falls from 3.07 (1x cost) to 2.851 (1.5x) to 2.647 (2x) — OOS t-stat falls from 4.15 to 3.62 at 2x cost. **The OOS signal survives 2x cost stress comfortably; this is a genuine point in its favor**, even though it doesn't change the IS-bar conclusion above.

---

## 9. Rejected candidates

58 of 60 cells, spanning all six pairs and both directions on most weekdays. Full detail in the registry CSV. Notable patterns (reported honestly, not cherry-picked to fit a narrative):
- **Every pair's "Monday SHORT" cell is the strong mirror-image of its "Monday LONG" result** (expected — they're not independent tests of the same data, just sign flips minus 2x cost) — e.g. GBPUSD Monday SHORT: OOS t=−5.06, AUDUSD Monday SHORT: OOS t=−5.21. This is not additional independent evidence for or against the Monday-drift hypothesis; it's the same signal viewed from the other side.
- **Tuesday–Friday cells show no consistent pattern across pairs** — several IS-positive cells reverse sign OOS (e.g. EURUSD Tuesday LONG: IS t=−1.43 → OOS t=+0.70; AUDUSD Tuesday LONG: IS t=−2.05 → OOS t=+1.04) — exactly the sign-instability the pre-registered bar is designed to catch and reject.
- **USDCAD and USDCHF show no cell clearing even a relaxed reading of the bar** — the weakest-performing pairs in this screen.

---

## 10. Promising candidates

**AUDUSD Monday LONG** (§8) — the only genuinely new candidate in this screen. Requires an independent confirmatory test on data not used in this screen (i.e., a fresh forward period, not a re-split of the same 2023-2026 window) before being considered for the next validation stage. **Not portfolio-qualified from this phase.**

---

## 11. Portfolio-qualified candidates

**None.** Per the explicit instruction, no candidate may be classified "live ready" from this phase, and none reached even the "F. PORTFOLIO QUALIFIED" tier on its own merits (§8) — the strongest candidate (AUDUSD Monday LONG) is held at "E" by the pre-registered IS-bar rule.

---

## 12. Portfolio diversification test (B9) — the one candidate worth running it on

Full detail: `reports/non_jpy_portfolio_comparison.csv`. Since AUDUSD Monday LONG is the only candidate to reach even the "promising" tier, this is the only candidate this test was run on (running it on all 58 rejected cells would be pointless — they already failed at the discovery stage).

**Method:** correlate AUDUSD Monday LONG's OOS daily R-multiples (84 Monday trades, 2025-01-01 to 2026-08-14) against the control portfolio's daily total R (summed across all six current-six strategies, from `data/phase26_all_trades.csv`) on the 84 calendar days where both had activity.

**Result: correlation = 0.29** — a moderate positive correlation, not near-zero and not strongly negative. **This is not a clean diversifier by this measure, but it is also not simply redundant with the existing book.**

**Drawdown-specific correlation:** of the control portfolio's historical worst 10 single days, only **1** coincided with an AUDUSD Monday trade day (2025-05-12) — on that one day, both control (−5.19R) and the candidate (−0.67R) were negative simultaneously. **n=1 is nowhere near enough to draw a conclusion** — reported as **INSUFFICIENT EVIDENCE**, not extrapolated into either "the candidate would have helped" or "the candidate would have hurt" on the account's worst days.

**Marginal drawdown/return contribution:** **NOT COMPUTED.** A proper marginal-contribution figure requires re-running the full portfolio Monte Carlo (as in `reports/current_6_strategy_revalidation.md`) with the candidate's trades interleaved into the historical trade stream — a distinct, larger undertaking than this screen-stage phase, correctly scoped to the next validation phase (§15) rather than approximated or fabricated here.

**JPY exposure impact:** mechanically, adding any AUDUSD-based strategy at a nonzero risk allocation would reduce the portfolio's JPY exposure percentage by construction (AUDUSD carries no JPY leg) — the exact resulting percentage was not computed, since that requires committing to a specific risk allocation for the candidate, which is a live-deployment decision explicitly out of scope for this research-only phase.

---

## 13. Remaining uncertainties

- Whether AUDUSD Monday LONG's OOS signal (t=4.15, robust to 2x cost) would survive an **independent** confirmatory test on genuinely fresh data (this screen's OOS period, once looked at, can no longer serve as an independent confirmation for any future test of the same hypothesis).
- Whether the underlying mechanism (if real) is AUDUSD-specific or reflects a broader USD/commodity-currency weekly-flow effect that might generalize further (e.g. to NZDUSD, not in this screen's universe) — not tested here.
- The true marginal portfolio-level drawdown/return impact of adding this candidate (§12) — not computed, flagged for the next phase.
- This screen used a broker feed (MetaQuotes-Demo) different from the 5ers production broker — whether the effect (or its cost-sensitivity) replicates on 5ers' actual spread conditions is untested.
- Multi-window walk-forward and Monte Carlo trade-reshuffling (both explicitly required by B8 for any "surviving" candidate) were **not run** on AUDUSD Monday LONG in this phase — it did not reach the tier (F) where the instruction requires that fuller gauntlet; if a future phase elevates it further, that full B8 battery should be completed before considering demo forward-testing.

---

## 14. Multiple-testing accounting (B11)

- **Total hypotheses tested:** 60 (6 pairs × 5 weekdays × 2 directions), all from a single pre-declared mechanism family, in a single pass, with the screening bar fixed before any result was seen.
- **Total candidates rejected:** 58.
- **Total parameter combinations:** 0 — no per-cell parameter tuning was performed (no grid search over hold-period, ATR window, or entry-time variants); this was a fixed-mechanism sweep across (pair, day, direction), not a parameter optimization.
- **Selection criterion:** the single pre-registered IS+OOS t≥2.0-same-sign-and-OOS-PF>1.0 bar (§7), applied uniformly.
- **Exploratory vs. confirmatory:** **everything in this report is exploratory.** AUDUSD Monday LONG's OOS period (2025-01-01 to 2026-08-14) was used to *evaluate* this screen — it cannot also serve as an independent *confirmation* for any future test of the same hypothesis. A future confirmatory test must use data outside this window (either a later forward period, or the 5ers/production broker's own feed for the same historical window, which is a different-enough data source to arguably count as semi-independent — flagged as a specific option for the next phase, not decided here).

---

## 15. Recommended next validation phase

**Not** live deployment (explicitly out of scope, B12). Recommended, in order:
1. An independent confirmatory test of AUDUSD Monday LONG on data outside this screen's window (a forward period from today, or the 5ers broker's own historical feed if/when accessible) — a single pre-registered hypothesis test, not a re-scan.
2. If that confirms: the full B8 robustness battery this phase did not complete (multi-window walk-forward, Monte Carlo trade-order reshuffling, parameter-neighborhood sensitivity even though no parameters were tuned here, regime analysis) — matching the discipline already applied to GBPUSD Monday in phases 7-8.
3. Only after both of the above: a proper marginal portfolio Monte Carlo (§12) with the candidate's simulated trades interleaved into the actual historical trade stream, to get a real (not correlation-proxy) drawdown/return contribution figure.
4. **Demo forward-test before any live consideration** — matching this project's own standing deployment discipline (`PROJECT_REPORT.md` §4: "strategy class → demo-only YAML → forward test, never straight to the funded account").

---

## 16. Final answer

**Does the existing portfolio need a non-JPY strategy?**

**INSUFFICIENT EVIDENCE to say the portfolio *needs* one, but the concentration finding (§2, and `reports/portfolio_concentration_framework.md`) makes a reasonable case that one *could help* if a genuinely qualifying candidate is found** — this screen did not find one that clears its own pre-registered bar, so it cannot recommend YES on the strength of what was actually discovered here. It also cannot recommend NO, since the underlying concentration concern (§2) is real and independently documented, not resolved by this screen's null result.

**If a future phase does pursue this further, the next candidate should have this profile** (not a specific pair recommendation — a profile, per the explicit instruction):

- **No JPY leg** — the baseline, non-negotiable requirement given §2's finding.
- **A session window meaningfully different from 00:00-07:00 UTC** — to avoid replicating the AMR cluster's same-session correlation mechanism (`reports/portfolio_concentration_framework.md` §3).
- **A mechanism that is not pure mean-reversion without a trend filter** — since that specific combination is the AMR family's own documented weak point; a trend-following, breakout, or regime-aware mechanic would structurally differ from the failure mode already observed live.
- **Demonstrated cost-robustness at 2x the assumed spread**, matching the bar every currently-live strategy was held to.
- **Correlation to the existing book's daily returns meaningfully below AUDUSD Monday LONG's observed 0.29** — this screen's own near-candidate is a useful benchmark for "not correlated enough to call a clean diversifier"; a future candidate should beat that number, not just clear a raw PF/expectancy bar.
- **Acceptable trade frequency** (roughly weekly, consistent with what a calendar/drift mechanism can realistically supply, or higher if a different mechanism family is used) — no specific number mandated, but a strategy trading only a handful of times a year would take too long to validate at the pace this project's live-forward-testing discipline requires.

---

*Research only. No candidate deployed. No existing strategy modified. No portfolio weights changed.*
