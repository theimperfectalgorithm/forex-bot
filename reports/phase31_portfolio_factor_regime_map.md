# Phase 31 — Portfolio Factor & Regime Map (Master Report)

**Research diagnosis only. No strategy, parameter, risk, pair, filter, or portfolio weight modified. No candidate deployed or promoted.**

**Script:** `src/phase31_factor_regime_map.py`. **Data validated via `src/research_data_validator.py` before any analysis** (both inputs passed cleanly — see §2).

---

## Executive summary

The current 6-strategy portfolio is **more concentrated than its strategy count suggests, on every dimension tested except raw currency-symbol diversity.** Four findings, ranked by evidentiary strength:

1. **Zero New York-session exposure** — every strategy enters new risk only during the Asian session (00:00-07:00 server) or the London open (07:00-09:00). This is a session gap, not a currency gap.
2. **HIGH volatility is the portfolio's only net-negative regime** — 3 of 6 strategies net-negative, combined historical R −6.91, vs. +120.64 (LOW vol) and +78.63 (NORMAL vol).
3. **Strategy correlation rises specifically on the portfolio's worst drawdown days** for most JPY-AMR pairs — the diversification visible on an average day partially disappears exactly when it matters most.
4. **Correlation-adjusted effective diversification is 2.67 of 6 nominal strategies** — not the naive risk-weight-only figure of 5.19.

**JPY concentration (94.7% of risk-weighted currency exposure) is real and contributes to all four findings, but is not shown to be the sole or even necessarily the dominant cause** — session and mechanism concentration (81.5% of risk-weighted trades share one mean-reversion mechanism) are at least as implicated. **This is the anti-bias finding the task explicitly asked this phase to be open to, and the evidence supports it.**

**Final verdict: C. HIGHLY CONCENTRATED** (not "D. fragile/single-factor," not "B. moderately concentrated" — see §16).

---

## 1. Data inventory

| Source | Path | Date range | Coverage | Trades | Type | Suitable for this phase? |
|---|---|---|---|---|---|---|
| Historical frozen-parameter reconstruction | `data/phase26_all_trades.csv` | 2023-08-01 to 2026-08-13 | Current 6 strategies | 2,712 | Backtest reconstruction (IS+OOS combined, frozen live parameters, EXP-105..111) | **Yes — primary source for this phase.** Carries `session`, `dow`, `hold_hours`, `vol_tercile`, `trend_tercile`, `r_multiple`, `dir` per trade — exactly the metadata needed, already validated in prior phases. |
| Production export | `reports/5ers_trade_export.csv` | 2026-07-20 to 2026-08-13 | Current 6 + demoted GBPJPY ARB | 36 unique tickets (33 current-six) | Live | Used for cross-checks only (§ various) — far too small (n=2-9 per strategy) for its own correlation/regime analysis, consistent with every prior phase's sample-size finding. |
| Pre-live acceptance criteria | `reports/current_6_strategy_revalidation.csv` | N/A (summary stats) | All 6 | N/A | Pre-live/IS+OOS summary | Used for family/mechanism verification (§4), not re-derived. |
| Strategy configs | `pairs/*.yaml` | N/A | All 6 | N/A | Config | Used to verify mechanism/session claims against code, not names. |
| Non-JPY calendar screen | `reports/non_jpy_diversification_research.md` + `non_jpy_candidate_registry.csv` | 2023-01-01 to 2026-08-14 | AUDUSD (candidate) | 84 (Monday LONG OOS) | Exploratory screen | Reused for §13 candidate comparison, not re-run. |
| Live scorecard | `reports/live_strategy_scorecard.csv`/`.md` | 2026-07-31 onward | All 6 | 19 (post-demotion) | Live decision framework | Reused for status context, not re-derived. |
| Phase 20/21/22 (named in the task) | — | — | — | — | — | **NOT directly consulted as separate files this phase** — their outputs are already folded into `current_6_strategy_revalidation.csv` and `phase26_all_trades.csv`, which this phase uses directly. Re-opening the raw phase20/21/22 scripts would duplicate, not add, evidence already captured in these two consolidated artifacts. |

**No required data was missing or materially inconsistent.** Both primary CSVs passed `src/research_data_validator.py`'s column-count-consistency, required-columns, and row-count checks before any analysis proceeded (`[validate]` lines in the script's own log output). No STOP condition was triggered.

---

## 2. Methodology

- **Historical population** (`phase26_all_trades.csv`) is the primary source for factor/session/regime/correlation/drawdown analysis (§§3-11) — it has the sample depth (2,712 trades, 774 trading days) these questions require; the live export does not.
- **Regime definitions are REUSED, not invented**: `vol_tercile` and `trend_tercile` come from the existing phase20/21 regime methodology already computed into this file — no new regime model was built for this phase, per explicit instruction.
- **Correlation methodology**: trade-level correlation is **not meaningful** (different strategies almost never share an exact trade timestamp, since AMR pairs fire independently within their shared session window) — documented explicitly rather than silently computed and misinterpreted. **Daily and weekly aggregated R** are used instead, which the task's own Part 7 listed as acceptable alternate views. Missing-strategy-days are treated as `NaN` in the correlation matrix (pandas' default pairwise-complete-observations behavior for `.corr()`) — **not zero-filled**, since a day a strategy didn't trade is not the same as a day it broke even, and zero-filling would artificially inflate correlation by adding spurious matching zero-days.
- **Session labeling caveat, disclosed rather than silently accepted**: GBPUSD Monday's `session` field in `phase26_all_trades.csv` reads `ASIAN` — this is a byproduct of the historical reconstruction's generic hour-of-day session bucketer (its entry falls at hour 0, which the bucketer classifies as Asian by clock time), **not** a claim that Monday Drift is mechanically an Asian-session mean-reversion strategy. Its actual config (`pairs/GBPUSD_monday.yaml`) specifies `session: monday`, a distinct calendar mechanic. This labeling artifact does not change this report's core finding (§3.3) that the book has zero New York-session entries — Monday Drift's *actual* entry time (00:00 server) genuinely is outside NY hours regardless of the label.

---

## 3. Factor map

Full CSV: `reports/portfolio_currency_factor_map.csv`. Summary in `reports/portfolio_factor_regime_map.md`. Risk-weighted currency exposure: **JPY 94.7%, CAD 33.8%, EUR 24.6%, AUD 22.4%, GBP 19.2%, USD 5.3%** (rows don't sum to 100% — each trade counts toward both legs).

---

## 4. Strategy-family map

Full CSV: `reports/portfolio_strategy_family_map.csv`.

| Family | Strategies | Risk-weighted share | Historical PF (aggregate, per-strategy) |
|---|---|---|---|
| mean_reversion | GBPJPY/EURJPY/AUDJPY/CADJPY AMR | **81.5%** | 1.415 / 1.164 / 1.153 / 1.086 |
| asian_range_breakout | CADJPY ARB | 13.2% | 1.270 |
| calendar_drift | GBPUSD Monday | 5.3% | 2.100 |

Four of six strategies (81.5% of risk-weighted trades) share one mechanism family — the single largest concentration by any dimension measured in this phase.

---

## 5. Session map

Full CSV: `reports/portfolio_session_exposure.csv`.

| Session (as-designed, verified against config) | Strategies | Risk-weighted share |
|---|---|---|
| Asian (00:00-07:00 server) | GBPJPY/EURJPY/AUDJPY/CADJPY AMR + GBPUSD Monday (Monday-specific, entry at hour 0) | 94.7% |
| London open (07:00-09:00 server) | CADJPY ARB | 13.2% |
| **New York** | **None** | **0.0%** |

**Answers to the task's five session questions:**
1. **Asian-hours risk creation:** the large majority — 4 of 6 strategies, 81.5% of risk-weighted trades, all initiate here.
2. **London risk creation:** CADJPY ARB only (13.2%), at the 07:00-09:00 open specifically, not the full London session.
3. **New York risk creation:** **zero** — no strategy initiates a new position during NY hours.
4/5. **Overlap and simultaneous entry:** the four AMR pairs all share the identical 00:00-07:00 window — average 2.21 other strategies open at any given trade's entry, maximum 5 simultaneously (§6).
6. **Session-transition exposure:** CADJPY ARB (avg hold 26.3h) and GBPUSD Monday (avg hold 20.2h) both routinely carry positions across multiple session boundaries; the AMR pairs (avg hold 2.95-4.23h) are force-flat by 07:00 server and do not.

---

## 6. Trade overlap analysis

Full CSV: `reports/portfolio_trade_overlap.csv`.

| Metric | Value |
|---|---|
| Average simultaneous strategies open (at any trade's entry) | 2.21 |
| Maximum simultaneous strategies open | 5 |
| Total historical trading days | 774 |
| Days with 2+ strategies entering | 722 (93.3%) |
| Days with 2+ losses | 240 (**31.0%**) |
| Days with 3+ losses | 119 (**15.4%**) |

**Nearly one in three historical trading days saw 2 or more strategies lose simultaneously; nearly 1 in 6 saw 3 or more.** This is not assumed independence — it's a direct count from the trade-level entry/exit interval overlap.

---

## 7. Correlation analysis

Full CSV: `reports/portfolio_return_correlation.csv`. Daily Pearson correlation matrix:

| | AUDJPY | CADJPY_AMR | CADJPY_ARB | EURJPY | GBPJPY | GBPUSD_MON |
|---|---|---|---|---|---|---|
| AUDJPY | 1.00 | 0.26 | 0.08 | 0.25 | 0.32 | 0.08 |
| CADJPY_AMR | 0.26 | 1.00 | 0.11 | 0.34 | 0.41 | 0.23 |
| CADJPY_ARB | 0.08 | 0.11 | 1.00 | −0.02 | 0.05 | 0.03 |
| EURJPY | 0.25 | 0.34 | −0.02 | 1.00 | 0.40 | 0.17 |
| GBPJPY | 0.32 | 0.41 | 0.05 | 0.40 | 1.00 | 0.18 |
| GBPUSD_MON | 0.08 | 0.23 | 0.03 | 0.17 | 0.18 | 1.00 |

**The four AMR pairs correlate with each other at 0.25-0.41 — consistently the highest pairs in the matrix.** CADJPY ARB and GBPUSD Monday correlate weakly with everything (0.03-0.23) — both are the two structurally different strategies (breakout, calendar-drift) in the book, consistent with §4's family finding. **Average pairwise correlation across all 15 pairs: 0.192.**

Spearman and weekly-aggregated views are in the CSV; they tell the same story (not reproduced here to avoid redundancy).

---

## 8. Drawdown-specific correlation (the key diversification question)

Full CSV: `reports/portfolio_drawdown_factor_analysis.csv`. Comparing correlation on the portfolio's worst-decile drawdown days (n=78 of 774) vs. all other days:

| Pair | Correlation, drawdown days | Correlation, normal days | Direction |
|---|---|---|---|
| EURJPY / GBPJPY | **0.557** | 0.373 | **Higher in drawdown** |
| AUDJPY / GBPJPY | **0.448** | 0.284 | **Higher in drawdown** |
| CADJPY_AMR / GBPJPY | **0.483** | 0.399 | **Higher in drawdown** |
| AUDJPY / EURJPY | **0.393** | 0.221 | **Higher in drawdown** |
| AUDJPY / CADJPY_AMR | **0.323** | 0.242 | **Higher in drawdown** |
| CADJPY_AMR / EURJPY | 0.362 | 0.329 | Roughly flat |
| CADJPY_ARB / GBPUSD_MON | **0.485** | −0.055 | **Sharply higher in drawdown** |
| GBPJPY / GBPUSD_MON | −0.318 | 0.272 | Lower/reversed in drawdown |

**Answer to the task's key question — "do the strategies become more correlated exactly when the portfolio is losing?" — YES, for the majority of AMR-pair combinations, and for one non-AMR pair (CADJPY ARB / GBPUSD Monday) too, though that specific pair's drawdown-day sample is thin (n=31).** This is the single most important quantitative finding in this phase: the correlation structure is not stable — it worsens specifically during adverse periods, which is exactly when low correlation would be most valuable.

---

## 9. Regime analysis

Full CSV: `reports/portfolio_regime_matrix.csv`. Volatility tercile, all six strategies:

| Regime | Strategies net-negative | Combined R |
|---|---|---|
| LOW | 0 of 6 | **+120.64** |
| NORMAL | 0 of 6 | **+78.63** |
| **HIGH** | **3 of 6** (AUDJPY AMR, CADJPY AMR, CADJPY ARB) | **−6.91** |

Trend tercile:

| Regime | Strategies net-negative | Combined R |
|---|---|---|
| LOW_TREND | 0 of 6 | +86.13 |
| HIGH_TREND | 1 of 6 | +56.49 |
| NORMAL_TREND | 2 of 6 | +51.49 |

**HIGH volatility is unambiguously the portfolio's weakest regime by both count-of-strategies-affected and combined-R.** No trend regime shows the same severity. **This confirms the finding already flagged in `reports/live_strategy_scorecard.md` and `reports/5ers_portfolio_update_aug13.md` for AUDJPY/CADJPY AMR specifically — this phase extends it to show CADJPY ARB shares the same HIGH-vol weakness despite being a structurally different (breakout, not mean-reversion) strategy, which is new information from this phase.**

---

## 10. Regime coincidence — portfolio-wide weak/strong regimes

**Portfolio-wide weak regime: HIGH volatility** (§9) — the only regime bucket where the portfolio is net-negative, and the only one with 3+ strategies simultaneously net-negative.

**Portfolio-wide strong regimes: LOW and NORMAL volatility, and LOW_TREND** — all show 0 of 6 strategies net-negative, i.e. every current strategy is individually profitable in these conditions historically.

---

## 11. Volatility exposure — the AMR family specifically

| Strategy | HIGH-vol PF | HIGH-vol expectancy R | LOW-vol PF | NORMAL-vol PF |
|---|---|---|---|---|
| GBPJPY AMR | 1.122 | +0.050 | 1.881 | 1.255 |
| EURJPY AMR | 1.164 | +0.057 | 1.164 | 1.163 |
| AUDJPY AMR | **0.827** | **−0.074** | 1.338 | 1.256 |
| CADJPY AMR | **0.833** | **−0.071** | 1.554 | 1.026 |

**AUDJPY and CADJPY AMR are net-losing specifically in HIGH volatility; GBPJPY and EURJPY AMR are not** (both stay above PF 1.1 even in HIGH vol). **The portfolio's AMR-family HIGH-vol weakness is not uniform across the four pairs — it is concentrated in exactly the same two (AUDJPY, CADJPY) already flagged ORANGE in the live scorecard**, which is a genuine cross-validation between this phase's historical analysis and the live decision framework's independent finding.

---

## 12. Directional factor map

Full data printed by the script; key pattern: **every AMR pair's SELL side is weaker than its BUY side** (EURJPY: BUY PF 1.494/SELL PF 0.840; AUDJPY: BUY 1.596/SELL 0.710; CADJPY: BUY 1.432/SELL 0.764) — reconfirming the pre-existing directional-asymmetry research this project has documented repeatedly. **Mapped to currency factors: every AMR pair's weak side is specifically "long JPY"** (SELL on a JPY-quote pair = short base / long JPY) — meaning the portfolio's directional weakness and its currency concentration are the same underlying pattern viewed two ways, not two independent findings.

---

## 13. Holding period factor

`<2h` trades dominate every AMR pair (339-442 trades each) and are strongly net-positive across the board; `2-12h` buckets are consistently the weakest for the AMR family (e.g. CADJPY AMR 2-6h: −14.66R, 6-12h: −15.85R) — trades that don't resolve quickly tend to be the AMR family's losing trades, consistent with a mean-reversion mechanic that works when the reversion happens fast and decays when it doesn't. CADJPY ARB and GBPUSD Monday are structurally different — both are >24h-hold-dominant strategies by design (ARB: 82 of 192 trades >24h, net +41.98R; Monday: 148 of 154 trades in the 12-24h bucket, its full weekly design).

---

## 14. Portfolio risk concentration

Full CSV: `reports/portfolio_factor_summary.csv`. By strategy: EURJPY AMR (24.6%) and AUDJPY AMR (22.4%) are the two largest risk-weighted allocations; GBPUSD Monday (5.3%) the smallest. **Risk contribution and trade count are explicitly distinguished** throughout this report and its CSVs — e.g. EURJPY AMR has the most historical trades (713) but CADJPY ARB's larger per-trade risk_pct (0.50% vs. 0.25%) means trade count alone would understate its risk contribution, which is why every risk figure in this report is trade-count × risk_pct weighted, not a raw trade tally.

---

## 15. Effective diversification

| Measure | Value | Interpretation |
|---|---|---|
| Naive effective N (risk-weight HHI only) | **5.19 of 6** | Ignores correlation — overstates true diversification |
| **Correlation-adjusted effective N** (1/(w′Σw), daily-R Pearson matrix) | **2.67 of 6** | The more honest measure — mathematically justified, not fabricated (standard portfolio-theory "effective number of bets" formula) |
| Average pairwise daily-R correlation | 0.192 | Modest positive, concentrated among the 4 AMR pairs specifically (0.25-0.41) |
| Days with 2+ simultaneous losses | 31.0% of 774 | |
| Days with 3+ simultaneous losses | 15.4% of 774 | |

**The portfolio behaves, in risk terms, closer to 2-3 independent strategies than 6** — this is the single clearest quantitative answer to "how diversified is the portfolio really."

---

## 16. Missing-factor analysis and candidate profile

Full detail: `reports/portfolio_missing_factor_analysis.md` and `reports/portfolio_candidate_profile.md`. Summary: the portfolio's clearest gaps are (1) zero New York-session exposure, (2) no HIGH-volatility-compatible strategy, (3) elevated drawdown-day correlation among the AMR family, and (4) correlation-adjusted effective diversification well below the nominal strategy count — with JPY concentration as a real but not sole contributor (session and mechanism concentration are at least as implicated).

---

## 17. AUDUSD Monday LONG candidate comparison (diagnostic only — not a promotion)

Full detail: `reports/portfolio_candidate_profile.md` §"AUDUSD Monday LONG checked against this profile". **Key finding: AUDUSD Monday LONG's HIGH-volatility performance (mean R +0.248/trade, its single best of three vol terciles) is a genuinely strong match for the portfolio's #1 quantified gap (§9/§11)** — but it does not address the session gap (still a Monday-only, 00:00-server design) and its correlation to the existing book (0.29) is actually **above**, not below, the current six's own 0.192 internal average. **Mixed result, reported as such — not promoted.**

---

## 18. Future non-JPY research profile

Full detail: `reports/portfolio_candidate_profile.md` §19. Summary: non-JPY, genuine London/NY session activity (not Asian or start-of-week), trend-following/regime-aware (not another trend-filter-free mean-reversion mechanic), historically HIGH-vol-compatible, correlation to the existing book below ~0.19 with an explicit drawdown-day correlation check.

---

## 19. Portfolio stress-test scenarios

| Scenario | Historical reconstruction possible? | Result |
|---|---|---|
| A. JPY shock | Partially — via HIGH-volatility regime data (§9), since a JPY-wide shock would plausibly manifest as elevated ATR across the JPY-cross AMR pairs simultaneously | Combined R −6.91 in HIGH vol across the 3 affected strategies — the closest reconstructable proxy for this scenario |
| B. High volatility | **Yes — directly reconstructed** (§9) | Portfolio's only net-negative regime |
| C. Strong trend | Partially — via trend_tercile (§9) | HIGH_TREND is mildly net-positive in aggregate (+56.49), with 1 of 6 strategies net-negative — not a severe scenario by this data |
| D. Asian-session adverse regime | **Not separately reconstructable from session alone** — 94.7% of the book already trades in this session under all conditions, so "Asian-session adverse" collapses into the HIGH-volatility scenario (B) rather than being a distinct, isolatable event in this dataset | See scenario B |
| E. Broad risk-off environment | **NOT AVAILABLE** — this dataset has no independent risk-on/risk-off classification (e.g. VIX-equivalent or cross-asset regime tag); building one would require a new data source/methodology, out of scope for this phase | Limitation stated explicitly, not estimated |
| F. Multiple simultaneous strategy failures | **Yes — directly reconstructed** (§6/§8) | 31.0% of days see 2+ losses, 15.4% see 3+; correlation on the worst-decile days is elevated for most AMR pairs |

---

## 20. Decision framework answers

1. **Is JPY concentration actually harmful?** Partially — it's the currency-level expression of findings that are at least as much about session and mechanism concentration (§16).
2. **Is strategy-family concentration harmful?** Yes — the clearest single concentration in the book (81.5% mean-reversion, §4), and the direct driver of the AMR-pair correlation cluster (§7).
3. **Is session concentration harmful?** Yes, in the sense that it's total (100% of trades occur in only 2 of the day's 3 major sessions, 0% in NY) — whether this has *caused* measurable harm beyond the volatility/correlation findings already documented is not separately established, but the concentration itself is unambiguous.
4. **Are current strategies correlated primarily because of JPY?** **Not established as the primary cause** — CADJPY ARB is also JPY (CAD/JPY) yet correlates weakly with the AMR pairs (0.05-0.11), while the AMR pairs correlate strongly with each other (0.25-0.41). **Mechanism and session match the correlation pattern better than currency alone does.**
5. **Are they correlated because of mechanism?** **Yes — best-supported answer.** The four same-mechanism AMR pairs are exactly the four highest-correlating strategies in the matrix.
6. **Are they correlated because of session?** Contributing, inseparable from #5 in this dataset (all four AMR pairs share both mechanism and session simultaneously — this analysis cannot fully decompose the two).
7. **Which factor contributes most to portfolio drawdown?** Regime (HIGH volatility, §9) combined with mechanism/session correlation (§8) — the drawdown-day correlation spike (§8) is the mechanism by which a HIGH-vol regime turns into a multi-strategy simultaneous loss event rather than an isolated one.
8. **Which regime causes the most simultaneous losses?** HIGH volatility (§9/§10).
9. **Single biggest diversification opportunity?** A strategy addressing the zero-NY-session gap (§16, most unambiguous finding).
10. **Second-best?** A HIGH-volatility-compatible strategy (§16, most concretely quantified gap).

---

## 21. Limitations

- Trade-level correlation is not meaningful for this book (documented in §2, not silently computed).
- Regime definitions are the existing vol/trend terciles already in `phase26_all_trades.csv` — no independent regime model was built or validated fresh in this phase.
- The "risk-off" stress scenario (§19E) could not be reconstructed from available data.
- GBPUSD Monday's `session` field in the source data is a labeling artifact, not a true session classification (§2) — disclosed, not silently used.
- The live production population (33-36 current-six trades) is far too small for its own correlation/regime/drawdown analysis — this report relies on the larger historical reconstruction (2,712 trades) throughout, with live data used only for scorecard-status cross-references already established in prior phases.
- Effective-N and correlation figures are historical (backtest-reconstruction-based); they characterize the *strategies' designed behavior*, not necessarily the live book's exact forward correlation, which remains too small a sample to measure directly.

---

## 22. Final verdict

**Portfolio classification: C. HIGHLY CONCENTRATED.**

Not A (well diversified) — correlation-adjusted effective N of 2.67 (of 6) and the drawdown-day correlation spike (§8) both directly contradict "well diversified."
Not B (moderately concentrated) — the concentration is not incidental or mild; 81.5% of risk-weighted trades share one mechanism, and every strategy shares one of only two sessions.
Not D (fragile/single-factor dominated) — the portfolio does have real structural variety (three distinct mechanism families, two distinct sessions, one non-JPY strategy) and multiple regimes (LOW/NORMAL volatility, LOW_TREND) where all six strategies are simultaneously profitable — this is concentration, not the near-total absence of diversification that "fragile/single-factor" would imply.

**Should we add a non-JPY strategy? INSUFFICIENT EVIDENCE to say a *non-JPY* strategy specifically is the fix** — the evidence (§16, §20 Q4-6) more strongly implicates session and mechanism concentration than currency alone. **A future candidate should be evaluated against the fuller profile in §16/§18, not selected on "non-JPY" as the primary criterion.**

**What should the next research candidate look like?** A profile, not a pair (full detail §18): non-JPY, genuine London/NY session, trend-aware mechanism (not another filter-free mean-reversion strategy), HIGH-volatility-compatible, and — the criterion this phase adds beyond what Phase 30 already established — correlation to the existing book below its own internal 0.192 average, checked specifically on drawdown days, not just average days.

---

## 23. Dashboard specification (not implemented — specification only)

**Portfolio panel:**

| Metric | Source |
|---|---|
| PORTFOLIO FACTOR MAP | `portfolio_currency_factor_map.csv`, refreshed per new closed trade |
| JPY EXPOSURE | % of risk-weighted trades/risk touching JPY |
| CURRENCY CONCENTRATION | Risk-weighted % by currency (§3) |
| STRATEGY FAMILY EXPOSURE | Risk-weighted % by mechanism (§4) |
| SESSION EXPOSURE | Risk-weighted % by session (§5), flagging the 0% NY figure prominently |
| REGIME EXPOSURE | Live trade count by vol/trend tercile, with a flag if HIGH-vol trade count is accumulating |
| CORRELATED LOSS DAYS | Rolling count of days with 2+/3+ simultaneous strategy losses |
| DRAWDOWN CONTRIBUTORS | Per-strategy $ /R contribution during the account's current drawdown, updated live |
| EFFECTIVE DIVERSIFICATION | Correlation-adjusted effective N, recomputed as live data accumulates (flagged NOT AVAILABLE below a minimum sample, per this project's established sample-size discipline) |

**Per-strategy panel:** FACTOR (currency + family), REGIME (current vol/trend bucket + historical PF in that bucket), SESSION, DIRECTION (BUY/SELL split), CORRELATION (to portfolio, rolling), DRAWDOWN CONTRIBUTION (current period).

**Not implemented in this phase**, per instruction.

---

## Safety check confirmation

All items in the task's final safety checklist confirmed: six live strategies unchanged, no production config/parameters/risk changed, no strategy paused, no candidate deployed, current 5ers export not committed, data validator used on both inputs before analysis, no silent CSV correction, no time-series shuffling, no future leakage (all analysis uses historical trade-level timestamps as-recorded), correlation methodology documented (§2), missing data handled explicitly (NaN not zero-filled, §2), regime definitions sourced from existing research (§2/§9), no new regime model invented, AUDUSD candidate not promoted (§17), no portfolio optimization performed anywhere in this phase.

---

*Prepared for TheImperfectAlgorithm. Reproducible via `python src/phase31_factor_regime_map.py`. No trading changes made.*
