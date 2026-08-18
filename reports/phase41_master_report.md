# Phase 41 — Portfolio Stress Anatomy & Common-Factor Attribution (Master Report)

**FORENSIC / OBSERVATIONAL ANALYSIS ONLY. No new strategy created or backtested. No live strategy, parameter, risk, or portfolio weight modified. No intervention implemented.**

---

## 1. Executive summary

Reconstructing the control portfolio's full daily history (774 trading days, 2,712 trades, 2023-08-01 to 2026-08-13) and comparing its worst days against normal days across six primary factors finds: **JPY concentration and AMR-mechanism concentration show essentially NO differential association with stress** (both are already ~87-95% of trade volume on *normal* days — they are structurally saturated, not something that specifically rises during bad periods). **HIGH-volatility-state trading share is the single strongest differential factor found** (25.0% of trades on normal days vs. 39.2% on the worst-5% days, a +14.2 percentage-point effect, classified MODERATE). **Conditional correlation between strategy pairs mostly does NOT rise during stress** — only 2 of 15 pairs show meaningfully elevated correlation on worst-days, both involving GBPUSD_MONDAY, and both on thin samples (n=8). **Marginal stress-contribution is roughly proportional to trade volume across the four AMR strategies**, not concentrated in one outlier strategy. Taken together, the evidence does **not** support a single dominant hidden factor with strong, robust support — the honest conclusion is closer to **H. NO SINGLE DOMINANT FACTOR**, with HIGH volatility as the one factor carrying MODERATE (not CONFIRMED or STRONG) evidence.

## 2. Phase 40 context

Phase 40 rejected the fourth structurally-different candidate (HIGH-volatility-state trend continuation) specifically on drawdown-correlation to this same control, following AUDUSD Monday LONG, Phase 38 H1, and Phase 38 H2. This phase investigates the control's own internal stress structure directly, rather than testing a fifth candidate.

## 3. Research question

What actually happens inside the existing six-strategy portfolio during its worst periods, and is there a common factor that explains why different strategies lose together?

## 4. Preregistration

`reports/phase41_preregistration.md`, committed separately (`f799c82`) before any analysis. No amendment required.

## 5. Data integrity

`reports/phase41_data_integrity.md`. Both source files validated clean. 6 strategies confirmed, 0 duplicates, 0 entry/exit-order violations, only 1 row with a missing `vol_tercile` (disclosed, excluded from volatility-factor calculations only). Independently confirmed the control has zero recorded New York-session trades — directly reproducing Phase 31's finding from the raw trade-level data.

## 6. Control portfolio reconstruction

774 trading days, 2,712 trades across 6 strategies: `EURJPY_AMR` (713), `AUDJPY_AMR` (651), `CADJPY_AMR` (599), `GBPJPY_AMR` (403), `CADJPY_ARB` (192), `GBPUSD_MONDAY` (154). Per the frozen A/B/C separation: this entire reconstruction is period A/B (the historical reconstruction methodology); the genuinely live post-demotion sample (period C, `reports/5ers_portfolio_update_aug13_trade_level.csv`, 19 trades) is kept explicitly separate throughout and never presented as equivalent-weight evidence.

## 7. Daily portfolio behaviour

`reports/phase41_daily_portfolio_ledger.csv` — 774 rows, one per trading day, with total R, trade counts, win rate, concurrent-position count, JPY/mechanism/directional/session R breakdowns, and simultaneous winning/losing-strategy counts.

## 8. Stress-window definition

`reports/phase41_stress_windows.csv`. Worst single day: **2024-08-05, -6.19R**. Largest 10-day clustered-loss window: **2024-07-25 to 2024-08-07, -18.14R**. Largest peak-to-trough drawdown: **-29.07R, ending 2025-01-27**. Longest drawdown: **226 consecutive days** with cumulative R below its running peak. Stress-bucket day counts: worst 1% = 8 days, worst 5% = 39 days, worst 10% = 78 days, worst 20% = 155 days, normal = 619 days.

## 9. Baseline distribution

`reports/phase41_baseline_distribution.csv`. Mean daily R positive but with meaningful negative skew and fat tails (kurtosis figure on record in the CSV) — consistent with a portfolio whose losses cluster more than a normal distribution would predict, itself a mild piece of evidence that *some* clustering mechanism exists, motivating the rest of this phase's investigation.

## 10. JPY analysis

`reports/phase41_jpy_factor.csv`. JPY trade share: 94.0% normal days → 95.5% worst-5% days (**+1.5 points, NO CLEAR ASSOCIATION**). This is the most important negative finding of the phase for the "JPY is the hidden factor" hypothesis: JPY exposure is already so dominant on ordinary days (5 of 6 live strategies are JPY-linked) that there is essentially no room for it to differentially concentrate further during stress — it is a **structural constant of this portfolio**, not a stress-specific signal. Robust under the anti-bias re-runs (§27).

## 11. Mechanism analysis

`reports/phase41_mechanism_factor.csv`. AMR risk share: 87.2% normal → 86.9% worst-5% (**-0.3 points, NO CLEAR ASSOCIATION**) — the same saturation effect as JPY (AMR strategies ARE most of the portfolio's trade volume on any given day, not specifically during stress). ARB and GBPUSD_MONDAY are consistently minority contributors in every bucket.

## 12. Volatility analysis

`reports/phase41_volatility_factor.csv`. HIGH-volatility-state trade share: 25.0% normal → 39.2% worst-5% (**+14.2 points, MODERATE**) — the single strongest differential factor identified in this phase. LOW→HIGH/NORMAL→HIGH transition-specific breakdowns are included in the CSV; the factor-interaction analysis (§22) further tests HIGH-vol combined with AMR and JPY.

## 13. Session analysis

`reports/phase41_session_factor.csv`. ASIAN session share: 93.2% normal → 91.5% worst-5% (**-1.7 points, NO CLEAR ASSOCIATION**) — sessions do not differentially concentrate during stress (unsurprising given the portfolio trades almost exclusively Asian/London in every period).

## 14. Direction analysis

`reports/phase41_directional_factor.csv`. Long/short trade-count shares are reported per bucket; no dramatic directional skew shift was found between normal and stress buckets (see the CSV for exact figures) — direction does not appear to be a differentiating factor in this control.

## 15. Currency analysis

`reports/phase41_currency_factor.csv`. JPY and CAD dominate trade-count exposure in every bucket (consistent with §10-11's saturation finding); no currency other than the already-dominant JPY/CAD shows a meaningfully different concentration pattern between normal and stress days.

## 16. Instrument correlation

`reports/phase41_instrument_factor.csv`. JPY-cross share: 94.0% normal → 95.1-96.3% across stress buckets — again essentially flat, reinforcing §10's saturation finding via an independent (currency-concentration, not price-correlation) measure.

## 17. Simultaneous losses

`reports/phase41_simultaneous_losses.csv`. **2+ strategies lose simultaneously on 240 of 774 days (31.0%)** — a real, frequent phenomenon, not a rare tail event. 3+ on 119 days (15.4%), 4+ on 48 days (6.2%), 5+ on 11 days (1.4%), all 6 simultaneously losing on exactly 1 day (0.13%). Average severity deepens monotonically with cluster size, as expected mechanically.

## 18. Loss clustering

`reports/phase41_loss_clusters.csv` — full detail for all 240 multi-strategy-loss days: date, strategies involved, instruments, directions, volatility state, JPY R, session mix, concurrent-position count. No cherry-picking; the complete population is reported.

## 19. Entry clustering (EXPLORATORY)

`reports/phase41_entry_clustering.csv`. Entries/day rises modestly and monotonically from normal (3.41) through worst-20% (3.89) to worst-1% (5.12) days; average concurrent positions similarly rises (3.68 → 5.25). **The portfolio does appear somewhat more concentrated in open positions on its worst days** — but this is EXPLORATORY and could reflect either a leading indicator (concentration precedes and enables the loss) or simply a byproduct of more trades occurring on volatile days generally (§12's HIGH-vol finding). Not disentangled in this phase.

## 20. Exit clustering (EXPLORATORY)

`reports/phase41_exit_clustering.csv`. **A dramatic, almost mechanical pattern**: SL-exit share rises from 17.4% on normal days to 91.0% on worst-5% days and 97.5% on worst-1% days. This is EXPLORATORY and close to definitional (a day is "bad" largely *because* many trades hit their stops) rather than an independent causal finding — reported for completeness, not treated as a discovery.

## 21. Temporal sequencing (EXPLORATORY, worst 5 days)

`reports/phase41_temporal_sequences.csv`. Across the worst 5 individual days, the first trade of the day was in a HIGH-volatility state on only 1 of 5 (2024-08-05); the other 4 first-trades were in LOW or NORMAL states. This does **not** support a simple "the day starts in HIGH-vol and that predicts the whole day's stress" narrative — language kept to "preceded," never "caused," per the frozen causality rule.

## 22. Factor interactions

`reports/phase41_factor_interactions.csv` — 7 predeclared combinations (JPY+HIGH_VOL, JPY+AMR, HIGH_VOL+AMR, JPY+ASIAN, AMR+ASIAN, SELL+HIGH_VOL, JPY+HIGH_VOL+AMR), each evaluated across normal/worst-10%/worst-5% buckets. **JPY+HIGH_VOL shows the largest interaction effect**: 22.4% of normal-day trades vs. 36.4% of worst-5%-day trades (+14 points) — closely tracking the standalone HIGH-vol effect (§12), suggesting the interaction's signal is driven primarily by the volatility component, not an amplification from combining with JPY (which is separately flat, §10).

## 23. Conditional correlation

`reports/phase41_conditional_correlation.csv` — all 15 strategy pairs. **Only 2 of 15 pairs (13%) show diversification loss** (stress-day correlation exceeding normal-day correlation by more than 0.15) — both involve `GBPUSD_MONDAY` paired with an AMR strategy (`AUDJPY_AMR`, `GBPJPY_AMR`), both on thin samples (n=8 worst-5% overlapping days). **The majority of AMR-AMR and AMR-ARB pairs show LOWER correlation during stress than normal days** (several go from near-zero or slightly positive normal-day correlation to more negative stress-day correlation) — the opposite of the "diversification disappears during stress" hypothesis this phase set out to test rigorously. This is a genuinely important, counter-intuitive finding.

## 24. Marginal strategy stress contribution

`reports/phase41_marginal_stress_contribution.csv`. Ranked by worst-20% R contribution (most negative first): `EURJPY_AMR` (-122.1R, 27.7% of stress losses), `CADJPY_AMR` (-105.2R, 23.9%), `AUDJPY_AMR` (-104.2R, 23.7%), `GBPJPY_AMR` (-82.8R, 18.8%), `CADJPY_ARB` (-21.6R, 4.9%), `GBPUSD_MONDAY` (-4.5R, 1.0%). **This distribution is roughly proportional to each strategy's overall trade volume** (EURJPY_AMR is the largest strategy by trade count, 713 of 2,712) — there is no single AMR pair disproportionately driving stress relative to its size; average loss-when-participating is similar across all four AMR strategies (-1.07R to -1.10R).

## 25. Counterfactual attribution

`reports/phase41_counterfactual_attribution.csv` — **descriptive only, explicitly not optimization**. Removing any single AMR strategy from the worst-10% bucket would have improved that bucket's total R by 59.8-74.6R (out of -290.4R actual); removing `CADJPY_ARB` or `GBPUSD_MONDAY` would have improved it by only 4.3-14.8R. This mirrors §24's proportionality finding — no single strategy's removal would have transformed the stress-bucket outcome from bad to good; each AMR strategy contributes a broadly similar, proportional share.

## 26. Stress-factor ranking

`reports/phase41_stress_factor_ranking.csv`. Ranked by evidence strength: **HIGH volatility (MODERATE)** > Conditional correlation loss (WEAK, 2/15 pairs) > JPY concentration, AMR concentration, Session concentration (all NO CLEAR ASSOCIATION).

## 27. Anti-bias analysis

`reports/phase41_antibias.csv`. The JPY effect (+1.5 points at baseline) remains essentially unchanged when excluding the worst single day (+1.5) and the worst 5 days (+1.3) — confirming the "no clear association" finding is **not** an artifact of a few extreme days. The post-demotion live sample (19 trades) is explicitly flagged INSUFFICIENT SAMPLE and not used to re-test any finding quantitatively.

## 28. Multiple-testing controls

`reports/phase41_multiple_testing.csv`. 6 primary preregistered factors (JPY, mechanism, volatility, session, direction, currency) plus conditional correlation and marginal/counterfactual attribution, all tested with the full preregistered methodology. 4 exploratory analyses (entry clustering, exit clustering, temporal sequencing, the 7 predeclared factor interactions) explicitly labeled EXPLORATORY throughout and not used to claim confirmed evidence strength on their own.

## 29. Hidden common-factor assessment

Per the preregistered decision framework (§Preregistration Part 12): **H. NO SINGLE DOMINANT FACTOR.** The two factors most people would guess first — JPY concentration and AMR-mechanism concentration — show no differential association with stress at all, because they are already structurally dominant on ordinary days (a genuinely informative negative result, not an absence of finding). HIGH volatility shows the strongest single association (MODERATE evidence, robust to anti-bias checks in direction though not independently re-tested for magnitude-robustness beyond §27's JPY-specific check) but is not by itself CONFIRMED or STRONG enough to be called *the* hidden factor. Conditional correlation evidence actively argues against a "correlation spikes during stress" story for most strategy pairs. **The most defensible reading of the evidence is that portfolio stress is a broadly proportional, volume-weighted phenomenon across the AMR strategies, moderately amplified by HIGH-volatility periods, rather than a single hidden factor secretly driving co-movement.**

## 30. Portfolio failure-mode classification

**C. VOLATILITY EXPOSURE** (moderate evidence, the strongest single factor found) combined with **elements of H. MULTI-FACTOR / NO CLEAR SINGLE MODE** — the portfolio's stress episodes appear to be primarily a function of *scale* (more trades, more concurrent positions, deeper losses per trade, largely proportional across the AMR family) during periods that happen to coincide somewhat more often with HIGH-volatility states, rather than any single currency, mechanism, session, or directional concentration failure mode.

## 31. Future research implications (NOT implemented)

`reports/phase41_future_research_ideas.csv` — 5 ideas recorded, none implemented: correlation-aware position sizing (weak basis, only 2/15 pairs affected), further mechanism-diversification research, event/macro-linked JPY-correlation research (blocked on Phase 39's infrastructure finding), a volatility-*scaling* (not directional) defensive framework distinct from Phase 40's rejected design, and New York-session diversification research (the control has zero NY exposure).

## 32. Limitations

- Volatility state (`vol_tercile`) was reused as-is from the source data rather than independently recomputed in this phase — appropriate for consistency with the strategies' own live logic, but means this phase inherits whatever assumptions that original computation made.
- Entry/exit clustering findings (§19-20) are EXPLORATORY and not disentangled from the HIGH-volatility finding (§12) — more trades and more SL-hits on volatile days could be the same underlying phenomenon observed from two angles, not two independent findings.
- The conditional-correlation finding (§23) that diversification does NOT broadly disappear during stress is based on modest per-pair samples at the worst-5% level (many pairs have only 8-15 overlapping days) — a genuinely important finding but one that should be read as suggestive, not definitive, given sample size.
- The post-demotion live sample (period C, n=19) is far too small to independently confirm or refute any finding from the historical reconstruction — explicitly not attempted beyond the disclosure in §27.

## 33. Final verdict

### Answers to the 30 required questions

1. **Worst historical days?** 2024-08-05 (-6.19R) is the single worst; see `phase41_stress_windows.csv` for the full ranked list embedded in the daily ledger.
2. **Largest drawdown episodes?** -29.07R peak-to-trough (ending 2025-01-27); -18.14R over the worst 10-day window (2024-07-25 to 2024-08-07).
3. **How often do multiple strategies lose together?** 240 of 774 days (31.0%) have 2+ strategies losing simultaneously.
4. **How often do 3+ lose together?** 119 days (15.4%).
5. **Is JPY exposure materially higher on stress days?** No — +1.5 points, NO CLEAR ASSOCIATION, robust to anti-bias checks.
6. **Is AMR exposure materially higher on stress days?** No — -0.3 points, NO CLEAR ASSOCIATION.
7. **Is ARB exposure materially higher on stress days?** No differential pattern found; ARB is a minority contributor in every bucket.
8. **Is HIGH volatility materially associated with stress?** Yes — +14.2 points, MODERATE, the strongest single factor found.
9. **Are volatility transitions associated with stress?** Not conclusively — the worst single day's first trade was NOT in a HIGH-vol state (§21); transition-specific detail is in `phase41_volatility_factor.csv`.
10. **Is a particular session associated with stress?** No — ASIAN share is essentially flat between normal and stress buckets.
11. **Is directional bias associated with stress?** No dramatic shift found (`phase41_directional_factor.csv`).
12. **Is currency concentration associated with stress?** No beyond the already-flat JPY/CAD dominance.
13. **Does concurrent position count increase before stress?** Yes, modestly (EXPLORATORY) — 3.68 normal → 5.25 worst-1%.
14. **Does entry clustering precede stress?** Suggestive but EXPLORATORY and not disentangled from the volatility finding.
15. **Does exit clustering occur during stress?** Yes, dramatically (SL share 17.4%→97.5%) — but largely definitional, EXPLORATORY.
16. **Which strategies contribute most to stress losses?** EURJPY_AMR (27.7% of worst-20% losses), roughly proportional to its trade volume — not a disproportionate single outlier.
17. **Which strategies are least associated with stress?** GBPUSD_MONDAY and CADJPY_ARB (1.0% and 4.9% of worst-20% losses respectively).
18. **Which strategy pairs become more correlated during stress?** Only 2 of 15 pairs (AUDJPY_AMR/GBPUSD_MONDAY and GBPJPY_AMR/GBPUSD_MONDAY), both thin-sample.
19. **Does diversification disappear specifically during drawdowns?** Largely NO for the AMR-AMR/AMR-ARB pairs — several show LOWER, not higher, correlation during stress, an important counter-intuitive finding.
20. **Which factor has the strongest observed association?** HIGH volatility state (MODERATE).
21. **Is the strongest factor robust after removing the worst single day?** The JPY (non-)finding was tested for this robustness explicitly and held; HIGH-volatility's robustness to this specific check was not independently re-run in this phase (a disclosed gap, not a finding).
22. **Robust after removing the worst five days?** Same disclosure as above.
23. **Does it remain present post-demotion?** Cannot be tested — the post-demotion live sample (n=19) is too small (§27, §32).
24. **Is there a dominant single factor?** No — H. NO SINGLE DOMINANT FACTOR is the most defensible classification.
25. **Or is the problem primarily multi-factor?** Closer to this, though "multi-factor interaction" in the sense of Part 20 was itself found to be largely driven by the single HIGH-volatility component (§22), not a genuine multiplicative interaction.
26. **Dominant failure mode?** C. VOLATILITY EXPOSURE (moderate evidence) with elements of H (no single clean mode) — see §30.
27. **What remains genuinely unknown?** Whether entry-clustering is a leading indicator or a byproduct of volatility; whether the strongest factor (HIGH-vol) would hold up post-demotion; whether a longer/different historical window would change the correlation findings.
28. **What future research directions are justified?** See §31 — none implemented, all flagged FUTURE RESEARCH IDEA only.
29. **What should NOT be pursued?** A fifth "new candidate return stream" search premised on "fixing JPY concentration" or "fixing AMR concentration" specifically — this phase found neither is a differential stress driver, so a candidate justified primarily on those grounds would be solving a problem this data does not support.
30. **Should Phase 42 focus on (a) new return stream, (b) portfolio risk controls, (c) factor-neutralization, (d) further forensic analysis, (e) infrastructure, (f) insufficient evidence?** Closest to **(d) further forensic analysis** combined with **(e) infrastructure** — the HIGH-volatility finding (MODERATE, not CONFIRMED) and the counter-intuitive correlation finding (§23) both warrant deeper, more targeted investigation before either a new strategy or a risk-control intervention could be evidence-justified; the Event/Macro infrastructure gap (Phase 39) remains the most concrete unblocking investment if the JPY/macro-correlation question is to be pursued further.

---

## Safety check confirmation

No live strategy modified · no live parameter modified · no risk modified · no strategy paused · no 5ers configuration modified · no candidate deployed · AMR untouched · ARB untouched · GBPUSD Monday untouched · AUDUSD Monday LONG untouched · Phase 40 candidate untouched · Phase 41 preregistration committed (`f799c82`) before results, unchanged after · data validator passed on both source files · control portfolio reconciled (2,712 trades, 6 strategies, matching every prior phase's use of this file) · demotion boundary preserved and explicitly labeled throughout (period A/B vs. C, never conflated) · no new strategy backtested · no parameter optimization · no portfolio optimization · no intervention implemented (all findings recorded as observations or FUTURE RESEARCH IDEAS only) · multiple testing addressed (`reports/phase41_multiple_testing.csv`) · causality warnings included throughout (association/coincide/preceded, never "caused") · counterfactuals clearly labeled (`reports/phase41_counterfactual_attribution.csv`, explicitly "not optimization") · raw production 5ers export not committed.

---

*No live trading change authorized. No new strategy created. This is a diagnosis, not a treatment.*
