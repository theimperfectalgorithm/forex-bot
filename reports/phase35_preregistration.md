# Phase 35 Pre-Registration — FROZEN BEFORE ANY CANDIDATE RESULT IS EXAMINED

**Written and committed before any candidate backtest is run. Not changed based on results. Any later methodological flaw is documented as a dated amendment in §13, never a silent edit.**

Continues [[Phase 34]]'s finding (B. SEARCH WAS TOO NARROW) by testing the 5 hypotheses Phase 34 identified — H1-H5 — each with exactly one pre-selected instrument and one economically-motivated parameter set, no grid search. The Phase 32 target profile is unchanged and not modified here.

---

## 1. Research universe (frozen)

Eligible instruments: AUDUSD, USDCAD, USDCHF, XAUUSD (EURUSD/GBPUSD remain excluded as settled dead ground, per Phase 33's unchanged finding). **One instrument is pre-assigned per hypothesis below — no hypothesis is tested on more than one instrument, and no instrument is swapped after seeing results.**

| Hypothesis | Instrument | Why this instrument, chosen before results |
|---|---|---|
| H1 — NY Open Range Breakout | **USDCAD** | Matches Phase 34's own search-map recommendation; non-JPY; distinct from H2's instrument |
| H2 — NY Session Momentum | **AUDUSD** | Deliberately different instrument from H1, per Phase 34's own reasoning ("reducing the risk that a single implementation choice determines the family's fate") |
| H3 — London/NY Overlap Continuation | **USDCHF** | The third instrument named in Phase 34's search map for this family, not yet used by H1/H2 |
| H4 — Multi-Timeframe Trend Continuation | **USDCAD** | Direct instrument retest per Phase 34's explicit recommendation — a *differently designed* strategy, not a re-run of Phase 33's rejected candidate |
| H5 — ATR-Scaled Volatility Expansion | **AUDUSD** | Deliberately avoids gold, per Phase 34's diagnosis that XAUUSD's drawdown-correlation failure plausibly reflects gold's own macro/hedge role, not the volatility-expansion mechanism itself |

## 2. Data periods and split (frozen, identical convention to Phase 33 for comparability)

- **Data range:** 2023-01-01 to 2026-08-14, MetaQuotes-Demo broker feed (disclosed limitation, unchanged from every prior phase).
- **TRAIN:** 2023-01-01 to 2024-08-31 (~20 months).
- **VALIDATION:** 2024-09-01 to 2025-04-30 (~8 months).
- **OOS:** 2025-05-01 to 2026-08-14 (~15.5 months) — never inspected before this point.
- All five parameter sets (§3-7) are fixed by economic reasoning before any window is inspected. VALIDATION serves as an intermediate integrity check only, never a tuning fold.

## 3. H1 — NEW YORK OPEN RANGE BREAKOUT (frozen mechanics)

- **Instrument/timeframe:** USDCAD, H1.
- **Range construction:** the first 2 hours of the NY session (13:00-15:00 UTC) — high/low of those H1 bars.
- **Breakout condition:** at or after 15:00 UTC, the first H1 close beyond the range high/low.
- **Stop:** opposite side of the range.
- **Target:** 1.5x the range width **(key parameter for the ±20% robustness check → 1.2x / 1.8x)**.
- **Session cutoff / max holding:** entries only considered through 20:00 UTC same day; positions capped at 48 H1 bars (~2 days) before being excluded from analysis if unresolved (documented limitation, not force-closed, matching Phase 33's convention).

## 4. H2 — NEW YORK SESSION MOMENTUM (frozen mechanics)

- **Instrument/timeframe:** AUDUSD, H1 signal and execution.
- **Genuinely momentum-based, not "AMR with a different session":** no mean-reversion component; entry follows, not fades, a confirmed directional move.
- **Momentum definition:** the absolute 3-hour price change (current H1 close vs. close 3 bars prior), compared against the instrument's own rolling 20-day average 3-hour absolute move. A signal fires when the current 3-hour move exceeds **1.0x** that rolling average **(key parameter → 0.8x / 1.2x)**, during NY session hours (13:00-21:00 UTC).
- **Direction:** the direction of the 3-hour move itself (continuation, not fade).
- **Exit:** SL at 1.0x ATR(14, H1); TP at 2.0x ATR(14, H1).
- **Max holding period:** entries only within 13:00-20:00 UTC; positions capped at 24 H1 bars.

## 5. H3 — LONDON/NEW YORK OVERLAP CONTINUATION (frozen mechanics)

- **Instrument/timeframe:** USDCHF, H1.
- **Overlap window:** 13:00-16:00 UTC (NY open through London close).
- **Continuation definition (genuinely different from H1 — a session-level directional-persistence test, not a range breakout):** the London session (07:00-13:00 UTC) is evaluated for directional efficiency — net displacement over that window divided by the sum of absolute H1 bar-to-bar moves. If this efficiency ratio exceeds **0.40** **(key parameter → 0.32 / 0.48)**, enter in the London session's own net direction at 13:00 UTC.
- **Exit:** SL at 1.0x ATR(14, H1); TP at 2.0x ATR(14, H1); force-flat at 16:00 UTC if neither is hit (a session-bounded design, distinct from H1's multi-day-capped breakout).
- **Max holding period:** bounded by the 13:00-16:00 UTC window itself.

## 6. H4 — MULTI-TIMEFRAME TREND CONTINUATION (frozen mechanics — NOT a repair of Phase 33's USDCAD candidate)

- **Instrument:** USDCAD. **Higher timeframe:** D1. **Execution timeframe:** H4.
- **Trend definition (D1):** bullish if today's D1 close > D1 close 20 trading days ago; bearish if the reverse. This is a slower, structurally different filter from Phase 33's single-H4-threshold design — the entire design intent being to test whether adding a higher-timeframe confirmation layer produces the "broad plateau" Phase 34 found missing.
- **Continuation/entry definition (H4):** only when the D1 filter confirms a direction, enter on an H4 close beyond its own prior **10-bar** high/low **(key parameter → 8 / 12 bars)** in the confirmed direction.
- **Exit:** SL at 1.5x ATR(20, H4); TP at 3.0x ATR(20, H4).
- **Max holding period:** positions capped at 60 H4 bars (~10 trading days).

## 7. H5 — ATR-SCALED VOLATILITY EXPANSION (frozen mechanics — NOT a parameter optimization of Phase 33's XAUUSD candidate)

- **Instrument/timeframe:** AUDUSD, H1.
- **Precondition (unchanged concept from Phase 33, applied to a new instrument):** at the London open (07:00 UTC), the preceding 4-hour realized range (03:00-06:00 UTC) must sit below its own rolling 30-day 33rd percentile.
- **Entry:** the first H1 close beyond that pre-London range, at or after 07:00 UTC.
- **Core design change (the specific hypothesis under test):** exit distance is scaled to REALIZED VOLATILITY AT ENTRY, not to the precondition range's fixed width. **Stop:** 1.0x ATR(14, H1) at the entry bar. **Target:** **2.5x** ATR(14, H1) at the entry bar **(key parameter → 2.0x / 3.0x)**.
- **Max holding period:** capped at 200 H1 bars (~8 days), matching Phase 33's XAUUSD convention for direct comparability of this specific design change.

## 8. Cost assumptions (frozen, identical to Phase 33/30 for consistency)

| Instrument | Normal (round-trip) | 1.5x | 2x |
|---|---|---|---|
| AUDUSD | 0.00018 | 0.00027 | 0.00036 |
| USDCAD | 0.00020 | 0.00030 | 0.00040 |
| USDCHF | 0.00020 | 0.00030 | 0.00040 |

## 9. Robustness tests required (frozen, identical structure to Phase 33, extended per Phase 34's recommendation)

1. **OOS sub-half consistency** — OOS split into two halves by trade-count median. Per Phase 34's explicit recommendation, a sub-half disagreement in a THIN sample (<40 total OOS trades) is classified **WARNING**, not automatic FAIL, unless corroborated by the parameter-perturbation check; ≥40 OOS trades with a sign disagreement is **FAIL**.
2. **Parameter perturbation** — ±20% of the single frozen key parameter (§3-7). A full sign reversal, or a >50% degradation in expectancy magnitude even without a sign flip, is **FAIL**. A same-sign result with <50% degradation is **PASS**.
3. **Cost stress** — 1.0x/1.5x/2.0x. OOS PF falling below 1.0 at 1.5x is **FAIL** (cost-fragile).
4. **Monte Carlo** — 10,000-draw trade-order reshuffle of OOS trades (candidate's own trades only — this preserves each trade's own regime/session context while testing sequencing risk, not a shuffle across different regimes).
5. **HIGH-volatility gate** — per Phase 34's explicit precondition: **UNKNOWN** if fewer than 10 OOS trades fall in the candidate's own HIGH-ATR tercile (terciles fixed from TRAIN+VAL only, no leakage). Otherwise STRONG (positive expectancy in HIGH tercile) / NEUTRAL (near zero) / WEAK (negative).
6. **Drawdown-correlation gate** — using the OOS-window-matched control (Phase 31/32's `data/phase26_all_trades.csv`, restricted to the candidate's own OOS date range, per the fair-comparison correction already established in Phase 33). **UNKNOWN** if fewer than 8 overlapping control-drawdown days exist (a slightly relaxed floor vs. Phase 33's informal 5-6, chosen in advance here, not after seeing results, to give the drawdown gate a marginally larger chance of producing a classification rather than defaulting to UNKNOWN as often as it did in Phase 33). STRONG DIVERSIFIER if drawdown-day correlation ≤ normal-day correlation; NEUTRAL if the difference is within 0.15; CORRELATED if drawdown-day correlation exceeds normal-day correlation by >0.15.

## 10. Portfolio integration (frozen, identical methodology to Phase 33)

Candidate's actual OOS trade stream blended with the control (`data/phase26_all_trades.csv`) restricted to the same OOS window, at 0.5x and 1.0x of the control's own median single-strategy daily-R-std — no weight optimization.

## 11. Multiple-testing controls (frozen)

- **Exactly 5 pre-registered hypotheses, each with exactly 1 parameter set** (no grid) — total disclosed parameter evaluations = 5 baseline + 5×2 perturbations = 15, all appearing in the final results regardless of outcome.
- The OOS window (2025-05-01 to 2026-08-14) is used exactly once per candidate. Any future revisit of this exact window for the same hypothesis is EXPLORATORY, not confirmatory, per the same rule established in Phase 33.
- **This is Phase 35's contribution to the project's cumulative multiple-testing count**: prior to this phase, 3 distinct strategy-family attempts had been made (Phase 30's calendar screen, Phase 33's two families) out of 16 taxonomized families (Phase 34). This phase adds 5 more family/hypothesis attempts, bringing the cumulative total to 8 of 16 (50%) — still well short of the point where the project's own Phase 34 analysis judged a stronger confirmation bar would become necessary.

## 12. Candidate classification rules (frozen — Part 24's 9 categories, applied mechanically and in this order)

- **A. REJECTED — NO EDGE**: OOS PF ≤ 1.0, or negative/statistically indistinguishable-from-zero OOS expectancy, or OOS trade count too small (<20) to evaluate at all.
- **B. REJECTED — OOS INSTABILITY**: OOS sub-half FAIL (§9.1).
- **C. REJECTED — PARAMETER FRAGILITY**: parameter-perturbation FAIL (§9.2).
- **D. REJECTED — COST FRAGILITY**: cost-stress FAIL (§9.3).
- **E. REJECTED — HIGH-VOLATILITY FAILURE**: HIGH-vol gate = WEAK (§9.5).
- **F. REJECTED — POOR DRAWDOWN DIVERSIFICATION**: drawdown-correlation gate = CORRELATED (§9.6).
- **G. REJECTED — POOR PORTFOLIO FIT**: passes A-F individually, but CONTROL+CANDIDATE shows materially worse combined max drawdown or max losing streak than CONTROL alone at 1.0x weight, with no offsetting HIGH-vol/drawdown-correlation benefit large enough to justify it (transparent qualitative judgment, numbers shown, not hidden).
- **H. PROMISING — MORE VALIDATION REQUIRED**: passes A-G, but at least one required category (Part 25's 10-point list) has insufficient evidence (e.g. an UNKNOWN classification on the HIGH-vol or drawdown-correlation gate) rather than an outright pass.
- **I. PORTFOLIO QUALIFIED — DEMO FORWARD TEST**: satisfies every item in Part 25's 10-point list with no UNKNOWN and no exception.

**These rules apply exactly as written, in this order, to every candidate, regardless of which candidate they favor or disfavor.**

---

*Frozen at the time of this commit. No candidate has been backtested yet. Any change after candidate results exist will be logged as a dated, explicit amendment below this line.*
