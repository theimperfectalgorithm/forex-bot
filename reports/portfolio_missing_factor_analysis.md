# Portfolio Missing-Factor Analysis

**Question:** what return/risk factor does the current 6-strategy portfolio NOT have? **Answer supported by the factor/regime analysis in `reports/phase31_portfolio_factor_regime_map.md`, not assumed in advance.**

---

## The evidence, ranked by how strongly it's supported

### 1. Zero New York-session exposure (strongly supported — the cleanest finding of this phase)

Every current strategy initiates new risk during either the Asian session (00:00-07:00 server) or the London open (07:00-09:00 server breakout for CADJPY ARB). **Zero of the six strategies enters a new position during New York hours (roughly 12:00-21:00 UTC).** This is not a JPY-specific gap — it would remain true even if every JPY strategy were replaced with non-JPY equivalents that kept the same session design. This is the single most unambiguous structural gap this analysis found.

### 2. HIGH-volatility regime is the portfolio's only net-negative regime (strongly supported)

Across the full historical population (`data/phase26_all_trades.csv`, 2,712 trades): **3 of 6 strategies (AUDJPY AMR, CADJPY AMR, CADJPY ARB) are net-negative specifically in the HIGH volatility tercile**, and the portfolio's combined R in HIGH volatility is **−6.91**, versus **+120.64 in LOW volatility** and **+78.63 in NORMAL volatility**. HIGH volatility is the only regime bucket (of the two regime dimensions tested — volatility and trend) where the portfolio loses money in aggregate. This is a genuine capability gap, not a JPY-specific one — the portfolio simply has no strategy whose edge is HIGH-volatility-compatible.

### 3. Correlation rises specifically during drawdown days (strongly supported, and arguably the most important finding)

For most JPY-AMR strategy pairs, the daily-return correlation computed only on the portfolio's worst-decile drawdown days is **higher** than the correlation on ordinary days — e.g. EURJPY AMR / GBPJPY AMR: 0.557 (drawdown days) vs. 0.373 (normal days); AUDJPY AMR / GBPJPY AMR: 0.448 vs. 0.284; CADJPY AMR / GBPJPY AMR: 0.483 vs. 0.399. **The strategies become more correlated exactly when the portfolio is losing** — the diversification the portfolio appears to have on an average day partially disappears on the days it's needed most. This is a structural, not incidental, property of the current book.

### 4. Effective diversification is materially lower than "six strategies" suggests (supported, mathematically justified — not fabricated)

A risk-weight-only concentration measure (HHI-based effective N) gives **5.19 of 6** — looks reasonably diversified. But that measure ignores correlation entirely. Once the actual daily-return correlation matrix is incorporated (effective N = 1/(w′Σw)), the number drops to **2.67 of 6** — **the portfolio behaves, in risk terms, closer to 2-3 independent strategies than 6.** This is the clearest single number answering "how diversified is the portfolio really."

### 5. JPY concentration is real but is one contributor among several, not the sole cause (partially supported — the anti-bias finding)

JPY exposure is genuinely high (94.7% of risk-weighted trade-count touches JPY on either leg) and does contribute to findings #1-4 (the AMR pairs' shared session and mechanism *is* what drives most of the correlation-during-drawdown effect). **But the HIGH-volatility weakness (#2) is not exclusively a JPY phenomenon** — CADJPY ARB (also JPY, but a structurally different breakout mechanic) shares the same HIGH-vol weakness as the mean-reversion AMR pairs, suggesting the common thread may be **strategy family and session** as much as **currency**. A hypothetical non-JPY strategy that still traded the Asian session with a similar mean-reversion, no-trend-filter mechanic would very plausibly inherit the same correlation-during-drawdown and HIGH-vol-weakness problems. **JPY concentration is not being dismissed here — it is being placed alongside, not above, session and mechanism concentration as a contributing factor.**

---

## What this analysis does NOT support

- It does **not** show that simply adding *any* non-JPY pair would fix the portfolio — a non-JPY strategy sharing the Asian session and a trend-filter-free mean-reversion mechanic would likely replicate findings #2 and #3 regardless of currency.
- It does **not** show JPY concentration is irrelevant — findings #3/#5 show it is a real contributor, just not proven to be the *dominant* one in isolation from session/mechanism.
- It does **not** identify a single "smoking gun" missing factor — the evidence points to a **combination**: session (no NY exposure), regime (no HIGH-vol-compatible edge), and mechanism (mean-reversion dominance, 81.5% of risk-weighted trades) reinforcing each other, with JPY concentration as the currency-level symptom of that combination rather than its sole root cause.

---

*Full supporting evidence: `reports/phase31_portfolio_factor_regime_map.md` (master report) and its underlying CSVs. No strategy modified.*
