# Future Candidate Profile — Phase 31

Derived from `reports/portfolio_missing_factor_analysis.md`. **A profile, not a pair or strategy recommendation.** Only characteristics actually supported by the Phase 31 factor/regime analysis are included.

**Future candidates should ideally be:**

- **Non-JPY**, but this alone is insufficient — see the session/mechanism caveat below.
- **Session exposure outside 00:00-09:00 server** (Asian + London-open) — ideally with meaningful **New York-session** participation, since the current book has literally zero NY-session entries (`portfolio_missing_factor_analysis.md` §1).
- **Not another trend-filter-free mean-reversion mechanic** — this specific combination (mean-reversion + no trend awareness) is the common thread linking the four AMR strategies' shared HIGH-volatility weakness and their elevated correlation during portfolio drawdowns. A trend-following, breakout-with-trend-context, or regime-aware mechanic would structurally differ from the existing book's dominant 81.5%-of-risk-weighted-trades family.
- **Demonstrated positive-or-neutral performance specifically in HIGH-volatility conditions** — the one regime bucket where the existing portfolio is net-negative (§2). This is the single most concrete, quantified capability gap this analysis found.
- **Correlation to the existing portfolio's daily returns materially below the 0.192 average pairwise correlation already observed among the current six** — and ideally, correlation that does **not** rise during the portfolio's own drawdown days (the opposite of what's currently observed for most AMR pairs, `portfolio_drawdown_factor_analysis.csv`).
- **Cost-robust under 2x the assumed spread** — the same bar every currently-live strategy was held to at deployment.
- **Trade frequency roughly weekly-to-daily** — enough to accumulate a validating sample at a reasonable pace, consistent with what this project's live-forward-testing discipline requires.

**Expected contribution, framed honestly:** a candidate meeting this profile would not be expected to be a large standalone return driver (the existing book's strongest strategies — GBPJPY AMR, GBPUSD Monday — already have full pre-live validation and no evidence of live deterioration); its value would be **structural** — raising the portfolio's correlation-adjusted effective N (currently 2.67 of 6, `portfolio_factor_summary.csv`) and reducing the correlation-during-drawdown effect, not necessarily topping the per-strategy PF leaderboard.

---

## AUDUSD Monday LONG checked against this profile (diagnostic only — NOT a promotion decision, per Phase 30's explicit instruction)

| Profile criterion | AUDUSD Monday LONG | Match? |
|---|---|---|
| Non-JPY | Yes (AUD/USD, no JPY leg) | ✓ |
| Session outside 00:00-09:00 server / NY participation | No — same Monday 00:00-server entry design as GBPUSD Monday; does not add NY-session exposure | ✗ |
| Not trend-filter-free mean-reversion | Correct — it's a calendar/drift mechanic, structurally different from AMR | ✓ |
| Positive/neutral in HIGH volatility | **Yes, and notably strong**: mean R +0.248/trade in its own HIGH-ATR-tercile bucket (59 trades, total +14.6), its *best* of the three vol terciles — the opposite pattern from the AMR portfolio's HIGH-vol weakness | ✓✓ |
| Correlation below 0.192 average / lower during drawdowns | Correlation to the historical control's daily R = **0.29** (`reports/non_jpy_portfolio_comparison.csv`) — **above**, not below, the existing book's own 0.192 average pairwise correlation. Drawdown-specific correlation: **INSUFFICIENT DATA** (only 1 of the control's worst 10 days coincided with a candidate trade day) | ✗ (on the one number available) |
| Cost-robust at 2x spread | Yes — OOS PF 2.647, t=3.62 at 2x assumed spread (`reports/non_jpy_diversification_research.md` §8) | ✓ |
| Weekly-ish trade frequency | ~1/week (Monday-only design) — on the low end of "acceptable" but consistent with GBPUSD Monday's own already-live cadence | ✓ (marginal) |

**Does AUDUSD Monday LONG address the missing factor? Partially — and the strongest match is the least expected one.** Its regime behavior (strong specifically in HIGH volatility) is a genuinely good match for the portfolio's single clearest capability gap (§2 above) — this is real, quantified evidence, not a coincidence dressed up as one. But it does **not** address the session gap (§1) — it's a Monday-only, start-of-week design just like the strategy already in the book, so it would not add New York exposure. And its correlation to the existing portfolio (0.29) is actually **higher** than the current six's own internal average pairwise correlation (0.192), meaning by that specific measure it would not obviously *lower* the portfolio's correlation-adjusted effective N as much as a genuinely low-correlation candidate would.

**This is not a promotion or deployment recommendation.** It is a factor-match diagnostic, consistent with the explicit instruction not to advance this candidate in this phase.

---

## Next non-JPY research direction (Part 19 — a profile, not "research EURUSD")

Research candidates that are:
- **Non-JPY**, with genuine **London/New York session** activity (not another Asian-session or start-of-week design).
- **Trend-following, breakout-with-trend-context, or otherwise regime-aware** — explicitly not a repeat of the trend-filter-free mean-reversion mechanic already dominating the book.
- **Historically compatible with HIGH-volatility conditions** — this is now the portfolio's best-evidenced capability gap and should be an explicit screening criterion for any future candidate, not an afterthought.
- **Correlation to the existing book below ~0.19** (the current six's own internal average) on daily returns, with a specific check on drawdown-day correlation (not just average-day correlation) before any candidate is taken further.

---

*Diagnostic profile only. No candidate promoted, validated, or deployed.*
