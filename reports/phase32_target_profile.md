# Phase 32 — Target Return-Stream Profile

**This is the deliverable Phase 33 will search against. Not a pair recommendation. Not a strategy. Derived entirely from Phase 32's factor-importance evidence (`reports/phase32_factor_ablation.csv`, `reports/phase32_factor_importance.csv`), not invented or assumed in advance.**

---

## THE IDEAL NEXT RETURN STREAM

**Volatility behavior (highest-priority characteristic, evidence-ranked #1):**
Positive or at minimum neutral performance specifically during HIGH-volatility conditions. This is the single most important characteristic tested — removing it cost the simulated portfolio **7.87R of additional maximum drawdown** (`phase32_factor_importance.csv`), more than any other factor. This directly targets the one regime (Phase 31 §9) where the existing portfolio is net-negative in aggregate.

**Drawdown correlation (evidence-ranked #2, and a distinct requirement from normal correlation):**
Correlation to the existing portfolio's daily returns that does NOT rise during the portfolio's own drawdown periods — ideally lower during drawdowns than on an average day. Removing this characteristic cost **5.46R of additional drawdown**, the second-largest effect measured. **This is not the same requirement as "low average correlation"** — Archetypes B and D in this phase both had acceptably low *normal* correlation (0.086, 0.111) but materially higher *drawdown* correlation (0.257, 0.314) and were correspondingly the two weakest performers in the simulation (`phase32_drawdown_diversification.csv`).

**Mechanism (evidence-ranked, large effect, not cleanly separable from #1/#2 in this model):**
Not another trend-filter-free mean-reversion strategy. The mechanism ablation (`phase32_mechanism_ablation.csv`) showed a hypothetical non-JPY strategy sharing the AMR family's mean-reversion design would still deepen portfolio drawdown by **14.7R relative to a trend-mechanism assumption**, and would itself be HIGH-vol-negative (−13.9R combined HIGH-vol contribution) — mechanism is closely intertwined with the volatility-behavior requirement above, not an independent add-on.

**Currency (real but the smallest cleanly-isolated effect):**
Non-JPY is preferred — the currency ablation showed a modest but real 1.5R drawdown difference attributable to the correlation assumption a JPY instrument would carry (`phase32_currency_ablation.csv`) — but this requirement should not be treated as more important than volatility behavior or drawdown correlation; the evidence in this phase does not support that ordering.

**Session (a real structural gap from Phase 31, but not shown to be sufficient on its own):**
London and/or New York session exposure is a genuine gap (zero NY exposure currently) — but the session simulation (`phase32_session_simulation.csv`) showed that a NY-only proxy with weaker underlying mechanism/correlation assumptions performed *worse* than a London-only proxy with stronger assumptions. **Session should be treated as a secondary, not primary, screening criterion** — a candidate should not be selected merely for trading NY hours if it fails the volatility/drawdown-correlation tests above.

**Normal-day correlation (real but evidence-ranked lowest of the four factors tested):**
Correlation to the existing book's daily returns below roughly 0.15-0.20 is a reasonable target, consistent with every archetype tested in this phase — but this phase's own evidence shows this is the **least** important of the four ablated characteristics (0.85R effect, roughly one-tenth the size of the HIGH-vol-compatibility effect). **Do not over-weight this criterion in Phase 33's screening relative to the others.**

---

## Explicit derivation, per factor (no invented thresholds)

| Characteristic | Target | Evidence source |
|---|---|---|
| HIGH-volatility behavior | POSITIVE or NEUTRAL, not NEGATIVE | `phase32_factor_importance.csv`: 7.87R effect (largest) |
| Drawdown-day correlation | Below ~0.10 (matching Archetypes A/C/E's realized 0.04-0.06, the three archetypes classified "STRONGER" diversifiers) | `phase32_drawdown_diversification.csv` |
| Mechanism | Not trend-filter-free mean-reversion; trend, breakout, or regime-aware preferred | `phase32_mechanism_ablation.csv`: 14.7R swing |
| Currency | Non-JPY preferred | `phase32_currency_ablation.csv`: 1.5R effect (smallest cleanly-isolated) |
| Session | London/NY preferred, but not sufficient alone | `phase32_session_simulation.csv`: NY-only proxy underperformed London-only proxy |
| Normal-day correlation | Below ~0.15-0.20 | `phase32_factor_importance.csv`: 0.85R effect (smallest of the four ablated factors) |

---

## What NOT to prioritize (equally derived from the evidence, not assumed)

- **Do not screen primarily on "non-JPY."** Currency showed the smallest cleanly-isolated effect of the factors tested, and Phase 31 already found JPY concentration is not proven to be the dominant cause of the portfolio's correlation structure.
- **Do not screen primarily on "trades New York hours."** The session simulation showed this alone does not guarantee improvement — a candidate's underlying mechanism and drawdown-correlation profile matter more than its session label.
- **Do not select a candidate based on low average/normal correlation alone.** This phase's evidence ranks it the least important of the four ablated factors — a candidate with excellent normal correlation but poor drawdown-day correlation (like the simulated Archetypes B and D) would be a weak addition despite looking good on a simple correlation screen.

---

## For Phase 33

Screen candidates in this priority order:
1. **Does it perform positively or neutrally in HIGH-volatility conditions** (empirically, not assumed)?
2. **Does its correlation to the current portfolio's daily returns stay low — or fall — specifically during the portfolio's own historical drawdown periods** (not just on an average day)?
3. **Is its mechanism something other than trend-filter-free mean-reversion** (trend-following, breakout-with-context, or another genuinely distinct family)?
4. Only then: is it non-JPY, and does it trade London/NY hours?

**Phase 33 will search for actual strategies matching this profile. This phase (32) authorizes no live trading change and identifies no specific pair or strategy.**
