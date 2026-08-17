# Phase 36 — Alternative Return-Stream Map

**Identification only. No backtesting performed. No specific strategy or parameter set proposed. This document exists to widen the candidate universe under consideration for a future phase, per RQ9.**

---

## 1. Cross-asset relationships

**Hypothesis:** returns on one asset (e.g. a commodity, a rate-sensitive currency) predict short-term moves in a related instrument (e.g. AUDUSD vs. iron ore/copper proxies, CAD vs. oil, JPY crosses vs. risk sentiment). **Why different:** entirely outside the price-action/technical-indicator search space this project has used exclusively so far. **Expected correlation to current portfolio:** untested, plausibly low if the driving factor (a specific commodity or macro series) doesn't overlap with the AMR family's own mean-reversion trigger. **Expected regime behavior:** untested. **Data requirements:** a second, non-FX data series (commodity futures, an index) — not yet sourced in this project. **Implementation difficulty:** Moderate-High (a second data feed, alignment/lag questions). **Overfitting risk:** Moderate — cross-asset lead-lag relationships are a well-studied but easy-to-overfit area. **Reason to test:** genuinely outside the technical-FX space RQ7 flags as over-searched. **Reason not to test yet:** data availability for the second asset is unconfirmed in this project's current toolchain.

## 2. Index futures / index CFDs (where data quality permits)

**Hypothesis:** equity-index behavior (e.g. US indices around the NY open) provides a return stream with a fundamentally different driver (equity risk sentiment) than any FX pair in the current book. **Why different:** a different asset class entirely, not just a different currency pair. **Expected correlation:** untested, but equity-risk-sentiment-driven moves plausibly correlate with JPY-cross risk-off moves during genuine market stress — this is a real question to test, not assume away. **Expected regime:** likely HIGH-vol compatible around known event windows (US data, FOMC). **Data requirements:** confirmed MT5 access to a suitable index CFD, not yet verified in this project. **Implementation difficulty:** Low-Moderate once data access is confirmed. **Overfitting risk:** Moderate. **Reason to test:** directly addresses the "different asset class" gap this project has never explored. **Reason not to test yet:** data availability unconfirmed; would need its own validation before treating as equivalent to the FX pairs already used.

## 3. Commodity relationships

**Hypothesis:** direct commodity trading (beyond the already-tested XAUUSD) — e.g. WTI/Brent crude, driven by different macro factors (supply/inventory data, OPEC actions) than either FX or gold. **Why different:** distinct fundamental drivers from every instrument tested so far, including XAUUSD. **Expected correlation:** untested. **Expected regime:** event-driven, likely HIGH-vol compatible around inventory/OPEC data releases. **Data requirements:** already-available via the same MT5 feed for major commodity CFDs (unconfirmed for this specific instrument, not yet checked). **Implementation difficulty:** Low if data is available. **Overfitting risk:** Moderate. **Reason to test:** low-cost extension of already-available infrastructure. **Reason not to test yet:** would replicate some of XAUUSD's own diagnosed correlation-to-portfolio risk (macro/hedge co-movement) unless specifically checked.

## 4. Volatility-sensitive systems (VIX-style or realized-vol-based)

**Hypothesis:** a system that explicitly trades based on a volatility *level* or *regime transition* signal (not just an ATR-normalized entry/exit within an otherwise price-based strategy, which is what H5/Phase33's XAUUSD candidate already did) — e.g. positioning ahead of confirmed volatility regime shifts using a dedicated volatility index or realized-vol term structure. **Why different:** targets Phase 32's #1 priority (HIGH-vol compatibility) as the PRIMARY signal, not a secondary classification applied after the fact. **Expected correlation:** plausibly low to the AMR family, which has no explicit volatility-regime awareness at all. **Expected regime:** by construction, HIGH-vol-relevant. **Data requirements:** a genuine volatility index or options-derived data series — not currently available via the MT5 feed used in this project. **Implementation difficulty:** High (data sourcing is the main barrier). **Overfitting risk:** Moderate. **Reason to test:** the most direct possible match to Phase 32's own top-ranked factor. **Reason not to test yet:** the required data does not appear to already be available in this project's toolchain — this is a real, disclosed gap, not a soft preference.

## 5. Relative-value / spread structures

**Hypothesis:** trade the spread/relationship between two correlated instruments (e.g. AUDUSD vs. NZDUSD, or a JPY-cross basket vs. a single JPY-cross) rather than any single instrument's outright direction. **Why different:** a structurally distinct return driver (convergence/divergence between two series) rather than directional prediction of one series. **Expected correlation:** potentially very low to the current book if constructed to be currency-market-neutral. **Expected regime:** typically range/mean-reversion-compatible for the spread itself, even when the underlying legs are trending — a genuinely different regime relationship than any current strategy. **Data requirements:** two correlated price series, already available. **Implementation difficulty:** Moderate (spread construction, cointegration/relationship stability checks). **Overfitting risk:** Moderate-High (spread relationships can be unstable and are easy to curve-fit in-sample). **Reason to test:** genuinely different mechanism and potentially strong diversification if properly currency-neutral. **Reason not to test yet:** requires new spread-construction and stability-testing infrastructure not yet built in this project.

## 6. Session-specific event structures

**Hypothesis:** trade a specific, recurring calendar event (e.g. a fixed weekly economic release, month-end/quarter-end flows, options-expiry-related flows) rather than a generic session window. **Why different:** an event-conditioned mechanism, distinct from both AMR's time-of-day design and H1-H5's generic session-window designs. **Expected correlation:** low if the event driver is unrelated to the AMR family's own Asian-session mean-reversion trigger. **Expected regime:** event-specific, often HIGH-vol at the event itself. **Data requirements:** an economic calendar feed — `core/news_calendar.py` already exists in this project for the live news-blackout gate, so partial infrastructure exists. **Implementation difficulty:** Low-Moderate, given the existing calendar infrastructure. **Overfitting risk:** Moderate (calendar effects can be numerous and easy to data-mine if not disciplined). **Reason to test:** builds on existing project infrastructure (`core/news_calendar.py`), lower implementation cost than most alternatives here. **Reason not to test yet:** would need the same discovery-vs-validation discipline (a small, pre-registered set of specific events, not a scan of every recurring release).

## 7. Cross-sectional FX (rank/relative strength across a basket)

**Hypothesis:** rank a basket of currencies by relative strength/momentum and trade the strongest-vs-weakest pair, rather than a single fixed-pair signal. **Why different:** this project's own phase 6 research (`PROJECT_REPORT.md` §4) already found "a genuine CADJPY edge via cross-sectional momentum" — this is a directly precedented mechanism, distinct from every price-level technical rule tested in Phases 30/33/35. **Expected correlation:** potentially low if the basket construction is genuinely currency-neutral. **Expected regime:** trending/momentum-compatible. **Data requirements:** multiple correlated FX series, already available. **Implementation difficulty:** Moderate (basket construction and rebalancing logic). **Overfitting risk:** Moderate. **Reason to test:** the single most evidence-backed alternative on this list — extends an already-validated project finding rather than a speculative new idea. **Reason not to test yet:** the specific basket construction and rebalancing rules would need their own pre-registration before testing, not a reason to avoid the family.

## 8. Multi-asset momentum (cross-asset, not cross-sectional-FX-only)

**Hypothesis:** combine momentum signals across FX, commodities, and (if available) indices into a single diversified momentum sleeve, rather than a single-instrument momentum system (as H2/Phase33's USDCAD attempted and both failed). **Why different:** diversifies the momentum *signal* itself across asset classes, not just testing momentum on one more FX pair. **Expected correlation:** plausibly the lowest of any option on this list, by design. **Expected regime:** trend-compatible, diversified across drivers. **Data requirements:** the same as items 1-3 combined — the most demanding data requirement on this list. **Implementation difficulty:** High. **Overfitting risk:** Moderate-High (more instruments, more parameters to combine). **Reason to test:** highest theoretical diversification ceiling. **Reason not to test yet:** highest implementation cost and requires solving items 1-3's data-access questions first — a natural "later," not "never."

## 9. Macro/event-conditioned systems

**Hypothesis:** condition entries on a macro regime classifier (e.g. rate-cycle direction, risk-on/risk-off classification) rather than a pure price-action trigger. **Why different:** this project's own Phase 31 (§19) explicitly found "risk-off environment" could NOT be reconstructed from the available data — this hypothesis directly requires building that missing classifier first. **Expected correlation:** potentially very informative specifically for the drawdown-correlation priority (Phase 32's #2), since a risk-off classifier would directly test whether the current book's drawdown-day correlation IS a risk-off phenomenon. **Expected regime:** by construction. **Data requirements:** a macro/risk-sentiment data source not currently in this project's toolchain — the single largest missing piece identified across Phases 31-36. **Implementation difficulty:** High. **Overfitting risk:** Moderate if the classifier itself is kept simple and pre-registered. **Reason to test:** would resolve a standing, repeatedly-flagged data gap (Phase 31's own "NOT AVAILABLE" for the risk-off stress scenario). **Reason not to test yet:** the classifier itself would need to be built and validated as a precondition, a nontrivial prerequisite project.

## 10. Other structurally distinct mechanisms (catch-all)

Options-derived positioning signals, order-flow/microstructure signals (unlikely to be feasible without tick-level or DOM data beyond this project's current MT5 access), and seasonality beyond calendar/day-of-week (e.g. month-of-year effects, not yet screened). Each would need its own dedicated scoping before any further ranking.

---

## Summary table

| # | Direction | Genuinely different? | Data available now? | Overfitting risk | Priority (see phase36_research_direction.md) |
|---|---|---|---|---|---|
| 7 | Cross-sectional FX | Yes (precedented) | Yes | Moderate | Highest — most evidence-backed |
| 6 | Session-specific event structures | Yes | Partial (calendar infra exists) | Moderate | High — lowest implementation cost |
| 5 | Relative-value/spread | Yes | Yes | Moderate-High | Moderate |
| 1 | Cross-asset relationships | Yes | Unconfirmed | Moderate | Moderate |
| 3 | Commodity relationships | Partial (echoes XAUUSD's own issue) | Unconfirmed | Moderate | Moderate |
| 2 | Index futures/CFDs | Yes | Unconfirmed | Moderate | Lower (data gap) |
| 4 | Volatility-sensitive systems | Yes, most directly targets Priority 1 | NO — confirmed gap | Moderate | Lower (data gap, despite high relevance) |
| 9 | Macro/event-conditioned | Yes, most directly targets Priority 2 | NO — confirmed gap | Moderate | Lower (data gap, despite high relevance) |
| 8 | Multi-asset momentum | Yes | Depends on 1-3 | Moderate-High | Lowest (compound data dependency) |

**No specific strategy is proposed. This is a map for scoping, not a set of hypotheses ready for pre-registration.**
