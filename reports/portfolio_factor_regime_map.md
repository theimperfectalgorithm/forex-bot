# Portfolio Factor & Strategy-Family Map (Summary)

Focused summary of Parts 3-4 of Phase 31. Full analysis, methodology, and every other section: `reports/phase31_portfolio_factor_regime_map.md` (master report).

---

## Currency factor map

Full detail: `reports/portfolio_currency_factor_map.csv`. Directional exposure computed per the explicit convention (BUY = long base / short quote; SELL = short base / long quote), not just symbol counting.

| Strategy | Instrument | Base/Quote | JPY leg? |
|---|---|---|---|
| GBPJPY AMR | GBPJPY | GBP/JPY | Yes |
| EURJPY AMR | EURJPY | EUR/JPY | Yes |
| AUDJPY AMR | AUDJPY | AUD/JPY | Yes |
| CADJPY AMR | CADJPY | CAD/JPY | Yes |
| CADJPY ARB | CADJPY | CAD/JPY | Yes |
| GBPUSD Monday | GBPUSD | GBP/USD | No |

**Risk-weighted currency exposure** (risk_pct × historical trade count, normalized): JPY 94.7%, CAD 33.8%, EUR 24.6%, AUD 22.4%, GBP 19.2%, USD 5.3%. (Currency rows don't sum to 100% — each trade counts toward both its base and quote currency.)

---

## Strategy family map

Full detail: `reports/portfolio_strategy_family_map.csv`. Mechanism verified from `pairs/*.yaml` configs and `PROJECT_REPORT.md`, not inferred from strategy names.

| Family | Strategies | Risk-weighted share |
|---|---|---|
| mean_reversion (asian_hours_reversion) | GBPJPY/EURJPY/AUDJPY/CADJPY AMR | **81.5%** |
| asian_range_breakout | CADJPY ARB | 13.2% |
| calendar_drift (monday_drift) | GBPUSD Monday | 5.3% |

**Four of six strategies (81.5% of risk-weighted trade count) share the identical mechanism family** — M15 z-score mean-reversion vs. SMA20, with no higher-timeframe trend filter by design (confirmed in `strategies/asian_hours_reversion.py` and reconfirmed in every prior phase of this project). This is the single largest concentration in the book by any dimension measured.

---

*Full methodology, session/regime/correlation/drawdown analysis, missing-factor synthesis, and final verdict: see the master report.*
