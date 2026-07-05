# Forex Bot — Complete Status Report (2026-07-04 to 2026-07-05)

Full record of the research campaign, code changes, findings, and forward plan
produced across the strategy-validation sessions. ~500 walk-forward backtests
were run against live MT5 data (MetaQuotes demo, $100k account).

---

## 1. WHERE THE PROJECT STOOD BEFORE

- 5-agent autonomous system (Market / Strategy / Risk / Execution / Reporting)
  trading 3 pairs live on demo: GBPJPY + EURJPY (london_breakout), EURUSD
  (sma_ema_combined). London + NY sessions only.
- 3 new GBPUSD strategies written but unvalidated (VolatilityRegimeTrend,
  MomentumDivergenceSession, RegimeFilteredMACross), all active: false.
- Risk gates: $90k hard floor, 5% daily loss, 2-consecutive-loss pair pause,
  dynamic lot sizing. No portfolio-level checks.
- No spread-aware backtesting; prior validations ignored transaction costs.

## 2. VALIDATION METHODOLOGY (used for every test)

- 36 months of MT5 H1/M15/H4/M5 data. In-sample (IS) = first 24 months,
  out-of-sample (OOS) = final 12 months. Selection on IS only.
- Pass criteria: profit factor > 1.3, max drawdown < 8%, >= 60% profitable
  months in-sample, AND positive OOS with acceptable DD.
- Per-pair spread deducted from every trade. SL assumed hit first when SL and
  TP touch in one bar (conservative). One trade/day/strategy, Friday 20:00
  close, session windows enforced — mirrors the live orchestrator.
- Indicator math verified by assertion against the live classes' exact
  windowed seeding (src/strategy_matrix_backtest.py `_self_check`).

## 3. WHAT WE FOUND — THE RESEARCH RECORD

### Phase 1 — Strategy matrix (33 runs): 0 passed
VRT / MDS / RFMC / AsianRangeBreakout across 9 majors. MDS-GBPUSD passed
in-sample (PF 1.71) but collapsed OOS (PF 0.63) in every parameter
neighbourhood = overfit. Failure notes written into the GBPUSD YAMLs.

### Phase 2 — mean_reversion sweep + ARB grid (62 runs): 1 passed
- mean_reversion: failed on all 9 majors.
- **PASS: asian_range_breakout GBPJPY, tp_multiplier 2.0, NO H4 filter**
  (IS PF 1.45 / DD 4.67% / 62.5% prof. months; OOS PF 1.19 / +$3,676).
  Robust across the whole TP grid — not a lucky cell.

### Phase 3 — Session-structure families, M15/H1/H4 (117 runs): 0 passed, 1 discovery
- London Open Range Breakout: catastrophic everywhere (PF 0.65–0.98).
- NY continuation, H4 Donchian/ATR: dead.
- **DISCOVERY: Asian-hours mean reversion (AMR) on JPY crosses** — all 36
  variants OOS-positive (PF 1.08–2.04) but IS PF plateaus 1.10–1.17:
  a real, parameter-insensitive, REGIME-STRENGTHENING edge (strong only in
  the last 12 months). Forward-test candidate, not activation candidate.

### Phase 4 — Professional-style EURUSD/GBPUSD (22 runs): 0 passed
False-breakout fade, WMR 16:00 fix flows, EURUSD-vs-GBPUSD relative value
(via EURGBP), vol-regime-conditioned Bollinger fade. Key science: London
breakouts lose CHASED **and** FADED — whipsaw eats both geometries.

### Phase 5 — Mechanical ICT 2022 model on M5 (16 runs): 0 passed
Liquidity sweep → displacement/MSS → FVG entry, London/NY killzones.
Incoherent cell-to-cell results = noise. ICT's testable core is the same
liquidity phenomenon already killed both directions.

**Cumulative verdict: EURUSD/GBPUSD are dead ground for price-derived
signals at retail data tier (~430 tests). All real edges found are
JPY-cross / session-structure.**

### Live-book audits (the most important findings)
- **EURUSD sma_ema_combined**: the SMA Run-1 book has a structural bug —
  its flat-filter (skip if |SMA50−SMA100| < 5p) contradicts its own cross
  trigger (SMAs are near-equal at a cross by definition): **zero trades in
  3 years**. The EMA pullback book: 435 trades, PF 0.95, −$3,717/3y.
  → Recommendation: deactivate.
- **london_breakout (GBPJPY/EURJPY)**: GBPJPY IS PF 0.95 / OOS 0.82
  (−$12.8k OOS at 1% risk); EURJPY IS 0.82 / OOS 0.77 (DD 39.6%).
  Adding them to the healthy portfolio flips it from +1.36%/mo & 0% bust
  to −0.25%/mo & **45% bust**. → Recommendation: deactivate.

### Phase 6 — New pairs + portfolio Monte Carlo
- **CADJPY: new edge, both families** — ARB tp2.0/noH4 (IS PF 1.15 /
  OOS 1.38), AMR z2.0 (IS 1.10 / OOS 1.35). The JPY thesis replicated on
  an out-of-family pair = strongest genuineness evidence.
- NZDJPY AMR OOS-strong (1.35–1.76), IS ~1.0 (same regime pattern).
- Weekly cross-sectional momentum (11 pairs): dead (PF 0.99).

### Phase 7 — Exits, calendar, gold
- **Exit study**: baseline SL/TP beat breakeven/trailing on 5/6 strategies
  (protective exits cut reverting winners). Adopted only BE@0.75R on
  AMR-GBPJPY (better in both windows, marginally).
- **GBPUSD Monday drift — first genuine EU/GU edge of the project**:
  +0.13%/day IS (t=+3.33, 65% win) and +0.21%/day OOS (t=+4.00, 73% win),
  positive every year 2023–2026, day-specific. Needs conversion to a
  bounded strategy + harness validation. EURUSD Monday weaker (OOS t=1.05).
- **XAUUSD**: ARB family provisionally passes (IS PF 1.33–1.45, DD < 3%)
  but OOS only flat-positive (PF ~1.05). Gold AMR / NY-momentum: dead.
  One small provisional slot. (5ers allows metals; verify program specifics.)

## 4. CODE CHANGES MADE (all local; nothing committed to git)

### Risk agent — src/agents/agent_risk.py (LIVE)
Four new prop-firm gates, tested against live MT5 (approve + reject paths):
1. Equity-based hard floor (floating losses count, not just balance).
2. Daily equity soft stop at −4% vs day's first-seen equity anchor
   (persisted in state; buffer before the firm's −5%).
3. Aggregate open-risk cap: sum of entry→SL risk across open positions +
   new trade ≤ 3% of balance; any open position missing an SL blocks
   new trades entirely.
4. Currency-concentration cap: max 2 open positions sharing a currency.

### Orchestrator — src/agents/main_agent.py + agent_strategy.py (LIVE)
- Strategy cache now keys asian_range_breakout entries as '<PAIR>@arb'
  (fixes the pair-name collision; london keys unchanged → state-file
  compatible). `key.split('@')[0]` recovers the MT5 symbol.
- `_breakout_pairs()` dispatches BOTH LondonBreakout and AsianRangeBreakout
  through the shared prepare()/check_breakout() path.
- `step_check_breakouts` iterates BREAKOUT_KEYS; state dicts include the
  new keys with .get() guards; open_trades records `strategy_key`.
- `_fresh_state` gained `day_start_equity` (risk anchor).
- **NOTE: requires a bot restart (VPS) to take effect.**

### Strategies
- strategies/asian_range_breakout.py: `tp_multiplier` + `h4_filter` now
  YAML-configurable; GBPJPY added to COMPATIBLE_PAIRS.
- strategies/asian_hours_reversion.py: NEW class (AMR) — full
  implementation + orchestrator integration spec in its docstring.
- strategies/registry.py: validation statuses recorded; AMR registered.

### Configs (pairs/)
- GBPJPY_asianrange.yaml — **ACTIVE (live on demo)**: tp 2.0, no H4, 0.5%.
- GBPJPY/EURJPY/AUDJPY_asianrev.yaml — AMR forward-test configs,
  active: false pending Asian-hours orchestration.
- GBPUSD_volregime / _momentum / _regimecross — failure verdicts written in.
- verify_architecture.py updated: 14/14 checks pass (GBPJPY appears twice).

### Research harnesses (src/, reusable)
strategy_matrix_backtest.py (core engine: windowed indicators, trade sim
with time exits, metrics), phase2_meanrev_arb_search.py,
phase3_session_structure_search.py, phase3b_amr_jpy_refine.py,
phase4_pro_eurusd_gbpusd.py, phase5_ict_backtest.py,
phase6_portfolio_model.py (portfolio assembly + 5ers Monte Carlo),
phase7_exits_calendar_gold.py (exit modes, calendar screen, gold),
revalidate_eurusd_live.py. Results CSVs + reports in data/.

## 5. THE PORTFOLIO MODEL (Book B+)

Composition (risk per trade):
| Slot | Pair | Strategy | Session | Risk |
|---|---|---|---|---|
| 1 | GBPJPY | asian_range_breakout (tp2.0, noH4) | Tokyo→London 07–09 | 0.50% |
| 2 | CADJPY | asian_range_breakout (tp2.0, noH4) | Tokyo→London 07–09 | 0.50% |
| 3 | GBPJPY | AMR z2.5/sl1.25/h<4 + BE@0.75R | Asian 00–07 | 0.25% |
| 4 | EURJPY | AMR z2.0/sl1.5/h<6 | Asian 00–07 | 0.25% |
| 5 | AUDJPY | AMR z2.0/sl1.5/h<4 | Asian 00–07 | 0.25% |
| 6 | CADJPY | AMR z2.0/sl1.5/h<4 | Asian 00–07 | 0.25% |
| 7 | XAUUSD | ARB (provisional) | Tokyo→London | 0.25% |
| (8) | GBPUSD | Monday drift (pending validation) | Monday all-day | TBD |

Monte Carlo vs 5ers step 1 (+8% target, −5% daily, −10% overall; 2,000
runs × 126 trading days): **+2.12%/month expectancy, 83% pass within 6
months, 2% bust, 15% timeout (no time limit at 5ers → timeouts pass
later), median 55 trading days (~2.6 months).**

Caveats: (a) AMR edge is regime-young — demo must confirm; (b) worst
historical 36-month stretch was −14.6% (correlated JPY drawdown; live
currency cap reduces this vs the naive backtest sum; start at half-risk
if all 5 JPY slots go live together); (c) gold slot is IS-strong /
OOS-unproven.

## 6. EXECUTION QUEUE (agreed plan, pending "go")

1. Deactivate losing live strategies: GBPJPY.yaml + EURJPY.yaml
   (london_breakout) and EURUSD.yaml → active: false.
2. Wire Asian-hours orchestration for AMR: 00:00–06:00 polling step +
   07:00 time-exit step + '@amr' cache keys (spec in
   strategies/asian_hours_reversion.py). Then activate the 3 AMR YAMLs.
3. Add CADJPY configs (ARB validated params + AMR) and XAUUSD ARB
   provisional config (needs pip-size handling for gold: 0.1, not the
   JPY/other logic; agent_risk lot calc already correct via tick values).
4. Build + validate Monday-GBPUSD strategy class (Monday entry, ATR-scaled
   SL, EOD exit, ~52 trades/yr) through the standard harness.
5. Build the news-calendar blackout gate (5ers rule: no entries/exits
   within ±5 min of high-impact news; config `news_filter` exists, no
   data feed behind it yet). Required before the challenge.
6. Restart VPS bot; demo forward-test the full book 4–8 weeks; compare
   monthly vs backtest. Buy the challenge only when demo shows ≥1%/month
   with max DD < 5%.
7. Before any non-$100k account: parameterize hardcoded constants
   (STARTING_BALANCE / HARD_FLOOR in agent_risk.py; $5,000 daily literal
   in main_agent.py). Note: at $5k, min-lot 0.01 forces >0.25% effective
   risk on JPY crosses.

## 7. PRINCIPLES ESTABLISHED (do not relitigate)

- Portfolio = sum of validated edges, not number of pairs. Negative-
  expectancy pairs make the portfolio strictly worse (measured: 45% bust).
- Movement ≠ money; predictability = money. EURUSD/GBPUSD are efficient,
  not generous. Edges found live in JPY-cross session structure + one
  calendar anomaly.
- Filters remove trades; they cannot create edge in a dead signal.
- Exit tuning polishes real edges only; BE/trailing usually hurts here.
- Every new idea passes the same bar: IS-selection only, OOS confirmation,
  spread-paid, walk-forward. No exceptions, including "popular" pairs.
