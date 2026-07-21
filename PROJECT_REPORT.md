# Forex Bot — Complete Project Report

**Date:** 2026-07-15 (rev 2, includes trade journal + health monitor) ·
**Supersedes:** PROJECT_STATUS_2026-07-05.md
**Purpose:** a single self-contained document from which any person or AI
can understand what this project is, what has been built, what was tried
and rejected (with evidence), what is running live right now, and what
comes next. Nothing in this document requires conversation context.

---

## 1. EXECUTIVE SUMMARY

This is an autonomous 5-agent MT5 trading system whose goal is to pass a
5ers prop-firm challenge (+8% step 1, then +5%, within −5% daily / −10%
overall drawdown). The project's owner documents everything publicly on
YouTube (TheImperfectAlgorithm); this repo is public; timeline decisions
are sometimes content-driven rather than statistically optimal.

Between 2026-07-04 and 2026-07-15 the project went through a complete
transformation:

- **~530 walk-forward backtests** across 8 research phases replaced the
  original 3 live strategies (all proven net losers) with a validated
  **8-slot portfolio** ("Book B+") spanning 6 instruments and 3 sessions.
- Portfolio Monte Carlo vs 5ers step-1 rules: **+2.1%/month expectancy,
  83% pass within 6 months, 2% bust, median 55 trading days**.
- The live forward-test (demo account **5052472770**, started $100k
  2026-07-01) has already caught and fixed **four real-world defects**
  that no backtest could see. Equity as of 2026-07-15: **≈ $99,965
  (−0.03%)** after 10 closed new-book trades.
- A second MT5 terminal + per-instance config system is ready for the
  5ers account the owner is purchasing now. The demo account continues
  permanently as the control group.

**The single most important lesson encoded in this project:** edges are
found by exhaustive falsification, not intuition. ~95% of everything
tested failed. What survived is small, session-structural, and mostly
JPY-cross — and every failure is documented below so it is never re-run.

---

## 2. SYSTEM ARCHITECTURE (as of commit a92cdc3)

### 2.1 The five agents (src/agents/)
| Agent | File | Role |
|---|---|---|
| Orchestrator | main_agent.py | 24/5 loop, wakes every 15 min, schedules all steps, owns daily state (data/state/daily_state.json) |
| 1 Market | agent_market.py | 00:00 UTC: balance vs hard floor, news flags → TRADE_DAY/AVOID |
| 2 Strategy | agent_strategy.py | loads strategies via registry, session prep + signal checks |
| 3 Risk | agent_risk.py | 9 gates before every trade (see 2.4) + lot sizing |
| 4 Execution | agent_execution.py | order placement, position monitoring, close detection, trade CSV |
| 5 Reporting | agent_reporting.py | 21:00 UTC daily summary |

### 2.2 Strategy dispatch — the key system
`pairs/*.yaml` files declare (pair, strategy, params, active). The
strategy cache in agent_strategy keys entries so one pair can run
multiple strategies: plain name for legacy strategies, **`<PAIR>@arb`**
(asian_range_breakout), **`<PAIR>@amr`** (asian_hours_reversion),
**`<PAIR>@mon`** (monday_drift). `key.split('@')[0]` recovers the MT5
symbol. Registry: strategies/registry.py (single source of truth,
carries validation verdicts as comments).

### 2.3 TIME COORDINATES — critical, easy to get wrong
**MT5 bar timestamps are SERVER time (UTC+3 in summer, UTC+2 in winter —
the MetaQuotes convention aligning the daily bar to NY close).** Every
strategy bar-hour rule and every backtest window is in SERVER
coordinates. The orchestrator therefore gates ALL session steps on
server minutes (`srv`), computed each cycle via
`agent_strategy.server_utc_offset_hours()` (live tick detection, DST
calendar fallback). Only day-cycle steps (market agent, report, Friday
close, state rollover) run on real UTC. **Fixed in e75d680 after live
evidence showed ARB and monday_drift could never fire and AMR ran on a
truncated window.** Real-UTC equivalents in summer: AMR entries
~21:00–03:00, AMR exit 04:00, ARB prep 04:45 / checks 05:00–09:30,
monday_drift entries SUNDAY 21:00–23:00, monday exit Monday 18:00.

### 2.4 Risk gates (agent_risk.run, all 9 in order)
1. MT5 connection + account read
2. Hard floor on BALANCE (starting_balance × (1 − hard_floor_pct))
3. Hard floor on EQUITY (floating losses count)
4. Daily loss limit — closed P&L (daily_loss_pct × starting_balance)
5. Daily EQUITY soft stop at −4% vs day's first-seen equity anchor
   (buffer inside the firm's −5%)
6. Per-pair 2-consecutive-loss daily pause
7. Aggregate open risk: Σ(entry→SL risk) + new trade ≤ 3% of balance;
   any open position missing an SL blocks all new trades
8. Currency concentration: max 2 open positions sharing a currency
9. News blackout: no entries within ±5 min of high-impact news
   (core/news_calendar.py, Forex Factory weekly JSON, 6h cache).
   **Fails OPEN in demo; MUST be flipped to fail-closed before a funded
   challenge.** Time exits (@amr/@mon) also defer inside a blackout.

Sizing: lots = balance × risk_percent(YAML) × risk_scale(config) /
(sl_pips × pip_value). Pip value from MT5 tick data except gold
(computed from contract size — MetaQuotes reports broken tick_value for
metals when market closed). MAX_LOT clamp from config.

### 2.5 Multi-account / multi-instance
Two MT5 terminals on the VPS: `C:\Program Files\MetaTrader 5\` (demo
5052472770) and `C:\MT5-5ers\` (awaiting 5ers login). One bot process
per terminal, from separate git clones of this repo. Per-instance
settings live in **gitignored `config/local_config.yaml`** (same schema
as global_config's `global:` block) — never edit the tracked config on
a VPS. Required keys per instance: `mt5_terminal_path` (pins the
process to its terminal at startup — mandatory with 2 terminals),
plus for the prop clone: `starting_balance`, `max_lot` (scale down!),
`risk_scale` (0.5 recommended initially). Password only via terminal's
saved login or MT5_PASSWORD env var. Never run one process switching
accounts (MT5 python API is a per-process singleton).

**Pair/strategy isolation (added 2026-07-21):** `pairs/*.yaml` is fully
git-tracked, so both clones see identical files after `git pull` —
`core/pair_manager.py`'s `get_active_pairs()` has no per-instance
filtering of its own. To let the demo clone keep freely adding/testing
new pairs via pushes to `main` without those going live on the funded
5ers clone, the 5ers clone's `local_config.yaml` carries an optional
top-level `locked_pairs` allowlist:
```yaml
locked_pairs:
  - pair: GBPJPY
    strategy: asian_range_breakout
  # ... one entry per validated {pair, strategy} in the current book
```
`get_active_pairs()` intersects the YAML-active set against this list
when present — a pair/strategy must be BOTH `active: true` in its YAML
AND listed here to trade; this only ever removes, never force-adds, so
a repo-wide `active: false` kill switch still applies everywhere. The
demo clone leaves `locked_pairs` undefined and is unaffected. Anything
excluded logs a WARNING in trading.log. **To promote a pair** to the
funded account: after it clears demo forward-testing, manually add its
`{pair, strategy}` entry to the 5ers clone's own `local_config.yaml` and
restart — a deliberate, on-machine action, never a side effect of a
`git push` made while working on the demo box.

### 2.6 Data collection & self-monitoring (added fc291ed, 2026-07-15)
- **Trade journal (core/trade_journal.py):** append-only ML-ready JSONL
  at data/journal/events.jsonl recording every SIGNAL, **REJECTION**
  (first-class, with gate stage + reason — the counterfactual half of a
  future meta-labeling dataset), ENTRY (26 context fields: spread, H1
  ATR, server hour/dow, minutes-to-next-high-impact-news, account
  balance/equity/open-risk, slippage vs signal price, intended risk,
  dual UTC+server timestamps) and EXIT. Hooked at every decision point
  in both the breakout and AMR/MON orchestrator paths. Fail-safe
  (journal errors can never block trading). Journal starts EMPTY
  2026-07-15 — earlier trades are only in MT5 history + trades_log.csv.
  MAE/MFE not captured live; reconstruct offline from bars if needed.
- **Strategy Health Monitor (core/health_monitor.py):** runs in the
  daily 21:00 report and standalone (`python -m core.health_monitor`).
  Compares each strategy key's live win rate to its backtest
  expectation (EXPECTED_WR table inside, conservative IS/OOS picks)
  via exact binomial statistics: GREEN / AMBER (p<0.10) / RED (p<0.02,
  n≥10 → logs a pause RECOMMENDATION; pausing stays a human decision
  per the standing checkpoint rules). This is the "learning loop"
  adopted from ML — as auditable statistics, not a neural net.
- **Meta-labeling: EXPLICITLY PARKED** until the journal holds ~1000+
  trades (~1 year). Price-predicting neural networks remain banned per
  the dead-ground principles (they are the overfitting failure mode of
  phases 1–5, scaled up).

### 2.7 Legacy behaviors removed/gated
- Daily 17:30 UTC EOD close: removed long ago in code; was still live
  on the owner's LAPTOP build through June (source of June's EXPERT
  closes — resolved, laptop retired 2026-07-01).
- 25-pip breakeven move (agent_execution): **excluded for @-keyed
  trades** (4691a22) — phase-7 study showed baseline beats breakeven on
  5/6 book strategies; observed live turning a +25p ARB trade into a
  $0 scratch. Legacy-style trades keep it.

---

## 3. THE LIVE BOOK — 8 validated slots (all active)

| # | Config file | Strategy | Session (server) | Risk | Validation (IS = Jul23–Jun25, OOS = Jul25–Jun26) |
|---|---|---|---|---|---|
| 1 | GBPJPY_asianrange.yaml | ARB tp2.0 noH4 | breakout 07–09 | 0.50% | IS PF 1.45/DD 4.7%/62.5%pm; OOS PF 1.19 +$3.7k |
| 2 | CADJPY_asianrange.yaml | ARB tp2.0 noH4 | breakout 07–09 | 0.50% | IS PF 1.15; OOS PF 1.38 +$6.4k |
| 3 | XAUUSD_asianrange.yaml | ARB tp1.5 noH4 min_range 30 | breakout 07–09 | 0.25% | PROVISIONAL: IS PF 1.45/DD 2.9%; OOS flat (PF 1.05) |
| 4 | GBPJPY_asianrev.yaml | AMR z2.5 sl1.25 h<4 | Asian 00–07 | 0.25% | IS PF 1.16/68%pm; OOS PF 2.03 |
| 5 | EURJPY_asianrev.yaml | AMR z2.0 sl1.5 h<6 | Asian 00–07 | 0.25% | IS PF 1.10/60%pm; OOS PF 1.47 |
| 6 | AUDJPY_asianrev.yaml | AMR z2.0 sl1.5 h<4 | Asian 00–07 | 0.25% | IS PF 1.17/60%pm; OOS PF 1.23 |
| 7 | CADJPY_asianrev.yaml | AMR z2.0 sl1.5 h<4 | Asian 00–07 | 0.25% | IS PF 1.10; OOS PF 1.35 |
| 8 | GBPUSD_monday.yaml | monday_drift sl1.25/tp1.0×ATR20d | Mon 00:00→21:00 | 0.25% | **strongest pass: IS PF 1.97/DD 0.66%/66.7%pm; OOS PF 3.08/DD 0.42%** |

Strategy mechanics:
- **ARB (asian_range_breakout):** Asian range 00–07 server; first H1
  close beyond an edge in hours {7,8}; SL at opposite edge, TP =
  tp_multiplier × range; runs to SL/TP (may hold days); Friday close.
- **AMR (asian_hours_reversion):** M15 z-score vs SMA20 during quiet
  hours; |z| ≥ threshold fades back to the mean (TP = SMA20, SL =
  sl_multiplier × that distance); force-flat at 07:00 server.
- **MON (monday_drift):** long GBPUSD at the close of server-Monday's
  00:00 H1 bar (= real Sunday ~22:00); ATR20d-scaled SL/TP; force-flat
  21:00 server Monday. Discovered via the phase-7 calendar screen
  (Monday drift t=+3.3 IS / +4.0 OOS, positive every year 2023–26).

**Caveats attached to the book:** AMR's edge is regime-young (strong
only in the last 12 months; parameter-insensitive so not overfit —
that's why it's on demo probation). Gold is IS-strong/OOS-flat. The
combined book's worst historical 36-month stretch was −14.6%
(correlated JPY drawdown; the live currency cap reduces this vs the
naive sum). Retired strategies remain in pairs/*.yaml with active:false
and their audit verdicts as comments: GBPJPY+EURJPY london_breakout
(OOS PF 0.82/0.77; including them flipped the portfolio MC to 45%
bust), EURUSD sma_ema_combined (SMA book structurally cannot fire —
its flat-filter vetoes its own cross trigger; EMA book PF 0.95).

---

## 4. RESEARCH RECORD — what was tried (do not re-run)

All harnesses live in src/ and are reusable; results CSVs in data/
(gitignored, regenerable). Methodology everywhere: 36 months MT5 data,
IS = first 24m / OOS = last 12m, selection on IS only, spread paid,
SL-first on ambiguous bars, criteria PF>1.3 / DD<8% / ≥60% profitable
months / positive OOS. Engine self-checks its windowed indicator math
against the live classes' exact seeding.

| Phase | Script | What was tested | Result |
|---|---|---|---|
| 1 | strategy_matrix_backtest.py | VRT/MDS/RFMC/ARB × 9 majors (33) | 0 pass. MDS-GBPUSD passed IS (PF 1.71), collapsed OOS = overfit |
| 2 | phase2_meanrev_arb_search.py | mean_reversion sweep + ARB grid + MDS fragility (62) | **1 pass: ARB-GBPJPY tp2.0/noH4** |
| 3 | phase3_session_structure_search.py | LORB / AMR / NY-continuation / H4 Donchian, M15-H4 (81) | LORB catastrophic (PF .65–.98); **AMR-JPY discovery** (all 36 variants OOS-positive) |
| 3b | phase3b_amr_jpy_refine.py | AMR refinement grid (36) | IS PF plateaus 1.10–1.17 = regime-strengthening, not overfit |
| 4 | phase4_pro_eurusd_gbpusd.py | Pro-style EU/GU: false-break fade, WMR fix flows, EURGBP RV, vol-regime MR (22) | 0 pass. London breakouts lose chased AND faded |
| 5 | phase5_ict_backtest.py | Mechanical ICT 2022 model on M5 (16) | 0 pass, incoherent cells = noise |
| 6 | phase6_portfolio_model.py | LIVE-book audits + NZDJPY/CADJPY + x-sect momentum + portfolio MC | london_breakout FAILED audit; **CADJPY new edge (both families)**; Book B MC 63% pass/0% bust |
| 7 | phase7_exits_calendar_gold.py | Exit modes (BE/trail), EU/GU calendar screen, XAUUSD | Baseline exits win 5/6; **GBPUSD Monday drift found (OOS t=+4.0)**; gold ARB provisional; Book B+ MC 83%/2% |
| 8 | phase8_monday_validation.py | Monday drift as bounded strategy (8) | **PASS (strongest of project)**; EURUSD control weak |
| — | revalidate_eurusd_live.py | Faithful dual-book audit of live EURUSD | SMA book 0 trades/3y (flat-filter bug); EMA PF 0.95 |

**Settled dead ground (~470 failures — never re-plow):** EURUSD/GBPUSD
price-derived signals at M5–H4 (indicator systems, session breakouts
both directions, fix flows, EURGBP relative value, regime-conditioned
reversion, mechanical ICT); mean_reversion (H4-range-gated RSI) on all
majors; London-open range breakout everywhere; NY-overlap continuation;
H4 Donchian/ATR; weekly cross-sectional momentum; AMR on non-JPY pairs;
gold AMR and gold NY-momentum; breakeven/trailing exits on the current
book (except BE@0.75R on AMR-GBPJPY: marginal, backtest-only).

**Principles established:** portfolio = validated edges, not pair
count (adding losers measurably raised bust risk to 45%). Movement ≠
money; the most liquid pairs are the hardest. Filters remove trades
but cannot create edge. Exit tuning polishes real edges only. Every
idea faces the same IS/OOS bar — including "popular" ideas (ICT) and
the owner's own live strategies.

---

## 5. LIVE FORWARD-TEST RECORD (account 5052472770)

**June (laptop era, old code):** 23 trades, +$849, PF 1.33 — a hot
month for what audits later proved are losing strategies. MAX_LOT cap
silently limited JPY breakout risk to ~0.2%. Two manual closes
detected (the anti-goal of the project). Laptop retired 2026-07-01.

**Week 1 (Jul 5–12, Book B+ era):** 7 closed: AMR 1W/4L (−$498),
ARB 0W/2L (−$556, one a $0 breakeven-rule scratch), plus one old-code
EURUSD loss. Equity trough −1.30%. Worst day −0.50%.

**Week 2 (Jul 13–15):** 3 closed, 3 wins +$1,029: CADJPY ARB full 2:1
TP +$633 (27h — the trade class the removed BE rule used to scratch),
XAUUSD ARB +$314 (gold slot's debut), EURJPY AMR +$82. GBPJPY ARB open
+~$240. **Equity ≈ $99,965 — first-week drawdown fully recovered.**

**Defects caught by the forward-test (the reason it exists):**
1. Laptop/VPS version drift (June): 17:30 EOD close still live on old
   deployment.
2. **Server-time coordinate bug** (e75d680): 3 of 8 slots structurally
   unable to fire, AMR truncated — invisible in backtests by
   construction.
3. **Legacy breakeven interference** (4691a22): live exit policy
   diverged from every validated backtest.
4. **MT5 update dialog outage** (Sun Jul 12 ~21:00–22:45 real): froze
   the terminal exactly through monday_drift's debut window; bot died
   22:45. Fixes: weekend maintenance task (bfcf595) + no-signal
   observability logging (a92cdc3).

**Known live quirks:** local terminal history sync is lazy — poke
`history_orders_get` before `history_deals_get` or recent days are
missing (bit twice). MAX_LOT 2.0 caps tight-SL JPY AMR trades to
~0.12–0.2% (below the 0.25% intended; safe direction). ARB realized
risk ran +10.7% over intended once (pip-value estimation drift —
monitoring). Zero AMR signals on some quiet days is normal.

---

## 6. DECISION RULES IN FORCE (agreed with owner — do not improvise)

1. **AMR checkpoint:** review at 20 closed AMR trades or Aug 1
   (whichever first). Standing at 6/20 (2W/4L). If WR still ≤~40–45%,
   cut all four AMR slots. **No parameter tweaks before the checkpoint.**
2. **Challenge gate (advisory):** demo ≥1%/month with max DD <5% over
   2–3 months. The owner is buying the 5ers account EARLY for YouTube
   content reasons (informed decision; a failed challenge is an
   episode). Mitigation: prop instance starts at risk_scale 0.5; 5ers
   has no time limit.
3. **No manual intervention on positions.** The system's edge includes
   hands staying off the terminal.
4. Every new strategy idea passes the standard harness bar before
   touching a YAML.

---

## 7. OPERATIONAL RUNBOOK

- **Deploy:** stop bot (Ctrl+C) → `git pull origin main` → `python
  verify_architecture.py` (expect **24/24**, active pairs AUDJPY,
  CADJPY×2, EURJPY, GBPJPY×2, GBPUSD, XAUUSD) → restart → check banner:
  strategy keys, account config line, and `MT5 terminal bound: ...
  account=5052472770`. **On the 5ers clone**, also glance at trading.log
  for a `pair_manager: LOCKED instance -- excluded pairs ...` WARNING
  after restart — see §2.5 `locked_pairs`. None expected unless someone
  pushed a new demo pair since the last deploy; if one appears and it's
  unexpected, investigate before assuming it's fine.
- **Weekend maintenance (prevents update-dialog outages):**
  scripts/weekend_maintenance.ps1 — Saturdays: stop bots → restart
  terminals (updates install with market closed) → start bots. Register
  via the schtasks line in its header; **edit the template paths first**.
- **Monitoring:** data/logs/trading.log (UTC timestamps). Signals log as
  `ARB/AMR/MON SIGNAL <key> ...`; operational failures now log as
  WARNINGs; @mon checks always log outcomes. Trade audit from any
  machine logged into the account: reconstruct positions from
  history_deals_get grouped by position_id (entry deal comment carries
  the strategy tag `5ers_<session>_<side>_<label>`).
- **Expected cadence:** ~25–40 trades/month across the book; AMR
  0–3/night, ARB ~5–6/month/pair, MON 1/week (Sunday ~22:00 real).
- **Next scheduled analysis:** first monthly demo-vs-backtest
  comparison early August.

---

## 8. ROADMAP

**Immediate (this week):**
- Pull a92cdc3 on VPS (observability) — pending.
- Sat Jul 18: first weekend-maintenance run (verify paths edited).
- Sun Jul 19 ~21:00–23:00 real UTC: monday_drift's real debut — check
  `Select-String trading.log -Pattern "MON"` Monday morning.

**The 5ers challenge instance ($5K 2-step CLASSIC, purchased 2026-07):**
- Official rules (screenshots on file, FAQ dated 2026-06-14): step 1 +8%,
  step 2 +5%, max daily loss 5%, max loss 10% STATIC from initial
  balance (equity stop-out $4,500), unlimited time, min 3 profitable
  days/step, $39. News: holding over news allowed; EXECUTING orders
  ±2 min around high-impact news (Forex Factory, SERVER time)
  prohibited — our gate uses the same feed with a ±5 min window
  (stricter, config `news_window_min`). Overnight/weekend holds
  allowed. Multiple logins same location/IP allowed (both bots on one
  VPS = compliant). Metals hours 01:05–23:50 EET (gold ARB unaffected).
- Phase-9 Monte Carlo (src/phase9_5k_challenge_sim.py) with 0.01-lot
  granularity: **risk_scale 1.0 → 2.56%/mo; step 1: 91% pass within a
  year / 9% bust / median 48 trading days; step 2: 94% / 6% / 28d.**
  risk_scale 0.5 is STRICTLY WORSE (84% / 12% bust / 81d): min-lot
  flooring keeps risk up while profit halves. Decision: run FULL risk.
- Instance config (config/local_config.yaml in its clone):
  starting_balance 5000, max_lot 0.5, risk_scale 1.0,
  mt5_terminal_path C:\MT5-5ers\terminal64.exe, **news_fail_closed:
  true** (a862894: unavailable calendar blocks entries on this
  instance; demo stays fail-open).
- Same 8-slot book, zero strategy changes — the demo-vs-prop twin-fill
  comparison is the experiment.

**When the 5ers account arrives:**
1. Log into the C:\MT5-5ers terminal (File → Login, search their
   server, save credentials). 2. Second clone + its local_config.yaml
   (starting_balance, max_lot scaled, risk_scale 0.5, terminal path).
   3. Flip news gate to FAIL-CLOSED for that instance (code change,
   small). 4. Verify 5ers symbol names (suffix mapping layer if their
   Market Watch shows e.g. GBPJPY.x). 5. Start; audit first fills vs
   demo (same signals, seconds apart → execution-quality comparison).

**Near-term engineering backlog:**
- AMR checkpoint execution (Aug 1 / 20 trades) — the health monitor
  now computes the supporting statistics daily.
- Monitor ARB realized-risk drift; consider computing pip value from
  cross rates instead of tick_value for JPY pairs.
- MAX_LOT interplay with AMR sizing (intended 0.25% often capped).
- Restart-resilience for the AMR window's midnight state rollover
  (documented residual: an in-window bot restart could allow one
  re-entry).

**Research directions (untested ground, in priority order):**
1. Tokyo fix (00:55 server) flows on JPY crosses.
2. Sydney-session structure on AUD/NZD (needs cross-midnight window
   support in the harness engine).
3. NZDJPY AMR (OOS 1.35–1.76, IS ~1.0 — same regime profile; candidate
   if the AMR family survives its checkpoint).
4. Second data source (free Dukascopy tick data) to cross-validate the
   book and enable USDCAD-oil correlation work.
5. Exit study round 2 ONLY if live data suggests it (partial-TP was
   never tested).
6. Meta-labeling (tiny logistic model on journaled signal context to
   size/skip signals) once data/journal/events.jsonl reaches ~1000+
   trades — the journal schema was designed for this.

**Explicitly NOT planned:** any new EURUSD/GBPUSD price-signal search
at retail data tier; ICT variants; martingale/grid anything; carry
trades (incompatible with Friday-close + daily-DD rules).

---

## 9. REFERENCE INDEX

- **Commit trail (all on origin/main):** a17520f risk gates → edbe008
  orchestrator/@arb+@amr → 8d4ddf5 strategies → e1f3ec9 Book B+ configs
  → 6d3ddc9 research harnesses → 708f12c news gate → 43284a6
  monday_drift → e75d680 **server-time fix** → 84358fd multi-account →
  9386186 portable flag → 9a62188 local_config overlay → 4691a22
  breakeven exclusion → bfcf595 weekend maintenance → a92cdc3
  no-signal observability → 893c832 this report → fc291ed trade
  journal + health monitor.
- **Key modules:** strategies/{asian_range_breakout, asian_hours_reversion,
  monday_drift, registry}.py · core/{news_calendar, trade_journal,
  health_monitor, data_loader, session_filter, pair_manager,
  strategy_loader}.py · src/agents/*.py · verify_architecture.py
  (22 checks).
- **Research artifacts:** src/strategy_matrix_backtest.py (core engine)
  + src/phase*.py + src/revalidate_eurusd_live.py; results in data/*.csv
  and data/phase*_report.txt (regenerable; gitignored).
- **AI-assistant memory:** persistent notes live outside the repo in the
  Claude project memory (index: MEMORY.md there); this document is the
  repo-side equivalent and should be updated at each milestone.
- **Accounts:** demo/control 5052472770 (MetaQuotes-Demo, $100k
  2026-07-01) · 5ers: pending purchase · retired laptop account 106040846.
