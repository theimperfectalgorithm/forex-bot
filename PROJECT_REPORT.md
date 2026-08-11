# Forex Bot — Complete Project Report

**Date:** 2026-08-11 (rev 3, 5ers live + dashboard + reliability hardening) ·
**Supersedes:** rev 2 (2026-07-15)
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

**Since rev 2 (2026-07-15 → 2026-08-11), in order:**
- The 5ers $5K Classic account went live (~2026-07-19, login 26520700).
  A cross-terminal MT5 contamination bug was found and fixed (bare
  `mt5.initialize()` calls could silently attach to the WRONG account's
  terminal when two run on one VPS) — see §2.5.
- A read-only mobile dashboard (`mcp/server.py` + `mcp/dashboard.html`,
  ports 8000/8001) was built, then hardened twice after a real outage —
  see §2.8.
- Two new live safety nets were added directly in response to real
  incidents: a pre-trade spread gate (rejects entries when live spread
  eats too much of the stop, after a rollover-spread incident cost real
  trades) and reverse-direction position reconciliation (catches an MT5
  position the bot doesn't know about — the class of bug a Reddit
  reviewer flagged and the cross-terminal incident made concrete). Both
  in §2.4.
- A VPS watchdog now self-heals any of the 4 scheduled processes within
  15 minutes if they're down for any reason, including an operator
  manually stopping one mid-deploy and forgetting to restart it (this
  cost a full trading day twice before the watchdog existed). §2.9.
- **A real, live-money-adjacent bug was found via the new dashboard's
  slippage analytics**, not by inspection: `agent_execution.py` trusted
  `order_send()`'s immediate `result.price` as the fill price, which is
  unreliable on market-execution/ECN brokers (5ers) — intermittently
  `0.0` for genuinely correctly-executed trades. Actual trading was
  never affected (SL/TP/PnL never depended on this value), but the
  *recorded* entry price — and everything derived from it (slippage,
  R-multiples) — was wrong for the affected trades. Fixed by confirming
  the real fill via `positions_get()` instead. §5.
- **The live book diverged between clones for the first time
  (2026-07-31):** `GBPJPY@arb` and `XAUUSD@arb` were demoted to
  demo-only on the 5ers account (0/3 record + stop-loss overshoot for
  the former; broker data-availability + lot-size floor for the
  latter), and `risk_scale` was cut to 0.5 while the reduced book
  rebuilds a track record. Demo continues running the full original
  8-slot book unchanged. §3.
- AMR (mean-reversion) is mid-way through a live investigation: a
  trending-JPY stretch in early August produced a real losing cluster.
  Root-caused to AMR having zero higher-timeframe trend filter (by
  design) — not a bug. A 2-week observation window is running before
  deciding whether to research and backtest a trend filter. §6, §8.

---

## 2. SYSTEM ARCHITECTURE

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

### 2.4 Risk gates (agent_risk.run, all 11 in order)
1. MT5 connection + account read
2. Hard floor on BALANCE (starting_balance × (1 − hard_floor_pct))
3. Hard floor on EQUITY (floating losses count)
4. **Untracked-position reconciliation** (added 2026-08-03): rejects new
   entries if main_agent's `_check_untracked_positions()` found an MT5
   position this bot placed (magic-matched) that wasn't in its own
   `open_trades` tracking, this UTC day. The position itself is adopted
   into normal monitoring the moment it's found (see §2.9); this gate
   only pauses NEW entries so a human reviews the log before the book
   resumes fully unattended. Pure state-flag read, no MT5 call.
5. Daily loss limit — closed P&L (daily_loss_pct × starting_balance)
6. Daily EQUITY soft stop at −4% vs day's first-seen equity anchor
   (buffer inside the firm's −5%)
7. Per-pair 2-consecutive-loss daily pause
8. Aggregate open risk: Σ(entry→SL risk) + new trade ≤ 3% of balance;
   any open position missing an SL blocks all new trades
9. Currency concentration: max 2 open positions sharing a currency
10. News blackout: no entries within ±5 min of high-impact news
    (core/news_calendar.py, Forex Factory weekly JSON, 6h cache).
    Fails OPEN on demo; **fails CLOSED on the 5ers instance**
    (`news_fail_closed: true`). Time exits (@amr/@mon) also defer
    inside a blackout.
11. **Spread gate** (added 2026-07-31, after a real incident): rejects
    an entry when live spread exceeds `spread_max_frac_of_sl` (config,
    default 30%) of the trade's SL distance. July 2026 journal data
    showed the live broker widening spreads to 12–31 pips at the
    server-midnight session rollover while AMR stops were 8–17 pips —
    the broker was rejecting most of those orders anyway
    (`retcode=10016 Invalid stops`), but marginal ones got through
    paying most of the stop away in spread alone. This makes that
    protection deliberate and journaled, on both accounts, and also
    catches the marginal cases the broker itself let through.

Sizing: lots = balance × risk_percent(YAML) × risk_scale(config) /
(sl_pips × pip_value). Pip value from MT5 tick data except gold
(computed from contract size — MetaQuotes reports broken tick_value for
metals when market closed). MAX_LOT clamp from config.

### 2.5 Multi-account / multi-instance
Two MT5 terminals on the VPS: `C:\Program Files\MetaTrader 5\` (demo
5052472770) and `C:\MT5-5ers\` (5ers, login 26520700, live since
2026-07-19). One bot process per terminal, from separate git clones of
this repo. Per-instance settings live in **gitignored
`config/local_config.yaml`** (same schema as global_config's `global:`
block) — never edit the tracked config on a VPS. Required keys per
instance: `mt5_terminal_path` (pins the process to its terminal at
startup — mandatory with 2 terminals), plus for the prop clone:
`starting_balance`, `max_lot` (scale down!), `risk_scale` (0.5 as of
2026-07-31, see §3). Password only via terminal's saved login or
MT5_PASSWORD env var. Never run one process switching accounts (MT5
python API is a per-process singleton).

**Cross-terminal contamination incident (2026-07-21, fixed same day):**
`_bind_mt5_terminal()` correctly binds each process to its own terminal
at startup, but a bare `mt5.initialize()` call **later in the same
process** (any of ~20 call sites across the codebase) was found to
silently re-attach to the *other* terminal when two MT5 terminals run
concurrently on one VPS — the demo process briefly read the 5ers
account's balance. Fixed centrally: `core/mt5_connect.py` monkey-patches
`MetaTrader5.initialize` process-wide at import time (imported first
thing in `main_agent.py` and `mcp/server.py`) to force every call,
bare or not, onto that instance's configured terminal/login — fixes
every call site including any not yet written, without touching 20
files individually. The incident's data fingerprint (two poisoned
`equity_curve.csv` rows showing the wrong account's balance) was found
and cleaned up manually after the fact.

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

### 2.8 Mobile dashboard (mcp/server.py + mcp/dashboard.html, added 2026-08)
Read-only "control center" web dashboard, one server process per VPS
clone (MT5's python API is a per-process singleton, so one process
can't serve both accounts) — demo on port 8000, 5ers on 8001, identical
code, per-clone `mcp/.env`. Bookmarked as
`http://<vps-ip>:<port>/dash?key=<DASH_TOKEN>`; the page fetches both
ports client-side and renders a tab per account, degrading gracefully
if one is down. `DASH_TOKEN` is a separate secret from `MCP_API_KEY`
(the MCP-protocol key never touches a browser).

Endpoints (all under `/api/*`, all plain `def` not `async def` — see
the reliability note below): `/summary` (live balance/equity/positions
+ challenge progress from that clone's `core.account_config`),
`/equity` (EOD curve + one live point), `/trades` (recent closed
trades, each with an R-multiple), `/stats` (win rate / profit factor /
expectancy over a period, with documented definitions since these
numbers get quoted publicly), `/journal` (entry+exit cards joined by
ticket, honest text straight from the bot's own journal), `/news`
(upcoming high-impact events), `/state` (session prep flags,
trade-allowed verdict, paused pairs), `/slippage` (added 2026-08-07 —
avg/worst slippage by pair and session; **this is what surfaced the
fill-price bug in §5**, nobody had ever looked at the already-collected
`slippage_pips` field before). A client-side canvas renderer generates
a 5-slide Instagram carousel + 1920×1080 YouTube thumbnail from the
stats data on demand, entirely in-browser (zero added VPS load).

**Reliability incident (2026-08-03) and fix:** the demo dashboard
server hung and had to be manually restarted. Root cause: several
`/api/*` routes did blocking file I/O (`open()`/`csv`/`json`) inside
`async def` handlers — a blocking call in an async route stalls
uvicorn's *entire* single event loop, not just that request, and a
pile-up of routine internet-scanner connections on the open port made
it worse. Fixed by declaring every data route as plain `def` (FastAPI
runs those in a thread pool automatically) plus a 10s client-side fetch
timeout. Separately hardened against the scanner noise itself: an IP
racking up 15+ failed-auth attempts in 60s gets an instant 429 with
near-zero server work, and `uvicorn`'s `limit_concurrency=100` caps any
other kind of burst. (A firewall IP-allowlist was considered and
rejected — mobile data has no stable IP to lock to via carrier-grade
NAT; a Cloudflare Tunnel was also considered but deferred, no domain
currently owned.)

### 2.9 VPS process reliability (scripts/watchdog.ps1, added 2026-08-06)
All 4 scheduled tasks (`ForexBot`, `ForexBot-5ers`, `ForexBotMCP`,
`ForexBotMCP-5ers`) already auto-start on VPS boot and auto-restart on
crash (`RestartCount`/`RestartInterval`, `ExecutionTimeLimit: PT0S` —
the default 72h kill switch was found and removed from all 4). That
setting only catches the scheduler's *own* launched process exiting on
its own, though — it does not catch an operator manually stopping a
process (mid-deploy, debugging) and forgetting the follow-up
`schtasks /run`, which is the actual pattern that has cost full trading
days on this VPS (e.g. a Tuesday deploy left the demo bot down through
all of the following Wednesday). `scripts/watchdog.ps1`, registered as
its own task running every 15 minutes, checks real process/port state
— bots by `main_agent.py`'s full path in `CommandLine`, dashboards by
listening port (8000/8001) since `mcp\server.py` is launched with a
relative path that can't tell the two clones apart — and restarts
(`schtasks /run`) whichever of the 4 isn't actually running, regardless
of *why*. Read-only except for that one call; verified end-to-end by
deliberately killing a process and confirming the watchdog brought it
back within one manual trigger.

---

## 3. THE LIVE BOOK — 8 slots on demo; 6 active on 5ers since 2026-07-31

**The book diverged between clones for the first time on 2026-07-31**
(see §2.5's `locked_pairs` mechanism). Demo keeps running the original,
full 8-slot table below, unchanged. On the 5ers clone, `locked_pairs`
now excludes:
- **`GBPJPY_asianrange.yaml` (row 1, ARB)** — 0 wins / 3 losses on the
  funded account, including its two biggest single losses, with
  realized R consistently overshooting plan (up to −1.65R against a 1R
  design — min-lot granularity + live spread on a $5K account inflates
  loss magnitude vs. backtest assumptions).
- **`XAUUSD_asianrange.yaml` (row 3, ARB)** — not performance-related:
  the 5ers broker has no H1 gold data available at the bot's 04:45 UTC
  London-prep check **every single day** (confirmed via a week of
  `WARNING: no H1 data for today` log lines), so it structurally misses
  the session; separately, gold's 400–600 point SL at 0.25% risk on $5K
  needs ~0.002–0.004 lots, below the 0.01 broker minimum, so it could
  never size correctly even when armed.

5ers also runs at **`risk_scale: 0.5`** (was 1.0) since the same date,
while the reduced 6-slot book (row 2 + rows 4–8 below) rebuilds a track
record. Both changes were one manual, on-machine `local_config.yaml`
edit on the 5ers clone — no code change, no effect on demo. See §5 for
how the reduced book has performed since, and §6/§8 for the live AMR
investigation running in parallel.

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

## 5. LIVE FORWARD-TEST RECORD (both accounts)

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

**5ers account (login 26520700, live since 2026-07-19, $5,000):**
- **Weeks 1–2 (Jul 19–31):** rough opening — 0/4 on day one, GBPJPY@arb
  compounding to its two worst-ever losses (−$34.19, −$40.78). Cross-
  terminal contamination (§2.5) briefly poisoned the equity history
  with the wrong account's balance during this window (cleaned up
  manually). Ended the period at roughly −2.7%.
- **2026-07-31: book demotion + risk cut** — see §3. Applied on-machine,
  verified live in the very next startup banner (`Strategy keys:
  CADJPY@arb, AUDJPY@amr, CADJPY@amr, EURJPY@amr, GBPJPY@amr,
  GBPUSD@mon`, `risk_scale=0.5`, and a `pair_manager: LOCKED instance
  -- excluded` WARNING naming the two demoted pairs).
- **Aug 3–7 (first post-demotion window, 10 trades):** 20% win rate,
  profit factor 0.22, expectancy −0.30R, net −$36.97. Losses are now
  uniformly small (−$2.68 to −$8.72) with **no outlier blowups** — the
  demotion fixed the catastrophic-loss failure mode, but the account
  had not yet turned the corner as of this check. Sample still too
  small (n=10) to draw a statistical conclusion.
- **2026-08-08: fill-price logging bug found and fixed** — see §2.8's
  slippage-card note. `agent_execution.place_trade()` logged `0.0` as
  the entry price for some trades (confirmed via raw journal +
  `trades_log.csv` cross-check on the same ticket) because
  `order_send()`'s immediate `result.price` is unreliable on this
  market-execution broker. Real SL/TP/PnL were never affected — only
  the recorded entry price, which corrupted slippage/R-multiple
  analysis for the affected trades. Fixed going forward
  (`_confirm_fill_price()`, confirms via `positions_get()` with short
  retries); historical corrupted rows left as-is by explicit decision.
- **Early Aug: AMR trending-JPY losing cluster** — a genuine multi-day
  CADJPY/AUDJPY uptrend (confirmed on the H1 chart, not a data
  artifact) ran through several AMR mean-reversion SELL signals in
  succession. Root-caused to `strategies/asian_hours_reversion.py`
  having **zero higher-timeframe trend filter by design** (its z-score
  is computed over only the last 5 hours of M15 bars) — not a bug. The
  strategy's own 2026-07-05 validation note had already flagged this
  exact risk ("persistence is unproven... run 2–3 months on demo before
  ANY challenge use") and was overridden by the later decision to run
  the identical book on both accounts from day one. See §6/§8 for the
  live decision in force.

---

## 6. DECISION RULES IN FORCE (agreed with owner — do not improvise)

1. **AMR trend-regime watch (in force as of 2026-08-11, ~2 weeks →
   ~Aug 25):** the early-August trending-JPY losing cluster (§5) is
   being observed, not reacted to. At the checkpoint, pull
   `core.health_monitor` output + the dashboard's per-pair stats. If
   AMR's win rate/expectancy on JPY crosses has recovered toward its
   backtested expectation → no action, treat as an expected trending-
   regime dip. If it's still degrading → scope, build, and properly
   backtest (full IS/OOS discipline, §4's methodology) a higher-
   timeframe trend filter for AMR **before** touching the live
   strategy — never patch a live signal blind. (Supersedes the original
   "review at 20 closed AMR trades or Aug 1" rule from rev 2, which was
   overtaken by the 5ers book-demotion decision on 2026-07-31 — GBPJPY
   breakout, not AMR broadly, turned out to be the acute problem; AMR's
   watch continues on its own track.)
2. **Challenge gate (advisory):** demo ≥1%/month with max DD <5% over
   2–3 months. The owner is buying the 5ers account EARLY for YouTube
   content reasons (informed decision; a failed challenge is an
   episode). Mitigation: prop instance runs at risk_scale 0.5 (cut from
   1.0 on 2026-07-31, see §3); 5ers has no time limit.
3. **No manual intervention on positions.** The system's edge includes
   hands staying off the terminal.
4. Every new strategy idea passes the standard harness bar before
   touching a YAML.
5. **Demoting/promoting a pair-strategy on the funded account is always
   a deliberate, on-machine `local_config.yaml` edit** — never a side
   effect of a `git push` made while working on the demo box (§2.5).
6. **Historical data corrections are opt-in, not automatic.** When a
   logging bug is found (e.g. §5's fill-price bug), the default is fix
   going forward only; backfilling/correcting already-written
   `trades_log.csv`/journal rows requires an explicit decision each
   time, not a blanket policy.

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
  the strategy tag `5ers_<session>_<side>_<label>`). **Easiest day-to-
  day check is now the mobile dashboard** (§2.8) — bookmark
  `http://<vps-ip>:8000/dash?key=<DASH_TOKEN>` (add `&tab=1` for the
  5ers tab directly). Watch for a `pair_manager: LOCKED instance --
  excluded` WARNING after any 5ers restart (§2.5), and an `UNTRACKED
  POSITION FOUND` ERROR (§2.4 gate 4 / §2.9) if reconciliation ever
  fires — both should be rare.
- **If a process seems down:** `scripts/watchdog.ps1` (§2.9) self-heals
  within 15 minutes on its own; no manual restart is normally needed.
  To force it immediately: `powershell -File
  C:\forex-bot\scripts\watchdog.ps1` then check
  `data\logs\watchdog.log`.
- **Expected cadence:** ~25–40 trades/month across the demo book (5ers
  runs a reduced 6-slot book, see §3); AMR 0–3/night, ARB ~5–6/month/
  pair, MON 1/week (Sunday ~22:00 real).
- **Next scheduled analysis:** AMR trend-regime checkpoint ~2026-08-25
  (§6, rule 1); end-of-August full month review (win rate/profit
  factor, whether risk_scale returns to 1.0 for September).

---

## 8. ROADMAP

**5ers challenge instance — status:** live since 2026-07-19, login
26520700. Official rules (screenshots on file, FAQ dated 2026-06-14):
step 1 +8%, step 2 +5%, max daily loss 5%, max loss 10% STATIC from
initial balance (equity stop-out $4,500), unlimited time, min 3
profitable days/step, $39. News: holding over news allowed; EXECUTING
orders ±2 min around high-impact news (Forex Factory, SERVER time)
prohibited — our gate uses the same feed with a ±5 min window
(stricter, `news_window_min`), and fails CLOSED on this instance
(`news_fail_closed: true`). Overnight/weekend holds allowed. Metals
hours 01:05–23:50 EET. Running the reduced 6-slot book at risk_scale
0.5 since 2026-07-31 (§3) — phase-9's original Monte Carlo recommended
full risk_scale 1.0 for the 8-slot book, but that study predates the
GBPJPY@arb demotion and the live min-lot-granularity findings; it has
not been rerun against the current reduced book (candidate for the
research backlog below).

**Immediate — in progress right now:**
- **AMR trend-regime watch**, checkpoint ~2026-08-25 (§6 rule 1).
- **New strategy research: JPY crosses in London/NY sessions**
  (diversification away from the current Asian-session-only
  correlation risk — AMR and ARB are both fundamentally session-
  structural bets on the same 00:00–09:00 UTC window). Chosen over
  three other candidates (commodity-bloc crosses AUD/NZD/CAD vs each
  other; London/NY gold; JPY-cross relative value) because it extends
  a *proven* edge (JPY-cross mean reversion, Asian hours) into
  untested territory (different session, likely trend-following
  rather than reversion) rather than starting from zero — cheapest to
  test, most likely to actually work. **Reminder: London/NY on
  EURUSD/GBPUSD is dead ground already (~470 failures, §4) — do not
  default back to the obvious majors.**
  - **Blocked on a data export**, in progress: local Mac CSV cache
    (`data/historical/`) only has 7 pairs (AUDUSD, EURJPY, EURUSD,
    GBPJPY, GBPUSD, NZDUSD, USDJPY) at H1/H4 — no M15 (AMR's own
    timeframe), and missing AUDJPY/CADJPY entirely despite both being
    live-traded pairs already. `scripts/export_historical_data.py`'s
    `PAIRS`/`TIMEFRAMES` lists need AUDJPY, CADJPY, M15, plus whichever
    new JPY crosses get chosen (NZDJPY, CHFJPY candidates), then a
    one-time run on a Windows/MT5-connected machine (VPS or a laptop),
    committed to git (these CSVs are ~3MB/pair/timeframe, NOT
    gitignored, so `git push`/`pull` is the sync mechanism — no scp/
    cloud-sync tooling needed) and pulled to continue development on
    Mac, same as every other backtest script.

**Near-term engineering backlog:**
- Monitor ARB realized-risk drift; consider computing pip value from
  cross rates instead of tick_value for JPY pairs.
- MAX_LOT interplay with AMR sizing (intended 0.25% often capped).
- Restart-resilience for the AMR window's midnight state rollover
  (documented residual: an in-window bot restart could allow one
  re-entry).
- Re-run phase-9's Monte Carlo against the actual current 6-slot/
  risk_scale-0.5 5ers book (the original study assumed the full 8-slot
  book at risk_scale 1.0 — no longer the live configuration).
- Consider historical-data backfill/correction for the fill-price bug
  (§5, §6 rule 6) if it turns out to matter for a specific piece of
  analysis — opt-in only, not automatic.

**Research directions (untested ground, in priority order):**
1. **JPY crosses in London/NY sessions** — in progress, see above.
2. Commodity-bloc crosses (AUDCAD, NZDCAD, AUDNZD) — phase 6 found a
   genuine CADJPY edge via cross-sectional momentum; these have never
   been tested at all and are far less efficiently arbed than EUR/GBP/
   USD majors.
3. Gold, London/NY session-specific (distinct from the existing
   Asian-hours, currently-provisional gold ARB) — gold's real
   directional action tends to cluster around US data/real-yield moves
   in London/NY hours; would need to account for the same broker
   data-availability gap that got XAUUSD demoted from 5ers (§3).
4. Tokyo fix (00:55 server) flows on JPY crosses.
5. Sydney-session structure on AUD/NZD (needs cross-midnight window
   support in the harness engine).
6. Second data source (free Dukascopy tick data) to cross-validate the
   book and enable USDCAD-oil correlation work.
7. Exit study round 2 ONLY if live data suggests it (partial-TP was
   never tested).
8. Meta-labeling (tiny logistic model on journaled signal context to
   size/skip signals) once data/journal/events.jsonl reaches ~1000+
   trades — the journal schema was designed for this.

**Explicitly NOT planned:** any new EURUSD/GBPUSD price-signal search
at retail data tier; ICT variants; martingale/grid anything; carry
trades (incompatible with Friday-close + daily-DD rules); an AMR trend
filter without first completing the §6-rule-1 backtest discipline;
auto-closing positions on daily-loss breach (conflicts with "pausing is
a human decision" — reserved for hard-floor breach only); a live/
automatically-recalculating Monte Carlo kill-switch (the existing
offline phase-9-style study is sufficient at current trade volume); AI
trade-reviewer/regime-detector features in the live trading loop
(content-generation use only, never a live risk decision).

---

## 9. REFERENCE INDEX

- **Commit trail through rev 2 (all on origin/main):** a17520f risk
  gates → edbe008 orchestrator/@arb+@amr → 8d4ddf5 strategies →
  e1f3ec9 Book B+ configs → 6d3ddc9 research harnesses → 708f12c news
  gate → 43284a6 monday_drift → e75d680 **server-time fix** → 84358fd
  multi-account → 9386186 portable flag → 9a62188 local_config overlay
  → 4691a22 breakeven exclusion → bfcf595 weekend maintenance →
  a92cdc3 no-signal observability → 893c832 this report (rev 1) →
  fc291ed trade journal + health monitor (rev 2).
- **Commit trail since rev 2:** a95b597 NEWS logger fix → db4f7db
  **cross-terminal MT5 contamination fix** (core/mt5_connect.py) →
  95492ed `locked_pairs` isolation → b9ae7f6 mobile dashboard v1 →
  c79d09e dashboard v2 control center → e445d7a/cd4f16f dashboard
  layout/week-nav fixes → 9a54f2a slippage aggregation card →
  b2f9bcf **spread gate** → c7297ab **position reconciliation** →
  9fb21de/a1e77bd dashboard reliability (blocking-I/O fix + rate
  limiting) → 6766d72 **VPS watchdog** → 0b64c02 **fill-price capture
  fix**.
- **Key modules:** strategies/{asian_range_breakout, asian_hours_reversion,
  monday_drift, registry}.py · core/{news_calendar, trade_journal,
  health_monitor, data_loader, session_filter, pair_manager,
  strategy_loader, mt5_connect, account_config}.py · src/agents/*.py
  (incl. `_confirm_fill_price`, `find_untracked_positions` in
  agent_execution.py; `_check_untracked_positions` in main_agent.py) ·
  mcp/{server, dashboard.html, backtest_engine}.py (mobile dashboard +
  MCP protocol server) · scripts/watchdog.ps1 · verify_architecture.py
  (24 checks).
- **Research artifacts:** src/strategy_matrix_backtest.py (core engine)
  + src/phase*.py + src/revalidate_eurusd_live.py; results in data/*.csv
  and data/phase*_report.txt (regenerable; gitignored). Historical bar
  cache for Mac-side backtesting: data/historical/*.csv (NOT
  gitignored, synced via git — see §8's data-export task).
- **AI-assistant memory:** persistent notes live outside the repo in the
  Claude project memory (index: MEMORY.md there); this document is the
  repo-side equivalent and should be updated at each milestone.
- **Accounts:** demo/control 5052472770 (MetaQuotes-Demo, $100k,
  2026-07-01) · 5ers 26520700 (Five Percent Online, $5,000, live
  2026-07-19, reduced 6-slot book + risk_scale 0.5 since 2026-07-31) ·
  retired laptop account 106040846.
