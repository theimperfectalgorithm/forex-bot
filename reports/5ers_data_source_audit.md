# 5ers Dashboard Data-Source Audit

**Scope:** Architecture inspection only. No strategy analysis, no interpretation, no portfolio changes. Answers the 13 questions from the prior forensic analysis's blocker ("could not access the actual 5ers MT5 terminal from this machine") plus the requested export/audit deliverables.

**Files inspected:** `mcp/server.py` (dashboard/API backend), `core/trade_journal.py` (journal writer), `src/agents/agent_execution.py` (order placement, trade closing, MT5 comment tagging), local `data/` directory, local MT5 connection.

---

## 1-13. Architecture questions

| # | Question | Answer |
|---|---|---|
| 1 | Where does the 5ers trade data originate? | Two flat files per bot instance/clone: `data/trades_log.csv` and `data/journal/events.jsonl`. There is no database. The dashboard (`mcp/server.py`) reads these files directly at request time — it does not store or cache trade data itself. |
| 2 | Does it come from MT5? | Yes, indirectly. `agent_execution.py` writes `trades_log.csv` rows at order-placement and at close-detection time, using `mt5.order_send()` results and `mt5.history_deals_get()` deal lookups (`_get_closed_deal`/`_format_exit_deal`, lines ~420-453). `core/trade_journal.py` additionally pulls live tick/rate data from MT5 (`market_context()`) at entry-logging time. |
| 3 | Which MT5 terminal/account does it use? | Whichever terminal the bot process is bound to via `core.mt5_connect` (imported at the top of `mcp/server.py` — pins `mt5.initialize()` to "this clone's terminal" to prevent cross-terminal contamination when two terminals run on the same VPS). Each clone (demo vs. 5ers) is a separate directory with its own `local_config.yaml` and its own MT5 terminal path, so a dashboard process only ever sees its own clone's data. **This machine's dashboard/bot process, if run here, would be bound to the DEMO terminal only** — confirmed directly (see §"Local access verification" below), not assumed. |
| 4 | SQLite/Postgres/CSV/JSON/etc? | CSV (`trades_log.csv`) + JSONL (`journal/events.jsonl`). No SQL database, no ORM, anywhere in `mcp/server.py`, `core/trade_journal.py`, or `agent_execution.py`. |
| 5 | Is there an internal API endpoint? | Yes: `/health`, `/api/summary`, `/api/equity`, `/api/trades?limit=N`, `/api/stats?days=N`, `/api/journal?limit=N`, `/api/slippage`, `/api/news`, `/api/state`, plus a browser view at `/dash` — all in `mcp/server.py`, gated by a `DASH_TOKEN` separate from the MCP tool-call `MCP_API_KEY`. These endpoints read the same two flat files described above; they are not a separate data store. |
| 6 | Does the dashboard have all historical fields we need? | No single source has everything — see the field table in §2. `trades_log.csv` has the execution skeleton (prices, lots, P&L, exit reason) for every closed trade. `journal/events.jsonl` has the richer fields (strategy attribution key, strategy_reason text, spread/ATR at entry, slippage, intended risk) but **only for trades opened after journaling was added to that clone** — it does not backfill history. |
| 7 | Is strategy attribution stored? | Yes, two independent ways: (a) `journal/events.jsonl`'s `key` field on signal/entry/exit events (e.g. `GBPJPY@arb`, `AUDJPY@amr`), joinable to `trades_log.csv` by MT5 `Ticket`; (b) the MT5 order `comment` field itself, written as `5ers_{session}_{signal}_{strategy}` (`agent_execution.py:257`) — this is stored inside MT5's own deal history, independent of either local file, so it survives even if local logs are lost. `trades_log.csv` alone has no strategy column — only `Pair`+`Session`, which is an approximation (unambiguous only if no two active strategies share both). |
| 8 | Is entry/exit time stored? | Yes. `trades_log.csv`: `Timestamp` (entry row) and `ExitTime` (exit row). `journal/events.jsonl`: `ts_utc`/`ts_server` on every event, plus `entry_time`/computed `hold_hours` on exit events. |
| 9 | Is exit reason stored? | Yes, `trades_log.csv` `ExitReason` column, values `TP`/`SL`/`MANUAL/OTHER`/`FRIDAY_CLOSE`/legacy `EOD_CLOSE`. **Per explicit project convention, `MANUAL/OTHER` does not mean discretionary manual intervention** — `_format_exit_deal()` (`agent_execution.py:445-453`) assigns it to any client-side close MT5 can't attribute to SL/TP, and in this bot that is the scheduled session-based/London-open strategy exit; `monitor_positions()` relabels the Friday 20:00 UTC forced close specifically to `FRIDAY_CLOSE` when known, but the generic scheduled exit still surfaces as `MANUAL/OTHER` in the raw data. The export script decodes this (see §3). |
| 10 | Is R stored? | Not directly. It's derived at read time from `PnL` (either file) divided by risk-at-entry: `risk_usd_intended` from a matched journal entry event (preferred, `source='journal'`), or a fallback approximation `abs(SLPips) × Lots × PIP_VALUE_USD` (`PIP_VALUE_USD = {'default': 10.0, 'JPY': 6.7, 'XAUUSD': 10.0}`, `source='fallback'`) when no journal match exists. Both the dashboard (`mcp/server.py`'s `_trade_r()`) and the new export script replicate this exact same logic. |
| 11 | Is spread stored? | Yes, but only at entry (and at rejection), not at exit. `market_context()` in `core/trade_journal.py` computes `spread_pips` live from `mt5.symbol_info_tick()` at the moment `log_entry()`/`log_signal()` fires, and is only available for trades that have a matching journal entry event. |
| 12 | Is ATR stored? | Yes, same caveat as spread — `atr14_h1_pips` from `market_context()`, entry-time only, journal-matched trades only. |
| 13 | Is strategy journal/reason stored? | Yes — `strategy_reason` is a free-text field passed into `log_signal()`/`log_entry()`, journal-matched trades only. |

**Additional fields not covered by the 13 questions but requested in the export schema:**
- **swap / commission**: **NOT AVAILABLE anywhere in the file-based system.** Neither `trades_log.csv`'s header nor any journal event schema includes these. `agent_execution.py`'s only references to "swap" are a Thursday-evening log-only warning message (`step_thursday_swap_warning`, `main_agent.py:954-962`) — never a persisted per-trade value. Raw MT5 deal objects returned by `mt5.history_deals_get()` do carry `.swap`/`.commission` attributes, but the code never reads or stores them. Obtaining these would require either (a) a new, explicit read of `deal.swap`/`deal.commission` added to `_format_exit_deal()`, or (b) a one-off direct MT5 history query on a machine with the target account's terminal.
- **strategy_version**: **NOT AVAILABLE for the general trade journal.** The only place a `strategy_version` string is tracked in this codebase is `src/amr_forward_tracker.py` (`STRATEGY_VERSION = 'phase22_model_B_buy_only@55e301e...'`), which is a separate, purpose-built forward-validation log for the AUDJPY AMR prospective tracker — it does not cover general trades or other strategies.

---

## 2. Field availability summary

| Field | Source | Coverage |
|---|---|---|
| trade_id, symbol, direction, entry/exit price, lots, SL, TP, profit, entry/exit time, raw exit reason | `trades_log.csv` | All CLOSED rows in whatever `data/` directory is pointed at |
| strategy (precise), signal_time, strategy_reason, spread, ATR, risk_percent, initial_risk (journal-sourced), holding_time | `journal/events.jsonl`, joined by Ticket | Only trades with a matching journal entry/exit event — i.e. only trades opened by a clone that had journaling deployed and running at the time |
| strategy (approximate fallback) | `Pair`+`Session` from `trades_log.csv` | All CLOSED rows, but only unambiguous if the pair+session combination is unique among concurrently active strategies |
| R | Derived: `profit / initial_risk` | Same coverage as `initial_risk` — 'journal' source when matched, 'fallback' approximation otherwise |
| swap, commission | Not persisted anywhere | NOT AVAILABLE from any file-based source on any machine; would need a raw MT5 deal-history query or a code change |
| strategy_version | Not persisted for general trades | NOT AVAILABLE except for the separate AMR forward-tracker log |

---

## 3. Read-only export mechanism built

**`scripts/export_5ers_trades.py`** (new file, read-only — every file it touches is opened `'r'` only; it never calls any MT5 trading function and never writes to any source file).

- Reads `trades_log.csv` (CLOSED rows) and `journal/events.jsonl` from a `--data-dir` you point it at, joins them by `Ticket`.
- Replicates the dashboard's exact `_trade_r()` R-calculation logic (journal-sourced risk preferred, `SLPips × Lots × PIP_VALUE_USD` fallback otherwise), tagging each row's `r_source` so journal-grade and approximated R are never silently blended.
- Decodes `MANUAL/OTHER` to `SCHEDULED_STRATEGY_EXIT` per the explicit project convention, while preserving the raw MT5 value in a separate `raw_exit_reason` column so nothing is lost.
- Classifies every trade attributed to `GBPJPY_ARB` or `XAUUSD_ARB` as `PRE_DEMOTION` / `POST_DEMOTION` relative to **2026-07-31** (the demotion date established in `reports/live_portfolio_validation_audit.md` / `PROJECT_REPORT.md` §4), by comparing each trade's entry time to that date. All other strategies get `N/A (not a demoted strategy)`.
- Requires an explicit `--account` label (no default) so an export can never be silently mislabeled as belonging to a particular account.
- Populates every field the current data genuinely can't answer with the literal string `NOT_AVAILABLE`, never a guess.

To run this against real 5ers data, execute it on a machine (or VPS session) whose `data/` directory belongs to the 5ers-bound clone, e.g.:
```
python scripts/export_5ers_trades.py --data-dir C:\forex-bot-5ers\data --account 5ERS-<login> --out reports/5ers_trade_export.csv
```

---

## 4. Local access verification (done directly, not assumed)

- `mt5.initialize()` on this machine succeeds and binds to **login=5052472770, server=MetaQuotes-Demo, balance=101366.23** — the DEMO account. This matches the same finding from the immediately-prior forensic phase (`project_5ers_drawdown_forensic` memory) — re-confirmed fresh for this task, not carried over.
- **No 5ers MT5 terminal exists on this machine**: `C:\MT5-5ers\` does not exist.
- **No 5ers bot clone exists on this machine**: `C:\forex-bot-5ers\` does not exist. Per the pending launch-runbook plan on file, the 5ers challenge clone (if launched) lives on the VPS, not on this local laptop.
- **This clone's local `data/journal/events.jsonl` does not exist at all** — the `data/journal/` directory is present but empty. This local checkout has never been run with journaling active.
- **This clone's local `data/trades_log.csv` exists but is stale dev/test data**: 19 CLOSED trades, `Timestamp` range **2026-06-04 to 2026-06-30**, symbols `GBPJPY`/`EURUSD` only (not the current 6-8 strategy live book: no CADJPY, no AMR pairs as currently configured, no GBPUSD Monday), and the file is untracked by git (gitignored local artifact, not synced from any server). This is local pre-live/dev-testing residue, not demo-account production data and **certainly not 5ers data**.

## 5. Export produced and what it actually contains

`reports/5ers_trade_export.csv` was generated by running the new script against **this machine's local `data/` directory only** (the only data this machine has), with `--account` explicitly set to `LOCAL-DEMO-5052472770-STALE-DEV-DATA` so it cannot be mistaken for 5ers or current-production data.

- **19 rows**, date range **2026-06-04 to 2026-06-30**.
- **0 journal-event matches** (`journal_entry_events: 0`, `journal_exit_events: 0`) — `events.jsonl` doesn't exist locally, so every row falls back to the `Pair`+`Session` strategy approximation, `NOT_AVAILABLE` for signal_time/spread/ATR/strategy_reason/risk_percent, and `fallback`-sourced R.
- **This export does NOT contain any 5ers trade data**, and does not represent the current live/demo 8-strategy book. It demonstrates the mechanism only.
- **Not truncated relative to what's available**: this file contains 100% of the CLOSED rows present in this machine's local `data/trades_log.csv` — 19 of 19. But "what's available on this machine" is a small, stale, unrepresentative slice; it is not the 5ers account's history.

## 6. Answer to the original blocker

**This does not solve the data-access blocker.** The dashboard/API architecture is now fully understood and a working, reusable export mechanism exists, but this local machine has zero access to the 5ers account, its terminal, or its clone's files — nothing has changed about that constraint. To actually export 5ers trade history, the same script (`scripts/export_5ers_trades.py`) needs to be run in an environment that has one of:
- The VPS's 5ers-bound clone's `data/` directory (if `C:\forex-bot-5ers\` has been launched there per the pending runbook), copied or run in place, pointed at with `--data-dir` and `--account 5ERS-<login>`.
- Direct MT5 terminal access to the 5ers account (a machine with `C:\MT5-5ers\` logged in), which would additionally allow pulling `swap`/`commission` directly from `mt5.history_deals_get()` deals — a capability the file-based export can never provide since those fields aren't persisted anywhere.

## 7. Is the data sufficient to rerun the prior forensic analysis?

**No.** The prior forensic protocol (`reports/5ers_current_drawdown_forensic_analysis.md`) needs real 5ers trade-level data for per-trade strategy/pair/direction attribution, exit reasons, holding times, and regime classification of the actual current losing period. This task's export contains none of that — it's mechanically ready (the join/R/demotion-classification logic is built and tested) but has nothing real to run on from this machine. The blocker is unchanged: **someone needs to run `scripts/export_5ers_trades.py --data-dir <5ers clone's data dir> --account 5ERS-<login>` from a location that actually has 5ers access**, after which the export will be complete and the forensic protocol can be rerun on real data.

---

*No strategy analysis performed. No interpretation of trade outcomes performed. No portfolio, account, or trading-code changes made. Per instruction, stopping here.*
