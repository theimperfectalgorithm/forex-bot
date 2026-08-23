# Phase 51 Preregistration — Live London-Open (Scheduled Session) Exit & Execution Audit

Committed BEFORE any forensic conclusion is drawn. This is an audit; no live change is authorized (see Part 3 of the task spec, reproduced in the master report's safety section).

## 1. Data-availability finding that shapes this preregistration (established before any hypothesis was tested)

Before freezing methodology, the actual available production evidence sources on this machine were enumerated:

| Requested source | Status |
|---|---|
| `reports/5ers_trade_export.csv` (validated flat export, built [[project_5ers_data_source_audit]]) | **AVAILABLE**, but covers only entry dates 2026-07-20 through 2026-08-13. |
| `journal/events.jsonl` (production journal) | **UNAVAILABLE on this machine** — lives at `C:\forex-bot-5ers\data\journal\events.jsonl` on the production host, confirmed not present here (`C:\forex-bot-5ers` does not exist on this machine, consistent with the standing finding in [[project_5ers_data_source_audit]] that this machine has zero direct 5ers/production access). |
| `data/trades_log.csv` (this repo's own copy, under `./data/`) | Present but **STALE / NOT PRODUCTION** — a pre-existing local research artifact, last entries 2026-06-04, 44 lines total, unrelated to the audit window. |
| `data/logs/trading.log` | Present but **STALE** — file modification date 2026-07-05, predates the audit window entirely. |
| MT5 deal/order history (live query) | **UNAVAILABLE** — this machine has no live MT5/5ers connection (standing fact, [[project_5ers_data_source_audit]]); no MT5 calls are made in this phase, consistent with the audit-only, no-production-touch safety rule. |
| Production execution/service logs, scheduler logs | **UNAVAILABLE** — not present on this machine. |
| Live strategy source (`strategies/*.py`) and pair YAML (`pairs/*.yaml`) | **AVAILABLE** — this repo's checked-in copy, previously validated source-faithful in Phase 47. |

**Consequence:** the primary audit window specified in the task (2026-08-14 through 2026-08-23) has **ZERO rows of production trade data available on this machine** — the only trade-level evidence source (`reports/5ers_trade_export.csv`) ends 2026-08-13, one day before the window opens, and no other source fills the gap. This is disclosed here, before any result, rather than silently substituting a different window or fabricating rows.

**Amended scope (methodological necessity, disclosed, not a results-driven change):** the audit is run over the full available trade-level window, **2026-07-20 through 2026-08-13** (72 trade rows, 6 live strategies), which becomes the sole analysis population. Per the task's own Part 5 rule ("if a source is unavailable, mark it UNAVAILABLE, do not infer it silently") and Part 6 ("include the immediately preceding period required to establish baseline"), this available window is treated entirely as **BASELINE PERIOD** context — it predates the task's specified CURRENT LIVE PERIOD. **CURRENT LIVE PERIOD (2026-08-14 to 2026-08-23): DATA_UNAVAILABLE for every deliverable**, reported as such rather than omitted.

## 2. Eligible strategies

Exactly the six live strategies named in the task: AUDJPY_AMR, CADJPY_AMR, EURJPY_AMR, GBPJPY_AMR, CADJPY_ARB, GBPUSD_MON. No substitutions, no additions.

## 3. Scheduled session-exit definition (from live source, not invented)

Read directly from `src/agents/main_agent.py` and `strategies/asian_hours_reversion.py` / `strategies/monday_drift.py`:

- **AMR books** (AUDJPY_AMR, CADJPY_AMR, EURJPY_AMR, GBPJPY_AMR): scheduled time-exit fires when `srv >= T_ASIAN_EXIT` (`main_agent.py:141`, `T_ASIAN_EXIT = 07:00`), where `srv` is **MT5 server minutes**, not UTC — confirmed by the file-level comment at `main_agent.py:120-128` ("all SESSION steps below are gated on server minutes"). The strategy source's own docstrings (`asian_hours_reversion.py:24-28`) and the orchestrator function's own docstring (`main_agent.py:587`) both describe this as "07:00 UTC," which is **stale/imprecise documentation** — the actual gate is server-time, verified from the constant definitions and surrounding comments. **DOCUMENTED: this is a documentation-language mismatch, not (on the evidence read) a functional bug** — flagged for engineering follow-up as a comment-accuracy fix only, per Part 27's regression-check instruction, not investigated further as a behavioral defect in this phase.
- **MON book** (GBPUSD_MON): scheduled time-exit fires when `srv >= T_MONDAY_EXIT` (`T_MONDAY_EXIT = 21:00` server minutes, Monday only). Same stale-comment pattern in `monday_drift.py:16,63`.
- **ARB book** (CADJPY_ARB): `strategies/asian_range_breakout.py` contains no time-exit logic of any kind (confirmed by source grep — zero exit-related lines). **CADJPY_ARB trades are therefore classified `LONDON_EXIT_NOT_EXPECTED` by design; ARB exits only via SL/TP (or Friday force-close).**
- **Server UTC offset**: `agent_strategy.server_utc_offset_hours()` returns UTC+3 during US DST, UTC+2 otherwise (`agent_strategy.py:103-128`). The full audit window (2026-07-20 to 2026-08-13) falls within US DST, so **server 07:00 = 04:00 UTC** and **server 21:00 Monday = 18:00 UTC Monday**, for the entire analysis population. This conversion is applied uniformly; if any trade's entry/exit straddled a DST transition it would be flagged, but none does in this window.
- No grace period is documented in source beyond the per-cycle retry (`step_asian_time_exit` retries every 15-minute cycle until every close succeeds, deferring only for an active news blackout). A **30-minute tolerance band** around the computed expected-exit UTC timestamp is frozen as the CORRECTLY_EXECUTED threshold (covers one to two 15-minute polling cycles plus a possible single blackout deferral); beyond 30 minutes but same calendar day is EXIT_DELAYED; no matching exit at all within 24h of the expected time is EXIT_SIGNAL_MISSING/EXIT_REQUEST_MISSING (undistinguishable from the export alone, see Part 4).

## 4. Evidence hierarchy (frozen, per task Part 12) as actually achievable here

Given Part 1's findings, only levels 5 and 6 of the task's 6-level hierarchy are reachable on this machine:
1. MT5 deal/order history — UNAVAILABLE
2. Production execution response/log — UNAVAILABLE
3. Production journal event — UNAVAILABLE
4. Production trade log (raw) — UNAVAILABLE (only a stale local copy exists, not used)
5. Live source code/configuration — AVAILABLE, used
6. Secondary derived export (`reports/5ers_trade_export.csv`) — AVAILABLE, used, and is the *only* trade-level evidence source in this phase

Because only level 6 evidence exists, **no finding in this phase can distinguish EXIT_SIGNAL_MISSING (C) from EXIT_REQUEST_MISSING (D) from EXECUTION_REJECTED (E) from EXECUTION_FAILED_UNKNOWN_REASON (F)** — those four categories require journal/execution-log/MT5-level evidence not present here. Any trade that would otherwise qualify for one of C/D/E/F is instead classified **J. DATA_UNAVAILABLE** with the specific reason recorded, rather than guessed among C-F.

## 5. Classification methodology (frozen)

For every CLOSED trade of an eligible strategy in the analysis window:
1. Determine `LONDON_EXIT_EXPECTED` (AMR: always; MON: always, since GBPUSD_MON only trades Mondays by construction; ARB: never → `LONDON_EXIT_NOT_EXPECTED`).
2. For `LONDON_EXIT_EXPECTED` trades, compute `expected_exit_utc` from the entry date (AMR: same calendar day 04:00 UTC; MON: the Monday of entry at 18:00 UTC).
3. Compare `expected_exit_utc` to the export's `exit_time`.
4. Compare `raw_exit_reason` (export's undecoded MT5 label) to `exit_reason` (export's `decode_exit_reason()` output) to check whether the blanket `MANUAL/OTHER -> SCHEDULED_STRATEGY_EXIT` relabeling (`scripts/export_5ers_trades.py:200-210`) is corroborated by timing, or merely assumed. This relabeling is unconditional in the export script — it does not check timestamp proximity — so it is treated here as an **unverified label**, re-verified independently by this audit's own timestamp comparison, not taken at face value.
5. Assign exactly one classification (A–J, task Part 11), using the 30-minute tolerance band from Part 3.

## 6. Intended-vs-actual P&L / counterfactual methodology (frozen, per Part 18-19)

No independent tick/bar price source is available on this machine for the audit window (no live MT5 connection; using Phase-26/historical CSVs to backfill live-period prices would violate Part 19's "no hindsight-selected/reconstructed price" spirit for a live forensic window). Therefore: **counterfactual P&L is computed ONLY for trades classified as a genuine timing deviation (EXIT_DELAYED or worse) AND only where the export itself already contains both an actual exit price and a plausible reference price at the intended time (none, on inspection — see Part 18 of the master report).** Where no independent price source exists, `INTENDED_EXIT_PRICE`/`INTENDED_EXIT_PNL`/`PNL_DIFFERENCE` are marked `NOT_AVAILABLE`, never estimated. This is frozen before results are examined.

## 7. Missing-data handling

Any trade lacking a required field (ticket, entry_time, exit_time, exit_reason) is retained in the trade-level CSV with the missing field marked `NOT_AVAILABLE` and is excluded from any aggregate statistic that field feeds, never dropped silently and never imputed.

## 8. Conclusion categories

Verbatim from task Part 32 (A–F). Verbatim event classifications from task Part 11 (A–J).

## 9. Production-safety rules

Verbatim from task Part 3 / Part 35: no code, YAML, parameter, risk, position, SL/TP, exit-logic, or execution-logic changes; no restarts, deployments, or manual position closes; no logging changes in production. This phase makes zero writes to any file under `strategies/`, `pairs/`, `src/agents/`, or `core/`, and issues no MT5 calls.

## 10. No post-result methodological changes

If a genuine methodological error is found after this document is committed, work stops and a disclosed, separately-committed amendment is required before continuing — consistent with every prior phase in this project.
