# Phase 51 — Live London-Open Exit & Execution Audit (Master Report)

**FORENSIC AUDIT ONLY. No production code, configuration, strategy, risk, parameter, or execution change made. No live position closed. No restart or deployment performed.**

---

## 1. Executive summary

The task's specified audit window (2026-08-14 to 2026-08-23) has **zero rows of trade-level evidence available on this machine** — the only usable evidence source, `reports/5ers_trade_export.csv`, ends 2026-08-13, and no journal, execution log, or live MT5 connection exists here (this machine's zero-5ers-access status was already established in [[project_5ers_data_source_audit]]). This is disclosed in the preregistration before any analysis, per the task's own missing-data rule, rather than silently substituted. The audit instead ran, honestly labeled as **BASELINE PERIOD** context, over the one window that does have data: 2026-07-20 to 2026-08-13 (72 export rows, 36 closed).

**Within that available window, the finding is unambiguous and positive: zero London-open (scheduled session) exit execution deviations were found.** Of 26 closed AMR trades where a scheduled session exit was expected, 10 hit their scheduled exit at the exact designed time and 16 were already closed by SL/TP before the scheduled exit time ever arrived — both are correct, by-design outcomes, not failures. CADJPY_ARB (4 closed trades) correctly has no scheduled exit by design. No GBPUSD_MONDAY trades closed in this window, so the Monday-exit mechanism is unverified either way.

**A genuine, unrelated, and significant finding surfaced during methodology validation** (documented as Amendment 1 in the preregistration): the server-time-converted "expected exit" hour computed from the checked-in source's own offset logic (04:00 UTC) was empirically contradicted by 9-for-9 second-precise observed exit timestamps at 07:00 UTC. The strategy source's own plain-English comments ("TIME EXIT at 07:00 UTC") turned out to match production behavior; this audit's mechanical derivation from the server-time-offset code did not. Whether this reflects a stale comment describing intended-but-superseded behavior, or a VPS/local code-version mismatch (a documented recurring pattern in this project, [[project_live_demo_june_findings]]: "VPS runs OLD code"), **cannot be resolved from the evidence available on this machine** and is reported as an open, disclosed uncertainty rather than resolved by assumption.

**The MANUAL/OTHER relabeling itself is now independently verified, not merely trusted.** The export tool's `decode_exit_reason()` blindly maps every raw `MANUAL/OTHER` label to `SCHEDULED_STRATEGY_EXIT` with no timestamp check (`scripts/export_5ers_trades.py:200-210`). This audit re-verified that relabel against actual exit timing for all 10 candidate trades and found it corroborated in every case — but the blind relabel itself remains a latent risk for any future genuinely-delayed or genuinely-manual close, which would receive the identical mislabel without a timing check like this one.

## 2. Phase 50 context

Phase 50 concluded that neither portfolio-stress research lead survived prospective validation, and recommended stopping portfolio-control research and continuing live observation. This phase does not revisit that question — it is a distinct, execution-layer forensic question triggered by the clarification that MANUAL/OTHER does not mean user intervention.

## 3. Research question

Are the six live strategies' scheduled session exits actually executing as designed, or are recent live losses partly created/amplified by execution-layer failures rather than strategy logic?

## 4. Preregistration

`reports/phase51_preregistration.md`, committed separately (`40a0fb3`) before any result. **Amendment 1** committed within the same file after an empirically-falsified methodological assumption was discovered (see §1, §22) — disclosed per the preregistration's own no-silent-amendment rule.

## 5. Data sources

Per preregistration Part 1: only `reports/5ers_trade_export.csv` (2026-07-20 to 2026-08-13) and this repo's checked-in `strategies/`/`pairs/`/`src/agents/` source were available. Journal, execution logs, MT5 deal history, and any evidence for the 08-14–08-23 window: **UNAVAILABLE**, not fabricated or inferred.

## 6. London-open definition

AMR: scheduled exit at 07:00 UTC (empirically confirmed, see Amendment 1 — not the 04:00 UTC value a literal reading of the server-time-offset code would produce). MON: scheduled exit at server 21:00 Monday, converted to 18:00 UTC via the documented DST offset — **unverified**, no observed Monday closes in the available window. ARB: no scheduled exit by design (confirmed by source inspection, zero exit-related lines in `asian_range_breakout.py`).

## 7. Expected-exit population

`reports/phase51_london_exit_expectations.csv` — 60 rows (all AMR/MON/ARB trades from the export); 26 closed AMR trades are `LONDON_EXIT_EXPECTED`; CADJPY_ARB trades are `LONDON_EXIT_NOT_EXPECTED`.

## 8. Event-chain reconstruction

`reports/phase51_event_chain.csv` — 300 rows (10 chain steps × 30 eligible closed trades). Only 4 of 10 chain steps (ENTRY, POSITION_OPEN, POSITION_CLOSED, TRADE_LOG_UPDATE) are directly OBSERVED from the available export; the middle of the chain (EXIT_SIGNAL_EVENT, EXECUTION_REQUEST, MT5_RESPONSE, JOURNAL_EXIT_EVENT) is UNAVAILABLE on this machine for every trade — meaning this audit can determine *whether* the position closed at the right time, but not *which internal step* would have failed had a deviation been found. Since zero deviations were found, this limitation did not block a conclusion this phase, but would block root-causing a *future* deviation without journal/execution-log access.

## 9. MT5 verification

`reports/phase51_mt5_verification.csv` — UNAVAILABLE, disclosed, no live MT5 connection on this machine, no calls attempted.

## 10. Journal verification

Not run — no journal file available on this machine (see §5).

## 11. Execution verification

`reports/phase51_execution_responses.csv` — UNAVAILABLE, disclosed.

## 12. Source audit

`reports/phase51_source_audit.csv` — 5 findings, most importantly: (a) `asian_hours_reversion.py`'s "NOT YET WIRED" docstring is stale — `step_asian_time_exit()` is actively wired in `main_agent.py` and firing in production (confirmed by export evidence); (b) the T_ASIAN_EXIT server-time-gating logic implies 04:00 UTC but observed behavior is 07:00 UTC (Amendment 1); (c) `decode_exit_reason()`'s blind MANUAL/OTHER relabel has no timestamp check built in.

## 13. Configuration audit

`reports/phase51_configuration_audit.csv` — documents the documented-vs-derived-vs-observed three-way comparison for AMR (mismatch, resolved empirically), MON (unresolved, no data), ARB (consistent, no exit expected or observed).

## 14. Trade-level findings

`reports/phase51_trade_level_audit.csv` — all 72 export rows, each individually classified. 36 CLOSED, 36 OPEN (marked `J. DATA_UNAVAILABLE`, correctly — an open trade's outcome isn't determined yet).

## 15. Deviation classifications

`reports/phase51_deviation_summary.csv`: **A. CORRECTLY_EXECUTED: 10. H. POSITION_ALREADY_CLOSED: 16.** No trade fell into B, C, D, E, F, G, or I.

## 16. Counterfactual P&L

`reports/phase51_pnl_counterfactual.csv` — 26 rows, `is_deviation=False` for all (since zero deviations exist in the available window). No INTENDED_EXIT_PRICE/PNL was computed for any trade, consistent with the frozen rule that counterfactuals are only computed for genuine deviations and only when an independent price source exists (none does here) — there being no deviations, this is moot, not a shortcut.

## 17. Strategy-level results

`reports/phase51_strategy_summary.csv`:

| Strategy | Closed | London-exit expected | Correctly executed | Already closed pre-exit | Deviations | Actual total R |
|---|---|---|---|---|---|---|
| AUDJPY_AMR | 9 | 9 | 3 | 6 | 0 | -3.12 |
| CADJPY_AMR | 6 | 6 | 4 | 2 | 0 | -2.04 |
| EURJPY_AMR | 9 | 9 | 2 | 7 | 0 | -1.00 |
| GBPJPY_AMR | 2 | 2 | 1 | 1 | 0 | +0.87 |
| CADJPY_ARB | 4 | 0 (N/A) | — | — | 0 | -0.84 |

Issue scope: **N/A — no execution deviations were found to scope.**

## 18. Baseline comparison

`reports/phase51_baseline_comparison.csv` — only one period is computable (the baseline). The task-specified current-live period cannot be compared against it; this is stated plainly rather than papered over with the baseline figure standing in.

## 19. Concurrency interaction

`reports/phase51_concurrency_execution.csv` — **not run.** The only field that would let a live-period concurrency cross-tab be built does not exist in the available export, and reconstructing it via the historical research ledger (`data/phase26_all_trades.csv`) would contaminate a live-period audit with historical research data, explicitly prohibited by the task's Part 6. Flagged as a legitimate Phase 52 candidate once a live-period source with position-count fields exists.

## 20. Volatility interaction

`reports/phase51_volatility_execution.csv` — not run, same limitation as §19.

## 21. Recent loss reconstruction

`reports/phase51_recent_loss_reconstruction.csv`. Baseline-period total actual R across the 5 non-ARB-excluded eligible strategies: **-6.13R** (AUDJPY_AMR/CADJPY_AMR/EURJPY_AMR/GBPJPY_AMR/CADJPY_ARB combined, closed trades only). Of that, -1.54R came from correctly-executed scheduled exits, -3.75R from SL/TP hits that closed the position before the scheduled exit could ever apply, -0.84R from CADJPY_ARB (no scheduled exit by design). **R attributable to London-exit execution deviations: 0.00R (0.0% of total).** This is the answer for the *only available* window; it cannot be extrapolated to the task's actual target window (08-14 to 08-23), which remains entirely unevidenced on this machine.

## 22. Source regression check

Per Phase 47's harness validation, this repo's checked-in strategy source remains internally consistent with itself (ARB has no exit logic, as before; AMR/MON both document a session-based time exit). The specific finding here is narrower than a full Phase-47-style re-run: **the AMR time-exit trigger, as literally implemented in the server-time-offset-aware `T_ASIAN_EXIT` gating code, computes to 04:00 UTC, but 9/9 observed production exits land at 07:00 UTC.** `SOURCE_CHANGED` cannot be asserted (no VPS code-version access), and `SOURCE_UNCHANGED` cannot be asserted either (the discrepancy is real and repeatable) — reported as **UNRESOLVED, evidence-backed discrepancy requiring VPS-side source inspection**, not guessed at.

## 23. Evidence matrix

`reports/phase51_evidence_matrix.csv` — levels 1-4 (MT5, execution log, journal, raw production trade log) all UNAVAILABLE; levels 5-6 (source code, derived export) AVAILABLE and used. No source disagreement was found among the levels actually reachable (source-comment language and export-observed timing *do* disagree with the server-offset-derived value, but that is a within-analysis discrepancy already reported in §22, not a disagreement between two independent evidence sources of comparable authority).

## 24. What is confirmed

Within the available window: all 26 AMR trades with an expected scheduled exit either executed it correctly (10) or had the position already closed by SL/TP first (16) — zero execution-layer deviations. CADJPY_ARB correctly never triggers a scheduled exit. The export's MANUAL/OTHER→SCHEDULED_STRATEGY_EXIT relabel is corroborated by timing for every trade it was applied to in this window.

## 25. What is probable

The 07:00 UTC vs 04:00 UTC discrepancy (§22) is very likely either a stale source comment describing an earlier, no-longer-accurate offset assumption, or a VPS/local desync of the kind already documented in this project's history — but which of the two cannot be established without VPS access.

## 26. What is unresolved

The entire task-specified audit window (08-14 to 08-23): completely unevidenced on this machine. The MON (Monday) scheduled-exit timing: unverified, zero observations. The root cause of the 07:00-vs-04:00 discrepancy: unresolved (§22). Concurrency/volatility interaction with execution reliability: not run (§19-20), no deviations existed to correlate against in any case.

## 27. Materiality

`reports/phase51_materiality.csv`: 26 expected exits, 10 correctly executed, 16 already-closed-pre-exit, **0 deviations of any kind (0.0% affected trade rate)**. This is the single most important number in this report: execution-layer London-exit failures are not a contributor to the baseline-period's -6.13R loss.

## 28. Safety assessment

No file under `strategies/`, `pairs/`, `src/agents/`, or `core/` was modified (`git diff --stat` empty for those paths). No MT5 call of any kind appears in any `src/phase51_*.py` file (verified by direct grep). No live position was closed. No restart or deployment occurred. `reports/5ers_trade_export.csv` remains unstaged. Full test suite: 46/46 passing.

## 29. Required engineering follow-up (documentation only — NOT performed in this phase)

1. Resolve the 07:00-vs-04:00 UTC discrepancy (§22) by inspecting the actual VPS-deployed source, not this local checkout.
2. Fix the stale "NOT YET WIRED" docstring in `asian_hours_reversion.py` (cosmetic — behavior is correct, comment is not).
3. Consider adding a timing-verified relabel (or at minimum a logged timestamp-delta) to `decode_exit_reason()` in `scripts/export_5ers_trades.py`, so a future genuinely-delayed or genuinely-manual close is not silently folded into `SCHEDULED_STRATEGY_EXIT` the same way a correctly-timed one is.
4. When the current-live-period data gap (08-14 to 08-23) is resolved (regenerate `reports/5ers_trade_export.csv` from the actual production host), re-run this exact audit over that period before drawing any conclusion about whether recent live losses have an execution-layer component.

None of these are fixed in this phase, per the explicit no-fix rule.

## 30. Final verdict

### Final classification (Part 32)

**F. INSUFFICIENT EVIDENCE** for the task-specified primary question (whether the 08-14 to 08-23 losses include an execution-layer component) — the only period with data shows **A. EXECUTION CORRECT** (zero deviations, all outcomes traceable to correct scheduled-exit execution or correct pre-emptive SL/TP), but that finding cannot be extended to the actual window the user asked about, which has zero evidence on this machine.

### Answers to the 30 required questions

1. How many trades should have received a scheduled exit? **26** (baseline window only; 0 known for the current-live window — data unavailable).
2. How many actually received it? **10** directly; **16** more were already closed by SL/TP before the scheduled time (not a failure to receive it — the position was gone first).
3. Delayed? **0.**
4. Rejected? **0.**
5. Misclassified? **0** (all 10 MANUAL/OTHER relabels independently timing-verified).
6. Genuinely unknown? **0** in the baseline window (36 OPEN trades await a future close, correctly marked pending, not "unknown").
7. Root cause of every confirmed deviation? **N/A — no deviations were confirmed.**
8. Strategy logic, execution logic, configuration, or logging? **N/A for the baseline window** (nothing to attribute); the *discrepancy* found (07:00 vs 04:00, §22) is a documentation/derivation issue, not a demonstrated execution failure.
9. Portfolio-wide or strategy-specific? **N/A — no issue found to scope.**
10. Which strategies affected? **None, in the available window.**
11. Actual R from correctly executed trades? **-1.54R** (scheduled-exit trades) + **-3.75R** (SL/TP-first trades, also correct execution) = **-5.29R** of the -6.13R total is fully execution-correct.
12. Actual R from London-exit deviations? **0.00R.**
13. Intended P&L at intended exits? **N/A — no deviations to counterfactual; see §16.**
14. Total P&L difference? **0.00R.**
15. % of recent losses attributable to the issue? **0.0%, for the only window with data.** Unknown for the actual current-live window.
16. Is the recent deviation rate abnormal vs baseline? **No baseline-vs-current comparison is possible** — only one period has data, and it IS the baseline by this report's own necessary redefinition.
17. Related to concurrency? **Not tested** — no deviations existed to correlate, and the live-period concurrency field is unavailable regardless (§19).
18. Related to volatility? **Not tested**, same reason (§20).
19. Did live source change since Phase 47? **Cannot be fully confirmed** — this repo's checkout is internally unchanged in structure, but the 07:00-vs-04:00 discrepancy (§22) raises the specific, disclosed possibility that the VPS-deployed code differs from this checkout in a way this audit cannot directly inspect.
20. Did live configuration change? **Not testable from available evidence.**
21. Is the issue already fixed in source but failing in deployment? **Possible but unconfirmed** — this is one of the two live explanations for §22's discrepancy, not established as fact.
22. Is this a known logging bug rather than an execution bug? **The MANUAL/OTHER label itself is a known, documented, and — for the 10 trades checked here — correctly-applied relabeling convention, not a bug.** No logging bug was found in the available window.
23. Does the evidence suggest the strategy itself is currently deteriorating? **Not addressed by this phase** — that is a strategy-performance question, out of this execution-forensic phase's scope, and unanswerable for the current-live window regardless (no data).
24. Does the evidence justify changing any strategy? **No.**
25. Does the evidence justify changing risk? **No.**
26. Does the evidence justify changing the scheduled-exit rule? **No.**
27. Does the evidence justify changing the execution system? **No — not for the available window; §22's discrepancy warrants investigation, not a change, since its cause is unconfirmed.**
28. What should be fixed, if anything? **Nothing based on this phase's evidence** — §29's four items are follow-up *investigations*, not confirmed fixes.
29. What should NOT be changed? **The AMR/MON scheduled-exit logic, the ARB no-exit design, the MANUAL/OTHER relabeling convention, and any risk/position-sizing parameter** — none of these show a defect in the available evidence.
30. Should Phase 52 investigate strategy performance or execution? **Neither, immediately — Phase 52 (or an ad hoc follow-up) should first re-acquire current-live-period trade data (regenerate the export from the actual production host covering 08-14 onward) so THIS SAME audit can be re-run over the period the user actually asked about. Only after that gap is closed does it become possible to say whether execution or strategy logic is the better next target.**

---

## Safety checklist confirmation

Preregistration committed (`40a0fb3`) before conclusions, amended once with full disclosure · no production code/YAML/parameter/risk/strategy/execution-logic modified · no live positions closed · no restart/deployment · MT5 records checked where available (none were — disclosed, not fabricated) · journal checked where available (none was) · execution logs checked where available (none were) · source checked (§12, §22) · configuration checked (§13) · MANUAL/OTHER never treated as user intervention anywhere in this report or its CSVs · intended P&L methodology frozen before results (§6 of preregistration) · no hindsight price selection (none computed at all, since no deviations existed) · counterfactuals labeled descriptive throughout · historical baseline and current-live period explicitly separated (§1, §18, §21) · raw production export not committed as part of this phase's changes · all findings evidence-graded (§23).

---

*No live trading change authorized. Zero execution-layer deviations found in the only available evidence window. The task's actual target window (2026-08-14 to 2026-08-23) remains entirely unevidenced on this machine — closing that data gap is the necessary next step before this question can be fully answered.*
