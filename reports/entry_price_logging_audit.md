# ENTRY PRICE LOGGING AUDIT — FINAL VERDICT

**Original bug:** `place_trade()` in `src/agents/agent_execution.py` logged the immediate `order_send()` API response's `result.price` field as the trade's entry price, and that field is intermittently `0.0` on market-execution/ECN-style brokers (confirmed on 5ers) even when the order filled correctly.

**Cause:** MT5 API inconsistency — `result.price` is not always reliably populated the instant `order_send()` returns; this is a known behavior class on prop-firm/ECN accounts, not a defect in this project's order-placement logic itself.

**Fixed:** YES

**Fix commit:** `0b64c02` ("Fix fill price capture: confirm via positions_get, not order_send's result.price")

**Fix deployed to 5ers:** Bracketed by trade evidence to between **2026-08-06 22:15 UTC** (last observed pre-fix trade) and **2026-08-09 22:00 UTC** (first observed post-fix trade) — consistent with the commit's timestamp (2026-08-07 19:09 UTC / 2026-08-08 00:39 +0530). **Exact VPS pull/restart timestamp: NOT AVAILABLE** (no deployment log accessible from this session — see §4).

**Current production logging:** CORRECT (verified against all 10 CLOSED trades from 2026-08-09 onward, including 2026-08-11, 08-12, 08-13)

**Trades affected:** 25 of 35 CLOSED trades (PRE-FIX, entry_price = 0.0)

**Actual fills recoverable:** 0 of 25 from any source accessible in this session (see §7) — theoretically recoverable via a direct MT5 broker deal-history query on the 5ers terminal, which this session cannot perform

**Historical execution analysis valid:** YES, for the analysis actually performed in `reports/5ers_current_portfolio_forensic_analysis.md` §8 — see §11 below for why

**Forensic report needs rerun:** NO — the prior report's own execution/spread section did not actually depend on `entry_price`; only its prose caveat overstated the impact (see §10/§11)

---

## 1. Find the original bug

Traced via `git log --follow -- src/agents/agent_execution.py` and `git log -p` on that file, searching for the `actual_entry = result.price` assignment.

**The bug was present from the project's first live-trading commit.** `git log --diff-filter=A -- src/agents/agent_execution.py` shows the file was created in `5fd0f2e` ("Add complete multi-agent live trading system", **2026-05-19 19:56:41 +0530**), and `git log -p --follow` confirms the line

```python
actual_entry = result.price
```

was present unchanged in `place_trade()` from that first commit through every subsequent commit touching the file, until `0b64c02` replaced it. **`git log --oneline 5fd0f2e..0b64c02~1 -- src/agents/agent_execution.py`** lists every intervening commit that touched this file (12 commits, e.g. `e15de7b` trade-monitoring fix, `66adc0b` deal-search-window fix, `c1d74ca` TP-headroom check) — **none of them touched this line**. The bug sat unnoticed in the code for ~2.5 months (2026-05-19 → 2026-08-07).

**The exact code path that caused it** (pre-fix, `src/agents/agent_execution.py`, `place_trade()`):
```python
result = mt5.order_send(request)
...
ticket       = result.order
actual_entry = result.price          # <-- the bug: trusted blindly
...
_write_trade_log({..., 'EntryPrice': actual_entry, ...})   # -> trades_log.csv
```
and downstream, `main_agent.py` passed that same `result['entry_price']` (= `actual_entry`) into `tj.log_entry(fill_price=result['entry_price'], ...)` — so **both** `trades_log.csv`'s `EntryPrice` and `journal/events.jsonl`'s `fill_price` were sourced from the identical corrupted value.

**Cause established directly from code, not inferred:** `mt5.order_send()`'s response object's `.price` field — an immediate, synchronous API response — was used as-is with no confirmation step, no null/zero check, and no retry. This is confirmed as an MT5 API retrieval/response-handling issue, not order execution, position lookup, or logging-timing issue per se (see §3).

---

## 2. Identify the fix

**Commit:** `0b64c02c431cdb59dbe2ff67ba4ebfd726a8dc7d`
**Date:** 2026-08-08 00:39:32 +0530 (author timezone) = **2026-08-07 19:09:32 UTC**
**Author:** theimperfectalgorithm
**Files changed:** `src/agents/agent_execution.py` (1 file, +29/-1 lines)
**Function(s) changed:** added new `_confirm_fill_price()`; modified `place_trade()`'s single line `actual_entry = result.price`

**Old logic:**
```python
ticket       = result.order
actual_entry = result.price
```

**New logic:**
```python
ticket       = result.order
actual_entry = _confirm_fill_price(ticket, result.price or entry_price, log)
```
where `_confirm_fill_price()` is:
```python
def _confirm_fill_price(ticket, fallback_price, log):
    for _ in range(3):
        positions = mt5.positions_get(ticket=ticket)
        if positions and positions[0].price_open:
            return positions[0].price_open      # broker-confirmed fill
        time.sleep(0.3)
    if fallback_price:
        log.warning(f"ticket {ticket}: could not confirm fill via positions_get() -- using fallback price {fallback_price}")
        return fallback_price                    # result.price, else the pre-order live-price snapshot
    log.warning(f"ticket {ticket}: could not confirm fill price at all -- logging 0.0")
    return 0.0                                    # now loud, never silent
```

**This is the "position lookup instead of order_send response" pattern** the audit's example list anticipated: the fix changed the entry-price source from `order_send()`'s immediate synchronous response to a subsequent `positions_get(ticket=...)` lookup of the just-opened position's `price_open` field — which reflects the broker's own confirmed fill — with up to 3 retries (0.3s apart) for the rare case it isn't visible in the first instants after the order returns. It is **not** an MT5 deal-history-based fix (that data source is used elsewhere in this codebase, for exit prices — see §8 — but not for this particular fix).

---

## 3. Why the old value was 0.0 — execution vs. logging

**Execution was NOT affected. Only the logged/recorded entry price was wrong.** Established directly from code, not inferred:

- `place_trade()` computes `sl_price` and `tp_price` from `anchor` (either the Asian-range high/low, or a live-price snapshot for use-live-anchor strategies) — this happens **before** `order_send()` is even called (lines ~228–240), and therefore entirely independent of the later, buggy `result.price` read. The actual stop-loss and take-profit sent to the broker in the order request were always correct.
- Exit price and PnL come from a completely separate code path: `_get_closed_deal()` / `_format_exit_deal()` query `mt5.history_deals_get()` for the closing deal directly from MT5's own broker-side deal history at close time — this path never reads `result.price` or the logged `entry_price` at all (verified by reading these functions; they take only `ticket` and a search-window as input).
- The fix commit's own message states this explicitly and the code confirms it: *"Actual order execution was never affected: SL/TP are computed from the Asian-range anchor before this value is even read, and PnL/ExitPrice come from MT5's own deal history at close, independent of it."*

**Was the order actually filled correctly but logged incorrectly? YES — this is a pure logging/recording defect, not an execution defect.** The order was placed, filled, and managed correctly by MT5 in every case; only the human/analytical record of *what price it filled at* was lost for the affected trades. There is no evidence anywhere in the code, commit history, or project documentation of execution itself being affected by this bug.

---

## 4. Exact fix deployment date

**Git commit → deployment/pull → bot restart → first trade using corrected logger: only partially reconstructible.**

- **Git commit timestamp:** 2026-08-07 19:09:32 UTC (established above, §2).
- **VPS deployment/pull timestamp:** **NOT AVAILABLE.** No deployment log, CI record, or VPS pull-history artifact is accessible from this session to establish exactly when `git pull` was run on the 5ers VPS clone or when the bot process was restarted afterward.
- **Bot restart timestamp:** **NOT AVAILABLE**, same reason.
- **First trade using the corrected logger (empirical, from production data):** the production export brackets the transition precisely — the last CLOSED trade with `entry_price = 0.0` opened at **2026-08-06 22:15:05 UTC** (EURJPY AMR), and the first CLOSED trade with a valid, non-zero `entry_price` opened at **2026-08-09 22:00:14 UTC** (GBPUSD Monday). No trades occur in the export between 08-06 22:15 and 08-09 22:00 (a quiet weekend window for this strategy mix), so the transition cannot be pinned closer than that 3-day bracket from trade data alone.
- **Consistency check:** the git commit timestamp (2026-08-07 19:09 UTC) falls squarely inside this empirical bracket, which is consistent with — though does not by itself prove — same-day-or-next-day deployment. This is circumstantial, trade-data-derived evidence, not a confirmed deployment log.

---

## 5. Audit current production logging

**Verified against the 10 most recent CLOSED trades in the production export (2026-08-09 through 2026-08-13), including all three explicitly requested dates (08-11, 08-12, 08-13):**

| trade_id | strategy | entry_time (UTC) | entry_price |
|---|---|---|---|
| 587348562 | AUDJPY_AMR | 2026-08-11 00:00:05 | 112.30000 |
| 587348576 | EURJPY_AMR | 2026-08-11 00:00:05 | 183.77200 |
| 587366610 | GBPJPY_AMR | 2026-08-11 00:45:05 | 214.91300 |
| 587497789 | CADJPY_ARB | 2026-08-11 06:00:06 | 114.29800 |
| 588015355 | EURJPY_AMR | 2026-08-12 02:45:06 | 183.93700 |
| 588589244 | CADJPY_AMR | 2026-08-12 23:30:05 | 114.30600 |
| 588619758 | EURJPY_AMR | 2026-08-13 00:45:05 | 183.64100 |

**All 7 trades from 08-11/08-12/08-13 (and all 10 since 08-09) have realistic, non-zero, correctly-scaled entry prices for their respective pairs.** `entry_price` is no longer 0.0 in current production data.

**Other fields, checked the same way:** `stop_loss`/`take_profit`, `profit`, `R`, `exit_price`, `exit_reason` are populated and sane for these same trades (all were already unaffected by this bug per §3, and remain so). `spread` and `ATR` are populated for these trades too — both were **always** independent of `entry_price` (they come from `market_context()`'s live tick/rate reads at journal-logging time, not from the order-placement price — see §8). **`signal_price` and `slippage` are NOT present as columns in the current export schema at all** (the export tool captures `signal_time`, not `signal_price`, and does not export a slippage field) — this is a schema gap, not a data-availability question this audit can resolve without re-running the export with an expanded column list.

---

## 6. Which historical trades were affected

Full per-trade table: `reports/entry_price_logging_audit.csv` (all 35 unique CLOSED trades from `reports/5ers_trade_export.csv`, each with trade_id, entry_time, entry_price, strategy, status, and PRE-FIX/POST-FIX classification).

| Classification | Count | Entry-time range |
|---|---|---|
| **PRE-FIX** (entry_price = 0.0) | **25** | 2026-07-20 21:15 → 2026-08-06 22:15 |
| **POST-FIX** (entry_price valid) | **10** | 2026-08-09 22:00 → 2026-08-13 00:45 |
| UNKNOWN | 0 | — |

**Not the entire 35-trade sample was affected** — 10 of 35 (28.6%) are POST-FIX and fully trustworthy for entry-price-dependent analysis. The 25 PRE-FIX trades span every strategy present in the export (AUDJPY/CADJPY/EURJPY/GBPJPY AMR, CADJPY/GBPJPY ARB, GBPUSD Monday) — the bug was universal across strategies and pairs, exactly as expected for a shared order-placement code path, not strategy-specific.

---

## 7. Can historical spread/execution analysis be recovered?

**For the 25 PRE-FIX trades, entry_price is not recoverable from any source accessible in this session:**

- **`trades_log.csv`'s `EntryPrice`**: 0.0 — the corrupted value itself.
- **`journal/events.jsonl`'s `fill_price`**: also 0.0 for the same trades — confirmed directly from the fix commit's own message ("Root cause traced to raw journal data — fill_price was 0.0 for the affected trades... proving the real fill price was lost before either log was written"). Both local application-level sources are corrupted identically; neither can serve as a fallback for the other.
- **Signal price**: the underlying journal event schema does capture a `signal_price` field (`log_signal()`/`log_entry()` in `core/trade_journal.py` both take a `signal_price` argument, read from a live price snapshot *before* order placement — unaffected by this bug in principle). **However, this session has no access to the raw `journal/events.jsonl` file** (no 5ers MT5/VPS access, consistent with every prior phase of this project), and **the production CSV export schema does not include a `signal_price` column** (only `signal_time`). So while signal_price plausibly exists somewhere in the system, it is **not accessible from any file this audit can currently read.**
- **MT5 deal history (`DEAL_ENTRY_IN` deals)**: MT5 stores its own broker-side record of the entry deal, independent of this bot's application-level logging — in principle this is the single most authoritative source and was never touched by the bug (the bug was in how the *bot* read back the price, not in what the *broker* recorded). **This has not been queried and cannot be from this session** — it would require direct MT5 terminal access to the 5ers account, which is not available here.

**Per-trade classification for all 25 PRE-FIX trades: C. Neither actual fill price nor signal price is available from any source this audit can currently reach.** (Full detail: `reports/entry_price_logging_audit.csv`, columns `fill_price_recoverable_locally` and `fill_price_recoverable_via_mt5_deal_history`.) **Category A (recoverable via direct MT5 deal history) remains theoretically possible but UNTESTED — this audit does not claim it is impossible, only that it was not attempted and cannot be attempted from this session.**

---

## 8. Current logging path — signal → order → fill → record

Traced directly from `src/agents/agent_execution.py::place_trade()` and `src/agents/main_agent.py`'s call site, plus `core/trade_journal.py::log_entry()`/`market_context()`:

| Value | Current source | Affected by the bug? |
|---|---|---|
| `signal_price` | Live price snapshot taken by the calling strategy code before `place_trade()` is invoked | No — always independent |
| `requested_price` (`entry_price` local var inside `place_trade()`) | `_get_live_price(symbol, signal)` — a fresh live-tick read at order-construction time, used to build the `order_send()` request and to anchor SL/TP for live-anchor strategies | No |
| `actual_fill_price` (`EntryPrice` in `trades_log.csv`, `fill_price` in the journal) | **Post-fix:** `_confirm_fill_price()` → `mt5.positions_get(ticket=...).price_open` (broker-confirmed), falling back to `result.price` then the live snapshot only if unconfirmed after 3 retries. **Pre-fix:** `result.price` directly, unconfirmed. | **Yes — this was the corrupted value** |
| `spread` (`spread_pips` in the journal) | `market_context()` → live `mt5.symbol_info_tick()` bid/ask read **at journal-logging time**, computed independently of `entry_price` | No — never affected |
| `slippage` (`slippage_pips` in the journal, not currently exported to CSV) | `core/trade_journal.py::log_entry()`: `(fill_price − signal_price) / pip`, i.e. **directly derived from the buggy `fill_price`** | **Yes — indirectly corrupted for the same 25 trades**, since it's a function of the same bad value (though not part of the current CSV export, so this didn't affect the prior forensic report, which never had a slippage column to begin with) |
| `ATR` (`atr14_h1_pips` in the journal) | `market_context()` → live `mt5.copy_rates_from_pos()` H1 bars, independent of `entry_price` | No — never affected |
| `SL` / `TP` | Computed from the Asian-range anchor / live-price snapshot **before** `order_send()` is even called | No — never affected |
| `PnL` / `ExitPrice` | `_get_closed_deal()` → `mt5.history_deals_get()`, MT5's own broker-side deal history at close, independent of `entry_price` entirely | No — never affected |

**Answer to the central question: the current logging path captures the actual, broker-confirmed execution price (`positions_get().price_open`), not merely the intended/requested price.** The intended/requested price (`_get_live_price()`'s snapshot) is now only used as a last-resort fallback if the broker-confirmed lookup fails after 3 retries — and even then, that fallback event is logged as a loud warning, never silently.

---

## 9. Regression protection

**NO REGRESSION TEST FOUND.**

Searched `tests/`, `src/`, `strategies/`, `scripts/`, and the whole repository for any reference to `_confirm_fill_price`, `entry_price`, `fill_price`, or a "fake MT5" test harness. The only test file in the repository is `tests/test_alignment_safety.py` (unrelated — covers the NZDJPY cross-symbol alignment bug from a different phase of this project). The fix commit's own message describes verification ("Verified against a fake-MT5 harness: immediate confirmation... delayed confirmation... broker never confirms... the original bug pattern") but **that harness was not committed to the repository** — `grep -rl "fake.mt5\|FakeMT5\|fake_mt5"` across all Python files returns nothing outside `agent_execution.py` itself. The verification described in the commit message appears to have been run once, ad hoc, and not preserved as a repeatable regression test.

**Consequence:** nothing in the current codebase would catch a regression to the old `result.price`-only behavior, or a new variant of the same class of bug (e.g. `positions_get()` itself returning stale/zero data under a different broker condition), other than manual dashboard inspection (the same slippage-card mechanism that caught it the first time). No test was created in this task, per instructions.

---

## 10. Reassess the forensic report's "25/35" statement

The prior report (`reports/5ers_current_portfolio_forensic_analysis.md`, §0) stated: *"`entry_price` reads `0.00000` for 25 of 35 (71%) CLOSED trades... this is why the spread/stop-distance analysis below uses an implied SL distance... instead of `entry_price − stop_loss`."* And in its final verdict text: *"entry_price unusable for 25/35 trades due to an already-known, already-fixed logging bug."*

**Verdict: B. Technically correct but misleading.**

- **Correct:** the 25/35 count is exactly right (confirmed independently in §6 above), and it is genuinely true that `entry_price` itself cannot be used for those 25 trades.
- **Misleading:** the phrasing implies the bug materially limited *the execution/cost analysis the report actually performed*. It did not. The report's §8 spread-forensics table used two fields, **neither of which depends on `entry_price`**: (a) `spread` — captured live via `market_context()`'s tick read at journal-logging time (§8 above confirms this is always independent of the fill-price bug), and (b) an *implied* SL distance computed as `initial_risk / (lots × pip_value)` — deliberately engineered in `src/phase27_5ers_current_portfolio_forensic.py` specifically to avoid needing `entry_price`. The report's own code comment already states this design choice explicitly. So the actual spread/SL-ratio bucket table in §8 of the prior report is **not built on corrupted data** — the corrupted field (`entry_price`) was correctly avoided, not silently used.
- **Were 25 trades genuinely executed without a known entry price? Yes** (per §3, execution itself was fine — only the *record* of the entry price is unknown).
- **Can the actual fill prices be recovered from another source? Not from anything accessible in this session** (§7) — theoretically possible via direct MT5 deal history, untested.

---

## 11. Does this materially invalidate the previous execution analysis?

**PARTIALLY — more precisely: NO for what was actually computed, but the report's own caveat oversold the limitation, and one execution-quality question genuinely remains unanswerable.**

**What remains valid, unchanged:**
- §8's spread-over-implied-SL-distance bucket table (the actual "execution quality" finding: most trades sit under a 10% spread-to-stop ratio, no bucket shows an extreme cost signature) — built entirely from fields independent of `entry_price`, per §10 above. **No rerun needed.**
- All of §2–§7, §9–§17 of the prior report (account performance, strategy attribution, directional analysis, exit-reason analysis, JPY concentration, drawdown attribution, Monte Carlo, regime analysis) — none of these sections used `entry_price` as an input at any point (verified by re-reading `src/phase27_5ers_current_portfolio_forensic.py`: `entry_price` is only referenced in the `entry_price_valid` flag and the now-avoided `sl_pips_est` calculation that was explicitly replaced). **No rerun needed.**

**What genuinely cannot be answered, and was correctly never claimed:**
- True slippage (fill price vs. signal price) for the 25 PRE-FIX trades — this was never computed or exported in the first place (§8 above: the export schema has no slippage column), so there is nothing to walk back; it simply remains a standing "NOT AVAILABLE" for those 25 trades specifically, same as it always was.
- A literal entry-price-anchored stop distance (`entry_price − stop_loss`) for the 25 PRE-FIX trades — genuinely unusable, but the prior report never used this metric (it used the implied version instead), so this doesn't change any stated conclusion.

**Recommended correction to the prior report's wording** (documentation only, not a rerun): the phrase *"entry_price unusable for 25/35 trades... could not perform a complete historical execution/spread analysis"* should be understood as *"entry_price itself is unusable for 25/35 trades, but the spread/cost analysis performed did not require it and stands as computed; only a true slippage analysis remains unavailable, and was never attempted."*

---

## 12. What should we do next?

- **No rerun of the forensic report is required** — its conclusions stand as computed, per §11.
- **Optional, not urgent:** if a true slippage analysis is ever wanted for the 25 PRE-FIX trades, the only path is a direct MT5 broker deal-history query (`DEAL_ENTRY_IN`) on the 5ers terminal — this is a new data-access task requiring 5ers MT5 access this project has not yet obtained from any session, not a rerun of existing work.
- **Optional, low-cost:** the export schema (`scripts/export_5ers_trades.py`) could be extended to include `signal_price` (already captured in the underlying journal, just not exported) on a future run — this would at least allow a signal-price-anchored spread check going forward, though it would not recover the historical PRE-FIX trades' true fill prices.
- **Regression risk noted but not acted on:** §9 found no test protects `_confirm_fill_price()` from regressing. Per this task's explicit instructions, no test was created — flagged for a future task if the user wants it addressed.
- **PROJECT_REPORT.md's own backlog already anticipated this exact question**: *"Consider historical-data backfill/correction for the fill-price bug... if it turns out to matter for a specific piece of analysis — opt-in only, not automatic."* This audit's conclusion is that it does **not** turn out to matter for the specific analysis already performed (§11) — so no backfill action is triggered by this audit.

---

*Diagnostic only. No code changed, no strategy modified, no configuration touched, no test created, no existing report overwritten. Reproducible: `reports/entry_price_logging_audit.csv` was generated by a one-off script (not committed — trivial re-derivation from `reports/5ers_trade_export.csv`, already excluded from git per policy).*
