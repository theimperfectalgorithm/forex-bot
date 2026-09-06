# Task018D safety fix and validation

Classification: **SAFE TO COMMIT BUT NOT SAFE TO DEPLOY LIVE**.

No staging, commit, production restart/stop, package installation, live MT5 call,
broker order, or production/configuration write was performed. The existing
Task018B patch remains unstaged. The running process has not been updated.

## Calendar evidence contract

A validated, fresh matching high-impact event establishes BLACKOUT for the
relevant currency within the inclusive configured window. CLEAR requires
explicit trustworthy evidence of complete coverage of the entire requested
interval, all relevant currencies/impacts, freshness/revisions, and absence of
truncation or unresolved records. Anything less is UNKNOWN. Dates bracketing
the window, a fresh cache, valid JSON, and a multi-day response prove none of
those completeness properties.

The [official Forex Factory calendar](https://www.forexfactory.com/calendar)
links its weekly JSON export and says event times are approximate and subject
to change. Neither that page nor the inspected
[official notices](https://www.forexfactory.com/notices) established a complete
coverage guarantee. The browser could not retrieve the linked versioned JSON
endpoint; this is a limitation of this investigation, not evidence that the
feed is complete or permanently unavailable. Repository calendar/cache code
also has no independently supported coverage contract or trusted bridge.
The market agent's separate calendar calls do not supply one: the
[official MT5 Python interface](https://www.mql5.com/en/docs/python_metatrader5)
does not document those MQL calendar methods as Python exports. No terminal
was queried to investigate this.

Therefore the current provider **cannot legitimately establish CLEAR** under
this contract. `_proves_coverage` returns False for every production snapshot;
cache fields claiming completeness are ignored. Its test-only monkeypatch is
an explicitly synthetic oracle for timing and existing order-path tests, not
provider evidence or a configurable production bypass. A future adapter needs
a separately reviewed evidence contract and implementation.

With the inspected effective news policy (enabled, fail-closed, five minutes),
deploying this code in LIVE would intentionally block every new entry until a
trusted source/bridge is supplied. Known news returns BLACKOUT; absence of a
known matching event returns UNKNOWN. The existing explicit filter-disabled
and fail-open policies remain labeled UNKNOWN permissions, as Task018B
requires; they are not CLEAR and were not enabled to restore trading. The
tracked fail-open default and instance override were not edited.

## Fixes and files

| File | Final change |
| --- | --- |
| `core/news_calendar.py` | No inferred completeness; immutable parsed snapshots; strict YAML; bounded refresh/backoff; pure final reevaluation; memory-only observational access. |
| `src/agents/agent_execution.py` | Retains the initial result and reevaluates it immediately before the sole new-entry `order_send`, after final price, sizing, and prop checks. |
| `src/agents/agent_risk.py` | Preserved Task018B structured risk-side gate; UNKNOWN fails closed under the strict policy. |
| `src/agents/main_agent.py` | Scheduled exits use `refresh=False`; legacy exits run before entry gates; main dispatch no longer suppresses them under reconciliation/session blocks. |
| `strategies/sma_ema_combined.py` | Adverse cross exits precede session/trend entry checks and return before entry-only H1 work can fail or delay them. |
| `core/runtime_paths.py` | Pytest rejects resolved production operational roots and descendants. |
| `conftest.py` | Collection-time filesystem audit guard, isolated news memory/config/cache, and explicit synthetic evidence for legacy order tests; retains Task018B strategy logger isolation. |
| `tests/test_task018b_news_failclosed.py` | Preserves Task018B regressions while replacing unsupported CLEAR expectations with UNKNOWN and testing memory/backoff behavior. |
| `tests/test_task018d_news_safety.py` | 48 adversarial cases covering actual risk/execution gates, strategy/orchestrator exits, main-loop dispatch, timing, duplicate keys, network bounds, and isolation. |
| `tests/test_task011_entry_safety.py`, `tests/test_task015_actual_risk_sizing.py`, `tests/test_task017_trading_mode.py` | Existing Task018B explicit news fixture opt-in, retained. |
| `tests/test_task015b_test_isolation.py` | Existing Task018B strategy logger and production handler regressions, retained. |
| `docs/task018d_news_safety.md` | This evidence/validation report. |

Other pre-existing untracked research/task/data files were not changed.

## Submission, exits, and configuration

Execution performs refresh/initial gating first, retains immutable event times,
then completes its final tick, monetary sizing, and broker-authoritative prop
query. The last news gate takes fresh UTC and evaluates the retained snapshot
without file access, logging, network, or a second fetch. A 301-second event
becomes BLACKOUT at exactly 300 seconds; a six-hour snapshot becomes UNKNOWN
at exactly six hours. Both reject submission under fail-closed policy. A bare
CLEAR result without a retained snapshot also cannot authorize submission.
Only local classification/result checks remain before calling `order_send`.

This removes the demonstrated intervening-broker-query race. It cannot make
the UTC read, Python execution, OS scheduling, MT5 transport, broker receipt,
and fill atomic. The residual scope is that final local interval plus broker
transport/execution latency and clock error. There is **no defensible finite
wall-clock maximum** in this process/MT5 interface; no millisecond guarantee
or invented safety margin is claimed. Policy is retained for one submission;
concurrent configuration editing is not an atomic cancellation mechanism.

Legacy cross exits ignore trade_allowed, NY news, reconciliation, entry-session,
PAUSED and SHADOW gates. The real strategy checks crosses outside the entry
session and with neutral H1 trend. A confirmed cross is returned immediately;
new entry evaluation can resume next cycle. Entry thresholds/SL/TP parameters
are unchanged. Inactive strategies remain inactive; this does not create a
strategy loader for deactivated books.

AMR/Monday scheduled exits defer only for a confirmed BLACKOUT in fresh,
already-validated memory. Missing, stale, malformed, or unavailable evidence
attempts the close. No network or disk calendar refresh occurs on that path.
Journal/reporting calendar observations also use memory only; missing evidence
produces no observational next-news value. Existing close/SLTP/Friday/accounting
paths retain their management semantics. Broker success remains authoritative
for acknowledgment and bookkeeping even if later diagnostics fail.

The news gate strictly loads both entire YAML documents. Duplicate news keys,
parent mappings, nested unrelated mappings, and ambiguous merge overrides
reject configuration before any weaker policy can be honored. Ordinary
overrides between separate global/local documents remain supported. Other
configuration consumers were not broadly redesigned; the entry news gate
fails closed on duplicates anywhere in either parsed document.

## Network and test isolation

Refresh uses a ten-second socket-operation timeout, reads at most 1 MiB plus
one overflow sentinel byte, and rejects oversized feed/cache content. A
process-wide fifteen-minute monotonic retry delay is set before and after
each attempt, including slow failures; valid memory is reused across symbols,
risk, and execution. Retrieval/validation failures become UNKNOWN. Atomic
cache replacement and old-cache preservation on failed publication remain.

The [Python urllib documentation](https://docs.python.org/3.11/library/urllib.request.html#urllib.request.urlopen)
defines an operation timeout, not a whole-request deadline. DNS and repeated
socket reads can exceed ten seconds. The size bound limits payload consumption,
not elapsed time. Entry refresh still occupies the single orchestrator thread
and can delay later cycle work. Exit/journal code itself never initiates that
refresh. This remains a deployment latency limitation; no async/thread bridge
or hard-deadline guarantee was invented.

Tests select temporary runtime/config/cache paths before collection and reset
in-memory news state per case. `data_dir` rejects production data descendants
and other operational roots. The permanent Python audit hook blocks ordinary
open/write/create/delete/rename/link mutations under data, logs, state,
journal(s), and reports, including hardcoded legacy targets. Source fixture
and repository reads remain allowed. These are ordinary Python API guards,
not an OS sandbox against native code, subprocesses, or pre-existing external
writers. Validation additionally forbade socket use and all live MT5 callable
functions. No test used a real terminal or network calendar.

## Validation and independent attack reproductions

Final validation on 2026-09-05, with `python -B` and pytest cache disabled:

| Check | Result |
| --- | --- |
| Task018B + Task018D | 166 passed |
| Task015/015B + Task016/016C + Task017/017B | 153 passed |
| Full pytest | 442 passed |
| Full pytest with only the permanent conftest filesystem guard | 442 passed; confirms the outer validation write guard did not mask isolation defects |
| In-memory syntax compilation | 181 Python files passed; no bytecode written |
| `git diff --check` | Passed; Git only reported existing LF/CRLF conversion notices |

The regression/full runs emitted 14 existing matplotlib/pyparsing deprecation
warnings. Task017D broker-success/diagnostic-failure behavior is covered in
the sizing and acknowledgment regressions. No rejection acknowledged a signal.

A separate inline Python program (not pytest test execution) called real
calendar/risk/execution/orchestrator functions with temporary files and broker
fakes. Timing probes supplied a synthetic completeness oracle only to reach
the otherwise unreachable CLEAR path. Results:

| Reproduced attack | Independent result |
| --- | --- |
| Valid low-impact records bracketing an omitted event | UNKNOWN; risk REJECTED; zero new-entry sends. |
| 301-second event, final prop query advances two seconds | Synthetic CLEAR -> BLACKOUT; zero sends. |
| Snapshot expires during final prop query | Synthetic CLEAR -> UNKNOWN; zero sends. |
| Duplicate fail-closed/filter/window/parent keys | Every case rejected; zero sends. |
| Legacy exit under market/news/reconciliation/PAUSED/SHADOW entry blocks | One legitimate close attempt per case; no new entries. |
| Unavailable calendar on AMR/Monday exits and journal | Two close attempts, zero calendar fetches. |
| Five failing entry evaluations across symbols and execution | UNKNOWN; one fetch attempt, zero sends. |
| Selecting data/state, data/logs, data/journal, journals, reports/runtime | All five rejected before writes. |

Permanent tests additionally execute the real main loop and real SMA strategy,
prove inclusive boundaries and forged cache completeness rejection, verify
fresh-memory blackout deferral, and exercise the real bounded urllib reader.

## Production fingerprints and residual classification

Each final test run and independent reproduction had its own immediate
before/after SHA-256 fingerprints of **518 files**, covering all existing
operational data/report roots plus config, pairs, and strategy sources. Every
final comparison was identical, including file additions/deletions. All five
shared this aggregate sorted manifest SHA-256:

`73de96d786f6eac7de3162d2a7a0f6d179ed41ee85e1bb68aaeedce32edc4b40`

The first combined validation window was explicitly failed because the live
bot appended two ordinary ORCHESTRATOR lines at 11:00:05 UTC. Hashing the exact
log prefix reproduced the original fingerprint, proving that those two
appends were the only log difference. The required validations were then
rerun with the unchanged results above; the failed window was not silently
accepted. Production processes were left running throughout. The ten prior
synthetic log lines were not removed or edited.

Commit suitability is based on the conservative entry behavior and passing
offline regressions. LIVE deployment remains unsuitable as a working trading
system until a trusted completeness source is provided and its timing/latency
contract reviewed. Further residuals are provider revisions/unsupported
records (conservatively UNKNOWN), accurate system-clock dependence, explicit
fail-open/disabled overrides, and unbounded OS/broker/request latency described
above. No actual fill-time or broker integration validation is claimed.
