# Task018F: native calendar bridge, shadow only

Classification: **SAFE TO COMMIT FOR SHADOW VALIDATION**.

Task018F adds five files; it changes no Task018D production authorization code,
production configuration, or operational data. Nothing was staged, committed,
attached to a terminal, deployed, or run against live MT5. The existing
Task018B/D working-tree changes remain separate and unstaged.

| New file | Responsibility |
| --- | --- |
| `mql5/CalendarBridgeShadow.mq5` | Non-trading MQL5 service; native queries, clock/identity observations, immutable payload publication. |
| `core/calendar_bridge.py` | Bounded local reader, strict schema validator, replay history, immutable evidence, bounded shadow reporter. |
| `src/calendar_bridge_shadow.py` | Explicit one-shot observation command; stdout/stderr logging only, no production imports. |
| `tests/test_task018f_calendar_bridge.py` | 101 offline native-output and adversarial tests. |
| `docs/task018f_bridge.md` | Contract, limitations, validation, and future shadow-use instructions. |

## Production separation

No production module imports the bridge. The existing news gate, its
`_proves_coverage` implementation, risk gate, and final execution reevaluation
are unchanged. The bridge cannot publish or construct `NewsResult` objects.
`BridgeEvidence.entries_allowed` always returns False, including for VALID
evidence with a shadow CLEAR classification. Its separate result type is also
rejected by the existing defensive `isinstance(NewsResult)` check.

Tests verify the dependency boundary, valid zero-event bridge evidence alongside
real production UNKNOWN, real risk rejection, zero mocked new-entry sends, and
rejection if a caller substitutes a bridge result into the news helper.
Creating a bridge file, running the observer, or seeing shadow CLEAR cannot
enable live entries. Bridge HIGH evidence also has no production effect.

## Native service and query semantics

The service has no includes, DLL imports, trading classes, order functions,
execution request structures, position management, or strategy logic. It uses
`CalendarCountries`, `CalendarValueHistory`, `CalendarEventById`,
`CalendarCountryById`, and before/after `CalendarValueLast` probes.

Every calendar operation immediately follows `ResetLastError()` and immediately
captures `GetLastError()`. History success requires integer `count >= 0`, zero
error, a matching array length, and the event-count cap. Zero is legitimate;
negative results and nonzero errors, explicitly including 5400 truncation,
cannot validate. Metadata lookups require successful Boolean returns, zero
errors, and matching parent identifiers. The correct lookup is
`CalendarEventById(values[i].event_id, event)`, never the occurrence's `id`.
These distinctions follow the [native interval contract](https://www.mql5.com/en/docs/calendar/calendarvaluehistory)
and [event lookup contract](https://www.mql5.com/en/docs/calendar/calendareventbyid).

One unfiltered history query covers whole server dates intersecting the window
plus evidence lifetime, with one-second endpoint padding. Native records are
fully enriched; no HIGH-only filtering or silent row dropping occurs. Failed
enrichment records a failure stage, false health evidence, and/or count mismatch.
Even if a prefix was serialized, the reader rejects the entire generation.
The country catalogue must advertise AUD, CAD, EUR, GBP, JPY and USD.

The change-ID probes start with zero; their zero record count means obtaining
the database identifier, not proving an empty calendar. Nonzero equal IDs and
zero captured probe errors are required. This catches reported changes during
construction but is not a transactional or upstream synchronization guarantee.
[Change identifier behavior](https://www.mql5.com/en/docs/calendar/calendarvaluelast).

## Version-one payload and manifest

Payload top-level keys are exactly:

`schema_version`, `source`, `instance_id`, `boot_id`, `sequence`, `identity`,
`clock`, `query`, `health`, `coverage`, `events`.

Source is `mql5-calendar-shadow`. Native identifiers and sequence are canonical
positive decimal strings; boot and instance tokens permit only a bounded set of
letters, digits, underscores and hyphens. No trusted `complete` flag exists.
Unknown keys, duplicate keys, nonfinite JSON, booleans masquerading as integers,
invalid identifiers, and unsupported schema versions are rejected.

| Section | Fields |
| --- | --- |
| identity | login, server, company, terminal_path, terminal_data_path |
| clock | generated_server_time, generated_utc_time, server_utc_offset_seconds, offset_sample_time, clock_status, clock_uncertainty_seconds, offset_before_seconds, offset_after_seconds, quote_age_before_seconds, quote_age_after_seconds |
| query | server_start, server_end, utc_start, utc_end, started_utc, elapsed_ms, return_count, error_code, query_success, failure_stage |
| health | terminal_connected, event_enrichment_complete, country_enrichment_complete, currency_catalog_valid, change_before, change_after, change_error_before, change_error_after |
| coverage | utc_start, utc_end, supported_currencies, returned_event_count |
| each event | value_id, event_id, country_id, country_code, currency, importance, time_mode, name, server_time, utc_time |

Time fields are integer seconds: UTC fields are UTC epochs; server fields encode
the native server-clock coordinate. No naive server timestamp is labeled UTC.
For uncertain time modes, `utc_time` is null, preserving the original native
server field without manufacturing an exact release instant.

The manifest contains exactly `schema_version`, `instance_id`, `boot_id`,
`sequence`, `payload_filename`, `payload_bytes`, `payload_sha256`, `published_utc`.

Publication builds UTF-8 bytes, excludes the encoder's terminating zero, hashes
those exact bytes with SHA-256, writes/flushed/closes a unique temporary payload,
then moves it to `calendar_<boot>_<sequence>.json` without replacement. Existing
generation names or temporary names cause failure. Only afterward is a complete
manifest written, flushed, closed, and moved over `manifest.json`.

`FileMove` is not assumed atomic. The reader requires complete manifest parsing,
the exact expected basename, a matching payload size and SHA-256, agreement of
manifest/payload identifiers, and an identical manifest on a second read.
Incomplete publication is rejected; the reader never opens `.tmp` files or
falls back by scanning older generations. If publication fails before replacing
the manifest, the old committed generation can remain visible only until its
original expiry. This is not instantaneous producer-failure detection.

The intended future directory is the verified terminal's
`TERMINAL_DATA_PATH\MQL5\Files\CalendarBridge\<instance_id>\`.
`FILE_COMMON` is never used. The service holds an exclusive writer handle for
the instance. File operations remain inside the MQL5 sandbox; the reader
rejects escaping payload paths, UNC directories and nonlocal Windows drives.
[MQL5 filesystem rules](https://www.mql5.com/en/docs/files/fileopen),
[FileMove contract](https://www.mql5.com/en/docs/files/filemove).

## Clock evidence and identity

The service monitors host UTC, estimated server time, and advancing quote time
once per second. Calendar retrieval itself targets a 60-second cadence.
Observed quote advancement and both quote-age checks must be within ten seconds.
An initial stale quote cannot satisfy startup clock validity. Host-clock versus
monotonic elapsed disagreement, backward quotes, and offset shifts impose a
90-second default hold-down. Samples before and after construction must agree.

The observed offset is quantized to 15-minute timezone increments only if the
raw difference is within two seconds, and must fall between UTC-12 and UTC+14.
It is not assumed permanently UTC+3. This is a plausibility/consistency policy,
not authenticated proof of the broker's timezone. The uncertainty field records
the two-second acceptance tolerance, not a metrologically verified error bound.

The host-derived UTC clock must still be trustworthy. `TimeTradeServer` is locally
calculated and `TimeCurrent` can be stale; their agreement alone does not prove
accurate UTC. Calendar records use the server's current timezone/DST setting.
Precise event timestamps are normalized exactly once by subtracting the
snapshot offset, with no second historical DST shift.
[Server-clock limitations](https://www.mql5.com/en/docs/dateandtime/timetradeserver),
[calendar timezone semantics](https://www.mql5.com/en/book/advanced/calendar/calendar_cache_tester).

Identity is checked before and after construction and serialized from native
account/terminal observations. Defaults name login 26520700, server
FivePercentOnline-Real and installation C:\MT5-5ers. `ExpectedDataPath` defaults
to empty, so the service refuses to start until a verified data directory is
explicitly supplied. Installation and data directories are never assumed equal.
The producer does not invent a company name; it requires a nonempty observed
one. The reader optionally pins that company in addition to mandatory login,
server, terminal path, data path, and instance ID.

These controls prevent accidental demo-snapshot consumption. A digest and
self-reported identity do not authenticate a malicious same-user writer.

## Reader and shadow classification

Evidence states: VALID, INVALID, STALE, IDENTITY_MISMATCH, UNAVAILABLE.
VALID means this shadow schema and its checks passed, not that production may
trust the provider. Every evidence result prohibits entries.

Local reads are bounded to an 8 KiB manifest, 1 MiB payload, and second manifest
read. Event count is capped at 5,000. No network, MT5 initialization, terminal
query, directory scan, or production logger initialization occurs in the reader.
Local filesystem work is bounded in amount, not by a hard OS scheduling deadline.

Counts, health flags, query success/error, identity, offsets, generated times,
elapsed time, explicit coverage, currencies and every event are checked. Query
duration over 15 seconds is rejected, without claiming native cancellation.
Expiry is 90 seconds from query start and is configurable downward in the reader.
Neither file mtime nor a newly published heartbeat renews calendar freshness.
The entire current evaluation window, including clock uncertainty, must lie
strictly inside the declared query interval; event bracketing is never evidence.

Supported requests cover the six required currencies. XAUUSD maps to USD only.
Known auxiliary calendar currencies can appear in the all-events response but
do not expand supported trading-symbol requests. Unrecognized currency values
fail validation conservatively; the allowlist is explicit in the reader.

HIGH, MODERATE, LOW, NONE and UNKNOWN remain explicit. HIGH events confidently
inside the window yield shadow BLACKOUT. NONE/UNKNOWN importance overlapping
the window yields shadow UNKNOWN. DATE/NOTIME/TENTATIVE/UNKNOWN timing on a
relevant HIGH or unresolved-importance event conservatively prevents shadow
CLEAR for that generation. Clock-uncertain boundaries are UNKNOWN. LOW and
MODERATE remain recognized below the HIGH policy. Precise out-of-window events
do not block shadow CLEAR. All of these classifications are observational.

A reader remembers accepted sequence/digest/query time. Identical rereads are
allowed until expiry; rollback or a changed payload at the same sequence is
invalid. A changed boot is quarantined until a newer generation with a later
query start arrives; retired boots cannot return. Boot history is capped at 16.
Replay history lasts for the reader object, not across observer process restarts.
One-shot CLI invocations therefore provide freshness checks, not persistent
replay protection.

## Explicit observation path

The observer is opt-in, with no automatic orchestrator hook or production
configuration change. For an **offline fixture directory**, use:

```powershell
python -B -m src.calendar_bridge_shadow --directory "<fixture directory>" --terminal-path "C:\MT5-5ers" --terminal-data-path "<pinned fixture data identity>" --login 26520700 --server FivePercentOnline-Real --instance-id "<fixture instance>" --symbol GBPUSD --existing-state UNKNOWN
```

`--existing-state` is a caller-supplied Task018 observation. The command does not
fetch or evaluate production news to obtain it. The reusable `ShadowReporter`
accepts the same explicit reference state, reports bridge/candidate/existing
states, an event-set digest, count and at most five event samples, and throttles
output to once per 60 seconds. It never refreshes the bridge or existing news.
Use a persistent reader/reporter for continuous future shadow comparisons.

## Validation

All Python runs used `python -B`, disabled pytest's cache provider, temporary
native-output fixtures, and guards against live MT5 calls and socket activity.

| Check | Result |
| --- | --- |
| Focused Task018F tests | 101 passed |
| Existing Task018B/D tests | 166 passed |
| Task015/015B, Task016/016C, Task017/017B | 153 passed |
| Full pytest | 543 passed; 14 existing matplotlib/pyparsing deprecation warnings |
| Python in-memory syntax compilation | 184 files passed; no bytecode written |
| Git/new-file whitespace checks | Passed |
| Static MQL5 execution-API scan | No OrderSend, OrderSendAsync, CTrade, PositionClose, PositionModify, MqlTradeRequest, TRADE_ACTION or DLL-import references |

Production fingerprints cover 518 existing operational/configuration/pair/
strategy files, including additions/deletions and reports. Focused and regression
windows were unchanged. The first full-suite window had an external MCP
authentication-failure append in `data/logs/mcp_access.log` at 12:17:38 UTC and
was rejected by the fingerprint wrapper despite passing tests. The full suite
was rerun: **543 passed in 15.67 seconds, all 518 fingerprinted files unchanged**.
The final before/after manifest SHA-256 was
`3f6d4ebdab91ee7c2aed0b5e8ba18ef7a82237ecd1f3438e5c70d65bf30c9ff1`.
No external service was stopped or modified. The full suite uses the permanent conftest filesystem
guard, without an outer filesystem guard masking that implementation.

MQL5 compilation was performed using a copy of the installed MetaEditor in a
unique temporary directory, with `/portable /compile /log`. No terminal binary,
terminal data directory, account state, or live service was opened. The official
documentation supports [command-line compilation](https://www.metatrader5.com/en/metaeditor/help/beginning/integration_ide)
and [portable editor data isolation](https://www.metatrader5.com/en/metaeditor/help/beginning/open).
The final compiler log reports **0 errors, 0 warnings**, with a 53,054-byte EX5
artifact retained only in the temporary compiler directory. The compiler process
returned 1 despite successful code generation; the explicit log and artifact
were checked rather than treating that return code as proof of failure/success.
The copied and repository sources match byte-for-byte, SHA-256:

`075071efc7babde5d326a964d3583a3521ee9c70da7f602c73f2ee334fe69587`

The service was never attached or executed. Python/native fixture interoperability
is specified and tested on the Python side; actual terminal-produced payloads,
live clock calibration and calendar availability still need later observation.

## Residuals and later shadow work

- Native query completeness is relative to the provider's database. Missing or
  late upstream revisions and uncertain releases remain possible.
- Clock consistency does not establish an independent trustworthy UTC source.
  A future authorization review must assess this separately.
- Native calls can stall; Python evidence expires while the independent service
  is blocked. There is no hard whole-request or broker-execution deadline.
- Interrupted writes never validate as partial payloads, but an older committed
  snapshot can remain fresh until its original expiry.
- The service stops after 1,440 publication attempts per boot (roughly a day at
  defaults) rather than deleting files. Payloads are individually capped; storage
  accumulated across boots requires explicit operator housekeeping. Do not leave
  this initial shadow service unattended indefinitely.
- Later work must provision the actual data path and instance, verify identity
  and native event/clock observations, exercise publication/disconnection/restart
  failures, and compare shadow output. No such deployment is part of Task018F.
- Production CLEAR integration would be a separate reviewed change. Nothing in
  this task enables it.
