# Research Data Integrity Policy

**Status:** in force starting Phase 30. Applies to every CSV/JSONL artifact produced or consumed by this project's research scripts (`src/phase*.py`), the production export tool (`scripts/export_5ers_trades.py`), and the MCP dashboard's data reads (`mcp/server.py`).

---

## 1. The incident that motivated this policy

`reports/current_6_strategy_revalidation.csv` (commit `6fd93a3`, 2026-07-13) was **hand-composed text**, not written through a CSV-serialization library (`csv.writer`/`pandas.to_csv`). One cell — `parameter_stability_status`, value `"STABLE (broad plateau, both z_thr and sl_mult)"` — was written **without quoting** the internal comma.

- **Where it was created:** authored directly as a markdown-adjacent data table by a prior research phase, not generated programmatically from a DataFrame or dict.
- **Why the comma wasn't quoted:** there was no automatic quoting step, because no CSV-writing library was in the authoring path — hand-typed text has no such safety net.
- **How the parser behaved:** pandas' default C engine raised a tokenization error; `engine='python', on_bad_lines='warn'` (used in an early fix attempt) **silently skipped** the 4 affected rows entirely, logging only a terse warning easy to miss in bulk console output.
- **Which fields were affected:** 4 of 7 data rows (GBPJPY_AMR, EURJPY_AMR, CADJPY_AMR, GBPUSD_MONDAY) — every strategy row whose `parameter_stability_status` text happened to contain a comma. The other 3 rows (CADJPY_ARB, AUDJPY_AMR) were unaffected because their equivalent text happened not to contain a comma — **a data corruption bug whose symptom is silent and value-content-dependent is exactly the most dangerous kind.**
- **Why the failure wasn't caught earlier:** no round-trip or schema validation was ever run against this file after it was written. It was read directly by three downstream analysis phases (27, 28, 29) before the corruption was noticed — purely because a human happened to compare an aggregate count and find it implausible, not because any automated check existed.

**Fixed (Phase 29) without modifying the historical artifact:** `src/phase27_5ers_current_portfolio_forensic.py::_load_summary_csv_robust()` reads the file with the stdlib `csv` module (which reports the true field count per row rather than silently realigning) and merges the split field back together for any row with exactly one extra field. The source file itself was deliberately left untouched — see §5.

---

## 2. Required schemas

Every research CSV must have its schema declared **before** being written, as one of:
- An explicit `set[str]` of required columns passed to `validate_required_columns()`.
- A full fixed column list (for frozen/versioned artifacts like `reports/*_scorecard.csv`) checked with `validate_no_unexpected_columns()`.

Ad hoc columns added after the fact without updating the schema declaration are themselves a validation failure, not a silent extension.

---

## 3. Validation requirements

**Every CSV a research script reads must pass, at minimum, before any analysis proceeds:**
1. `validate_column_count_consistency()` — the single check that would have caught the incident in §1 immediately.
2. `validate_required_columns()`.
3. `validate_row_count()` (exact for frozen artifacts, min/max range for growing ones — see §4).

**Every CSV a research script writes must be written through `csv.DictWriter` or `pandas.to_csv`** — never string concatenation, f-strings, or manual `open(...).write()` — and should be spot-checked with `validate_roundtrip()` when the schema is new or changed.

**Production trade exports specifically** must additionally pass:
4. `validate_lifecycle_pairing()` (every ticket has exactly one OPEN + one CLOSED row).
5. `validate_no_missing_values()` on `strategy`/`account`/`trade_id`.
6. `validate_numeric_columns()` / `validate_datetime_columns()` on the relevant fields, with `NOT_AVAILABLE` as the only permitted non-numeric/non-date sentinel.

---

## 4. Structural rules vs. current-snapshot expectations

This distinction is **required**, per the explicit Phase 30 instruction not to hardcode a specific row count as a permanent rule:

| Rule type | Example | Validator call |
|---|---|---|
| **Structural** (must always hold, regardless of how much the account has traded) | Every ticket has exactly one OPEN + one CLOSED row; `account` and `strategy` are never blank; no column-count mismatches | `validate_lifecycle_pairing()`, `validate_no_missing_values()`, `validate_column_count_consistency()` |
| **Current-snapshot** (true today, expected to grow) | "72 rows, 36 tickets" as of 2026-08-13 | `validate_row_count(min_rows=<last known count>)` — a **floor**, never an exact match, for any file expected to keep growing |

**A validator that hardcodes "must equal exactly 72 rows" forever would itself become a false-positive integrity failure the next time the account closes a trade.** Every production-export check in this policy uses `min_rows`, never `expected=`, for that reason.

---

## 5. Raw-data handling

- Raw production data (`reports/5ers_trade_export.csv`, and the VPS-side `trades_log.csv`/`journal/events.jsonl` it's derived from) is **never committed to git** — established policy, unchanged by this document.
- Historical research artifacts with a known-but-unfixed source defect (like `current_6_strategy_revalidation.csv`) are **not silently edited** after the fact — the corruption is worked around in the reading code (as in §1's fix), with the workaround documented inline, so the git history remains an honest record of what was actually produced at the time. A source artifact is only edited directly when the edit is the explicit subject of a task (it is not here).

---

## 6. Production-data handling

- Every script that reads `reports/5ers_trade_export.csv` (or any future production export) must call the structural + snapshot validators from §3 **before** any metric is computed — not after, not "if convenient."
- A validation failure on production data is always fatal to the run (§8) — no research conclusion may be drawn from a file that failed its own schema check, even partially.

---

## 7. Reproducibility requirements

- Every research script under `src/phase*.py` must be re-runnable from a clean checkout given only the committed inputs (or, for production-data-dependent scripts, a fresh export placed at the documented path) — no hidden state, no manual pre-processing steps not captured in the script itself.
- Every generated report must cite the exact script and commit/experiment ID that produced it (already this project's convention — continued here, not newly introduced).

---

## 8. When analysis MUST STOP

Per the explicit Phase 30 stop conditions, analysis must halt (not continue with a caveat) if:
- Any `ResearchDataError` is raised by a validator and cannot be resolved by fixing the *reading* code (never by silently coercing/dropping in the *analysis* code).
- A data schema does not match what the analysis code expects.
- Historical data contains unexplained gaps.
- Train/validation/OOS boundaries cannot be established from the data's own timestamps.
- Results would depend on an assumption not explicitly documented in the report.

**`ResearchDataError` is intentionally not caught anywhere in the analysis pipeline** — it is designed to propagate and halt the script, matching this requirement structurally, not just by convention.

---

## 9. When a dataset is safe to use

A dataset is safe to use in a research conclusion only when:
1. It has passed every applicable validator in §3 with zero failures (not "failures ignored because the row count looked plausible").
2. Its provenance (which script/export run produced it, and when) is stated in the consuming report.
3. Any known pre-existing defect in the file (like `current_6_strategy_revalidation.csv`'s comma issue) is either fixed at the reading layer with the workaround documented, or the file is regenerated cleanly.

---

## 10. How generated research artifacts should be committed

- Derived reports/CSVs (scorecards, forensic analyses, registries) **are** committed — they're the project's evidentiary record.
- Raw production trade data, credentials, and VPS-specific private paths/secrets are **never** committed — unchanged existing policy, re-stated here for completeness.
- Every new CSV artifact should be written via `csv.DictWriter`/`pandas.to_csv` (§3) so that its own commit is inherently round-trip-safe from the moment it's created — this policy exists specifically so the §1 incident's root cause (hand-composed text) cannot recur.

---

*This policy applies going forward from Phase 30. It does not retroactively modify any previously-committed artifact.*
