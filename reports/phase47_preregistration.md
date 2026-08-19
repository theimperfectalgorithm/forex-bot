# Phase 47 Preregistration — Live-Source Reproduction & Robustness Harness

**Frozen before any reproduction result is inspected. Committed separately, before any Phase 47 result exists. Not modified after seeing results.**

INFRASTRUCTURE + REPRODUCTION ONLY. No live strategy code, YAML, execution logic, or risk setting modified. No deployment. No robustness *conclusions* — this phase only establishes whether the testing machinery is trustworthy.

---

## 1. The six frozen strategies (verified against the repository)

`AUDJPY_AMR`, `CADJPY_AMR`, `EURJPY_AMR`, `GBPJPY_AMR` (source: `strategies/asian_hours_reversion.py` + `pairs/*_asianrev.yaml`), `CADJPY_ARB` (source: `strategies/asian_range_breakout.py` + `pairs/CADJPY_asianrange.yaml`), `GBPUSD_MON`/`GBPUSD_MONDAY` (source: `strategies/monday_drift.py` + `pairs/GBPUSD_monday.yaml`). No candidate strategy added.

## 2. Source/configuration snapshot methodology

Every source and YAML file is read directly and its content quoted/paraphrased into the deliverable CSVs — never inferred, never reconstructed from memory. A SHA-256 hash of each source/config file, computed at read time, is recorded in `reports/phase47_source_inventory.csv` / `phase47_live_config_snapshot.csv` as the immutable "this is what was actually present" record for this phase.

## 3. Historical data source

MT5 M15 (AMR z-score signal), H1 (ARB Asian range + Monday drift ATR), and H4 (ARB trend filter) bars, pulled fresh via the same `MetaTrader5` Python API already used throughout this project. Date range: 2023-08-01 to 2026-08-13, matching the control ledger's own coverage (`data/phase26_all_trades.csv`).

## 4. Execution assumptions (frozen, disclosed limitations)

Signal generation is reproduced **exactly** as written in the live strategy source (z-score/SMA/STD formulas, H4 trend sign test, Asian-range construction, ATR20d calculation) — copied logic, not reinterpreted. **Execution/fill assumptions are approximate**: entry price = the signal bar's own close (matching the live code's own `entry_price` field), no explicit slippage model (none exists in the live code either — it uses the raw signal-bar close), spread not separately modeled (consistent with every backtest in this project's history, which has never had broker-level tick/spread data). This is disclosed as **APPROXIMATE REPRODUCTION**, not EXACT REPRODUCTION, before any result is seen.

## 5. Matching methodology and tolerance (frozen)

A reconstructed signal is compared against the historical ledger (`data/phase26_all_trades.csv`) by **(pair, entry date, direction)** — the historical ledger does not retain a shared instrument-level trade ID with any live signal log, so exact-timestamp matching is not possible; date-level matching is the finest grain the data supports, disclosed here, not chosen after seeing results. A match is classified:
- **EXACT MATCH**: same pair, same entry date, same direction, R-multiple within ±15% of the historical value.
- **ACCEPTABLE MATCH**: same pair, same entry date, same direction, R-multiple outside ±15% but same sign.
- **MISMATCH**: same pair, same entry date, opposite direction, or a reconstructed signal exists with no historical counterpart on that date (false positive) or vice versa (false negative).
- **UNMATCHABLE**: reconstruction could not be attempted (missing MT5 history for that date/pair).

## 6. Minimum reproduction threshold for Stage A PASS (frozen, per Part 14)

A strategy receives:
- **A. REPRODUCTION PASS**: ≥85% of historical trades reach EXACT or ACCEPTABLE MATCH.
- **B. REPRODUCTION PASS WITH LIMITATIONS**: 60-85% match rate, or ≥85% match rate on direction/date alone (signal reproduction) but with disclosed execution-price divergence.
- **C. REPRODUCTION FAILURE**: <60% match rate.
- **D. INSUFFICIENT DATA**: MT5 history unavailable for a material fraction of the required period/pair/timeframe.

These thresholds are chosen before any reproduction is run, based on this project's own established Gate-1 sample-adequacy conventions (never loosened after seeing results, per Part 13's explicit instruction).

## 7. Signal reproduction vs. execution reproduction (frozen distinction, per Part 13)

A strategy may reach a PASS classification on **signal reproduction** (correct pair/date/direction) while still having unreproducible **execution** detail (exact fill price, exact R) — both are reported separately in `reports/phase47_reproduction_metrics.csv`, never conflated into one number.

## 8. Documented-vs-live discrepancy handling (frozen, per Parts 7-8)

Every discrepancy found between a strategy's docstring, its source implementation, and its live YAML/execution-layer code is recorded in `reports/phase47_documented_vs_live.csv` with **all three states preserved side by side**. No discrepancy is corrected, normalized, or resolved in this phase, regardless of which state appears intended.

## 9. Parameter/cost harness design (Stage B, frozen, per Parts 17-20)

**Continuous/discrete perturbable parameters** (per strategy, identified directly from source): AMR — `z_threshold`, `sl_multiplier`, `entry_end_hour`(discrete, per-pair); ARB — `tp_multiplier`, `min_range_pips`; Monday — `sl_atr_mult`, `tp_atr_mult`. **Categorical/non-perturbable**: `h4_filter` (binary on/off, not a ±20% perturbation target — its two live states are documented as-is, not perturbed), session window boundaries (frozen structural choices, not perturbed per Part 18's explicit instruction), pair/instrument selection.

The parameter harness (`src/phase47_parameter_harness.py`) and cost harness (`src/phase47_cost_harness.py`) are built and **sanity-tested only** in this phase (Parts 19-20) — changing exactly one parameter/cost value and verifying no other output changes. **No final ±20% or 2x-cost robustness conclusion is produced in this phase**, per Part 21's explicit prohibition — that is reserved for Phase 48, contingent on this phase's Stage A/B readiness verdict.

## 10. Determinism and immutability testing (frozen, per Parts 23-24)

The same historical replay is run twice per strategy; results must be byte-for-byte identical (same trade list, same R values) or the phase STOPS per Part 23. SHA-256 hashes of every live strategy/config file are captured before and after every harness run; any hash change is a hard failure per Part 24 — the harness must never write to `strategies/`, `pairs/`, or any live configuration path.

## 11. Portfolio reproduction methodology

The six strategies' reconstructed trade lists are combined without deduplication logic beyond what each strategy's own source already enforces (one-trade-per-day-per-pair) — no cross-strategy conflict resolution is invented, since the live system runs each strategy independently.

## 12. Final classification framework (frozen, per Part 28)

A (REPRODUCTION READY — PROCEED TO PHASE 48) requires **every** strategy at A or B reproduction AND passing determinism AND passing immutability AND both harnesses passing their sanity tests. Any strategy at C or D, or any failed determinism/immutability/sanity test, caps the phase at **B (PARTIALLY READY)** at best, and **C (REPRODUCTION FAILURE — DO NOT PROCEED)** if the failure is material and widespread. Phase 48 is **not** authorized to run final robustness analysis unless this phase concludes A.

---

*No amendment has been made to this document after any Phase 47 result was produced.*
