# Phase 40 — Data Integrity (Part 10)

Run before substantive analysis, per the preregistered methodology.

## Checks performed

1. **`research_data_validator.validate_column_count_consistency`** run on `data/phase26_all_trades.csv` (the drawdown-correlation control source) — passed (1/1 checks).
2. **MT5 H1 pulls** for EURUSD/GBPUSD/AUDUSD/USDCAD, 2019-01-01 to 2026-08-14: each pull asserts monotonic timestamps, zero duplicate candles, and strictly positive OHLC (identical convention to every prior phase's MT5 puller) — all four passed.
3. **No look-ahead in volatility calculation**: the normalized-ATR series is a standard trailing rolling mean (pandas `.rolling(14).mean()` over the true-range series) — by construction, the value at bar *t* uses only bars ≤ *t*. The entry decision at bar *t+1* explicitly reads the **prior** bar's (*t*'s) already-realized state via a `.shift(1)`-equivalent indexing pattern, verified by inspection of `src/phase40_volatility_conditioned.py::high_vol_trades()`.
4. **Regime labels are point-in-time**: the TRAIN-period tercile thresholds (`q1`, `q2`) are computed once from TRAIN-only data and applied as fixed numeric thresholds to VALIDATION/OOS — never re-estimated on out-of-train data, verified by inspection (threshold computation occurs before any VALIDATION/OOS filtering in the script).
5. **No OOS data leaks into normalization**: the ATR rolling window itself is a local trailing calculation (no cross-period contamination), and the tercile threshold values are TRAIN-derived constants applied unchanged going forward — confirmed no re-fitting occurs anywhere in the pipeline.
6. **UTC/session-hour handling**: uses the project's established raw-MT5-server-hour convention (matching `src/phase19_london_ny_volatility_persistence.py`'s NY=[12,21) definition, adjusted per this phase's own frozen 13:00-21:00 window) — not a naive UTC assumption, consistent with the project's documented server-time-fix history.

## Result

**No material integrity failure found.** Proceeding to baseline reproduction (Part 11).
