"""
Forex Bot - H1 Bollinger Band Mean Reversion: Train/Forward-Test Walk-Forward

Same rigorous methodology as the M15 EMA-crossover search
(src/m15_walk_forward_search.py), applied to a Bollinger Band mean-reversion
strategy on EURUSD H1.  Background research only -- not for live deployment.

Data split (strict 50/50, no leakage):
  Training     : 2020-01-01 to 2022-06-30  (~2.5 years)
  Forward test : 2022-07-01 to 2024-12-31  (~2.5 years, untouched until the end)

Strategy logic:
  Bollinger Bands(period, k) on H1 close.
  BUY  when close crosses below the lower band  (expect reversion up to mid).
  SELL when close crosses above the upper band  (expect reversion down to mid).
  TP   : dynamic -- exit when price reaches the *previous* bar's middle band
         (SMA), avoiding same-bar lookahead.
  SL   : fixed at entry = sl_mult x band width (upper-lower at entry), beyond
         the entry price.
  H4 filter : only trade when |H4 SMA50 - SMA200| <= neutral_threshold_pips
              (ranging market). Skip when H4 is strongly trending.
  Session   : entries only allowed 08:00-22:00 UTC (London+NY combined).
              Open trades are still monitored for SL/TP at any hour.

Grid:
  BB period           : 15, 20, 25, 30                 (4)
  Std deviations (k)   : 1.5, 2.0, 2.5                   (3)
  H4 neutral threshold : 20, 30, 40 pips                  (3)
  SL multiplier        : 1.0x, 1.5x, 2.0x band width      (3)
  Total combinations: 4 * 3 * 3 * 3 = 108

Optimization targets (training data ONLY):
  45% <= win rate <= 65%
  max drawdown < 4%
  profit factor > 1.2
  >= 60 trades
  profitable in >= 60% of training months

Top 5 training-passing configs (ranked by profit factor desc, max DD asc
tiebreak) are forward-tested exactly once, unmodified, on 2022-2024 data.

Requirements:
  MetaTrader 5 must be OPEN and LOGGED IN before running this script.
"""

import sys
import time as _time
from datetime import datetime, timezone
from itertools import product

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

# ── SETTINGS ──────────────────────────────────────────────────────────────────

SYMBOL          = 'EURUSD'
PIP_SIZE        = 0.0001
PIP_VALUE_USD   = 1.00
INITIAL_BALANCE = 10_000.00

FETCH_FROM = datetime(2019, 9, 1, tzinfo=timezone.utc)   # buffer for indicator warmup
FETCH_TO   = datetime(2025, 1, 1, tzinfo=timezone.utc)

TRAIN_START = pd.Timestamp('2020-01-01', tz='UTC')
TRAIN_END   = pd.Timestamp('2022-07-01', tz='UTC')   # exclusive
FWD_START   = pd.Timestamp('2022-07-01', tz='UTC')
FWD_END     = pd.Timestamp('2025-01-01', tz='UTC')   # exclusive

SESSION_START_HOUR = 8
SESSION_END_HOUR   = 22   # exclusive

H4_FAST_PERIOD = 50
H4_SLOW_PERIOD = 200

# ── GRID ──────────────────────────────────────────────────────────────────────

BB_PERIODS       = [15, 20, 25, 30]
BB_STDDEVS       = [1.5, 2.0, 2.5]
H4_THRESH_PIPS   = [20, 30, 40]
SL_MULTIPLIERS   = [1.0, 1.5, 2.0]

# ── TARGETS (training data only) ─────────────────────────────────────────────

TARGET_WR_MIN     = 45.0
TARGET_WR_MAX     = 65.0
TARGET_MAX_DD     = 4.0
TARGET_PF_MIN     = 1.2
TARGET_MIN_TRADES = 60
TARGET_MONTHLY_PROFITABLE_PCT = 60.0

TOP_N = 5


# ── 1. CONNECT + FETCH ────────────────────────────────────────────────────────

def connect_mt5():
    print("Connecting to MetaTrader 5...")
    if not mt5.initialize():
        print(f"ERROR: Could not connect — {mt5.last_error()}")
        sys.exit(1)
    info    = mt5.terminal_info()
    account = mt5.account_info()
    print(f"Connected : {info.name}")
    print(f"Account   : {account.login}  ({account.server})\n")


def fetch(symbol: str, timeframe, date_from: datetime, date_to: datetime, label: str) -> pd.DataFrame:
    print(f"Requesting {label} data from {date_from.date()} to {date_to.date()} ...")
    rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)
    if rates is None or len(rates) == 0:
        print(f"ERROR: No {label} data returned — {mt5.last_error()}")
        mt5.shutdown()
        sys.exit(1)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low'}, inplace=True)
    print(f"  -> {len(df):,} bars  ({df.index[0]}  to  {df.index[-1]})")
    return df[['Close', 'High', 'Low']]


# ── 2. INDICATORS ──────────────────────────────────────────────────────────────

def calc_bollinger(close: pd.Series, period: int, k: float):
    mid = close.rolling(window=period).mean()
    std = close.rolling(window=period).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    return mid, upper, lower


def calc_sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(window=period).mean()


def build_h4_neutral_series(h4_df: pd.DataFrame, threshold_pips: float) -> pd.Series:
    """Bool series: True where H4 is 'neutral/ranging' (|SMA50-SMA200| <= threshold),
    shifted by one bar so only fully-closed H4 bars are used (no lookahead)."""
    sma_fast = calc_sma(h4_df['Close'], H4_FAST_PERIOD)
    sma_slow = calc_sma(h4_df['Close'], H4_SLOW_PERIOD)
    diff_pips = (sma_fast - sma_slow).abs() / PIP_SIZE
    neutral = diff_pips <= threshold_pips
    return neutral.shift(1)


def align_to_h1(h1_index: pd.DatetimeIndex, series: pd.Series) -> np.ndarray:
    """merge_asof (backward) to bring the latest known H4 state onto each H1 bar."""
    left  = pd.DataFrame({'time': h1_index})
    right = series.dropna().rename('val').reset_index().rename(columns={'index': 'time'})
    if 'time' not in right.columns:
        right.columns = ['time', 'val']
    merged = pd.merge_asof(left, right, on='time', direction='backward')
    return merged['val'].values


# ── 3. ENTRY SIGNAL ELIGIBILITY (independent of SL multiplier) ──────────────

def build_eligibility(close_arr, mid_arr, upper_arr, lower_arr, h4_neutral_arr, session_mask):
    """
    Returns int8 array: +1 = BUY entry eligible (close crossed below lower
    band), -1 = SELL entry eligible (close crossed above upper band),
    0 = no entry. Requires H4 neutral filter + session window.
    """
    below_lower = close_arr < lower_arr
    above_upper = close_arr > upper_arr

    # "crosses" -- this bar triggers the condition, previous bar did not
    cross_below = below_lower & ~np.roll(below_lower, 1)
    cross_above = above_upper & ~np.roll(above_upper, 1)
    cross_below[0] = below_lower[0]
    cross_above[0] = above_upper[0]

    buy_ok  = cross_below & h4_neutral_arr & session_mask
    sell_ok = cross_above & h4_neutral_arr & session_mask

    eligibility = np.zeros(len(close_arr), dtype=np.int8)
    eligibility[buy_ok]  = 1
    eligibility[sell_ok] = -1
    return eligibility


# ── 4. BACKTEST ENGINE ────────────────────────────────────────────────────────

def run_backtest(close, high, low, times_ns, eligibility, band_width_arr,
                  mid_shifted_arr, sl_mult, idx_start, idx_end):
    """
    Walks bars idx_start..idx_end (inclusive). One trade open at a time.
    TP target updates every bar to the previous bar's middle-band value
    (dynamic mean-reversion target, no same-bar lookahead).
    SL is fixed at entry: sl_mult x band-width-at-entry beyond entry price.
    Conservative tie-break: SL wins if both SL and TP are hit same bar.
    Any trade still open at idx_end is force-closed at that bar's close.

    Returns list of (exit_time_ns, pnl_usd) tuples.
    """
    trades = []
    direction   = 0
    entry_price = 0.0
    sl_price    = 0.0

    for i in range(idx_start, idx_end + 1):
        c = close[i]
        h = high[i]
        l = low[i]

        if direction != 0:
            tp_price = mid_shifted_arr[i]
            close_price = None

            if not np.isnan(tp_price):
                if direction == 1:
                    sl_hit = l <= sl_price
                    tp_hit = h >= tp_price
                else:
                    sl_hit = h >= sl_price
                    tp_hit = l <= tp_price

                if sl_hit:
                    close_price = sl_price
                elif tp_hit:
                    close_price = tp_price
            else:
                # tp target unavailable (warmup edge) -- only check SL
                if direction == 1 and l <= sl_price:
                    close_price = sl_price
                elif direction == -1 and h >= sl_price:
                    close_price = sl_price

            if close_price is not None:
                pips = ((close_price - entry_price) if direction == 1
                         else (entry_price - close_price)) / PIP_SIZE
                pnl = pips * PIP_VALUE_USD
                trades.append((times_ns[i], pnl))
                direction = 0

        if direction == 0:
            sig = eligibility[i]
            if sig != 0 and not np.isnan(band_width_arr[i]):
                direction   = 1 if sig == 1 else -1
                entry_price = c
                sl_dist     = sl_mult * band_width_arr[i]
                sl_price    = (c - sl_dist) if direction == 1 else (c + sl_dist)

    if direction != 0:
        c = close[idx_end]
        pips = ((c - entry_price) if direction == 1 else (entry_price - c)) / PIP_SIZE
        pnl = pips * PIP_VALUE_USD
        trades.append((times_ns[idx_end], pnl))

    return trades


# ── 5. METRICS ─────────────────────────────────────────────────────────────────

def compute_metrics(trades: list) -> dict:
    n = len(trades)
    if n == 0:
        return {'trades': 0, 'win_rate': 0, 'pnl': 0, 'max_dd': 100, 'profit_factor': 0,
                'pct_months_profitable': 0}

    pnls = np.array([t[1] for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    win_rate = len(wins) / n * 100
    total_pnl = pnls.sum()

    equity = INITIAL_BALANCE + np.cumsum(pnls)
    equity = np.insert(equity, 0, INITIAL_BALANCE)
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak * 100
    max_dd = dd.max()

    gross_win  = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float('inf')

    times = pd.to_datetime([t[0] for t in trades], utc=True).tz_localize(None)
    month_pnl = pd.Series(pnls, index=times).groupby(times.to_period('M')).sum()
    pct_months_profitable = (month_pnl > 0).sum() / len(month_pnl) * 100 if len(month_pnl) else 0

    return {
        'trades': n,
        'win_rate': round(win_rate, 1),
        'pnl': round(total_pnl, 2),
        'max_dd': round(max_dd, 2),
        'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else float('inf'),
        'pct_months_profitable': round(pct_months_profitable, 1),
    }


def passes_targets(m: dict) -> bool:
    return (
        m['trades'] >= TARGET_MIN_TRADES and
        TARGET_WR_MIN <= m['win_rate'] <= TARGET_WR_MAX and
        m['max_dd'] < TARGET_MAX_DD and
        m['profit_factor'] > TARGET_PF_MIN and
        m['pct_months_profitable'] >= TARGET_MONTHLY_PROFITABLE_PCT
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    connect_mt5()
    h1 = fetch(SYMBOL, mt5.TIMEFRAME_H1, FETCH_FROM, FETCH_TO, 'H1')
    h4 = fetch(SYMBOL, mt5.TIMEFRAME_H4, FETCH_FROM, FETCH_TO, 'H4')
    mt5.shutdown()
    print()

    # ── Precompute H4 neutral-market arrays (3 thresholds) ──
    h4_neutral_arrays = {}
    for thresh in H4_THRESH_PIPS:
        neutral_series = build_h4_neutral_series(h4, thresh)
        h4_neutral_arrays[thresh] = align_to_h1(h1.index, neutral_series).astype(bool)

    # ── Session mask ──
    hours = h1.index.hour.values
    session_mask = (hours >= SESSION_START_HOUR) & (hours < SESSION_END_HOUR)

    close_arr = h1['Close'].values
    high_arr  = h1['High'].values
    low_arr   = h1['Low'].values
    times_ns  = h1.index.values.astype('int64')

    # ── Precompute Bollinger Bands for every (period, k) pair ──
    bb_cache = {}
    for period in BB_PERIODS:
        for k in BB_STDDEVS:
            mid, upper, lower = calc_bollinger(h1['Close'], period, k)
            band_width = (upper - lower).values
            mid_shifted = mid.shift(1).values
            bb_cache[(period, k)] = {
                'mid': mid.values, 'upper': upper.values, 'lower': lower.values,
                'band_width': band_width, 'mid_shifted': mid_shifted,
            }

    # ── Slice indices for train / forward windows ──
    idx = pd.Series(np.arange(len(h1)), index=h1.index)
    train_mask = (h1.index >= TRAIN_START) & (h1.index < TRAIN_END)
    fwd_mask   = (h1.index >= FWD_START)   & (h1.index < FWD_END)
    train_start_idx, train_end_idx = idx[train_mask].iloc[0], idx[train_mask].iloc[-1]
    fwd_start_idx,   fwd_end_idx   = idx[fwd_mask].iloc[0],   idx[fwd_mask].iloc[-1]

    print(f"Training window : {h1.index[train_start_idx]}  to  {h1.index[train_end_idx]}  "
          f"({train_end_idx - train_start_idx + 1:,} bars)")
    print(f"Forward window  : {h1.index[fwd_start_idx]}  to  {h1.index[fwd_end_idx]}  "
          f"({fwd_end_idx - fwd_start_idx + 1:,} bars)\n")

    grid = list(product(BB_PERIODS, BB_STDDEVS, H4_THRESH_PIPS, SL_MULTIPLIERS))
    print(f"Total grid combinations: {len(grid):,}\n")

    passing = []
    t0 = _time.time()
    eligibility_cache = {}

    for n_done, (period, k, h4_thresh, sl_mult) in enumerate(grid, 1):
        elig_key = (period, k, h4_thresh)
        elig = eligibility_cache.get(elig_key)
        bb = bb_cache[(period, k)]
        if elig is None:
            h4_arr = h4_neutral_arrays[h4_thresh]
            elig = build_eligibility(close_arr, bb['mid'], bb['upper'], bb['lower'], h4_arr, session_mask)
            eligibility_cache[elig_key] = elig

        trades = run_backtest(close_arr, high_arr, low_arr, times_ns, elig,
                               bb['band_width'], bb['mid_shifted'], sl_mult,
                               train_start_idx, train_end_idx)
        metrics = compute_metrics(trades)

        if passes_targets(metrics):
            passing.append({
                'bb_period': period, 'bb_std': k, 'h4_thresh_pips': h4_thresh, 'sl_mult': sl_mult,
                **{f'train_{key}': v for key, v in metrics.items()},
            })

        if n_done % 20 == 0 or n_done == len(grid):
            elapsed = _time.time() - t0
            print(f"  [{n_done:,}/{len(grid):,}]  elapsed={elapsed:.0f}s  passing so far={len(passing)}", flush=True)

    print(f"\nGrid search complete. {len(passing)} / {len(grid)} configs passed all training targets.\n", flush=True)

    if not passing:
        print("No configuration passed all training targets. Reporting honestly: "
              "the search found no Bollinger mean-reversion candidate meeting win rate, "
              "drawdown, profit factor, trade count, and monthly-consistency targets "
              "simultaneously on 2020-2022 H1 training data.")
        return

    results_df = pd.DataFrame(passing)
    results_df.to_csv('data/h1_bollinger_search_passing_configs.csv', index=False)
    print(f"All {len(passing)} passing configs saved to data/h1_bollinger_search_passing_configs.csv\n")

    # ── Rank by profit factor desc, max DD asc tiebreak ──
    results_df = results_df.sort_values(by=['train_profit_factor', 'train_max_dd'],
                                         ascending=[False, True])
    top5 = results_df.head(TOP_N).to_dict('records')

    print(f"Top {TOP_N} configs (ranked by training profit factor, DD tiebreak):")
    for i, cfg in enumerate(top5, 1):
        print(f"  #{i}  BB({cfg['bb_period']},{cfg['bb_std']})  H4_neutral<={cfg['h4_thresh_pips']}p  "
              f"SL={cfg['sl_mult']}x  | train PF={cfg['train_profit_factor']}  WR={cfg['train_win_rate']}%  "
              f"P&L=${cfg['train_pnl']:+,.2f}  DD={cfg['train_max_dd']}%  trades={cfg['train_trades']}")
    print()

    # ── Forward test top 5, ONCE, unmodified ──
    print("=" * 100)
    print("FORWARD TEST -- running top 5 configs on 2022-07-01 to 2024-12-31 (never seen during optimization)")
    print("=" * 100)

    final_rows = []
    for cfg in top5:
        elig_key = (cfg['bb_period'], cfg['bb_std'], cfg['h4_thresh_pips'])
        elig = eligibility_cache[elig_key]
        bb = bb_cache[(cfg['bb_period'], cfg['bb_std'])]
        trades_fwd = run_backtest(close_arr, high_arr, low_arr, times_ns, elig,
                                   bb['band_width'], bb['mid_shifted'], cfg['sl_mult'],
                                   fwd_start_idx, fwd_end_idx)
        fwd_metrics = compute_metrics(trades_fwd)

        label = f"BB({cfg['bb_period']},{cfg['bb_std']}) H4<={cfg['h4_thresh_pips']}p SL{cfg['sl_mult']}x"

        final_rows.append({
            'config': label,
            'train_wr': cfg['train_win_rate'], 'train_dd': cfg['train_max_dd'], 'train_pf': cfg['train_profit_factor'],
            'train_pnl': cfg['train_pnl'], 'train_trades': cfg['train_trades'],
            'fwd_wr': fwd_metrics['win_rate'], 'fwd_dd': fwd_metrics['max_dd'], 'fwd_pf': fwd_metrics['profit_factor'],
            'fwd_pnl': fwd_metrics['pnl'], 'fwd_trades': fwd_metrics['trades'],
        })

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv('data/h1_bollinger_forward_test_results.csv', index=False)

    def verdict(row) -> str:
        if row['fwd_trades'] < 15:
            return 'INSUFFICIENT FWD DATA'
        pf_drop = (row['train_pf'] - row['fwd_pf']) if row['train_pf'] != float('inf') else 0
        if row['fwd_pnl'] <= 0 or row['fwd_pf'] < 1.0:
            return 'FAILS -- degrades to unprofitable'
        if pf_drop > 0.5 or row['fwd_dd'] > row['train_dd'] + 3 or abs(row['fwd_wr'] - row['train_wr']) > 10:
            return 'WEAK -- significant degradation'
        return 'ROBUST -- holds up on unseen data'

    final_df['verdict'] = final_df.apply(verdict, axis=1)

    print()
    print(f"{'Config':<45} {'TrWR':>6} {'TrDD':>6} {'TrPF':>6} {'FwWR':>6} {'FwDD':>6} {'FwPF':>6}  Verdict")
    print("-" * 120)
    for _, r in final_df.iterrows():
        print(f"{r['config']:<45} {r['train_wr']:>5.1f}% {r['train_dd']:>5.1f}% {r['train_pf']:>6.2f} "
              f"{r['fwd_wr']:>5.1f}% {r['fwd_dd']:>5.1f}% {r['fwd_pf']:>6.2f}  {r['verdict']}")
    print()

    n_robust = (final_df['verdict'] == 'ROBUST -- holds up on unseen data').sum()
    print(f"Honest assessment: {n_robust} / {TOP_N} top-training configs showed genuine robustness "
          f"on forward-test (unseen) data.")
    if n_robust == 0:
        print("ALL configurations degraded on unseen data. This indicates the training-data results "
              "were likely overfit to 2020-2022 conditions rather than reflecting a real, durable edge.")

    print(f"\nFull results saved:")
    print(f"  data/h1_bollinger_search_passing_configs.csv")
    print(f"  data/h1_bollinger_forward_test_results.csv")
    print("\nDone!")


if __name__ == '__main__':
    main()
