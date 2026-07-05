"""
Forex Bot - M15 EURUSD Strategy Search: Train/Forward-Test Walk-Forward Validation

Systematic grid search for a profitable EURUSD M15 strategy.  Background
research only -- not for live deployment.

Data split (strict 50/50, no leakage):
  Training     : 2020-01-01 to 2022-06-30  (~2.5 years)
  Forward test : 2022-07-01 to 2024-12-31  (~2.5 years, untouched until the end)

Strategy framework:
  Entry      : EMA fast/slow crossover on M15
  Trend      : EMA(50) or SMA(50) direction on H1 or H4 (4 combinations)
  RSI filter : period + threshold pair (entry only allowed in the
               direction confirmed by RSI, same convention as every prior
               episode: RSI < buy_threshold for BUY, RSI > sell_threshold
               for SELL)
  Session    : London + NY combined, 08:00-22:00 UTC (entries only;
               open trades are still monitored for SL/TP outside this window)
  Exit       : fixed SL/TP, TP = 2 x SL (2:1 RR)

Grid (after dropping invalid fast>=slow EMA pairs):
  EMA fast   : 5, 8, 10, 12, 20
  EMA slow   : 20, 50, 100, 200          (19 valid fast<slow pairs)
  Trend TF   : H1, H4                    (2)
  Trend type : EMA(50), SMA(50)          (2)
  RSI period : 9, 14, 21                 (3)
  RSI thresh : 40/60, 45/55, 30/70       (3)
  SL pips    : 20, 30, 40, 50            (4)
  Total combinations: 19 * 2 * 2 * 3 * 3 * 4 = 2,736

Optimization targets (training data ONLY):
  40% <= win rate <= 46%
  max drawdown < 4%
  profit factor > 1.2
  >= 80 trades
  profitable in >= 60% of training months

Top 5 training-passing configs (ranked by profit factor, tie-break P&L)
are forward-tested exactly once, unmodified, on the unseen 2022-2024 window.

Requirements:
  MetaTrader 5 must be OPEN and LOGGED IN before running this script.
"""

import sys
import time as _time
from datetime import datetime, timedelta, timezone
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
FWD_END     = pd.Timestamp('2025-01-01', tz='UTC')    # exclusive

SESSION_START_HOUR = 8
SESSION_END_HOUR   = 22   # exclusive

# ── GRID ──────────────────────────────────────────────────────────────────────

EMA_FAST_VALUES = [5, 8, 10, 12, 20]
EMA_SLOW_VALUES = [20, 50, 100, 200]
TREND_TFS       = ['H1', 'H4']
TREND_TYPES     = ['EMA', 'SMA']
RSI_PERIODS     = [9, 14, 21]
RSI_THRESHOLDS  = [(40, 60), (45, 55), (30, 70)]   # (sell_threshold, buy_threshold)
SL_PIPS_VALUES  = [20, 30, 40, 50]
RR              = 2.0   # TP = RR * SL

TREND_MA_PERIOD = 50

# ── TARGETS (training data only) ─────────────────────────────────────────────

TARGET_WR_MIN     = 40.0
TARGET_WR_MAX     = 46.0
TARGET_MAX_DD     = 4.0
TARGET_PF_MIN     = 1.2
TARGET_MIN_TRADES = 80
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

def calc_rsi(close: pd.Series, period: int) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def calc_sma(close: pd.Series, period: int) -> pd.Series:
    return close.rolling(window=period).mean()


def build_trend_series(htf_df: pd.DataFrame, ma_type: str, period: int) -> pd.Series:
    """Bullish(1)/Bearish(-1) trend on the higher timeframe, shifted by one
    bar so only fully-closed HTF bars are used (no lookahead)."""
    ma = calc_ema(htf_df['Close'], period) if ma_type == 'EMA' else calc_sma(htf_df['Close'], period)
    trend = np.where(htf_df['Close'] > ma, 1, -1)
    trend_series = pd.Series(trend, index=htf_df.index).shift(1)
    return trend_series


def align_to_m15(m15_index: pd.DatetimeIndex, trend_series: pd.Series) -> np.ndarray:
    """merge_asof (backward) to bring the latest known HTF trend onto each M15 bar."""
    left  = pd.DataFrame({'time': m15_index})
    right = trend_series.dropna().rename('trend').reset_index().rename(columns={'index': 'time'})
    if 'time' not in right.columns:
        right.columns = ['time', 'trend']
    merged = pd.merge_asof(left, right, on='time', direction='backward')
    return merged['trend'].values


# ── 3. ENTRY SIGNAL ELIGIBILITY (independent of SL) ──────────────────────────

def build_eligibility(close_arr, ema_fast_dict, ema_slow_dict, fast_p, slow_p,
                       trend_arr, rsi_arr, sell_th, buy_th, session_mask):
    """
    Returns int8 array: +1 = BUY entry eligible at this bar, -1 = SELL eligible,
    0 = no entry. Conditions: EMA crossover this bar + RSI confirmation +
    trend alignment + within session window.
    """
    fast = ema_fast_dict[fast_p]
    slow = ema_slow_dict[slow_p]
    position  = (fast > slow).astype(np.int8)
    crossover = np.diff(position, prepend=position[0])

    buy_cross  = crossover == 1
    sell_cross = crossover == -1

    buy_ok  = buy_cross  & (rsi_arr < buy_th)  & (trend_arr == 1)  & session_mask
    sell_ok = sell_cross & (rsi_arr > sell_th) & (trend_arr == -1) & session_mask

    eligibility = np.zeros(len(close_arr), dtype=np.int8)
    eligibility[buy_ok]  = 1
    eligibility[sell_ok] = -1
    return eligibility


# ── 4. BACKTEST ENGINE (numpy, sequential state machine) ─────────────────────

def run_backtest(close, high, low, times_ns, eligibility, sl_pips, tp_pips,
                  idx_start, idx_end):
    """
    Walks bars idx_start..idx_end (inclusive). One trade open at a time.
    Conservative tie-break: SL wins if both SL and TP are hit same bar.
    Any trade still open at idx_end is force-closed at that bar's close
    (mark-to-market) so training/forward windows stay cleanly separated.

    Returns list of (exit_time_ns, pnl_usd) tuples.
    """
    sl_dist = sl_pips * PIP_SIZE
    tp_dist = tp_pips * PIP_SIZE

    trades = []
    direction   = 0      # 0 = flat, 1 = long, -1 = short
    entry_price = 0.0
    sl_price    = 0.0
    tp_price    = 0.0

    for i in range(idx_start, idx_end + 1):
        c = close[i]
        h = high[i]
        l = low[i]

        if direction != 0:
            close_price = None
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

            if close_price is not None:
                pips = ((close_price - entry_price) if direction == 1
                         else (entry_price - close_price)) / PIP_SIZE
                pnl = pips * PIP_VALUE_USD
                trades.append((times_ns[i], pnl))
                direction = 0

        if direction == 0:
            sig = eligibility[i]
            if sig != 0:
                direction   = 1 if sig == 1 else -1
                entry_price = c
                if direction == 1:
                    sl_price = c - sl_dist
                    tp_price = c + tp_dist
                else:
                    sl_price = c + sl_dist
                    tp_price = c - tp_dist

    # Force-close any still-open trade at the window boundary
    if direction != 0:
        c = close[idx_end]
        pips = ((c - entry_price) if direction == 1 else (entry_price - c)) / PIP_SIZE
        pnl = pips * PIP_VALUE_USD
        trades.append((times_ns[idx_end], pnl))

    return trades


# ── 5. METRICS ─────────────────────────────────────────────────────────────────

def compute_metrics(trades: list, require_monthly: bool = True) -> dict:
    n = len(trades)
    if n == 0:
        return {'trades': 0, 'win_rate': 0, 'pnl': 0, 'max_dd': 100, 'profit_factor': 0,
                'pct_months_profitable': 0}

    pnls  = np.array([t[1] for t in trades])
    wins  = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    win_rate = len(wins) / n * 100
    total_pnl = pnls.sum()

    equity = INITIAL_BALANCE + np.cumsum(pnls)
    equity = np.insert(equity, 0, INITIAL_BALANCE)
    peak = np.maximum.accumulate(equity)
    dd   = (peak - equity) / peak * 100
    max_dd = dd.max()

    gross_win  = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float('inf')

    pct_months_profitable = None
    if require_monthly:
        times = pd.to_datetime([t[0] for t in trades], utc=True)
        month_pnl = pd.Series(pnls, index=times).groupby(times.to_period('M')).sum()
        pct_months_profitable = (month_pnl > 0).sum() / len(month_pnl) * 100 if len(month_pnl) else 0

    return {
        'trades': n,
        'win_rate': round(win_rate, 1),
        'pnl': round(total_pnl, 2),
        'max_dd': round(max_dd, 2),
        'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else float('inf'),
        'pct_months_profitable': round(pct_months_profitable, 1) if pct_months_profitable is not None else None,
    }


def passes_targets(m: dict) -> bool:
    return (
        m['trades'] >= TARGET_MIN_TRADES and
        TARGET_WR_MIN <= m['win_rate'] <= TARGET_WR_MAX and
        m['max_dd'] < TARGET_MAX_DD and
        m['profit_factor'] > TARGET_PF_MIN and
        m['pct_months_profitable'] is not None and
        m['pct_months_profitable'] >= TARGET_MONTHLY_PROFITABLE_PCT
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    connect_mt5()
    m15 = fetch(SYMBOL, mt5.TIMEFRAME_M15, FETCH_FROM, FETCH_TO, 'M15')
    h1  = fetch(SYMBOL, mt5.TIMEFRAME_H1,  FETCH_FROM, FETCH_TO, 'H1')
    h4  = fetch(SYMBOL, mt5.TIMEFRAME_H4,  FETCH_FROM, FETCH_TO, 'H4')
    mt5.shutdown()
    print()

    # ── Precompute trend arrays (4 combinations: TF x MA type) ──
    trend_arrays = {}
    for tf_label, htf_df in [('H1', h1), ('H4', h4)]:
        for ma_type in TREND_TYPES:
            trend_series = build_trend_series(htf_df, ma_type, TREND_MA_PERIOD)
            trend_arrays[(tf_label, ma_type)] = align_to_m15(m15.index, trend_series)

    # ── Precompute RSI arrays ──
    rsi_arrays = {p: calc_rsi(m15['Close'], p).values for p in RSI_PERIODS}

    # ── Precompute EMA arrays for every fast/slow period needed ──
    all_ema_periods = sorted(set(EMA_FAST_VALUES) | set(EMA_SLOW_VALUES))
    ema_arrays = {p: calc_ema(m15['Close'], p).values for p in all_ema_periods}

    # ── Session mask ──
    hours = m15.index.hour.values
    session_mask = (hours >= SESSION_START_HOUR) & (hours < SESSION_END_HOUR)

    # ── Arrays for backtest ──
    close_arr = m15['Close'].values
    high_arr  = m15['High'].values
    low_arr   = m15['Low'].values
    times_ns  = m15.index.values.astype('int64')

    # ── Slice indices for train / forward windows ──
    idx = pd.Series(np.arange(len(m15)), index=m15.index)
    train_mask = (m15.index >= TRAIN_START) & (m15.index < TRAIN_END)
    fwd_mask   = (m15.index >= FWD_START)   & (m15.index < FWD_END)
    train_start_idx, train_end_idx = idx[train_mask].iloc[0], idx[train_mask].iloc[-1]
    fwd_start_idx,   fwd_end_idx   = idx[fwd_mask].iloc[0],   idx[fwd_mask].iloc[-1]

    print(f"Training window : {m15.index[train_start_idx]}  to  {m15.index[train_end_idx]}  "
          f"({train_end_idx - train_start_idx + 1:,} bars)")
    print(f"Forward window  : {m15.index[fwd_start_idx]}  to  {m15.index[fwd_end_idx]}  "
          f"({fwd_end_idx - fwd_start_idx + 1:,} bars)\n")

    # ── Valid EMA fast/slow pairs ──
    ema_pairs = [(f, s) for f in EMA_FAST_VALUES for s in EMA_SLOW_VALUES if f < s]

    grid = list(product(ema_pairs, TREND_TFS, TREND_TYPES, RSI_PERIODS, RSI_THRESHOLDS, SL_PIPS_VALUES))
    print(f"Total grid combinations: {len(grid):,}\n")

    passing = []
    t0 = _time.time()
    eligibility_cache = {}

    for n_done, ((fast_p, slow_p), trend_tf, trend_type, rsi_p, (sell_th, buy_th), sl_pips) in enumerate(grid, 1):
        elig_key = (fast_p, slow_p, trend_tf, trend_type, rsi_p, sell_th, buy_th)
        elig = eligibility_cache.get(elig_key)
        if elig is None:
            trend_arr = trend_arrays[(trend_tf, trend_type)]
            rsi_arr   = rsi_arrays[rsi_p]
            elig = build_eligibility(close_arr, ema_arrays, ema_arrays, fast_p, slow_p,
                                      trend_arr, rsi_arr, sell_th, buy_th, session_mask)
            eligibility_cache[elig_key] = elig

        tp_pips = sl_pips * RR
        trades = run_backtest(close_arr, high_arr, low_arr, times_ns, elig,
                               sl_pips, tp_pips, train_start_idx, train_end_idx)
        metrics = compute_metrics(trades, require_monthly=True)

        if passes_targets(metrics):
            passing.append({
                'fast': fast_p, 'slow': slow_p, 'trend_tf': trend_tf, 'trend_type': trend_type,
                'rsi_period': rsi_p, 'sell_th': sell_th, 'buy_th': buy_th,
                'sl_pips': sl_pips, 'tp_pips': tp_pips,
                **{f'train_{k}': v for k, v in metrics.items()},
            })

        if n_done % 300 == 0 or n_done == len(grid):
            elapsed = _time.time() - t0
            print(f"  [{n_done:,}/{len(grid):,}]  elapsed={elapsed:.0f}s  "
                  f"passing so far={len(passing)}")

    print(f"\nGrid search complete. {len(passing)} / {len(grid)} configs passed all training targets.\n")

    if not passing:
        print("No configuration passed all training targets. Reporting honestly: "
              "the search found no candidate meeting win rate, drawdown, profit factor, "
              "trade count, and monthly-consistency targets simultaneously on 2020-2022 training data.")
        return

    results_df = pd.DataFrame(passing)
    results_df.to_csv('data/m15_search_passing_configs.csv', index=False)
    print(f"All {len(passing)} passing configs saved to data/m15_search_passing_configs.csv\n")

    # ── Rank by profit factor (tie-break: P&L) ──
    results_df = results_df.sort_values(by=['train_profit_factor', 'train_pnl'], ascending=False)
    top5 = results_df.head(TOP_N).to_dict('records')

    print(f"Top {TOP_N} configs (ranked by training profit factor):")
    for i, cfg in enumerate(top5, 1):
        print(f"  #{i}  EMA {cfg['fast']}/{cfg['slow']}  trend={cfg['trend_tf']}-{cfg['trend_type']}{TREND_MA_PERIOD}  "
              f"RSI{cfg['rsi_period']} {cfg['sell_th']}/{cfg['buy_th']}  SL={cfg['sl_pips']}p  "
              f"| train PF={cfg['train_profit_factor']}  WR={cfg['train_win_rate']}%  "
              f"P&L=${cfg['train_pnl']:+,.2f}  DD={cfg['train_max_dd']}%")
    print()

    # ── Forward test top 5, ONCE, unmodified ──
    print("=" * 100)
    print("FORWARD TEST -- running top 5 configs on 2022-07-01 to 2024-12-31 (never seen during optimization)")
    print("=" * 100)

    final_rows = []
    for i, cfg in enumerate(top5, 1):
        elig_key = (cfg['fast'], cfg['slow'], cfg['trend_tf'], cfg['trend_type'],
                    cfg['rsi_period'], cfg['sell_th'], cfg['buy_th'])
        elig = eligibility_cache[elig_key]
        trades_fwd = run_backtest(close_arr, high_arr, low_arr, times_ns, elig,
                                   cfg['sl_pips'], cfg['tp_pips'], fwd_start_idx, fwd_end_idx)
        fwd_metrics = compute_metrics(trades_fwd, require_monthly=True)

        label = (f"EMA{cfg['fast']}/{cfg['slow']} {cfg['trend_tf']}-{cfg['trend_type']}{TREND_MA_PERIOD} "
                 f"RSI{cfg['rsi_period']}({cfg['sell_th']}/{cfg['buy_th']}) SL{cfg['sl_pips']}p")

        final_rows.append({
            'config': label,
            'train_wr': cfg['train_win_rate'], 'train_dd': cfg['train_max_dd'], 'train_pf': cfg['train_profit_factor'],
            'train_pnl': cfg['train_pnl'], 'train_trades': cfg['train_trades'],
            'fwd_wr': fwd_metrics['win_rate'], 'fwd_dd': fwd_metrics['max_dd'], 'fwd_pf': fwd_metrics['profit_factor'],
            'fwd_pnl': fwd_metrics['pnl'], 'fwd_trades': fwd_metrics['trades'],
        })

    final_df = pd.DataFrame(final_rows)
    final_df.to_csv('data/m15_search_forward_test_results.csv', index=False)

    # ── Verdict per config ──
    def verdict(row) -> str:
        if row['fwd_trades'] < 20:
            return 'INSUFFICIENT FWD DATA'
        pf_drop = row['train_pf'] - row['fwd_pf'] if row['train_pf'] != float('inf') else 0
        if row['fwd_pnl'] <= 0 or row['fwd_pf'] < 1.0:
            return 'FAILS -- degrades to unprofitable'
        if pf_drop > 0.5 or row['fwd_dd'] > row['train_dd'] + 3 or abs(row['fwd_wr'] - row['train_wr']) > 10:
            return 'WEAK -- significant degradation'
        return 'ROBUST -- holds up on unseen data'

    final_df['verdict'] = final_df.apply(verdict, axis=1)

    print()
    print(f"{'Config':<55} {'TrWR':>6} {'TrDD':>6} {'TrPF':>6} {'FwWR':>6} {'FwDD':>6} {'FwPF':>6}  Verdict")
    print("-" * 130)
    for _, r in final_df.iterrows():
        print(f"{r['config']:<55} {r['train_wr']:>5.1f}% {r['train_dd']:>5.1f}% {r['train_pf']:>6.2f} "
              f"{r['fwd_wr']:>5.1f}% {r['fwd_dd']:>5.1f}% {r['fwd_pf']:>6.2f}  {r['verdict']}")
    print()

    n_robust = (final_df['verdict'] == 'ROBUST -- holds up on unseen data').sum()
    print(f"Honest assessment: {n_robust} / {TOP_N} top-training configs showed genuine robustness "
          f"on forward-test (unseen) data.")
    if n_robust == 0:
        print("ALL configurations degraded on unseen data. This indicates the training-data results "
              "were likely overfit to 2020-2022 conditions rather than reflecting a real, durable edge.")

    print(f"\nFull results saved:")
    print(f"  data/m15_search_passing_configs.csv")
    print(f"  data/m15_search_forward_test_results.csv")
    print("\nDone!")


if __name__ == '__main__':
    main()
