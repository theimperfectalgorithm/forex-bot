"""
Forex Bot - H1 ATR(14) 3.0x/6.0x SMA Strategy: Final Walk-Forward Validation

Same rigorous train/forward methodology as the M15 EMA search
(0/2,736 passed forward test) and the Bollinger Band search (0/108 passed
training). This is a FIXED configuration -- no optimization, no grid --
run once on training, once on forward, unmodified, to check whether the
earlier +$856.73 / 3-year result holds up under the same scrutiny.

Strategy (fixed):
  Entry      : H1 50/200 SMA crossover
  RSI filter : RSI(14) < 60 for BUY crossovers, RSI(14) > 40 for SELL
               crossovers (original episode convention -- this is the
               actual filter used to produce the earlier +$856.73 result,
               re-validated here under strict train/forward separation)
  SL         : 3.0 x ATR(14) on H1, fixed at entry
  TP         : 6.0 x ATR(14) on H1, fixed at entry (2:1 RR)
  Session    : entries only 08:00-22:00 UTC (London+NY combined)
  Direction  : trade only in the direction the crossover indicates
               (inherent to the entry logic -- no contrarian trades)

Data split (strict 50/50, no leakage -- run as two separate backtests,
each force-closing any open trade at its own window boundary):
  Training     : 2020-01-01 to 2022-06-30
  Forward test : 2022-07-01 to 2024-12-31

Sub-period check: one continuous backtest over the full 2020-2024 range,
then completed trades bucketed by entry date into 6 equal ~10-month windows
(same bucketing convention as the earlier 6-period H1 ATR walk-forward).

Requirements:
  MetaTrader 5 must be OPEN and LOGGED IN before running this script.
"""

import sys
from datetime import datetime, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

# ── SETTINGS ──────────────────────────────────────────────────────────────────

SYMBOL        = 'EURUSD'
PIP_SIZE      = 0.0001
PIP_VALUE_USD = 1.00
INITIAL_BALANCE = 10_000.00

FETCH_FROM = datetime(2019, 9, 1, tzinfo=timezone.utc)   # buffer for indicator warmup
FETCH_TO   = datetime(2025, 1, 1, tzinfo=timezone.utc)

TRAIN_START = pd.Timestamp('2020-01-01', tz='UTC')
TRAIN_END   = pd.Timestamp('2022-07-01', tz='UTC')   # exclusive
FWD_START   = pd.Timestamp('2022-07-01', tz='UTC')
FWD_END     = pd.Timestamp('2025-01-01', tz='UTC')   # exclusive

SESSION_START_HOUR = 8
SESSION_END_HOUR   = 22   # exclusive

RSI_PERIOD   = 14
RSI_BUY_THRESHOLD  = 60   # BUY allowed when RSI < this (original episode convention)
RSI_SELL_THRESHOLD = 40   # SELL allowed when RSI > this

ATR_PERIOD  = 14
SL_ATR_MULT = 3.0
TP_ATR_MULT = 6.0

N_SUBPERIODS = 6


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


def fetch(symbol, timeframe, date_from, date_to, label) -> pd.DataFrame:
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


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def build_eligibility(close_arr, sma50_arr, sma200_arr, rsi_arr, session_mask):
    position  = (sma50_arr > sma200_arr).astype(np.int8)
    crossover = np.diff(position, prepend=position[0])

    buy_ok  = (crossover == 1)  & (rsi_arr < RSI_BUY_THRESHOLD)  & session_mask
    sell_ok = (crossover == -1) & (rsi_arr > RSI_SELL_THRESHOLD) & session_mask

    elig = np.zeros(len(close_arr), dtype=np.int8)
    elig[buy_ok]  = 1
    elig[sell_ok] = -1
    return elig


# ── 3. BACKTEST ENGINE ────────────────────────────────────────────────────────

def run_backtest(close, high, low, times_ns, eligibility, atr_arr, idx_start, idx_end):
    """
    One trade at a time. SL/TP fixed at entry from ATR(14) at that bar.
    Conservative tie-break: SL wins if both hit same bar.
    Any trade open at idx_end is force-closed at that bar's close.
    Returns list of (entry_time_ns, exit_time_ns, pnl_usd, sl_pips, tp_pips).
    """
    trades = []
    direction   = 0
    entry_price = 0.0
    entry_time  = 0
    sl_price    = 0.0
    tp_price    = 0.0
    sl_pips     = 0.0
    tp_pips     = 0.0

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
                trades.append((entry_time, times_ns[i], pnl, sl_pips, tp_pips))
                direction = 0

        if direction == 0:
            sig = eligibility[i]
            if sig != 0:
                atr = atr_arr[i]
                if not np.isnan(atr):
                    direction   = 1 if sig == 1 else -1
                    entry_price = c
                    entry_time  = times_ns[i]
                    sl_dist     = atr * SL_ATR_MULT
                    tp_dist     = atr * TP_ATR_MULT
                    sl_pips     = sl_dist / PIP_SIZE
                    tp_pips     = tp_dist / PIP_SIZE
                    if direction == 1:
                        sl_price = c - sl_dist
                        tp_price = c + tp_dist
                    else:
                        sl_price = c + sl_dist
                        tp_price = c - tp_dist

    if direction != 0:
        c = close[idx_end]
        pips = ((c - entry_price) if direction == 1 else (entry_price - c)) / PIP_SIZE
        pnl = pips * PIP_VALUE_USD
        trades.append((entry_time, times_ns[idx_end], pnl, sl_pips, tp_pips))

    return trades


# ── 4. METRICS ─────────────────────────────────────────────────────────────────

def compute_metrics(trades: list) -> dict:
    n = len(trades)
    if n == 0:
        return {'trades': 0, 'win_rate': 0, 'pnl': 0, 'max_dd': 0, 'profit_factor': 0,
                'pct_months_profitable': 0, 'avg_sl_pips': 0, 'avg_tp_pips': 0}

    pnls    = np.array([t[2] for t in trades])
    sl_pips = np.array([t[3] for t in trades])
    tp_pips = np.array([t[4] for t in trades])
    wins    = pnls[pnls > 0]
    losses  = pnls[pnls <= 0]

    win_rate  = len(wins) / n * 100
    total_pnl = pnls.sum()

    equity = INITIAL_BALANCE + np.cumsum(pnls)
    equity = np.insert(equity, 0, INITIAL_BALANCE)
    peak   = np.maximum.accumulate(equity)
    dd     = (peak - equity) / peak * 100
    max_dd = dd.max()

    gross_win  = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float('inf')

    exit_times = pd.to_datetime([t[1] for t in trades], utc=True).tz_localize(None)
    month_pnl = pd.Series(pnls, index=exit_times).groupby(exit_times.to_period('M')).sum()
    pct_months_profitable = (month_pnl > 0).sum() / len(month_pnl) * 100 if len(month_pnl) else 0

    return {
        'trades': n,
        'win_rate': round(win_rate, 1),
        'pnl': round(total_pnl, 2),
        'max_dd': round(max_dd, 2),
        'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else float('inf'),
        'pct_months_profitable': round(pct_months_profitable, 1),
        'avg_sl_pips': round(sl_pips.mean(), 1),
        'avg_tp_pips': round(tp_pips.mean(), 1),
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    connect_mt5()
    h1 = fetch(SYMBOL, mt5.TIMEFRAME_H1, FETCH_FROM, FETCH_TO, 'H1')
    mt5.shutdown()
    print()

    close_arr = h1['Close'].values
    high_arr  = h1['High'].values
    low_arr   = h1['Low'].values
    times_ns  = h1.index.values.astype('int64')

    sma50  = h1['Close'].rolling(50).mean().values
    sma200 = h1['Close'].rolling(200).mean().values
    rsi    = calc_rsi(h1['Close'], RSI_PERIOD).values
    atr    = calc_atr(h1['High'], h1['Low'], h1['Close'], ATR_PERIOD).values

    hours = h1.index.hour.values
    session_mask = (hours >= SESSION_START_HOUR) & (hours < SESSION_END_HOUR)

    eligibility = build_eligibility(close_arr, sma50, sma200, rsi, session_mask)

    idx = pd.Series(np.arange(len(h1)), index=h1.index)
    train_mask = (h1.index >= TRAIN_START) & (h1.index < TRAIN_END)
    fwd_mask   = (h1.index >= FWD_START)   & (h1.index < FWD_END)
    train_start_idx, train_end_idx = idx[train_mask].iloc[0], idx[train_mask].iloc[-1]
    fwd_start_idx,   fwd_end_idx   = idx[fwd_mask].iloc[0],   idx[fwd_mask].iloc[-1]

    print(f"Training window : {h1.index[train_start_idx]}  to  {h1.index[train_end_idx]}  "
          f"({train_end_idx - train_start_idx + 1:,} bars)")
    print(f"Forward window  : {h1.index[fwd_start_idx]}  to  {h1.index[fwd_end_idx]}  "
          f"({fwd_end_idx - fwd_start_idx + 1:,} bars)\n")

    # ── Separate, leakage-free backtests for train and forward ──
    train_trades = run_backtest(close_arr, high_arr, low_arr, times_ns, eligibility, atr,
                                 train_start_idx, train_end_idx)
    fwd_trades   = run_backtest(close_arr, high_arr, low_arr, times_ns, eligibility, atr,
                                 fwd_start_idx, fwd_end_idx)

    train_m = compute_metrics(train_trades)
    fwd_m   = compute_metrics(fwd_trades)

    w = 64
    print("=" * w)
    print("   H1 ATR(14) 3.0x/6.0x  --  FINAL WALK-FORWARD VALIDATION")
    print("   50/200 SMA crossover + RSI(<60 BUY / >40 SELL) + ATR SL/TP  |  EURUSD H1")
    print("=" * w)
    print(f"  {'Metric':<26}  {'Training (20-22)':>16}  {'Forward (22-24)':>16}")
    print("-" * w)
    print(f"  {'Total Trades':<26}  {train_m['trades']:>16}  {fwd_m['trades']:>16}")
    print(f"  {'Win Rate':<26}  {train_m['win_rate']:>15.1f}%  {fwd_m['win_rate']:>15.1f}%")
    print(f"  {'Total P&L':<26}  ${train_m['pnl']:>+15,.2f}  ${fwd_m['pnl']:>+15,.2f}")
    print(f"  {'Max Drawdown':<26}  {train_m['max_dd']:>15.2f}%  {fwd_m['max_dd']:>15.2f}%")
    print(f"  {'Profit Factor':<26}  {train_m['profit_factor']:>16}  {fwd_m['profit_factor']:>16}")
    print(f"  {'Profitable Months %':<26}  {train_m['pct_months_profitable']:>15.1f}%  {fwd_m['pct_months_profitable']:>15.1f}%")
    print(f"  {'Avg SL size (pips)':<26}  {train_m['avg_sl_pips']:>16.1f}  {fwd_m['avg_sl_pips']:>16.1f}")
    print(f"  {'Avg TP size (pips)':<26}  {train_m['avg_tp_pips']:>16.1f}  {fwd_m['avg_tp_pips']:>16.1f}")
    print("=" * w)
    print()

    # ── 6-subperiod walk-forward breakdown (continuous backtest, bucketed) ──
    full_start_idx = idx[(h1.index >= TRAIN_START) & (h1.index < FWD_END)].iloc[0]
    full_end_idx   = idx[(h1.index >= TRAIN_START) & (h1.index < FWD_END)].iloc[-1]
    full_trades = run_backtest(close_arr, high_arr, low_arr, times_ns, eligibility, atr,
                                full_start_idx, full_end_idx)

    boundaries = pd.date_range(TRAIN_START, FWD_END, periods=N_SUBPERIODS + 1, tz='UTC')
    entry_times = pd.to_datetime([t[0] for t in full_trades], utc=True)
    pnls_all    = np.array([t[2] for t in full_trades])

    print("=" * 64)
    print(f"   6-SUBPERIOD WALK-FORWARD BREAKDOWN  (~10 months each, continuous backtest)")
    print("=" * 64)
    print(f"  {'Period':<28}  {'Trades':>7}  {'Win Rate':>9}  {'P&L':>12}")
    print("-" * 64)

    sub_results = []
    for i in range(N_SUBPERIODS):
        p_start, p_end = boundaries[i], boundaries[i + 1]
        mask = (entry_times >= p_start) & (entry_times < p_end)
        chunk_pnls = pnls_all[mask]
        n = len(chunk_pnls)
        wr = (chunk_pnls > 0).sum() / n * 100 if n else 0
        pnl = chunk_pnls.sum() if n else 0.0
        label = f"{p_start.date()} to {(p_end - pd.Timedelta(days=1)).date()}"
        flag = '  <-- NEGATIVE' if pnl < 0 else ''
        print(f"  {label:<28}  {n:>7}  {wr:>8.1f}%  ${pnl:>+10,.2f}{flag}")
        sub_results.append({'period': label, 'trades': n, 'win_rate': round(wr, 1), 'pnl': round(pnl, 2)})

    print("-" * 64)
    total_n = len(full_trades)
    total_pnl = pnls_all.sum()
    print(f"  {'TOTAL':<28}  {total_n:>7}  {'':>9}  ${total_pnl:>+10,.2f}")
    print("=" * 64)
    print()

    n_negative = sum(1 for r in sub_results if r['pnl'] < 0)
    print(f"Negative sub-periods: {n_negative} / {N_SUBPERIODS}")
    print()

    # ── Honest assessment ──
    print("=" * 64)
    print("   HONEST ASSESSMENT")
    print("=" * 64)

    pf_drop = (train_m['profit_factor'] - fwd_m['profit_factor']) if train_m['profit_factor'] != float('inf') else 0
    wr_drop = train_m['win_rate'] - fwd_m['win_rate']
    pnl_ratio = (fwd_m['pnl'] / train_m['pnl']) if train_m['pnl'] not in (0, None) else None

    print(f"  Train PF={train_m['profit_factor']}  Fwd PF={fwd_m['profit_factor']}  (drop: {pf_drop:+.2f})")
    print(f"  Train WR={train_m['win_rate']}%  Fwd WR={fwd_m['win_rate']}%  (drop: {wr_drop:+.1f}pp)")
    print(f"  Train P&L=${train_m['pnl']:+,.2f}  Fwd P&L=${fwd_m['pnl']:+,.2f}")

    both_profitable = train_m['pnl'] > 0 and fwd_m['pnl'] > 0
    pf_holds = fwd_m['profit_factor'] >= 1.0 and pf_drop < 0.5
    consistent = both_profitable and pf_holds and abs(wr_drop) < 10

    if consistent:
        verdict = "CONSISTENT -- forward test result is in the same ballpark as training, no regime-fitting collapse detected."
    elif both_profitable:
        verdict = "PARTIALLY CONSISTENT -- both windows profitable, but with meaningful degradation forward."
    else:
        verdict = "FAILS -- forward test does not hold up; same regime-fitting collapse pattern as the M15 search."

    print(f"\n  Verdict: {verdict}\n")

    print("  Comparison to other strategies tested:")
    print(f"    M15 EMA search        : 0 / 2,736 configs passed forward test (total collapse)")
    print(f"    Bollinger Band search  : 0 / 108   configs passed even training (structural PF failure)")
    print(f"    H1 ATR 3.0x/6.0x (this): "
          f"{'PASSES' if consistent else ('PARTIAL' if both_profitable else 'FAILS')} -- "
          f"{'genuinely different in quality' if consistent or both_profitable else 'same failure pattern'}")
    print()
    print("Done!")


if __name__ == '__main__':
    main()
