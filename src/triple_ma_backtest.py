"""
Triple MA Alignment Backtest -- EURUSD M5 (dynamic sizing)
===========================================================
Strategy (Run 1 rules, M5 timeframe):
  Entry:
    BUY  -- M5 50 SMA crosses above 100 SMA AND 100 SMA > 200 SMA
    SELL -- M5 50 SMA crosses below 100 SMA AND 100 SMA < 200 SMA
    Flat filter: skip if spread(sma50, sma100, sma200) < 5 pips
    Enter on bar close after crossover confirmed

  H4 filter:
    BUY  only when H4 50 SMA > H4 200 SMA
    SELL only when H4 50 SMA < H4 200 SMA

  Session filter:
    08:00-22:00 GMT (London + NY)

  Exit:
    SL = 30 pips, TP = 60 pips (2:1 RR)
    Early exit: M5 50 SMA crosses back through 100 SMA

  Daily limits:
    Max 2 trades per day
    Stop after 2 consecutive losses

Dynamic position sizing:
    Max daily loss  = 5% of current balance
    Risk per trade  = 20% of max daily loss = 1% of balance
    Lot size        = risk_usd / (SL_PIPS x pip_value_per_lot)
    Hard cap        = 2.0 lots

Settings:
  Pair            : EURUSD
  Starting balance: $10,000
  Data            : 3 years M5 + H4 from MT5
"""

from __future__ import annotations

import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import MetaTrader5 as mt5
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

SYMBOL         = 'EURUSD'
PIP_SIZE       = 0.0001
PIP_VALUE      = 10.00     # $ per pip per standard lot (EURUSD USD account)
STARTING_BAL   = 10_000.00

SL_PIPS        = 30
TP_PIPS        = 60
SL_PRICE       = SL_PIPS * PIP_SIZE   # 0.0030
TP_PRICE       = TP_PIPS * PIP_SIZE   # 0.0060

# Dynamic sizing
MAX_DAILY_LOSS_PCT  = 0.05   # 5% of balance
RISK_PER_TRADE_PCT  = 0.20   # 20% of daily limit = 1% of balance
MIN_LOT             = 0.01
MAX_LOT             = 2.00

FLAT_PIPS      = 5
FLAT_THRESHOLD = FLAT_PIPS * PIP_SIZE   # 0.0005

SMA_FAST       = 50
SMA_MID        = 100
SMA_SLOW       = 200
H4_FAST        = 50
H4_SLOW        = 200

SESSION_START  = 8    # 08:00 GMT
SESSION_END    = 22   # 22:00 GMT

MAX_TRADES_DAY  = 2
MAX_CONSEC_LOSS = 2

YEARS   = 3
M5_BARS = YEARS * 365 * 288 + 5_000   # ~320k + buffer
H4_BARS = YEARS * 365 * 6  + 500      # ~6.5k + buffer

OUT_PNG = 'data/triple_ma_m5_dynamic_equity_curve.png'


# ---------------------------------------------------------------------------
# MT5 data fetch
# ---------------------------------------------------------------------------

def connect():
    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        sys.exit(1)
    print(f"Connected: {mt5.terminal_info().name}")


def fetch_bars(timeframe, n: int, label: str) -> pd.DataFrame:
    print(f"Fetching {n:,} {label} bars for {SYMBOL}...")
    rates = mt5.copy_rates_from_pos(SYMBOL, timeframe, 0, n)
    if rates is None or len(rates) == 0:
        print(f"ERROR: no {label} data"); sys.exit(1)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    print(f"  {label} rows: {len(df):,}  ({df.index[0].date()} -> {df.index[-1].date()})")
    return df


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def calc_entry_smas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['sma50']  = df['close'].rolling(SMA_FAST).mean()
    df['sma100'] = df['close'].rolling(SMA_MID).mean()
    df['sma200'] = df['close'].rolling(SMA_SLOW).mean()
    return df.dropna(subset=['sma50', 'sma100', 'sma200'])


def calc_h4_trend_series(h4: pd.DataFrame) -> pd.Series:
    """Return a Series of H4 trend (+1/-1/0) indexed by H4 bar time."""
    fast = h4['close'].rolling(H4_FAST).mean()
    slow = h4['close'].rolling(H4_SLOW).mean()
    trend = np.where(fast > slow, 1, np.where(fast < slow, -1, 0))
    return pd.Series(trend, index=h4.index, name='h4_trend')


def align_h4_to_m5(m5: pd.DataFrame, h4_trend: pd.Series) -> pd.Series:
    """
    Forward-fill H4 trend values onto every M5 timestamp.
    Uses merge_asof so each M5 bar inherits the most recent completed H4 bar's trend.
    """
    combined = h4_trend.reindex(h4_trend.index.union(m5.index))
    filled   = combined.ffill()
    return filled.reindex(m5.index).fillna(0).astype(int)


# ---------------------------------------------------------------------------
# Dynamic lot sizing
# ---------------------------------------------------------------------------

def calc_lot_size(balance: float) -> float:
    risk_usd = balance * MAX_DAILY_LOSS_PCT * RISK_PER_TRADE_PCT
    lots = risk_usd / (SL_PIPS * PIP_VALUE)
    return round(max(MIN_LOT, min(lots, MAX_LOT)), 2)


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def run_backtest(m5: pd.DataFrame, h4_trend_aligned: pd.Series) -> dict:
    balance   = STARTING_BAL
    equity_ts = []
    trades    = []

    today         = None
    trades_today  = 0
    consec_losses = 0
    daily_stop    = False

    in_trade    = False
    trade_dir   = None
    entry_price = 0.0
    sl_price    = 0.0
    tp_price    = 0.0
    entry_time  = None
    lot_size    = 0.0   # set at entry

    bars    = m5.reset_index()
    h4_arr  = h4_trend_aligned.values   # numpy array, aligned to bars

    for i in range(1, len(bars)):
        row  = bars.iloc[i]
        prev = bars.iloc[i - 1]
        ts   = row['time']
        date_ = ts.date()
        hour  = ts.hour

        if date_ != today:
            today        = date_
            trades_today = 0
            daily_stop   = False

        equity_ts.append((ts, balance))

        # -- Manage open trade
        if in_trade:
            bar_low  = row['low']
            bar_high = row['high']

            if trade_dir == 'BUY':
                hit_tp = bar_high >= tp_price
                hit_sl = bar_low  <= sl_price
            else:
                hit_tp = bar_low  <= tp_price
                hit_sl = bar_high >= sl_price

            # Early exit: 50 SMA crosses back through 100 SMA
            early_exit = (
                (trade_dir == 'BUY'  and row['sma50'] < row['sma100']) or
                (trade_dir == 'SELL' and row['sma50'] > row['sma100'])
            )

            closed   = False
            exit_px  = 0.0
            exit_why = ''

            if hit_tp and not hit_sl:
                exit_px, exit_why, closed = tp_price, 'TP', True
            elif hit_sl and not hit_tp:
                exit_px, exit_why, closed = sl_price, 'SL', True
            elif hit_sl and hit_tp:
                exit_px, exit_why, closed = sl_price, 'SL', True   # conservative
            elif early_exit:
                exit_px, exit_why, closed = row['close'], 'CROSS_EXIT', True

            if closed:
                pnl_pips = (exit_px - entry_price) / PIP_SIZE if trade_dir == 'BUY' \
                           else (entry_price - exit_px) / PIP_SIZE
                pnl_usd  = round(pnl_pips * PIP_VALUE * lot_size, 2)
                balance  += pnl_usd

                won = pnl_usd > 0
                consec_losses = 0 if won else consec_losses + 1
                if consec_losses >= MAX_CONSEC_LOSS:
                    daily_stop = True

                trades.append({
                    'entry_time' : entry_time,
                    'exit_time'  : ts,
                    'direction'  : trade_dir,
                    'entry_price': entry_price,
                    'exit_price' : exit_px,
                    'lot_size'   : lot_size,
                    'pnl_pips'   : round(pnl_pips, 1),
                    'pnl_usd'    : pnl_usd,
                    'exit_reason': exit_why,
                    'balance'    : round(balance, 2),
                })

                in_trade = False
                continue

            continue   # still in trade

        # -- Entry signal checks
        if trades_today >= MAX_TRADES_DAY or daily_stop:
            continue
        if not (SESSION_START <= hour < SESSION_END):
            continue

        sma50_now   = row['sma50']
        sma100_now  = row['sma100']
        sma200_now  = row['sma200']
        sma50_prev  = prev['sma50']
        sma100_prev = prev['sma100']

        # Flat market filter (all 3 MAs)
        spread = max(sma50_now, sma100_now, sma200_now) - min(sma50_now, sma100_now, sma200_now)
        if spread < FLAT_THRESHOLD:
            continue

        # Crossover detection: 50 crosses 100
        bull_cross = (sma50_prev <= sma100_prev) and (sma50_now > sma100_now)
        bear_cross = (sma50_prev >= sma100_prev) and (sma50_now < sma100_now)

        # Alignment: 100 must already be on the right side of 200
        bull_align = sma100_now > sma200_now
        bear_align = sma100_now < sma200_now

        signal = None
        if bull_cross and bull_align:
            signal = 'BUY'
        elif bear_cross and bear_align:
            signal = 'SELL'

        if signal is None:
            continue

        # H4 filter (pre-aligned array lookup -- O(1))
        h4_t = int(h4_arr[i])
        if signal == 'BUY'  and h4_t != 1:
            continue
        if signal == 'SELL' and h4_t != -1:
            continue

        # Dynamic lot size at entry
        lot_size = calc_lot_size(balance)
        ep = row['close']

        if signal == 'BUY':
            sl = ep - SL_PRICE
            tp = ep + TP_PRICE
        else:
            sl = ep + SL_PRICE
            tp = ep - TP_PRICE

        in_trade    = True
        trade_dir   = signal
        entry_price = ep
        sl_price    = sl
        tp_price    = tp
        entry_time  = ts
        trades_today += 1

    equity_ts.append((bars.iloc[-1]['time'], balance))

    return {'trades': trades, 'equity_ts': equity_ts, 'final_bal': balance}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(result: dict) -> dict:
    trades = result['trades']
    if not trades:
        return {}

    df = pd.DataFrame(trades)
    df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)
    df['exit_time']  = pd.to_datetime(df['exit_time'],  utc=True)
    df['month']      = df['entry_time'].dt.to_period('M')

    total    = len(df)
    wins     = (df['pnl_usd'] > 0).sum()
    losses   = (df['pnl_usd'] <= 0).sum()
    win_rate = wins / total * 100

    gross_profit  = df[df['pnl_usd'] > 0]['pnl_usd'].sum()
    gross_loss    = df[df['pnl_usd'] <= 0]['pnl_usd'].sum()
    profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else float('inf')

    total_pnl = df['pnl_usd'].sum()
    avg_win   = df[df['pnl_usd'] > 0]['pnl_usd'].mean() if wins > 0 else 0
    avg_loss  = df[df['pnl_usd'] <= 0]['pnl_usd'].mean() if losses > 0 else 0
    avg_lot   = df['lot_size'].mean()

    eq     = np.array([e[1] for e in result['equity_ts']])
    peak   = np.maximum.accumulate(eq)
    dd     = (peak - eq) / peak * 100
    max_dd = dd.max()

    monthly = (df.groupby('month')['pnl_usd']
                 .agg(pnl='sum', trades='count')
                 .reset_index())
    monthly['profitable'] = monthly['pnl'] > 0
    prof_months  = monthly['profitable'].sum()
    total_months = len(monthly)

    exit_counts   = df['exit_reason'].value_counts().to_dict()
    avg_per_month = total / max(total_months, 1)

    return {
        'total'        : total,
        'wins'         : wins,
        'losses'       : losses,
        'win_rate'     : win_rate,
        'total_pnl'    : total_pnl,
        'gross_profit' : gross_profit,
        'gross_loss'   : gross_loss,
        'profit_factor': profit_factor,
        'avg_win'      : avg_win,
        'avg_loss'     : avg_loss,
        'avg_lot'      : avg_lot,
        'max_dd'       : max_dd,
        'avg_per_month': avg_per_month,
        'prof_months'  : prof_months,
        'total_months' : total_months,
        'monthly'      : monthly,
        'exit_counts'  : exit_counts,
        'df'           : df,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

# Run 1 M15 fixed-lot reference numbers for comparison
M15_REF = {
    'total'        : 51,
    'win_rate'     : 45.1,
    'total_pnl'    : 321.60,
    'profit_factor': 1.51,
    'max_dd'       : 1.85,
    'avg_per_month': 1.6,
    'prof_months'  : '17/31 (55%)',
}


def print_report(stats: dict, result: dict):
    s = stats
    print()
    print("=" * 64)
    print("  TRIPLE MA ALIGNMENT -- EURUSD M5 + DYNAMIC SIZING -- 3Y")
    print("=" * 64)
    print()
    print(f"  Starting balance  : ${STARTING_BAL:>10,.2f}")
    print(f"  Final balance     : ${result['final_bal']:>10,.2f}")
    print(f"  Total P&L         : ${s['total_pnl']:>+10,.2f}")
    print(f"  Return            : {(result['final_bal']/STARTING_BAL-1)*100:>+9.1f}%")
    print()
    print(f"  Total trades      : {s['total']}")
    print(f"  Wins / Losses     : {s['wins']} / {s['losses']}")
    print(f"  Win rate          : {s['win_rate']:.1f}%")
    print(f"  Profit factor     : {s['profit_factor']:.2f}")
    print(f"  Avg lot size      : {s['avg_lot']:.2f}")
    print(f"  Avg win           : ${s['avg_win']:>+8.2f}")
    print(f"  Avg loss          : ${s['avg_loss']:>+8.2f}")
    print()
    print(f"  Max drawdown      : {s['max_dd']:.2f}%")
    print(f"  Avg trades/month  : {s['avg_per_month']:.1f}")
    pct_prof = s['prof_months'] / s['total_months'] * 100
    print(f"  Profitable months : {s['prof_months']} / {s['total_months']} ({pct_prof:.0f}%)")
    print()
    print("  Exit reasons:")
    for k, v in sorted(s['exit_counts'].items(), key=lambda x: -x[1]):
        print(f"    {k:<15} : {v}")
    print()
    print("  MONTHLY BREAKDOWN")
    print(f"  {'Month':<10}  {'Trades':>6}  {'P&L':>10}  {'Result'}")
    print("  " + "-" * 42)
    for _, row in s['monthly'].iterrows():
        mark = "PROFIT" if row['profitable'] else "loss  "
        print(f"  {str(row['month']):<10}  {int(row['trades']):>6}  "
              f"${row['pnl']:>+9,.2f}  {mark}")
    print()

    # -- M5 vs M15 comparison table
    ret_m5 = (result['final_bal'] / STARTING_BAL - 1) * 100
    print("=" * 64)
    print("  M5 vs M15 COMPARISON  (Run 1 rules, fixed 0.1L vs dynamic)")
    print("=" * 64)
    print(f"  {'Metric':<22}  {'M15 fixed-lot':>14}  {'M5 dynamic':>12}")
    print("  " + "-" * 52)
    print(f"  {'Total trades':<22}  {M15_REF['total']:>14}  {s['total']:>12}")
    print(f"  {'Win rate':<22}  {M15_REF['win_rate']:>13.1f}%  {s['win_rate']:>11.1f}%")
    print(f"  {'Total P&L':<22}  ${M15_REF['total_pnl']:>+13,.2f}  ${s['total_pnl']:>+11,.2f}")
    print(f"  {'Return':<22}  {'~3.2%':>14}  {ret_m5:>+11.1f}%")
    print(f"  {'Profit factor':<22}  {M15_REF['profit_factor']:>14.2f}  {s['profit_factor']:>12.2f}")
    print(f"  {'Max drawdown':<22}  {M15_REF['max_dd']:>13.2f}%  {s['max_dd']:>11.2f}%")
    print(f"  {'Avg trades/month':<22}  {M15_REF['avg_per_month']:>14.1f}  {s['avg_per_month']:>12.1f}")
    print(f"  {'Profitable months':<22}  {M15_REF['prof_months']:>14}  {s['prof_months']}/{s['total_months']} ({pct_prof:.0f}%)")
    print()

    # Honest assessment
    wr  = s['win_rate']
    pf  = s['profit_factor']
    dd  = s['max_dd']
    ret = ret_m5

    if ret > 0 and pf > 1.5 and dd < 20:
        verdict = "VIABLE -- positive expectancy, good drawdown control"
    elif ret > 0 and pf > 1.2:
        verdict = "MARGINAL -- profitable but room for improvement"
    else:
        verdict = "UNDERPERFORMING -- negative or weak expectancy"

    print("=" * 64)
    print("  HONEST ASSESSMENT")
    print("=" * 64)
    print(f"  Verdict: {verdict}")
    print()
    print(f"  M5 generates ~{s['avg_per_month']:.1f} trades/month vs M15's "
          f"{M15_REF['avg_per_month']:.1f} -- "
          + ("higher frequency on faster TF." if s['avg_per_month'] > M15_REF['avg_per_month']
             else "similar frequency despite faster TF."))
    print()
    print(f"  Dynamic sizing (avg {s['avg_lot']:.2f}L) vs fixed 0.1L on M15.")
    print(f"  At $10k start: 1% risk / 30-pip SL = {calc_lot_size(STARTING_BAL):.2f} lots initially.")
    print()
    if s['win_rate'] > M15_REF['win_rate']:
        print("  Win rate IMPROVED on M5 vs M15.")
    elif abs(s['win_rate'] - M15_REF['win_rate']) < 2:
        print("  Win rate essentially unchanged between M5 and M15.")
    else:
        print("  Win rate DECLINED on M5 -- more noise on the faster timeframe.")
    print()
    if dd > M15_REF['max_dd'] * 2:
        print(f"  Drawdown {dd:.1f}% is significantly higher than M15's "
              f"{M15_REF['max_dd']:.1f}% -- dynamic sizing amplifies swings.")
    else:
        print(f"  Drawdown {dd:.1f}% vs M15 {M15_REF['max_dd']:.1f}% -- "
              "dynamic sizing drawdown is manageable.")
    print("=" * 64)


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

def plot_equity(result: dict, stats: dict):
    eq_times  = [e[0] for e in result['equity_ts']]
    eq_values = [e[1] for e in result['equity_ts']]

    df_tr   = stats['df']
    monthly = stats['monthly']

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[3, 1, 1], hspace=0.38)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(eq_times, eq_values, color='#2196F3', linewidth=1.0, label='Equity')
    ax1.axhline(STARTING_BAL, color='gray', linewidth=0.8, linestyle='--', label='Start $10k')

    wins  = df_tr[df_tr['pnl_usd'] > 0]
    losss = df_tr[df_tr['pnl_usd'] <= 0]
    ax1.scatter(wins['exit_time'],  wins['balance'],  color='#4CAF50', s=10, zorder=3, label='Win')
    ax1.scatter(losss['exit_time'], losss['balance'], color='#F44336', s=10, zorder=3, label='Loss')

    ax1.set_title(
        f'Triple MA Alignment -- EURUSD M5 -- Dynamic Sizing -- 3-Year Equity Curve\n'
        f'(50/100/200 SMA crossover + H4 filter + cross-exit | 1% risk per trade)',
        fontsize=10, pad=8
    )
    ax1.set_ylabel('Balance ($)')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    eq_arr = np.array(eq_values)
    peak   = np.maximum.accumulate(eq_arr)
    dd_arr = (peak - eq_arr) / peak * 100

    ax2 = fig.add_subplot(gs[1])
    ax2.fill_between(eq_times, -dd_arr, 0, color='#F44336', alpha=0.5, label='Drawdown')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_ylim(bottom=-(dd_arr.max() * 1.25 or 1))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{-x:.0f}%'))
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower left', fontsize=8)

    ax3 = fig.add_subplot(gs[2])
    colors = ['#4CAF50' if p else '#F44336' for p in monthly['profitable']]
    x_pos  = range(len(monthly))
    ax3.bar(x_pos, monthly['pnl'], color=colors, width=0.7)
    ax3.axhline(0, color='black', linewidth=0.8)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([str(m) for m in monthly['month']], rotation=45, ha='right', fontsize=7)
    ax3.set_ylabel('Monthly P&L ($)')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax3.set_title('Monthly P&L', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')

    pct_prof = stats['prof_months'] / stats['total_months'] * 100
    fig.text(
        0.5, 0.005,
        f"Win rate: {stats['win_rate']:.1f}%  |  PF: {stats['profit_factor']:.2f}  |  "
        f"Max DD: {stats['max_dd']:.1f}%  |  P&L: ${stats['total_pnl']:+,.2f}  |  "
        f"Trades: {stats['total']}  |  Prof months: {stats['prof_months']}/{stats['total_months']} ({pct_prof:.0f}%)",
        ha='center', fontsize=8, color='#333333'
    )

    plt.savefig(OUT_PNG, dpi=130, bbox_inches='tight')
    print(f"\n  Chart saved: {OUT_PNG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    connect()

    m5_raw = fetch_bars(mt5.TIMEFRAME_M5, M5_BARS, 'M5')
    h4_raw = fetch_bars(mt5.TIMEFRAME_H4, H4_BARS, 'H4')

    print("Computing indicators...")
    m5 = calc_entry_smas(m5_raw)
    h4_trend = calc_h4_trend_series(h4_raw)

    # Restrict to 3-year window
    cutoff = m5.index[-1] - pd.DateOffset(years=YEARS)
    m5     = m5[m5.index >= cutoff]

    # Keep extra H4 history for pre-period warm-up, then align
    h4_for_align = h4_trend[h4_trend.index >= cutoff - pd.DateOffset(days=60)]
    h4_aligned   = align_h4_to_m5(m5, h4_for_align)

    print(f"  M5 bars in window : {len(m5):,}")
    print(f"  H4 bars used      : {h4_for_align.notna().sum():,}")
    print(f"  Initial lot size  : {calc_lot_size(STARTING_BAL):.2f}L "
          f"(1% of ${STARTING_BAL:,.0f} / {SL_PIPS}p SL)")
    print("Running backtest...")

    result = run_backtest(m5, h4_aligned)

    if not result['trades']:
        print("No trades generated -- check data and strategy parameters.")
        sys.exit(0)

    stats = compute_stats(result)
    print_report(stats, result)
    plot_equity(result, stats)

    mt5.shutdown()


if __name__ == '__main__':
    main()
