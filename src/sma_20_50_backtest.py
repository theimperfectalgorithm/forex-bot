"""
Forex Bot - 20/50 SMA Crossover Backtest (3-Year, EURUSD M15)

Tests whether shorter SMA windows (20/50) react fast enough on the M15
timeframe to outperform the 50/200 crossover that failed the 3-year
stress test (-$43.20, 43.4% win rate).

Both strategies use the same RSI-14/60-40 confirmation filter:
  BUY  when fast SMA crosses above slow SMA  AND  RSI < 60
  SELL when fast SMA crosses below slow SMA  AND  RSI > 40

Data fetched once from MT5 and shared between both runs.
Settings: EURUSD M15, ~3 years, $10,000 balance, 0.1 lots ($1/pip).

Requirements:
  MetaTrader 5 must be OPEN and LOGGED IN before running this script.
"""

import os
import sys
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── SETTINGS ──────────────────────────────────────────────────────────────────

SYMBOL          = 'EURUSD'
MONTHS          = 36
INITIAL_BALANCE = 10_000.00
PIP_SIZE        = 0.0001
PIP_VALUE_USD   = 1.00
RSI_PERIOD      = 14
BUY_THRESHOLD   = 60
SELL_THRESHOLD  = 40

# 50/200 3-year results from stress_test.py — used for the comparison table
BASELINE_50_200 = {
    'label'   : '50/200 SMA (stress test)',
    'trades'  : 235,
    'winners' : 102,
    'losers'  : 133,
    'win_rate': 43.4,
    'pnl'     : -43.20,
    'balance' : 9_956.80,
    'max_dd'  : 12.4,
    'best'    : 346.60,
    'worst'   : -361.60,
}


# ── 1. CONNECT TO MT5 ─────────────────────────────────────────────────────────

def connect_mt5():
    print("Connecting to MetaTrader 5...")
    if not mt5.initialize():
        print(f"ERROR: Could not connect — {mt5.last_error()}")
        print("Make sure MetaTrader 5 is open and logged in.")
        sys.exit(1)
    info    = mt5.terminal_info()
    account = mt5.account_info()
    print(f"Connected : {info.name}")
    print(f"Account   : {account.login}  ({account.server})\n")


# ── 2. FETCH M15 DATA ─────────────────────────────────────────────────────────

def fetch_data() -> pd.DataFrame:
    date_to   = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=MONTHS * 30)

    print(f"Requesting M15 data from {date_from.date()} to {date_to.date()} ...")
    rates = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M15, date_from, date_to)

    if rates is None or len(rates) == 0:
        print(f"ERROR: No data returned — {mt5.last_error()}")
        mt5.shutdown()
        sys.exit(1)

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df.rename(columns={'close': 'Close'}, inplace=True)

    months_actual = (df.index[-1] - df.index[0]).days / 30
    print(f"Received   : {len(df):,} bars  "
          f"({df.index[0].date()} to {df.index[-1].date()},  "
          f"~{months_actual:.1f} months)\n")
    return df[['Close']]


# ── 3. RSI ────────────────────────────────────────────────────────────────────

def calculate_rsi(close: pd.Series, period: int) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ── 4. STRATEGY — parameterised SMA windows ───────────────────────────────────

def apply_strategy(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    data = df.copy()
    data['SMA_fast'] = data['Close'].rolling(window=fast).mean()
    data['SMA_slow'] = data['Close'].rolling(window=slow).mean()
    data['RSI']      = calculate_rsi(data['Close'], RSI_PERIOD)
    data.dropna(inplace=True)

    data['Position'] = (data['SMA_fast'] > data['SMA_slow']).astype(int)
    crossover        = data['Position'].diff()

    data['Signal'] = 0.0
    data.loc[(crossover == 1)  & (data['RSI'] < BUY_THRESHOLD),  'Signal'] =  1.0
    data.loc[(crossover == -1) & (data['RSI'] > SELL_THRESHOLD), 'Signal'] = -1.0
    return data


# ── 5. BACKTEST ───────────────────────────────────────────────────────────────

def run_backtest(data: pd.DataFrame):
    signal_rows = data[data['Signal'].isin([1.0, -1.0])]
    trades      = []
    equity      = [INITIAL_BALANCE]
    balance     = INITIAL_BALANCE
    open_trade  = None

    for timestamp, row in signal_rows.iterrows():
        signal = row['Signal']
        price  = row['Close']

        if open_trade is not None:
            direction   = open_trade['direction']
            entry_price = open_trade['entry_price']
            entry_time  = open_trade['entry_time']

            pips    = ((price - entry_price) if direction == 'LONG'
                       else (entry_price - price)) / PIP_SIZE
            pnl     = round(pips * PIP_VALUE_USD, 2)
            balance = round(balance + pnl, 2)

            trades.append({
                'Trade #'    : len(trades) + 1,
                'Direction'  : direction,
                'Entry Date' : entry_time.strftime('%Y-%m-%d %H:%M'),
                'Exit Date'  : timestamp.strftime('%Y-%m-%d %H:%M'),
                'Entry Price': round(entry_price, 5),
                'Exit Price' : round(price, 5),
                'Pips'       : round(pips, 1),
                'P&L (USD)'  : pnl,
                'Balance'    : balance,
                'Result'     : 'WIN' if pnl > 0 else 'LOSS',
            })
            equity.append(balance)

        open_trade = {
            'direction'  : 'LONG' if signal == 1.0 else 'SHORT',
            'entry_price': price,
            'entry_time' : timestamp,
        }

    return trades, equity


# ── 6. COMPUTE STATS ──────────────────────────────────────────────────────────

def compute_stats(trades: list, equity: list, data: pd.DataFrame) -> dict:
    df      = pd.DataFrame(trades)
    winners = df[df['P&L (USD)'] > 0]
    losers  = df[df['P&L (USD)'] <= 0]

    peak   = INITIAL_BALANCE
    max_dd = 0.0
    for e in equity:
        peak   = max(peak, e)
        max_dd = max(max_dd, (peak - e) / peak * 100)

    months = (data.index[-1] - data.index[0]).days / 30

    return {
        'months'   : round(months, 1),
        'trades'   : len(df),
        'winners'  : len(winners),
        'losers'   : len(losers),
        'win_rate' : round(len(winners) / len(df) * 100, 1) if len(df) else 0,
        'pnl'      : round(df['P&L (USD)'].sum(), 2),
        'balance'  : equity[-1],
        'max_dd'   : round(max_dd, 1),
        'avg_win'  : round(winners['P&L (USD)'].mean(), 2) if not winners.empty else 0,
        'avg_loss' : round(losers['P&L (USD)'].mean(),  2) if not losers.empty  else 0,
        'best'     : round(df['P&L (USD)'].max(), 2),
        'worst'    : round(df['P&L (USD)'].min(), 2),
        'per_month': round(df['P&L (USD)'].sum() / months, 2) if months else 0,
    }


# ── 7. PRINT COMPARISON TABLE ─────────────────────────────────────────────────

def print_comparison(new: dict):
    b = BASELINE_50_200

    def chg(new_val, old_val, higher_better=True):
        diff = new_val - old_val
        sign = '+' if diff >= 0 else ''
        tag  = ('better' if (diff > 0) == higher_better and diff != 0
                else ('worse' if diff != 0 else 'same'))
        return f"  {sign}{diff:.1f}  ({tag})"

    w = 62
    print()
    print("=" * w)
    print("     SMA COMPARISON — 3 Years  |  EURUSD M15  |  0.1 Lots")
    print(f"     RSI-{RSI_PERIOD} filter: BUY < {BUY_THRESHOLD}, SELL > {SELL_THRESHOLD}")
    print("=" * w)
    print(f"  {'Metric':<24}  {'50/200 SMA':>12}  {'20/50 SMA':>12}  Change")
    print("-" * w)
    print(f"  {'Total Trades':<24}  {b['trades']:>12}  {new['trades']:>12}"
          f"{chg(new['trades'], b['trades'])}")
    print(f"  {'Winning Trades':<24}  {b['winners']:>12}  {new['winners']:>12}")
    print(f"  {'Losing Trades':<24}  {b['losers']:>12}  {new['losers']:>12}")
    print(f"  {'Win Rate':<24}  {b['win_rate']:>11.1f}%  {new['win_rate']:>11.1f}%"
          f"{chg(new['win_rate'], b['win_rate'])}")
    print(f"  {'Total P&L':<24}  ${b['pnl']:>+11,.2f}  ${new['pnl']:>+11,.2f}"
          f"{chg(new['pnl'], b['pnl'])}")
    print(f"  {'Avg P&L / month':<24}  ${b['pnl']/b['months'] if 'months' in b else b['pnl']/35.9:>+11,.2f}"
          f"  ${new['per_month']:>+11,.2f}")
    print(f"  {'Final Balance':<24}  ${b['balance']:>11,.2f}  ${new['balance']:>11,.2f}")
    print(f"  {'Max Drawdown':<24}  {b['max_dd']:>11.1f}%  {new['max_dd']:>11.1f}%"
          f"{chg(new['max_dd'], b['max_dd'], higher_better=False)}")
    print("-" * w)
    print(f"  {'Avg Win':<24}  {'':>12}  ${new['avg_win']:>+11,.2f}")
    print(f"  {'Avg Loss':<24}  {'':>12}  ${new['avg_loss']:>+11,.2f}")
    print(f"  {'Best Trade':<24}  ${b['best']:>+11,.2f}  ${new['best']:>+11,.2f}")
    print(f"  {'Worst Trade':<24}  ${b['worst']:>+11,.2f}  ${new['worst']:>+11,.2f}")
    print("=" * w)
    print()


# ── 8. PLOT — overlay both equity curves ──────────────────────────────────────

def plot_comparison(trades_new: list, equity_new: list,
                    trades_old: list, equity_old: list,
                    data: pd.DataFrame):
    df_new = pd.DataFrame(trades_new)
    df_old = pd.DataFrame(trades_old)

    exit_new = pd.to_datetime(df_new['Exit Date'])
    exit_old = pd.to_datetime(df_old['Exit Date'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11))

    # ── Overlay equity curves ──
    ax1.axhline(INITIAL_BALANCE, color='grey', linewidth=1, linestyle='--',
                label=f'Starting balance  ${INITIAL_BALANCE:,.0f}', zorder=1)

    # 50/200 — red (it lost money)
    full_old = [exit_old.iloc[0]] + list(exit_old)
    ax1.plot(full_old, equity_old, color='#d62728', linewidth=1.6,
             linestyle='--', alpha=0.75, label='50/200 SMA  (baseline)', zorder=2)

    # 20/50 — blue or green depending on outcome
    full_new = [exit_new.iloc[0]] + list(exit_new)
    new_color = '#2ca02c' if equity_new[-1] >= INITIAL_BALANCE else '#1f77b4'
    ax1.plot(full_new, equity_new, color=new_color, linewidth=2,
             label='20/50 SMA  (this test)', zorder=3)

    ax1.fill_between(full_new, INITIAL_BALANCE, equity_new,
                     where=[e >= INITIAL_BALANCE for e in equity_new],
                     alpha=0.12, color='green')
    ax1.fill_between(full_new, INITIAL_BALANCE, equity_new,
                     where=[e < INITIAL_BALANCE for e in equity_new],
                     alpha=0.12, color='red')

    ax1.set_title(
        f'20/50 vs 50/200 SMA Crossover  +  RSI-{RSI_PERIOD}/{BUY_THRESHOLD}-{SELL_THRESHOLD}  '
        f'|  EURUSD M15  |  0.1 Lots\n'
        f'{data.index[0].date()}  to  {data.index[-1].date()}',
        fontsize=13, fontweight='bold')
    ax1.set_ylabel('Account Balance (USD)')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')

    # ── 20/50 per-trade P&L bars ──
    bar_colors = ['#2ca02c' if p > 0 else '#d62728' for p in df_new['P&L (USD)']]
    ax2.bar(df_new['Trade #'], df_new['P&L (USD)'],
            color=bar_colors, alpha=0.8, width=0.7)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_title('20/50 SMA — Individual Trade P&L', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Trade Number')
    ax2.set_ylabel('P&L (USD)')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()

    output_dir  = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'sma_20_50_vs_50_200.png')
    plt.savefig(output_path, dpi=120)
    print(f"Chart saved : {output_path}")


# ── 9. SAVE TRADE LOG ─────────────────────────────────────────────────────────

def save_trade_log(trades: list):
    df          = pd.DataFrame(trades)
    output_dir  = os.path.join(os.path.dirname(__file__), '..', 'data')
    output_path = os.path.join(output_dir, 'trade_log_sma_20_50.csv')
    df.to_csv(output_path, index=False)
    print(f"Trade log   : {output_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

connect_mt5()
raw = fetch_data()
mt5.shutdown()

# Run 50/200 first (for the overlay chart) then 20/50
print("Running 50/200 SMA strategy ...")
data_50_200           = apply_strategy(raw, fast=50, slow=200)
trades_50_200, eq_50_200 = run_backtest(data_50_200)

print("Running 20/50 SMA strategy ...")
data_20_50            = apply_strategy(raw, fast=20, slow=50)
trades_20_50, eq_20_50   = run_backtest(data_20_50)

stats = compute_stats(trades_20_50, eq_20_50, data_20_50)

print_comparison(stats)
save_trade_log(trades_20_50)
plot_comparison(trades_20_50, eq_20_50, trades_50_200, eq_50_200, raw)
print("\nDone!")
