"""
Forex Bot - M15 Backtest Engine

Runs a full backtest of the 50/200-period SMA crossover strategy on
EURUSD M15 data fetched directly from MetaTrader 5.

Trade rules:
  BUY  signal -> enter LONG,  hold until the next SELL signal
  SELL signal -> enter SHORT, hold until the next BUY  signal
  (The bot is always in the market once the first signal fires)

Position sizing:
  Fixed 0.1 lots per trade.
  EURUSD pip value at 0.1 lots = $1.00 per pip
  (0.0001 price move x 10,000 units = $1.00)

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

# ── SETTINGS ──────────────────────────────────────────────────────────────────

SYMBOL          = 'EURUSD'
MONTHS          = 6
INITIAL_BALANCE = 10_000.00
LOT_SIZE        = 0.1
PIP_SIZE        = 0.0001   # 1 pip for EURUSD (4th decimal place)
PIP_VALUE_USD   = 1.00     # $1.00 per pip for 0.1 lots on EURUSD


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
    print(f"Account   : {account.login}  ({account.server})")
    print(f"Balance   : {account.balance} {account.currency}\n")


# ── 2. FETCH M15 DATA ─────────────────────────────────────────────────────────

def fetch_data() -> pd.DataFrame:
    date_to   = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=MONTHS * 30)

    rates = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M15, date_from, date_to)

    if rates is None or len(rates) == 0:
        print(f"ERROR: No data returned — {mt5.last_error()}")
        mt5.shutdown()
        sys.exit(1)

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df.rename(columns={'close': 'Close'}, inplace=True)
    return df[['Close']]


# ── 3. APPLY SMA CROSSOVER STRATEGY ──────────────────────────────────────────

def apply_strategy(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data['SMA_50']   = data['Close'].rolling(window=50).mean()
    data['SMA_200']  = data['Close'].rolling(window=200).mean()
    data.dropna(inplace=True)
    data['Position'] = (data['SMA_50'] > data['SMA_200']).astype(int)
    # +1 = 50 SMA just crossed above 200 SMA -> BUY
    # -1 = 50 SMA just crossed below 200 SMA -> SELL
    data['Signal']   = data['Position'].diff()
    return data


# ── 4. BACKTEST ENGINE ────────────────────────────────────────────────────────

def run_backtest(data: pd.DataFrame):
    """
    Walk through every signal in chronological order.
    At each BUY  signal: close any open SHORT, then open a LONG.
    At each SELL signal: close any open LONG,  then open a SHORT.

    Returns
    -------
    trades : list of dicts — one entry per completed trade
    equity : list of floats — balance after each closed trade (plus opening value)
    """
    signal_rows = data[data['Signal'].isin([1.0, -1.0])]

    trades     = []
    equity     = [INITIAL_BALANCE]
    balance    = INITIAL_BALANCE
    open_trade = None   # dict with keys: direction, entry_price, entry_time

    for timestamp, row in signal_rows.iterrows():
        signal = row['Signal']
        price  = row['Close']

        # ── Close the open trade if there is one ──
        if open_trade is not None:
            direction   = open_trade['direction']
            entry_price = open_trade['entry_price']
            entry_time  = open_trade['entry_time']

            if direction == 'LONG':
                pips = (price - entry_price) / PIP_SIZE
            else:
                pips = (entry_price - price) / PIP_SIZE

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

        # ── Open a new trade in the direction of the signal ──
        open_trade = {
            'direction'  : 'LONG' if signal == 1.0 else 'SHORT',
            'entry_price': price,
            'entry_time' : timestamp,
        }

    return trades, equity


# ── 5. PRINT SUMMARY ──────────────────────────────────────────────────────────

def print_summary(trades: list, equity: list):
    if not trades:
        print("No completed trades.")
        return

    df        = pd.DataFrame(trades)
    winners   = df[df['P&L (USD)'] > 0]
    losers    = df[df['P&L (USD)'] <= 0]
    total_pnl = df['P&L (USD)'].sum()
    win_rate  = len(winners) / len(df) * 100
    avg_win   = winners['P&L (USD)'].mean() if not winners.empty else 0.0
    avg_loss  = losers['P&L (USD)'].mean()  if not losers.empty  else 0.0
    best      = df['P&L (USD)'].max()
    worst     = df['P&L (USD)'].min()

    # Peak-to-trough drawdown
    peak      = INITIAL_BALANCE
    max_dd    = 0.0
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100
        if dd > max_dd:
            max_dd = dd

    print()
    print("=" * 46)
    print("           BACKTEST RESULTS")
    print("=" * 46)
    print(f"  Symbol          : {SYMBOL}  (M15, {MONTHS} months)")
    print(f"  Lot Size        : {LOT_SIZE}  (${PIP_VALUE_USD:.2f}/pip)")
    print("-" * 46)
    print(f"  Initial Balance : ${INITIAL_BALANCE:>10,.2f}")
    print(f"  Final Balance   : ${equity[-1]:>10,.2f}")
    print(f"  Total P&L       : ${total_pnl:>+10,.2f}")
    print(f"  Max Drawdown    : {max_dd:>9.1f}%")
    print("-" * 46)
    print(f"  Total Trades    : {len(df):>10}")
    print(f"  Winning Trades  : {len(winners):>10}")
    print(f"  Losing Trades   : {len(losers):>10}")
    print(f"  Win Rate        : {win_rate:>9.1f}%")
    print("-" * 46)
    print(f"  Avg Win         : ${avg_win:>+10,.2f}")
    print(f"  Avg Loss        : ${avg_loss:>+10,.2f}")
    print(f"  Best Trade      : ${best:>+10,.2f}")
    print(f"  Worst Trade     : ${worst:>+10,.2f}")
    print("=" * 46)
    print()


# ── 6. PLOT EQUITY CURVE + PER-TRADE P&L ─────────────────────────────────────

def plot_results(trades: list, equity: list):
    df         = pd.DataFrame(trades)
    trade_nums = list(range(len(equity)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # ── Equity curve ──
    curve_color = '#2ca02c' if equity[-1] >= INITIAL_BALANCE else '#d62728'
    ax1.plot(trade_nums, equity, color=curve_color, linewidth=2, zorder=3)
    ax1.axhline(INITIAL_BALANCE, color='grey', linewidth=1,
                linestyle='--', label=f'Starting balance  ${INITIAL_BALANCE:,.0f}')
    ax1.fill_between(trade_nums, INITIAL_BALANCE, equity,
                     where=[e >= INITIAL_BALANCE for e in equity],
                     alpha=0.15, color='green', label='Profit zone')
    ax1.fill_between(trade_nums, INITIAL_BALANCE, equity,
                     where=[e < INITIAL_BALANCE for e in equity],
                     alpha=0.15, color='red', label='Loss zone')
    ax1.set_title('Equity Curve — EURUSD M15  |  50/200 SMA Crossover  |  0.1 Lots',
                  fontsize=13, fontweight='bold')
    ax1.set_xlabel('Trade Number')
    ax1.set_ylabel('Account Balance (USD)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Per-trade P&L bars ──
    bar_colors = ['#2ca02c' if p > 0 else '#d62728' for p in df['P&L (USD)']]
    ax2.bar(df['Trade #'], df['P&L (USD)'], color=bar_colors, alpha=0.8, width=0.7)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_title('Individual Trade P&L', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Trade Number')
    ax2.set_ylabel('P&L (USD)')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()

    output_dir  = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'backtest_equity_curve.png')
    plt.savefig(output_path, dpi=120)
    print(f"Chart saved  : {output_path}")


# ── 7. SAVE TRADE LOG ─────────────────────────────────────────────────────────

def save_trade_log(trades: list):
    df          = pd.DataFrame(trades)
    output_dir  = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'trade_log.csv')
    df.to_csv(output_path, index=False)
    print(f"Trade log    : {output_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

connect_mt5()

print(f"Fetching {MONTHS} months of {SYMBOL} M15 data...")
raw = fetch_data()
mt5.shutdown()
print(f"Fetched {len(raw):,} bars.  Disconnected from MT5.\n")

data           = apply_strategy(raw)
trades, equity = run_backtest(data)

print_summary(trades, equity)
save_trade_log(trades)
plot_results(trades, equity)
print("\nDone!")
