"""
Forex Bot - Intraday Moving Average Crossover Strategy (MT5)

Connects to a running MetaTrader 5 terminal, fetches 6 months of EURUSD
data on both the M15 and M5 timeframes, then runs the same 50/200-period
SMA crossover strategy we built on the daily chart.

Note: On intraday charts "periods" replace "days":
  M15  50-period SMA  = last 50 x 15 min = ~12.5 hours of price history
  M15 200-period SMA  = last 200 x 15 min = ~50 hours (~2 trading days)
  M5   50-period SMA  = last 50 x  5 min  = ~4 hours of price history
  M5  200-period SMA  = last 200 x  5 min = ~16.5 hours (~1 trading day)

Requirements:
  - MetaTrader 5 terminal must be OPEN and LOGGED IN before running this script
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


# ── 1. CONNECT TO MT5 ────────────────────────────────────────────────────────

def connect_mt5():
    """Start connection to the MT5 terminal. Exits if it fails."""
    print("Connecting to MetaTrader 5...")

    if not mt5.initialize():
        print(f"ERROR: Could not connect to MT5 — {mt5.last_error()}")
        print("Make sure MetaTrader 5 is open and you are logged in.")
        sys.exit(1)

    info = mt5.terminal_info()
    account = mt5.account_info()
    print(f"Connected to: {info.name}")
    print(f"Account:      {account.login}  ({account.server})")
    print(f"Balance:      {account.balance} {account.currency}\n")


# ── 2. FETCH INTRADAY DATA FROM MT5 ─────────────────────────────────────────

def fetch_data(symbol: str, timeframe: int, months: int) -> pd.DataFrame:
    """
    Pull historical OHLCV bars from MT5 for the given symbol and timeframe.

    Parameters
    ----------
    symbol    : e.g. 'EURUSD'
    timeframe : mt5.TIMEFRAME_M5 or mt5.TIMEFRAME_M15
    months    : how many months of history to fetch

    Returns a DataFrame with a DatetimeIndex and a 'Close' column.
    """
    date_to   = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=months * 30)

    rates = mt5.copy_rates_range(symbol, timeframe, date_from, date_to)

    if rates is None or len(rates) == 0:
        print(f"ERROR: No data returned for {symbol} — {mt5.last_error()}")
        print("Check that the symbol is available in your MT5 terminal.")
        mt5.shutdown()
        sys.exit(1)

    df = pd.DataFrame(rates)

    # MT5 timestamps are in UTC seconds — convert to a proper DatetimeIndex
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df.rename(columns={'close': 'Close'}, inplace=True)

    return df[['Close']]


# ── 3. MOVING AVERAGE CROSSOVER STRATEGY ─────────────────────────────────────

def apply_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 50/200-period SMAs and find crossover buy/sell signals.
    Returns a new DataFrame with SMA columns and a Signal column added.
    """
    data = df.copy()

    data['SMA_50']  = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

    # Drop leading rows where the 200-period MA hasn't warmed up yet
    data.dropna(inplace=True)

    # 1 when short MA is above long MA, 0 when below
    data['Position'] = (data['SMA_50'] > data['SMA_200']).astype(int)

    # Difference between consecutive positions reveals crossover moments:
    #  +1 = just crossed UP   -> BUY  (Golden Cross)
    #  -1 = just crossed DOWN -> SELL (Death Cross)
    data['Signal'] = data['Position'].diff()

    return data


# ── 4. PRINT SIGNAL SUMMARY ───────────────────────────────────────────────────

def print_signals(data: pd.DataFrame, label: str):
    """Print a readable summary of all detected signals."""
    buy_signals  = data[data['Signal'] ==  1]
    sell_signals = data[data['Signal'] == -1]

    print(f"  {label}: {len(data)} bars analysed")
    print(f"    BUY  signals: {len(buy_signals)}")
    print(f"    SELL signals: {len(sell_signals)}")

    if not buy_signals.empty:
        print("    BUY dates:")
        for ts, row in buy_signals.iterrows():
            print(f"      {ts.strftime('%Y-%m-%d %H:%M')}  price: {row['Close']:.5f}")

    if not sell_signals.empty:
        print("    SELL dates:")
        for ts, row in sell_signals.iterrows():
            print(f"      {ts.strftime('%Y-%m-%d %H:%M')}  price: {row['Close']:.5f}")

    print()


# ── 5. PLOT ONE TIMEFRAME ON A GIVEN AXES OBJECT ─────────────────────────────

def plot_timeframe(ax, data: pd.DataFrame, title: str):
    """Draw price, both MAs, and buy/sell markers on the given axes."""
    buy_signals  = data[data['Signal'] ==  1]
    sell_signals = data[data['Signal'] == -1]

    # Price (thin line — lots of bars on intraday charts)
    ax.plot(data.index, data['Close'],
            label='EURUSD', color='#1f77b4', linewidth=0.8, alpha=0.7)

    # Moving averages
    ax.plot(data.index, data['SMA_50'],
            label='50-period SMA', color='#ff7f0e', linewidth=1.5, linestyle='--')
    ax.plot(data.index, data['SMA_200'],
            label='200-period SMA', color='#d62728', linewidth=1.5, linestyle='-.')

    # Signal markers
    if not buy_signals.empty:
        ax.scatter(buy_signals.index, buy_signals['Close'],
                   marker='^', color='green', s=120, zorder=5,
                   label=f'BUY (Golden Cross) x{len(buy_signals)}')

    if not sell_signals.empty:
        ax.scatter(sell_signals.index, sell_signals['Close'],
                   marker='v', color='red', s=120, zorder=5,
                   label=f'SELL (Death Cross) x{len(sell_signals)}')

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('Price (USD per 1 EUR)', fontsize=10)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Readable date labels on the x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')


# ── 6. MAIN ───────────────────────────────────────────────────────────────────

SYMBOL  = 'EURUSD'
MONTHS  = 6

connect_mt5()

print(f"Fetching {MONTHS} months of {SYMBOL} data ...\n")

# Fetch both timeframes
raw_m15 = fetch_data(SYMBOL, mt5.TIMEFRAME_M15, MONTHS)
raw_m5  = fetch_data(SYMBOL, mt5.TIMEFRAME_M5,  MONTHS)

print(f"M15 bars downloaded : {len(raw_m15):,}")
print(f"M5  bars downloaded : {len(raw_m5):,}\n")

# Apply the crossover strategy to each
data_m15 = apply_strategy(raw_m15)
data_m5  = apply_strategy(raw_m5)

# Print signal summaries
print("Signal summary:")
print_signals(data_m15, 'M15 (15-min chart)')
print_signals(data_m5,  'M5  ( 5-min chart)')

# Disconnect from MT5 — we have all the data we need
mt5.shutdown()
print("Disconnected from MT5.\n")

# ── 7. SIDE-BY-SIDE CHART ─────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=False)

plot_timeframe(
    ax1, data_m15,
    f'EURUSD M15 — 50/200 SMA Crossover  ({MONTHS} months)'
)
plot_timeframe(
    ax2, data_m5,
    f'EURUSD M5  — 50/200 SMA Crossover  ({MONTHS} months)'
)

fig.suptitle('Intraday Moving Average Crossover Strategy — MT5 Data',
             fontsize=15, fontweight='bold', y=1.01)
fig.tight_layout()

# Save
output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'intraday_strategy.png')
plt.savefig(output_path, dpi=120, bbox_inches='tight')
print(f"Chart saved to: {output_path}")
print("Done! Open the file in an image viewer to see both timeframe charts.")
