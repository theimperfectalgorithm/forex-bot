"""
Forex Bot - Moving Average Crossover Strategy

Strategy rules:
  BUY  signal: 50-day SMA crosses ABOVE the 200-day SMA (Golden Cross)
  SELL signal: 50-day SMA crosses BELOW the 200-day SMA (Death Cross)

We download 2 years of data so the 200-day MA has enough history to be useful.
"""

import os
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (saves to file instead of opening a window)
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ── 1. DOWNLOAD DATA ──────────────────────────────────────────────────────────
# We need 2 years so the 200-day moving average has enough data to calculate
end_date = datetime.now()
start_date = end_date - timedelta(days=730)  # ~2 years

print(f"Downloading EURUSD data from {start_date.date()} to {end_date.date()}...")
raw = yf.download('EUR=X', start=start_date, end=end_date, progress=False)

# Flatten multi-level columns that yfinance sometimes creates
raw.columns = raw.columns.get_level_values(0)

# Keep only the closing price and work with a clean copy
data = raw[['Close']].copy()
print(f"Downloaded {len(data)} trading days of data\n")

# ── 2. CALCULATE MOVING AVERAGES ──────────────────────────────────────────────
# A moving average smooths out daily noise by averaging the last N closing prices.
# 50-day  = short-term trend
# 200-day = long-term trend
data['SMA_50']  = data['Close'].rolling(window=50).mean()
data['SMA_200'] = data['Close'].rolling(window=200).mean()

# Drop rows where the 200-day MA couldn't be calculated yet (first 199 rows)
data.dropna(inplace=True)
print(f"Rows with both MAs available: {len(data)}")

# ── 3. GENERATE BUY / SELL SIGNALS ───────────────────────────────────────────
# Position: 1 means "50MA is above 200MA", 0 means "50MA is below 200MA"
data['Position'] = (data['SMA_50'] > data['SMA_200']).astype(int)

# Signal: look for the moment the position CHANGES
# +1 = 50MA just crossed ABOVE 200MA  → BUY
# -1 = 50MA just crossed BELOW 200MA  → SELL
data['Signal'] = data['Position'].diff()

buy_signals  = data[data['Signal'] ==  1]
sell_signals = data[data['Signal'] == -1]

print(f"\nBUY  signals found: {len(buy_signals)}")
print(f"SELL signals found: {len(sell_signals)}")

if not buy_signals.empty:
    print("\nBUY dates:")
    for date in buy_signals.index:
        print(f"  {date.date()}  ->  price: {buy_signals.loc[date, 'Close']:.5f}")

if not sell_signals.empty:
    print("\nSELL dates:")
    for date in sell_signals.index:
        print(f"  {date.date()}  ->  price: {sell_signals.loc[date, 'Close']:.5f}")

# ── 4. PLOT EVERYTHING ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 7))

# Price line
ax.plot(data.index, data['Close'],
        label='EURUSD Price', color='#1f77b4', linewidth=1.2, alpha=0.8)

# Moving average lines
ax.plot(data.index, data['SMA_50'],
        label='50-day SMA', color='#ff7f0e', linewidth=1.8, linestyle='--')
ax.plot(data.index, data['SMA_200'],
        label='200-day SMA', color='#d62728', linewidth=1.8, linestyle='-.')

# Buy signals — green upward triangles
ax.scatter(buy_signals.index, buy_signals['Close'],
           marker='^', color='green', s=150, zorder=5,
           label='BUY signal (Golden Cross)')

# Sell signals — red downward triangles
ax.scatter(sell_signals.index, sell_signals['Close'],
           marker='v', color='red', s=150, zorder=5,
           label='SELL signal (Death Cross)')

# Chart formatting
ax.set_title('EURUSD — Moving Average Crossover Strategy\n'
             '50-day SMA vs 200-day SMA',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Price (USD per 1 EUR)', fontsize=12)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()

# ── 5. SAVE THE CHART ─────────────────────────────────────────────────────────
output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'ma_crossover_strategy.png')
plt.savefig(output_path, dpi=120)
print(f"\nChart saved to: {output_path}")
print("Done! Open the file in an image viewer to see the strategy signals.")
