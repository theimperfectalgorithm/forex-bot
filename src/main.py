"""
Simple Forex Bot - Download and plot EURUSD data
"""

import os
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Calculate date range (1 year back from today)
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

print(f"Downloading EURUSD data from {start_date.date()} to {end_date.date()}...")

# Download EURUSD data (using EUR=X ticker)
# yfinance uses EUR=X for EURUSD exchange rate
data = yf.download('EUR=X', start=start_date, end=end_date, progress=False)

print(f"Downloaded {len(data)} records")
print("\nFirst few rows:")
print(data.head())
print("\nData statistics:")
print(data['Close'].describe())

# Create the plot
plt.figure(figsize=(14, 6))
plt.plot(data.index, data['Close'], linewidth=1.5, color='#1f77b4')
plt.title('EURUSD Exchange Rate - Last 12 Months', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'eurusd_plot.png')
plt.savefig(output_path, dpi=100)
print(f"\nPlot saved to {output_path}")

print("Done! Open the plot file in an image viewer to see the graph.")
