"""
Forex Bot - M15 Backtest with RSI Confirmation Filter

Extends the baseline SMA crossover strategy (backtest.py) by adding RSI(14)
as an entry filter to reduce low-quality trades.

Filter rules:
  BUY  signal fires only when 50 SMA crosses ABOVE 200 SMA  AND  RSI < 50
  SELL signal fires only when 50 SMA crosses BELOW 200 SMA  AND  RSI > 50

Rationale:
  A BUY crossover with RSI already above 50 means momentum has been bullish
  for a while — the move may be exhausted. Waiting for RSI < 50 means we only
  enter longs when there is still room for momentum to build upward.
  The inverse logic applies to SELL signals.

Same settings as backtest.py: $10,000 balance, 0.1 lots, EURUSD M15, 6 months.

Requirements:
  MetaTrader 5 must be OPEN and LOGGED IN before running this script.
"""

import os
import sys
from datetime import datetime, timedelta, timezone

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── SETTINGS ──────────────────────────────────────────────────────────────────

SYMBOL          = 'EURUSD'
MONTHS          = 6
INITIAL_BALANCE = 10_000.00
LOT_SIZE        = 0.1
PIP_SIZE        = 0.0001
PIP_VALUE_USD   = 1.00

# Known results from backtest.py — used for the comparison table
BASELINE = {
    'trades'  : 66,
    'winners' : 25,
    'losers'  : 41,
    'win_rate': 37.9,
    'pnl'     : 108.40,
    'balance' : 10_108.40,
    'max_dd'  : 3.8,
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


# ── 3. RSI CALCULATION ────────────────────────────────────────────────────────

def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder's RSI using exponential smoothing (alpha = 1/period).
    Values range from 0 to 100.
      > 70  = overbought (price may be due for a pullback)
      < 30  = oversold   (price may be due for a bounce)
      ~ 50  = momentum neutral
    """
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)

    # com = period - 1 gives Wilder's smoothing factor of 1/period
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()

    rs  = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ── 4. APPLY STRATEGY WITH RSI FILTER ────────────────────────────────────────

def apply_strategy(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data['SMA_50']  = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['RSI']     = calculate_rsi(data['Close'], period=14)

    data.dropna(inplace=True)

    data['Position'] = (data['SMA_50'] > data['SMA_200']).astype(int)
    crossover        = data['Position'].diff()

    # Start with no signals, then selectively allow crossovers that pass RSI
    data['Signal'] = 0.0
    data.loc[(crossover == 1)  & (data['RSI'] < 50), 'Signal'] =  1.0
    data.loc[(crossover == -1) & (data['RSI'] > 50), 'Signal'] = -1.0

    return data


# ── 5. BACKTEST ENGINE (unchanged from backtest.py) ───────────────────────────

def run_backtest(data: pd.DataFrame):
    signal_rows = data[data['Signal'].isin([1.0, -1.0])]

    trades     = []
    equity     = [INITIAL_BALANCE]
    balance    = INITIAL_BALANCE
    open_trade = None

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


# ── 6. COMPUTE SUMMARY STATS ──────────────────────────────────────────────────

def compute_stats(trades: list, equity: list) -> dict:
    df       = pd.DataFrame(trades)
    winners  = df[df['P&L (USD)'] > 0]
    losers   = df[df['P&L (USD)'] <= 0]
    peak     = INITIAL_BALANCE
    max_dd   = 0.0
    for e in equity:
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100
        if dd > max_dd:
            max_dd = dd
    return {
        'trades'  : len(df),
        'winners' : len(winners),
        'losers'  : len(losers),
        'win_rate': len(winners) / len(df) * 100 if len(df) else 0,
        'pnl'     : round(df['P&L (USD)'].sum(), 2),
        'balance' : equity[-1],
        'avg_win' : round(winners['P&L (USD)'].mean(), 2) if not winners.empty else 0.0,
        'avg_loss': round(losers['P&L (USD)'].mean(),  2) if not losers.empty  else 0.0,
        'best'    : round(df['P&L (USD)'].max(), 2),
        'worst'   : round(df['P&L (USD)'].min(), 2),
        'max_dd'  : round(max_dd, 1),
    }


# ── 7. PRINT COMPARISON TABLE ─────────────────────────────────────────────────

def print_comparison(new: dict):
    b = BASELINE

    def delta(new_val, old_val, higher_better=True):
        """Return a +/- change string."""
        diff = new_val - old_val
        sign = '+' if diff >= 0 else ''
        arrow = ('  (better)' if (diff > 0) == higher_better and diff != 0
                 else ('  (worse)'  if diff != 0 else ''))
        return f"{sign}{diff:.1f}{arrow}"

    print()
    print("=" * 62)
    print("         STRATEGY COMPARISON  —  M15  EURUSD  0.1 Lots")
    print("=" * 62)
    print(f"  {'Metric':<22}  {'Baseline (SMA only)':>16}  {'RSI Filter':>12}")
    print("-" * 62)
    print(f"  {'Total Trades':<22}  {b['trades']:>16}  {new['trades']:>12}  "
          f"{delta(new['trades'], b['trades'], higher_better=False)}")
    print(f"  {'Winning Trades':<22}  {b['winners']:>16}  {new['winners']:>12}")
    print(f"  {'Losing Trades':<22}  {b['losers']:>16}  {new['losers']:>12}")
    print(f"  {'Win Rate':<22}  {b['win_rate']:>15.1f}%  {new['win_rate']:>11.1f}%  "
          f"{delta(new['win_rate'], b['win_rate'])}")
    print(f"  {'Total P&L':<22}  ${b['pnl']:>+14,.2f}  ${new['pnl']:>+10,.2f}  "
          f"{delta(new['pnl'], b['pnl'])}")
    print(f"  {'Final Balance':<22}  ${b['balance']:>14,.2f}  ${new['balance']:>10,.2f}")
    print(f"  {'Max Drawdown':<22}  {b['max_dd']:>15.1f}%  {new['max_dd']:>11.1f}%  "
          f"{delta(new['max_dd'], b['max_dd'], higher_better=False)}")
    print(f"  {'Avg Win':<22}  ${new['avg_win']:>+10,.2f}")
    print(f"  {'Avg Loss':<22}  ${new['avg_loss']:>+10,.2f}")
    print(f"  {'Best Trade':<22}  ${new['best']:>+10,.2f}")
    print(f"  {'Worst Trade':<22}  ${new['worst']:>+10,.2f}")
    print("=" * 62)
    print()


# ── 8. PLOT EQUITY CURVE ──────────────────────────────────────────────────────

def plot_results(trades: list, equity: list):
    df         = pd.DataFrame(trades)
    trade_nums = list(range(len(equity)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # ── Equity curve — new strategy vs baseline ──
    curve_color = '#2ca02c' if equity[-1] >= INITIAL_BALANCE else '#d62728'
    ax1.plot(trade_nums, equity, color=curve_color, linewidth=2,
             label='RSI-filtered strategy', zorder=3)
    ax1.axhline(INITIAL_BALANCE, color='grey', linewidth=1, linestyle='--',
                label=f'Starting balance  ${INITIAL_BALANCE:,.0f}')
    ax1.axhline(BASELINE['balance'], color='#ff7f0e', linewidth=1,
                linestyle=':', label=f"Baseline final  ${BASELINE['balance']:,.2f}")
    ax1.fill_between(trade_nums, INITIAL_BALANCE, equity,
                     where=[e >= INITIAL_BALANCE for e in equity],
                     alpha=0.15, color='green', label='Profit zone')
    ax1.fill_between(trade_nums, INITIAL_BALANCE, equity,
                     where=[e < INITIAL_BALANCE for e in equity],
                     alpha=0.15, color='red', label='Loss zone')
    ax1.set_title('Equity Curve — EURUSD M15  |  50/200 SMA  +  RSI(14) Filter  |  0.1 Lots',
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
    output_path = os.path.join(output_dir, 'backtest_rsi_equity_curve.png')
    plt.savefig(output_path, dpi=120)
    print(f"Chart saved  : {output_path}")


# ── 9. SAVE TRADE LOG ─────────────────────────────────────────────────────────

def save_trade_log(trades: list):
    df          = pd.DataFrame(trades)
    output_dir  = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'trade_log_rsi.csv')
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
stats          = compute_stats(trades, equity)

print_comparison(stats)
save_trade_log(trades)
plot_results(trades, equity)
print("\nDone!")
