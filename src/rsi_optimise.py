"""
Forex Bot - RSI Parameter Optimisation

Tests five RSI configurations on top of the 50/200 SMA crossover strategy
to find which RSI period and threshold produces the best backtest results.

Variants tested:
  Label          RSI period  Buy threshold  Sell threshold
  ------------  ----------  -------------  --------------
  RSI-14/50  *      14         < 50           > 50    (current baseline)
  RSI-7/50           7         < 50           > 50    (faster RSI)
  RSI-21/50         21         < 50           > 50    (slower RSI)
  RSI-14/60-40      14         < 60           > 40    (more permissive)
  RSI-14/40-60      14         < 40           > 60    (contrarian / mean-reversion)

All variants use the same data: EURUSD M15, 6 months from MT5.
Same trade settings: $10,000 starting balance, 0.1 lots.

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
PIP_SIZE        = 0.0001
PIP_VALUE_USD   = 1.00

# All five RSI variants to test, in order
VARIANTS = [
    {'label': 'RSI-14/50 (current)', 'period': 14, 'buy_th': 50, 'sell_th': 50},
    {'label': 'RSI-7/50  (faster)',  'period':  7, 'buy_th': 50, 'sell_th': 50},
    {'label': 'RSI-21/50 (slower)',  'period': 21, 'buy_th': 50, 'sell_th': 50},
    {'label': 'RSI-14/60-40 (permissive)', 'period': 14, 'buy_th': 60, 'sell_th': 40},
    {'label': 'RSI-14/40-60 (contrarian)', 'period': 14, 'buy_th': 40, 'sell_th': 60},
]


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


# ── 2. FETCH M15 DATA (once, reused for all variants) ─────────────────────────

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


# ── 3. RSI (Wilder's smoothing) ───────────────────────────────────────────────

def calculate_rsi(close: pd.Series, period: int) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ── 4. STRATEGY — parameterised by RSI settings ───────────────────────────────

def apply_strategy(df: pd.DataFrame, period: int,
                   buy_th: float, sell_th: float) -> pd.DataFrame:
    """
    50/200 SMA crossover filtered by RSI.
      BUY  fires when crossover is up   AND RSI < buy_th
      SELL fires when crossover is down AND RSI > sell_th
    """
    data = df.copy()
    data['SMA_50']  = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['RSI']     = calculate_rsi(data['Close'], period)
    data.dropna(inplace=True)

    data['Position'] = (data['SMA_50'] > data['SMA_200']).astype(int)
    crossover        = data['Position'].diff()

    data['Signal'] = 0.0
    data.loc[(crossover == 1)  & (data['RSI'] < buy_th),  'Signal'] =  1.0
    data.loc[(crossover == -1) & (data['RSI'] > sell_th), 'Signal'] = -1.0
    return data


# ── 5. BACKTEST ENGINE ────────────────────────────────────────────────────────

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

def compute_stats(trades: list, equity: list) -> dict:
    if not trades:
        return {'trades': 0, 'winners': 0, 'losers': 0, 'win_rate': 0.0,
                'pnl': 0.0, 'balance': INITIAL_BALANCE, 'max_dd': 0.0}

    df      = pd.DataFrame(trades)
    winners = df[df['P&L (USD)'] > 0]
    losers  = df[df['P&L (USD)'] <= 0]

    peak   = INITIAL_BALANCE
    max_dd = 0.0
    for e in equity:
        peak  = max(peak, e)
        max_dd = max(max_dd, (peak - e) / peak * 100)

    return {
        'trades'  : len(df),
        'winners' : len(winners),
        'losers'  : len(losers),
        'win_rate': round(len(winners) / len(df) * 100, 1),
        'pnl'     : round(df['P&L (USD)'].sum(), 2),
        'balance' : equity[-1],
        'max_dd'  : round(max_dd, 1),
    }


# ── 7. PRINT COMPARISON TABLE ─────────────────────────────────────────────────

def print_comparison(results: list):
    # Also include the no-RSI baseline for context
    no_rsi = {'label': 'No RSI (SMA only)',
               'stats': {'trades': 66, 'winners': 25, 'losers': 41,
                         'win_rate': 37.9, 'pnl': 108.40,
                         'balance': 10_108.40, 'max_dd': 3.8}}

    rows = [no_rsi] + results

    print()
    print("=" * 80)
    print("              RSI OPTIMISATION RESULTS  —  EURUSD M15  0.1 Lots")
    print("=" * 80)
    print(f"  {'Variant':<32}  {'Trades':>6}  {'Win%':>6}  "
          f"{'P&L':>9}  {'Balance':>10}  {'MaxDD':>6}")
    print("-" * 80)

    best_pnl = max(r['stats']['pnl'] for r in results)

    for row in rows:
        s     = row['stats']
        label = row['label']
        marker = '  <-- BEST' if s['pnl'] == best_pnl else ''
        print(f"  {label:<32}  {s['trades']:>6}  {s['win_rate']:>5.1f}%  "
              f"${s['pnl']:>+8,.2f}  ${s['balance']:>9,.2f}  "
              f"{s['max_dd']:>5.1f}%{marker}")

    print("=" * 80)
    print()

    # Identify and explain the winner
    best = max(results, key=lambda r: r['stats']['pnl'])
    print(f"  Best variant: {best['label']}")
    print(f"  P&L: ${best['stats']['pnl']:+,.2f}  |  "
          f"Win rate: {best['stats']['win_rate']:.1f}%  |  "
          f"Trades: {best['stats']['trades']}  |  "
          f"Max DD: {best['stats']['max_dd']:.1f}%")
    print()


# ── 8. PLOT ALL EQUITY CURVES ─────────────────────────────────────────────────

def plot_all(results: list):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#d62728']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # ── All equity curves on one chart ──
    ax1.axhline(INITIAL_BALANCE, color='grey', linewidth=1,
                linestyle='--', label='Starting balance', zorder=1)
    for i, r in enumerate(results):
        eq = r['equity']
        ax1.plot(range(len(eq)), eq, linewidth=1.8,
                 color=colors[i], label=r['label'], zorder=2)

    ax1.set_title('Equity Curves — All RSI Variants  |  EURUSD M15  |  0.1 Lots',
                  fontsize=13, fontweight='bold')
    ax1.set_xlabel('Trade Number')
    ax1.set_ylabel('Account Balance (USD)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── P&L bar chart for quick comparison ──
    labels = [r['label'].split('(')[0].strip() for r in results]
    pnls   = [r['stats']['pnl'] for r in results]
    bar_colors = ['#2ca02c' if p >= 0 else '#d62728' for p in pnls]

    bars = ax2.bar(labels, pnls, color=bar_colors, alpha=0.8, width=0.55)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.axhline(108.40, color='grey', linewidth=1, linestyle='--',
                label='No-RSI baseline (+$108.40)')

    for bar, val in zip(bars, pnls):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (5 if val >= 0 else -12),
                 f'${val:+,.0f}', ha='center', va='bottom', fontsize=9)

    ax2.set_title('Total P&L by RSI Variant', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Total P&L (USD)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')

    fig.tight_layout()

    output_dir  = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'rsi_optimisation.png')
    plt.savefig(output_path, dpi=120)
    print(f"Chart saved : {output_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

connect_mt5()
print(f"Fetching {MONTHS} months of {SYMBOL} M15 data (once, shared across all variants)...")
raw = fetch_data()
mt5.shutdown()
print(f"Fetched {len(raw):,} bars.  Disconnected from MT5.\n")

results = []
for v in VARIANTS:
    print(f"Running: {v['label']} ...")
    data           = apply_strategy(raw, v['period'], v['buy_th'], v['sell_th'])
    trades, equity = run_backtest(data)
    stats          = compute_stats(trades, equity)
    results.append({'label': v['label'], 'stats': stats, 'equity': equity})

print_comparison(results)
plot_all(results)
print("\nDone!")
