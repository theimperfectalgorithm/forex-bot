"""
Forex Bot - H1 Timeframe Backtest (3-Year, EURUSD)

Tests the 50/200 SMA crossover + RSI-14/60-40 filter on the H1 (1-hour)
timeframe to determine whether a higher timeframe resolves the noise
problems that caused both M15 variants to lose money over 3 years.

On H1, the 50-period SMA covers ~2 trading days and the 200-period SMA
covers ~8 trading days — a much more meaningful trend signal than the
same windows on M15 (which covered hours, not days).

Settings: EURUSD H1, ~3 years, $10,000 balance, 0.1 lots ($1/pip).

3-year M15 results for comparison:
  50/200 SMA + RSI-14/60-40 : -$43.20  (43.4% WR, 12.4% DD)
  20/50  SMA + RSI-14/60-40 : -$689.60 (39.6% WR, 14.3% DD)

Requirements:
  MetaTrader 5 must be OPEN and LOGGED IN before running this script.
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).parent.parent))
from core import data_loader

# ── SETTINGS ──────────────────────────────────────────────────────────────────

SYMBOL          = 'EURUSD'
MONTHS          = 36
INITIAL_BALANCE = 10_000.00
PIP_SIZE        = 0.0001
PIP_VALUE_USD   = 1.00
RSI_PERIOD      = 14
BUY_THRESHOLD   = 60
SELL_THRESHOLD  = 40

# M15 3-year results — hardcoded from previous runs
M15_RESULTS = [
    {
        'label'    : 'M15  50/200 SMA',
        'trades'   : 235,
        'winners'  : 102,
        'losers'   : 133,
        'win_rate' : 43.4,
        'pnl'      : -43.20,
        'balance'  : 9_956.80,
        'max_dd'   : 12.4,
        'best'     : 346.60,
        'worst'    : -361.60,
        'per_month': -1.20,
    },
    {
        'label'    : 'M15  20/50  SMA',
        'trades'   : 970,
        'winners'  : 384,
        'losers'   : 586,
        'win_rate' : 39.6,
        'pnl'      : -689.60,
        'balance'  : 9_310.40,
        'max_dd'   : 14.3,
        'best'     : 333.10,
        'worst'    : -301.40,
        'per_month': -19.19,
    },
]


# ── 1. CONNECT TO MT5 ─────────────────────────────────────────────────────────

def connect_mt5():
    if not MT5_AVAILABLE:
        print("MetaTrader5 not available -- using offline CSV data from "
              "data/historical/\n")
        return
    print("Connecting to MetaTrader 5...")
    if not mt5.initialize():
        print(f"ERROR: Could not connect — {mt5.last_error()}")
        print("Make sure MetaTrader 5 is open and logged in.")
        sys.exit(1)
    info    = mt5.terminal_info()
    account = mt5.account_info()
    print(f"Connected : {info.name}")
    print(f"Account   : {account.login}  ({account.server})\n")


# ── 2. FETCH H1 DATA ──────────────────────────────────────────────────────────

def fetch_data() -> pd.DataFrame:
    date_to   = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=MONTHS * 30)

    print(f"Requesting H1 data from {date_from.date()} to {date_to.date()} ...")
    df = data_loader.get_bars(SYMBOL, 'H1', date_from, date_to)

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


# ── 4. STRATEGY ───────────────────────────────────────────────────────────────

def apply_strategy(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data['SMA_50']  = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['RSI']     = calculate_rsi(data['Close'], RSI_PERIOD)
    data.dropna(inplace=True)

    data['Position'] = (data['SMA_50'] > data['SMA_200']).astype(int)
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
        'label'    : 'H1   50/200 SMA',
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


# ── 7. PRINT 3-WAY COMPARISON ─────────────────────────────────────────────────

def print_comparison(h1: dict):
    rows = M15_RESULTS + [h1]
    w    = 72

    print()
    print("=" * w)
    print("   FULL COMPARISON — 3 Years  |  EURUSD  |  RSI-14/60-40  |  0.1 Lots")
    print("=" * w)
    print(f"  {'Metric':<22} {'M15 50/200':>12} {'M15 20/50':>12} {'H1 50/200':>12}  Winner")
    print("-" * w)

    def best_of(values, higher_better=True):
        fn = max if higher_better else min
        return values.index(fn(values))

    metrics = [
        ('Total Trades',    [r['trades']   for r in rows], True,  '{:>12,}'),
        ('Win Rate',        [r['win_rate'] for r in rows], True,  '{:>11.1f}%'),
        ('Total P&L',       [r['pnl']      for r in rows], True,  '${:>+11,.2f}'),
        ('Avg P&L/month',   [r['per_month']for r in rows], True,  '${:>+11,.2f}'),
        ('Final Balance',   [r['balance']  for r in rows], True,  '${:>11,.2f}'),
        ('Max Drawdown',    [r['max_dd']   for r in rows], False, '{:>11.1f}%'),
        ('Best Trade',      [r['best']     for r in rows], True,  '${:>+11,.2f}'),
        ('Worst Trade',     [r['worst']    for r in rows], True,  '${:>+11,.2f}'),
    ]

    winner_labels = ['M15 50/200', 'M15 20/50', 'H1 50/200']

    for name, values, hb, fmt in metrics:
        wi      = best_of(values, hb)
        cells   = '  '.join(fmt.format(v) for v in values)
        print(f"  {name:<22}  {cells}  {winner_labels[wi]}")

    print("=" * w)
    print()
    print(f"  H1 detail:")
    print(f"    Winning trades  : {h1['winners']}  /  {h1['trades']}")
    print(f"    Losing trades   : {h1['losers']}  /  {h1['trades']}")
    print(f"    Avg win         : ${h1['avg_win']:>+,.2f}")
    print(f"    Avg loss        : ${h1['avg_loss']:>+,.2f}")
    print(f"    Data window     : ~{h1['months']:.0f} months")
    print()


# ── 8. PLOT ───────────────────────────────────────────────────────────────────

def plot_results(trades: list, equity: list, data: pd.DataFrame):
    df         = pd.DataFrame(trades)
    exit_times = pd.to_datetime(df['Exit Date'])
    full_index = [exit_times.iloc[0]] + list(exit_times)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11))

    # ── H1 equity curve ──
    curve_color = '#2ca02c' if equity[-1] >= INITIAL_BALANCE else '#d62728'
    ax1.plot(full_index, equity, color=curve_color, linewidth=2,
             label='H1 50/200 SMA (this test)', zorder=3)
    ax1.axhline(INITIAL_BALANCE, color='grey', linewidth=1, linestyle='--',
                label=f'Starting balance  ${INITIAL_BALANCE:,.0f}', zorder=1)

    # Reference lines for the two M15 final balances
    ax1.axhline(M15_RESULTS[0]['balance'], color='#ff7f0e', linewidth=1.2,
                linestyle=':', label=f"M15 50/200 final  ${M15_RESULTS[0]['balance']:,.2f}")
    ax1.axhline(M15_RESULTS[1]['balance'], color='#d62728', linewidth=1.2,
                linestyle=':', label=f"M15 20/50  final  ${M15_RESULTS[1]['balance']:,.2f}")

    ax1.fill_between(full_index, INITIAL_BALANCE, equity,
                     where=[e >= INITIAL_BALANCE for e in equity],
                     alpha=0.15, color='green', label='Profit zone')
    ax1.fill_between(full_index, INITIAL_BALANCE, equity,
                     where=[e < INITIAL_BALANCE for e in equity],
                     alpha=0.15, color='red', label='Loss zone')

    ax1.set_title(
        f'H1 50/200 SMA + RSI-{RSI_PERIOD}/{BUY_THRESHOLD}-{SELL_THRESHOLD}  '
        f'|  EURUSD H1  |  0.1 Lots  |  3 Years\n'
        f'{data.index[0].date()}  to  {data.index[-1].date()}',
        fontsize=13, fontweight='bold')
    ax1.set_ylabel('Account Balance (USD)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')

    # ── Per-trade P&L bars ──
    bar_colors = ['#2ca02c' if p > 0 else '#d62728' for p in df['P&L (USD)']]
    ax2.bar(df['Trade #'], df['P&L (USD)'], color=bar_colors, alpha=0.8, width=0.7)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_title('H1 50/200 SMA — Individual Trade P&L', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Trade Number')
    ax2.set_ylabel('P&L (USD)')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()

    output_dir  = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'h1_backtest_equity_curve.png')
    plt.savefig(output_path, dpi=120)
    print(f"Chart saved : {output_path}")


# ── 9. SAVE TRADE LOG ─────────────────────────────────────────────────────────

def save_trade_log(trades: list):
    df          = pd.DataFrame(trades)
    output_dir  = os.path.join(os.path.dirname(__file__), '..', 'data')
    output_path = os.path.join(output_dir, 'trade_log_h1.csv')
    df.to_csv(output_path, index=False)
    print(f"Trade log   : {output_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

connect_mt5()
raw = fetch_data()
if MT5_AVAILABLE:
    mt5.shutdown()

data           = apply_strategy(raw)
trades, equity = run_backtest(data)
stats          = compute_stats(trades, equity, data)

print_comparison(stats)
save_trade_log(trades)
plot_results(trades, equity, raw)
print("\nDone!")
