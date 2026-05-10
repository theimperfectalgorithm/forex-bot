"""
Forex Bot - 3-Year Stress Test (EURUSD M15)

Runs the winning RSI-14/60-40 strategy over the maximum available
M15 history from MT5 (~3 years, from May 2023) to check whether the
6-month backtest result holds up or was a lucky window.

Strategy: 50/200 SMA crossover + RSI-14 filter
  BUY  when 50 SMA crosses above 200 SMA  AND  RSI < 60
  SELL when 50 SMA crosses below 200 SMA  AND  RSI > 40

Settings: EURUSD M15, $10,000 starting balance, 0.1 lots ($1/pip)

Known 6-month baseline (RSI-14/60-40):
  Trades: 27  |  Win rate: 51.9%  |  P&L: +$388.50  |  Max DD: 2.8%

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
MONTHS          = 36           # Request 3 years; MT5 will return what it has
INITIAL_BALANCE = 10_000.00
PIP_SIZE        = 0.0001
PIP_VALUE_USD   = 1.00

RSI_PERIOD      = 14
BUY_THRESHOLD   = 60          # Best config from rsi_optimise.py
SELL_THRESHOLD  = 40

# 6-month results to compare against
BASELINE_6M = {
    'months'  : 6,
    'trades'  : 27,
    'win_rate': 51.9,
    'pnl'     : 388.50,
    'balance' : 10_388.50,
    'max_dd'  : 2.8,
    'best'    : None,
    'worst'   : None,
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


# ── 2. FETCH DATA ─────────────────────────────────────────────────────────────

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

    actual_start = df.index[0].date()
    actual_end   = df.index[-1].date()
    months_actual = (df.index[-1] - df.index[0]).days / 30
    print(f"Received   : {len(df):,} bars  "
          f"({actual_start} to {actual_end},  ~{months_actual:.1f} months)\n")

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
        'months'    : round(months, 1),
        'trades'    : len(df),
        'winners'   : len(winners),
        'losers'    : len(losers),
        'win_rate'  : round(len(winners) / len(df) * 100, 1) if len(df) else 0,
        'pnl'       : round(df['P&L (USD)'].sum(), 2),
        'balance'   : equity[-1],
        'max_dd'    : round(max_dd, 1),
        'avg_win'   : round(winners['P&L (USD)'].mean(), 2) if not winners.empty else 0,
        'avg_loss'  : round(losers['P&L (USD)'].mean(),  2) if not losers.empty  else 0,
        'best'      : round(df['P&L (USD)'].max(), 2),
        'worst'     : round(df['P&L (USD)'].min(), 2),
        'per_month' : round(df['P&L (USD)'].sum() / months, 2) if months else 0,
    }


# ── 7. PRINT RESULTS ──────────────────────────────────────────────────────────

def print_results(s: dict):
    b = BASELINE_6M

    def chg(new, old, higher_better=True):
        diff = new - old
        tag  = 'better' if (diff > 0) == higher_better and diff != 0 else \
               ('worse' if diff != 0 else 'same')
        return f"{'+' if diff >= 0 else ''}{diff:.1f}  ({tag})"

    print()
    print("=" * 58)
    print("          3-YEAR STRESS TEST RESULTS")
    print(f"     RSI-{RSI_PERIOD}/{BUY_THRESHOLD}-{SELL_THRESHOLD}  |  "
          f"EURUSD M15  |  0.1 Lots")
    print("=" * 58)
    print(f"  {'Metric':<26}  {'6 months':>10}  {'3 years':>10}")
    print("-" * 58)
    print(f"  {'Data window':<26}  {'~6 months':>10}  "
          f"  ~{s['months']:.0f} mths")
    print(f"  {'Total Trades':<26}  {b['trades']:>10}  {s['trades']:>10}  "
          f"{chg(s['trades'], b['trades'], higher_better=True)}")
    print(f"  {'Winning Trades':<26}  {'':>10}  {s['winners']:>10}")
    print(f"  {'Losing Trades':<26}  {'':>10}  {s['losers']:>10}")
    print(f"  {'Win Rate':<26}  {b['win_rate']:>9.1f}%  {s['win_rate']:>9.1f}%  "
          f"{chg(s['win_rate'], b['win_rate'])}")
    print(f"  {'Total P&L':<26}  ${b['pnl']:>+9,.2f}  ${s['pnl']:>+9,.2f}  "
          f"{chg(s['pnl'], b['pnl'])}")
    print(f"  {'Avg P&L per month':<26}  "
          f"${b['pnl']/6:>+9,.2f}  ${s['per_month']:>+9,.2f}")
    print(f"  {'Final Balance':<26}  ${b['balance']:>9,.2f}  ${s['balance']:>9,.2f}")
    print(f"  {'Max Drawdown':<26}  {b['max_dd']:>9.1f}%  {s['max_dd']:>9.1f}%  "
          f"{chg(s['max_dd'], b['max_dd'], higher_better=False)}")
    print("-" * 58)
    print(f"  {'Avg Win':<26}  {'':>10}  ${s['avg_win']:>+9,.2f}")
    print(f"  {'Avg Loss':<26}  {'':>10}  ${s['avg_loss']:>+9,.2f}")
    print(f"  {'Best Single Trade':<26}  {'':>10}  ${s['best']:>+9,.2f}")
    print(f"  {'Worst Single Trade':<26}  {'':>10}  ${s['worst']:>+9,.2f}")
    print("=" * 58)
    print()


# ── 8. PLOT EQUITY CURVE ──────────────────────────────────────────────────────

def plot_equity(trades: list, equity: list, data: pd.DataFrame):
    df = pd.DataFrame(trades)

    # Build a time-indexed equity series for a proper date x-axis
    exit_times = pd.to_datetime(df['Exit Date'])
    eq_series  = pd.Series(equity[1:], index=exit_times)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # ── Equity curve ──
    curve_color = '#2ca02c' if equity[-1] >= INITIAL_BALANCE else '#d62728'
    full_index  = [exit_times.iloc[0]] + list(exit_times)
    ax1.plot(full_index, equity, color=curve_color, linewidth=1.8, zorder=3)
    ax1.axhline(INITIAL_BALANCE, color='grey', linewidth=1, linestyle='--',
                label=f'Starting balance  ${INITIAL_BALANCE:,.0f}')
    ax1.fill_between(full_index, INITIAL_BALANCE, equity,
                     where=[e >= INITIAL_BALANCE for e in equity],
                     alpha=0.15, color='green', label='Profit zone')
    ax1.fill_between(full_index, INITIAL_BALANCE, equity,
                     where=[e < INITIAL_BALANCE for e in equity],
                     alpha=0.15, color='red', label='Loss zone')
    ax1.set_title(
        f'3-Year Stress Test — EURUSD M15  |  '
        f'RSI-{RSI_PERIOD}/{BUY_THRESHOLD}-{SELL_THRESHOLD}  |  0.1 Lots\n'
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
    ax2.set_title('Individual Trade P&L', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Trade Number')
    ax2.set_ylabel('P&L (USD)')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()

    output_dir  = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'stress_test_equity_curve.png')
    plt.savefig(output_path, dpi=120)
    print(f"Chart saved : {output_path}")


# ── 9. SAVE TRADE LOG ─────────────────────────────────────────────────────────

def save_trade_log(trades: list):
    df          = pd.DataFrame(trades)
    output_dir  = os.path.join(os.path.dirname(__file__), '..', 'data')
    output_path = os.path.join(output_dir, 'trade_log_stress_test.csv')
    df.to_csv(output_path, index=False)
    print(f"Trade log   : {output_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

connect_mt5()
raw = fetch_data()
mt5.shutdown()

data           = apply_strategy(raw)
trades, equity = run_backtest(data)
stats          = compute_stats(trades, equity, data)

print_results(stats)
save_trade_log(trades)
plot_equity(trades, equity, data)
print("\nDone!")
