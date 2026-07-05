"""
Forex Bot - ATR-Based SL/TP Backtest (Episode 11)

Tests the locked Episode 10 H1 50/200 SMA crossover + RSI-14/60-40 strategy
with a volatility-adaptive ATR(14) stop loss / take profit, instead of the
fixed 50/100 pip SL/TP -- entry logic and RSI filter are unchanged.

  Fixed (Episode 10, locked) : SL = 50 pips        TP = 100 pips
  ATR (this test)            : SL = 1.5 x ATR(14)   TP = 3.0 x ATR(14)

Both backtests run on the *same* fetched H1 dataset and the *same* signal
sequence (signal generation does not depend on SL/TP), so the only variable
under test is how the SL/TP distance is sized at entry.

Settings: EURUSD H1, ~3 years, $10,000 balance, 0.1 lots ($1/pip).

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

# Fixed SL/TP (Episode 10, locked)
FIXED_SL_PIPS = 50
FIXED_TP_PIPS = 100
FIXED_SL_DIST = FIXED_SL_PIPS * PIP_SIZE
FIXED_TP_DIST = FIXED_TP_PIPS * PIP_SIZE

# ATR-based SL/TP variants under test (period, sl_mult, tp_mult, label)
ATR_PERIOD = 14
ATR_VARIANTS = [
    {'label': 'ATR 1.5x/3.0x', 'sl_mult': 1.5, 'tp_mult': 3.0},
    {'label': 'ATR 3.0x/6.0x', 'sl_mult': 3.0, 'tp_mult': 6.0},
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


# ── 2. FETCH H1 OHLC DATA ──────────────────────────────────────────────────────

def fetch_data() -> pd.DataFrame:
    date_to   = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=MONTHS * 30)

    print(f"Requesting H1 data from {date_from.date()} to {date_to.date()} ...")
    rates = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_H1, date_from, date_to)

    if rates is None or len(rates) == 0:
        print(f"ERROR: No data returned — {mt5.last_error()}")
        mt5.shutdown()
        sys.exit(1)

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low'}, inplace=True)

    months_actual = (df.index[-1] - df.index[0]).days / 30
    print(f"Received   : {len(df):,} bars  "
          f"({df.index[0].date()} to {df.index[-1].date()},  "
          f"~{months_actual:.1f} months)\n")
    return df[['Close', 'High', 'Low']]


# ── 3. INDICATORS: RSI + ATR ──────────────────────────────────────────────────

def calculate_rsi(close: pd.Series, period: int) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Wilder's ATR: smoothed average of True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# ── 4. STRATEGY (identical entry/filter logic to locked Episode 10) ─────────

def apply_strategy(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data['SMA_50']  = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['RSI']     = calculate_rsi(data['Close'], RSI_PERIOD)
    data['ATR']     = calculate_atr(data['High'], data['Low'], data['Close'], ATR_PERIOD)
    data.dropna(inplace=True)

    data['Position'] = (data['SMA_50'] > data['SMA_200']).astype(int)
    crossover        = data['Position'].diff()

    data['Signal'] = 0.0
    data.loc[(crossover == 1)  & (data['RSI'] < BUY_THRESHOLD),  'Signal'] =  1.0
    data.loc[(crossover == -1) & (data['RSI'] > SELL_THRESHOLD), 'Signal'] = -1.0
    return data


# ── 5. BAR-BY-BAR BACKTEST (parameterised by SL/TP sizing function) ─────────

def run_backtest(data: pd.DataFrame, sl_tp_fn):
    """
    sl_tp_fn(row) -> (sl_dist, tp_dist) in price units, evaluated at entry.
    Conservative tie-break: if both SL and TP are hit in the same bar, SL wins.
    """
    trades     = []
    equity     = [INITIAL_BALANCE]
    balance    = INITIAL_BALANCE
    open_trade = None

    for timestamp, row in data.iterrows():
        signal = row['Signal']
        bar_hi = row['High']
        bar_lo = row['Low']
        bar_cl = row['Close']

        if open_trade is not None:
            direction   = open_trade['direction']
            entry_price = open_trade['entry_price']
            entry_time  = open_trade['entry_time']
            sl_price    = open_trade['sl']
            tp_price    = open_trade['tp']

            close_price  = None
            close_reason = None

            if direction == 'LONG':
                sl_hit = bar_lo <= sl_price
                tp_hit = bar_hi >= tp_price
            else:
                sl_hit = bar_hi >= sl_price
                tp_hit = bar_lo <= tp_price

            if sl_hit:                  # conservative: SL wins same-bar ties
                close_price, close_reason = sl_price, 'SL'
            elif tp_hit:
                close_price, close_reason = tp_price, 'TP'

            if close_price is None and signal in [-1.0, 1.0]:
                close_price  = bar_cl
                close_reason = 'Signal'

            if close_price is not None:
                pips    = ((close_price - entry_price) if direction == 'LONG'
                           else (entry_price - close_price)) / PIP_SIZE
                pnl     = round(pips * PIP_VALUE_USD, 2)
                balance = round(balance + pnl, 2)

                trades.append({
                    'Trade #'    : len(trades) + 1,
                    'Direction'  : direction,
                    'Entry Date' : entry_time.strftime('%Y-%m-%d %H:%M'),
                    'Exit Date'  : timestamp.strftime('%Y-%m-%d %H:%M'),
                    'Entry Price': round(entry_price, 5),
                    'Exit Price' : round(close_price, 5),
                    'Exit Reason': close_reason,
                    'Pips'       : round(pips, 1),
                    'P&L (USD)'  : pnl,
                    'Balance'    : balance,
                    'Result'     : 'WIN' if pnl > 0 else 'LOSS',
                    'SL Pips'    : open_trade['sl_pips'],
                    'TP Pips'    : open_trade['tp_pips'],
                })
                equity.append(balance)
                open_trade = None

        if open_trade is None and signal in [1.0, -1.0]:
            direction = 'LONG' if signal == 1.0 else 'SHORT'
            sl_dist, tp_dist = sl_tp_fn(row)
            open_trade = {
                'direction'  : direction,
                'entry_price': bar_cl,
                'entry_time' : timestamp,
                'sl'         : bar_cl - sl_dist if direction == 'LONG' else bar_cl + sl_dist,
                'tp'         : bar_cl + tp_dist if direction == 'LONG' else bar_cl - tp_dist,
                'sl_pips'    : round(sl_dist / PIP_SIZE, 1),
                'tp_pips'    : round(tp_dist / PIP_SIZE, 1),
            }

    return trades, equity


def fixed_sl_tp(row) -> tuple:
    return FIXED_SL_DIST, FIXED_TP_DIST


def make_atr_sl_tp_fn(sl_mult: float, tp_mult: float):
    def _fn(row) -> tuple:
        atr = row['ATR']
        return atr * sl_mult, atr * tp_mult
    return _fn


# ── 6. COMPUTE STATS ──────────────────────────────────────────────────────────

def compute_stats(trades: list, equity: list, data: pd.DataFrame) -> dict:
    df      = pd.DataFrame(trades)
    winners = df[df['P&L (USD)'] > 0]
    losers  = df[df['P&L (USD)'] <= 0]
    tp_hits = df[df['Exit Reason'] == 'TP']
    sl_hits = df[df['Exit Reason'] == 'SL']

    peak   = INITIAL_BALANCE
    max_dd = 0.0
    for e in equity:
        peak   = max(peak, e)
        max_dd = max(max_dd, (peak - e) / peak * 100)

    months = (data.index[-1] - data.index[0]).days / 30

    return {
        'months'     : round(months, 1),
        'trades'     : len(df),
        'winners'    : len(winners),
        'losers'     : len(losers),
        'tp_hits'    : len(tp_hits),
        'sl_hits'    : len(sl_hits),
        'sig_exits'  : len(df[df['Exit Reason'] == 'Signal']),
        'win_rate'   : round(len(winners) / len(df) * 100, 1) if len(df) else 0,
        'pnl'        : round(df['P&L (USD)'].sum(), 2),
        'balance'    : equity[-1],
        'max_dd'     : round(max_dd, 1),
        'avg_win'    : round(winners['P&L (USD)'].mean(), 2) if not winners.empty else 0,
        'avg_loss'   : round(losers['P&L (USD)'].mean(),  2) if not losers.empty  else 0,
        'best'       : round(df['P&L (USD)'].max(), 2),
        'worst'      : round(df['P&L (USD)'].min(), 2),
        'per_month'  : round(df['P&L (USD)'].sum() / months, 2) if months else 0,
        'avg_sl_pips': round(df['SL Pips'].mean(), 1),
        'avg_tp_pips': round(df['TP Pips'].mean(), 1),
        'min_sl_pips': round(df['SL Pips'].min(), 1),
        'max_sl_pips': round(df['SL Pips'].max(), 1),
        'trades_pm'  : round(len(df) / months, 1),
        'df'         : df,
    }


def widen_narrow_breakdown(atr_df: pd.DataFrame) -> dict:
    n        = len(atr_df)
    wider    = int((atr_df['SL Pips'] > FIXED_SL_PIPS).sum())
    narrower = int((atr_df['SL Pips'] < FIXED_SL_PIPS).sum())
    equal    = int((atr_df['SL Pips'] == FIXED_SL_PIPS).sum())
    return {
        'wider'        : wider,
        'narrower'     : narrower,
        'equal'        : equal,
        'pct_wider'    : round(wider / n * 100, 1)    if n else 0,
        'pct_narrower' : round(narrower / n * 100, 1) if n else 0,
        'pct_equal'    : round(equal / n * 100, 1)    if n else 0,
    }


# ── 7. PRINT COMPARISON (N-way: fixed + any number of ATR variants) ─────────

def print_comparison(runs: list):
    """runs: list of (label, stats_dict) tuples, first entry must be 'Fixed'."""
    labels = [label for label, _ in runs]
    col_w  = max(11, max(len(l) for l in labels) + 1)
    w      = 24 + (col_w + 2) * len(runs)

    def row(name, key, fmt, suffix=''):
        cells = '  '.join(fmt.format(s[key]) + suffix for _, s in runs)
        print(f"  {name:<22}  {cells}")

    print()
    print("=" * w)
    print("   FIXED 50p/100p SL-TP  vs  ATR(14)-BASED SL-TP VARIANTS")
    print("   H1 50/200 SMA + RSI-14/60-40  |  EURUSD  |  3 Years  |  0.1 Lots")
    print("=" * w)
    header = '  '.join(f'{l:>{col_w}}' for l in labels)
    print(f"  {'Metric':<22}  {header}")
    print("-" * w)
    row('Total Trades',        'trades',      '{:>' + str(col_w) + '}')
    row('Win Rate',            'win_rate',    '{:>' + str(col_w-1) + '.1f}', '%')
    row('Total P&L',           'pnl',         '${:>+' + str(col_w-1) + ',.2f}')
    row('Avg P&L / month',     'per_month',   '${:>+' + str(col_w-1) + ',.2f}')
    row('Final Balance',       'balance',     '${:>' + str(col_w-1) + ',.2f}')
    row('Max Drawdown',        'max_dd',      '{:>' + str(col_w-1) + '.1f}', '%')
    row('Best Trade',          'best',        '${:>+' + str(col_w-1) + ',.2f}')
    row('Worst Trade',         'worst',       '${:>+' + str(col_w-1) + ',.2f}')
    print("-" * w)
    row('TP hits',             'tp_hits',     '{:>' + str(col_w) + '}')
    row('SL hits',             'sl_hits',     '{:>' + str(col_w) + '}')
    row('Signal exits',        'sig_exits',   '{:>' + str(col_w) + '}')
    row('Avg win',             'avg_win',     '${:>+' + str(col_w-1) + ',.2f}')
    row('Avg loss',            'avg_loss',    '${:>+' + str(col_w-1) + ',.2f}')
    print("-" * w)
    row('Avg SL size (pips)',  'avg_sl_pips', '{:>' + str(col_w) + '.1f}')
    row('Avg TP size (pips)',  'avg_tp_pips', '{:>' + str(col_w) + '.1f}')
    print("=" * w)
    print()


def print_widen_narrow(label: str, wn: dict, n_trades: int):
    print(f"  {label}  vs fixed 50p baseline  ({n_trades} trades):")
    print(f"    Wider   (SL > 50p) : {wn['wider']:>3}  ({wn['pct_wider']:.1f}%)")
    print(f"    Narrower(SL < 50p) : {wn['narrower']:>3}  ({wn['pct_narrower']:.1f}%)")
    print(f"    Equal   (SL = 50p) : {wn['equal']:>3}  ({wn['pct_equal']:.1f}%)")
    print()


# ── 8. PLOT EQUITY CURVES ──────────────────────────────────────────────────────

def plot_results(runs: list, data):
    """runs: list of (label, trades, equity) tuples."""
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e']

    fig, ax = plt.subplots(figsize=(16, 7))
    for i, (label, trades, equity) in enumerate(runs):
        exit_dates = pd.to_datetime(pd.DataFrame(trades)['Exit Date'])
        idx        = [exit_dates.iloc[0]] + list(exit_dates)
        ax.plot(idx, equity, color=colors[i % len(colors)], linewidth=2,
                label=f'{label}  (final ${equity[-1]:,.2f})')

    ax.axhline(INITIAL_BALANCE, color='grey', linewidth=1, linestyle='--',
               label=f'Starting balance  ${INITIAL_BALANCE:,.0f}')

    ax.set_title('Fixed SL/TP vs ATR(14) SL/TP Variants -- H1 50/200 SMA  |  EURUSD  |  3 Years',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Account Balance (USD)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

    fig.tight_layout()
    output_dir  = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'h1_atr_vs_fixed_equity_curve.png')
    plt.savefig(output_path, dpi=120)
    print(f"Chart saved : {output_path}")


# ── 9. SAVE TRADE LOGS ─────────────────────────────────────────────────────────

def save_trade_log(trades: list, filename: str):
    df          = pd.DataFrame(trades)
    output_dir  = os.path.join(os.path.dirname(__file__), '..', 'data')
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"Trade log   : {output_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

connect_mt5()
raw = fetch_data()
mt5.shutdown()

data = apply_strategy(raw)

fixed_trades, fixed_equity = run_backtest(data, fixed_sl_tp)
fixed_stats = compute_stats(fixed_trades, fixed_equity, data)

runs       = [('Fixed 50p/100p', fixed_stats)]
plot_runs  = [('Fixed 50p/100p', fixed_trades, fixed_equity)]

for variant in ATR_VARIANTS:
    fn = make_atr_sl_tp_fn(variant['sl_mult'], variant['tp_mult'])
    trades, equity = run_backtest(data, fn)
    stats = compute_stats(trades, equity, data)

    label = variant['label']
    runs.append((label, stats))
    plot_runs.append((label, trades, equity))

    safe_name = label.lower().replace(' ', '_').replace('.', '').replace('/', '_')
    save_trade_log(trades, f'trade_log_h1_{safe_name}.csv')

print_comparison(runs)

for label, stats in runs[1:]:
    wn = widen_narrow_breakdown(stats['df'])
    print_widen_narrow(label, wn, stats['trades'])

save_trade_log(fixed_trades, 'trade_log_h1_fixed_sl_tp.csv')
plot_results(plot_runs, raw)
print("\nDone!")
