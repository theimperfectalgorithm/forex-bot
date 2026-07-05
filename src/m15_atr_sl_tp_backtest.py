"""
Forex Bot - M15 ATR(14) 3.0x/6.0x SL/TP Backtest

Tests whether ATR-based stop sizing (which fixed the H1 50/200 SMA strategy --
see h1_atr_sl_tp_backtest.py: ATR 3.0x/6.0x turned +$577.90 fixed-stop into
+$856.73) also rescues the M15 version of the same strategy, which lost
money even before adding any stop at all (Episode 7/8 baseline: -$43.20).

Same crossover entry logic and RSI-14/60-40 filter as every prior episode.
Only the timeframe (M15) and SL/TP sizing (ATR-based) change.

  M15 baseline (Episode 7/8, no SL/TP) : -$43.20  (43.4% WR, 12.4% DD, 235 trades)
  H1  ATR 3.0x/6.0x (just tested)      : +$856.73 (44.6% WR, 1.4% DD,  56 trades)
  M15 ATR 3.0x/6.0x (this test)        : ?

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

ATR_PERIOD  = 14
SL_ATR_MULT = 3.0
TP_ATR_MULT = 6.0

FIXED_SL_PIPS = 50   # reference point only, for wider/narrower comparison

# Hardcoded reference results from prior episodes (same strategy, different
# timeframe / stop sizing) -- used for the final 3-way comparison table.
M15_BASELINE_NO_SLTP = {
    'label'   : 'M15 baseline (no SL/TP, Ep.7/8)',
    'trades'  : 235,
    'win_rate': 43.4,
    'pnl'     : -43.20,
    'max_dd'  : 12.4,
    'avg_sl_pips': None,
    'avg_tp_pips': None,
}

H1_ATR_3X_6X = {
    'label'   : 'H1 ATR 3.0x/6.0x',
    'trades'  : 56,
    'win_rate': 44.6,
    'pnl'     : 856.73,
    'max_dd'  : 1.4,
    'avg_sl_pips': 38.6,
    'avg_tp_pips': 77.2,
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


# ── 2. FETCH M15 OHLC DATA ──────────────────────────────────────────────────────

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


# ── 4. STRATEGY (identical entry/filter logic to every prior episode) ───────

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


# ── 5. BAR-BY-BAR BACKTEST WITH ATR SL/TP ────────────────────────────────────

def run_backtest(data: pd.DataFrame):
    """Conservative tie-break: if both SL and TP are hit in the same bar, SL wins."""
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

            if sl_hit:
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
            atr      = row['ATR']
            sl_dist  = atr * SL_ATR_MULT
            tp_dist  = atr * TP_ATR_MULT
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


# ── 7. PRINT 3-WAY COMPARISON ─────────────────────────────────────────────────

def print_comparison(m15_atr: dict):
    base = M15_BASELINE_NO_SLTP
    h1   = H1_ATR_3X_6X
    w    = 78

    print()
    print("=" * w)
    print("   M15 ATR(14) 3.0x/6.0x  vs  M15 BASELINE (no SL/TP)  vs  H1 ATR(14) 3.0x/6.0x")
    print("   50/200 SMA crossover + RSI-14/60-40  |  EURUSD  |  3 Years  |  0.1 Lots")
    print("=" * w)
    print(f"  {'Metric':<22}  {'M15 baseline':>14}  {'M15 ATR 3x/6x':>14}  {'H1 ATR 3x/6x':>14}")
    print("-" * w)
    print(f"  {'Total Trades':<22}  {base['trades']:>14}  {m15_atr['trades']:>14}  {h1['trades']:>14}")
    print(f"  {'Win Rate':<22}  {base['win_rate']:>13.1f}%  {m15_atr['win_rate']:>13.1f}%  {h1['win_rate']:>13.1f}%")
    print(f"  {'Total P&L':<22}  ${base['pnl']:>+13,.2f}  ${m15_atr['pnl']:>+13,.2f}  ${h1['pnl']:>+13,.2f}")
    print(f"  {'Max Drawdown':<22}  {base['max_dd']:>13.1f}%  {m15_atr['max_dd']:>13.1f}%  {h1['max_dd']:>13.1f}%")
    print("-" * w)
    print(f"  {'Avg SL size (pips)':<22}  {'n/a':>14}  {m15_atr['avg_sl_pips']:>14.1f}  {h1['avg_sl_pips']:>14.1f}")
    print(f"  {'Avg TP size (pips)':<22}  {'n/a':>14}  {m15_atr['avg_tp_pips']:>14.1f}  {h1['avg_tp_pips']:>14.1f}")
    print(f"  {'SL range (min-max)':<22}  {'n/a':>14}  "
          f"{m15_atr['min_sl_pips']:.1f}-{m15_atr['max_sl_pips']:.1f}p")
    print("-" * w)
    print(f"  {'TP hits':<22}  {'n/a':>14}  {m15_atr['tp_hits']:>14}")
    print(f"  {'SL hits':<22}  {'n/a':>14}  {m15_atr['sl_hits']:>14}")
    print(f"  {'Signal exits':<22}  {'n/a':>14}  {m15_atr['sig_exits']:>14}")
    print(f"  {'Avg win':<22}  {'n/a':>14}  ${m15_atr['avg_win']:>+13,.2f}")
    print(f"  {'Avg loss':<22}  {'n/a':>14}  ${m15_atr['avg_loss']:>+13,.2f}")
    print("=" * w)
    print()


def print_widen_narrow(wn: dict, n_trades: int):
    print(f"  M15 ATR 3.0x/6.0x vs fixed 50p reference  ({n_trades} trades):")
    print(f"    Wider   (SL > 50p) : {wn['wider']:>3}  ({wn['pct_wider']:.1f}%)")
    print(f"    Narrower(SL < 50p) : {wn['narrower']:>3}  ({wn['pct_narrower']:.1f}%)")
    print(f"    Equal   (SL = 50p) : {wn['equal']:>3}  ({wn['pct_equal']:.1f}%)")
    print()


# ── 8. PLOT EQUITY CURVE ──────────────────────────────────────────────────────

def plot_results(trades: list, equity: list, data: pd.DataFrame):
    df         = pd.DataFrame(trades)
    exit_times = pd.to_datetime(df['Exit Date'])
    full_index = [exit_times.iloc[0]] + list(exit_times)

    fig, ax = plt.subplots(figsize=(16, 7))
    curve_color = '#2ca02c' if equity[-1] >= INITIAL_BALANCE else '#d62728'
    ax.plot(full_index, equity, color=curve_color, linewidth=2,
            label=f'M15 ATR(14) 3.0x/6.0x  (final ${equity[-1]:,.2f})')
    ax.axhline(INITIAL_BALANCE, color='grey', linewidth=1, linestyle='--',
               label=f'Starting balance  ${INITIAL_BALANCE:,.0f}')

    ax.set_title('M15 50/200 SMA + ATR(14) 3.0x/6.0x SL-TP  |  EURUSD  |  3 Years',
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
    output_path = os.path.join(output_dir, 'm15_atr_sl_tp_equity_curve.png')
    plt.savefig(output_path, dpi=120)
    print(f"Chart saved : {output_path}")


# ── 9. SAVE TRADE LOG ─────────────────────────────────────────────────────────

def save_trade_log(trades: list):
    df          = pd.DataFrame(trades)
    output_dir  = os.path.join(os.path.dirname(__file__), '..', 'data')
    output_path = os.path.join(output_dir, 'trade_log_m15_atr_sl_tp.csv')
    df.to_csv(output_path, index=False)
    print(f"Trade log   : {output_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

connect_mt5()
raw = fetch_data()
mt5.shutdown()

data           = apply_strategy(raw)
trades, equity = run_backtest(data)
stats          = compute_stats(trades, equity, data)
wn             = widen_narrow_breakdown(stats['df'])

print_comparison(stats)
print_widen_narrow(wn, stats['trades'])
save_trade_log(trades)
plot_results(trades, equity, raw)
print("\nDone!")
