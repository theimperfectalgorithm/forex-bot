"""
Forex Bot - Walk-Forward Style Validation: H1 ATR(14) 3.0x/6.0x SL/TP

Splits the existing 3-year H1 50/200 SMA + ATR(14) 3.0x/6.0x backtest
(Jul 2023 - Jun 2026, total +$856.73) into 6 calendar half-year sub-periods
and reports trades / win rate / P&L separately for each, to check whether
the total profit is spread fairly evenly or concentrated in one or two
windows (a sign of overfitting / regime dependence rather than a real edge).

Same crossover entry logic and RSI-14/60-40 filter as every prior episode.
Same single continuous backtest run as h1_atr_sl_tp_backtest.py -- this
script only changes how the *results* are sliced and reported.

Settings: EURUSD H1, ~3 years, $10,000 balance, 0.1 lots ($1/pip).

Requirements:
  MetaTrader 5 must be OPEN and LOGGED IN before running this script.
"""

import sys
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5
import pandas as pd

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

# Calendar half-year sub-periods covering Jul 2023 -- Jun 2026
SUB_PERIODS = [
    ('2023-07-01', '2023-12-31', 'H2 2023'),
    ('2024-01-01', '2024-06-30', 'H1 2024'),
    ('2024-07-01', '2024-12-31', 'H2 2024'),
    ('2025-01-01', '2025-06-30', 'H1 2025'),
    ('2025-07-01', '2025-12-31', 'H2 2025'),
    ('2026-01-01', '2026-06-30', 'H1 2026'),
]


# ── 1. CONNECT TO MT5 ─────────────────────────────────────────────────────────

def connect_mt5():
    print("Connecting to MetaTrader 5...")
    if not mt5.initialize():
        print(f"ERROR: Could not connect — {mt5.last_error()}")
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
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# ── 4. STRATEGY ───────────────────────────────────────────────────────────────

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
    trades     = []
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
                    'Entry Time' : entry_time,
                    'Exit Time'  : timestamp,
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

    return trades


# ── 6. SPLIT INTO SUB-PERIODS AND REPORT ─────────────────────────────────────

def split_and_report(trades: list):
    df = pd.DataFrame(trades)
    df['Entry Time'] = pd.to_datetime(df['Entry Time'], utc=True)

    rows = []
    total_pnl_check = 0.0

    for start_str, end_str, label in SUB_PERIODS:
        start = pd.Timestamp(start_str, tz='UTC')
        end   = pd.Timestamp(end_str, tz='UTC') + pd.Timedelta(days=1)  # inclusive end day
        chunk = df[(df['Entry Time'] >= start) & (df['Entry Time'] < end)]

        n       = len(chunk)
        winners = chunk[chunk['P&L (USD)'] > 0]
        win_rate = round(len(winners) / n * 100, 1) if n else 0.0
        pnl      = round(chunk['P&L (USD)'].sum(), 2)
        total_pnl_check += pnl

        rows.append({
            'period'  : label,
            'trades'  : n,
            'win_rate': win_rate,
            'pnl'     : pnl,
        })

    return rows, round(total_pnl_check, 2)


def print_report(rows: list, total_pnl: float):
    w = 56
    print()
    print("=" * w)
    print("   WALK-FORWARD SPLIT -- H1 ATR(14) 3.0x/6.0x SL/TP")
    print("   50/200 SMA + RSI-14/60-40  |  EURUSD H1  |  6 x 6-month windows")
    print("=" * w)
    print(f"  {'Period':<12}  {'Trades':>8}  {'Win Rate':>9}  {'P&L':>12}")
    print("-" * w)

    pnls = [r['pnl'] for r in rows]
    best_idx  = pnls.index(max(pnls))
    worst_idx = pnls.index(min(pnls))

    for i, r in enumerate(rows):
        flag = ''
        if i == best_idx and r['pnl'] > 0:
            flag = '  <- best'
        elif i == worst_idx and r['pnl'] < 0:
            flag = '  <- worst'
        print(f"  {r['period']:<12}  {r['trades']:>8}  {r['win_rate']:>8.1f}%  "
              f"${r['pnl']:>+10,.2f}{flag}")

    print("-" * w)
    print(f"  {'TOTAL':<12}  {sum(r['trades'] for r in rows):>8}  {'':>9}  "
          f"${total_pnl:>+10,.2f}")
    print("=" * w)
    print()

    # Concentration check: what % of total profit comes from the single best period?
    positive_pnls = [p for p in pnls if p > 0]
    n_profitable  = len(positive_pnls)
    n_losing      = len([p for p in pnls if p < 0])

    print(f"  Profitable periods : {n_profitable} / {len(rows)}")
    print(f"  Losing periods     : {n_losing} / {len(rows)}")

    if total_pnl > 0 and positive_pnls:
        best_pnl = max(pnls)
        concentration = round(best_pnl / total_pnl * 100, 1)
        print(f"  Best single period contributes: ${best_pnl:+,.2f}  "
              f"({concentration:.1f}% of total profit)")

        sorted_pnls = sorted(positive_pnls, reverse=True)
        if len(sorted_pnls) >= 2:
            top2 = sorted_pnls[0] + sorted_pnls[1]
            top2_pct = round(top2 / total_pnl * 100, 1)
            print(f"  Top 2 periods contribute      : ${top2:+,.2f}  "
                  f"({top2_pct:.1f}% of total profit)")

    print()


# ── MAIN ──────────────────────────────────────────────────────────────────────

connect_mt5()
raw = fetch_data()
mt5.shutdown()

data   = apply_strategy(raw)
trades = run_backtest(data)
rows, total_pnl = split_and_report(trades)
print_report(rows, total_pnl)
print("Done!")
