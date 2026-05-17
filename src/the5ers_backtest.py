"""
Forex Bot -- The5ers $100K Session Breakout Backtest (H4 Trend Filter)

Strategy:
  0. H4 trend filter: 50/200 SMA on H4 bars at London open time
       Bullish (Close > SMA50 > SMA200)  -> BUY breakouts only
       Bearish (Close < SMA50 < SMA200)  -> SELL breakouts only
       Neutral                           -> skip day (no trade)
  1. Asian session (00:00-07:00 UTC): record High and Low of the session
  2. London open (08:00 UTC): take breakout ONLY in trend direction
  3. NY open (13:00 UTC): same check if London produced no signal that day
  4. CANCEL: if price already moved > 20 pips past the breakout level (no chasing)
  5. Stop loss  = 50% of Asian session range
  6. Take profit = 100% of Asian session range (2:1 risk-reward)
  7. EOD close at 22:00 UTC if neither SL nor TP was hit

The5ers Phase 1 Challenge Rules (Growth Plan $100K):
  Starting balance   : $100,000
  Profit target      : 10% -> reach $110,000
  Max daily loss     : 5% of current balance
  Max total loss     : 10% of STARTING balance -> hard floor $90,000
  Min profitable days: 3 per month
  Time limit         : Unlimited

Dynamic position sizing:
  Risk per trade = 1% of balance  (= 20% of the 5% daily limit)
  SL pips        = 50% of Asian session range (varies every day)
  Lot size       = risk_usd / (SL_pips x pip_value_per_lot)

Pairs: GBPUSD, EURUSD, AUDUSD, NZDUSD, EURJPY, GBPJPY

Requirements:
  MetaTrader 5 must be OPEN and LOGGED IN before running this script.
"""

import os
import sys
from datetime import datetime, timedelta, timezone, date as date_type

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from collections import defaultdict

# -- SETTINGS -------------------------------------------------------------------

STARTING_BALANCE     = 100_000.00
TARGET_BALANCE       = 110_000.00   # 10% profit target
HARD_FLOOR           = 90_000.00    # 10% loss -- hard stop
MAX_DAILY_LOSS_PCT   = 0.05         # 5% of balance that day
RISK_PER_TRADE_PCT   = 0.01         # 1% per trade (20% of daily limit)
MAX_CONSEC_LOSSES    = 2            # per pair -> skip next trading day
MIN_ASIAN_RANGE_PIPS = 10           # skip days with tiny Asian range
MAX_OVERSHOOT_PIPS   = 20           # cancel if already 20+ pips past breakout
MONTHS               = 36

ASIAN_END_HOUR   = 7    # 00:00-06:45 UTC bars included in Asian range
LONDON_HOUR      = 8    # London breakout check starts at 08:00 UTC
NY_HOUR          = 13   # NY breakout check starts at 13:00 UTC
EOD_HOUR         = 22   # Close any open trade at 22:00 UTC

MIN_LOT = 0.01
MAX_LOT = 2.0

PAIRS = {
    'GBPUSD': {'pip_size': 0.0001, 'pip_value': 10.00},
    'EURUSD': {'pip_size': 0.0001, 'pip_value': 10.00},
    'AUDUSD': {'pip_size': 0.0001, 'pip_value': 10.00},
    'NZDUSD': {'pip_size': 0.0001, 'pip_value': 10.00},
    'EURJPY': {'pip_size': 0.01,   'pip_value':  6.67},
    'GBPJPY': {'pip_size': 0.01,   'pip_value':  6.67},
}

# -- 1. MT5 CONNECTION & DATA FETCH --------------------------------------------

def connect_mt5():
    print("Connecting to MetaTrader 5...")
    if not mt5.initialize():
        print(f"ERROR: {mt5.last_error()}")
        sys.exit(1)
    info    = mt5.terminal_info()
    account = mt5.account_info()
    print(f"Connected : {info.name}")
    print(f"Account   : {account.login}  ({account.server})\n")


def _fetch(symbol, date_from, date_to):
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, date_from, date_to)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High',
                       'low': 'Low', 'close': 'Close'}, inplace=True)
    return df[['Open', 'High', 'Low', 'Close']]


def fetch_all_data() -> dict:
    date_to   = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=MONTHS * 30)

    print(f"Fetching M15 data  {date_from.date()} to {date_to.date()}")
    print("-" * 60)

    all_data = {}
    for sym in PAIRS:
        df = _fetch(sym, date_from, date_to)
        if df is None:
            print(f"  {sym}: SKIPPED -- no data")
            continue
        all_data[sym] = df
        print(f"  {sym}: {len(df):,} bars  "
              f"({df.index[0].date()} to {df.index[-1].date()})")

    print()
    return all_data


# -- 2. POSITION SIZING --------------------------------------------------------

def calc_lots(balance: float, sym: str, sl_pips: float) -> float:
    risk_usd = balance * RISK_PER_TRADE_PCT
    pip_val  = PAIRS[sym]['pip_value']
    lots     = risk_usd / (sl_pips * pip_val)
    return round(max(MIN_LOT, min(lots, MAX_LOT)), 2)


# -- 2b. H4 TREND FILTER -------------------------------------------------------

def prepare_h4(m15_df: pd.DataFrame) -> pd.DataFrame:
    """Resample M15 -> H4 and compute 50/200 SMA trend."""
    h4 = m15_df.resample('4h').agg(
        Open=('Open', 'first'), High=('High', 'max'),
        Low=('Low', 'min'),   Close=('Close', 'last')
    ).dropna()
    h4['SMA50']  = h4['Close'].rolling(50).mean()
    h4['SMA200'] = h4['Close'].rolling(200).mean()
    h4['Trend']  = 0
    bull = (h4['Close'] > h4['SMA50']) & (h4['SMA50'] > h4['SMA200'])
    bear = (h4['Close'] < h4['SMA50']) & (h4['SMA50'] < h4['SMA200'])
    h4.loc[bull, 'Trend'] =  1
    h4.loc[bear, 'Trend'] = -1
    return h4


def get_h4_trend(h4_df: pd.DataFrame, at_ts: pd.Timestamp) -> int:
    """Return the trend of the last H4 bar that CLOSED before at_ts."""
    prior = h4_df[h4_df.index < at_ts]
    if prior.empty:
        return 0
    return int(prior['Trend'].iloc[-1])


# -- 3. BREAKOUT DETECTION -----------------------------------------------------

def find_breakout(session_bars: pd.DataFrame,
                  asian_high: float, asian_low: float,
                  sym: str, allowed_direction: int = 0):
    """
    Scan session bars for the FIRST breakout of the Asian range.
    allowed_direction: +1 = BUY only, -1 = SELL only, 0 = both directions.
    Returns a signal dict or None.

    Cancels if:
      - Both directions breached in the same bar (ambiguous)
      - Price already moved > 20 pips past the breakout level
    """
    pip_size = PAIRS[sym]['pip_size']

    for ts, bar in session_bars.iterrows():
        buy_break  = bar['High'] > asian_high
        sell_break = bar['Low']  < asian_low

        if buy_break and sell_break:
            return None  # Ambiguous -- skip

        if buy_break and allowed_direction >= 0:
            overshoot = (bar['High'] - asian_high) / pip_size
            if overshoot > MAX_OVERSHOOT_PIPS:
                return None
            return {
                'direction'   : 'LONG',
                'entry_price' : asian_high,
                'trigger_ts'  : ts,
            }

        if sell_break and allowed_direction <= 0:
            overshoot = (asian_low - bar['Low']) / pip_size
            if overshoot > MAX_OVERSHOOT_PIPS:
                return None
            return {
                'direction'   : 'SHORT',
                'entry_price' : asian_low,
                'trigger_ts'  : ts,
            }

    return None  # No breakout this session


# -- 4. TRADE EXECUTION --------------------------------------------------------

def execute_trade(signal: dict, bars_from_trigger: pd.DataFrame,
                  sl_pips: float, tp_pips: float, sym: str):
    """
    Run bar-by-bar from the trigger bar until SL, TP, or EOD close.
    Entry is assumed at signal['entry_price'] (pending order fill).
    Returns (exit_price, exit_reason, exit_ts).
    """
    pip_size    = PAIRS[sym]['pip_size']
    direction   = signal['direction']
    entry_price = signal['entry_price']

    if direction == 'LONG':
        sl_price = entry_price - sl_pips * pip_size
        tp_price = entry_price + tp_pips * pip_size
    else:
        sl_price = entry_price + sl_pips * pip_size
        tp_price = entry_price - tp_pips * pip_size

    for ts, bar in bars_from_trigger.iterrows():
        hi = bar['High']
        lo = bar['Low']

        if direction == 'LONG':
            sl_hit = lo <= sl_price
            tp_hit = hi >= tp_price
        else:
            sl_hit = hi >= sl_price
            tp_hit = lo <= tp_price

        if sl_hit and tp_hit:
            return sl_price, 'SL', ts   # Conservative: SL wins tie
        if sl_hit:
            return sl_price, 'SL', ts
        if tp_hit:
            return tp_price, 'TP', ts

    # EOD: close at the last bar's close price
    last = bars_from_trigger.iloc[-1]
    return last['Close'], 'EOD', bars_from_trigger.index[-1]


# -- 5. BACKTEST ENGINE --------------------------------------------------------

def run_backtest(all_data: dict):
    # Build sorted list of unique trading days (Mon-Fri) across all pairs
    all_days = sorted({
        ts.date()
        for sym in all_data
        for ts in all_data[sym].index
        if ts.weekday() < 5
    })

    # Pre-compute H4 trend for each pair
    h4_data = {sym: prepare_h4(df) for sym, df in all_data.items()}
    print(f"H4 trend filter: ON  (50/200 SMA -- BUY/SELL only with trend, skip neutral days)")
    print(f"Processing {len(all_days)} trading days across {len(all_data)} pairs ...\n")

    balance     = STARTING_BALANCE
    all_trades  = []
    equity_log  = []      # [(date, balance, daily_pnl, day_start_balance)]
    hard_stop   = False
    target_hit  = False

    # Per-pair pause tracking (consecutive losses)
    pair_consec_losses = {sym: 0 for sym in all_data}
    pair_pause_until   = {sym: None for sym in all_data}

    # Monthly tracking
    monthly_start = {}   # {period: balance_at_start}

    for day in all_days:
        if hard_stop or target_hit:
            break

        # -- Hard floor check at start of day ---------------------------------
        if balance <= HARD_FLOOR:
            print(f"HARD STOP on {day}  --  balance ${balance:,.2f} at or below $90,000")
            hard_stop = True
            break

        # -- Target check at start of day -------------------------------------
        if balance >= TARGET_BALANCE:
            print(f"TARGET REACHED on {day}  --  balance ${balance:,.2f}")
            target_hit = True
            break

        # -- Monthly tracking -------------------------------------------------
        period = pd.Period(day, 'M')
        if period not in monthly_start:
            monthly_start[period] = balance

        day_start_balance = balance
        daily_pnl         = 0.0
        daily_loss_limit  = balance * MAX_DAILY_LOSS_PCT

        # -- Process each pair ------------------------------------------------
        for sym in all_data:
            # Daily loss limit: don't open new trades if already at limit
            if daily_pnl <= -daily_loss_limit:
                break

            # Per-pair pause (2 consecutive losses)
            if (pair_pause_until[sym] is not None
                    and day <= pair_pause_until[sym]):
                continue

            m15 = all_data[sym]
            day_ts  = pd.Timestamp(day, tz='UTC')
            next_ts = day_ts + pd.Timedelta(days=1)

            day_bars = m15[(m15.index >= day_ts) & (m15.index < next_ts)]
            if len(day_bars) < 20:
                continue   # Not enough data for this pair on this day

            # -- Asian range (00:00-06:45 UTC) --------------------------------
            asian = day_bars[day_bars.index.hour < ASIAN_END_HOUR]
            if len(asian) < 4:   # Need at least 1 hour of data
                continue

            asian_high       = asian['High'].max()
            asian_low        = asian['Low'].min()
            range_pips       = (asian_high - asian_low) / PAIRS[sym]['pip_size']

            if range_pips < MIN_ASIAN_RANGE_PIPS:
                continue   # Range too tight, skip day

            sl_pips = range_pips * 0.50   # 50% of range
            tp_pips = range_pips * 1.00   # 100% of range
            lots    = calc_lots(balance, sym, sl_pips)

            # -- H4 trend direction at London open ----------------------------
            london_open_ts = pd.Timestamp(day, tz='UTC').replace(
                hour=LONDON_HOUR, minute=0, second=0, microsecond=0
            )
            trend = get_h4_trend(h4_data[sym], london_open_ts)
            if trend == 0:
                continue   # No clear H4 trend -- skip this pair today

            # -- London session breakout (08:00-12:45 UTC) --------------------
            london_bars = day_bars[
                (day_bars.index.hour >= LONDON_HOUR) &
                (day_bars.index.hour <  NY_HOUR)
            ]
            signal = find_breakout(london_bars, asian_high, asian_low, sym,
                                   allowed_direction=trend)

            # -- NY session breakout (13:00-21:45 UTC) if London produced none -
            if signal is None:
                ny_bars = day_bars[
                    (day_bars.index.hour >= NY_HOUR) &
                    (day_bars.index.hour <  EOD_HOUR)
                ]
                signal = find_breakout(ny_bars, asian_high, asian_low, sym,
                                       allowed_direction=trend)

            if signal is None:
                continue   # No valid signal today for this pair

            # -- Execute trade bar-by-bar from trigger bar to 22:00 UTC ------
            trigger_ts    = signal['trigger_ts']
            tradeable_bars = day_bars[
                (day_bars.index >= trigger_ts) &
                (day_bars.index.hour < EOD_HOUR)
            ]
            if tradeable_bars.empty:
                continue

            exit_price, exit_reason, exit_ts = execute_trade(
                signal, tradeable_bars, sl_pips, tp_pips, sym
            )

            direction   = signal['direction']
            entry_price = signal['entry_price']
            pip_size    = PAIRS[sym]['pip_size']

            pips = ((exit_price - entry_price) if direction == 'LONG'
                    else (entry_price - exit_price)) / pip_size
            pnl  = round(pips * PAIRS[sym]['pip_value'] * lots, 2)

            balance   = round(balance + pnl, 2)
            daily_pnl += pnl

            # SL/TP price levels for the trade log
            if direction == 'LONG':
                sl_level = entry_price - sl_pips * pip_size
                tp_level = entry_price + tp_pips * pip_size
            else:
                sl_level = entry_price + sl_pips * pip_size
                tp_level = entry_price - tp_pips * pip_size

            all_trades.append({
                'Date'          : str(day),
                'Symbol'        : sym,
                'Direction'     : direction,
                'Session'       : ('London' if trigger_ts.hour < NY_HOUR else 'New York'),
                'Entry Time'    : trigger_ts.strftime('%H:%M'),
                'Exit Time'     : exit_ts.strftime('%H:%M'),
                'Entry Price'   : round(entry_price, 5),
                'Exit Price'    : round(exit_price, 5),
                'SL'            : round(sl_level, 5),
                'TP'            : round(tp_level, 5),
                'Asian High'    : round(asian_high, 5),
                'Asian Low'     : round(asian_low, 5),
                'Range (pips)'  : round(range_pips, 1),
                'SL (pips)'     : round(sl_pips, 1),
                'TP (pips)'     : round(tp_pips, 1),
                'Lots'          : lots,
                'Pips'          : round(pips, 1),
                'P&L (USD)'     : pnl,
                'Balance'       : balance,
                'Exit Reason'   : exit_reason,
                'Result'        : 'WIN' if pnl > 0 else ('LOSS' if pnl < 0 else 'BE'),
            })

            # Consecutive loss tracking
            if pnl <= 0:
                pair_consec_losses[sym] += 1
                if pair_consec_losses[sym] >= MAX_CONSEC_LOSSES:
                    pair_pause_until[sym]   = day + timedelta(days=1)
                    pair_consec_losses[sym] = 0
            else:
                pair_consec_losses[sym] = 0

            # Recheck daily limit after this trade
            if daily_pnl <= -daily_loss_limit:
                break

        equity_log.append((day, balance, daily_pnl, day_start_balance))

    print(f"  Backtest complete. Final balance: ${balance:,.2f}\n")
    return all_trades, equity_log, monthly_start


# -- 6. STATISTICS -------------------------------------------------------------

def compute_pair_stats(trades: list, sym: str, months: float) -> dict:
    df = pd.DataFrame([t for t in trades if t['Symbol'] == sym])
    if df.empty:
        return None

    winners = df[df['P&L (USD)'] > 0]
    losers  = df[df['P&L (USD)'] < 0]

    df['date_dt'] = pd.to_datetime(df['Date'])
    df['ym']      = df['date_dt'].dt.to_period('M')
    active_days   = df.groupby('ym')['Date'].nunique().mean()

    pnl = df['P&L (USD)'].sum()
    return {
        'symbol'    : sym,
        'trades'    : len(df),
        'winners'   : len(winners),
        'losers'    : len(losers),
        'win_rate'  : round(len(winners) / len(df) * 100, 1),
        'pnl'       : round(pnl, 2),
        'per_month' : round(pnl / months, 2),
        'avg_win'   : round(winners['P&L (USD)'].mean(), 2) if not winners.empty else 0,
        'avg_loss'  : round(losers['P&L (USD)'].mean(),  2) if not losers.empty  else 0,
        'tp_hits'   : int((df['Exit Reason'] == 'TP').sum()),
        'sl_hits'   : int((df['Exit Reason'] == 'SL').sum()),
        'eod_hits'  : int((df['Exit Reason'] == 'EOD').sum()),
        'avg_lots'  : round(df['Lots'].mean(), 2),
        'days_mo'   : round(float(active_days), 1),
        'london_pct': round((df['Session'] == 'London').mean() * 100, 1),
    }


def compute_combined_stats(trades: list, equity_log: list,
                           monthly_start: dict) -> dict:
    df = pd.DataFrame(trades)
    if df.empty:
        return {}

    winners = df[df['P&L (USD)'] > 0]
    losers  = df[df['P&L (USD)'] < 0]

    # Drawdown from equity log
    balances = [v for _, v, _, _ in equity_log]
    peak     = STARTING_BALANCE
    max_dd   = 0.0
    for b in balances:
        peak   = max(peak, b)
        max_dd = max(max_dd, peak - b)

    # Worst single trading day
    worst_day_loss = min((pnl for _, _, pnl, _ in equity_log), default=0)

    # Profitable days per month
    monthly_profit_days = defaultdict(int)
    for day, bal, dpnl, dstart in equity_log:
        if dpnl > 0:
            monthly_profit_days[pd.Period(day, 'M')] += 1

    avg_profitable_days = (np.mean(list(monthly_profit_days.values()))
                           if monthly_profit_days else 0)
    min_profitable_days = (min(monthly_profit_days.values())
                           if monthly_profit_days else 0)

    first = pd.to_datetime(df['Date']).min()
    last  = pd.to_datetime(df['Date']).max()
    months = max((last - first).days / 30.0, 1.0)

    pnl_total = df['P&L (USD)'].sum()
    final_bal = equity_log[-1][1] if equity_log else STARTING_BALANCE

    # Monthly returns
    monthly_pnl = defaultdict(float)
    df['ym'] = pd.to_datetime(df['Date']).dt.to_period('M')
    for ym, grp in df.groupby('ym'):
        monthly_pnl[ym] = grp['P&L (USD)'].sum()

    return {
        'trades'            : len(df),
        'winners'           : len(winners),
        'losers'            : len(losers),
        'win_rate'          : round(len(winners) / len(df) * 100, 1),
        'pnl'               : round(pnl_total, 2),
        'pct_return'        : round(pnl_total / STARTING_BALANCE * 100, 2),
        'final_balance'     : round(final_bal, 2),
        'max_dd'            : round(max_dd, 2),
        'max_dd_pct'        : round(max_dd / STARTING_BALANCE * 100, 2),
        'worst_day'         : round(worst_day_loss, 2),
        'worst_day_pct'     : round(abs(worst_day_loss) / STARTING_BALANCE * 100, 2),
        'avg_win'           : round(winners['P&L (USD)'].mean(), 2) if not winners.empty else 0,
        'avg_loss'          : round(losers['P&L (USD)'].mean(),  2) if not losers.empty  else 0,
        'best_trade'        : round(df['P&L (USD)'].max(), 2),
        'worst_trade'       : round(df['P&L (USD)'].min(), 2),
        'tp_hits'           : int((df['Exit Reason'] == 'TP').sum()),
        'sl_hits'           : int((df['Exit Reason'] == 'SL').sum()),
        'eod_hits'          : int((df['Exit Reason'] == 'EOD').sum()),
        'avg_lots'          : round(df['Lots'].mean(), 2),
        'min_lots'          : round(df['Lots'].min(), 2),
        'max_lots'          : round(df['Lots'].max(), 2),
        'months'            : round(months, 1),
        'per_month'         : round(pnl_total / months, 2),
        'avg_profitable_days': round(avg_profitable_days, 1),
        'min_profitable_days': int(min_profitable_days),
        'monthly_pnl'       : dict(monthly_pnl),
        'monthly_start'     : monthly_start,
    }


# -- 7. PRINT RESULTS ----------------------------------------------------------

def print_per_pair(pair_stats: list):
    w = 108
    print("=" * w)
    print("  PER-PAIR RESULTS  |  Session Breakout + H4 Trend Filter  |  Dynamic Lot Sizing")
    print(f"  Asian range SL=50% / TP=100%  |  1% risk/trade  |  MAX 2 lots  |  ~3 years  |  $100,000 account")
    print("=" * w)
    print(f"  {'Pair':<8} {'Trades':>7} {'Win%':>6} {'P&L ($)':>12} "
          f"{'$/mo':>9} {'AvgWin':>9} {'AvgLoss':>9} "
          f"{'TP':>5} {'SL':>5} {'EOD':>5} {'AvgLot':>8} {'Days/mo':>9} {'London%':>8}")
    print("-" * w)
    for s in pair_stats:
        if s is None:
            continue
        print(f"  {s['symbol']:<8} {s['trades']:>7} {s['win_rate']:>5.1f}% "
              f"${s['pnl']:>+11,.2f} ${s['per_month']:>+8,.2f} "
              f"${s['avg_win']:>+8,.2f} ${s['avg_loss']:>+8,.2f} "
              f"{s['tp_hits']:>5} {s['sl_hits']:>5} {s['eod_hits']:>5} "
              f"{s['avg_lots']:>8.2f} {s['days_mo']:>9.1f} {s['london_pct']:>7.0f}%")
    print("=" * w)
    print()


def print_combined(c: dict):
    w = 68
    print("=" * w)
    print("  COMBINED RESULTS  |  6 PAIRS  |  H4 TREND FILTER  |  SHARED $100,000 ACCOUNT")
    print("=" * w)
    rows = [
        ('Total Trades',           f"{c['trades']:,}"),
        ('Winning / Losing',       f"{c['winners']:,} / {c['losers']:,}  ({c['win_rate']:.1f}%)"),
        ('Total P&L',              f"${c['pnl']:>+,.2f}"),
        ('Total Return',           f"{c['pct_return']:>+.2f}%"),
        ('Avg P&L / month',        f"${c['per_month']:>+,.2f}"),
        ('Final Balance',          f"${c['final_balance']:>,.2f}"),
        ('Max Drawdown ($)',        f"${c['max_dd']:,.2f}  ({c['max_dd_pct']:.2f}%)"),
        ('Worst Trading Day',      f"${c['worst_day']:,.2f}  ({c['worst_day_pct']:.2f}%)"),
        ('Avg Win / Avg Loss',     f"${c['avg_win']:+,.2f}  /  ${c['avg_loss']:+,.2f}"),
        ('Best / Worst Trade',     f"${c['best_trade']:+,.2f}  /  ${c['worst_trade']:+,.2f}"),
        ('TP / SL / EOD exits',    f"{c['tp_hits']} / {c['sl_hits']} / {c['eod_hits']}"),
        ('Avg Lot Size',           f"{c['avg_lots']:.2f}  (range: {c['min_lots']:.2f}-{c['max_lots']:.2f})"),
        ('Avg Profitable Days/mo', f"{c['avg_profitable_days']:.1f}"),
        ('Min Profitable Days/mo', f"{c['min_profitable_days']}"),
        ('Data Window',            f"~{c['months']:.0f} months"),
    ]
    for label, val in rows:
        print(f"  {label:<28}  {val}")
    print("=" * w)
    print()


def print_monthly(c: dict):
    monthly_pnl   = c['monthly_pnl']
    monthly_start = c['monthly_start']

    w = 62
    print("=" * w)
    print("  MONTHLY BREAKDOWN")
    print("=" * w)
    print(f"  {'Month':<10}  {'Start Bal':>12}  {'P&L':>10}  {'Return':>8}  {'Cumul'}  ")
    print("-" * w)

    cumulative = 0.0
    for ym in sorted(monthly_pnl.keys()):
        pnl    = monthly_pnl[ym]
        start  = monthly_start.get(ym, STARTING_BALANCE)
        ret    = pnl / start * 100
        cumulative += pnl
        marker = " <<< TARGET" if (STARTING_BALANCE + cumulative) >= TARGET_BALANCE else ""
        print(f"  {str(ym):<10}  ${start:>11,.2f}  ${pnl:>+9,.2f}  {ret:>+7.2f}%  "
              f"${STARTING_BALANCE + cumulative:>10,.2f}{marker}")
    print("=" * w)
    print()


# -- 8. THE5ERS ASSESSMENT ----------------------------------------------------

def the5ers_assessment(c: dict, equity_log: list):
    w = 72
    print("=" * w)
    print("  THE5ERS PHASE 1 CHALLENGE ASSESSMENT  --  $100K GROWTH PLAN")
    print("=" * w)
    print(f"  Starting balance   : ${STARTING_BALANCE:,.0f}")
    print(f"  Profit target      : 10%  ->  reach ${TARGET_BALANCE:,.0f}")
    print(f"  Max daily loss     : 5% of balance  (${STARTING_BALANCE * 0.05:,.0f} on day 1)")
    print(f"  Hard floor         : ${HARD_FLOOR:,.0f}  (10% from starting balance)")
    print(f"  Min profitable days: 3 per month")
    print(f"  Time limit         : Unlimited")
    print()

    final_bal = c['final_balance']
    max_dd    = c['max_dd']
    worst_day = abs(c['worst_day'])
    min_days  = c['min_profitable_days']

    # Check if hard floor was ever breached
    floor_breach = any(bal <= HARD_FLOOR for _, bal, _, _ in equity_log)

    # Check worst daily loss vs limit (5% of starting balance, conservative check)
    worst_day_pct_of_start = worst_day / STARTING_BALANCE * 100

    # Estimate months to target
    monthly_rate = c['per_month']
    remaining    = TARGET_BALANCE - STARTING_BALANCE   # $10,000
    if monthly_rate > 0:
        months_to_target = remaining / monthly_rate
    else:
        months_to_target = float('inf')

    target_pass  = final_bal >= TARGET_BALANCE
    floor_pass   = not floor_breach
    dd_pass      = max_dd    < (STARTING_BALANCE * 0.10)
    daily_pass   = worst_day < (STARTING_BALANCE * 0.05)
    days_pass    = min_days  >= 3

    def status(p): return "PASS" if p else "FAIL"

    print(f"  {'Requirement':<32}  {'Target':>12}  {'Actual':>12}  Status")
    print("  " + "-" * 66)
    print(f"  {'Profit target reached':<32}  {'$110,000':>12}  "
          f"${final_bal:>11,.2f}  {status(target_pass)}")
    print(f"  {'Hard floor never breached':<32}  {'> $90,000':>12}  "
          f"{'YES' if floor_pass else 'NO -- BREACHED':>12}  {status(floor_pass)}")
    print(f"  {'Max drawdown':<32}  {'< $10,000':>12}  "
          f"${max_dd:>11,.2f}  {status(dd_pass)}")
    print(f"  {'Worst daily loss':<32}  {'< $5,000':>12}  "
          f"${worst_day:>11,.2f}  {status(daily_pass)}")
    print(f"  {'Min profitable days/month':<32}  {'>= 3':>12}  "
          f"{min_days:>12}  {status(days_pass)}")
    print("=" * w)

    all_pass = target_pass and floor_pass and dd_pass and daily_pass and days_pass
    verdict  = "PASSED" if all_pass else "FAILED"
    print(f"\n  VERDICT: {verdict}\n")

    if not all_pass:
        print("  Gaps:")
        if not target_pass:
            gap = TARGET_BALANCE - final_bal
            print(f"    Still ${gap:,.2f} short of $110,000 target")
            if monthly_rate > 0:
                print(f"    At current rate (${monthly_rate:+,.0f}/mo): "
                      f"~{months_to_target:.0f} months to target")
            else:
                print(f"    Strategy is losing money -- target unreachable at this rate")
        if not days_pass:
            print(f"    Worst month only {min_days} profitable days (need 3+)")

    if monthly_rate > 0 and not target_pass:
        print()
        print(f"  Projection at current monthly rate (${monthly_rate:+,.0f}/mo):")
        print(f"    Months to target : ~{months_to_target:.0f} months")
        bal = STARTING_BALANCE
        for m in range(1, int(months_to_target) + 3):
            bal += monthly_rate
            if bal >= TARGET_BALANCE:
                print(f"    Month {m:>2}: ${bal:,.2f}  <<< TARGET HIT")
                break
            else:
                print(f"    Month {m:>2}: ${bal:,.2f}")

    print()
    print("  Key observations:")
    print(f"  - Win rate      : {c['win_rate']:.1f}%  (break-even for 2:1 RR = 33.3%)")
    print(f"  - Avg win/loss  : ${c['avg_win']:+,.2f} / ${c['avg_loss']:+,.2f}  "
          f"(actual RR = {abs(c['avg_win']/c['avg_loss']):.2f}:1 if non-zero)")
    print(f"  - EOD closes    : {c['eod_hits']} trades  "
          f"({c['eod_hits']/max(c['trades'],1)*100:.1f}% ended without SL or TP hit)")
    print(f"  - Dynamic lots  : {c['min_lots']:.2f} to {c['max_lots']:.2f}  "
          f"(grows automatically as balance grows)")
    print("=" * w)
    print()


# -- 9. PLOT -------------------------------------------------------------------

def plot_results(trades: list, equity_log: list, c: dict):
    df = pd.DataFrame(trades)
    if df.empty:
        return

    pair_colors = {
        'GBPUSD': '#1f77b4', 'EURUSD': '#ff7f0e',
        'AUDUSD': '#2ca02c', 'NZDUSD': '#d62728',
        'EURJPY': '#9467bd', 'GBPJPY': '#8c564b',
    }

    fig = plt.figure(figsize=(18, 16))
    gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.52, wspace=0.38)
    ax1 = fig.add_subplot(gs[0, :])   # Equity curve (full width)
    ax2 = fig.add_subplot(gs[1, :])   # Monthly returns (full width)
    ax3 = fig.add_subplot(gs[2, 0])   # P&L per pair
    ax4 = fig.add_subplot(gs[2, 1])   # Win rate per pair
    ax5 = fig.add_subplot(gs[3, 0])   # Lot size evolution
    ax6 = fig.add_subplot(gs[3, 1])   # Trade distribution pie

    eq_dates  = [pd.Timestamp(d) for d, _, _, _ in equity_log]
    eq_values = [v for _, v, _, _ in equity_log]

    # ---- 1. Equity curve ---------------------------------------------------
    final = eq_values[-1]
    color = '#2ca02c' if final >= TARGET_BALANCE else '#d62728'

    ax1.plot(eq_dates, eq_values, color=color, linewidth=1.8, zorder=3,
             label='Account balance')
    ax1.axhline(STARTING_BALANCE, color='grey', linewidth=1, linestyle='--',
                label=f'Start  ${STARTING_BALANCE:,.0f}', zorder=1)
    ax1.axhline(TARGET_BALANCE, color='#2ca02c', linewidth=1.5, linestyle='-.',
                label=f'Target ${TARGET_BALANCE:,.0f}  (+10%)', zorder=2)
    ax1.axhline(HARD_FLOOR, color='#d62728', linewidth=1.5, linestyle='-.',
                label=f'Floor  ${HARD_FLOOR:,.0f}  (-10%)', zorder=2)

    ax1.fill_between(eq_dates, STARTING_BALANCE, eq_values,
                     where=[v >= STARTING_BALANCE for v in eq_values],
                     alpha=0.12, color='green')
    ax1.fill_between(eq_dates, STARTING_BALANCE, eq_values,
                     where=[v < STARTING_BALANCE for v in eq_values],
                     alpha=0.12, color='red')

    ax1.set_ylim(min(min(eq_values), HARD_FLOOR) * 0.995,
                 max(max(eq_values), TARGET_BALANCE) * 1.005)
    ax1.set_title(
        f'The5ers $100K Challenge -- Equity Curve  |  6 Pairs Session Breakout  |  '
        f'Dynamic Lots  |  Asian Range SL/TP\n'
        f'Final balance: ${final:,.2f}  |  Max DD: ${c["max_dd"]:,.0f}  |  '
        f'{c["win_rate"]:.1f}% win rate  |  {c["trades"]} trades',
        fontsize=11, fontweight='bold')
    ax1.set_ylabel('Balance (USD)')
    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')

    # ---- 2. Monthly returns bar chart --------------------------------------
    monthly_pnl = c['monthly_pnl']
    months_sorted = sorted(monthly_pnl.keys())
    mo_labels = [str(m) for m in months_sorted]
    mo_values = [monthly_pnl[m] for m in months_sorted]
    mo_colors = ['#2ca02c' if v > 0 else '#d62728' for v in mo_values]

    ax2.bar(range(len(mo_values)), mo_values, color=mo_colors, alpha=0.85)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_xticks(range(len(mo_labels)))
    ax2.set_xticklabels(mo_labels, rotation=45, ha='right', fontsize=8)
    ax2.set_title('Monthly P&L  (green = profit, red = loss)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('P&L (USD)')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    # ---- 3. P&L per pair --------------------------------------------------
    pair_pnl   = df.groupby('Symbol')['P&L (USD)'].sum().sort_values()
    bar_colors = ['#2ca02c' if v >= 0 else '#d62728' for v in pair_pnl]
    ax3.bar(range(len(pair_pnl)), pair_pnl.values, color=bar_colors, alpha=0.85)
    ax3.set_xticks(range(len(pair_pnl)))
    ax3.set_xticklabels(pair_pnl.index)
    ax3.axhline(0, color='black', linewidth=0.8)
    ax3.set_title('Total P&L per Pair', fontsize=11, fontweight='bold')
    ax3.set_ylabel('P&L (USD)')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    for i, (sym, v) in enumerate(pair_pnl.items()):
        offset = abs(pair_pnl.values).max() * 0.04
        ax3.text(i, v + (offset if v >= 0 else -offset * 2.5),
                 f'${v:+,.0f}', ha='center', fontsize=9, fontweight='bold')

    # ---- 4. Win rate per pair ---------------------------------------------
    wr = df.groupby('Symbol').apply(
        lambda g: (g['P&L (USD)'] > 0).mean() * 100).sort_values()
    wc = ['#2ca02c' if v >= 33.3 else '#d62728' for v in wr]
    ax4.bar(range(len(wr)), wr.values, color=wc, alpha=0.85)
    ax4.set_xticks(range(len(wr)))
    ax4.set_xticklabels(wr.index)
    ax4.axhline(33.3, color='grey', linewidth=1, linestyle='--',
                label='33.3% breakeven (2:1 RR)')
    ax4.set_title('Win Rate per Pair', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Win Rate (%)')
    ax4.set_ylim(0, 100)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    for i, (sym, v) in enumerate(wr.items()):
        ax4.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')

    # ---- 5. Lot size evolution --------------------------------------------
    df_s = df.copy()
    df_s['exit_dt'] = pd.to_datetime(df_s['Date'])
    df_s.sort_values('exit_dt', inplace=True)

    for sym, grp in df_s.groupby('Symbol'):
        ax5.scatter(grp['exit_dt'], grp['Lots'], alpha=0.55, s=18,
                    color=pair_colors.get(sym, 'grey'), label=sym, zorder=3)

    if len(df_s) >= 10:
        roll = df_s.set_index('exit_dt')['Lots'].rolling('30D', min_periods=1).mean()
        ax5.plot(roll.index, roll.values, color='black', linewidth=1.8,
                 label='30-day avg', zorder=4)

    ax5.set_title('Lot Size Evolution  (Dynamic Sizing)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Lot Size')
    ax5.legend(fontsize=8, ncol=2)
    ax5.grid(True, alpha=0.3)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=30, ha='right')

    # ---- 6. Trade distribution pie ----------------------------------------
    counts = df.groupby('Symbol').size().sort_values(ascending=False)
    ax6.pie(counts.values, labels=counts.index, autopct='%1.0f%%', startangle=90,
            colors=[pair_colors.get(s, 'grey') for s in counts.index])
    ax6.set_title('Trade Distribution by Pair', fontsize=11, fontweight='bold')

    out_dir  = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'the5ers_backtest.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"Chart saved : {out_path}")


# -- 10. SAVE TRADE LOG --------------------------------------------------------

def save_trade_log(trades: list):
    df       = pd.DataFrame(trades)
    out_dir  = os.path.join(os.path.dirname(__file__), '..', 'data')
    out_path = os.path.join(out_dir, 'trade_log_the5ers.csv')
    df.to_csv(out_path, index=False)
    print(f"Trade log   : {out_path}")


# -- MAIN ----------------------------------------------------------------------

connect_mt5()
all_data = fetch_all_data()
mt5.shutdown()

if not all_data:
    print("No data fetched. Exiting.")
    sys.exit(1)

trades, equity_log, monthly_start = run_backtest(all_data)

if not trades:
    print("No trades generated. Check data availability and session hours.")
    sys.exit(0)

# Date range
first_date = pd.to_datetime(trades[0]['Date'])
last_date  = pd.to_datetime(trades[-1]['Date'])
months     = max((last_date - first_date).days / 30.0, 1.0)

pair_stats = [compute_pair_stats(trades, sym, months) for sym in all_data]
combined   = compute_combined_stats(trades, equity_log, monthly_start)

print_per_pair(pair_stats)
print_combined(combined)
print_monthly(combined)
the5ers_assessment(combined, equity_log)
save_trade_log(trades)
plot_results(trades, equity_log, combined)

print("Done!")
