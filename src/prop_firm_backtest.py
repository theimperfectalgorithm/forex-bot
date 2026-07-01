"""
Forex Bot -- Professional 3-Timeframe Prop Firm Backtest

Strategy:
  H4  -- 50/200 SMA trend filter (only trade in direction of H4 trend)
  H1  -- 50/200 SMA crossover + RSI-14/60-40 fires the signal
  M15 -- Wait for M15 price to pull back to M15 50 SMA before entering

Pairs: EURUSD, GBPUSD, EURJPY, AUDUSD, USDCHF

Dynamic position sizing:
  Risk per trade = 0.5% of current balance
  Lot size = risk / (SL_pips x pip_value_per_lot)
  SL = 50 pips  |  TP = 100 pips  |  2:1 risk-reward

Daily risk management:
  Max daily loss    = 5% of balance at start of day (shared across all pairs)
  Per-pair pause    = 2 consecutive losses on that pair stops it for the day
  Hard stop         = 10% total drawdown from starting balance -- trading halts permanently

Backtest: $10,000 starting balance | 3 years | all data from MetaTrader 5

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
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent.parent))
from core import data_loader

# -- SETTINGS -------------------------------------------------------------------

STARTING_BALANCE      = 10_000.00
MONTHS                = 36
SL_PIPS               = 50
TP_PIPS               = 100
RSI_PERIOD            = 14
BUY_THRESHOLD         = 60
SELL_THRESHOLD        = 40
PULLBACK_TIMEOUT_BARS = 16        # 4 hours on M15 before signal expires
MAX_CONSEC_LOSSES     = 2         # consecutive losses on one pair -> pair stops for the day
RISK_PER_TRADE_PCT    = 0.005     # 0.5% of balance per trade
MAX_DAILY_LOSS_PCT    = 0.05      # 5% -- stops ALL pairs if hit
MAX_TOTAL_DD_PCT      = 0.10      # 10% -- hard stop from starting balance
MIN_LOT               = 0.01
MAX_LOT               = 2.00

PAIRS = {
    'EURUSD': {'pip_size': 0.0001, 'pip_value': 10.00},
    'GBPUSD': {'pip_size': 0.0001, 'pip_value': 10.00},
    'EURJPY': {'pip_size': 0.01,   'pip_value':  6.67},
    'AUDUSD': {'pip_size': 0.0001, 'pip_value': 10.00},
    'USDCHF': {'pip_size': 0.0001, 'pip_value': 10.00},
}

# -- 1. MT5 CONNECTION & DATA FETCH ---------------------------------------------

def connect_mt5():
    if not MT5_AVAILABLE:
        print("MetaTrader5 not available -- using offline CSV data from "
              "data/historical/\n")
        return
    print("Connecting to MetaTrader 5...")
    if not mt5.initialize():
        print(f"ERROR: {mt5.last_error()}")
        print("Make sure MetaTrader 5 is open and logged in.")
        sys.exit(1)
    info    = mt5.terminal_info()
    account = mt5.account_info()
    print(f"Connected : {info.name}")
    print(f"Account   : {account.login}  ({account.server})\n")


def _fetch(symbol: str, timeframe: str, date_from: datetime, date_to: datetime):
    try:
        df = data_loader.get_bars(symbol, timeframe, date_from, date_to)
    except (ValueError, FileNotFoundError):
        return None
    return df[['Open', 'High', 'Low', 'Close']]


def fetch_all_data() -> dict:
    date_to   = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=MONTHS * 30)

    print(f"Fetching data  {date_from.date()} to {date_to.date()}")
    print("-" * 60)

    all_data = {}
    for sym in PAIRS:
        h4  = _fetch(sym, 'H4',  date_from, date_to)
        h1  = _fetch(sym, 'H1',  date_from, date_to)
        m15 = _fetch(sym, 'M15', date_from, date_to)

        if h4 is None or h1 is None or m15 is None:
            print(f"  {sym}: SKIPPED -- missing data")
            continue

        all_data[sym] = {'H4': h4, 'H1': h1, 'M15': m15}
        h4_range  = f"{h4.index[0].date()} to {h4.index[-1].date()}"
        print(f"  {sym}: H4={len(h4):,}  H1={len(h1):,}  M15={len(m15):,}  ({h4_range})")

    print()
    return all_data


# -- 2. INDICATOR PREPARATION ---------------------------------------------------

def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def prepare_h4(h4_df: pd.DataFrame) -> pd.DataFrame:
    """Compute H4 50/200 SMA trend.  Trend: +1 bullish, -1 bearish, 0 neutral."""
    df          = h4_df.copy()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200']= df['Close'].rolling(200).mean()
    df          = df.dropna(subset=['SMA200'])
    df['Trend'] = 0
    df.loc[df['SMA50'] > df['SMA200'], 'Trend'] =  1
    df.loc[df['SMA50'] < df['SMA200'], 'Trend'] = -1
    return df


def prepare_h1(h1_df: pd.DataFrame) -> pd.DataFrame:
    """Compute H1 50/200 SMA crossover + RSI-14 signal.  Signal: +1 BUY, -1 SELL, 0 none."""
    df           = h1_df.copy()
    df['SMA50']  = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df['RSI']    = _rsi(df['Close'], RSI_PERIOD)
    df           = df.dropna(subset=['SMA200'])

    df['Position'] = (df['SMA50'] > df['SMA200']).astype(int)
    crossover      = df['Position'].diff()

    df['Signal'] = 0
    df.loc[(crossover ==  1) & (df['RSI'] < BUY_THRESHOLD),  'Signal'] =  1
    df.loc[(crossover == -1) & (df['RSI'] > SELL_THRESHOLD), 'Signal'] = -1
    return df


def prepare_m15(m15_df: pd.DataFrame) -> pd.DataFrame:
    """Add M15 50 SMA for pullback detection."""
    df          = m15_df.copy()
    df['SMA50'] = df['Close'].rolling(50).mean()
    return df.dropna(subset=['SMA50'])


# -- 3. LOOKUP HELPERS ----------------------------------------------------------

def h4_trend_at(h4_df: pd.DataFrame, ts: pd.Timestamp) -> int:
    """Return H4 trend at or before timestamp ts."""
    idx = h4_df.index.searchsorted(ts, side='right') - 1
    if idx < 0:
        return 0
    return int(h4_df.iloc[idx]['Trend'])


def h1_signal_at(h1_df: pd.DataFrame, bar_ts: pd.Timestamp) -> int:
    """Return the H1 signal value for the bar that opened at bar_ts."""
    try:
        return int(h1_df.at[bar_ts, 'Signal'])
    except KeyError:
        return 0


# -- 4. POSITION SIZING ---------------------------------------------------------

def calc_lots(balance: float, sym: str) -> float:
    """Dynamic lot size: risk 0.5% of balance, bounded by SL distance."""
    risk_usd = balance * RISK_PER_TRADE_PCT
    lots     = risk_usd / (SL_PIPS * PAIRS[sym]['pip_value'])
    return round(max(MIN_LOT, min(lots, MAX_LOT)), 2)


# -- 5. BACKTEST ENGINE ---------------------------------------------------------

def run_backtest(all_data: dict):
    print("Preparing indicators ...")
    h4_prep  = {sym: prepare_h4(all_data[sym]['H4'])   for sym in all_data}
    h1_prep  = {sym: prepare_h1(all_data[sym]['H1'])   for sym in all_data}
    m15_prep = {sym: prepare_m15(all_data[sym]['M15']) for sym in all_data}

    # Unified sorted M15 timestamps across all pairs
    all_ts = sorted(set().union(*[set(m15_prep[sym].index) for sym in m15_prep]))
    print(f"Running backtest over {len(all_ts):,} M15 timestamps ({len(all_data)} pairs) ...")

    balance    = STARTING_BALANCE
    all_trades = []
    equity_log = []       # [(timestamp, balance), ...]
    hard_stop  = False

    # Per-pair mutable state
    states = {sym: {
        'open_trade'     : None,   # active trade dict or None
        'pending_signal' : None,   # {'direction': 'LONG'/'SHORT', 'bars': int}
        'enter_next'     : None,   # 'LONG' | 'SHORT' -- enter at open of next bar
        'consec_losses'  : 0,
        'stopped_today'  : False,
        'last_h1_checked': None,   # prevents rechecking same H1 bar twice
    } for sym in all_data}

    current_date     = None
    daily_pnl        = 0.0
    daily_loss_limit = 0.0

    step = max(len(all_ts) // 10, 1)

    for i, ts in enumerate(all_ts):

        # -- Progress report --------------------------------------------------
        if i % step == 0:
            print(f"  {i/len(all_ts)*100:5.1f}%  {ts.date()}  ${balance:,.2f}")

        ts_date = ts.date()

        # -- Daily rollover ---------------------------------------------------
        if ts_date != current_date:
            current_date = ts_date

            # Hard stop: check 10% total drawdown at start of each day
            loss_pct = (STARTING_BALANCE - balance) / STARTING_BALANCE
            if loss_pct >= MAX_TOTAL_DD_PCT:
                print(f"\n  HARD STOP on {ts_date}  -- account down {loss_pct*100:.1f}%")
                hard_stop = True
                break

            daily_pnl        = 0.0
            daily_loss_limit = balance * MAX_DAILY_LOSS_PCT

            for st in states.values():
                st['consec_losses'] = 0
                st['stopped_today'] = False

        equity_log.append((ts, balance))

        # -- Process each pair at this M15 bar -------------------------------
        for sym in all_data:
            if ts not in m15_prep[sym].index:
                continue

            bar   = m15_prep[sym].loc[ts]
            state = states[sym]
            pc    = PAIRS[sym]

            # -- A. Manage open trade: bar-by-bar SL/TP check -----------------
            if state['open_trade'] is not None:
                tr  = state['open_trade']
                dir_ = tr['direction']
                sl   = tr['sl']
                tp   = tr['tp']
                hi   = bar['High']
                lo   = bar['Low']

                sl_hit = (dir_ == 'LONG' and lo <= sl) or (dir_ == 'SHORT' and hi >= sl)
                tp_hit = (dir_ == 'LONG' and hi >= tp) or (dir_ == 'SHORT' and lo <= tp)

                if sl_hit or tp_hit:
                    # Conservative tie-break: SL wins if both triggered in same bar
                    if sl_hit and tp_hit:
                        close_p, reason = sl, 'SL'
                    elif sl_hit:
                        close_p, reason = sl, 'SL'
                    else:
                        close_p, reason = tp, 'TP'

                    pips    = ((close_p - tr['entry']) if dir_ == 'LONG'
                               else (tr['entry'] - close_p)) / pc['pip_size']
                    pnl     = round(pips * pc['pip_value'] * tr['lots'], 2)
                    balance = round(balance + pnl, 2)
                    daily_pnl += pnl

                    all_trades.append({
                        'Symbol'      : sym,
                        'Direction'   : dir_,
                        'Entry Date'  : tr['entry_time'].strftime('%Y-%m-%d %H:%M'),
                        'Exit Date'   : ts.strftime('%Y-%m-%d %H:%M'),
                        'Entry Price' : round(tr['entry'], 5),
                        'Exit Price'  : round(close_p, 5),
                        'SL'          : round(sl, 5),
                        'TP'          : round(tp, 5),
                        'Lots'        : tr['lots'],
                        'Risk ($)'    : tr['risk_usd'],
                        'Pips'        : round(pips, 1),
                        'P&L (USD)'   : pnl,
                        'Balance'     : balance,
                        'Exit Reason' : reason,
                        'Result'      : 'WIN' if pnl > 0 else 'LOSS',
                    })
                    state['open_trade'] = None

                    if pnl <= 0:
                        state['consec_losses'] += 1
                        if state['consec_losses'] >= MAX_CONSEC_LOSSES:
                            state['stopped_today'] = True
                    else:
                        state['consec_losses'] = 0

                    # Global daily loss limit: stop ALL pairs if hit
                    if daily_pnl <= -daily_loss_limit:
                        for st in states.values():
                            st['stopped_today'] = True

            # -- Skip entry logic if this pair (or all pairs) is stopped -------
            if state['stopped_today'] or daily_pnl <= -daily_loss_limit:
                state['pending_signal'] = None
                state['enter_next']     = None
                continue

            # -- B. Execute entry flagged from previous bar's pullback ---------
            if state['enter_next'] is not None and state['open_trade'] is None:
                direction = state['enter_next']
                entry_p   = bar['Open']
                lots      = calc_lots(balance, sym)
                risk_usd  = round(balance * RISK_PER_TRADE_PCT, 2)

                if direction == 'LONG':
                    sl_p = entry_p - SL_PIPS * pc['pip_size']
                    tp_p = entry_p + TP_PIPS * pc['pip_size']
                else:
                    sl_p = entry_p + SL_PIPS * pc['pip_size']
                    tp_p = entry_p - TP_PIPS * pc['pip_size']

                state['open_trade'] = {
                    'direction' : direction,
                    'entry'     : entry_p,
                    'entry_time': ts,
                    'sl'        : sl_p,
                    'tp'        : tp_p,
                    'lots'      : lots,
                    'risk_usd'  : risk_usd,
                }
                state['enter_next'] = None

            # -- C. H1 signal detection (fires on first M15 bar of each H1) ----
            # The H1 bar that just closed has open_time = ts - 1H.
            # We check it once (last_h1_checked guard) to avoid reprocessing.
            if ts.minute == 0:
                prev_h1 = ts - pd.Timedelta(hours=1)

                if state['last_h1_checked'] != prev_h1:
                    state['last_h1_checked'] = prev_h1
                    sig      = h1_signal_at(h1_prep[sym], prev_h1)
                    h4_trend = h4_trend_at(h4_prep[sym], ts)

                    if sig == 1 and h4_trend == 1:       # BUY signal, H4 bullish
                        if state['open_trade'] is None:
                            state['pending_signal'] = {'direction': 'LONG', 'bars': 0}
                    elif sig == -1 and h4_trend == -1:   # SELL signal, H4 bearish
                        if state['open_trade'] is None:
                            state['pending_signal'] = {'direction': 'SHORT', 'bars': 0}
                    elif sig != 0:
                        # H1 fired but H4 trend opposes -- cancel any pending
                        state['pending_signal'] = None

            # -- D. M15 pullback entry refinement -----------------------------
            if state['pending_signal'] is not None and state['open_trade'] is None:
                ps = state['pending_signal']
                ps['bars'] += 1

                if ps['bars'] > PULLBACK_TIMEOUT_BARS:
                    # Signal expired -- price never pulled back
                    state['pending_signal'] = None
                else:
                    sma    = bar['SMA50']
                    close_ = bar['Close']

                    pulled_back = (
                        (ps['direction'] == 'LONG'  and close_ <= sma) or
                        (ps['direction'] == 'SHORT' and close_ >= sma)
                    )
                    if pulled_back:
                        # Flag entry for the NEXT M15 bar's open (avoids look-ahead)
                        state['enter_next']     = ps['direction']
                        state['pending_signal'] = None

    print(f"  100.0%  Complete.  Final balance: ${balance:,.2f}\n")
    return all_trades, equity_log


# -- 6. STATISTICS --------------------------------------------------------------

def compute_stats(trades: list, equity_log: list, sym: str = None) -> dict:
    """Compute statistics for one pair (sym) or all pairs combined (sym=None)."""
    df = pd.DataFrame(trades)
    if sym:
        df = df[df['Symbol'] == sym]
    if df.empty:
        return None

    winners = df[df['P&L (USD)'] > 0]
    losers  = df[df['P&L (USD)'] <= 0]

    df = df.copy()
    df['exit_dt'] = pd.to_datetime(df['Exit Date'])
    df['ym']      = df['exit_dt'].dt.to_period('M')

    active_by_month = df.groupby('ym')['exit_dt'].apply(lambda g: g.dt.date.nunique())
    avg_days_mo     = float(active_by_month.mean()) if not active_by_month.empty else 0.0

    first  = df['exit_dt'].min()
    last   = df['exit_dt'].max()
    months = max((last - first).days / 30.0, 1.0)

    pnl_total = df['P&L (USD)'].sum()

    # Drawdown and worst day (combined only)
    max_dd        = 0.0
    worst_day_pct = 0.0
    final_balance = None

    if sym is None:
        eq_vals = [v for _, v in equity_log]
        peak    = STARTING_BALANCE
        for v in eq_vals:
            peak   = max(peak, v)
            max_dd = max(max_dd, (peak - v) / peak * 100)

        final_balance = equity_log[-1][1] if equity_log else STARTING_BALANCE

        day_pnl = df.groupby(df['exit_dt'].dt.date)['P&L (USD)'].sum()
        worst   = day_pnl.min() if not day_pnl.empty else 0.0
        worst_day_pct = abs(min(worst, 0)) / STARTING_BALANCE * 100

    return {
        'symbol'        : sym or 'COMBINED',
        'trades'        : len(df),
        'winners'       : len(winners),
        'losers'        : len(losers),
        'win_rate'      : round(len(winners) / len(df) * 100, 1),
        'pnl'           : round(pnl_total, 2),
        'per_month'     : round(pnl_total / months, 2),
        'pct_return'    : round(pnl_total / STARTING_BALANCE * 100, 2),
        'final_balance' : round(final_balance, 2) if final_balance is not None else None,
        'max_dd'        : round(max_dd, 1),
        'worst_day_pct' : round(worst_day_pct, 2),
        'avg_win'       : round(winners['P&L (USD)'].mean(), 2) if not winners.empty else 0.0,
        'avg_loss'      : round(losers['P&L (USD)'].mean(),  2) if not losers.empty  else 0.0,
        'best'          : round(df['P&L (USD)'].max(), 2),
        'worst'         : round(df['P&L (USD)'].min(), 2),
        'tp_hits'       : int((df['Exit Reason'] == 'TP').sum()),
        'sl_hits'       : int((df['Exit Reason'] == 'SL').sum()),
        'avg_lots'      : round(df['Lots'].mean(), 2),
        'min_lots'      : round(df['Lots'].min(), 2),
        'max_lots'      : round(df['Lots'].max(), 2),
        'days_mo'       : round(avg_days_mo, 1),
        'trades_mo'     : round(len(df) / months, 1),
        'months'        : round(months, 1),
    }


# -- 7. PRINT RESULTS -----------------------------------------------------------

def print_per_pair(pair_stats: list):
    w = 104
    print("=" * w)
    print("  PER-PAIR RESULTS  |  H4 Trend + H1 Signal + M15 Pullback  |  Dynamic Lot Sizing")
    print(f"  SL = {SL_PIPS} pips  |  TP = {TP_PIPS} pips  |  "
          f"{RISK_PER_TRADE_PCT*100:.1f}% risk/trade  |  ~3 years  |  Shared ${STARTING_BALANCE:,.0f}")
    print("=" * w)
    print(f"  {'Pair':<8} {'Trades':>7} {'Win%':>6} {'P&L ($)':>12} "
          f"{'$/mo':>9} {'AvgWin':>9} {'AvgLoss':>9} "
          f"{'TP':>5} {'SL':>5} {'AvgLot':>8} {'Days/mo':>9}")
    print("-" * w)
    for s in pair_stats:
        if s is None:
            continue
        print(f"  {s['symbol']:<8} {s['trades']:>7} {s['win_rate']:>5.1f}% "
              f"${s['pnl']:>+11,.2f} ${s['per_month']:>+8,.2f} "
              f"${s['avg_win']:>+8,.2f} ${s['avg_loss']:>+8,.2f} "
              f"{s['tp_hits']:>5} {s['sl_hits']:>5} "
              f"{s['avg_lots']:>8.2f} {s['days_mo']:>9.1f}")
    print("=" * w)
    print()


def print_combined(c: dict):
    w = 64
    print("=" * w)
    print("  COMBINED RESULTS -- 5 PAIRS  |  SHARED $10,000 ACCOUNT")
    print("=" * w)
    rows = [
        ('Total Trades',          f"{c['trades']:,}"),
        ('Winning Trades',        f"{c['winners']:,}  ({c['win_rate']:.1f}%)"),
        ('Losing Trades',         f"{c['losers']:,}"),
        ('Total P&L',             f"${c['pnl']:>+,.2f}"),
        ('Total Return',          f"{c['pct_return']:>+.2f}%"),
        ('Avg P&L / month',       f"${c['per_month']:>+,.2f}"),
        ('Final Balance',         f"${c['final_balance']:>,.2f}"),
        ('Max Drawdown',          f"{c['max_dd']:.1f}%"),
        ('Worst Single-Day Loss', f"{c['worst_day_pct']:.2f}%  of starting balance"),
        ('Avg Win',               f"${c['avg_win']:>+,.2f}"),
        ('Avg Loss',              f"${c['avg_loss']:>+,.2f}"),
        ('Best Trade',            f"${c['best']:>+,.2f}"),
        ('Worst Trade',           f"${c['worst']:>+,.2f}"),
        ('TP hits / SL hits',     f"{c['tp_hits']} / {c['sl_hits']}"),
        ('Trades / month',        f"{c['trades_mo']:.1f}"),
        ('Active days / month',   f"{c['days_mo']:.1f}"),
        ('Avg Lot Size',          f"{c['avg_lots']:.2f}"),
        ('Lot range',             f"{c['min_lots']:.2f}  to  {c['max_lots']:.2f}"),
        ('Data window',           f"~{c['months']:.0f} months"),
    ]
    for label, val in rows:
        print(f"  {label:<28}  {val}")
    print("=" * w)
    print()


# -- 8. PROP FIRM ASSESSMENT ----------------------------------------------------

def prop_firm_assessment(c: dict):
    target_60d   = STARTING_BALANCE * 0.08     # 8% profit in 60 days
    monthly_need = target_60d / 2.0            # = $400/month on a $10k account

    w = 70
    print("=" * w)
    print("  PROP FIRM READINESS -- 60-DAY CHALLENGE ASSESSMENT")
    print("=" * w)
    print(f"  Account size        : ${STARTING_BALANCE:,.0f}")
    print(f"  Profit target       : 8% in 60 days = ${target_60d:,.0f}  "
          f"(${monthly_need:,.0f}/month)")
    print(f"  Max daily loss      : 5%  (${STARTING_BALANCE * 0.05:,.0f})")
    print(f"  Max total DD        : 10%  (${STARTING_BALANCE * 0.10:,.0f})")
    print(f"  Min trading days    : 10 / month")
    print()

    actual_monthly  = c['per_month']
    actual_dd       = c['max_dd']
    actual_day_loss = c['worst_day_pct']
    actual_days     = c['days_mo']

    def status(pass_): return "PASS" if pass_ else "FAIL"

    ret_pass  = actual_monthly  >= monthly_need
    dd_pass   = actual_dd       <  10.0
    day_pass  = actual_day_loss <   5.0
    days_pass = actual_days     >=  10.0

    print(f"  {'Requirement':<30}  {'Target':>10}  {'Actual':>10}  Status")
    print("  " + "-" * 64)
    print(f"  {'Monthly return':<30}  {'$'+f'{monthly_need:,.0f}/mo':>10}  "
          f"{'$'+f'{actual_monthly:+,.0f}/mo':>10}  {status(ret_pass)}")
    print(f"  {'Max total drawdown':<30}  {'< 10%':>10}  "
          f"{actual_dd:>9.1f}%  {status(dd_pass)}")
    print(f"  {'Worst single-day loss':<30}  {'< 5%':>10}  "
          f"{actual_day_loss:>9.2f}%  {status(day_pass)}")
    print(f"  {'Active trading days':<30}  {'>= 10/mo':>10}  "
          f"{actual_days:>9.1f}  {status(days_pass)}")
    print("=" * w)

    all_pass = ret_pass and dd_pass and day_pass and days_pass
    verdict  = "READY" if all_pass else "NOT READY"
    print(f"\n  VERDICT: Strategy is {verdict} for a prop firm challenge.\n")

    if not all_pass:
        print("  Gaps to bridge:")
        if not ret_pass:
            gap_x       = monthly_need / max(actual_monthly, 0.01)
            lots_needed = c['avg_lots'] * gap_x
            est_dd      = actual_dd * gap_x
            print(f"    Return : ${actual_monthly:+,.0f}/mo  vs  "
                  f"${monthly_need:+,.0f}/mo needed  ({gap_x:.1f}x gap)")
            print(f"    To close the gap via lot size alone:")
            print(f"      Average lots required  : {lots_needed:.2f}  "
                  f"(current avg {c['avg_lots']:.2f})")
            safe = "SAFE" if est_dd < 10 else "UNSAFE -- would exceed 10% DD limit"
            print(f"      Projected drawdown     : {est_dd:.1f}%  ({safe})")
        if not days_pass:
            deficit = 10.0 - actual_days
            print(f"    Days   : {actual_days:.1f}/mo  vs  10 required  "
                  f"({deficit:.1f} more active days/month needed)")

    print()
    print("  Roadmap to prop firm readiness:")
    print("  1. Let the account compound -- lot sizes grow automatically as balance rises")
    print("  2. Add more pairs (GBPJPY, NZDUSD) to increase active trading days/month")
    print("  3. Walk-forward test on 2025-2026 out-of-sample data before attempting")
    print("  4. Paper trade for 30 days -- verify live timing matches backtest closely")
    print("  5. Only fund a challenge once paper-trade equity curve closely matches this")
    print("=" * w)
    print()


# -- 9. PLOTTING ----------------------------------------------------------------

def plot_results(trades: list, equity_log: list):
    df = pd.DataFrame(trades)
    if df.empty:
        print("No trades to plot.")
        return

    # Thin equity log to ~1 point per H1 bar (every 4th M15 bar) for speed
    step     = max(len(equity_log) // 6000, 1)
    eq_ts    = [t for t, _ in equity_log[::step]]
    eq_val   = [v for _, v in equity_log[::step]]
    # Always include the final point
    if equity_log[-1] not in zip(eq_ts, eq_val):
        eq_ts.append(equity_log[-1][0])
        eq_val.append(equity_log[-1][1])

    pair_colors = {
        'EURUSD': '#1f77b4', 'GBPUSD': '#ff7f0e',
        'EURJPY': '#2ca02c', 'AUDUSD': '#d62728', 'USDCHF': '#9467bd',
    }

    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.38)
    ax1 = fig.add_subplot(gs[0, :])   # full-width equity curve
    ax2 = fig.add_subplot(gs[1, 0])   # P&L per pair
    ax3 = fig.add_subplot(gs[1, 1])   # win rate per pair
    ax4 = fig.add_subplot(gs[2, 0])   # lot size evolution
    ax5 = fig.add_subplot(gs[2, 1])   # trade distribution pie

    # -- Equity curve ----------------------------------------------------------
    final  = eq_val[-1]
    color  = '#2ca02c' if final >= STARTING_BALANCE else '#d62728'
    ax1.plot(eq_ts, eq_val, color=color, linewidth=1.8, zorder=3, label='Account balance')
    ax1.axhline(STARTING_BALANCE, color='grey', linewidth=1, linestyle='--',
                label=f'Start  ${STARTING_BALANCE:,.0f}', zorder=1)
    ax1.fill_between(eq_ts, STARTING_BALANCE, eq_val,
                     where=[v >= STARTING_BALANCE for v in eq_val],
                     alpha=0.15, color='green', label='Profit')
    ax1.fill_between(eq_ts, STARTING_BALANCE, eq_val,
                     where=[v < STARTING_BALANCE for v in eq_val],
                     alpha=0.15, color='red', label='Loss')
    ax1.set_title(
        f'Combined Equity Curve  |  5 Pairs  |  H4+H1+M15 Strategy  |  '
        f'Dynamic Lot Sizing  |  Shared $10,000\n'
        f'Final balance: ${final:,.2f}  |  '
        f'SL={SL_PIPS}p / TP={TP_PIPS}p  |  {RISK_PER_TRADE_PCT*100:.1f}% risk/trade',
        fontsize=11, fontweight='bold')
    ax1.set_ylabel('Balance (USD)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')

    # -- P&L per pair ----------------------------------------------------------
    pair_pnl   = df.groupby('Symbol')['P&L (USD)'].sum().sort_values()
    bar_colors = ['#2ca02c' if v >= 0 else '#d62728' for v in pair_pnl]
    ax2.bar(range(len(pair_pnl)), pair_pnl.values, color=bar_colors, alpha=0.85)
    ax2.set_xticks(range(len(pair_pnl)))
    ax2.set_xticklabels(pair_pnl.index)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.set_title('Total P&L per Pair', fontsize=11, fontweight='bold')
    ax2.set_ylabel('P&L (USD)')
    ax2.grid(True, alpha=0.3, axis='y')
    for idx, (sym, v) in enumerate(pair_pnl.items()):
        offset = abs(pair_pnl.values).max() * 0.04
        ax2.text(idx, v + (offset if v >= 0 else -offset * 2.5),
                 f'${v:+,.0f}', ha='center', fontsize=9, fontweight='bold')

    # -- Win rate per pair -----------------------------------------------------
    win_rates  = df.groupby('Symbol').apply(
        lambda g: (g['P&L (USD)'] > 0).mean() * 100).sort_values()
    wr_colors  = ['#2ca02c' if v >= 50 else '#ff7f0e' for v in win_rates]
    ax3.bar(range(len(win_rates)), win_rates.values, color=wr_colors, alpha=0.85)
    ax3.set_xticks(range(len(win_rates)))
    ax3.set_xticklabels(win_rates.index)
    ax3.axhline(50, color='grey', linewidth=1, linestyle='--', label='50% break-even')
    ax3.set_title('Win Rate per Pair', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Win Rate (%)')
    ax3.set_ylim(0, 100)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    for idx, (sym, v) in enumerate(win_rates.items()):
        ax3.text(idx, v + 2, f'{v:.1f}%', ha='center', fontsize=9, fontweight='bold')

    # -- Lot size over time ----------------------------------------------------
    df_s = df.copy()
    df_s['exit_dt'] = pd.to_datetime(df_s['Exit Date'])
    df_s.sort_values('exit_dt', inplace=True)

    for sym, grp in df_s.groupby('Symbol'):
        ax4.scatter(grp['exit_dt'], grp['Lots'],
                    alpha=0.6, s=18, zorder=3,
                    color=pair_colors.get(sym, 'grey'), label=sym)

    if len(df_s) >= 10:
        roll = df_s.set_index('exit_dt')['Lots'].rolling('30D', min_periods=1).mean()
        ax4.plot(roll.index, roll.values, color='black', linewidth=1.8,
                 label='30-day rolling avg', zorder=4)

    ax4.set_title('Lot Size Evolution  (Dynamic Sizing)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Lot Size')
    ax4.legend(fontsize=8, ncol=2)
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, ha='right')

    # -- Trade distribution pie ------------------------------------------------
    counts = df.groupby('Symbol').size().sort_values(ascending=False)
    ax5.pie(counts.values, labels=counts.index, autopct='%1.0f%%', startangle=90,
            colors=[pair_colors.get(s, 'grey') for s in counts.index])
    ax5.set_title('Trade Distribution by Pair', fontsize=11, fontweight='bold')

    out_dir  = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'prop_firm_backtest.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"Chart saved : {out_path}")


# -- 10. SAVE TRADE LOG ---------------------------------------------------------

def save_trade_log(trades: list):
    df       = pd.DataFrame(trades)
    out_dir  = os.path.join(os.path.dirname(__file__), '..', 'data')
    out_path = os.path.join(out_dir, 'trade_log_prop_firm.csv')
    df.to_csv(out_path, index=False)
    print(f"Trade log   : {out_path}")


# -- MAIN -----------------------------------------------------------------------

connect_mt5()
all_data = fetch_all_data()
if MT5_AVAILABLE:
    mt5.shutdown()

if not all_data:
    print("No data fetched. Exiting.")
    sys.exit(1)

trades, equity_log = run_backtest(all_data)

if not trades:
    print("No trades generated. Check MT5 data availability and strategy parameters.")
    sys.exit(0)

pair_stats = [compute_stats(trades, equity_log, sym) for sym in all_data]
combined   = compute_stats(trades, equity_log)

print_per_pair(pair_stats)
print_combined(combined)
prop_firm_assessment(combined)
save_trade_log(trades)
plot_results(trades, equity_log)

print("Done!")
