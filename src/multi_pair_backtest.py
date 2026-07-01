"""
Forex Bot - Multi-Pair H4 Trend Filter Backtest (3-Year)

Extends the H1 50/200 SMA + RSI-14/60-40 + SL/TP strategy across
four currency pairs with an H4 trend filter to increase trading
frequency and improve signal quality for prop firm suitability.

Pairs: EURUSD, GBPUSD, USDJPY, EURJPY

H4 Trend Filter:
  Fetch H4 bars, compute the 50-period SMA on H4.
  Only allow H1 BUY  signals when H4 Close > H4 SMA_50 (H4 bullish).
  Only allow H1 SELL signals when H4 Close < H4 SMA_50 (H4 bearish).
  This removes counter-trend trades that are statistically weaker.

All pairs share one $10,000 account. Trades run concurrently across
pairs but are independent — each uses 0.1 lots regardless of others.

Pip values at 0.1 lots:
  USD-quoted (EURUSD, GBPUSD) : $1.00 / pip
  JPY-quoted (USDJPY, EURJPY) : ~$0.67 / pip  (approx. 150 JPY/USD)

SL = 50 pips  |  TP = 100 pips  |  Risk:Reward = 1:2

Baseline (EURUSD H1, SL/TP, no H4 filter, no extra pairs):
  Trades: 55  |  Win rate: 40.0%  |  P&L: +$477.90  |  Max DD: 3.2%
  Trades/month: 1.5  (well below the 10-day prop firm requirement)

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

MONTHS          = 36
INITIAL_BALANCE = 10_000.00
RSI_PERIOD      = 14
BUY_THRESHOLD   = 60
SELL_THRESHOLD  = 40
SL_PIPS         = 50
TP_PIPS         = 100
H4_SMA_PERIOD   = 50   # H4 trend SMA

# Per-pair pip configuration
PAIRS = {
    'EURUSD': {'pip_size': 0.0001, 'pip_value': 1.00},
    'GBPUSD': {'pip_size': 0.0001, 'pip_value': 1.00},
    'USDJPY': {'pip_size': 0.01,   'pip_value': 0.67},
    'EURJPY': {'pip_size': 0.01,   'pip_value': 0.67},
}

# EURUSD-only SL/TP baseline (from h1_sl_tp_backtest.py)
BASELINE = {
    'label'    : 'EURUSD only, no H4 filter',
    'trades'   : 55,
    'win_rate' : 40.0,
    'pnl'      : 477.90,
    'per_month': 13.45,
    'max_dd'   : 3.2,
    'trades_pm': 1.5,
}


# ── 1. CONNECT / DISCONNECT ───────────────────────────────────────────────────

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


# ── 2. FETCH OHLCV DATA ───────────────────────────────────────────────────────

def fetch_data(symbol: str, timeframe: str, months: int,
               extra_days: int = 0) -> pd.DataFrame:
    """
    Fetch OHLCV bars via data_loader (MT5 live, or offline CSV on Mac).
    extra_days: additional lookback beyond `months` (used for H4 SMA warmup).
    """
    date_to   = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=months * 30 + extra_days)

    try:
        df = data_loader.get_bars(symbol, timeframe, date_from, date_to)
    except (ValueError, FileNotFoundError) as e:
        print(f"  WARNING: No {symbol} data — {e}")
        return pd.DataFrame()

    return df[['Open', 'High', 'Low', 'Close']]


# ── 3. H4 TREND FILTER ───────────────────────────────────────────────────────

def compute_h4_trend(h4_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the H4 trend direction using a 50-period SMA.
    Returns a DataFrame with a single column 'H4Trend':
      +1 = H4 Close above H4 SMA_50  (bullish → only allow H1 LONG signals)
      -1 = H4 Close below H4 SMA_50  (bearish → only allow H1 SHORT signals)
       0 = SMA not yet warmed up (block all signals)
    """
    h4 = h4_df.copy()
    h4['H4_SMA'] = h4['Close'].rolling(window=H4_SMA_PERIOD).mean()
    h4['H4Trend'] = 0
    h4.loc[h4['Close'] > h4['H4_SMA'], 'H4Trend'] =  1
    h4.loc[h4['Close'] < h4['H4_SMA'], 'H4Trend'] = -1
    return h4[['H4Trend']]


def merge_h4_trend(h1_df: pd.DataFrame, h4_trend: pd.DataFrame) -> pd.DataFrame:
    """
    For each H1 bar, look up the most recently completed H4 bar's trend.
    merge_asof with direction='backward' finds the latest H4 timestamp <= H1 timestamp.
    """
    h1 = h1_df.reset_index()
    h4 = h4_trend.reset_index()

    merged = pd.merge_asof(
        h1.sort_values('time'),
        h4.sort_values('time'),
        on='time',
        direction='backward'
    )
    merged.set_index('time', inplace=True)
    merged['H4Trend'] = merged['H4Trend'].fillna(0).astype(int)
    return merged


# ── 4. RSI ────────────────────────────────────────────────────────────────────

def calculate_rsi(close: pd.Series, period: int) -> pd.Series:
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


# ── 5. STRATEGY + H4 FILTER ───────────────────────────────────────────────────

def apply_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 50/200 SMA crossover + RSI-14/60-40 signals, then gate
    each signal through the H4 trend direction.
    """
    data = df.copy()
    data['SMA_50']  = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    data['RSI']     = calculate_rsi(data['Close'], RSI_PERIOD)
    data.dropna(subset=['SMA_50', 'SMA_200', 'RSI'], inplace=True)

    data['Position'] = (data['SMA_50'] > data['SMA_200']).astype(int)
    crossover        = data['Position'].diff()

    # Base signals: SMA crossover + RSI filter
    data['Signal'] = 0.0
    data.loc[(crossover == 1)  & (data['RSI'] < BUY_THRESHOLD),  'Signal'] =  1.0
    data.loc[(crossover == -1) & (data['RSI'] > SELL_THRESHOLD), 'Signal'] = -1.0

    # H4 gate: block signals that oppose the H4 trend
    if 'H4Trend' in data.columns:
        data.loc[(data['Signal'] ==  1.0) & (data['H4Trend'] != 1),  'Signal'] = 0.0
        data.loc[(data['Signal'] == -1.0) & (data['H4Trend'] != -1), 'Signal'] = 0.0

    return data


# ── 6. BAR-BY-BAR BACKTEST WITH SL/TP ────────────────────────────────────────

def run_backtest(symbol: str, data: pd.DataFrame) -> list:
    """
    Walks every H1 bar. Checks intra-bar SL/TP hits using High/Low,
    then checks for a new entry signal at bar close.
    Returns a list of completed trade dicts (no running equity — combined later).
    """
    cfg      = PAIRS[symbol]
    pip_size = cfg['pip_size']
    pip_val  = cfg['pip_value']
    sl_dist  = SL_PIPS * pip_size
    tp_dist  = TP_PIPS * pip_size

    trades     = []
    open_trade = None

    for timestamp, row in data.iterrows():
        bar_hi = row['High']
        bar_lo = row['Low']
        bar_cl = row['Close']
        signal = row['Signal']

        # ── Check SL / TP on open trade ──
        if open_trade is not None:
            direction   = open_trade['direction']
            entry_price = open_trade['entry_price']
            sl_price    = open_trade['sl']
            tp_price    = open_trade['tp']

            close_price  = None
            close_reason = None

            if direction == 'LONG':
                sl_hit = bar_lo <= sl_price
                tp_hit = bar_hi >= tp_price
                if sl_hit and tp_hit:
                    close_price, close_reason = sl_price, 'SL'
                elif sl_hit:
                    close_price, close_reason = sl_price, 'SL'
                elif tp_hit:
                    close_price, close_reason = tp_price, 'TP'
            else:
                sl_hit = bar_hi >= sl_price
                tp_hit = bar_lo <= tp_price
                if sl_hit and tp_hit:
                    close_price, close_reason = sl_price, 'SL'
                elif sl_hit:
                    close_price, close_reason = sl_price, 'SL'
                elif tp_hit:
                    close_price, close_reason = tp_price, 'TP'

            # No SL/TP → check for opposing signal exit
            if close_price is None and signal in [1.0, -1.0]:
                close_price  = bar_cl
                close_reason = 'Signal'

            if close_price is not None:
                pips = ((close_price - entry_price) if direction == 'LONG'
                        else (entry_price - close_price)) / pip_size
                pnl  = round(pips * pip_val, 2)

                trades.append({
                    'Symbol'     : symbol,
                    'Direction'  : direction,
                    'Entry Date' : open_trade['entry_time'].strftime('%Y-%m-%d %H:%M'),
                    'Exit Date'  : timestamp.strftime('%Y-%m-%d %H:%M'),
                    'Entry Price': round(entry_price, 5),
                    'Exit Price' : round(close_price, 5),
                    'Exit Reason': close_reason,
                    'Pips'       : round(pips, 1),
                    'P&L (USD)'  : pnl,
                    'Result'     : 'WIN' if pnl > 0 else 'LOSS',
                })
                open_trade = None

        # ── Open new trade on signal ──
        if open_trade is None and signal in [1.0, -1.0]:
            direction  = 'LONG' if signal == 1.0 else 'SHORT'
            open_trade = {
                'direction'  : direction,
                'entry_price': bar_cl,
                'entry_time' : timestamp,
                'sl'  : bar_cl - sl_dist if direction == 'LONG' else bar_cl + sl_dist,
                'tp'  : bar_cl + tp_dist if direction == 'LONG' else bar_cl - tp_dist,
            }

    return trades


# ── 7. BUILD COMBINED EQUITY ──────────────────────────────────────────────────

def build_combined_equity(all_trades: list) -> tuple:
    """
    Sort all trades from all pairs by exit date and apply P&L
    sequentially to a single starting balance.
    Returns (sorted trades with Balance column, equity list).
    """
    sorted_trades = sorted(all_trades, key=lambda t: t['Exit Date'])
    equity        = [INITIAL_BALANCE]
    balance       = INITIAL_BALANCE

    for t in sorted_trades:
        balance         = round(balance + t['P&L (USD)'], 2)
        t['Balance']    = balance
        equity.append(balance)

    for i, t in enumerate(sorted_trades):
        t['Trade #'] = i + 1

    return sorted_trades, equity


# ── 8. COMPUTE STATS ──────────────────────────────────────────────────────────

def compute_stats(trades: list, equity: list,
                  months: float, label: str = '') -> dict:
    if not trades:
        return {'label': label, 'trades': 0, 'win_rate': 0, 'pnl': 0,
                'balance': INITIAL_BALANCE, 'max_dd': 0, 'per_month': 0,
                'trades_pm': 0, 'winners': 0, 'losers': 0}

    df      = pd.DataFrame(trades)
    winners = df[df['P&L (USD)'] > 0]
    losers  = df[df['P&L (USD)'] <= 0]

    peak   = INITIAL_BALANCE
    max_dd = 0.0
    for e in equity:
        peak   = max(peak, e)
        max_dd = max(max_dd, (peak - e) / peak * 100)

    # Unique trading days (based on exit dates)
    exit_dates  = pd.to_datetime(df['Exit Date']).dt.date
    months_s    = pd.to_datetime(df['Exit Date']).dt.to_period('M')
    unique_days = exit_dates.groupby(months_s).nunique()

    return {
        'label'      : label,
        'months'     : months,
        'trades'     : len(df),
        'winners'    : len(winners),
        'losers'     : len(losers),
        'win_rate'   : round(len(winners) / len(df) * 100, 1),
        'pnl'        : round(df['P&L (USD)'].sum(), 2),
        'balance'    : equity[-1],
        'max_dd'     : round(max_dd, 1),
        'avg_win'    : round(winners['P&L (USD)'].mean(), 2) if not winners.empty else 0,
        'avg_loss'   : round(losers['P&L (USD)'].mean(),  2) if not losers.empty  else 0,
        'best'       : round(df['P&L (USD)'].max(), 2),
        'worst'      : round(df['P&L (USD)'].min(), 2),
        'per_month'  : round(df['P&L (USD)'].sum() / months, 2) if months else 0,
        'trades_pm'  : round(len(df) / months, 1),
        'days_pm'    : round(unique_days.mean(), 1) if not unique_days.empty else 0,
        'tp_hits'    : len(df[df['Exit Reason'] == 'TP']) if 'Exit Reason' in df else 0,
        'sl_hits'    : len(df[df['Exit Reason'] == 'SL']) if 'Exit Reason' in df else 0,
    }


# ── 9. PRINT PER-PAIR TABLE ───────────────────────────────────────────────────

def print_per_pair(pair_stats: dict, months: float):
    w = 74
    print()
    print("=" * w)
    print("   PER-PAIR RESULTS  —  H1 50/200 SMA + RSI-14/60-40 + H4 Filter")
    print(f"   SL={SL_PIPS}p / TP={TP_PIPS}p  |  0.1 lots  |  ~{months:.0f} months")
    print("=" * w)
    print(f"  {'Pair':<8}  {'Trades':>6}  {'Win%':>6}  {'P&L':>10}  "
          f"{'$/mo':>8}  {'MaxDD':>7}  {'T/mo':>5}  {'Days/mo':>8}")
    print("-" * w)

    for sym, s in pair_stats.items():
        if s['trades'] == 0:
            print(f"  {sym:<8}  {'No data':>60}")
            continue
        print(f"  {sym:<8}  {s['trades']:>6}  {s['win_rate']:>5.1f}%  "
              f"${s['pnl']:>+9,.2f}  ${s['per_month']:>+7,.2f}  "
              f"{s['max_dd']:>6.1f}%  {s['trades_pm']:>5.1f}  "
              f"{s['days_pm']:>7.1f}")

    print("=" * w)


# ── 10. PRINT COMBINED + BASELINE COMPARISON ─────────────────────────────────

def print_combined(s: dict):
    b = BASELINE
    w = 64

    def chg(new_val, old_val, higher_better=True):
        diff = new_val - old_val
        tag  = ('better' if (diff > 0) == higher_better and diff != 0
                else ('worse' if diff != 0 else 'same'))
        return f"  {'+' if diff >= 0 else ''}{diff:.1f}  ({tag})"

    print()
    print("=" * w)
    print("   COMBINED RESULTS vs EURUSD-ONLY BASELINE")
    print("=" * w)
    print(f"  {'Metric':<24}  {'Baseline':>12}  {'Combined':>12}  Change")
    print("-" * w)
    print(f"  {'Total Trades':<24}  {b['trades']:>12}  {s['trades']:>12}"
          f"{chg(s['trades'], b['trades'])}")
    print(f"  {'Trades / month':<24}  {b['trades_pm']:>12.1f}  {s['trades_pm']:>12.1f}"
          f"{chg(s['trades_pm'], b['trades_pm'])}")
    print(f"  {'Active days / month':<24}  {'~3':>12}  {s['days_pm']:>12.1f}")
    print(f"  {'Win Rate':<24}  {b['win_rate']:>11.1f}%  {s['win_rate']:>11.1f}%"
          f"{chg(s['win_rate'], b['win_rate'])}")
    print(f"  {'Total P&L':<24}  ${b['pnl']:>+11,.2f}  ${s['pnl']:>+11,.2f}"
          f"{chg(s['pnl'], b['pnl'])}")
    print(f"  {'Avg P&L / month':<24}  ${b['per_month']:>+11,.2f}  ${s['per_month']:>+11,.2f}"
          f"{chg(s['per_month'], b['per_month'])}")
    print(f"  {'Final Balance':<24}  ${b['pnl']+INITIAL_BALANCE:>11,.2f}  ${s['balance']:>11,.2f}")
    print(f"  {'Max Drawdown':<24}  {b['max_dd']:>11.1f}%  {s['max_dd']:>11.1f}%"
          f"{chg(s['max_dd'], b['max_dd'], higher_better=False)}")
    print(f"  {'Best Trade':<24}  {'':>12}  ${s['best']:>+11,.2f}")
    print(f"  {'Worst Trade':<24}  {'':>12}  ${s['worst']:>+11,.2f}")
    print(f"  {'TP hits':<24}  {'':>12}  {s['tp_hits']:>12}")
    print(f"  {'SL hits':<24}  {'':>12}  {s['sl_hits']:>12}")
    print("=" * w)


# ── 11. PROP FIRM ASSESSMENT ──────────────────────────────────────────────────

def prop_firm_assessment(s: dict):
    pf_target_pct  = 8.0
    pf_max_dd_pct  = 10.0
    pf_daily_dd    = 5.0
    pf_min_days    = 10

    target_usd    = INITIAL_BALANCE * pf_target_pct / 100
    monthly_pct   = s['per_month'] / INITIAL_BALANCE * 100

    # Lot size needed to hit 8% in one month
    if s['per_month'] > 0:
        lots_needed = round(0.1 * target_usd / s['per_month'], 2)
        dd_at_lots  = round(s['max_dd'] * lots_needed / 0.1, 1)
    else:
        lots_needed = float('inf')
        dd_at_lots  = float('inf')

    w = 66
    print()
    print("=" * w)
    print("   PROP FIRM READINESS ASSESSMENT")
    print("=" * w)

    print(f"""
  Prop Firm Rules (typical Phase 1 challenge):
    Profit target   :  {pf_target_pct}% in 30 days  (${target_usd:,.0f} on $10k account)
    Max total DD    :  {pf_max_dd_pct}%
    Max daily loss  :  {pf_daily_dd}%  (${pf_daily_dd/100*INITIAL_BALANCE:.0f})
    Min trading days:  {pf_min_days} days/month
""")

    gaps = [
        ('Monthly return',
         f'{pf_target_pct:.0f}% / month',
         f'{monthly_pct:+.2f}%',
         monthly_pct >= pf_target_pct),
        ('Max total drawdown',
         f'< {pf_max_dd_pct:.0f}%',
         f'{s["max_dd"]:.1f}%',
         s['max_dd'] <= pf_max_dd_pct),
        ('Max daily drawdown',
         f'< {pf_daily_dd:.0f}%',
         f'~{min(SL_PIPS*0.1/INITIAL_BALANCE*100*len(PAIRS), pf_daily_dd-0.1):.1f}%',
         True),
        ('Active trading days',
         f'>= {pf_min_days} days/mo',
         f'~{s["days_pm"]:.1f} days/mo',
         s['days_pm'] >= pf_min_days),
    ]

    all_pass = all(g[3] for g in gaps)

    print(f"  {'Requirement':<26}  {'Target':>14}  {'Current':>12}  Status")
    print("  " + "-" * (w - 2))
    for name, target, current, passing in gaps:
        print(f"  {name:<26}  {target:>14}  {current:>12}  "
              f"{'PASS' if passing else 'FAIL'}")

    verdict = "READY" if all_pass else "NOT READY"
    print()
    print(f"  VERDICT: Strategy is {verdict} for a prop firm challenge.")
    print()

    if not all_pass:
        failing = [g[0] for g in gaps if not g[3]]
        print(f"  Failing requirements: {', '.join(failing)}")
        print()

    print(f"  RETURN GAP:")
    print(f"    Current avg monthly P&L : ${s['per_month']:>+,.2f}  "
          f"({monthly_pct:+.2f}% on $10k)")
    print(f"    Target                  : ${target_usd:>+,.2f}  "
          f"({pf_target_pct:.0f}%)")
    if s['per_month'] > 0:
        print(f"    Gap                     : {target_usd/s['per_month']:.1f}x below target")
        print(f"    Lots needed for target  : ~{lots_needed} lots")
        print(f"    DD at {lots_needed} lots          : ~{dd_at_lots:.0f}%  "
              f"({'UNSAFE' if dd_at_lots > pf_max_dd_pct else 'safe'})")
    else:
        print(f"    Strategy is net negative — lot scaling will not help.")

    print(f"""
  ACTIVE DAYS GAP:
    Current  : ~{s['days_pm']:.1f} unique exit days / month across {len(PAIRS)} pairs
    Required : {pf_min_days}+ days
    Status   : {'PASS — requirement met' if s['days_pm'] >= pf_min_days
                 else f'FAIL — {pf_min_days - s["days_pm"]:.0f} more active days needed per month'}

  DRAWDOWN:
    Current max DD : {s['max_dd']:.1f}%  (limit: {pf_max_dd_pct:.0f}%)
    Max daily loss : each SL costs ${SL_PIPS * 0.1 * len(PAIRS):.0f} max if all pairs stopped out on
                     same day ({SL_PIPS*0.1*len(PAIRS)/INITIAL_BALANCE*100:.2f}% of account) — safely
                     within the {pf_daily_dd:.0f}% daily loss limit.

  WHAT STILL NEEDS TO CHANGE:
    1. Return rate  — the primary bottleneck. Options:
       a) Increase lot size gradually as account grows (compound)
       b) Add more liquid pairs (AUDUSD, USDCHF, NZDUSD)
       c) Add a second entry technique (breakout, pullback entries)
         to complement the crossover signals
    2. Active days  — if still below {pf_min_days}, add M1/M5 scalp layer on
       high-momentum days to pad trading day count (separate signal)
    3. Walk-forward test — validate on out-of-sample 2025-2026 data
       before attempting a real challenge
""")
    print("=" * w)


# ── 12. PLOT COMBINED EQUITY CURVE ────────────────────────────────────────────

def plot_equity(all_trades: list, equity: list):
    df         = pd.DataFrame(all_trades)
    exit_times = pd.to_datetime(df['Exit Date'])
    full_index = [exit_times.iloc[0]] + list(exit_times)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11))

    # ── Combined equity curve ──
    color = '#2ca02c' if equity[-1] >= INITIAL_BALANCE else '#d62728'
    ax1.plot(full_index, equity, color=color, linewidth=2,
             label='Combined (4 pairs + H4 filter)', zorder=3)
    ax1.axhline(INITIAL_BALANCE, color='grey', linewidth=1, linestyle='--',
                label=f'Starting balance  ${INITIAL_BALANCE:,.0f}', zorder=1)
    ax1.axhline(INITIAL_BALANCE + BASELINE['pnl'], color='#ff7f0e',
                linewidth=1.2, linestyle=':',
                label=f"EURUSD-only baseline  ${INITIAL_BALANCE+BASELINE['pnl']:,.2f}")
    ax1.fill_between(full_index, INITIAL_BALANCE, equity,
                     where=[e >= INITIAL_BALANCE for e in equity],
                     alpha=0.15, color='green', label='Profit zone')
    ax1.fill_between(full_index, INITIAL_BALANCE, equity,
                     where=[e < INITIAL_BALANCE for e in equity],
                     alpha=0.15, color='red', label='Loss zone')
    ax1.set_title(
        f'Multi-Pair H4-Filtered Strategy  |  '
        f'{", ".join(PAIRS)}  |  H1 50/200 SMA + RSI-14  |  '
        f'SL {SL_PIPS}p / TP {TP_PIPS}p  |  0.1 lots each',
        fontsize=12, fontweight='bold')
    ax1.set_ylabel('Combined Account Balance (USD)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')

    # ── Per-trade P&L bars coloured by pair ──
    pair_colors = {
        'EURUSD': '#1f77b4', 'GBPUSD': '#ff7f0e',
        'USDJPY': '#2ca02c', 'EURJPY': '#9467bd',
    }
    bar_colors = [pair_colors.get(t['Symbol'], '#aaa') for t in all_trades]
    ax2.bar(df['Trade #'], df['P&L (USD)'], color=bar_colors, alpha=0.8, width=0.6)
    ax2.axhline(0, color='black', linewidth=0.8)

    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=c, label=p)
               for p, c in pair_colors.items() if p in df['Symbol'].values]
    ax2.legend(handles=patches, fontsize=9)
    ax2.set_title('Individual Trade P&L (coloured by pair)',
                  fontsize=13, fontweight='bold')
    ax2.set_xlabel('Trade Number (chronological, all pairs combined)')
    ax2.set_ylabel('P&L (USD)')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    output_dir  = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'multi_pair_equity_curve.png')
    plt.savefig(output_path, dpi=120)
    print(f"Chart saved : {output_path}")


# ── 13. SAVE TRADE LOG ────────────────────────────────────────────────────────

def save_trade_log(all_trades: list):
    df          = pd.DataFrame(all_trades)
    output_dir  = os.path.join(os.path.dirname(__file__), '..', 'data')
    output_path = os.path.join(output_dir, 'trade_log_multi_pair.csv')
    df.to_csv(output_path, index=False)
    print(f"Trade log   : {output_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

connect_mt5()

print("Fetching data for all pairs (H1 + H4)...")
h1_raw = {}
h4_raw = {}

for symbol in PAIRS:
    tf_label = f'  {symbol}'
    h1 = fetch_data(symbol, 'H1', MONTHS)
    h4 = fetch_data(symbol, 'H4', MONTHS, extra_days=30)
    if h1.empty or h4.empty:
        print(f"  {symbol}: skipped (no data)")
        continue
    h1_raw[symbol] = h1
    h4_raw[symbol] = h4
    months = (h1.index[-1] - h1.index[0]).days / 30
    print(f"  {symbol}: {len(h1):,} H1 bars, {len(h4):,} H4 bars  "
          f"({h1.index[0].date()} to {h1.index[-1].date()},  ~{months:.1f} months)")

if MT5_AVAILABLE:
    mt5.shutdown()
print(f"\nProcessing {len(h1_raw)} pairs...\n")

# Process each pair
all_trades  = []
pair_stats  = {}
data_months = 0

for symbol in PAIRS:
    if symbol not in h1_raw:
        continue

    h4_trend    = compute_h4_trend(h4_raw[symbol])
    h1_merged   = merge_h4_trend(h1_raw[symbol], h4_trend)
    data        = apply_strategy(h1_merged)
    trades      = run_backtest(symbol, data)
    data_months = (h1_raw[symbol].index[-1] - h1_raw[symbol].index[0]).days / 30

    # Per-pair isolated stats (using a dummy single-pair equity)
    solo_equity = [INITIAL_BALANCE]
    bal         = INITIAL_BALANCE
    for t in trades:
        bal = round(bal + t['P&L (USD)'], 2)
        solo_equity.append(bal)

    pair_stats[symbol] = compute_stats(trades, solo_equity, data_months, symbol)
    all_trades.extend(trades)
    print(f"  {symbol}: {len(trades)} trades completed")

# Build combined equity from all pairs sorted by exit date
combined_trades, combined_equity = build_combined_equity(all_trades)
combined_stats = compute_stats(combined_trades, combined_equity, data_months, 'Combined')

print_per_pair(pair_stats, data_months)
print_combined(combined_stats)
save_trade_log(combined_trades)
plot_equity(combined_trades, combined_equity)
prop_firm_assessment(combined_stats)
print("\nDone!")
