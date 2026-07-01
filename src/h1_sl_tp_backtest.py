"""
Forex Bot - H1 Backtest with Stop Loss / Take Profit + Prop Firm Assessment

Part 1: Adds fixed SL (50 pips) and TP (100 pips) to the H1 50/200 SMA
        crossover strategy and reruns the 3-year backtest.

        The backtest engine is rewritten to walk bar-by-bar and check each
        bar's High and Low for SL/TP hits BEFORE checking for a new signal.
        This correctly simulates intra-bar order execution.

Part 2: Assesses whether the strategy is currently prop firm ready and
        produces a clear gap analysis and roadmap.

SL/TP logic:
  LONG  trade: SL = entry - 50 pips,  TP = entry + 100 pips
  SHORT trade: SL = entry + 50 pips,  TP = entry - 100 pips
  Conservative tie-break: if both SL and TP are hit in the same bar, SL wins.

Settings: EURUSD H1, ~3 years, $10,000 balance, 0.1 lots ($1/pip).

H1 baseline (no SL/TP) for comparison:
  Trades: 55  |  Win rate: 43.6%  |  P&L: +$458.50  |  Max DD: 8.5%

Requirements:
  MetaTrader 5 must be OPEN and LOGGED IN before running this script.
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from collections import defaultdict
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
SL_PIPS         = 50
TP_PIPS         = 100
SL_DIST         = SL_PIPS * PIP_SIZE   # 0.0050
TP_DIST         = TP_PIPS * PIP_SIZE   # 0.0100

# H1 no-SL/TP baseline
H1_BASELINE = {
    'label'    : 'H1  No SL/TP',
    'trades'   : 55,
    'winners'  : 24,
    'losers'   : 31,
    'win_rate' : 43.6,
    'pnl'      : 458.50,
    'balance'  : 10_458.50,
    'max_dd'   : 8.5,
    'best'     : 573.40,
    'worst'    : -334.80,
    'per_month': 12.90,
    'avg_win'  : 141.15,
    'avg_loss' : -94.48,
}


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


# ── 2. FETCH H1 OHLCV DATA ────────────────────────────────────────────────────
# We need High and Low (not just Close) to detect intra-bar SL/TP hits.

def fetch_data() -> pd.DataFrame:
    date_to   = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=MONTHS * 30)

    print(f"Requesting H1 data from {date_from.date()} to {date_to.date()} ...")
    df = data_loader.get_bars(SYMBOL, 'H1', date_from, date_to)

    months_actual = (df.index[-1] - df.index[0]).days / 30
    print(f"Received   : {len(df):,} bars  "
          f"({df.index[0].date()} to {df.index[-1].date()},  "
          f"~{months_actual:.1f} months)\n")
    return df[['Close', 'High', 'Low']]


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


# ── 5. BAR-BY-BAR BACKTEST WITH SL/TP ────────────────────────────────────────

def run_backtest(data: pd.DataFrame):
    """
    Walks every bar in order:
      1. If a trade is open, check whether this bar's High or Low triggered
         the SL or TP. If so, close the trade at the SL/TP price.
      2. If no trade is open (or just closed), check for a new entry signal
         at this bar's close price.

    Conservative tie-break: if a bar's range covers both SL and TP, the
    SL is assumed to have been hit first (worst-case assumption).
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

        # ── Check SL / TP on the open trade ──
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
                if sl_hit and tp_hit:       # conservative: SL wins tie
                    close_price, close_reason = sl_price, 'SL'
                elif sl_hit:
                    close_price, close_reason = sl_price, 'SL'
                elif tp_hit:
                    close_price, close_reason = tp_price, 'TP'
            else:  # SHORT
                sl_hit = bar_hi >= sl_price
                tp_hit = bar_lo <= tp_price
                if sl_hit and tp_hit:
                    close_price, close_reason = sl_price, 'SL'
                elif sl_hit:
                    close_price, close_reason = sl_price, 'SL'
                elif tp_hit:
                    close_price, close_reason = tp_price, 'TP'

            # If neither SL nor TP was hit, check for an opposing signal exit
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
                })
                equity.append(balance)
                open_trade = None

        # ── Open a new trade on a signal (if flat) ──
        if open_trade is None and signal in [1.0, -1.0]:
            direction = 'LONG' if signal == 1.0 else 'SHORT'
            open_trade = {
                'direction'  : direction,
                'entry_price': bar_cl,
                'entry_time' : timestamp,
                'sl'         : bar_cl - SL_DIST if direction == 'LONG' else bar_cl + SL_DIST,
                'tp'         : bar_cl + TP_DIST if direction == 'LONG' else bar_cl - TP_DIST,
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

    # Monthly P&L breakdown (group closed trades by year-month)
    df['Exit Month'] = pd.to_datetime(df['Exit Date']).dt.to_period('M')
    monthly = df.groupby('Exit Month')['P&L (USD)'].sum()

    return {
        'months'    : round(months, 1),
        'trades'    : len(df),
        'winners'   : len(winners),
        'losers'    : len(losers),
        'tp_hits'   : len(tp_hits),
        'sl_hits'   : len(sl_hits),
        'sig_exits' : len(df[df['Exit Reason'] == 'Signal']),
        'win_rate'  : round(len(winners) / len(df) * 100, 1) if len(df) else 0,
        'pnl'       : round(df['P&L (USD)'].sum(), 2),
        'balance'   : equity[-1],
        'max_dd'    : round(max_dd, 1),
        'avg_win'   : round(winners['P&L (USD)'].mean(), 2) if not winners.empty else 0,
        'avg_loss'  : round(losers['P&L (USD)'].mean(),  2) if not losers.empty  else 0,
        'best'      : round(df['P&L (USD)'].max(), 2),
        'worst'     : round(df['P&L (USD)'].min(), 2),
        'per_month' : round(df['P&L (USD)'].sum() / months, 2) if months else 0,
        'monthly'   : monthly,
        'trades_pm' : round(len(df) / months, 1),
    }


# ── 7. PRINT RESULTS + SL/TP vs BASELINE COMPARISON ──────────────────────────

def print_results(s: dict):
    b  = H1_BASELINE
    w  = 58

    def chg(new_val, old_val, higher_better=True):
        diff = new_val - old_val
        tag  = ('better' if (diff > 0) == higher_better and diff != 0
                else ('worse' if diff != 0 else 'same'))
        return f"  {'+' if diff >= 0 else ''}{diff:.1f}  ({tag})"

    print()
    print("=" * w)
    print("   PART 1 — H1 BACKTEST: SL/TP vs NO SL/TP")
    print(f"   SL = {SL_PIPS} pips  |  TP = {TP_PIPS} pips  |  "
          f"RR = 1:{TP_PIPS // SL_PIPS}")
    print("=" * w)
    print(f"  {'Metric':<22}  {'No SL/TP':>10}  {'SL/TP':>10}  Change")
    print("-" * w)
    print(f"  {'Total Trades':<22}  {b['trades']:>10}  {s['trades']:>10}"
          f"{chg(s['trades'], b['trades'])}")
    print(f"  {'Win Rate':<22}  {b['win_rate']:>9.1f}%  {s['win_rate']:>9.1f}%"
          f"{chg(s['win_rate'], b['win_rate'])}")
    print(f"  {'Total P&L':<22}  ${b['pnl']:>+9,.2f}  ${s['pnl']:>+9,.2f}"
          f"{chg(s['pnl'], b['pnl'])}")
    print(f"  {'Avg P&L / month':<22}  ${b['per_month']:>+9,.2f}  "
          f"${s['per_month']:>+9,.2f}")
    print(f"  {'Final Balance':<22}  ${b['balance']:>9,.2f}  ${s['balance']:>9,.2f}")
    print(f"  {'Max Drawdown':<22}  {b['max_dd']:>9.1f}%  {s['max_dd']:>9.1f}%"
          f"{chg(s['max_dd'], b['max_dd'], higher_better=False)}")
    print(f"  {'Best Trade':<22}  ${b['best']:>+9,.2f}  ${s['best']:>+9,.2f}")
    print(f"  {'Worst Trade':<22}  ${b['worst']:>+9,.2f}  ${s['worst']:>+9,.2f}")
    print("-" * w)
    print(f"  {'TP hits':<22}  {'':>10}  {s['tp_hits']:>10}")
    print(f"  {'SL hits':<22}  {'':>10}  {s['sl_hits']:>10}")
    print(f"  {'Signal exits':<22}  {'':>10}  {s['sig_exits']:>10}")
    print(f"  {'Avg win':<22}  ${b['avg_win']:>+9,.2f}  ${s['avg_win']:>+9,.2f}")
    print(f"  {'Avg loss':<22}  ${b['avg_loss']:>+9,.2f}  ${s['avg_loss']:>+9,.2f}")
    print("=" * w)

    print()
    print("  Monthly P&L breakdown:")
    print(f"  {'Month':<12}  {'P&L':>10}  {'Running':>10}")
    running = INITIAL_BALANCE
    for period, val in s['monthly'].items():
        running += val
        bar = '#' * int(abs(val) / 20) if val != 0 else ''
        sign = '+' if val >= 0 else ''
        print(f"  {str(period):<12}  ${val:>+8,.2f}  ${running:>9,.2f}  {sign}{bar}")
    print()


# ── 8. PROP FIRM READINESS ASSESSMENT ────────────────────────────────────────

def prop_firm_assessment(s: dict):
    w = 62

    # Typical prop firm rules (using common challenge parameters)
    pf_profit_target_pct = 8.0    # % of account in Phase 1
    pf_max_dd_pct        = 10.0   # max total drawdown %
    pf_daily_dd_pct      = 5.0    # max daily loss %
    pf_min_days          = 10     # min trading days per month
    pf_window_days       = 30     # days to hit target

    # What our strategy delivers (at 0.1 lots on $10k)
    monthly_pnl_usd  = s['per_month']             # $/month at 0.1 lots
    monthly_pnl_pct  = monthly_pnl_usd / INITIAL_BALANCE * 100
    current_dd_pct   = s['max_dd']
    trades_per_month = s['trades_pm']

    # Days with trades (proxy: each trade = 1 trading day minimum)
    active_days_pm   = min(trades_per_month * 2, 22)  # rough estimate

    # Lot size needed to hit profit target in one month
    target_usd        = INITIAL_BALANCE * pf_profit_target_pct / 100
    if monthly_pnl_usd > 0:
        lots_needed   = round(0.1 * target_usd / monthly_pnl_usd, 2)
        dd_at_lots    = round(current_dd_pct * lots_needed / 0.1, 1)
    else:
        lots_needed   = float('inf')
        dd_at_lots    = float('inf')

    print("=" * w)
    print("   PART 2 — PROP FIRM READINESS ASSESSMENT")
    print("=" * w)

    print(f"""
  Typical Prop Firm Challenge Rules:
    Profit target   :  {pf_profit_target_pct}% in {pf_window_days} days
    Max total DD    :  {pf_max_dd_pct}%
    Max daily loss  :  {pf_daily_dd_pct}%
    Min trading days:  {pf_min_days} days/month
""")

    print("  Current Strategy Performance (0.1 lots on $10,000):")
    print(f"    Avg monthly P&L   :  ${monthly_pnl_usd:+,.2f}  "
          f"({monthly_pnl_pct:+.2f}% / month)")
    print(f"    Max drawdown      :  {current_dd_pct:.1f}%")
    print(f"    Trades per month  :  {trades_per_month:.1f}")
    print(f"    Est. active days  :  ~{active_days_pm:.0f} days/month")
    print()

    print("  GAP ANALYSIS:")
    print(f"  {'Requirement':<28}  {'Target':>10}  {'Current':>10}  Status")
    print("  " + "-" * (w - 2))

    gaps = [
        ('Monthly return',    f'{pf_profit_target_pct:.0f}%',
         f'{monthly_pnl_pct:+.2f}%',
         monthly_pnl_pct >= pf_profit_target_pct),
        ('Max total drawdown', f'< {pf_max_dd_pct:.0f}%',
         f'{current_dd_pct:.1f}%',
         current_dd_pct <= pf_max_dd_pct),
        ('Max daily drawdown', f'< {pf_daily_dd_pct:.0f}%',
         f'~{min(SL_PIPS * 0.1 / INITIAL_BALANCE * 100, pf_daily_dd_pct):.1f}%',
         True),
        ('Min trading days/mo', f'>= {pf_min_days} days',
         f'~{active_days_pm:.0f} days',
         active_days_pm >= pf_min_days),
    ]

    for name, target, current, passing in gaps:
        status = 'PASS' if passing else 'FAIL'
        print(f"  {name:<28}  {target:>10}  {current:>10}  {status}")

    print()
    print("  VERDICT: Strategy is NOT prop firm ready at 0.1 lots.\n")

    print("  SPECIFIC GAPS:")
    print(f"""
  1. RETURN RATE — the biggest gap
       Current  : {monthly_pnl_pct:+.2f}% / month  (${monthly_pnl_usd:+,.2f})
       Required : {pf_profit_target_pct:.0f}% in 30 days  (${target_usd:,.2f})
       Gap      : {pf_profit_target_pct - monthly_pnl_pct:.1f}x short of target

     To hit {pf_profit_target_pct}% in one month at current win rate, you need ~{lots_needed} lots.
     But scaling to {lots_needed} lots multiplies the drawdown to ~{dd_at_lots:.0f}%
     — well beyond the {pf_max_dd_pct}% prop firm limit.

  2. TRADE FREQUENCY — ~{trades_per_month:.1f} trades/month on H1
       The 50/200 SMA on H1 generates very few signals (~{trades_per_month:.1f}/month).
       Prop firms require activity across {pf_min_days}+ distinct days.
       At {trades_per_month:.1f} trades/month you likely won't satisfy the
       minimum trading days rule.

  3. DRAWDOWN HEADROOM — currently {current_dd_pct:.1f}%, limit is {pf_max_dd_pct:.0f}%
       The current {current_dd_pct:.1f}% drawdown just fits at 0.1 lots.
       Any lot size increase shrinks this margin rapidly.
""")

    print("=" * w)
    print("   ROADMAP TO PROP FIRM READINESS")
    print("=" * w)
    print(f"""
  Step 1 — Add a second currency pair  (priority: HIGH)
    Run the same strategy on GBPUSD or USDJPY alongside EURUSD.
    More pairs = more signals per month, more active trading days,
    and diversified drawdown. Target: 3-5 pairs.

  Step 2 — Add a second timeframe filter  (priority: HIGH)
    Use H4 or Daily SMA direction as a trend bias filter.
    Only take H1 signals that align with the H4/Daily trend.
    This reduces bad trades rather than just stopping them out.

  Step 3 — Dynamic position sizing  (priority: MEDIUM)
    Scale lot size based on account balance and current drawdown.
    Start at 0.5 lots (5x current) — returns jump to ~{monthly_pnl_usd * 5:+.0f}/month
    (~{monthly_pnl_pct * 5:.1f}%), drawdown stays under {current_dd_pct * 5 / 5 * 1.2:.1f}% if risk
    is managed per-trade.

  Step 4 — Trailing stop  (priority: MEDIUM)
    Replace the fixed TP with a trailing stop that locks in profit
    as the trade moves in your favour. This captures larger trend
    moves (some H1 trends run 200-400 pips) rather than capping at 100.

  Step 5 — Walk-forward validation  (priority: HIGH before going live)
    Backtest on 2023-2024 only, then forward-test on 2025-2026 data
    you haven't optimised on. If results hold, the edge is real.
    If they collapse, the strategy is overfit.

  Realistic timeline:
    Steps 1+2 implemented and backtested         :  2-3 weeks
    Step 3+4 added and validated                 :  1-2 weeks
    Step 5 walk-forward + demo account live test :  4-8 weeks
    Prop firm challenge attempt                  :  after demo is profitable
""")


# ── 9. PLOT EQUITY CURVE ──────────────────────────────────────────────────────

def plot_results(trades: list, equity: list, data: pd.DataFrame):
    df         = pd.DataFrame(trades)
    exit_times = pd.to_datetime(df['Exit Date'])
    full_index = [exit_times.iloc[0]] + list(exit_times)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 11))

    # ── Equity curve ──
    curve_color = '#2ca02c' if equity[-1] >= INITIAL_BALANCE else '#d62728'
    ax1.plot(full_index, equity, color=curve_color, linewidth=2,
             label=f'H1 50/200 SMA  SL={SL_PIPS}p / TP={TP_PIPS}p', zorder=3)
    ax1.axhline(INITIAL_BALANCE, color='grey', linewidth=1, linestyle='--',
                label=f'Starting balance  ${INITIAL_BALANCE:,.0f}', zorder=1)
    ax1.axhline(H1_BASELINE['balance'], color='#ff7f0e', linewidth=1.2,
                linestyle=':', label=f"H1 no SL/TP final  ${H1_BASELINE['balance']:,.2f}")
    ax1.fill_between(full_index, INITIAL_BALANCE, equity,
                     where=[e >= INITIAL_BALANCE for e in equity],
                     alpha=0.15, color='green', label='Profit zone')
    ax1.fill_between(full_index, INITIAL_BALANCE, equity,
                     where=[e < INITIAL_BALANCE for e in equity],
                     alpha=0.15, color='red', label='Loss zone')

    ax1.set_title(
        f'H1 50/200 SMA + RSI-{RSI_PERIOD}/{BUY_THRESHOLD}-{SELL_THRESHOLD}  '
        f'|  SL {SL_PIPS} pips / TP {TP_PIPS} pips  |  EURUSD H1  |  3 Years',
        fontsize=13, fontweight='bold')
    ax1.set_ylabel('Account Balance (USD)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha='right')

    # ── Per-trade P&L bars, coloured by exit reason ──
    colors = {'TP': '#2ca02c', 'SL': '#d62728', 'Signal': '#1f77b4'}
    bar_colors = [colors.get(r, '#999999') for r in df['Exit Reason']]
    bars = ax2.bar(df['Trade #'], df['P&L (USD)'],
                   color=bar_colors, alpha=0.85, width=0.7)
    ax2.axhline(0, color='black', linewidth=0.8)

    # Legend patches
    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color=c, label=l)
                      for l, c in [('TP hit', '#2ca02c'),
                                   ('SL hit', '#d62728'),
                                   ('Signal exit', '#1f77b4')]]
    ax2.legend(handles=legend_patches, fontsize=9)
    ax2.set_title('Individual Trade P&L  (coloured by exit reason)',
                  fontsize=13, fontweight='bold')
    ax2.set_xlabel('Trade Number')
    ax2.set_ylabel('P&L (USD)')
    ax2.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()

    output_dir  = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'h1_sl_tp_equity_curve.png')
    plt.savefig(output_path, dpi=120)
    print(f"Chart saved : {output_path}")


# ── 10. SAVE TRADE LOG ────────────────────────────────────────────────────────

def save_trade_log(trades: list):
    df          = pd.DataFrame(trades)
    output_dir  = os.path.join(os.path.dirname(__file__), '..', 'data')
    output_path = os.path.join(output_dir, 'trade_log_h1_sl_tp.csv')
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

print_results(stats)
save_trade_log(trades)
plot_results(trades, equity, raw)

prop_firm_assessment(stats)
print("Done!")
