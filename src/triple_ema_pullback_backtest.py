"""
Triple EMA Pullback — EURUSD M15  (3-year backtest)
=====================================================
Indicators : EMA 5 / 20 / 50 on M15  |  EMA 50 on H1 (trend filter)

Entry:
  BUY  — H1 close > H1 EMA50  AND  M15 EMA5 > EMA20 > EMA50
          Pullback: bar low within 3 pips of EMA5 or EMA20
          Trigger : next bar close > EMA5
  SELL — mirror

Exit:
  SL  : 15 pips fixed from entry
  TP  : 30 pips fixed from entry (2:1 RR)
  Early: EMA5 crosses back through EMA20 — close at bar close

Session : 12:00–16:00 UTC only
Limits  : max 2 trades/day  |  stop after 2 consecutive losses (resets daily)

Backtest:
  Pair    : EURUSD M15
  Balance : $10,000
  Lots    : 0.10 fixed
  Period  : 3 years from MT5
"""

from __future__ import annotations

import sys
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Settings ─────────────────────────────────────────────────────────────────
SYMBOL        = 'EURUSD'
PIP_SIZE      = 0.0001
PIP_VALUE     = 10.00          # USD per pip per 1.0 lot
LOT_SIZE      = 0.10
STARTING_BAL  = 10_000.00
YEARS         = 3

SL_PIPS       = 15
TP_PIPS       = 30

EMA_FAST      = 5
EMA_MID       = 20
EMA_SLOW      = 50
H1_EMA_N      = 50

SESSION_START = 12             # UTC hour  (inclusive)
SESSION_END   = 16             # UTC hour  (exclusive)

MAX_TRADES_DAY  = 2
MAX_CONSEC_LOSS = 2
TOUCH_PIPS      = 3            # pullback tolerance

BARS_M15 = YEARS * 365 * 96 + 5_000
BARS_H1  = YEARS * 365 * 24 + 2_000

# ── Triple MA SMA Run 1 reference (M15, same settings, SL30/TP60) ─────────────
RUN1 = dict(trades=None, win_rate=45.1, pnl=320.0, ret=3.2,
            max_dd=-1.85, pf=1.51, prof_months=None)

# ── Data ──────────────────────────────────────────────────────────────────────
def fetch_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not mt5.initialize():
        sys.exit(f"MT5 init failed: {mt5.last_error()}")

    m15_raw = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M15, 0, BARS_M15)
    h1_raw  = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1,  0, BARS_H1)
    mt5.shutdown()

    if m15_raw is None or h1_raw is None:
        sys.exit("MT5 data fetch failed")

    def to_df(raw):
        df = pd.DataFrame(raw)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        return df

    m15 = to_df(m15_raw)
    h1  = to_df(h1_raw)

    cutoff = m15.index[-1] - pd.DateOffset(years=YEARS)
    return m15[m15.index >= cutoff].copy(), h1[h1.index >= cutoff].copy()


# ── Indicators ────────────────────────────────────────────────────────────────
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def compute_indicators(m15: pd.DataFrame, h1: pd.DataFrame) -> pd.DataFrame:
    m15 = m15.copy()
    m15['ema5']  = _ema(m15['close'], EMA_FAST)
    m15['ema20'] = _ema(m15['close'], EMA_MID)
    m15['ema50'] = _ema(m15['close'], EMA_SLOW)

    h1 = h1.copy()
    h1['ema50'] = _ema(h1['close'], H1_EMA_N)
    h1_trend = pd.Series(
        np.where(h1['close'] > h1['ema50'], 1, -1),
        index=h1.index
    )

    # Forward-fill H1 trend to M15 timestamps (O(1) lookup per bar)
    combined = h1_trend.reindex(h1_trend.index.union(m15.index))
    m15['h1_trend'] = combined.ffill().reindex(m15.index).fillna(0).astype(int)

    return m15


# ── Backtest ──────────────────────────────────────────────────────────────────
def run_backtest(m15: pd.DataFrame, cross_exit: str | None = 'ema20') -> tuple[list, list]:
    """
    cross_exit : 'ema20' = exit when EMA5 crosses EMA20  (original)
                 'ema50' = exit when EMA5 crosses EMA50  (Test B)
                  None   = no early exit at all           (Test A)
    """
    ema5_a  = m15['ema5'].values
    ema20_a = m15['ema20'].values
    ema50_a = m15['ema50'].values
    h1_a    = m15['h1_trend'].values
    highs   = m15['high'].values
    lows    = m15['low'].values
    closes  = m15['close'].values
    times   = m15.index

    TOUCH = TOUCH_PIPS * PIP_SIZE
    SL    = SL_PIPS    * PIP_SIZE
    TP    = TP_PIPS    * PIP_SIZE

    balance      = STARTING_BAL
    equity_curve = []          # (datetime, balance)
    trades       = []          # closed trade dicts

    # Open trade state
    in_trade    = False
    trade_dir   = ''
    entry_price = sl_price = tp_price = 0.0
    entry_time  = None

    # Pullback state
    pullback_pending = False
    pullback_dir     = ''

    # Daily state
    daily_trades  = 0
    consec_losses = 0
    current_day   = None

    WARMUP = EMA_SLOW + 10     # bars to skip while EMAs stabilise

    for i in range(WARMUP, len(m15)):
        t   = times[i]
        day = t.date()

        if day != current_day:
            current_day   = day
            daily_trades  = 0
            consec_losses = 0

        equity_curve.append((t, float(balance)))

        e5  = ema5_a[i]
        e20 = ema20_a[i]
        e50 = ema50_a[i]
        h1t = h1_a[i]
        lo  = lows[i]
        hi  = highs[i]
        cl  = closes[i]

        # ── Manage open trade ──────────────────────────────────────
        if in_trade:
            hit_sl = (trade_dir == 'BUY'  and lo <= sl_price) or \
                     (trade_dir == 'SELL' and hi >= sl_price)
            hit_tp = (trade_dir == 'BUY'  and hi >= tp_price) or \
                     (trade_dir == 'SELL' and lo <= tp_price)

            # Both SL and TP within same bar → assume SL (conservative)
            if hit_sl and hit_tp:
                hit_tp = False

            if hit_sl:
                pnl = -SL_PIPS * PIP_VALUE * LOT_SIZE
                balance += pnl
                trades.append({'entry_time': entry_time, 'exit_time': t,
                               'direction': trade_dir, 'entry': entry_price,
                               'exit': sl_price, 'pnl': pnl,
                               'reason': 'SL', 'result': 'LOSS'})
                consec_losses += 1
                in_trade = pullback_pending = False

            elif hit_tp:
                pnl = TP_PIPS * PIP_VALUE * LOT_SIZE
                balance += pnl
                trades.append({'entry_time': entry_time, 'exit_time': t,
                               'direction': trade_dir, 'entry': entry_price,
                               'exit': tp_price, 'pnl': pnl,
                               'reason': 'TP', 'result': 'WIN'})
                consec_losses = 0
                in_trade = pullback_pending = False

            else:
                # EMA cross exit checked at bar close (parameterised)
                if cross_exit is not None:
                    cross_ref = e20 if cross_exit == 'ema20' else e50
                    cross = (trade_dir == 'BUY'  and e5 < cross_ref) or \
                            (trade_dir == 'SELL' and e5 > cross_ref)
                    if cross:
                        sign = 1 if trade_dir == 'BUY' else -1
                        pnl  = (cl - entry_price) * sign / PIP_SIZE * PIP_VALUE * LOT_SIZE
                        balance += pnl
                        trades.append({'entry_time': entry_time, 'exit_time': t,
                                       'direction': trade_dir, 'entry': entry_price,
                                       'exit': cl, 'pnl': pnl,
                                       'reason': 'EMA_CROSS',
                                       'result': 'WIN' if pnl > 0 else 'LOSS'})
                        consec_losses = 0 if pnl > 0 else consec_losses + 1
                        in_trade = pullback_pending = False
            continue

        # ── Trade limits ───────────────────────────────────────────
        if daily_trades >= MAX_TRADES_DAY or consec_losses >= MAX_CONSEC_LOSS:
            continue

        # ── Session filter ─────────────────────────────────────────
        if not (SESSION_START <= t.hour < SESSION_END):
            pullback_pending = False
            continue

        # ── EMA alignment (with H1 trend) ──────────────────────────
        bullish = (e5 > e20 > e50) and h1t == 1
        bearish = (e5 < e20 < e50) and h1t == -1

        if not bullish and not bearish:
            pullback_pending = False
            continue

        # Invalidate pending pullback if alignment changed
        if pullback_pending:
            if (pullback_dir == 'BUY'  and not bullish) or \
               (pullback_dir == 'SELL' and not bearish):
                pullback_pending = False

        # ── Entry trigger from previous bar's pullback ─────────────
        entered = False
        if pullback_pending:
            if pullback_dir == 'BUY' and bullish and cl > e5:
                entry_price  = cl
                sl_price     = cl - SL
                tp_price     = cl + TP
                in_trade     = True
                trade_dir    = 'BUY'
                entry_time   = t
                daily_trades += 1
                pullback_pending = False
                entered = True

            elif pullback_dir == 'SELL' and bearish and cl < e5:
                entry_price  = cl
                sl_price     = cl + SL
                tp_price     = cl - TP
                in_trade     = True
                trade_dir    = 'SELL'
                entry_time   = t
                daily_trades += 1
                pullback_pending = False
                entered = True

            else:
                # Entry trigger not met — reset, check for fresh touch below
                pullback_pending = False

        if entered:
            continue

        # ── Detect pullback touch on current bar ───────────────────
        if bullish:
            if abs(lo - e5) <= TOUCH or abs(lo - e20) <= TOUCH:
                pullback_pending = True
                pullback_dir     = 'BUY'
        elif bearish:
            if abs(hi - e5) <= TOUCH or abs(hi - e20) <= TOUCH:
                pullback_pending = True
                pullback_dir     = 'SELL'

    return trades, equity_curve


# ── Statistics ────────────────────────────────────────────────────────────────
def compute_stats(trades: list, equity_curve: list, m15: pd.DataFrame,
                  label: str = 'Original (5x20)') -> dict:
    """Print per-variant stats and return a summary dict for the comparison table."""
    start_dt = m15.index[0].strftime('%Y-%m-%d')
    end_dt   = m15.index[-1].strftime('%Y-%m-%d')

    print()
    print('=' * 62)
    print(f'  Triple EMA Pullback  [{label}]  --  EURUSD M15')
    print(f'  {start_dt}  to  {end_dt}  ({YEARS} years)')
    print('=' * 62)

    if not trades:
        print('  No trades generated.')
        return {}

    df = pd.DataFrame(trades)
    df['exit_time']  = pd.to_datetime(df['exit_time'])
    df['entry_time'] = pd.to_datetime(df['entry_time'])

    total   = len(df)
    wins    = (df['result'] == 'WIN').sum()
    losses  = total - wins
    wr      = wins / total * 100

    total_pnl  = df['pnl'].sum()
    gross_win  = df[df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(df[df['pnl'] <= 0]['pnl'].sum())
    pf         = gross_win / gross_loss if gross_loss > 0 else float('inf')
    ret_pct    = total_pnl / STARTING_BAL * 100

    # Drawdown
    eq_times  = [x[0] for x in equity_curve]
    eq_vals   = [x[1] for x in equity_curve]
    eq_s      = pd.Series(eq_vals, index=eq_times)
    roll_max  = eq_s.cummax()
    dd_s      = (eq_s - roll_max) / roll_max * 100
    max_dd    = dd_s.min()

    # Monthly breakdown
    df['month'] = df['exit_time'].dt.to_period('M')
    monthly_pnl = df.groupby('month')['pnl'].sum()
    prof_months = (monthly_pnl > 0).sum()
    total_months = len(monthly_pnl)
    avg_per_month = total / total_months if total_months else 0

    # Exit reason counts
    reasons = df['reason'].value_counts()

    print(f'  Starting balance   : ${STARTING_BAL:>10,.2f}')
    print(f'  Final balance      : ${STARTING_BAL + total_pnl:>10,.2f}')
    print(f'  Total P&L          : ${total_pnl:>+10,.2f}  ({ret_pct:+.1f}%)')
    print(f'  Max drawdown       : {max_dd:.2f}%')
    print()
    print(f'  Total trades       : {total}')
    print(f'  Wins / Losses      : {wins} / {losses}')
    print(f'  Win rate           : {wr:.1f}%')
    print(f'  Profit factor      : {pf:.2f}')
    print()
    print(f'  Avg trades / month : {avg_per_month:.1f}')
    print(f'  Profitable months  : {prof_months} / {total_months}')
    print()
    print('  Exit breakdown:')
    for reason, cnt in reasons.items():
        w = (df[df['reason'] == reason]['result'] == 'WIN').sum()
        print(f'    {reason:<14} {cnt:>4} trades  ({w} W / {cnt-w} L)')
    print()

    # ── Monthly P&L table ─────────────────────────────────────────
    print('  Monthly P&L:')
    print(f'  {"Month":<10}  {"P&L":>9}  {"Trades":>6}  {"WR":>7}')
    print(f'  {"-"*10}  {"-"*9}  {"-"*6}  {"-"*7}')
    for mth in monthly_pnl.index:
        mth_df   = df[df['month'] == mth]
        m_pnl    = mth_df['pnl'].sum()
        m_trades = len(mth_df)
        m_wr     = (mth_df['result'] == 'WIN').sum() / m_trades * 100 if m_trades else 0
        flag     = ' [+]' if m_pnl > 0 else ' [-]'
        print(f'  {str(mth):<10}  ${m_pnl:>+8.2f}  {m_trades:>6}  {m_wr:>6.1f}%{flag}')
    print()

    # ── Equity curve ──────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f'Triple EMA Pullback [{label}]  --  EURUSD M15 (3 Years)',
                 fontsize=12, y=0.98)

    ax1.plot(eq_times, eq_vals, color='#2196F3', linewidth=0.8, label='Equity')
    ax1.axhline(STARTING_BAL, color='#9E9E9E', linestyle='--',
                linewidth=0.6, label=f'Start ${STARTING_BAL:,.0f}')
    ax1.set_ylabel('Balance (USD)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.25)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    ax2.fill_between(eq_times, dd_s.values, 0, color='#F44336', alpha=0.45,
                     label='Drawdown')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()

    slug = label.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
    out  = Path(__file__).parent.parent / 'data' / f'triple_ema_pullback_{slug}.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Equity curve -> {out}')
    print()

    return {
        'label'       : label,
        'total'       : total,
        'wr'          : wr,
        'pnl'         : total_pnl,
        'ret'         : ret_pct,
        'pf'          : pf,
        'max_dd'      : max_dd,
        'prof_months' : prof_months,
        'tot_months'  : total_months,
        'avg_month'   : avg_per_month,
        'eq_times'    : eq_times,
        'eq_vals'     : eq_vals,
    }


# ── Multi-variant comparison ───────────────────────────────────────────────────
def print_comparison(results: list[dict]) -> None:
    """Print a side-by-side comparison of all run variants + SMA Run 1 reference."""
    sma1 = dict(label='SMA Run 1', total=None, wr=45.1, pnl=320.0, ret=3.2,
                pf=1.51, max_dd=-1.85, prof_months=None, tot_months=37)
    all_r = results + [sma1]

    col_w = 18
    hdr   = f'  {"Metric":<22}' + ''.join(f'  {r["label"]:>{col_w}}' for r in all_r)
    sep   = f'  {"-"*22}' + ('  ' + '-' * col_w) * len(all_r)

    print()
    print('=' * (26 + (col_w + 2) * len(all_r)))
    print('  FULL COMPARISON  --  All variants vs SMA Run 1')
    print('=' * (26 + (col_w + 2) * len(all_r)))
    print(hdr)
    print(sep)

    def _row(metric, key, fmt, higher_better=True):
        vals = [r.get(key) for r in all_r]
        # Find best (excluding None)
        valid = [v for v in vals if v is not None]
        if valid:
            best = max(valid) if higher_better else min(valid)
        else:
            best = None

        cells = []
        for v in vals:
            if v is None:
                cells.append('--')
            else:
                s = fmt.format(v)
                cells.append(s + ' *' if v == best else s)
        row = f'  {metric:<22}' + ''.join(f'  {c:>{col_w}}' for c in cells)
        print(row)

    _row('Total P&L',         'pnl',         '${:+,.2f}',  higher_better=True)
    _row('Return %',          'ret',         '{:+.1f}%',   higher_better=True)
    _row('Win rate',          'wr',          '{:.1f}%',    higher_better=True)
    _row('Profit factor',     'pf',          '{:.2f}',     higher_better=True)
    _row('Max drawdown',      'max_dd',      '{:.2f}%',    higher_better=False)
    _row('Total trades',      'total',       '{:d}',       higher_better=True)
    _row('Profitable months', 'prof_months', '{:d}',       higher_better=True)
    _row('Avg trades/month',  'avg_month',   '{:.1f}',     higher_better=True)
    print()

    # Overall winner by P&L
    best_r = max(results, key=lambda r: r['pnl'])
    print(f'  >> Best EMA variant by P&L : {best_r["label"]}  (${best_r["pnl"]:+,.2f})')
    sma_winner = best_r['label'] if best_r['pnl'] > sma1['pnl'] else sma1['label']
    print(f'  >> Best overall by P&L     : {sma_winner}')
    print()


# ── Overlay equity chart ───────────────────────────────────────────────────────
def plot_overlay(results: list[dict]) -> None:
    colours = ['#2196F3', '#4CAF50', '#FF9800']
    fig, ax = plt.subplots(figsize=(14, 6))
    for r, col in zip(results, colours):
        ax.plot(r['eq_times'], r['eq_vals'], color=col,
                linewidth=0.9, label=r['label'])
    ax.axhline(STARTING_BAL, color='#9E9E9E', linestyle='--',
               linewidth=0.6, label=f'Start ${STARTING_BAL:,.0f}')
    ax.set_title('Triple EMA Pullback -- All Variants  (EURUSD M15, 3 Years)',
                 fontsize=12)
    ax.set_ylabel('Balance (USD)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    out = Path(__file__).parent.parent / 'data' / 'triple_ema_pullback_overlay.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Overlay chart -> {out}')


# ── Main ──────────────────────────────────────────────────────────────────────
VARIANTS = [
    ('Original (5x20 exit)', 'ema20'),
    ('Test A -- No exit',    None),
    ('Test B -- 5x50 exit',  'ema50'),
]

if __name__ == '__main__':
    print('Fetching MT5 data ...')
    m15, h1 = fetch_data()
    print(f'  M15 : {len(m15):,} bars  ({m15.index[0].date()} to {m15.index[-1].date()})')
    print(f'  H1  : {len(h1):,} bars  ({h1.index[0].date()} to {h1.index[-1].date()})')

    print('Computing indicators ...')
    m15 = compute_indicators(m15, h1)

    all_stats = []
    for label, ce in VARIANTS:
        print(f'\nRunning: {label} ...')
        trades, equity_curve = run_backtest(m15, cross_exit=ce)
        print(f'  Trades: {len(trades)}')
        stats = compute_stats(trades, equity_curve, m15, label=label)
        all_stats.append(stats)

    print_comparison(all_stats)
    plot_overlay(all_stats)
    print('Done.')
