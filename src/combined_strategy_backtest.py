"""
Combined Backtest -- SMA Run 1 + EMA Pullback (Test A)  |  EURUSD M15
======================================================================
Both strategies run simultaneously on the same EURUSD M15 data.

SMA Run 1:
  Entry  : SMA50 crosses SMA100 with SMA100 on right side of SMA200
  Filter : H4 50/200 SMA trend  |  session 08:00-22:00 UTC  |  flat<5p
  Exit   : SL 30p  |  TP 60p  |  SMA50/100 cross-exit

EMA Pullback Test A  (no cross-exit):
  Entry  : EMA5>EMA20>EMA50 (bullish) + H1 EMA50 trend + pullback touch
           within 3 pips of EMA5 or EMA20 -> next bar closes beyond EMA5
  Filter : H1 EMA50 trend  |  session 12:00-16:00 UTC
  Exit   : SL 15p  |  TP 30p  (no early exit)

Dynamic sizing (per strategy, per trade):
  Max daily loss  = 5% of current balance
  Risk per trade  = 10% of max daily loss  (= 0.5% of balance)
  Lot size        = risk / (SL_pips x pip_value)
  Hard cap        = 2.00 lots
  Hard floor      = stop all trading if balance drops 10% below start

  At $10,000 start:
    SMA  (30p SL): 0.25% x $10k / (30 x $10) = ~0.08 lots
    EMA  (15p SL): 0.25% x $10k / (15 x $10) = ~0.17 lots

Limits (per strategy):
  Max 2 trades per day
  Stop after 2 consecutive losses (resets daily)
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

# ── Shared settings ──────────────────────────────────────────────────────────
SYMBOL       = 'EURUSD'
PIP_SIZE     = 0.0001
PIP_VALUE    = 10.00      # USD per pip per 1.0 lot
STARTING_BAL = 10_000.00
YEARS        = 3

MAX_TRADES_DAY  = 2
MAX_CONSEC_LOSS = 2

# Dynamic sizing
MAX_DAILY_LOSS_PCT = 0.05   # 5% of balance
RISK_PER_TRADE_PCT = 0.05   # 5% of daily limit = 0.25% of balance
MIN_LOT            = 0.01
MAX_LOT            = 2.00
HARD_FLOOR_PCT     = 0.10   # halt all trading if balance drops this far below start
HARD_FLOOR         = STARTING_BAL * (1 - HARD_FLOOR_PCT)  # $9,000

# ── SMA Run 1 settings ───────────────────────────────────────────────────────
SMA_FAST        = 50
SMA_MID         = 100
SMA_SLOW        = 200
H4_FAST         = 50
H4_SLOW         = 200
SMA_SL_PIPS     = 30
SMA_TP_PIPS     = 60
SMA_SESSION_S   = 8    # UTC hour inclusive
SMA_SESSION_E   = 22   # UTC hour exclusive
SMA_FLAT_PIPS   = 5
SMA_FLAT        = SMA_FLAT_PIPS * PIP_SIZE

# ── EMA Pullback settings ────────────────────────────────────────────────────
EMA_FAST_N      = 5
EMA_MID_N       = 20
EMA_SLOW_N      = 50
H1_EMA_N        = 50
EMA_SL_PIPS     = 15
EMA_TP_PIPS     = 30
EMA_SESSION_S   = 12   # UTC hour inclusive
EMA_SESSION_E   = 16   # UTC hour exclusive
EMA_TOUCH_PIPS  = 3
EMA_TOUCH       = EMA_TOUCH_PIPS * PIP_SIZE

# Reference results for comparison table
FIXED_COMBINED = dict(trades=779, wr=36.1, pnl=1048.20, ret=10.5, pf=1.14, max_dd=2.50)
SOLO_SMA       = dict(trades=51,  wr=45.1, pnl=321.60,  ret=3.2,  pf=1.51, max_dd=1.85)
SOLO_EMA       = dict(trades=706, wr=36.0, pnl=840.00,  ret=8.4,  pf=1.12, max_dd=3.15)


def calc_lot(balance: float, sl_pips: int) -> float:
    """Compute dynamic lot size: risk = 0.5% of balance, risk/pip = SL distance."""
    risk_usd = balance * MAX_DAILY_LOSS_PCT * RISK_PER_TRADE_PCT
    lots     = risk_usd / (sl_pips * PIP_VALUE)
    return round(max(MIN_LOT, min(lots, MAX_LOT)), 2)

BARS_M15 = YEARS * 365 * 96 + 5_000
BARS_H4  = YEARS * 365 *  6 + 1_000
BARS_H1  = YEARS * 365 * 24 + 2_000


# ── Data fetch ────────────────────────────────────────────────────────────────
def fetch_data():
    if not mt5.initialize():
        sys.exit(f"MT5 init failed: {mt5.last_error()}")

    def _get(tf, n, label):
        raw = mt5.copy_rates_from_pos(SYMBOL, tf, 0, n)
        if raw is None:
            sys.exit(f"No {label} data")
        df = pd.DataFrame(raw)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        return df

    m15 = _get(mt5.TIMEFRAME_M15, BARS_M15, 'M15')
    h4  = _get(mt5.TIMEFRAME_H4,  BARS_H4,  'H4')
    h1  = _get(mt5.TIMEFRAME_H1,  BARS_H1,  'H1')
    mt5.shutdown()

    cutoff = m15.index[-1] - pd.DateOffset(years=YEARS)
    return (m15[m15.index >= cutoff].copy(),
            h4[h4.index  >= cutoff].copy(),
            h1[h1.index  >= cutoff].copy())


# ── Indicators ────────────────────────────────────────────────────────────────
def _ffill_to_m15(series: pd.Series, m15_idx) -> np.ndarray:
    """Forward-fill a higher-timeframe series to M15 timestamps."""
    combined = series.reindex(series.index.union(m15_idx))
    return combined.ffill().reindex(m15_idx).fillna(0).values


def compute_indicators(m15: pd.DataFrame, h4: pd.DataFrame, h1: pd.DataFrame):
    m15 = m15.copy()

    # SMA strategy indicators (rolling SMAs)
    m15['sma50']  = m15['close'].rolling(SMA_FAST).mean()
    m15['sma100'] = m15['close'].rolling(SMA_MID).mean()
    m15['sma200'] = m15['close'].rolling(SMA_SLOW).mean()

    # H4 trend for SMA strategy
    h4_fast = h4['close'].rolling(H4_FAST).mean()
    h4_slow = h4['close'].rolling(H4_SLOW).mean()
    h4_trend = pd.Series(
        np.where(h4_fast > h4_slow, 1, np.where(h4_fast < h4_slow, -1, 0)),
        index=h4.index
    )
    m15['h4_trend'] = _ffill_to_m15(h4_trend, m15.index)

    # EMA strategy indicators (exponential MAs)
    m15['ema5']  = m15['close'].ewm(span=EMA_FAST_N,  adjust=False).mean()
    m15['ema20'] = m15['close'].ewm(span=EMA_MID_N,   adjust=False).mean()
    m15['ema50_m15'] = m15['close'].ewm(span=EMA_SLOW_N, adjust=False).mean()

    # H1 trend for EMA strategy
    h1_ema50  = h1['close'].ewm(span=H1_EMA_N, adjust=False).mean()
    h1_trend  = pd.Series(
        np.where(h1['close'] > h1_ema50, 1, -1),
        index=h1.index
    )
    m15['h1_trend'] = _ffill_to_m15(h1_trend, m15.index)

    return m15.dropna(subset=['sma50', 'sma100', 'sma200'])


# ── Combined backtest ─────────────────────────────────────────────────────────
def run_combined(m15: pd.DataFrame) -> tuple[list, list]:
    # Arrays for speed
    sma50_a   = m15['sma50'].values
    sma100_a  = m15['sma100'].values
    sma200_a  = m15['sma200'].values
    h4_a      = m15['h4_trend'].values.astype(int)
    ema5_a    = m15['ema5'].values
    ema20_a   = m15['ema20'].values
    ema50m_a  = m15['ema50_m15'].values
    h1_a      = m15['h1_trend'].values.astype(int)
    highs     = m15['high'].values
    lows      = m15['low'].values
    closes    = m15['close'].values
    times     = m15.index

    SMA_SL = SMA_SL_PIPS * PIP_SIZE
    SMA_TP = SMA_TP_PIPS * PIP_SIZE
    EMA_SL = EMA_SL_PIPS * PIP_SIZE
    EMA_TP = EMA_TP_PIPS * PIP_SIZE

    balance      = STARTING_BAL
    equity_curve = []
    all_trades   = []

    # ── SMA strategy state ────────────────────────────────────────
    sma_in    = False
    sma_dir   = ''
    sma_ep = sma_sl = sma_tp = 0.0
    sma_et    = None
    sma_lot   = 0.0
    sma_day   = None
    sma_dtrd  = 0      # daily_trades
    sma_stop  = False  # daily_stop flag (consec losses)

    # ── EMA strategy state ────────────────────────────────────────
    ema_in    = False
    ema_dir   = ''
    ema_ep = ema_sl = ema_tp = 0.0
    ema_et    = None
    ema_lot   = 0.0
    ema_day   = None
    ema_dtrd  = 0
    ema_cons  = 0      # consecutive losses (reset daily)

    floor_hit = False  # latched once balance breaches hard floor

    ema_pb_pend = False
    ema_pb_dir  = ''

    WARMUP = SMA_SLOW + 5

    for i in range(WARMUP, len(m15)):
        t   = times[i]
        day = t.date()

        # ── Daily resets ──────────────────────────────────────────
        if day != sma_day:
            sma_day  = day
            sma_dtrd = 0
            sma_stop = False

        if day != ema_day:
            ema_day  = day
            ema_dtrd = 0
            ema_cons = 0

        equity_curve.append((t, float(balance)))

        lo = lows[i]; hi = highs[i]; cl = closes[i]
        e5 = ema5_a[i]; e20 = ema20_a[i]; e50m = ema50m_a[i]
        s50 = sma50_a[i]; s100 = sma100_a[i]; s200 = sma200_a[i]

        # ── Manage SMA trade ──────────────────────────────────────
        if sma_in:
            hit_sl = (sma_dir == 'BUY' and lo <= sma_sl) or \
                     (sma_dir == 'SELL' and hi >= sma_sl)
            hit_tp = (sma_dir == 'BUY' and hi >= sma_tp) or \
                     (sma_dir == 'SELL' and lo <= sma_tp)
            cross  = (sma_dir == 'BUY'  and s50 < s100) or \
                     (sma_dir == 'SELL' and s50 > s100)

            if hit_sl and hit_tp:
                hit_tp = False  # conservative

            exit_px = exit_why = None
            if hit_tp:
                exit_px, exit_why = sma_tp, 'TP'
            elif hit_sl:
                exit_px, exit_why = sma_sl, 'SL'
            elif cross:
                exit_px, exit_why = cl, 'CROSS_EXIT'

            if exit_px is not None:
                sign   = 1 if sma_dir == 'BUY' else -1
                pnl    = (exit_px - sma_ep) * sign / PIP_SIZE * PIP_VALUE * sma_lot
                balance += pnl
                won    = pnl > 0
                if not won:
                    sma_stop = True
                if balance < HARD_FLOOR:
                    floor_hit = True
                all_trades.append({
                    'entry_time': sma_et, 'exit_time': t,
                    'strategy': 'SMA', 'direction': sma_dir,
                    'entry': sma_ep, 'exit': exit_px,
                    'pnl': round(pnl, 2), 'reason': exit_why,
                    'result': 'WIN' if won else 'LOSS',
                    'lots': sma_lot, 'confluence': False,
                })
                sma_in = False

        # ── Manage EMA trade ──────────────────────────────────────
        if ema_in:
            hit_sl = (ema_dir == 'BUY' and lo <= ema_sl) or \
                     (ema_dir == 'SELL' and hi >= ema_sl)
            hit_tp = (ema_dir == 'BUY' and hi >= ema_tp) or \
                     (ema_dir == 'SELL' and lo <= ema_tp)

            if hit_sl and hit_tp:
                hit_tp = False

            exit_px = exit_why = None
            if hit_tp:
                exit_px, exit_why = ema_tp, 'TP'
            elif hit_sl:
                exit_px, exit_why = ema_sl, 'SL'

            if exit_px is not None:
                sign   = 1 if ema_dir == 'BUY' else -1
                pnl    = (exit_px - ema_ep) * sign / PIP_SIZE * PIP_VALUE * ema_lot
                balance += pnl
                won    = pnl > 0
                ema_cons = 0 if won else ema_cons + 1
                if balance < HARD_FLOOR:
                    floor_hit = True
                all_trades.append({
                    'entry_time': ema_et, 'exit_time': t,
                    'strategy': 'EMA', 'direction': ema_dir,
                    'entry': ema_ep, 'exit': exit_px,
                    'pnl': round(pnl, 2), 'reason': exit_why,
                    'result': 'WIN' if won else 'LOSS',
                    'lots': ema_lot, 'confluence': False,
                })
                ema_in = False
                ema_pb_pend = False

        # ═══════════════════════════════════════════════════════════
        # Entry signal detection
        # ═══════════════════════════════════════════════════════════

        # ── Hard floor: halt all new entries ─────────────────────
        if floor_hit:
            continue

        # ── SMA signal ────────────────────────────────────────────
        sma_signal = None
        if not sma_in and sma_dtrd < MAX_TRADES_DAY and not sma_stop:
            if SMA_SESSION_S <= t.hour < SMA_SESSION_E:
                spread = max(s50, s100, s200) - min(s50, s100, s200)
                if spread >= SMA_FLAT and i > 0:
                    s50p  = sma50_a[i-1]
                    s100p = sma100_a[i-1]
                    bull  = (s50p <= s100p) and (s50 > s100) and (s100 > s200)
                    bear  = (s50p >= s100p) and (s50 < s100) and (s100 < s200)
                    h4t   = int(h4_a[i])
                    if bull and h4t == 1:
                        sma_signal = 'BUY'
                    elif bear and h4t == -1:
                        sma_signal = 'SELL'

        # ── EMA signal detection ──────────────────────────────────
        ema_signal = None
        if not ema_in and ema_dtrd < MAX_TRADES_DAY and ema_cons < MAX_CONSEC_LOSS:
            if EMA_SESSION_S <= t.hour < EMA_SESSION_E:
                bull_align = (e5 > e20 > e50m) and h1_a[i] == 1
                bear_align = (e5 < e20 < e50m) and h1_a[i] == -1

                if not bull_align and not bear_align:
                    ema_pb_pend = False
                else:
                    # Invalidate pending if alignment flipped
                    if ema_pb_pend:
                        if (ema_pb_dir == 'BUY'  and not bull_align) or \
                           (ema_pb_dir == 'SELL' and not bear_align):
                            ema_pb_pend = False

                    # Entry trigger from previous bar's pullback
                    triggered = False
                    if ema_pb_pend:
                        if ema_pb_dir == 'BUY' and bull_align and cl > e5:
                            ema_signal = 'BUY'
                            triggered  = True
                        elif ema_pb_dir == 'SELL' and bear_align and cl < e5:
                            ema_signal = 'SELL'
                            triggered  = True
                        ema_pb_pend = False  # always reset after check

                    # Detect fresh pullback touch (only if not just triggered)
                    if not triggered:
                        if bull_align and (abs(lows[i] - e5) <= EMA_TOUCH or
                                           abs(lows[i] - e20) <= EMA_TOUCH):
                            ema_pb_pend = True
                            ema_pb_dir  = 'BUY'
                        elif bear_align and (abs(highs[i] - e5) <= EMA_TOUCH or
                                             abs(highs[i] - e20) <= EMA_TOUCH):
                            ema_pb_pend = True
                            ema_pb_dir  = 'SELL'
            else:
                ema_pb_pend = False   # outside session, clear pending

        # ── Enter trades ──────────────────────────────────────────
        # Detect confluence: both fire same bar, same direction
        confluence = (sma_signal is not None and
                      ema_signal is not None and
                      sma_signal == ema_signal)

        if sma_signal and not sma_in:
            sma_lot  = calc_lot(balance, SMA_SL_PIPS)
            sma_ep   = cl
            sma_sl   = cl - SMA_SL if sma_signal == 'BUY' else cl + SMA_SL
            sma_tp   = cl + SMA_TP if sma_signal == 'BUY' else cl - SMA_TP
            sma_in   = True
            sma_dir  = sma_signal
            sma_et   = t
            sma_dtrd += 1

        if ema_signal and not ema_in:
            ema_lot  = calc_lot(balance, EMA_SL_PIPS)
            ema_ep   = cl
            ema_sl   = cl - EMA_SL if ema_signal == 'BUY' else cl + EMA_SL
            ema_tp   = cl + EMA_TP if ema_signal == 'BUY' else cl - EMA_TP
            ema_in   = True
            ema_dir  = ema_signal
            ema_et   = t
            ema_dtrd += 1

        # Tag the just-opened trades as confluence if both fired
        if confluence:
            # Mark both open trades (last appended won't help — tag at close time instead)
            # Use a running flag so we can tag when they close
            pass  # handled by recording confluence flag in the entry dict below

    # (confluence tagging is done post-hoc — see stats)
    return all_trades, equity_curve


# ── Stats ─────────────────────────────────────────────────────────────────────
def _trade_stats(subset: pd.DataFrame) -> dict:
    if subset.empty:
        return dict(n=0, wr=0.0, pnl=0.0, pf=0.0)
    wins  = (subset['pnl'] > 0).sum()
    n     = len(subset)
    gw    = subset[subset['pnl'] > 0]['pnl'].sum()
    gl    = abs(subset[subset['pnl'] <= 0]['pnl'].sum())
    return dict(n=n, wr=wins/n*100, pnl=subset['pnl'].sum(),
                pf=gw/gl if gl else float('inf'))


def compute_stats(all_trades: list, equity_curve: list, m15: pd.DataFrame) -> dict:
    if not all_trades:
        return {}

    df = pd.DataFrame(all_trades)
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time']  = pd.to_datetime(df['exit_time'])

    # Detect confluence: both SMA and EMA entered on the same bar (same entry_time + direction)
    sma_df  = df[df['strategy'] == 'SMA'][['entry_time', 'direction']].copy()
    ema_df  = df[df['strategy'] == 'EMA'][['entry_time', 'direction']].copy()
    merged  = sma_df.merge(ema_df, on=['entry_time', 'direction'], how='inner')
    conf_keys = set(zip(merged['entry_time'], merged['direction']))

    def is_conf(row):
        return (row['entry_time'], row['direction']) in conf_keys
    df['confluence'] = df.apply(is_conf, axis=1)

    # Subsets
    sma_s   = df[df['strategy'] == 'SMA']
    ema_s   = df[df['strategy'] == 'EMA']
    conf_s  = df[df['confluence']]        # trades that were part of a confluence entry
    solo_s  = df[~df['confluence']]

    # Equity / drawdown
    eq_times = [x[0] for x in equity_curve]
    eq_vals  = [x[1] for x in equity_curve]
    eq_s     = pd.Series(eq_vals, index=eq_times)
    roll_max = eq_s.cummax()
    dd_s     = (eq_s - roll_max) / roll_max * 100
    max_dd   = dd_s.min()

    # Monthly
    df['month'] = df['exit_time'].dt.to_period('M')
    monthly_pnl = df.groupby('month')['pnl'].sum()
    prof_months = (monthly_pnl > 0).sum()
    total_months = len(monthly_pnl)

    n_total = len(df)
    wins    = (df['pnl'] > 0).sum()
    wr      = wins / n_total * 100
    total_pnl = df['pnl'].sum()
    gw      = df[df['pnl'] > 0]['pnl'].sum()
    gl      = abs(df[df['pnl'] <= 0]['pnl'].sum())
    pf      = gw / gl if gl else float('inf')
    avg_month = n_total / total_months if total_months else 0

    return dict(
        df=df, sma_s=sma_s, ema_s=ema_s, conf_s=conf_s, solo_s=solo_s,
        n=n_total, wr=wr, pnl=total_pnl, pf=pf,
        max_dd=max_dd, avg_month=avg_month,
        prof_months=prof_months, total_months=total_months,
        monthly_pnl=monthly_pnl, eq_times=eq_times, eq_vals=eq_vals, dd_s=dd_s,
    )


# ── Report ────────────────────────────────────────────────────────────────────
def print_report(s: dict, m15: pd.DataFrame) -> None:
    start_dt = m15.index[0].strftime('%Y-%m-%d')
    end_dt   = m15.index[-1].strftime('%Y-%m-%d')

    sma_st = _trade_stats(s['sma_s'])
    ema_st = _trade_stats(s['ema_s'])

    ret_pct = s['pnl'] / STARTING_BAL * 100

    # Lot size info
    df_all = s['df']
    sma_avg_lot = s['sma_s']['lots'].mean() if not s['sma_s'].empty else 0.0
    ema_avg_lot = s['ema_s']['lots'].mean() if not s['ema_s'].empty else 0.0
    sma_max_lot = s['sma_s']['lots'].max()  if not s['sma_s'].empty else 0.0
    ema_max_lot = s['ema_s']['lots'].max()  if not s['ema_s'].empty else 0.0
    sma_init_lot = calc_lot(STARTING_BAL, SMA_SL_PIPS)
    ema_init_lot = calc_lot(STARTING_BAL, EMA_SL_PIPS)

    floor_triggered = (STARTING_BAL + s['pnl']) < HARD_FLOOR

    print()
    print('=' * 68)
    print('  COMBINED SYSTEM (DYNAMIC SIZING)  --  SMA Run 1 + EMA Pullback')
    print(f'  EURUSD M15  |  {start_dt}  to  {end_dt}  ({YEARS} years)')
    print('=' * 68)
    print(f'  Starting balance   : ${STARTING_BAL:>10,.2f}')
    print(f'  Final balance      : ${STARTING_BAL + s["pnl"]:>10,.2f}')
    print(f'  Total P&L          : ${s["pnl"]:>+10,.2f}  ({ret_pct:+.1f}%)')
    print(f'  Max drawdown       : {s["max_dd"]:.2f}%')
    print(f'  Hard floor         : ${HARD_FLOOR:,.2f}  {"[TRIGGERED]" if floor_triggered else "[not hit]"}')
    print()
    print(f'  Total trades       : {s["n"]}  (SMA {sma_st["n"]} + EMA {ema_st["n"]})')
    print(f'  Win rate           : {s["wr"]:.1f}%')
    print(f'  Profit factor      : {s["pf"]:.2f}')
    print(f'  Avg trades / month : {s["avg_month"]:.1f}')
    print(f'  Profitable months  : {s["prof_months"]} / {s["total_months"]}')
    print()

    # ── Dynamic sizing summary ────────────────────────────────────
    print('  DYNAMIC LOT SIZING SUMMARY')
    risk_pct = MAX_DAILY_LOSS_PCT * RISK_PER_TRADE_PCT * 100
    print(f'  Risk per trade     : {risk_pct:.2f}% of balance  ({RISK_PER_TRADE_PCT*100:.0f}% of daily limit)')
    print(f'  {"Strategy":<14}  {"SL":>4}  {"Init lot":>9}  {"Avg lot":>9}  {"Max lot":>9}  {"P&L":>10}')
    print(f'  {"-"*14}  {"-"*4}  {"-"*9}  {"-"*9}  {"-"*9}  {"-"*10}')
    print(f'  {"SMA Run 1":<14}  {SMA_SL_PIPS:>3}p  {sma_init_lot:>9.2f}  {sma_avg_lot:>9.2f}  {sma_max_lot:>9.2f}  ${sma_st["pnl"]:>+9,.2f}')
    print(f'  {"EMA Pullback":<14}  {EMA_SL_PIPS:>3}p  {ema_init_lot:>9.2f}  {ema_avg_lot:>9.2f}  {ema_max_lot:>9.2f}  ${ema_st["pnl"]:>+9,.2f}')
    print()

    # ── Per-strategy stats ────────────────────────────────────────
    print('  PER-STRATEGY BREAKDOWN')
    print(f'  {"Strategy":<14}  {"Trades":>6}  {"WR":>7}  {"P&L":>10}  {"PF":>6}')
    print(f'  {"-"*14}  {"-"*6}  {"-"*7}  {"-"*10}  {"-"*6}')
    print(f'  {"SMA Run 1":<14}  {sma_st["n"]:>6}  {sma_st["wr"]:>6.1f}%  ${sma_st["pnl"]:>+9,.2f}  {sma_st["pf"]:>6.2f}')
    print(f'  {"EMA Pullback":<14}  {ema_st["n"]:>6}  {ema_st["wr"]:>6.1f}%  ${ema_st["pnl"]:>+9,.2f}  {ema_st["pf"]:>6.2f}')
    print()

    # ── Monthly P&L ───────────────────────────────────────────────
    print('  MONTHLY P&L')
    print(f'  {"Month":<10}  {"Combined":>10}  {"SMA":>9}  {"EMA":>9}  {"Trd":>4}  {"Avg lot":>8}')
    print(f'  {"-"*10}  {"-"*10}  {"-"*9}  {"-"*9}  {"-"*4}  {"-"*8}')
    df  = s['df']
    df['month'] = df['exit_time'].dt.to_period('M')
    for mth in s['monthly_pnl'].index:
        mth_df   = df[df['month'] == mth]
        m_pnl    = mth_df['pnl'].sum()
        m_sma    = mth_df[mth_df['strategy']=='SMA']['pnl'].sum()
        m_ema    = mth_df[mth_df['strategy']=='EMA']['pnl'].sum()
        m_n      = len(mth_df)
        m_avglot = mth_df['lots'].mean()
        flag     = '[+]' if m_pnl > 0 else '[-]'
        print(f'  {str(mth):<10}  ${m_pnl:>+9,.2f}  ${m_sma:>+8,.2f}  ${m_ema:>+8,.2f}  {m_n:>4}  {m_avglot:>8.2f}  {flag}')
    print()

    # ── Comparison: dynamic vs fixed ─────────────────────────────
    print('=' * 68)
    print('  COMPARISON  --  Dynamic Sizing vs Fixed 0.10 Lots')
    print('=' * 68)
    fc = FIXED_COMBINED
    hdr = f'  {"Metric":<22}  {"Dynamic":>14}  {"Fixed 0.10L":>14}'
    sep = f'  {"-"*22}  {"-"*14}  {"-"*14}'
    print(hdr); print(sep)

    def _row2(label, dval, fval, fmt, higher_better=True):
        best = (max if higher_better else min)(dval, fval)
        def _c(v):
            return (fmt.format(v) + ' *') if v == best else fmt.format(v)
        print(f'  {label:<22}  {_c(dval):>14}  {_c(fval):>14}')

    _row2('Total P&L',         s['pnl'],        fc['pnl'],    '${:+,.2f}', True)
    _row2('Return %',          ret_pct,          fc['ret'],    '{:+.1f}%',  True)
    _row2('Win rate',          s['wr'],           fc['wr'],     '{:.1f}%',   True)
    _row2('Profit factor',     s['pf'],           fc['pf'],     '{:.2f}',    True)
    _row2('Max drawdown',      abs(s['max_dd']),  fc['max_dd'], '{:.2f}%',   False)
    _row2('Total trades',      s['n'],            fc['trades'], '{:d}',      False)
    _row2('Profitable months', s['prof_months'],  19,           '{:d}',      True)
    print(f'  {"Avg SMA lot":<22}  {sma_avg_lot:>14.2f}  {"0.10":>14}')
    print(f'  {"Avg EMA lot":<22}  {ema_avg_lot:>14.2f}  {"0.10":>14}')
    print(f'  {"Hard floor hit":<22}  {"YES" if floor_triggered else "NO":>14}  {"N/A":>14}')
    print()


# ── Equity chart ──────────────────────────────────────────────────────────────
def plot_equity(s: dict) -> None:
    eq_times = s['eq_times']
    eq_vals  = s['eq_vals']
    dd_vals  = s['dd_s'].values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(
        'Combined System (Dynamic Sizing): SMA Run 1 + EMA Pullback  --  EURUSD M15 (3 Years)',
        fontsize=11, y=0.99
    )

    ax1.plot(eq_times, eq_vals, color='#2196F3', linewidth=0.9, label='Equity (dynamic lots)')
    ax1.axhline(STARTING_BAL, color='#9E9E9E', linestyle='--',
                linewidth=0.6, label=f'Start ${STARTING_BAL:,.0f}')
    ax1.axhline(HARD_FLOOR, color='#F44336', linestyle=':',
                linewidth=0.8, label=f'Hard floor ${HARD_FLOOR:,.0f}')

    # Mark SMA wins/losses
    sma_df = s['sma_s'].copy()
    sma_df['exit_time'] = pd.to_datetime(sma_df['exit_time'])
    # Mark EMA wins/losses
    ema_df = s['ema_s'].copy()
    ema_df['exit_time'] = pd.to_datetime(ema_df['exit_time'])

    ax1.set_ylabel('Balance (USD)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.25)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

    ax2.fill_between(eq_times, dd_vals, 0, color='#F44336', alpha=0.45,
                     label='Drawdown')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()

    out = Path(__file__).parent.parent / 'data' / 'combined_strategy_equity.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Chart -> {out}')


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Fetching MT5 data ...')
    m15, h4, h1 = fetch_data()
    print(f'  M15 : {len(m15):,} bars  ({m15.index[0].date()} to {m15.index[-1].date()})')
    print(f'  H4  : {len(h4):,} bars')
    print(f'  H1  : {len(h1):,} bars')

    print('Computing indicators ...')
    m15 = compute_indicators(m15, h4, h1)

    print('Running combined backtest ...')
    all_trades, equity_curve = run_combined(m15)
    print(f'  Total trade records: {len(all_trades)} '
          f'(SMA {sum(1 for t in all_trades if t["strategy"]=="SMA")} '
          f'+ EMA {sum(1 for t in all_trades if t["strategy"]=="EMA")})')

    stats = compute_stats(all_trades, equity_curve, m15)
    print_report(stats, m15)
    plot_equity(stats)
    print('Done.')
