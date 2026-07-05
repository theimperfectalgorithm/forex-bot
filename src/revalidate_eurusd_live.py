"""
Forex Bot - Re-validation of the LIVE EURUSD strategy (sma_ema_combined)
=========================================================================
The EURUSD dual strategy is trading live money right now, but was
validated with older, spread-blind backtests. Phases 1-5 (2026-07-04/05)
showed indicator systems on EURUSD have no edge net of spread -- this
script re-tests the exact live logic with the current methodology.

Faithful replication of strategies/sma_ema_combined.py:
  - Two INDEPENDENT books that can be open simultaneously:
      SMA Run 1   : M15 SMA50/100 cross, flat filter (|50-100| >= 5p),
                    price vs SMA200 gate, H1 EMA50 trend filter,
                    SL 30p / TP 60p, ADVERSE-CROSS EXIT.
      EMA Pullback: M15 EMA5>20>50 alignment + H1 trend, two-step
                    (touch EMA20 within 3p / beyond, then confirmation
                    close back on the trend side), SL 15p / TP 30p.
  - Session: signals + cross-exits evaluated ONLY on bars opening
    12:00-15:45 UTC (London/NY overlap) -- as live.
  - Per book, per day: max 2 entries; book pauses for the day after 2
    consecutive losses; pullback-pending state resets at midnight
    (daily_state is rebuilt) -- all as live.
  - Friday close 20:00 UTC (orchestrator). Positions may run overnight
    Mon-Thu, as live.
  - Indicators seeded exactly as live: 250-bar M15 window (SMA/EMA),
    60-bar H1 window (EMA50); H1 trend taken from the last CLOSED H1 bar.
  - Risk 0.25% per trade (agent_risk EURUSD_RISK_PCT), spread 1.0 pip.

Requirements: MetaTrader 5 OPEN and LOGGED IN.
Output: data/revalidate_eurusd_results.csv + console report
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy_matrix_backtest import (
    MONTHS, IS_MONTHS, windowed_ema, metrics, REPO_ROOT,
)
from phase3_session_structure_search import fetch_tf

PIP = 0.0001
SPREAD = 1.0
RISK_PCT = 0.0025
BAL0 = 100_000.0

SMA_SL, SMA_TP = 30.0, 60.0
EMA_SL, EMA_TP = 15.0, 30.0
FLAT_PIPS, TOUCH_PIPS = 5.0, 3.0
MAX_DAILY, MAX_CONSEC = 2, 2

OUT_CSV = REPO_ROOT / 'data' / 'revalidate_eurusd_results.csv'


def rolling_sma(a, n):
    return pd.Series(a).rolling(n).mean().to_numpy()


def simulate(m15: pd.DataFrame, h1: pd.DataFrame):
    closes = m15['Close'].to_numpy()
    opens  = m15['Open'].to_numpy()
    highs  = m15['High'].to_numpy()
    lows   = m15['Low'].to_numpy()
    times  = m15.index
    hours  = times.hour
    wday   = times.weekday
    dates  = times.date
    n = len(m15)

    sma50  = rolling_sma(closes, 50)
    sma100 = rolling_sma(closes, 100)
    sma200 = rolling_sma(closes, 200)
    ema5   = windowed_ema(closes, 5, 250)
    ema20  = windowed_ema(closes, 20, 250)
    ema50m = windowed_ema(closes, 50, 250)

    # H1 EMA50 trend mapped to each M15 bar (last CLOSED H1 bar)
    h1c = h1['Close'].to_numpy()
    h1ema = windowed_ema(h1c, 50, 60)
    h1trend = np.where(h1c > h1ema, 1, np.where(h1c < h1ema, -1, 0))
    h1_close_t = (h1.index + pd.Timedelta(hours=1)).values
    m15_close_t = (times + pd.Timedelta(minutes=15)).values
    idx = np.searchsorted(h1_close_t, m15_close_t, side='right') - 1
    trend_at = np.zeros(n, dtype=int)
    ok = idx >= 0
    trend_at[ok] = h1trend[idx[ok]]

    balance = BAL0
    books = {'SMA': None, 'EMA': None}          # open position per book
    day = {'cur': None, 'sma_n': 0, 'ema_n': 0, 'sma_cl': 0, 'ema_cl': 0,
           'pb_pend': False, 'pb_dir': ''}
    trades = []
    equity = np.empty(n)

    def close_pos(book, i, price, reason):
        nonlocal balance
        p = books[book]
        pips = ((price - p['entry']) if p['dir'] == 'BUY'
                else (p['entry'] - price)) / PIP - SPREAD
        pnl = pips * p['usd_per_pip']
        balance += pnl
        trades.append({'entry_time': p['time'], 'exit_time': times[i],
                       'book': book, 'dir': p['dir'], 'reason': reason,
                       'pips': round(pips, 1), 'pnl': round(pnl, 2),
                       'balance': round(balance, 2)})
        # consecutive-loss tracking (per book, per day)
        key = 'sma_cl' if book == 'SMA' else 'ema_cl'
        day[key] = day[key] + 1 if pnl < 0 else 0
        books[book] = None

    for i in range(n):
        if dates[i] != day['cur']:
            day.update({'cur': dates[i], 'sma_n': 0, 'ema_n': 0,
                        'sma_cl': 0, 'ema_cl': 0,
                        'pb_pend': False, 'pb_dir': ''})

        # ---- manage open positions (SL/TP any bar; Friday close)
        for book in ('SMA', 'EMA'):
            p = books[book]
            if p is None:
                continue
            if p['dir'] == 'BUY':
                if lows[i] <= p['sl']:
                    close_pos(book, i, p['sl'], 'SL'); continue
                if highs[i] >= p['tp']:
                    close_pos(book, i, p['tp'], 'TP'); continue
            else:
                if highs[i] >= p['sl']:
                    close_pos(book, i, p['sl'], 'SL'); continue
                if lows[i] <= p['tp']:
                    close_pos(book, i, p['tp'], 'TP'); continue
            if wday[i] == 4 and hours[i] >= 20:
                close_pos(book, i, opens[i], 'FridayClose')

        in_window = 12 <= hours[i] < 16
        if in_window and i >= 250 and not np.isnan(sma200[i]) \
                and trend_at[i] != 0:
            tr = trend_at[i]
            c = closes[i]
            bull_x = sma50[i - 1] <= sma100[i - 1] and sma50[i] > sma100[i]
            bear_x = sma50[i - 1] >= sma100[i - 1] and sma50[i] < sma100[i]

            # ---- SMA adverse cross-exit (window-only, as live)
            p = books['SMA']
            if p is not None and ((p['dir'] == 'BUY' and bear_x)
                                  or (p['dir'] == 'SELL' and bull_x)):
                close_pos('SMA', i, c, 'CrossExit')

            # ---- SMA entries
            if (books['SMA'] is None and day['sma_n'] < MAX_DAILY
                    and day['sma_cl'] < MAX_CONSEC):
                flat = abs(sma50[i] - sma100[i]) / PIP < FLAT_PIPS
                sig = None
                if not flat and bull_x and tr == 1 and c > sma200[i]:
                    sig = 'BUY'
                elif not flat and bear_x and tr == -1 and c < sma200[i]:
                    sig = 'SELL'
                if sig:
                    risk = balance * RISK_PCT
                    books['SMA'] = {
                        'dir': sig, 'entry': c, 'time': times[i],
                        'sl': c - SMA_SL * PIP if sig == 'BUY' else c + SMA_SL * PIP,
                        'tp': c + SMA_TP * PIP if sig == 'BUY' else c - SMA_TP * PIP,
                        'usd_per_pip': risk / SMA_SL}
                    day['sma_n'] += 1

            # ---- EMA pullback state machine
            if (books['EMA'] is None and day['ema_n'] < MAX_DAILY
                    and day['ema_cl'] < MAX_CONSEC):
                e5, e20, e50 = ema5[i], ema20[i], ema50m[i]
                bull_al = e5 > e20 > e50
                bear_al = e5 < e20 < e50
                touch = TOUCH_PIPS * PIP
                if not day['pb_pend']:
                    if bull_al and tr == 1 and (abs(c - e20) <= touch or c < e20):
                        day['pb_pend'], day['pb_dir'] = True, 'BUY'
                    elif bear_al and tr == -1 and (abs(c - e20) <= touch or c > e20):
                        day['pb_pend'], day['pb_dir'] = True, 'SELL'
                else:
                    sig = None
                    if day['pb_dir'] == 'BUY':
                        if c > e20 and bull_al and tr == 1:
                            sig = 'BUY'
                            day.update({'pb_pend': False, 'pb_dir': ''})
                        elif not bull_al or tr != 1:
                            day.update({'pb_pend': False, 'pb_dir': ''})
                    elif day['pb_dir'] == 'SELL':
                        if c < e20 and bear_al and tr == -1:
                            sig = 'SELL'
                            day.update({'pb_pend': False, 'pb_dir': ''})
                        elif not bear_al or tr != -1:
                            day.update({'pb_pend': False, 'pb_dir': ''})
                    if sig:
                        risk = balance * RISK_PCT
                        books['EMA'] = {
                            'dir': sig, 'entry': c, 'time': times[i],
                            'sl': c - EMA_SL * PIP if sig == 'BUY' else c + EMA_SL * PIP,
                            'tp': c + EMA_TP * PIP if sig == 'BUY' else c - EMA_TP * PIP,
                            'usd_per_pip': risk / EMA_SL}
                        day['ema_n'] += 1

        # ---- mark-to-market
        fl = 0.0
        for p in books.values():
            if p is not None:
                fp = ((closes[i] - p['entry']) if p['dir'] == 'BUY'
                      else (p['entry'] - closes[i])) / PIP
                fl += fp * p['usd_per_pip']
        equity[i] = balance + fl

    tdf = pd.DataFrame(trades)
    eq = pd.Series(equity, index=times + pd.Timedelta(minutes=15))
    return tdf, eq


def report(label, m):
    if m.get('trades', 0) == 0:
        print(f'  {label}: no trades')
        return
    print(f"  {label}: n={m['trades']}  WR={m['win_rate']}%  PF={m['pf']}  "
          f"P&L=${m['pnl']:+,.2f}  maxDD={m['max_dd_pct']}%  "
          f"prof.months={m['prof_months_pct']}%  worst day={m['worst_day_pct']}%")


def main():
    now = pd.Timestamp.now(tz='UTC')
    oos_start = now - pd.DateOffset(months=MONTHS - IS_MONTHS)
    is_start  = now - pd.DateOffset(months=MONTHS)

    print('Fetching EURUSD M15/H1 ...')
    m15 = fetch_tf('EURUSD', 'M15')
    h1  = fetch_tf('EURUSD', 'H1')
    tdf, eq = simulate(m15, h1)

    print(f'\nTotal simulated trades: {len(tdf)} '
          f'({m15.index[0].date()} -> {m15.index[-1].date()}, '
          f'risk 0.25%/trade, spread {SPREAD}p)\n')

    print('=== FULL PERIOD (36 months) ===')
    report('ALL  ', metrics(tdf, eq, is_start, now))
    for book in ('SMA', 'EMA'):
        sub = tdf[tdf['book'] == book]
        report(book, metrics(sub, eq, is_start, now))

    print('\n=== IN-SAMPLE (first 24 months) ===')
    report('ALL  ', metrics(tdf, eq, is_start, oos_start))
    for book in ('SMA', 'EMA'):
        report(book, metrics(tdf[tdf['book'] == book], eq, is_start, oos_start))

    print('\n=== OUT-OF-SAMPLE (last 12 months) ===')
    report('ALL  ', metrics(tdf, eq, oos_start, now))
    for book in ('SMA', 'EMA'):
        report(book, metrics(tdf[tdf['book'] == book], eq, oos_start, now))

    exits = tdf['reason'].value_counts().to_dict()
    print(f'\nExit breakdown: {exits}')
    tdf.to_csv(OUT_CSV, index=False)
    print(f'Saved: {OUT_CSV}')


if __name__ == '__main__':
    main()
