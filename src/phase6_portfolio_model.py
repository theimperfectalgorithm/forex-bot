"""
Forex Bot - Phase 6: The portfolio model + 5ers pass probability
=================================================================
1. AUDIT: faithful backtest of the LIVE london_breakout (GBPJPY/EURJPY):
   H4 strict alignment (last>SMA50>SMA200), M15 Asian range 00-07 UTC,
   M15 close beyond edge during London (08-13) / NY (13-22), one trade
   per session, 2-consec-loss daily pause, SL/TP anchored at the range
   edge (50% / 100% of range), Friday close, risk 1%.
2. NEW-EDGE SWEEP: the validated JPY-cross families on the two untested
   JPY crosses (NZDJPY, CADJPY): ARB (2 best variants) + AMR (2 best).
   Plus a first screen of weekly cross-sectional momentum (long the
   strongest / short the weakest of 11 pairs, 1-week hold).
3. PORTFOLIO: combine daily P&L of all qualifying components and Monte
   Carlo (5-day block bootstrap, 2000 runs x 126 trading days) against
   the 5ers step-1 rules: +8% target, -5% daily, -10% overall.

Output: data/phase6_portfolio_report.txt + console
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy_matrix_backtest import (
    SPREAD_PIPS, MONTHS, IS_MONTHS, run_sim, metrics, REPO_ROOT,
)
from phase2_meanrev_arb_search import signals_arb_p
from phase3b_amr_jpy_refine import signals_amr_v
from phase3_session_structure_search import fetch_tf

SPREAD_PIPS.setdefault('NZDJPY', 2.0)
SPREAD_PIPS.setdefault('CADJPY', 2.0)

OUT_TXT = REPO_ROOT / 'data' / 'phase6_portfolio_report.txt'
BAL0 = 100_000.0
lines = []


def say(s=''):
    print(s)
    lines.append(s)


# ── 1. Faithful london_breakout simulation ──────────────────────────────────

def sim_london_breakout(m15, h4, pip, spread):
    closes = m15['Close'].to_numpy()
    opens  = m15['Open'].to_numpy()
    highs  = m15['High'].to_numpy()
    lows   = m15['Low'].to_numpy()
    times  = m15.index
    hours, wday, dates = times.hour, times.weekday, times.date

    # strict H4 alignment mapped to M15 (closed H4 bars only)
    h4c = h4['Close'].to_numpy()
    s50  = pd.Series(h4c).rolling(50).mean().to_numpy()
    s200 = pd.Series(h4c).rolling(200).mean().to_numpy()
    tr_h4 = np.where((h4c > s50) & (s50 > s200), 1,
             np.where((h4c < s50) & (s50 < s200), -1, 0))
    h4_ct = (h4.index + pd.Timedelta(hours=4)).values
    m15_ct = (times + pd.Timedelta(minutes=15)).values
    idx = np.searchsorted(h4_ct, m15_ct, side='right') - 1
    trend = np.zeros(len(m15), int)
    trend[idx >= 0] = tr_h4[idx[idx >= 0]]

    balance = BAL0
    pos = None
    trades = []
    equity = np.empty(len(m15))
    day = {'cur': None, 'hi': None, 'lo': None, 'ldn': False, 'ny': False,
           'cl': 0, 'trend': 0}

    for i in range(len(m15)):
        if dates[i] != day['cur']:
            day = {'cur': dates[i], 'hi': None, 'lo': None,
                   'ldn': False, 'ny': False, 'cl': 0, 'trend': 0}
        if hours[i] < 7:
            day['hi'] = highs[i] if day['hi'] is None else max(day['hi'], highs[i])
            day['lo'] = lows[i]  if day['lo'] is None else min(day['lo'], lows[i])
        if hours[i] == 7 and day['trend'] == 0:
            day['trend'] = trend[i]              # prep-time trend snapshot

        if pos is not None:
            ep = er = None
            if pos['dir'] == 'BUY':
                if lows[i] <= pos['sl']: ep, er = pos['sl'], 'SL'
                elif highs[i] >= pos['tp']: ep, er = pos['tp'], 'TP'
            else:
                if highs[i] >= pos['sl']: ep, er = pos['sl'], 'SL'
                elif lows[i] <= pos['tp']: ep, er = pos['tp'], 'TP'
            if ep is None and wday[i] == 4 and hours[i] >= 20:
                ep, er = opens[i], 'FridayClose'
            if ep is not None:
                pips = ((ep - pos['entry']) if pos['dir'] == 'BUY'
                        else (pos['entry'] - ep)) / pip - spread
                pnl = pips * pos['upp']
                balance += pnl
                trades.append({'entry_time': pos['t'], 'exit_time': times[i],
                               'pnl': round(pnl, 2)})
                day['cl'] = day['cl'] + 1 if pnl < 0 else 0
                pos = None

        sess = ('london' if 8 <= hours[i] < 13 else
                'ny' if 13 <= hours[i] < 22 else None)
        if (pos is None and sess and day['hi'] is not None
                and not day[('ldn' if sess == 'london' else 'ny')]
                and day['cl'] < 2 and day['trend'] != 0):
            rng = (day['hi'] - day['lo']) / pip
            if rng >= 10:
                c = closes[i]
                sig = None
                if day['trend'] == 1 and c > day['hi']:
                    sig, edge = 'BUY', day['hi']
                elif day['trend'] == -1 and c < day['lo']:
                    sig, edge = 'SELL', day['lo']
                if sig:
                    sl_pips = rng * 0.5
                    risk = balance * 0.01
                    pos = {'dir': sig, 'entry': c, 't': times[i],
                           'sl': edge - sl_pips * pip if sig == 'BUY' else edge + sl_pips * pip,
                           'tp': edge + rng * pip if sig == 'BUY' else edge - rng * pip,
                           'upp': risk / sl_pips}
                    day['ldn' if sess == 'london' else 'ny'] = True

        fl = 0.0
        if pos is not None:
            fl = (((closes[i] - pos['entry']) if pos['dir'] == 'BUY'
                   else (pos['entry'] - closes[i])) / pip) * pos['upp']
        equity[i] = balance + fl

    return pd.DataFrame(trades), pd.Series(equity, index=times + pd.Timedelta(minutes=15))


# ── 2. Weekly cross-sectional momentum screen ───────────────────────────────

def sim_xsect(h1_data, pairs, risk_pct=0.005):
    """Mon 00:00: rank pairs by 20-day return / ATR20d; long best, short
    worst, hold 1 week. Coarse screen -- no intraweek SL."""
    daily = {}
    for p in pairs:
        c = h1_data[p]['Close'].resample('1D').last().dropna()
        daily[p] = c
    df = pd.DataFrame(daily).dropna()
    rets20 = df.pct_change(20)
    vol20  = df.pct_change().rolling(20).std() * np.sqrt(20)
    score = (rets20 / vol20).dropna()
    weekly_pnl = []
    mondays = [d for d in score.index if d.weekday() == 0][4:]
    for k in range(len(mondays) - 1):
        d0, d1 = mondays[k], mondays[k + 1]
        s = score.loc[d0]
        best, worst = s.idxmax(), s.idxmin()
        pnl = 0.0
        for pair, sign in ((best, 1), (worst, -1)):
            r = df[pair].loc[d1] / df[pair].loc[d0] - 1
            v = vol20[pair].loc[d0]
            if v and v > 0:
                pnl += sign * r / v * (BAL0 * risk_pct)   # vol-normalised
        weekly_pnl.append({'exit_time': d1, 'pnl': pnl})
    t = pd.DataFrame(weekly_pnl)
    wins = t[t['pnl'] > 0]['pnl'].sum()
    loss = abs(t[t['pnl'] <= 0]['pnl'].sum())
    return t, (wins / loss if loss > 0 else np.inf)


# ── 3. portfolio helpers ─────────────────────────────────────────────────────

def daily_pnl(tdf):
    if tdf.empty:
        return pd.Series(dtype=float)
    s = tdf.set_index(pd.to_datetime(tdf['exit_time'], utc=True))['pnl']
    return s.resample('1D').sum()


def stat_line(name, tdf, eq, is_start, oos_start, now):
    mi = metrics(tdf, eq, is_start, oos_start)
    mo = metrics(tdf, eq, oos_start, now)
    say(f"  {name:<26} IS: n={mi.get('trades',0):>3} PF={mi.get('pf',0):<5} "
        f"DD={mi.get('max_dd_pct',0):>5.2f}%  "
        f"OOS: n={mo.get('trades',0):>3} PF={mo.get('pf',0):<5} "
        f"P&L=${mo.get('pnl',0):>+9,.2f}")
    return mi, mo


def monte_carlo(port_daily, n_runs=2000, days=126, seed=11):
    rng = np.random.default_rng(seed)
    arr = port_daily.to_numpy()
    n = len(arr)
    block = 5
    passes = busts = timeouts = 0
    days_to_pass = []
    for _ in range(n_runs):
        cum = 0.0
        outcome = 'timeout'
        d = 0
        while d < days:
            j = rng.integers(0, n - block)
            for x in arr[j:j + block]:
                d += 1
                cum += x
                if x <= -0.05 * BAL0 or cum <= -0.10 * BAL0:
                    outcome = 'bust'
                    break
                if cum >= 0.08 * BAL0:
                    outcome = 'pass'
                    days_to_pass.append(d)
                    break
            if outcome != 'timeout':
                break
        if outcome == 'pass': passes += 1
        elif outcome == 'bust': busts += 1
        else: timeouts += 1
    med = int(np.median(days_to_pass)) if days_to_pass else None
    return passes / n_runs, busts / n_runs, timeouts / n_runs, med


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    now = pd.Timestamp.now(tz='UTC')
    oos_start = now - pd.DateOffset(months=MONTHS - IS_MONTHS)
    is_start  = now - pd.DateOffset(months=MONTHS)

    say('================ PHASE 6: PORTFOLIO MODEL ================\n')
    components = {}     # name -> (tdf, weight-in-book bool)

    # ---- 1. london_breakout audit (LIVE book, risk 1%)
    say('--- 1. LIVE london_breakout audit (risk 1%) ---')
    for pair in ('GBPJPY', 'EURJPY'):
        m15 = fetch_tf(pair, 'M15'); h4 = fetch_tf(pair, 'H4')
        tdf, eq = sim_london_breakout(m15, h4, 0.01, SPREAD_PIPS[pair])
        mi, mo = stat_line(f'london_breakout {pair}', tdf, eq,
                           is_start, oos_start, now)
        components[f'LB-{pair}'] = (tdf, mi, mo)

    # ---- 2. new-edge sweep: ARB + AMR on NZDJPY / CADJPY
    say('\n--- 2. New JPY crosses: ARB + AMR ---')
    for pair in ('NZDJPY', 'CADJPY'):
        h1 = fetch_tf(pair, 'H1'); h4 = fetch_tf(pair, 'H4')
        m15 = fetch_tf(pair, 'M15')
        for tp_m, h4f in ((2.0, False), (1.5, True)):
            cands = signals_arb_p(h1, h4, 0.01, tp_m, h4f)
            tdf, eq = run_sim(h1, cands, 0.01, SPREAD_PIPS[pair], 0.005)
            stat_line(f'arb(tp{tp_m},h4={"Y" if h4f else "N"}) {pair}',
                      tdf, eq, is_start, oos_start, now)
            components[f'ARB-{pair}-tp{tp_m}{"Y" if h4f else "N"}'] = \
                (tdf, *[metrics(tdf, eq, a, b) for a, b in
                        ((is_start, oos_start), (oos_start, now))])
        for z, slm, eh in ((2.0, 1.5, 4), (2.5, 1.25, 4)):
            cands = signals_amr_v(m15, 0.01, SPREAD_PIPS[pair], z, slm, eh)
            tdf, eq = run_sim(m15, cands, 0.01, SPREAD_PIPS[pair], 0.0025,
                              time_exit_hour=7)
            stat_line(f'amr(z{z},sl{slm}) {pair}', tdf, eq,
                      is_start, oos_start, now)
            components[f'AMR-{pair}-z{z}'] = \
                (tdf, *[metrics(tdf, eq, a, b) for a, b in
                        ((is_start, oos_start), (oos_start, now))])

    # ---- validated book components (re-simulated for trade lists)
    say('\n--- 3. Validated book components ---')
    h1 = fetch_tf('GBPJPY', 'H1'); h4g = fetch_tf('GBPJPY', 'H4')
    cands = signals_arb_p(h1, h4g, 0.01, 2.0, False)
    tdf, eq = run_sim(h1, cands, 0.01, SPREAD_PIPS['GBPJPY'], 0.005)
    stat_line('ARB GBPJPY (LIVE demo)', tdf, eq, is_start, oos_start, now)
    components['ARB-GBPJPY'] = (tdf, None, None)

    for pair, z, slm, eh in (('GBPJPY', 2.5, 1.25, 4),
                             ('EURJPY', 2.0, 1.5, 6),
                             ('AUDJPY', 2.0, 1.5, 4)):
        m15 = fetch_tf(pair, 'M15')
        cands = signals_amr_v(m15, 0.01, SPREAD_PIPS[pair], z, slm, eh)
        tdf, eq = run_sim(m15, cands, 0.01, SPREAD_PIPS[pair], 0.0025,
                          time_exit_hour=7)
        stat_line(f'AMR {pair} (fwd-test cand.)', tdf, eq,
                  is_start, oos_start, now)
        components[f'AMR-{pair}'] = (tdf, None, None)

    # ---- cross-sectional momentum screen
    say('\n--- 4. Weekly cross-sectional momentum screen (11 pairs) ---')
    XP = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDUSD', 'NZDUSD',
          'GBPJPY', 'EURJPY', 'AUDJPY', 'NZDJPY', 'CADJPY']
    h1_data = {p: fetch_tf(p, 'H1') for p in XP}
    xt, xpf = sim_xsect(h1_data, XP)
    say(f'  xsect momentum: {len(xt)} weeks, PF={xpf:.2f}, '
        f'P&L=${xt["pnl"].sum():+,.2f}')

    # ---- 5. PORTFOLIO ASSEMBLY + MONTE CARLO
    say('\n--- 5. Portfolio Monte Carlo (5ers step 1: +8% / -5% day / -10%) ---')
    books = {
        'A: validated only (ARB-GBPJPY)': ['ARB-GBPJPY'],
        'B: A + AMR trio': ['ARB-GBPJPY', 'AMR-GBPJPY', 'AMR-EURJPY',
                            'AMR-AUDJPY'],
        'C: B + london_breakout (live)': ['ARB-GBPJPY', 'AMR-GBPJPY',
                                          'AMR-EURJPY', 'AMR-AUDJPY',
                                          'LB-GBPJPY', 'LB-EURJPY'],
    }
    for label, names in books.items():
        series = [daily_pnl(components[n][0]) for n in names
                  if n in components and not components[n][0].empty]
        port = pd.concat(series, axis=1).fillna(0.0).sum(axis=1)
        port = port.asfreq('D', fill_value=0.0)
        port = port[port.index.dayofweek < 5]          # business days
        mo_mean = port.resample('ME').sum().mean()
        mo_std  = port.resample('ME').sum().std()
        cum = port.cumsum()
        dd = (cum - cum.cummax()).min()
        p_pass, p_bust, p_to, med = monte_carlo(port)
        say(f'\n  BOOK {label}')
        say(f'    monthly mean=${mo_mean:+,.0f} ({mo_mean/BAL0*100:.2f}%)  '
            f'std=${mo_std:,.0f}  hist maxDD=${dd:,.0f} ({dd/BAL0*100:.2f}%)')
        say(f'    6-month MC: PASS {p_pass*100:.0f}%  BUST {p_bust*100:.0f}%  '
            f'timeout {p_to*100:.0f}%  '
            f'median days-to-pass={med if med else "n/a"}')

    OUT_TXT.write_text('\n'.join(lines), encoding='utf-8')
    say(f'\nSaved: {OUT_TXT}')


if __name__ == '__main__':
    main()
