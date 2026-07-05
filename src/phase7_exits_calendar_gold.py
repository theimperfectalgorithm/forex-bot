"""
Forex Bot - Phase 7: Exit management + EU/GU calendar screen + XAUUSD
======================================================================
1. EXIT STUDY (validated book only): does moving SL to breakeven at
   0.5R / 0.75R / 1.0R, or trailing by the original SL distance after
   1R, improve ARB / AMR? Conservative intrabar model: the move is
   ARMED when a bar's favorable extreme reaches the trigger and takes
   effect from the NEXT bar (same-bar reversals still hit the original
   SL). Selection on in-sample only; OOS reported for the winner.
2. CALENDAR SCREEN (EURUSD/GBPUSD): day-of-week drift, post-big-day
   reversal, month-end drift -- the last untested conditioning angle.
3. XAUUSD: the validated session-structure families on gold (pip=0.1,
   spread 3.5 pips, ~2.8y of MT5 history): ARB grid, AMR, NY-overlap
   continuation.
4. Re-run the Book-B Monte Carlo with the best exit modes (+gold if it
   qualifies) against 5ers step 1 (+8% / -5% day / -10%).

Output: data/phase7_report.txt
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
from phase3_session_structure_search import fetch_tf, signals_nyc
from phase6_portfolio_model import daily_pnl, monte_carlo, BAL0

SPREAD_PIPS.setdefault('CADJPY', 2.0)
SPREAD_PIPS.setdefault('XAUUSD', 3.5)

OUT_TXT = REPO_ROOT / 'data' / 'phase7_report.txt'
lines = []


def say(s=''):
    print(s)
    lines.append(s)


# ── exit-mode simulator (extends run_sim with BE / trailing) ────────────────

def run_sim_exits(bars, candidates, pip, spread_pips, risk_pct,
                  time_exit_hour=None, be_r=None, trail_after_r=None):
    cand = {i: (d, slp, tpp) for i, d, slp, tpp in candidates}
    opens  = bars['Open'].to_numpy()
    highs  = bars['High'].to_numpy()
    lows   = bars['Low'].to_numpy()
    closes = bars['Close'].to_numpy()
    times  = bars.index
    wd, hrs, dts = times.weekday, times.hour, times.date

    balance, pos, last_date = BAL0, None, None
    trades = []
    equity = np.empty(len(bars))

    for i in range(len(bars)):
        if pos is not None:
            # apply pending SL move decided on the PREVIOUS bar
            if pos.get('pending_sl') is not None:
                if pos['dir'] == 'BUY':
                    pos['sl'] = max(pos['sl'], pos['pending_sl'])
                else:
                    pos['sl'] = min(pos['sl'], pos['pending_sl'])
                pos['pending_sl'] = None

            ep = er = None
            if time_exit_hour is not None and hrs[i] >= time_exit_hour:
                ep, er = opens[i], 'TimeExit'
            elif pos['dir'] == 'BUY':
                if lows[i] <= pos['sl']:
                    ep = pos['sl']
                    er = 'BE/Trail' if pos['sl'] >= pos['entry'] else 'SL'
                elif highs[i] >= pos['tp']:
                    ep, er = pos['tp'], 'TP'
            else:
                if highs[i] >= pos['sl']:
                    ep = pos['sl']
                    er = 'BE/Trail' if pos['sl'] <= pos['entry'] else 'SL'
                elif lows[i] <= pos['tp']:
                    ep, er = pos['tp'], 'TP'
            if ep is None and wd[i] == 4 and hrs[i] >= 20:
                ep, er = opens[i], 'FridayClose'

            if ep is not None:
                pips = ((ep - pos['entry']) if pos['dir'] == 'BUY'
                        else (pos['entry'] - ep)) / pip - spread_pips
                pnl = pips * pos['upp']
                balance += pnl
                trades.append({'entry_time': pos['t'], 'exit_time': times[i],
                               'reason': er, 'pips': round(pips, 1),
                               'pnl': round(pnl, 2)})
                pos = None
            else:
                # arm BE / trail from this bar's favorable extreme
                fav = (highs[i] - pos['entry'] if pos['dir'] == 'BUY'
                       else pos['entry'] - lows[i]) / pip
                slp = pos['slp']
                if be_r is not None and fav >= be_r * slp:
                    tgt = pos['entry']
                    pos['pending_sl'] = tgt
                if trail_after_r is not None and fav >= trail_after_r * slp:
                    pos['peak'] = max(pos.get('peak', 0.0), fav)
                    dist = slp * pip
                    tgt = (pos['entry'] + (pos['peak'] * pip) - dist
                           if pos['dir'] == 'BUY'
                           else pos['entry'] - (pos['peak'] * pip) + dist)
                    pos['pending_sl'] = tgt

        if pos is None and i in cand:
            d, slp, tpp = cand[i]
            if dts[i] != last_date and slp > 0:
                c = closes[i]
                pos = {'dir': d, 'entry': c, 't': times[i], 'slp': slp,
                       'sl': c - slp * pip if d == 'BUY' else c + slp * pip,
                       'tp': c + tpp * pip if d == 'BUY' else c - tpp * pip,
                       'upp': balance * risk_pct / slp, 'pending_sl': None}
                last_date = dts[i]

        fl = 0.0
        if pos is not None:
            fl = (((closes[i] - pos['entry']) if pos['dir'] == 'BUY'
                   else (pos['entry'] - closes[i])) / pip) * pos['upp']
        equity[i] = balance + fl

    return (pd.DataFrame(trades),
            pd.Series(equity, index=times + (times[1] - times[0])))


EXIT_MODES = {
    'baseline'  : {},
    'BE@0.5R'   : {'be_r': 0.5},
    'BE@0.75R'  : {'be_r': 0.75},
    'BE@1.0R'   : {'be_r': 1.0},
    'trail@1R'  : {'trail_after_r': 1.0},
}


def main():
    now = pd.Timestamp.now(tz='UTC')
    oos_start = now - pd.DateOffset(months=MONTHS - IS_MONTHS)
    is_start  = now - pd.DateOffset(months=MONTHS)

    say('================ PHASE 7 ================\n')

    # ---------- data ----------
    data = {}
    for p in ('GBPJPY', 'EURJPY', 'AUDJPY', 'CADJPY'):
        data[p] = {'h1': fetch_tf(p, 'H1'), 'h4': fetch_tf(p, 'H4'),
                   'm15': fetch_tf(p, 'M15')}

    BOOK = {
        'ARB-GBPJPY': ('arb', 'GBPJPY', (2.0, False), 0.005, None),
        'ARB-CADJPY': ('arb', 'CADJPY', (2.0, False), 0.005, None),
        'AMR-GBPJPY': ('amr', 'GBPJPY', (2.5, 1.25, 4), 0.0025, 7),
        'AMR-EURJPY': ('amr', 'EURJPY', (2.0, 1.5, 6), 0.0025, 7),
        'AMR-AUDJPY': ('amr', 'AUDJPY', (2.0, 1.5, 4), 0.0025, 7),
        'AMR-CADJPY': ('amr', 'CADJPY', (2.0, 1.5, 4), 0.0025, 7),
    }

    # ---------- 1. exit study ----------
    say('--- 1. EXIT STUDY (select on IS, confirm OOS) ---')
    best_mode, best_trades = {}, {}
    for name, (kind, pair, prm, risk, teh) in BOOK.items():
        pip = 0.01
        if kind == 'arb':
            cands = signals_arb_p(data[pair]['h1'], data[pair]['h4'],
                                  pip, prm[0], prm[1])
            bars = data[pair]['h1']
        else:
            cands = signals_amr_v(data[pair]['m15'], pip, SPREAD_PIPS[pair],
                                  prm[0], prm[1], prm[2])
            bars = data[pair]['m15']
        results = {}
        for mode, kw in EXIT_MODES.items():
            tdf, eq = run_sim_exits(bars, cands, pip, SPREAD_PIPS[pair],
                                    risk, time_exit_hour=teh, **kw)
            mi = metrics(tdf, eq, is_start, oos_start)
            mo = metrics(tdf, eq, oos_start, now)
            results[mode] = (mi.get('pnl', 0), mi.get('pf', 0),
                             mo.get('pnl', 0), mo.get('pf', 0), tdf)
        base = results['baseline']
        win = max(results, key=lambda m: results[m][0])   # IS pnl
        w = results[win]
        say(f'  {name:<12} baseline IS ${base[0]:>+8,.0f}/PF {base[1]:<5} '
            f'OOS ${base[2]:>+8,.0f}/PF {base[3]:<5} | best-IS: {win:<9} '
            f'IS ${w[0]:>+8,.0f}/PF {w[1]:<5} OOS ${w[2]:>+8,.0f}/PF {w[3]}')
        # adopt winner only if it beats baseline on BOTH IS and OOS pnl
        if win != 'baseline' and w[0] > base[0] and w[2] > base[2]:
            best_mode[name], best_trades[name] = win, w[4]
        else:
            best_mode[name], best_trades[name] = 'baseline', base[4]
    say(f'  ADOPTED: ' + ', '.join(f'{k}={v}' for k, v in best_mode.items()))

    # ---------- 2. EU/GU calendar screen ----------
    say('\n--- 2. EURUSD/GBPUSD calendar screen (daily bars) ---')
    for pair in ('EURUSD', 'GBPUSD'):
        h1 = fetch_tf(pair, 'H1')
        d = h1['Close'].resample('1D').last().dropna()
        o = h1['Open'].resample('1D').first().dropna()
        idx = d.index.intersection(o.index)
        ret = (d[idx] / o[idx] - 1) * 100
        ret = ret[ret.index.dayofweek < 5]
        say(f'  {pair} day-of-week mean daily return (%):')
        for wd, nm in enumerate(['Mon', 'Tue', 'Wed', 'Thu', 'Fri']):
            sub = ret[ret.index.dayofweek == wd]
            t = sub.mean() / (sub.std() / np.sqrt(len(sub))) if len(sub) else 0
            say(f'    {nm}: {sub.mean():+.4f}%  (n={len(sub)}, t={t:+.2f})')
        big = ret.shift(1).abs() > 1.5 * ret.rolling(20).std().shift(1)
        rev = -np.sign(ret.shift(1)[big]) * ret[big]
        t_rev = (rev.mean() / (rev.std() / np.sqrt(len(rev)))
                 if len(rev) > 5 else 0)
        say(f'    post-big-day reversal: mean {rev.mean():+.4f}%/day '
            f'(n={len(rev)}, t={t_rev:+.2f})')
        me = ret[ret.index.is_month_end
                 | (ret.index + pd.Timedelta(days=1)).is_month_end]
        t_me = me.mean() / (me.std() / np.sqrt(len(me))) if len(me) > 5 else 0
        say(f'    month-end drift: mean {me.mean():+.4f}%/day '
            f'(n={len(me)}, t={t_me:+.2f})')

    # ---------- 3. XAUUSD ----------
    say('\n--- 3. XAUUSD session-structure campaign (pip=0.1, spread 3.5p) ---')
    g = {'h1': fetch_tf('XAUUSD', 'H1'), 'h4': fetch_tf('XAUUSD', 'H4'),
         'm15': fetch_tf('XAUUSD', 'M15')}
    say(f"  data: {g['h1'].index[0].date()} -> {g['h1'].index[-1].date()}")
    gpip = 0.1
    gold_pass = []
    for tp_m, h4f in ((1.5, True), (1.5, False), (2.0, True), (2.0, False)):
        cands = signals_arb_p(g['h1'], g['h4'], gpip, tp_m, h4f, min_range=30)
        tdf, eq = run_sim(g['h1'], cands, gpip, 3.5, 0.0025)
        mi = metrics(tdf, eq, is_start, oos_start)
        mo = metrics(tdf, eq, oos_start, now)
        lbl = f'arb(tp{tp_m},h4={"Y" if h4f else "N"})'
        say(f'  {lbl:<18} IS: n={mi.get("trades",0):>3} PF={mi.get("pf",0):<5} '
            f'DD={mi.get("max_dd_pct",0):>5.2f}%  OOS: n={mo.get("trades",0):>3} '
            f'PF={mo.get("pf",0):<5} P&L=${mo.get("pnl",0):>+8,.0f}')
        if (mi.get('pf', 0) > 1.3 and mi.get('max_dd_pct', 99) < 8
                and mo.get('pnl', 0) > 0):
            gold_pass.append((lbl, tdf))
    for z, slm, eh in ((2.0, 1.5, 4), (2.5, 1.25, 4)):
        cands = signals_amr_v(g['m15'], gpip, 3.5, z, slm, eh)
        tdf, eq = run_sim(g['m15'], cands, gpip, 3.5, 0.0025, time_exit_hour=7)
        mi = metrics(tdf, eq, is_start, oos_start)
        mo = metrics(tdf, eq, oos_start, now)
        lbl = f'amr(z{z},sl{slm})'
        say(f'  {lbl:<18} IS: n={mi.get("trades",0):>3} PF={mi.get("pf",0):<5} '
            f'DD={mi.get("max_dd_pct",0):>5.2f}%  OOS: n={mo.get("trades",0):>3} '
            f'PF={mo.get("pf",0):<5} P&L=${mo.get("pnl",0):>+8,.0f}')
        if (mi.get('pf', 0) > 1.3 and mi.get('max_dd_pct', 99) < 8
                and mo.get('pnl', 0) > 0):
            gold_pass.append((lbl, tdf))
    for thr in (0.75, 1.25):
        cands = signals_nyc(g['h1'], gpip, thr)
        tdf, eq = run_sim(g['h1'], cands, gpip, 3.5, 0.0025, time_exit_hour=20)
        mi = metrics(tdf, eq, is_start, oos_start)
        mo = metrics(tdf, eq, oos_start, now)
        lbl = f'nyc(thr{thr})'
        say(f'  {lbl:<18} IS: n={mi.get("trades",0):>3} PF={mi.get("pf",0):<5} '
            f'DD={mi.get("max_dd_pct",0):>5.2f}%  OOS: n={mo.get("trades",0):>3} '
            f'PF={mo.get("pf",0):<5} P&L=${mo.get("pnl",0):>+8,.0f}')
        if (mi.get('pf', 0) > 1.3 and mi.get('max_dd_pct', 99) < 8
                and mo.get('pnl', 0) > 0):
            gold_pass.append((lbl, tdf))
    say(f'  gold qualifiers: {[l for l, _ in gold_pass] or "none"}')

    # ---------- 4. final Monte Carlo ----------
    say('\n--- 4. BOOK B+ Monte Carlo with adopted exits (+gold if any) ---')
    series = [daily_pnl(t) for t in best_trades.values() if not t.empty]
    series += [daily_pnl(t) for _, t in gold_pass[:1]]   # at most 1 gold slot
    port = pd.concat(series, axis=1).fillna(0.0).sum(axis=1)
    port = port.asfreq('D', fill_value=0.0)
    port = port[port.index.dayofweek < 5]
    mo_mean = port.resample('ME').sum().mean()
    mo_std  = port.resample('ME').sum().std()
    cum = port.cumsum()
    dd = (cum - cum.cummax()).min()
    p_pass, p_bust, p_to, med = monte_carlo(port)
    say(f'  components: {list(best_trades.keys()) + [l for l, _ in gold_pass[:1]]}')
    say(f'  monthly mean=${mo_mean:+,.0f} ({mo_mean/BAL0*100:.2f}%)  '
        f'std=${mo_std:,.0f}  hist maxDD=${dd:,.0f} ({dd/BAL0*100:.2f}%)')
    say(f'  6-month MC: PASS {p_pass*100:.0f}%  BUST {p_bust*100:.0f}%  '
        f'timeout {p_to*100:.0f}%  median days-to-pass={med}')

    OUT_TXT.write_text('\n'.join(lines), encoding='utf-8')
    say(f'\nSaved: {OUT_TXT}')


if __name__ == '__main__':
    main()
