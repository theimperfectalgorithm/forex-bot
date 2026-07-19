"""
Forex Bot - Phase 9: $5K challenge Monte Carlo (5ers 2-step CLASSIC)
=====================================================================
Question: does Book B+ still clear the challenge when run on a $5,000
account, where the 0.01 minimum lot quantizes position sizes -- and
should the instance run risk_scale 1.0 or 0.5?

Simulates the full 8-slot book with $5K-granular sizing:
  lots = max(0.01, round(risk_usd / (sl_pips x pip_value), 2))
(so min-lot flooring and rounding distortion are IN the P&L), then
Monte-Carlos the combined daily P&L against the CLASSIC rules:
  step 1: +8% target | step 2: +5% | -5% daily | -10% STATIC floor |
  unlimited time (horizons reported at 126 and 252 trading days) |
  minimum 3 profitable days (counted; never binding in practice).

Reference: the $100k continuous-sizing model = 83% pass / 2% bust /
median 55 days (phase 7). This isolates what $5K granularity changes.

Exits are the LIVE book's behavior (baseline SL/TP; AMR 07:00 exit,
MON 21:00 exit; no breakeven -- matches deployment, 4691a22).

Requirements: MetaTrader 5 OPEN and LOGGED IN.
Output: data/phase9_5k_report.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy_matrix_backtest import SPREAD_PIPS, REPO_ROOT
from phase2_meanrev_arb_search import signals_arb_p
from phase3b_amr_jpy_refine import signals_amr_v
from phase3_session_structure_search import fetch_tf
from phase8_monday_validation import signals_monday

SPREAD_PIPS.setdefault('CADJPY', 2.0)
SPREAD_PIPS.setdefault('XAUUSD', 3.5)

OUT_TXT = REPO_ROOT / 'data' / 'phase9_5k_report.txt'
BAL0 = 5_000.0
MIN_LOT = 0.01

# pip value per 1.0 lot, USD account (approximations adequate for a
# granularity study; JPY value floats with USDJPY -- use current-era level)
PVAL = {'GBPJPY': 5.4, 'EURJPY': 5.4, 'AUDJPY': 5.4, 'CADJPY': 5.4,
        'GBPUSD': 10.0, 'XAUUSD': 10.0}

lines = []
def say(s=''):
    print(s)
    lines.append(s)


def run_sim_5k(bars, cands, pip, spread, risk_pct, pval,
               scale, time_exit_hour=None):
    """phase-6 engine semantics with $5K min-lot-granular sizing."""
    cand = {i: (d, slp, tpp) for i, d, slp, tpp in cands}
    o = bars['Open'].to_numpy();  h = bars['High'].to_numpy()
    l = bars['Low'].to_numpy();   c = bars['Close'].to_numpy()
    times = bars.index
    wd, hrs, dts = times.weekday, times.hour, times.date
    balance, pos, last_date = BAL0, None, None
    trades = []
    for i in range(len(bars)):
        if pos is not None:
            ep = er = None
            if time_exit_hour is not None and hrs[i] >= time_exit_hour:
                ep, er = o[i], 'TimeExit'
            elif pos['dir'] == 'BUY':
                if l[i] <= pos['sl']: ep, er = pos['sl'], 'SL'
                elif h[i] >= pos['tp']: ep, er = pos['tp'], 'TP'
            else:
                if h[i] >= pos['sl']: ep, er = pos['sl'], 'SL'
                elif l[i] <= pos['tp']: ep, er = pos['tp'], 'TP'
            if ep is None and wd[i] == 4 and hrs[i] >= 20:
                ep, er = o[i], 'FridayClose'
            if ep is not None:
                pips = ((ep - pos['e']) if pos['dir'] == 'BUY'
                        else (pos['e'] - ep)) / pip - spread
                pnl = pips * pos['lots'] * pval
                balance += pnl
                trades.append({'exit_time': times[i], 'pnl': round(pnl, 2),
                               'lots': pos['lots']})
                pos = None
        if pos is None and i in cand:
            d, slp, tpp = cand[i]
            if dts[i] != last_date and slp > 0:
                risk_usd = balance * risk_pct * scale
                lots = max(MIN_LOT, round(risk_usd / (slp * pval), 2))
                pos = {'dir': d, 'e': c[i], 'lots': lots,
                       'sl': c[i] - slp*pip if d == 'BUY' else c[i] + slp*pip,
                       'tp': c[i] + tpp*pip if d == 'BUY' else c[i] - tpp*pip}
                last_date = dts[i]
    return pd.DataFrame(trades)


def mc_5k(port_daily, target_pct, n_runs=3000, days=252, seed=17):
    rng = np.random.default_rng(seed)
    arr = port_daily.to_numpy()
    block = 5
    res = {'pass': 0, 'bust': 0, 'timeout': 0}
    d2p, p126 = [], 0
    for _ in range(n_runs):
        cum, d, outcome, prof_days = 0.0, 0, 'timeout', 0
        while d < days:
            j = rng.integers(0, len(arr) - block)
            for x in arr[j:j + block]:
                d += 1
                cum += x
                if x > 0:
                    prof_days += 1
                if x <= -0.05 * BAL0 or cum <= -0.10 * BAL0:
                    outcome = 'bust'; break
                if cum >= target_pct * BAL0 and prof_days >= 3:
                    outcome = 'pass'; d2p.append(d); break
            if outcome != 'timeout':
                break
        res[outcome] += 1
        if outcome == 'pass' and d <= 126:
            p126 += 1
    n = n_runs
    return {'pass_252': res['pass']/n, 'pass_126': p126/n,
            'bust': res['bust']/n, 'timeout': res['timeout']/n,
            'median_days': int(np.median(d2p)) if d2p else None}


def main():
    say('========== PHASE 9: $5K CLASSIC CHALLENGE SIM ==========\n')
    data = {}
    for p in ('GBPJPY', 'EURJPY', 'AUDJPY', 'CADJPY', 'XAUUSD', 'GBPUSD'):
        say(f'fetching {p} ...')
        data[p] = {'h1': fetch_tf(p, 'H1'), 'h4': fetch_tf(p, 'H4'),
                   'm15': fetch_tf(p, 'M15')}
    say('')

    # (name, bars, candidate fn output, pip, spread, risk, pval, time_exit)
    def slots():
        out = []
        for pair, tp_m, mr in (('GBPJPY', 2.0, 10), ('CADJPY', 2.0, 10)):
            cands = signals_arb_p(data[pair]['h1'], data[pair]['h4'],
                                  0.01, tp_m, False, min_range=mr)
            out.append((f'{pair}@arb', data[pair]['h1'], cands, 0.01,
                        SPREAD_PIPS[pair], 0.005, PVAL[pair], None))
        cands = signals_arb_p(data['XAUUSD']['h1'], data['XAUUSD']['h4'],
                              0.1, 1.5, False, min_range=30)
        out.append(('XAUUSD@arb', data['XAUUSD']['h1'], cands, 0.1,
                    3.5, 0.0025, PVAL['XAUUSD'], None))
        for pair, z, slm, eh in (('GBPJPY', 2.5, 1.25, 4),
                                 ('EURJPY', 2.0, 1.5, 6),
                                 ('AUDJPY', 2.0, 1.5, 4),
                                 ('CADJPY', 2.0, 1.5, 4)):
            cands = signals_amr_v(data[pair]['m15'], 0.01,
                                  SPREAD_PIPS[pair], z, slm, eh)
            out.append((f'{pair}@amr', data[pair]['m15'], cands, 0.01,
                        SPREAD_PIPS[pair], 0.0025, PVAL[pair], 7))
        cands = signals_monday(data['GBPUSD']['h1'], 1.25, 1.0)
        out.append(('GBPUSD@mon', data['GBPUSD']['h1'], cands, 0.0001,
                    SPREAD_PIPS['GBPUSD'], 0.0025, PVAL['GBPUSD'], 21))
        return out

    SLOTS = slots()
    for scale in (1.0, 0.5):
        say(f'--- risk_scale {scale} ---')
        series = []
        for name, bars, cands, pip, spread, risk, pval, teh in SLOTS:
            tdf = run_sim_5k(bars, cands, pip, spread, risk, pval,
                             scale, time_exit_hour=teh)
            if tdf.empty:
                continue
            med_lots = tdf['lots'].median()
            say(f'  {name:12} n={len(tdf):>4}  median lots={med_lots:.2f}  '
                f'net=${tdf["pnl"].sum():+8.2f}')
            s = tdf.set_index(pd.to_datetime(tdf['exit_time'], utc=True))['pnl']
            series.append(s.resample('1D').sum())
        port = pd.concat(series, axis=1).fillna(0.0).sum(axis=1)
        port = port.asfreq('D', fill_value=0.0)
        port = port[port.index.dayofweek < 5]
        mo = port.resample('ME').sum()
        say(f'  portfolio: monthly mean=${mo.mean():+,.2f} '
            f'({mo.mean()/BAL0*100:.2f}%)  std=${mo.std():,.2f}')
        s1 = mc_5k(port, 0.08)
        s2 = mc_5k(port, 0.05)
        say(f'  STEP 1 (+8%):  pass(1y) {s1["pass_252"]*100:.0f}%  '
            f'pass(6mo) {s1["pass_126"]*100:.0f}%  bust {s1["bust"]*100:.0f}%  '
            f'median {s1["median_days"]} days')
        say(f'  STEP 2 (+5%):  pass(1y) {s2["pass_252"]*100:.0f}%  '
            f'pass(6mo) {s2["pass_126"]*100:.0f}%  bust {s2["bust"]*100:.0f}%  '
            f'median {s2["median_days"]} days\n')

    OUT_TXT.write_text('\n'.join(lines), encoding='utf-8')
    say(f'Saved: {OUT_TXT}')


if __name__ == '__main__':
    main()
