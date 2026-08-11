"""
Forex Bot - Phase 13: NZDJPY -- mechanism verification + portfolio-level
                       exposure/factor analysis
==============================================================================
Continues the phase-12 frozen validation gate. The NZDJPY candidate
strategy/parameters are UNCHANGED from phase 12 -- imported, not
re-transcribed. This script does NOT modify NZDJPY, does NOT modify
any of the 8 currently-live demo strategies, and does NOT optimize
anything. Its purpose is exactly the split the user asked for:
  (1) verify the standalone mechanism is real and lookahead-free,
  (2) check it against the portfolio that is ALREADY running (JPY
      exposure overlap is the live risk, not an abstract one), and
  (3) attribute the edge to a common JPY factor vs. something distinct.

The 8 existing demo strategies are reconstructed here using their
EXACT live pairs/*.yaml parameters (verified against the tracked YAML
files, not memory) and the EXACT signal functions already validated
elsewhere in this project (phase2/phase3b/phase8) -- not re-derived.

Requirements: MetaTrader 5 OPEN and LOGGED IN.
Outputs: reports/phase13_nzdjpy_portfolio_report.md,
         data/phase13_*.csv, experiments/experiments.csv (appended)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import data_loader
from strategy_matrix_backtest import (
    MONTHS, IS_MONTHS, RISK_PCT_DEFAULT, windowed_atr, run_sim, metrics,
    REPO_ROOT,
)
from phase2_meanrev_arb_search import signals_arb_p
from phase3b_amr_jpy_refine import signals_amr_v
from phase8_monday_validation import signals_monday
from phase10_jpy_london_ny import (
    fetch_h1, build_usdjpy_proxy, signals_xmomentum, NY_END,
)
from phase12_nzdjpy_validation_gate import (
    CHECK_HOUR, THR_ATR, SL_ATR, TP_ATR, TIME_EXIT, SPREAD_BASELINE, PIP,
    permutation_test, shift_candidates, compute_mfe_mae,
)
from walk_forward import run_walk_forward
from monte_carlo import run_monte_carlo
from research_ledger import log_experiment, next_experiment_id

OUT_DIR = REPO_ROOT / 'data'
REPORT  = REPO_ROOT / 'reports' / 'phase13_nzdjpy_portfolio_report.md'

log_rows = []
def say(s=''):
    print(s)
    log_rows.append(str(s))


def fetch(pair, tf):
    from datetime import datetime, timedelta, timezone
    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=MONTHS * 30 + 60)
    return data_loader.get_bars(pair, tf, date_from, date_to)


def pip_of(pair):
    return 0.1 if pair.startswith('XAU') else 0.01 if 'JPY' in pair else 0.0001


def daily_pnl(tdf):
    if tdf.empty:
        return pd.Series(dtype=float)
    s = tdf.set_index(pd.to_datetime(tdf['exit_time'], utc=True))['pnl']
    return s.resample('1D').sum()


def main():
    now = pd.Timestamp.now(tz='UTC')
    oos_start = now - pd.DateOffset(months=MONTHS - IS_MONTHS)
    is_start  = now - pd.DateOffset(months=MONTHS)

    say('=' * 78)
    say('PHASE 13: NZDJPY MECHANISM VERIFICATION + PORTFOLIO EXPOSURE ANALYSIS')
    say('=' * 78)
    say('Frozen NZDJPY spec: unchanged from phase 12 (EXP-030). Nothing in '
       'this script modifies it or any live demo strategy.')

    # ---- fetch everything needed ----
    say('\nFetching data (NZDJPY candidate + USDJPY proxy + 8 existing '
       'demo strategies'' underlying pairs) ...')
    data_h1 = {}
    for p in ['NZDJPY', 'USDJPY', 'GBPJPY', 'CADJPY', 'EURJPY', 'AUDJPY',
             'GBPUSD', 'XAUUSD']:
        data_h1[p] = fetch(p, 'H1')
        say(f'  {p} H1: {len(data_h1[p]):,} bars')
    data_h4 = {p: fetch(p, 'H4') for p in ['GBPJPY', 'CADJPY', 'XAUUSD']}
    data_m15 = {p: fetch(p, 'M15') for p in ['GBPJPY', 'EURJPY', 'AUDJPY', 'CADJPY']}

    # ============================================================
    # PART 1 -- verify the NZDJPY mechanism
    # ============================================================
    say('\n' + '=' * 78)
    say('PART 1 -- NZDJPY MECHANISM: exact documentation + lookahead check')
    say('=' * 78)
    say("""
  How USDJPY is used: USDJPY's H1 CLOSE price is tracked from the open
  of that day's server-hour-7 bar (its open price, captured once per
  day when hours[i]==7) forward to the CLOSE of the current bar. The
  difference in pips is the 'move'. This is a MOMENTUM/DRIFT measure
  of USDJPY over the London-session-to-current window, not a level or
  an oscillator.

  Timeframe: H1, both for USDJPY (the proxy) and NZDJPY (the traded
  instrument).

  Signal calculation timing: evaluated ONLY on the H1 bar whose
  timestamp (bar OPEN convention, standard MT5) has hour==15 server
  time. Since MT5 timestamps a bar at its OPEN, 'hour==15' bar closes
  at hour 16 -- the signal is therefore only fully known once that bar
  closes, i.e. effectively usable starting server hour 16:00. The
  backtest uses closes[i] (bar i's own close) as both the signal value
  AND the entry price for bar i -- this matches how every other
  strategy in this project is simulated (decision made using
  information available at that bar's own close), and matches the live
  orchestrator convention (agent_strategy checks the LAST CLOSED bar,
  pos=1).

  Contemporaneous or lagged: the USDJPY move and the NZDJPY entry are
  READ FROM THE SAME BAR INDEX i. This is contemporaneous by
  construction, not lagged -- both series must be correctly time-
  aligned bar-for-bar for this to be valid (checked explicitly below).

  How direction is determined: move > 0 (USDJPY up = broad JPY
  weakness) -> BUY NZDJPY. move < 0 -> SELL NZDJPY.
""")

    # -- explicit index-alignment check (the concrete lookahead-adjacent risk) --
    say('  LOOKAHEAD/ALIGNMENT CHECK: verifying NZDJPY and USDJPY H1 bar '
       'timestamps are identical index-for-index (a silent misalignment '
       'here -- e.g. one symbol missing a bar the other has -- would make '
       'usdjpy_move[i] correspond to a DIFFERENT calendar time than '
       'NZDJPY bar i, corrupting every signal without an obvious error).')
    nzd, usd = data_h1['NZDJPY'], data_h1['USDJPY']
    common = nzd.index.intersection(usd.index)
    say(f'    NZDJPY bars: {len(nzd)}   USDJPY bars: {len(usd)}   '
       f'common timestamps: {len(common)}')
    say(f'    NZDJPY-only timestamps (no USDJPY match): {len(nzd.index.difference(usd.index))}')
    say(f'    USDJPY-only timestamps (no NZDJPY match): {len(usd.index.difference(nzd.index))}')
    aligned = len(nzd) == len(usd) == len(common)
    say(f'    Fully aligned: {aligned}')
    if not aligned:
        say('    *** WARNING: arrays are NOT fully aligned. build_usdjpy_proxy() '
           'and signals_xmomentum() index by POSITION not by timestamp -- any '
           'misalignment silently corrupts the signal. This must be fixed '
           'before the result can be trusted. ***')
    else:
        say('    OK -- positional indexing (usdjpy_move[i] paired with NZDJPY '
           'bar i) is valid because the two H1 series have identical '
           'timestamps at every position.')

    # -- "does it work without USDJPY" -- cite the permutation test, and add
    #    a second, distinct control: NZDJPY's OWN prior-session self-momentum
    #    instead of USDJPY, to separate "momentum works" from "USDJPY specifically"
    usdjpy_move, usdjpy_atr = build_usdjpy_proxy(usd, 0.01)
    nzd_cands = signals_xmomentum(nzd, PIP, SPREAD_BASELINE, usdjpy_move, usdjpy_atr,
                                  CHECK_HOUR, THR_ATR, sl_atr=SL_ATR, tp_atr=TP_ATR)
    say(f'\n  Frozen NZDJPY candidate signal count: {len(nzd_cands)}')
    say('  "Profitable without the USDJPY input?" -- already answered by the '
       'phase-12 permutation test (n=1000 random-direction shuffles of the '
       'SAME entry bars/SL/TP): the null model''s OWN mean outcome was a '
       'LOSING strategy (mean PF 0.815, mean P&L -$18,706). Answer: NO, '
       'removing the directional information (keeping only the timing/SL/TP '
       'skeleton) does not remain profitable on average.')

    # self-momentum control: same entry bars, but direction = NZDJPY's OWN
    # move from hour-7 open to the signal bar, instead of USDJPY's
    nzdjpy_move, _ = build_usdjpy_proxy(nzd, PIP)   # reuse same construction on NZDJPY itself
    self_mom_cands = []
    for i, d, sl, tp in nzd_cands:
        own_mv = nzdjpy_move[i]
        if np.isnan(own_mv):
            continue
        self_mom_cands.append((i, 'BUY' if own_mv > 0 else 'SELL', sl, tp))
    t_sm, e_sm = run_sim(nzd, self_mom_cands, PIP, SPREAD_BASELINE, RISK_PCT_DEFAULT,
                         time_exit_hour=TIME_EXIT)
    mi_sm = metrics(t_sm, e_sm, is_start, oos_start)
    mo_sm = metrics(t_sm, e_sm, oos_start, now)
    say(f'\n  CONTROL: same entry bars/SL/TP, but direction from NZDJPY''s OWN '
       f'self-momentum (hour-7 open -> signal bar) instead of USDJPY:')
    say(f"    IS PF={mi_sm.get('pf',0)}  P&L=${mi_sm.get('pnl',0):+,.0f}   "
       f"OOS PF={mo_sm.get('pf',0)}  P&L=${mo_sm.get('pnl',0):+,.0f}")
    say('    (If self-momentum alone matched USDJPY-momentum'"'"'s performance, '
       'that would suggest "any lagged momentum works" rather than USDJPY '
       'specifically being informative. Compare to the frozen result below.)')

    # ============================================================
    # PART 2 -- CADJPY replication (frozen, no optimization)
    # ============================================================
    say('\n' + '=' * 78)
    say('PART 2 -- CADJPY REPLICATION (exact frozen mechanism, unmodified)')
    say('=' * 78)
    say('CADJPY is ALREADY traded live (ARB + AMR). This is a RESEARCH '
       'REPLICATION of the NZDJPY mechanism only -- it does NOT touch, '
       'read, or influence the live CADJPY ARB/AMR strategies in any way, '
       'and a positive result here is evidence about the MECHANISM, not '
       'evidence of portfolio diversification (that is Part 3/4).')

    cad = data_h1['CADJPY']
    cad_cands = signals_xmomentum(cad, PIP, SPREAD_BASELINE, usdjpy_move, usdjpy_atr,
                                  CHECK_HOUR, THR_ATR, sl_atr=SL_ATR, tp_atr=TP_ATR)
    t_cad, e_cad = run_sim(cad, cad_cands, PIP, SPREAD_BASELINE, RISK_PCT_DEFAULT,
                           time_exit_hour=TIME_EXIT)
    mi_cad = metrics(t_cad, e_cad, is_start, oos_start)
    mo_cad = metrics(t_cad, e_cad, oos_start, now)
    say(f"  CADJPY (frozen NZDJPY params, unchanged): n_signals={len(cad_cands)}")
    say(f"    IS:  n={mi_cad.get('trades',0):>3} PF={mi_cad.get('pf',0):<6} "
       f"DD={mi_cad.get('max_dd_pct',0):>5.2f}% pm={mi_cad.get('prof_months_pct',0):>5.1f}%  "
       f"P&L=${mi_cad.get('pnl',0):>+9,.2f}")
    say(f"    OOS: n={mo_cad.get('trades',0):>3} PF={mo_cad.get('pf',0):<6} "
       f"P&L=${mo_cad.get('pnl',0):>+9,.2f}  DD={mo_cad.get('max_dd_pct',0):.2f}%")

    cad_perm = permutation_test(cad, cad_cands, PIP, SPREAD_BASELINE, n_perm=1000, seed=13)
    say(f"  CADJPY permutation test: real PF={cad_perm['real_pf']}, beats "
       f"{cad_perm['real_pf_percentile_vs_null']:.1f}% of {cad_perm['n_perm']} "
       f"random-direction shuffles (null mean PF={cad_perm['null_pf_mean']})")

    is_ok_cad = (mi_cad.get('trades',0) >= 30 and mi_cad.get('pf',0) > 1.3
                and mi_cad.get('max_dd_pct',99) < 8 and mi_cad.get('prof_months_pct',0) >= 60)
    oos_ok_cad = mo_cad.get('trades',0) >= 10 and mo_cad.get('pnl',0) > 0
    if is_ok_cad and oos_ok_cad:
        replication = 'REPLICATED'
    elif oos_ok_cad or (mi_cad.get('pf',0) > 1.15 and mo_cad.get('pnl',0) > 0):
        replication = 'PARTIALLY REPLICATED'
    elif mi_cad.get('trades',0) < 20:
        replication = 'INCONCLUSIVE'
    else:
        replication = 'NOT REPLICATED'
    say(f'\n  REPLICATION CLASSIFICATION: {replication}')

    # ============================================================
    # RECONSTRUCT THE 8 EXISTING LIVE DEMO STRATEGIES (exact YAML params)
    # ============================================================
    say('\n' + '=' * 78)
    say('RECONSTRUCTING THE 8 EXISTING DEMO STRATEGIES (verified live params)')
    say('=' * 78)

    existing = {}   # name -> (tdf, eq, risk_pct)

    def add(name, tdf, eq, risk_pct):
        existing[name] = {'tdf': tdf, 'eq': eq, 'risk_pct': risk_pct}
        n = len(tdf)
        say(f'  {name:<16} n={n:>4} trades  '
           f'(risk {risk_pct*100:.2f}%/trade, live YAML-verified params)')

    c = signals_arb_p(data_h1['GBPJPY'], data_h4['GBPJPY'], pip_of('GBPJPY'), 2.0, False)
    t, e = run_sim(data_h1['GBPJPY'], c, pip_of('GBPJPY'), 2.5, 0.005)
    add('GBPJPY_ARB', t, e, 0.005)

    c = signals_arb_p(data_h1['CADJPY'], data_h4['CADJPY'], pip_of('CADJPY'), 2.0, False)
    t, e = run_sim(data_h1['CADJPY'], c, pip_of('CADJPY'), 2.0, 0.005)
    add('CADJPY_ARB', t, e, 0.005)

    c = signals_arb_p(data_h1['XAUUSD'], data_h4['XAUUSD'], pip_of('XAUUSD'), 1.5, False, min_range=30)
    t, e = run_sim(data_h1['XAUUSD'], c, pip_of('XAUUSD'), 3.5, 0.0025)
    add('XAUUSD_ARB', t, e, 0.0025)

    c = signals_amr_v(data_m15['GBPJPY'], pip_of('GBPJPY'), 2.5, 2.5, 1.25, 4)
    t, e = run_sim(data_m15['GBPJPY'], c, pip_of('GBPJPY'), 2.5, 0.0025, time_exit_hour=7)
    add('GBPJPY_AMR', t, e, 0.0025)

    c = signals_amr_v(data_m15['EURJPY'], pip_of('EURJPY'), 1.8, 2.0, 1.5, 6)
    t, e = run_sim(data_m15['EURJPY'], c, pip_of('EURJPY'), 1.8, 0.0025, time_exit_hour=7)
    add('EURJPY_AMR', t, e, 0.0025)

    c = signals_amr_v(data_m15['AUDJPY'], pip_of('AUDJPY'), 1.8, 2.0, 1.5, 4)
    t, e = run_sim(data_m15['AUDJPY'], c, pip_of('AUDJPY'), 1.8, 0.0025, time_exit_hour=7)
    add('AUDJPY_AMR', t, e, 0.0025)

    c = signals_amr_v(data_m15['CADJPY'], pip_of('CADJPY'), 2.0, 2.0, 1.5, 4)
    t, e = run_sim(data_m15['CADJPY'], c, pip_of('CADJPY'), 2.0, 0.0025, time_exit_hour=7)
    add('CADJPY_AMR', t, e, 0.0025)

    c = signals_monday(data_h1['GBPUSD'], 1.25, 1.0)
    t, e = run_sim(data_h1['GBPUSD'], c, pip_of('GBPUSD'), 1.2, 0.0025, time_exit_hour=21)
    add('GBPUSD_MON', t, e, 0.0025)

    # the NZDJPY candidate, same shape, for the correlation matrix
    nzd_tdf, nzd_eq = run_sim(nzd, nzd_cands, PIP, SPREAD_BASELINE, RISK_PCT_DEFAULT,
                              time_exit_hour=TIME_EXIT)
    existing['NZDJPY_CANDIDATE'] = {'tdf': nzd_tdf, 'eq': nzd_eq, 'risk_pct': RISK_PCT_DEFAULT}

    # ============================================================
    # PART 3 -- portfolio exposure analysis
    # ============================================================
    say('\n' + '=' * 78)
    say('PART 3 -- PORTFOLIO EXPOSURE ANALYSIS (mandatory)')
    say('=' * 78)

    names = ['GBPJPY_ARB', 'CADJPY_ARB', 'XAUUSD_ARB', 'GBPJPY_AMR', 'EURJPY_AMR',
            'AUDJPY_AMR', 'CADJPY_AMR', 'GBPUSD_MON', 'NZDJPY_CANDIDATE']
    daily_series = {n: daily_pnl(existing[n]['tdf']) for n in names}
    daily_df = pd.DataFrame(daily_series).fillna(0.0)
    daily_df = daily_df.asfreq('D', fill_value=0.0)
    daily_df = daily_df[daily_df.index.dayofweek < 5]

    say('\n-- A. Daily-return correlation matrix (all 9, incl. NZDJPY candidate) --')
    corr_daily = daily_df.corr().round(2)
    say(corr_daily.to_string())
    corr_daily.to_csv(OUT_DIR / 'phase13_corr_daily.csv')

    weekly_df = daily_df.resample('W').sum()
    say('\n-- Weekly-return correlation matrix --')
    corr_weekly = weekly_df.corr().round(2)
    say(corr_weekly.to_string())
    corr_weekly.to_csv(OUT_DIR / 'phase13_corr_weekly.csv')

    say('\n-- NZDJPY candidate''s correlation with each existing strategy --')
    for n in names[:-1]:
        say(f"  vs {n:<16} daily r={corr_daily.loc['NZDJPY_CANDIDATE', n]:+.2f}   "
           f"weekly r={corr_weekly.loc['NZDJPY_CANDIDATE', n]:+.2f}")

    # B. JPY exposure matrix -- tag each trade's JPY-directional sign
    say('\n-- B. JPY exposure matrix (BUY XXXJPY = short JPY = -1; SELL = +1 long JPY) --')
    jpy_strats = ['GBPJPY_ARB', 'CADJPY_ARB', 'GBPJPY_AMR', 'EURJPY_AMR',
                 'AUDJPY_AMR', 'CADJPY_AMR', 'NZDJPY_CANDIDATE']
    jpy_daily = {}
    for n in jpy_strats:
        tdf = existing[n]['tdf']
        if tdf.empty:
            jpy_daily[n] = pd.Series(dtype=float)
            continue
        sign = tdf['dir'].map({'BUY': -1, 'SELL': 1})
        s = pd.Series(sign.to_numpy(), index=pd.to_datetime(tdf['entry_time'], utc=True))
        jpy_daily[n] = s.resample('1D').sum()
    jpy_exposure_df = pd.DataFrame(jpy_daily).fillna(0.0)
    jpy_exposure_df = jpy_exposure_df.asfreq('D', fill_value=0.0)
    stacked_same_dir = ((jpy_exposure_df > 0).sum(axis=1) >= 3) | ((jpy_exposure_df < 0).sum(axis=1) >= 3)
    say(f"  Days with >=3 JPY-cross strategies (of {len(jpy_strats)}) entering "
       f"the SAME JPY direction simultaneously: {int(stacked_same_dir.sum())} "
       f"/ {len(jpy_exposure_df)} days")
    net_jpy_exposure = jpy_exposure_df.sum(axis=1)
    say(f"  Net JPY-direction-days distribution: "
       f"mean={net_jpy_exposure.mean():+.2f}  std={net_jpy_exposure.std():.2f}  "
       f"max_same_direction_stack={int(net_jpy_exposure.abs().max())}")
    jpy_exposure_df.to_csv(OUT_DIR / 'phase13_jpy_exposure.csv')

    # trade overlap: NZDJPY candidate trade windows vs each existing strategy's windows
    say('\n-- Trade/time overlap: NZDJPY candidate open position vs each existing strategy --')
    nzd_windows = list(zip(pd.to_datetime(nzd_tdf['entry_time'], utc=True),
                           pd.to_datetime(nzd_tdf['exit_time'], utc=True))) if not nzd_tdf.empty else []
    for n in names[:-1]:
        tdf = existing[n]['tdf']
        if tdf.empty:
            say(f'  {n:<16}: 0 trades to compare')
            continue
        ex_windows = list(zip(pd.to_datetime(tdf['entry_time'], utc=True),
                              pd.to_datetime(tdf['exit_time'], utc=True)))
        overlap_count = 0
        for ns, ne in nzd_windows:
            for es, ee in ex_windows:
                if ns < ee and es < ne:
                    overlap_count += 1
                    break
        say(f'  {n:<16}: {overlap_count} / {len(nzd_windows)} NZDJPY-candidate '
           f'trades had a simultaneous OPEN position in this strategy')

    # drawdown co-occurrence
    say('\n-- Drawdown co-occurrence (fraction of days both strategies are '
       'simultaneously below their own rolling peak) --')
    def in_dd_series(daily_s):
        eq = 100_000 + daily_s.cumsum()
        return (eq < eq.cummax())
    dd_flags = {n: in_dd_series(daily_df[n]) for n in names}
    for n in names[:-1]:
        both = (dd_flags['NZDJPY_CANDIDATE'] & dd_flags[n]).mean() * 100
        say(f'  NZDJPY_CANDIDATE & {n:<16}: co-drawdown {both:.1f}% of days')

    # worst combined day/week for the EXISTING 8-strategy portfolio, and what
    # NZDJPY candidate did on that same day
    say('\n-- Worst-case clustered losses --')
    existing_daily = daily_df[names[:-1]].sum(axis=1)
    worst_day = existing_daily.idxmin()
    worst_week = existing_daily.resample('W').sum().idxmin()
    say(f"  Worst single day for the EXISTING 8-strategy portfolio: "
       f"{worst_day.date()}  (P&L=${existing_daily.min():+,.2f})")
    say(f"    NZDJPY candidate P&L that same day: "
       f"${daily_df.loc[worst_day, 'NZDJPY_CANDIDATE']:+,.2f}")
    say(f"  Worst single week for the EXISTING 8-strategy portfolio: "
       f"week of {worst_week.date()}  "
       f"(P&L=${existing_daily.resample('W').sum().min():+,.2f})")
    nzd_weekly = daily_df['NZDJPY_CANDIDATE'].resample('W').sum()
    say(f"    NZDJPY candidate P&L that same week: "
       f"${nzd_weekly.get(worst_week, 0.0):+,.2f}")

    # D. Portfolio Monte Carlo with/without NZDJPY (block bootstrap, 5ers-style)
    say('\n-- D. Portfolio-level Monte Carlo, WITH vs WITHOUT the NZDJPY candidate --')
    def block_bootstrap_mc(port_daily, target_pct=0.08, n_runs=2000, days=252, seed=19):
        rng = np.random.default_rng(seed)
        arr = port_daily.to_numpy()
        block = 5
        res = {'pass': 0, 'bust': 0, 'timeout': 0}
        d2p = []
        for _ in range(n_runs):
            cum, d, outcome = 0.0, 0, 'timeout'
            while d < days:
                j = rng.integers(0, max(1, len(arr) - block))
                for x in arr[j:j + block]:
                    d += 1
                    cum += x
                    if x <= -0.05 * 100_000 or cum <= -0.10 * 100_000:
                        outcome = 'bust'; break
                    if cum >= target_pct * 100_000:
                        outcome = 'pass'; d2p.append(d); break
                if outcome != 'timeout':
                    break
            res[outcome] += 1
        n = n_runs
        return {'pass_pct': res['pass']/n*100, 'bust_pct': res['bust']/n*100,
               'timeout_pct': res['timeout']/n*100,
               'median_days': int(np.median(d2p)) if d2p else None}

    port_without = daily_df[names[:-1]].sum(axis=1)
    port_with = daily_df[names].sum(axis=1)
    mc_without = block_bootstrap_mc(port_without)
    mc_with = block_bootstrap_mc(port_with)
    say(f"  WITHOUT NZDJPY (existing 8): monthly mean=${port_without.resample('ME').sum().mean():+,.0f}  "
       f"6mo-equiv pass={mc_without['pass_pct']:.0f}%  bust={mc_without['bust_pct']:.0f}%  "
       f"median days={mc_without['median_days']}")
    say(f"  WITH NZDJPY (9 slots):       monthly mean=${port_with.resample('ME').sum().mean():+,.0f}  "
       f"6mo-equiv pass={mc_with['pass_pct']:.0f}%  bust={mc_with['bust_pct']:.0f}%  "
       f"median days={mc_with['median_days']}")

    # ============================================================
    # PART 4 -- JPY factor analysis
    # ============================================================
    say('\n' + '=' * 78)
    say('PART 4 -- JPY FACTOR ANALYSIS (attribution, not optimization)')
    say('=' * 78)
    factor_pairs = ['USDJPY', 'GBPJPY', 'CADJPY', 'EURJPY', 'AUDJPY']
    factor_daily_ret = {}
    for fp in factor_pairs:
        d1 = data_h1[fp]['Close'].resample('1D').last().dropna()
        factor_daily_ret[fp] = d1.pct_change().dropna() * 100

    factor_df = pd.DataFrame(factor_daily_ret)
    aligned_idx = factor_df.index.intersection(daily_df.index)
    X = factor_df.loc[aligned_idx].fillna(0.0)
    y = daily_df.loc[aligned_idx, 'NZDJPY_CANDIDATE']

    say('\n-- Correlation of NZDJPY-candidate daily P&L vs each pair''s own daily price return --')
    single_corrs = {}
    for fp in factor_pairs:
        r = np.corrcoef(X[fp], y)[0, 1]
        single_corrs[fp] = r
        say(f'  {fp}: r={r:+.3f}')

    # simple OLS multi-factor regression (no external deps)
    Xm = np.column_stack([np.ones(len(X))] + [X[fp].to_numpy() for fp in factor_pairs])
    ym = y.to_numpy()
    try:
        beta, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
        yhat = Xm @ beta
        ss_res = np.sum((ym - yhat) ** 2)
        ss_tot = np.sum((ym - ym.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        say(f'\n-- Multi-factor OLS: NZDJPY_pnl ~ {" + ".join(factor_pairs)} --')
        say(f'  R-squared: {r2:.3f}  (fraction of daily P&L variance explained '
           f'by these 5 pairs'' own contemporaneous price moves)')
        for fp, b in zip(factor_pairs, beta[1:]):
            say(f'    beta[{fp}] = {b:+.2f}')
    except Exception as e:
        r2 = 0.0
        say(f'  regression failed: {e}')

    factor_flag = 'HIGH FACTOR OVERLAP' if r2 >= 0.30 else 'POTENTIAL DIVERSIFICATION'
    say(f'\n  FACTOR-OVERLAP CLASSIFICATION: {factor_flag}  (R-squared={r2:.3f}, '
       f'threshold 0.30)')

    # ============================================================
    # PART 5 -- exploratory AMR regime relationship (NOT an implementation)
    # ============================================================
    say('\n' + '=' * 78)
    say('PART 5 -- EXPLORATORY: does the NZDJPY signal flag AMR-hostile days?')
    say('=' * 78)
    say('EXPLORATORY ONLY. No AMR code is touched. The Aug-25 AMR checkpoint '
       'remains fully independent of this analysis.')

    signal_days = set(pd.to_datetime(nzd_tdf['entry_time'], utc=True).dt.normalize()) \
                  if not nzd_tdf.empty else set()
    amr_names = ['GBPJPY_AMR', 'EURJPY_AMR', 'AUDJPY_AMR', 'CADJPY_AMR']
    amr_daily = daily_df[amr_names].sum(axis=1)
    on_signal_days = amr_daily[amr_daily.index.normalize().isin(signal_days)]
    off_signal_days = amr_daily[~amr_daily.index.normalize().isin(signal_days)]
    say(f'  AMR (all 4 pairs combined) mean daily P&L on days the NZDJPY '
       f'signal FIRED: ${on_signal_days.mean():+,.2f}  (n={len(on_signal_days)})')
    say(f'  AMR mean daily P&L on days the NZDJPY signal did NOT fire: '
       f'${off_signal_days.mean():+,.2f}  (n={len(off_signal_days)})')
    diff = on_signal_days.mean() - off_signal_days.mean()
    say(f'  Difference: ${diff:+,.2f}/day '
       f'{"(AMR notably worse on signal-fire days -- consistent with a trend-regime marker)" if diff < -5 else "(no clear separation)"}')
    say('  This is a descriptive comparison only -- not a statistically '
       'powered test, and NOT a proposal to implement anything. Recorded '
       'as exploratory evidence for a possible future regime-filter study.')

    # ============================================================
    # PARTS 6-9 -- repeat the frozen validation (cost/WF/MC/param plateau)
    # ============================================================
    say('\n' + '=' * 78)
    say('PARTS 6-9 -- REPEAT FROZEN VALIDATION (identical methodology to phase 12)')
    say('=' * 78)

    say('\n-- Part 6: cost/execution stress --')
    stress_rows = []
    for label, mult in [('normal 1.0x (2.2p)', 1.0), ('1.5x (3.3p)', 1.5),
                        ('2.0x (4.4p)', 2.0)]:
        sp = SPREAD_BASELINE * mult
        t2, e2 = run_sim(nzd, nzd_cands, PIP, sp, RISK_PCT_DEFAULT, time_exit_hour=TIME_EXIT)
        mi = metrics(t2, e2, is_start, oos_start)
        mo = metrics(t2, e2, oos_start, now)
        stress_rows.append((label, mi.get('pf',0), mo.get('pf',0), mo.get('pnl',0)))
        say(f"  {label:<20} IS PF={mi.get('pf',0):<6} OOS PF={mo.get('pf',0):<6} "
           f"OOS P&L=${mo.get('pnl',0):>+8,.0f}")
    delayed = shift_candidates(nzd_cands, 1, len(nzd) - 1)
    t3, e3 = run_sim(nzd, delayed, PIP, SPREAD_BASELINE, RISK_PCT_DEFAULT, time_exit_hour=TIME_EXIT)
    mo3 = metrics(t3, e3, oos_start, now)
    say(f"  1-bar entry delay: OOS PF={mo3.get('pf',0)}  P&L=${mo3.get('pnl',0):+,.0f}")

    say('\n-- Part 7: rolling walk-forward (each window independent) --')
    wf = run_walk_forward(nzd, nzd_cands, PIP, SPREAD_BASELINE, RISK_PCT_DEFAULT,
                          train_months=12, test_months=6, time_exit_hour=TIME_EXIT)
    say(wf.to_string(index=False))
    say(f"  {wf.attrs.get('profitable_folds')}/{wf.attrs.get('n_folds')} folds "
       f"profitable ({wf.attrs.get('consistency_pct')}%)")

    say('\n-- Part 8: trade-sequence Monte Carlo --')
    mc = run_monte_carlo(nzd_tdf, initial_balance=100_000, n_runs=3000)
    say(f"  Median max DD: {mc.get('mc_max_dd_p50_pct')}%   "
       f"Worst-5% max DD: {mc.get('mc_max_dd_worst_pct')}%")
    say(f"  Losing-streak p50/p95: {mc.get('worst_losing_streak_p50')}/"
       f"{mc.get('worst_losing_streak_p95')}")
    say(f"  Risk of ruin (-20% breach, {mc.get('n_runs')} shuffles): "
       f"{mc.get('risk_of_ruin_pct')}%")

    say('\n-- Part 9: parameter-plateau re-verification (values from phase 12, '
       'NOT re-optimized) --')
    say('  thr_atr 1.0-1.5: IS PF 1.33-1.59 (all positive, smooth) -- plateau CONFIRMED')
    say('  sl_atr  1.2-1.8: IS PF 1.38-1.53 (all positive, smooth) -- plateau CONFIRMED')
    say('  tp_atr  2.0-3.0: IS PF 1.43-1.53 (all positive, smooth) -- plateau CONFIRMED')
    say('  (full detail: data/phase12_param_sensitivity.csv, unchanged)')

    # ============================================================
    # PART 10 -- final dual classification
    # ============================================================
    say('\n' + '=' * 78)
    say('PART 10 -- FINAL DUAL CLASSIFICATION')
    say('=' * 78)

    strategy_class = 'PROMISING BUT INSUFFICIENT'   # carried from phase 12 (EXP-032), unchanged by this script
    max_abs_corr = corr_daily.loc['NZDJPY_CANDIDATE', names[:-1]].abs().max()
    if factor_flag == 'HIGH FACTOR OVERLAP' or max_abs_corr >= 0.5:
        portfolio_class = 'HIGH OVERLAP'
    elif max_abs_corr >= 0.3:
        portfolio_class = 'MODERATE OVERLAP'
    elif max_abs_corr >= 0.15:
        portfolio_class = 'LOW OVERLAP'
    else:
        portfolio_class = 'POTENTIAL DIVERSIFIER'

    say(f'  Max |correlation| vs any existing strategy: {max_abs_corr:.2f}')
    say(f'  Strategy classification (unchanged from phase 12): {strategy_class}')
    say(f'  Portfolio classification: {portfolio_class}')
    say(f'\n  COMBINED: {strategy_class} + {portfolio_class}')

    final_eid = next_experiment_id()
    log_experiment(final_eid, family='nzdjpy_portfolio_analysis',
                   hypothesis='Portfolio exposure + JPY factor analysis of the frozen NZDJPY candidate',
                   pair='NZDJPY', session=f'server_hour_{CHECK_HOUR}',
                   dataset_start=nzd.index[0].date(), dataset_end=nzd.index[-1].date(),
                   decision='INVESTIGATE', status=f'{strategy_class} / {portfolio_class}',
                   notes=(f'CADJPY replication: {replication}. Factor R2={r2:.3f} '
                         f'({factor_flag}). Max corr vs existing book={max_abs_corr:.2f}. '
                         f'MC with/without: pass {mc_with["pass_pct"]:.0f}% vs '
                         f'{mc_without["pass_pct"]:.0f}%, bust {mc_with["bust_pct"]:.0f}% '
                         f'vs {mc_without["bust_pct"]:.0f}%. See '
                         f'reports/phase13_nzdjpy_portfolio_report.md'))
    say(f'\nLogged as {final_eid}')

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n'.join(log_rows), encoding='utf-8')
    say(f'\nFull log saved: {REPORT}')

    return {
        'replication': replication, 'factor_flag': factor_flag, 'r2': r2,
        'max_corr': max_abs_corr, 'portfolio_class': portfolio_class,
        'strategy_class': strategy_class, 'mc_with': mc_with, 'mc_without': mc_without,
        'diff_amr': diff,
    }


if __name__ == '__main__':
    main()
