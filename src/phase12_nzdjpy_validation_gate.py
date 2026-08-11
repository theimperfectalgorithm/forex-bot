"""
Forex Bot - Phase 12: NZDJPY cross-asset-momentum -- FROZEN VALIDATION GATE
=============================================================================
This is a VALIDATION run, not a discovery or optimization run. The
strategy rules and parameters below are FROZEN at the exact values
that produced the phase-10/10b result. Nothing here selects, tunes, or
re-picks a parameter based on any result computed in this script.
Every signal function is IMPORTED from phase10_jpy_london_ny.py
unchanged (not re-transcribed) so there is zero risk of the frozen
spec silently drifting from the code that actually produced the
original finding.

=========================== FROZEN SPECIFICATION ===========================
Instrument traded:    NZDJPY (H1)
Reference/proxy:      USDJPY (H1) -- read-only signal input, NEVER traded
Entry trigger:        at the H1 bar whose SERVER hour == 15 (check_hour),
                       compute USDJPY's move (pips) from that day's
                       server-hour-7 bar open to this bar's close, and
                       that day's USDJPY ATR(14,H1) as of hour 7.
                       If |move| >= 1.25 x that ATR (thr_atr):
                         move > 0 -> BUY NZDJPY (JPY broadly weak)
                         move < 0 -> SELL NZDJPY (JPY broadly strong)
Entry price:           NZDJPY close of that same hour-15 bar (no
                       modeled slippage in the frozen baseline -- entry-
                       delay/slippage is tested SEPARATELY as a stress
                       scenario, section F, never folded into baseline)
Stop-loss:             1.5 x NZDJPY's own ATR(14,H1) at the signal bar
Take-profit:           2.5 x NZDJPY's own ATR(14,H1) at the signal bar
                       (min TP floor: max(3 pips, 2x spread), essentially
                       never binding given ATR-scaled distances)
Exit priority:         SL/TP intrabar (SL wins if both touched same
                       bar) OR a hard time-exit at server hour >= 21,
                       OR Friday server hour >= 20 force-close.
                       CONSEQUENCE OF DESIGN: entries only ever occur at
                       hour 15 and are force-flat by hour 21 THE SAME
                       DAY, so in practice this strategy carries ZERO
                       overnight or weekend exposure -- Friday-close
                       essentially never engages.
Position sizing:       0.5% of current (compounding) balance risked per
                       trade (RISK_PCT_DEFAULT), lots derived as
                       risk_usd / sl_pips. NOTE: this offline research
                       engine does not model a broker MAX_LOT clamp or
                       minimum-lot rounding (the live bot's execution
                       layer does -- see phase9's $5K sizing work); this
                       is a known simplification of the research
                       backtest vs. live fills, disclosed here.
Spread assumption:     2.2 pips fixed, deducted from every trade
                       (NZDJPY's assumed retail spread throughout this
                       project). Conservative/stress spread scenarios
                       are tested separately in section F.
Slippage assumption:   NONE in the frozen baseline. Entry-delay is
                       tested as a separate stress scenario (section F),
                       never folded into the frozen baseline itself.
News filter:           NONE applied in this offline backtest. The live
                       system's news-blackout gate (core/news_calendar.py)
                       is a LIVE execution-time risk control, not part
                       of this historical simulation -- disclosed, not
                       modeled here.
Trading frequency cap: one trade per day (enforced by the engine's
                       last_trade_date dedup; also structurally true
                       since the signal only checks a single bar/day).
Session:               entries confined to a single fixed server hour
                       (15) inside the project's established London/NY
                       server-hour convention (07-21).
Timeframe:              H1 for both the traded instrument and the proxy.
Universe:               NZDJPY only. No other pair is part of this
                       frozen candidate.
Dataset:                36 months H1, MT5 live fetch. IS = first 24
                       months (used for the ORIGINAL discovery/selection
                       in phase10/10b). OOS = final 12 months (NEVER
                       used for any selection decision -- touched here
                       for confirmation only, exactly once).
==============================================================================

Requirements: MetaTrader 5 OPEN and LOGGED IN.
Outputs: reports/phase12_nzdjpy_validation_report.md,
         experiments/experiments.csv (appended, multiple rows),
         data/phase12_*.csv (per-section detail tables)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy_matrix_backtest import (
    MONTHS, IS_MONTHS, RISK_PCT_DEFAULT, windowed_atr, run_sim, metrics,
    REPO_ROOT,
)
from phase10_jpy_london_ny import (
    fetch_h1, build_usdjpy_proxy, signals_xmomentum, NY_END,
)
from walk_forward import run_walk_forward
from monte_carlo import run_monte_carlo
from research_ledger import log_experiment, next_experiment_id

PAIR = 'NZDJPY'
PIP = 0.01
SPREAD_BASELINE = 2.2

# ---- FROZEN PARAMETERS -- do not change anywhere below this line ----------
CHECK_HOUR = 15
THR_ATR    = 1.25
SL_ATR     = 1.5
TP_ATR     = 2.5
TIME_EXIT  = NY_END   # 21, server hour
# -----------------------------------------------------------------------

OUT_DIR = REPO_ROOT / 'data'
REPORT  = REPO_ROOT / 'reports' / 'phase12_nzdjpy_validation_report.md'

FROZEN_SPEC = {
    'instrument': PAIR, 'proxy': 'USDJPY', 'timeframe': 'H1',
    'check_hour_server': CHECK_HOUR, 'thr_atr': THR_ATR,
    'sl_atr_mult': SL_ATR, 'tp_atr_mult': TP_ATR,
    'time_exit_server_hour': TIME_EXIT, 'friday_close_server_hour': 20,
    'risk_pct_per_trade': RISK_PCT_DEFAULT, 'spread_pips': SPREAD_BASELINE,
    'slippage_pips': 0, 'news_filter': 'none (offline backtest)',
    'max_trades_per_day': 1, 'sizing': 'compounding balance x risk_pct / sl_pips',
    'lot_clamp_modeled': False,
    'entry_price': 'signal-bar close, no delay', 'exit_priority': 'SL > TP > TimeExit(21) > FridayClose(Fri 20:00)',
    'dataset_months': MONTHS, 'is_months': IS_MONTHS, 'oos_months': MONTHS - IS_MONTHS,
    'source_experiments': 'phase10 EXP set + phase10b refinement (best cell)',
}

log_rows = []
def say(s=''):
    print(s)
    log_rows.append(s)


# ── MFE / MAE replay ─────────────────────────────────────────────────────

def compute_mfe_mae(tdf: pd.DataFrame, h1: pd.DataFrame, pip: float) -> pd.DataFrame:
    """For each closed trade, replay the bars it was open and compute the
    Maximum Favorable / Adverse Excursion in R-multiples (1R = that
    trade's own sl_pips). Pure analysis -- never used to alter entries,
    exits, or the frozen signal logic."""
    if tdf.empty:
        return pd.DataFrame()
    highs = h1['High']; lows = h1['Low']
    out = []
    for _, tr in tdf.iterrows():
        window = h1[(h1.index >= tr['entry_time']) & (h1.index <= tr['exit_time'])]
        if window.empty or tr['sl_pips'] <= 0:
            continue
        if tr['dir'] == 'BUY':
            mfe_pips = (window['High'].max() - tr['entry']) / pip
            mae_pips = (tr['entry'] - window['Low'].min()) / pip
        else:
            mfe_pips = (tr['entry'] - window['Low'].min()) / pip
            mae_pips = (window['High'].max() - tr['entry']) / pip
        out.append({
            'entry_time': tr['entry_time'], 'dir': tr['dir'],
            'sl_pips': tr['sl_pips'], 'pnl': tr['pnl'],
            'mfe_r': round(mfe_pips / tr['sl_pips'], 2),
            'mae_r': round(mae_pips / tr['sl_pips'], 2),
            'result_r': round((tr['pips']) / tr['sl_pips'], 2) if 'pips' in tr else np.nan,
        })
    return pd.DataFrame(out)


# ── entry-delay stress wrapper ───────────────────────────────────────────

def shift_candidates(cands: list, n_bars: int, max_i: int) -> list:
    return [(min(i + n_bars, max_i), d, sl, tp) for i, d, sl, tp in cands]


# ── random-direction permutation null model ──────────────────────────────

def permutation_test(h1, real_cands, pip, spread, n_perm=1000, seed=11):
    """Same entry bars, same ATR-based SL/TP construction -- but the
    BUY/SELL direction is a coin flip instead of the real USDJPY-proxy
    direction. Isolates whether the direction-selection logic adds
    value beyond generic session-time NZDJPY trading with these SL/TP
    mechanics. Reports the percentile rank of the REAL strategy's PF
    and total P&L against this null distribution -- a permutation-test
    style significance check."""
    rng = np.random.default_rng(seed)
    real_tdf, real_eq = run_sim(h1, real_cands, pip, spread, RISK_PCT_DEFAULT,
                                time_exit_hour=TIME_EXIT)
    real_pf = real_tdf[real_tdf.pnl > 0].pnl.sum() / abs(real_tdf[real_tdf.pnl <= 0].pnl.sum()) \
        if not real_tdf.empty and (real_tdf.pnl <= 0).any() else np.inf
    real_pnl = real_tdf.pnl.sum() if not real_tdf.empty else 0.0

    null_pfs, null_pnls = [], []
    for _ in range(n_perm):
        shuffled = [(i, rng.choice(['BUY', 'SELL']), sl, tp)
                   for i, _d, sl, tp in real_cands]
        tdf, _ = run_sim(h1, shuffled, pip, spread, RISK_PCT_DEFAULT,
                         time_exit_hour=TIME_EXIT)
        if tdf.empty:
            continue
        gl = abs(tdf[tdf.pnl <= 0].pnl.sum())
        pf = tdf[tdf.pnl > 0].pnl.sum() / gl if gl > 0 else np.inf
        null_pfs.append(pf)
        null_pnls.append(tdf.pnl.sum())

    null_pfs = np.array(null_pfs)
    null_pnls = np.array(null_pnls)
    pf_percentile = float((null_pfs < real_pf).mean() * 100) if len(null_pfs) else np.nan
    pnl_percentile = float((null_pnls < real_pnl).mean() * 100) if len(null_pnls) else np.nan
    return {
        'real_pf': round(real_pf, 3), 'real_pnl': round(real_pnl, 2),
        'null_pf_mean': round(float(null_pfs.mean()), 3),
        'null_pf_p50': round(float(np.median(null_pfs)), 3),
        'null_pf_p95': round(float(np.percentile(null_pfs, 95)), 3),
        'null_pnl_mean': round(float(null_pnls.mean()), 2),
        'real_pf_percentile_vs_null': round(pf_percentile, 1),
        'real_pnl_percentile_vs_null': round(pnl_percentile, 1),
        'n_perm': len(null_pfs),
    }


def main():
    now = pd.Timestamp.now(tz='UTC')
    oos_start = now - pd.DateOffset(months=MONTHS - IS_MONTHS)
    is_start  = now - pd.DateOffset(months=MONTHS)
    is_mid    = is_start + pd.DateOffset(months=IS_MONTHS // 2)

    say('=' * 78)
    say('PHASE 12: NZDJPY FROZEN VALIDATION GATE')
    say('=' * 78)

    # ---- log the frozen spec FIRST, before any computation below ----
    spec_eid = next_experiment_id()
    log_experiment(spec_eid, family='nzdjpy_validation_gate',
                   hypothesis='FROZEN SPEC RECORD -- see params_json for the '
                              'complete, immutable specification under test',
                   pair=PAIR, session=f'server_hour_{CHECK_HOUR}', timeframe='H1',
                   params=FROZEN_SPEC, decision='INVESTIGATE', status='VALIDATION',
                   notes='Frozen before any validation test in this script ran.')
    say(f'\nFrozen spec logged as {spec_eid} in experiments/experiments.csv')
    for k, v in FROZEN_SPEC.items():
        say(f'  {k}: {v}')

    say('\nFetching NZDJPY + USDJPY H1 ...')
    nzd = fetch_h1(PAIR)
    usd = fetch_h1('USDJPY')
    say(f'  NZDJPY: {len(nzd):,} bars ({nzd.index[0].date()} -> {nzd.index[-1].date()})')

    usdjpy_move, usdjpy_atr = build_usdjpy_proxy(usd, 0.01)
    cands = signals_xmomentum(nzd, PIP, SPREAD_BASELINE, usdjpy_move, usdjpy_atr,
                              CHECK_HOUR, THR_ATR, sl_atr=SL_ATR, tp_atr=TP_ATR)
    say(f'  Frozen candidate signal count (full 36mo): {len(cands)}')

    tdf, eq = run_sim(nzd, cands, PIP, SPREAD_BASELINE, RISK_PCT_DEFAULT,
                      time_exit_hour=TIME_EXIT)
    tdf.to_csv(OUT_DIR / 'phase12_frozen_trades.csv', index=False)

    # ================= A/B/C: IS / internal-split / OOS =================
    say('\n' + '=' * 78)
    say('A/B/C. IN-SAMPLE / INTERNAL SPLIT / OUT-OF-SAMPLE')
    say('=' * 78)
    say('NOTE: only ~3 years of history exist, so a true 3-way independent '
       'split (dev/validation/final-OOS) is not honestly available. IS-early '
       'and IS-late below are BOTH inside the original discovery window (not '
       'independent) -- an internal consistency check only. OOS is the '
       'genuinely held-out final 12 months, touched exactly once, here.')

    m_is_early = metrics(tdf, eq, is_start, is_mid)
    m_is_late  = metrics(tdf, eq, is_mid, oos_start)
    m_is_full  = metrics(tdf, eq, is_start, oos_start)
    m_oos      = metrics(tdf, eq, oos_start, now)

    for label, m in [('IS-early (internal, first 12mo of IS)', m_is_early),
                     ('IS-late (internal, second 12mo of IS)', m_is_late),
                     ('IS-FULL (24mo, original selection window)', m_is_full),
                     ('OOS (final 12mo, held out, touched once)', m_oos)]:
        say(f'  {label}: n={m.get("trades",0):>3}  PF={m.get("pf",0):<6}  '
           f'DD={m.get("max_dd_pct",0):>5.2f}%  pm={m.get("prof_months_pct",0):>5.1f}%  '
           f'P&L=${m.get("pnl",0):>+9,.2f}  WR={m.get("win_rate",0)}%')

    # ================= D: rolling walk-forward =================
    say('\n' + '=' * 78)
    say('D. ROLLING WALK-FORWARD (each window reported independently)')
    say('=' * 78)
    wf = run_walk_forward(nzd, cands, PIP, SPREAD_BASELINE, RISK_PCT_DEFAULT,
                          train_months=12, test_months=6, time_exit_hour=TIME_EXIT)
    say(wf.to_string(index=False))
    say(f"  {wf.attrs.get('data_constraint_note', '')}")
    say(f"  Consistency: {wf.attrs.get('profitable_folds')}/{wf.attrs.get('n_folds')} "
       f"folds profitable ({wf.attrs.get('consistency_pct')}%)")
    wf.to_csv(OUT_DIR / 'phase12_walk_forward.csv', index=False)

    # ================= E: trade-sequence Monte Carlo =================
    say('\n' + '=' * 78)
    say('E. TRADE-SEQUENCE MONTE CARLO (order-shuffle, drawdown/streak only -- '
       'not meaningless shuffled-P&L "confidence intervals")')
    say('=' * 78)
    mc = run_monte_carlo(tdf, initial_balance=100_000, n_runs=3000)
    hist_equity = 100_000 + tdf['pnl'].cumsum()
    hist_peak = hist_equity.cummax()
    hist_dd_pct = ((hist_equity - hist_peak) / hist_peak * 100).min()
    losing = (tdf['pnl'] <= 0).to_numpy()
    hist_streak = cur = 0
    for L in losing:
        cur = cur + 1 if L else 0
        hist_streak = max(hist_streak, cur)
    say(f'  Historical (actual, single path) max DD: {hist_dd_pct:.2f}%')
    say(f'  Historical (actual) longest losing streak: {hist_streak} trades')
    say(f"  MC median max DD:     {mc.get('mc_max_dd_p50_pct')}%")
    say(f"  MC worst-5% max DD:   {mc.get('mc_max_dd_worst_pct')}%")
    say(f"  MC losing-streak p50/p95: {mc.get('worst_losing_streak_p50')}/"
       f"{mc.get('worst_losing_streak_p95')}")
    say(f"  Risk of ruin (breach -20% at any point, {mc.get('n_runs')} shuffles): "
       f"{mc.get('risk_of_ruin_pct')}%")

    # ================= F: cost / execution stress =================
    say('\n' + '=' * 78)
    say('F/12. COST + EXECUTION STRESS (spread multiples, entry delay)')
    say('=' * 78)
    stress_rows = []
    for label, mult in [('normal (1.0x = 2.2p)', 1.0), ('conservative (1.5x = 3.3p)', 1.5),
                        ('stress (2.0x = 4.4p)', 2.0), ('extreme (3.0x = 6.6p)', 3.0)]:
        sp = SPREAD_BASELINE * mult
        t2, e2 = run_sim(nzd, cands, PIP, sp, RISK_PCT_DEFAULT, time_exit_hour=TIME_EXIT)
        mi = metrics(t2, e2, is_start, oos_start)
        mo = metrics(t2, e2, oos_start, now)
        stress_rows.append({'scenario': label, 'spread_pips': sp,
                           'is_pf': mi.get('pf', 0), 'is_pnl': mi.get('pnl', 0),
                           'oos_pf': mo.get('pf', 0), 'oos_pnl': mo.get('pnl', 0)})
        say(f"  {label:<28} IS PF={mi.get('pf',0):<6} P&L=${mi.get('pnl',0):>+9,.0f}  "
           f"OOS PF={mo.get('pf',0):<6} P&L=${mo.get('pnl',0):>+9,.0f}")

    delayed = shift_candidates(cands, 1, len(nzd) - 1)
    t3, e3 = run_sim(nzd, delayed, PIP, SPREAD_BASELINE, RISK_PCT_DEFAULT,
                     time_exit_hour=TIME_EXIT)
    mi3 = metrics(t3, e3, is_start, oos_start)
    mo3 = metrics(t3, e3, oos_start, now)
    say(f"  1-bar entry delay (execute next bar's close instead of signal "
       f"bar's close):")
    say(f"    IS PF={mi3.get('pf',0):<6} P&L=${mi3.get('pnl',0):>+9,.0f}  "
       f"OOS PF={mo3.get('pf',0):<6} P&L=${mo3.get('pnl',0):>+9,.0f}")
    pd.DataFrame(stress_rows).to_csv(OUT_DIR / 'phase12_cost_stress.csv', index=False)

    # ================= G/11: parameter sensitivity (no re-selection) =================
    say('\n' + '=' * 78)
    say('G/11. PARAMETER SENSITIVITY -- neighbors reported, NONE selected')
    say('=' * 78)
    sens_rows = []

    def eval_variant(ch, thr, sl_a, tp_a):
        c = signals_xmomentum(nzd, PIP, SPREAD_BASELINE, usdjpy_move, usdjpy_atr,
                              ch, thr, sl_atr=sl_a, tp_atr=tp_a)
        t, e = run_sim(nzd, c, PIP, SPREAD_BASELINE, RISK_PCT_DEFAULT, time_exit_hour=TIME_EXIT)
        mi = metrics(t, e, is_start, oos_start)
        mo = metrics(t, e, oos_start, now)
        return mi, mo

    say('\n  -- check_hour neighbors (thr/sl/tp held at frozen values) --')
    for ch in (13, 14, 15, 16):
        mi, mo = eval_variant(ch, THR_ATR, SL_ATR, TP_ATR)
        tag = ' <== FROZEN' if ch == CHECK_HOUR else ''
        say(f"    check_hour={ch}: IS PF={mi.get('pf',0):<6} DD={mi.get('max_dd_pct',0):>5.2f}%  "
           f"OOS PF={mo.get('pf',0):<6} P&L=${mo.get('pnl',0):>+8,.0f}{tag}")
        sens_rows.append({'param': 'check_hour', 'value': ch, **mi, 'oos_pf': mo.get('pf',0), 'oos_pnl': mo.get('pnl',0)})

    say('\n  -- thr_atr neighbors (+/-10%/20%, others held at frozen values) --')
    for thr in (THR_ATR*0.8, THR_ATR*0.9, THR_ATR, THR_ATR*1.1, THR_ATR*1.2):
        mi, mo = eval_variant(CHECK_HOUR, round(thr, 3), SL_ATR, TP_ATR)
        tag = ' <== FROZEN' if abs(thr - THR_ATR) < 1e-9 else ''
        say(f"    thr_atr={thr:.3f}: IS PF={mi.get('pf',0):<6} DD={mi.get('max_dd_pct',0):>5.2f}%  "
           f"OOS PF={mo.get('pf',0):<6} P&L=${mo.get('pnl',0):>+8,.0f}{tag}")
        sens_rows.append({'param': 'thr_atr', 'value': round(thr,3), **mi, 'oos_pf': mo.get('pf',0), 'oos_pnl': mo.get('pnl',0)})

    say('\n  -- sl_atr neighbors (+/-10%/20%, others held at frozen values) --')
    for sl_a in (SL_ATR*0.8, SL_ATR*0.9, SL_ATR, SL_ATR*1.1, SL_ATR*1.2):
        mi, mo = eval_variant(CHECK_HOUR, THR_ATR, round(sl_a, 3), TP_ATR)
        tag = ' <== FROZEN' if abs(sl_a - SL_ATR) < 1e-9 else ''
        say(f"    sl_atr={sl_a:.3f}: IS PF={mi.get('pf',0):<6} DD={mi.get('max_dd_pct',0):>5.2f}%  "
           f"OOS PF={mo.get('pf',0):<6} P&L=${mo.get('pnl',0):>+8,.0f}{tag}")
        sens_rows.append({'param': 'sl_atr', 'value': round(sl_a,3), **mi, 'oos_pf': mo.get('pf',0), 'oos_pnl': mo.get('pnl',0)})

    say('\n  -- tp_atr neighbors (+/-10%/20%, others held at frozen values) --')
    for tp_a in (TP_ATR*0.8, TP_ATR*0.9, TP_ATR, TP_ATR*1.1, TP_ATR*1.2):
        mi, mo = eval_variant(CHECK_HOUR, THR_ATR, SL_ATR, round(tp_a, 3))
        tag = ' <== FROZEN' if abs(tp_a - TP_ATR) < 1e-9 else ''
        say(f"    tp_atr={tp_a:.3f}: IS PF={mi.get('pf',0):<6} DD={mi.get('max_dd_pct',0):>5.2f}%  "
           f"OOS PF={mo.get('pf',0):<6} P&L=${mo.get('pnl',0):>+8,.0f}{tag}")
        sens_rows.append({'param': 'tp_atr', 'value': round(tp_a,3), **mi, 'oos_pf': mo.get('pf',0), 'oos_pnl': mo.get('pnl',0)})

    pd.DataFrame(sens_rows).to_csv(OUT_DIR / 'phase12_param_sensitivity.csv', index=False)

    # ================= H: year-by-year =================
    say('\n' + '=' * 78)
    say('H. YEAR-BY-YEAR (concentration check)')
    say('=' * 78)
    tdf2 = tdf.copy()
    tdf2['entry_time'] = pd.to_datetime(tdf2['entry_time'], utc=True)
    tdf2['year'] = tdf2['entry_time'].dt.year
    year_rows = []
    for yr, g in tdf2.groupby('year'):
        w = g[g.pnl > 0]; gl = abs(g[g.pnl <= 0].pnl.sum())
        pf = w.pnl.sum()/gl if gl > 0 else np.inf
        year_rows.append({'year': yr, 'trades': len(g), 'wr': round(len(w)/len(g)*100,1),
                          'pf': round(pf,2), 'pnl': round(g.pnl.sum(),2)})
        say(f"  {yr}: n={len(g):>3}  WR={len(w)/len(g)*100:>5.1f}%  PF={pf:<6.2f}  "
           f"P&L=${g.pnl.sum():>+9,.2f}")
    total_pnl = tdf2.pnl.sum()
    best_year_pnl = max(r['pnl'] for r in year_rows) if year_rows else 0
    concentration = (best_year_pnl / total_pnl * 100) if total_pnl > 0 else 0
    say(f"  Single best year contributes {concentration:.1f}% of total P&L "
       f"{'-- CONCENTRATION FLAG' if concentration > 60 else '(broad)'}")
    pd.DataFrame(year_rows).to_csv(OUT_DIR / 'phase12_year_by_year.csv', index=False)

    # ================= I: monthly =================
    say('\n' + '=' * 78)
    say('I. MONTHLY PERFORMANCE')
    say('=' * 78)
    monthly = tdf2.set_index('entry_time')['pnl'].resample('ME').agg(['count', 'sum'])
    monthly.columns = ['trades', 'pnl']
    say(monthly.to_string())
    monthly.to_csv(OUT_DIR / 'phase12_monthly.csv')
    zero_months = (monthly['trades'] == 0).sum()
    say(f"  Months with zero trades: {zero_months} / {len(monthly)}")

    # ================= J: drawdown analysis =================
    say('\n' + '=' * 78)
    say('J. DRAWDOWN ANALYSIS (historical single path)')
    say('=' * 78)
    dd_series = (hist_equity - hist_peak)
    in_dd = dd_series < 0
    # duration: count consecutive trades below prior peak
    run_len, lengths = 0, []
    for v in in_dd:
        if v:
            run_len += 1
        else:
            if run_len > 0:
                lengths.append(run_len)
            run_len = 0
    if run_len > 0:
        lengths.append(run_len)
    avg_dd_duration = np.mean(lengths) if lengths else 0
    max_dd_duration = max(lengths) if lengths else 0
    recovery_factor = total_pnl / abs(hist_equity.min() - 100_000) if hist_equity.min() < 100_000 else np.inf
    say(f"  Max drawdown: {hist_dd_pct:.2f}%  (${hist_peak.max()-hist_equity.min():,.2f} at worst)")
    say(f"  Avg drawdown duration: {avg_dd_duration:.1f} trades")
    say(f"  Max drawdown duration: {max_dd_duration} trades")
    say(f"  Recovery factor (net P&L / max peak-to-trough $): {recovery_factor:.2f}")

    # ================= K: MFE/MAE =================
    say('\n' + '=' * 78)
    say('K. MFE / MAE ANALYSIS')
    say('=' * 78)
    mfe_mae = compute_mfe_mae(tdf, nzd, PIP)
    if not mfe_mae.empty:
        mfe_mae.to_csv(OUT_DIR / 'phase12_mfe_mae.csv', index=False)
        say(f"  Median MFE: {mfe_mae['mfe_r'].median():.2f}R   "
           f"Median MAE: {mfe_mae['mae_r'].median():.2f}R")
        say(f"  Winners: median MFE captured = "
           f"{(mfe_mae[mfe_mae.pnl>0]['result_r'] / mfe_mae[mfe_mae.pnl>0]['mfe_r']).median()*100:.0f}% "
           f"of the best available excursion")
        say(f"  Losers: median MAE vs 1R stop = "
           f"{mfe_mae[mfe_mae.pnl<=0]['mae_r'].median():.2f}R "
           f"(should be ~1.0R if stops are the binding constraint)")

    # ================= L: losing-streak (historical, already reported in E) ===
    say('\n' + '=' * 78)
    say('L. LOSING-STREAK ANALYSIS (see section E for historical + MC figures)')
    say('=' * 78)
    say(f"  Historical longest losing streak: {hist_streak} consecutive trades")
    say(f"  Monte Carlo p50/p95 across {mc.get('n_runs')} order-shuffles: "
       f"{mc.get('worst_losing_streak_p50')}/{mc.get('worst_losing_streak_p95')}")

    # ================= M: trade frequency =================
    say('\n' + '=' * 78)
    say('M. TRADE FREQUENCY')
    say('=' * 78)
    n_days = (tdf2['entry_time'].max() - tdf2['entry_time'].min()).days
    say(f"  Total trades: {len(tdf2)} over {n_days} days "
       f"({len(tdf2)/(n_days/7):.2f}/week, {len(tdf2)/(n_days/30):.2f}/month)")
    say(f"  Entry hour distribution: {tdf2['entry_time'].dt.hour.value_counts().to_dict()} "
       f"(should be 100% at hour {CHECK_HOUR} by construction)")
    gaps = tdf2['entry_time'].sort_values().diff().dt.days.dropna()
    say(f"  Days between trades: median={gaps.median():.1f}  max={gaps.max():.0f}")

    # ================= benchmark: random-direction permutation test =========
    say('\n' + '=' * 78)
    say('BENCHMARK: random-direction null model (permutation test)')
    say('=' * 78)
    say('Buy-and-hold is not a meaningful null model for an intraday FX '
       'directional strategy. The correct null here is: same entry bars, '
       'same ATR-based SL/TP construction, but the BUY/SELL direction is a '
       'coin flip instead of the USDJPY-proxy signal. This isolates whether '
       'the DIRECTION-PICKING logic adds value beyond generic session-time '
       'trading with these SL/TP mechanics.')
    perm = permutation_test(nzd, cands, PIP, SPREAD_BASELINE, n_perm=1000)
    for k, v in perm.items():
        say(f'  {k}: {v}')
    say(f"\n  Interpretation: the real strategy's profit factor exceeded "
       f"{perm['real_pf_percentile_vs_null']:.1f}% of {perm['n_perm']} "
       f"random-direction shuffles of the SAME entry bars/SL/TP construction.")

    # ================= scorecard =================
    say('\n' + '=' * 78)
    say('FINAL VALIDATION SCORECARD')
    say('=' * 78)

    def pf(m): return m.get('pf', 0)
    is_ok = (m_is_full.get('trades',0) >= 30 and pf(m_is_full) > 1.3
            and m_is_full.get('max_dd_pct',99) < 8 and m_is_full.get('prof_months_pct',0) >= 60)
    oos_ok = m_oos.get('trades',0) >= 10 and m_oos.get('pnl',0) > 0 and m_oos.get('max_dd_pct',99) < 8
    wf_ok = wf.attrs.get('consistency_pct', 0) >= 75
    mc_ok = mc.get('risk_of_ruin_pct', 100) < 5
    cost_ok = stress_rows[2]['is_pf'] > 1.0 and stress_rows[2]['oos_pf'] > 1.0   # 2x spread
    sens_df = pd.DataFrame(sens_rows)
    thr_pfs = sens_df[sens_df.param=='thr_atr']['pf']
    sl_pfs  = sens_df[sens_df.param=='sl_atr']['pf']
    tp_pfs  = sens_df[sens_df.param=='tp_atr']['pf']
    plateau_ok = all((s.max() - s.min()) < 0.6 for s in (thr_pfs, sl_pfs, tp_pfs) if len(s))
    year_ok = concentration < 60
    perm_ok = perm['real_pf_percentile_vs_null'] >= 90
    dd_ok = hist_dd_pct > -10

    scorecard = [
        ('In-sample (24mo)', f"PF={pf(m_is_full)} DD={m_is_full.get('max_dd_pct',0)}% pm={m_is_full.get('prof_months_pct',0)}%", is_ok),
        ('Out-of-sample (12mo, held out)', f"PF={pf(m_oos)} P&L=${m_oos.get('pnl',0):+,.0f}", oos_ok),
        ('Walk-forward', f"{wf.attrs.get('profitable_folds')}/{wf.attrs.get('n_folds')} folds profitable", wf_ok),
        ('Monte Carlo (risk of ruin)', f"{mc.get('risk_of_ruin_pct')}%", mc_ok),
        ('Cost stress (2x spread)', f"IS PF={stress_rows[2]['is_pf']} OOS PF={stress_rows[2]['oos_pf']}", cost_ok),
        ('Parameter sensitivity (plateau)', f"thr/sl/tp PF range < 0.6 spread: {plateau_ok}", plateau_ok),
        ('Year consistency', f"best year = {concentration:.0f}% of total P&L", year_ok),
        ('Direction vs random (permutation)', f"beats {perm['real_pf_percentile_vs_null']:.0f}% of random-direction shuffles", perm_ok),
        ('Drawdown', f"{hist_dd_pct:.2f}%", dd_ok),
    ]
    say(f"{'Test':<38} {'Result':<50} {'Pass/Fail'}")
    for name, result, ok in scorecard:
        say(f"{name:<38} {result:<50} {'PASS' if ok else 'FAIL'}")

    n_pass = sum(1 for _, _, ok in scorecard if ok)
    if is_ok and oos_ok and wf_ok and mc_ok and perm_ok and n_pass >= 7:
        status = 'VALIDATED'
    elif oos_ok and n_pass >= 5:
        status = 'PROMISING BUT INSUFFICIENT'
    elif is_ok and not oos_ok:
        status = 'OVERFIT_RISK'
    else:
        status = 'FAILED'
    say(f"\n{n_pass}/{len(scorecard)} checks pass.  FINAL STATUS: {status}")

    # ---- log final summary experiment ----
    final_eid = next_experiment_id()
    log_experiment(final_eid, family='nzdjpy_validation_gate',
                   hypothesis='Full frozen validation gate result summary',
                   pair=PAIR, session=f'server_hour_{CHECK_HOUR}',
                   dataset_start=nzd.index[0].date(), dataset_end=nzd.index[-1].date(),
                   is_metrics=m_is_full, oos_metrics=m_oos,
                   wf_folds=wf.attrs.get('n_folds'),
                   wf_consistency_pct=wf.attrs.get('consistency_pct'),
                   mc_risk_of_ruin_pct=mc.get('risk_of_ruin_pct'),
                   decision=('KEEP' if status == 'VALIDATED' else 'INVESTIGATE'),
                   status=status,
                   notes=f'{n_pass}/{len(scorecard)} scorecard checks passed; '
                        f'ref spec {spec_eid}')
    say(f'\nFinal summary logged as {final_eid}')

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n'.join(log_rows), encoding='utf-8')
    say(f'\nFull log saved: {REPORT}')


if __name__ == '__main__':
    main()
