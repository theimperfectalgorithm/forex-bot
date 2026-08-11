"""
Forex Bot - Phase 20: Volatility Regime x Existing-Strategy Diagnostics
============================================================================
DIAGNOSTIC ONLY. Does not create, optimize, or modify any strategy. Does
not change the demo account. Reconstructs all 8 live demo strategies from
their exact frozen live parameters (same reconstruction as phase13/19b)
and asks: does entry-time volatility regime (ATR percentile, the measure
phase19 found to be the more predictive volatility variable) explain
differences in these strategies' historical performance?

Primary volatility variable (frozen, reusing the project's existing
methodology unchanged since phase14/16/17/19): ATR = Wilder(14), 66-bar
rolling window (windowed_atr, this project's standard implementation),
percentile-ranked over the pair's full available history
(pd.Series(atr).rank(pct=True)). Evaluated at the trade's own entry bar
-- windowed_atr's own docstring guarantees this is the value the live
class would compute if that bar were the most recent closed bar, i.e.
genuinely available at entry, no lookahead.

Fixed regimes (frozen before any results are examined):
  LOW         = ATR percentile [0, 25)
  NORMAL-LOW  = ATR percentile [25, 50)
  NORMAL-HIGH = ATR percentile [50, 75)
  HIGH        = ATR percentile [75, 100]

Requirements: MetaTrader 5 OPEN and LOGGED IN (or CSV cache).
Output: reports/phase20_diagnostics_log.txt, data/phase20_trades.csv
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import data_loader
from strategy_matrix_backtest import run_sim, windowed_atr, REPO_ROOT
from phase2_meanrev_arb_search import signals_arb_p
from phase3b_amr_jpy_refine import signals_amr_v
from phase8_monday_validation import signals_monday
from phase15_downmove_reversion_baseline import ASIAN, LONDON, OVERLAP, NY, session_of_hour

MONTHS = 36
REGIME_BINS = [0, 0.25, 0.50, 0.75, 1.0001]
REGIME_LABELS = ['LOW', 'NORMAL-LOW', 'NORMAL-HIGH', 'HIGH']
MIN_SAMPLE = 20   # below this, a regime cell is flagged insufficient, not judged

LOG: list[str] = []


def say(msg=''):
    print(msg)
    LOG.append(str(msg))


def fetch(pair, tf):
    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=MONTHS * 30 + 30)
    return data_loader.get_bars(pair, tf, date_from, date_to)


def pip_of(pair):
    return 0.01 if (pair.endswith('JPY') or pair == 'XAUUSD') else 0.0001


def atr_pctile_series(df: pd.DataFrame, pip: float) -> pd.Series:
    highs, lows, closes = df['High'].to_numpy(), df['Low'].to_numpy(), df['Close'].to_numpy()
    atr = windowed_atr(highs, lows, closes, 14, 66) / pip
    return pd.Series(atr, index=df.index).rank(pct=True)


def mfe_mae(df: pd.DataFrame, pip: float, entry_time, exit_time, direction, entry_px, atr_at_entry):
    """Simple post-hoc MFE/MAE in ATR units, using only bars between entry
    and exit (inclusive) -- diagnostic only, does not affect any trade
    decision."""
    win = df.loc[entry_time:exit_time]
    if win.empty or atr_at_entry <= 0 or np.isnan(atr_at_entry):
        return np.nan, np.nan
    if direction == 'BUY':
        mfe = (win['High'].max() - entry_px) / pip / atr_at_entry
        mae = (entry_px - win['Low'].min()) / pip / atr_at_entry
    else:
        mfe = (entry_px - win['Low'].min()) / pip / atr_at_entry
        mae = (win['High'].max() - entry_px) / pip / atr_at_entry
    return mfe, mae


def build_strategy_trades(name, pair, tf, cands_df, signal_fn, run_kwargs, session_hour_ref='entry') -> pd.DataFrame:
    """Reconstruct one strategy's trades and attach entry-time regime info."""
    df = cands_df
    pip = pip_of(pair)
    pctile = atr_pctile_series(df, pip)
    atr_raw = windowed_atr(df['High'].to_numpy(), df['Low'].to_numpy(), df['Close'].to_numpy(), 14, 66) / pip
    atr_series = pd.Series(atr_raw, index=df.index)

    cands = signal_fn()
    tdf, _ = run_sim(df, cands, pip, **run_kwargs)
    if tdf.empty:
        return pd.DataFrame()

    tdf = tdf.copy()
    tdf['strategy'] = name
    tdf['pair'] = pair
    tdf['atr_pctile_entry'] = tdf['entry_time'].map(pctile)
    tdf['atr_at_entry'] = tdf['entry_time'].map(atr_series)
    tdf['regime'] = pd.cut(tdf['atr_pctile_entry'], bins=REGIME_BINS, labels=REGIME_LABELS, right=False)
    tdf['session'] = tdf['entry_time'].apply(lambda t: session_of_hour(t.hour))
    tdf['year'] = tdf['entry_time'].dt.year
    # R-multiple: realized pips / risked pips (sl_pips) -- reconstructs R without needing usd_per_pip
    with np.errstate(divide='ignore', invalid='ignore'):
        tdf['r_multiple'] = np.where(tdf['sl_pips'] > 0, tdf['pips'] / tdf['sl_pips'], np.nan)

    mfes, maes = [], []
    for _, row in tdf.iterrows():
        mfe, mae = mfe_mae(df, pip, row['entry_time'], row['exit_time'], row['dir'], row['entry'], row['atr_at_entry'])
        mfes.append(mfe); maes.append(mae)
    tdf['mfe_atr'] = mfes
    tdf['mae_atr'] = maes
    return tdf


def summarize(tdf: pd.DataFrame) -> dict:
    if tdf.empty or len(tdf) < MIN_SAMPLE:
        return dict(n=len(tdf), insufficient=True)
    wins = tdf[tdf.pnl > 0]['pnl'].sum()
    losses = -tdf[tdf.pnl < 0]['pnl'].sum()
    pf = wins / losses if losses > 0 else np.nan
    cum = tdf['pnl'].cumsum()
    running_max = cum.cummax()
    dd = (cum - running_max).min()
    # losing streak
    is_loss = (tdf['pnl'] < 0).to_numpy()
    max_streak, cur = 0, 0
    for x in is_loss:
        cur = cur + 1 if x else 0
        max_streak = max(max_streak, cur)
    return dict(n=len(tdf), insufficient=False, win_rate=float((tdf.pnl > 0).mean()),
                avg_r=float(tdf['r_multiple'].mean()), median_r=float(tdf['r_multiple'].median()),
                expectancy=float(tdf['pnl'].mean()), pf=float(pf), max_dd=float(dd),
                avg_mfe=float(tdf['mfe_atr'].mean()), avg_mae=float(tdf['mae_atr'].mean()),
                max_losing_streak=int(max_streak), total_pnl=float(tdf['pnl'].sum()))


def report_by_regime(tdf: pd.DataFrame, label: str):
    say(f'\n-- {label}: by fixed regime --')
    rows = []
    for reg in REGIME_LABELS:
        sub = tdf[tdf.regime == reg]
        s = summarize(sub)
        rows.append(dict(regime=reg, **s))
    out = pd.DataFrame(rows)
    say(out.to_string(index=False))
    for _, r in out.iterrows():
        if r.get('insufficient'):
            say(f'  ** {r["regime"]}: insufficient sample (n={r["n"]}, need >={MIN_SAMPLE}) -- not judged **')
    return out


def report_continuous(tdf: pd.DataFrame, label: str):
    say(f'\n-- {label}: continuous relationship (quintile bins of ATR percentile) --')
    if len(tdf) < 40:
        say('  insufficient total sample for quintile analysis')
        return
    tdf = tdf.copy()
    tdf['q'] = pd.qcut(tdf['atr_pctile_entry'], 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
    out = tdf.groupby('q', observed=True).agg(n=('pnl', 'size'), expectancy=('pnl', 'mean'),
                                               win_rate=('pnl', lambda s: (s > 0).mean()))
    say(out.to_string())
    monotonic = out['expectancy'].is_monotonic_increasing or out['expectancy'].is_monotonic_decreasing
    say(f'  Monotonic expectancy across quintiles: {monotonic}')


def report_by_year(tdf: pd.DataFrame, label: str):
    say(f'\n-- {label}: by year (HIGH vs LOW+NORMAL-LOW regime) --')
    rows = []
    for yr in [2023, 2024, 2025, 2026]:
        sub = tdf[tdf.year == yr]
        hi = sub[sub.regime == 'HIGH']
        lo = sub[sub.regime.isin(['LOW', 'NORMAL-LOW'])]
        if len(hi) < MIN_SAMPLE or len(lo) < MIN_SAMPLE:
            rows.append(dict(year=yr, n_hi=len(hi), n_lo=len(lo), note='insufficient'))
            continue
        rows.append(dict(year=yr, n_hi=len(hi), exp_hi=hi['pnl'].mean(),
                          n_lo=len(lo), exp_lo=lo['pnl'].mean()))
    say(pd.DataFrame(rows).to_string(index=False))


def report_by_session(tdf: pd.DataFrame, label: str):
    say(f'\n-- {label}: by actual trading session --')
    sess_counts = tdf['session'].value_counts()
    say(f'  Session distribution: {sess_counts.to_dict()}')
    if sess_counts.nunique() == 1 or len(sess_counts) == 1:
        say('  Strategy trades in a single session window -- session-consistency check not applicable '
            '(this is expected, not a filter being imposed).')
        return
    rows = []
    for sess, sub in tdf.groupby('session'):
        hi = sub[sub.regime == 'HIGH']
        lo = sub[sub.regime.isin(['LOW', 'NORMAL-LOW'])]
        if len(hi) < MIN_SAMPLE or len(lo) < MIN_SAMPLE:
            continue
        rows.append(dict(session=sess, n_hi=len(hi), exp_hi=hi['pnl'].mean(),
                          n_lo=len(lo), exp_lo=lo['pnl'].mean()))
    if rows:
        say(pd.DataFrame(rows).to_string(index=False))
    else:
        say('  insufficient per-session sample for a regime split')


def main():
    say('=' * 90)
    say('PHASE 20 -- VOLATILITY REGIME x EXISTING-STRATEGY DIAGNOSTICS (observational, no strategy change)')
    say('=' * 90)
    say('Primary volatility variable (frozen): ATR percentile = rank_pct(windowed_atr(14,66)) at the')
    say('trade\'s own entry bar -- reused unchanged from phase14/16/17/19, not re-derived here.')
    say(f'Fixed regimes (frozen before any results examined): {REGIME_LABELS} = {REGIME_BINS}')
    say(f'Minimum sample size for a judged cell: {MIN_SAMPLE} trades; below that, flagged insufficient.')

    all_trades = []

    # ---- ARB family ----
    say('\n' + '=' * 90)
    say('RECONSTRUCTING STRATEGIES')
    say('=' * 90)
    for pair, tp_mult, use_h4, min_range in [('GBPJPY', 2.0, False, 10), ('CADJPY', 2.0, False, 10),
                                              ('XAUUSD', 1.5, False, 30)]:
        try:
            h1 = fetch(pair, 'H1')
            h4 = fetch(pair, 'H4')
        except Exception as e:
            say(f'{pair}_ARB: SKIP ({e})'); continue
        pip = pip_of(pair)
        name = f'{pair}_ARB'
        tdf = build_strategy_trades(name, pair, 'H1', h1,
                                     lambda h1=h1, h4=h4, pip=pip, tp_mult=tp_mult, use_h4=use_h4, min_range=min_range:
                                     signals_arb_p(h1, h4, pip, tp_mult, use_h4, min_range=min_range),
                                     dict(spread_pips=2.0, risk_pct=0.005))
        say(f'{name}: {len(tdf)} trades reconstructed')
        all_trades.append(tdf)

    # ---- AMR family ----
    for pair, z_thr, sl_mult, end_hour in [('GBPJPY', 2.5, 1.25, 4), ('EURJPY', 2.0, 1.5, 6),
                                            ('AUDJPY', 2.0, 1.5, 4), ('CADJPY', 2.0, 1.5, 4)]:
        try:
            m15 = fetch(pair, 'M15')
        except Exception as e:
            say(f'{pair}_AMR: SKIP ({e})'); continue
        pip = pip_of(pair)
        name = f'{pair}_AMR'
        tdf = build_strategy_trades(name, pair, 'M15', m15,
                                     lambda m15=m15, pip=pip, z_thr=z_thr, sl_mult=sl_mult, end_hour=end_hour:
                                     signals_amr_v(m15, pip, 2.0, z_thr, sl_mult, end_hour),
                                     dict(spread_pips=2.0, risk_pct=0.0025))
        say(f'{name}: {len(tdf)} trades reconstructed')
        all_trades.append(tdf)

    # ---- Monday Drift ----
    pair = 'GBPUSD'
    try:
        h1 = fetch(pair, 'H1')
        pip = pip_of(pair)
        name = 'GBPUSD_MONDAY'
        tdf = build_strategy_trades(name, pair, 'H1', h1,
                                     lambda h1=h1: signals_monday(h1, 1.25, 1.0),
                                     dict(spread_pips=1.2, risk_pct=0.0025, time_exit_hour=21))
        say(f'{name}: {len(tdf)} trades reconstructed')
        all_trades.append(tdf)
    except Exception as e:
        say(f'{pair}_MONDAY: SKIP ({e})')

    all_trades = [t for t in all_trades if not t.empty]
    combined = pd.concat(all_trades, ignore_index=True)
    out_dir = REPO_ROOT / 'data'
    combined.to_csv(out_dir / 'phase20_trades.csv', index=False)

    # ---- Parts 4/5/6/7/8/9/10: per-strategy deep dive ----
    for name, tdf in [(t['strategy'].iloc[0], t) for t in all_trades]:
        say('\n' + '=' * 90)
        say(f'STRATEGY: {name}')
        say('=' * 90)
        report_by_regime(tdf, name)
        report_continuous(tdf, name)
        report_by_year(tdf, name)
        report_by_session(tdf, name)
        say(f'  Contribution to combined P&L: ${tdf["pnl"].sum():+,.2f} '
            f'({100*tdf["pnl"].sum()/combined["pnl"].sum():.1f}% of total)' if combined["pnl"].sum() != 0 else '')

    # ---- Part 11: portfolio view ----
    say('\n' + '=' * 90)
    say('PART 11 -- PORTFOLIO VIEW (combined across all 8 strategies)')
    say('=' * 90)
    daily = combined.copy()
    daily['date'] = daily['entry_time'].dt.date
    day_regime = daily.groupby('date')['atr_pctile_entry'].mean()
    day_regime_bin = pd.cut(day_regime, bins=REGIME_BINS, labels=REGIME_LABELS, right=False)
    day_pnl = daily.groupby('date')['pnl'].sum()
    port = pd.DataFrame({'pnl': day_pnl, 'regime': day_regime_bin})
    rows = []
    for reg in REGIME_LABELS:
        sub = port[port.regime == reg]
        if len(sub) < MIN_SAMPLE:
            rows.append(dict(regime=reg, n_days=len(sub), note='insufficient'))
            continue
        cum = sub['pnl'].cumsum()
        dd = (cum - cum.cummax()).min()
        rows.append(dict(regime=reg, n_days=len(sub), mean_daily_pnl=sub['pnl'].mean(),
                          std_daily_pnl=sub['pnl'].std(), worst_day=sub['pnl'].min(), max_dd=dd))
    say(pd.DataFrame(rows).to_string(index=False))

    # ---- Part 12: clustering / correlation ----
    say('\n' + '=' * 90)
    say('PART 12 -- CLUSTERED-LOSS ANALYSIS')
    say('=' * 90)
    daily_losers = daily[daily.pnl < 0].groupby('date')['strategy'].nunique()
    daily_losers_by_regime = pd.DataFrame({'n_losing_strategies': daily_losers}).join(
        pd.DataFrame({'regime': day_regime_bin}), how='left')
    rows = []
    for reg in REGIME_LABELS:
        sub = daily_losers_by_regime[daily_losers_by_regime.regime == reg]
        if len(sub) < MIN_SAMPLE:
            continue
        rows.append(dict(regime=reg, n_days_with_any_loss=len(sub),
                          mean_simultaneous_losers=sub['n_losing_strategies'].mean(),
                          pct_days_2plus_losers=(sub['n_losing_strategies'] >= 2).mean()))
    say(pd.DataFrame(rows).to_string(index=False))

    report_path = REPO_ROOT / 'reports' / 'phase20_diagnostics_log.txt'
    report_path.write_text('\n'.join(LOG), encoding='utf-8')
    say(f'\nFull log written to {report_path}')


if __name__ == '__main__':
    main()
