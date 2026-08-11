"""
Forex Bot - Phase 21: AMR Regime Mechanism Research
========================================================================
Follow-up to phase 20 (AUDJPY/CADJPY AMR show cross-year-confirmed
volatility-regime deterioration; GBPJPY/EURJPY AMR mixed). Central
question: is volatility itself the causal variable, or a proxy for
trend/persistence? MECHANISM RESEARCH ONLY -- does not modify AMR, add
a filter, or change any parameter. The 2026-08-25 AMR checkpoint is
unaffected.

All regime/explanatory variables are computed from data strictly at or
before each trade's own entry bar (bar i, the signal-confirmation bar --
run_sim enters at that bar's own close, so bar i's own state is legitimate
entry-time information, unchanged convention from phase16/17/20). MFE/MAE
(Part 11) is the only place post-entry data is used, and only for
path/outcome analysis, never for regime classification.

Requirements: MetaTrader 5 OPEN and LOGGED IN (or CSV cache).
Output: reports/phase21_mechanism_log.txt, data/phase21_amr_trades.csv
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
from phase3b_amr_jpy_refine import signals_amr_v

MONTHS = 36
MIN_SAMPLE = 20
TREND_WINDOW = 20     # efficiency-ratio / persistence window, matches phase16's precedent
SHORT_WINDOW = 8       # short-horizon slope/vol-change window, matches phase17's precedent

AMR_PAIRS = [
    ('GBPJPY', 2.5, 1.25, 4),
    ('EURJPY', 2.0, 1.5, 6),
    ('AUDJPY', 2.0, 1.5, 4),
    ('CADJPY', 2.0, 1.5, 4),
]

LOG: list[str] = []


def say(msg=''):
    print(msg)
    LOG.append(str(msg))


def fetch(pair, tf):
    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=MONTHS * 30 + 30)
    return data_loader.get_bars(pair, tf, date_from, date_to)


def efficiency_ratio(closes: np.ndarray, window: int) -> np.ndarray:
    """Kaufman efficiency ratio (unchanged from phase16): |net move| / sum
    |bar-to-bar moves| over a trailing window. 1.0=trend, ~0=noise/MR."""
    net = np.abs(closes - np.roll(closes, window))
    net[:window] = np.nan
    diffs = np.abs(np.diff(closes, prepend=closes[0]))
    roll_sum = pd.Series(diffs).rolling(window).sum().to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        er = net / roll_sum
    er[:window] = np.nan
    return er


def build_amr_trades(pair, z_thr, sl_mult, end_hour) -> pd.DataFrame:
    m15 = fetch(pair, 'M15')
    pip = 0.01 if pair.endswith('JPY') else 0.0001
    closes, highs, lows = m15['Close'].to_numpy(), m15['High'].to_numpy(), m15['Low'].to_numpy()
    atr = windowed_atr(highs, lows, closes, 14, 66) / pip
    atr_pctile = pd.Series(atr, index=m15.index).rank(pct=True)

    # ---- Part 3 pre-specified trend/persistence variables (entry-time, backward-only) ----
    diffs = pd.Series(closes).diff()
    up = (diffs > 0).astype(float)
    persistence_20 = up.rolling(TREND_WINDOW, min_periods=TREND_WINDOW).mean().to_numpy()
    er_20 = efficiency_ratio(closes, TREND_WINDOW)
    with np.errstate(divide='ignore', invalid='ignore'):
        ret_20_atr = (pd.Series(closes) - pd.Series(closes).shift(TREND_WINDOW)).to_numpy() / np.where(atr > 0, atr, np.nan)
        ret_8_atr = (pd.Series(closes) - pd.Series(closes).shift(SHORT_WINDOW)).to_numpy() / np.where(atr > 0, atr, np.nan)
    recent_high_20 = pd.Series(highs).rolling(TREND_WINDOW, min_periods=TREND_WINDOW).max().to_numpy()
    recent_low_20 = pd.Series(lows).rolling(TREND_WINDOW, min_periods=TREND_WINDOW).min().to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        dist_from_high_atr = (recent_high_20 - closes) / np.where(atr > 0, atr, np.nan)
        dist_from_low_atr = (closes - recent_low_20) / np.where(atr > 0, atr, np.nan)
        pos_in_range_20 = (closes - recent_low_20) / np.maximum(recent_high_20 - recent_low_20, 1e-9)

    # ---- Part 6: pre-entry volatility transition (purely backward-looking) ----
    bar_ret = pd.Series(closes).diff().to_numpy() / pip
    vol_recent_4 = pd.Series(bar_ret).rolling(4).std().to_numpy()      # bars [i-3..i]
    vol_prior_4 = pd.Series(bar_ret).shift(4).rolling(4).std().to_numpy()  # bars [i-7..i-4]
    with np.errstate(divide='ignore', invalid='ignore'):
        vol_change_ratio = vol_recent_4 / vol_prior_4

    cands = signals_amr_v(m15, pip, 2.0, z_thr, sl_mult, end_hour)
    tdf, _ = run_sim(m15, cands, pip, 2.0, 0.0025)
    if tdf.empty:
        return pd.DataFrame()

    idx_map = {t: i for i, t in enumerate(m15.index)}
    entry_idx = tdf['entry_time'].map(idx_map)

    tdf = tdf.copy()
    tdf['pair'] = pair
    tdf['entry_idx'] = entry_idx
    tdf['atr_pctile'] = entry_idx.map(pd.Series(atr_pctile.to_numpy()))
    tdf['persistence_20'] = entry_idx.map(pd.Series(persistence_20))
    tdf['efficiency_ratio_20'] = entry_idx.map(pd.Series(er_20))
    tdf['ret_20_atr'] = entry_idx.map(pd.Series(ret_20_atr))
    tdf['ret_8_atr'] = entry_idx.map(pd.Series(ret_8_atr))
    tdf['dist_from_high_atr'] = entry_idx.map(pd.Series(dist_from_high_atr))
    tdf['dist_from_low_atr'] = entry_idx.map(pd.Series(dist_from_low_atr))
    tdf['pos_in_range_20'] = entry_idx.map(pd.Series(pos_in_range_20))
    tdf['vol_change_ratio'] = entry_idx.map(pd.Series(vol_change_ratio))
    tdf['year'] = tdf['entry_time'].dt.year
    with np.errstate(divide='ignore', invalid='ignore'):
        tdf['r_multiple'] = np.where(tdf['sl_pips'] > 0, tdf['pips'] / tdf['sl_pips'], np.nan)

    # regime bins (frozen, unchanged from phase20)
    tdf['vol_regime'] = pd.cut(tdf['atr_pctile'], [0, 0.25, 0.5, 0.75, 1.0001],
                                labels=['LOW', 'NORMAL-LOW', 'NORMAL-HIGH', 'HIGH'], right=False)
    tdf['vol_tercile'] = pd.cut(tdf['atr_pctile'], [0, 1/3, 2/3, 1.0001],
                                 labels=['LOW', 'NORMAL', 'HIGH'], right=False)
    # trend tercile from efficiency ratio, computed on THIS strategy's OWN entry population
    # (fixed terciles of the population being studied, not searched for best split)
    er_valid = tdf['efficiency_ratio_20'].dropna()
    er_terciles = er_valid.quantile([1/3, 2/3]).to_numpy() if len(er_valid) >= 30 else [np.nan, np.nan]
    tdf['trend_tercile'] = pd.cut(tdf['efficiency_ratio_20'], [-np.inf, er_terciles[0], er_terciles[1], np.inf],
                                   labels=['LOW TREND', 'NORMAL TREND', 'HIGH TREND'])

    # MFE/MAE (Part 11, post-entry outcome/path analysis only)
    mfes, maes, hit_time_to_worst = [], [], []
    for _, row in tdf.iterrows():
        win = m15.loc[row['entry_time']:row['exit_time']]
        a = atr[idx_map.get(row['entry_time'], 0)] if row['entry_time'] in idx_map else np.nan
        if win.empty or not a or np.isnan(a) or a <= 0:
            mfes.append(np.nan); maes.append(np.nan); continue
        if row['dir'] == 'BUY':
            mfe = (win['High'].max() - row['entry']) / pip / a
            mae = (row['entry'] - win['Low'].min()) / pip / a
        else:
            mfe = (row['entry'] - win['Low'].min()) / pip / a
            mae = (win['High'].max() - row['entry']) / pip / a
        mfes.append(mfe); maes.append(mae)
    tdf['mfe_atr'] = mfes
    tdf['mae_atr'] = maes
    return tdf


def cell_stats(sub: pd.DataFrame) -> dict:
    if len(sub) < MIN_SAMPLE:
        return dict(n=len(sub), insufficient=True)
    wins = sub[sub.pnl > 0]['pnl'].sum()
    losses = -sub[sub.pnl < 0]['pnl'].sum()
    pf = wins / losses if losses > 0 else np.nan
    return dict(n=len(sub), insufficient=False, win_rate=float((sub.pnl > 0).mean()),
                avg_r=float(sub['r_multiple'].mean()), median_r=float(sub['r_multiple'].median()),
                expectancy=float(sub['pnl'].mean()), pf=float(pf))


def main():
    say('=' * 90)
    say('PHASE 21 -- AMR REGIME MECHANISM RESEARCH (volatility cause vs. trend proxy)')
    say('Mechanism research only. AMR is NOT modified. No filter created. No threshold optimized.')
    say(f'Trend/persistence window: {TREND_WINDOW} bars. Short-horizon window: {SHORT_WINDOW} bars.')
    say(f'Minimum sample per judged cell: {MIN_SAMPLE} trades.')
    say('=' * 90)

    all_amr = {}
    for pair, z_thr, sl_mult, end_hour in AMR_PAIRS:
        say(f'\nReconstructing {pair} AMR ...')
        tdf = build_amr_trades(pair, z_thr, sl_mult, end_hour)
        all_amr[pair] = tdf
        say(f'  {len(tdf)} trades, {tdf["dir"].value_counts().to_dict() if not tdf.empty else {}}')

    combined = pd.concat([t.assign(pair=p) for p, t in all_amr.items() if not t.empty], ignore_index=True)
    combined.to_csv(REPO_ROOT / 'data' / 'phase21_amr_trades.csv', index=False)

    # ---- PART 2: reconfirm volatility relationship ----
    say('\n' + '=' * 90)
    say('PART 2 -- RECONFIRM VOLATILITY RELATIONSHIP (same definition as phase20)')
    say('=' * 90)
    for pair, tdf in all_amr.items():
        say(f'\n-- {pair} AMR: by vol_regime --')
        rows = [dict(regime=r, **cell_stats(tdf[tdf.vol_regime == r])) for r in ['LOW', 'NORMAL-LOW', 'NORMAL-HIGH', 'HIGH']]
        say(pd.DataFrame(rows).to_string(index=False))

    # ---- PART 3/4: trend/persistence vs volatility ----
    say('\n' + '=' * 90)
    say('PART 3/4 -- TREND/PERSISTENCE VARIABLES VS VOLATILITY')
    say('=' * 90)
    trend_vars = ['persistence_20', 'efficiency_ratio_20', 'ret_20_atr', 'ret_8_atr', 'dist_from_high_atr']
    for pair, tdf in all_amr.items():
        say(f'\n-- {pair} AMR --')
        say('  By trend tercile (efficiency_ratio_20):')
        rows = [dict(trend=t, **cell_stats(tdf[tdf.trend_tercile == t])) for t in ['LOW TREND', 'NORMAL TREND', 'HIGH TREND']]
        say('  ' + pd.DataFrame(rows).to_string(index=False).replace('\n', '\n  '))

        # conditioning check: does vol regime still separate performance WITHIN each trend tercile?
        say('  Volatility effect WITHIN each trend tercile (does vol remain predictive after conditioning on trend?):')
        for t in ['LOW TREND', 'NORMAL TREND', 'HIGH TREND']:
            sub = tdf[tdf.trend_tercile == t]
            lo = sub[sub.vol_tercile == 'LOW']
            hi = sub[sub.vol_tercile == 'HIGH']
            if len(lo) >= MIN_SAMPLE and len(hi) >= MIN_SAMPLE:
                say(f'    {t}: vol_LOW exp={lo["pnl"].mean():+.2f} (n={len(lo)})  '
                    f'vol_HIGH exp={hi["pnl"].mean():+.2f} (n={len(hi)})  diff={hi["pnl"].mean()-lo["pnl"].mean():+.2f}')
            else:
                say(f'    {t}: insufficient sample (n_lo={len(lo)}, n_hi={len(hi)})')

        say('  Trend effect WITHIN each volatility regime (does trend remain predictive after conditioning on vol?):')
        for v in ['LOW', 'NORMAL', 'HIGH']:
            sub = tdf[tdf.vol_tercile == v]
            lo = sub[sub.trend_tercile == 'LOW TREND']
            hi = sub[sub.trend_tercile == 'HIGH TREND']
            if len(lo) >= MIN_SAMPLE and len(hi) >= MIN_SAMPLE:
                say(f'    vol_{v}: trend_LOW exp={lo["pnl"].mean():+.2f} (n={len(lo)})  '
                    f'trend_HIGH exp={hi["pnl"].mean():+.2f} (n={len(hi)})  diff={hi["pnl"].mean()-lo["pnl"].mean():+.2f}')
            else:
                say(f'    vol_{v}: insufficient sample (n_lo={len(lo)}, n_hi={len(hi)})')

    # ---- PART 5: 2D regime matrix ----
    say('\n' + '=' * 90)
    say('PART 5 -- 2D VOLATILITY x TREND REGIME MATRIX')
    say('=' * 90)
    for pair, tdf in all_amr.items():
        say(f'\n-- {pair} AMR: expectancy matrix (rows=vol tercile, cols=trend tercile) --')
        rows = []
        for v in ['LOW', 'NORMAL', 'HIGH']:
            row = {'vol_tercile': v}
            for t in ['LOW TREND', 'NORMAL TREND', 'HIGH TREND']:
                sub = tdf[(tdf.vol_tercile == v) & (tdf.trend_tercile == t)]
                s = cell_stats(sub)
                row[f'{t}_n'] = s['n']
                row[f'{t}_exp'] = s.get('expectancy', np.nan)
                row[f'{t}_pf'] = s.get('pf', np.nan)
            rows.append(row)
        say(pd.DataFrame(rows).to_string(index=False))

    # ---- PART 6: volatility transition ----
    say('\n' + '=' * 90)
    say('PART 6 -- VOLATILITY TRANSITION (already-high vol vs. rapidly-expanding-into-entry vol)')
    say('=' * 90)
    say('vol_change_ratio = std(bar returns, last 4 bars before entry) / std(bar returns, 4 bars before that)')
    say('-- purely backward-looking, no post-entry data.')
    for pair, tdf in all_amr.items():
        sub = tdf.dropna(subset=['vol_change_ratio'])
        if len(sub) < 40:
            say(f'{pair} AMR: insufficient sample'); continue
        sub = sub.copy()
        sub['expansion'] = sub['vol_change_ratio'] >= sub['vol_change_ratio'].median()
        rows = []
        for label, mask in [('EXPANDING (>=median)', sub['expansion']), ('CONTRACTING/STABLE (<median)', ~sub['expansion'])]:
            rows.append(dict(group=label, **cell_stats(sub[mask])))
        say(f'-- {pair} AMR --')
        say(pd.DataFrame(rows).to_string(index=False))
        # also cross with already-high ATR level
        hi_atr = sub[sub.vol_tercile == 'HIGH']
        if len(hi_atr) >= 40:
            rows2 = []
            hi_atr = hi_atr.copy()
            hi_atr['expansion'] = hi_atr['vol_change_ratio'] >= hi_atr['vol_change_ratio'].median()
            for label, mask in [('HIGH-ATR + EXPANDING', hi_atr['expansion']), ('HIGH-ATR + STABLE', ~hi_atr['expansion'])]:
                rows2.append(dict(group=label, **cell_stats(hi_atr[mask])))
            say('  within HIGH-ATR-regime trades only:')
            say('  ' + pd.DataFrame(rows2).to_string(index=False).replace('\n', '\n  '))

    # ---- PART 7: market location ----
    say('\n' + '=' * 90)
    say('PART 7 -- MARKET LOCATION (position within recent range)')
    say('=' * 90)
    for pair, tdf in all_amr.items():
        sub = tdf.dropna(subset=['pos_in_range_20'])
        if len(sub) < 60:
            say(f'{pair} AMR: insufficient sample'); continue
        sub = sub.copy()
        sub['loc_bin'] = pd.cut(sub['pos_in_range_20'], [0, 1/3, 2/3, 1.0001], labels=['NEAR_LOW', 'MID_RANGE', 'NEAR_HIGH'])
        rows = [dict(location=l, **cell_stats(sub[sub.loc_bin == l])) for l in ['NEAR_LOW', 'MID_RANGE', 'NEAR_HIGH']]
        say(f'-- {pair} AMR --')
        say(pd.DataFrame(rows).to_string(index=False))

    # ---- PART 8: directional asymmetry ----
    say('\n' + '=' * 90)
    say('PART 8 -- DIRECTIONAL ASYMMETRY (BUY vs SELL, by vol regime)')
    say('=' * 90)
    for pair, tdf in all_amr.items():
        say(f'\n-- {pair} AMR --')
        rows = []
        for d in ['BUY', 'SELL']:
            for v in ['LOW', 'NORMAL', 'HIGH']:
                sub = tdf[(tdf.dir == d) & (tdf.vol_tercile == v)]
                rows.append(dict(direction=d, vol_tercile=v, **cell_stats(sub)))
        say(pd.DataFrame(rows).to_string(index=False))

    # ---- PART 9: year consistency ----
    say('\n' + '=' * 90)
    say('PART 9 -- YEAR CONSISTENCY (HIGH vs LOW vol tercile expectancy, by year)')
    say('=' * 90)
    for pair, tdf in all_amr.items():
        say(f'\n-- {pair} AMR --')
        rows = []
        for yr in [2023, 2024, 2025, 2026]:
            sub = tdf[tdf.year == yr]
            hi = sub[sub.vol_tercile == 'HIGH']
            lo = sub[sub.vol_tercile == 'LOW']
            if len(hi) < MIN_SAMPLE or len(lo) < MIN_SAMPLE:
                rows.append(dict(year=yr, n_hi=len(hi), n_lo=len(lo), note='insufficient'))
                continue
            rows.append(dict(year=yr, n_hi=len(hi), exp_hi=hi['pnl'].mean(), n_lo=len(lo), exp_lo=lo['pnl'].mean()))
        say(pd.DataFrame(rows).to_string(index=False))

    # ---- PART 11: MFE/MAE path analysis ----
    say('\n' + '=' * 90)
    say('PART 11 -- MFE/MAE PATH ANALYSIS (post-entry, outcome-only, not used for regime classification)')
    say('=' * 90)
    for pair, tdf in all_amr.items():
        say(f'\n-- {pair} AMR: mean MFE/MAE by vol tercile --')
        rows = []
        for v in ['LOW', 'NORMAL', 'HIGH']:
            sub = tdf[tdf.vol_tercile == v].dropna(subset=['mfe_atr', 'mae_atr'])
            if len(sub) < MIN_SAMPLE:
                continue
            rows.append(dict(vol_tercile=v, n=len(sub), mean_mfe=sub['mfe_atr'].mean(),
                              mean_mae=sub['mae_atr'].mean(), mfe_mae_ratio=sub['mfe_atr'].mean() / max(sub['mae_atr'].mean(), 1e-9)))
        say(pd.DataFrame(rows).to_string(index=False))

    report_path = REPO_ROOT / 'reports' / 'phase21_mechanism_log.txt'
    report_path.write_text('\n'.join(LOG), encoding='utf-8')
    say(f'\nFull log written to {report_path}')


if __name__ == '__main__':
    main()
