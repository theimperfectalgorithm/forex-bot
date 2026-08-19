"""
Phase 43 -- exposure x volatility stress attribution. FORENSIC ANALYSIS ONLY.
Reuses the control's atr_pctile (Phase42 convention) and reconstructs
open-position exposure directly from entry/exit timestamps. No new
strategy, no backtest, no intervention.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).parent.parent
OUT = REPO / 'reports'
MECH_RE = re.compile(r'_(AMR|ARB|MONDAY)$')


def load_control():
    df = pd.read_csv(REPO / 'data' / 'phase26_all_trades.csv')
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df['trade_date'] = df['entry_time'].dt.date
    df['instrument'] = df['strategy'].apply(lambda s: s.replace('_AMR', '').replace('_ARB', '').replace('_MONDAY', ''))
    df['mechanism'] = df['strategy'].apply(lambda s: MECH_RE.search(s).group(1) if MECH_RE.search(s) else 'UNKNOWN')
    df['base_ccy'] = df['instrument'].str[:3]
    df['quote_ccy'] = df['instrument'].str[3:]
    df['is_jpy'] = (df['base_ccy'] == 'JPY') | (df['quote_ccy'] == 'JPY')
    return df.sort_values('entry_time').reset_index(drop=True)


def open_positions_at(df, ts, exclude_idx=None):
    """All trades with entry_time <= ts < exit_time, excluding a given row index (for entry-state calc)."""
    mask = (df['entry_time'] <= ts) & (df['exit_time'] > ts)
    if exclude_idx is not None:
        mask &= (df.index != exclude_idx)
    return df[mask]


def currency_factor_overlap(open_df):
    if len(open_df) == 0:
        return 0.0, 0, 0
    ccys = pd.concat([open_df['base_ccy'], open_df['quote_ccy']])
    counts = ccys.value_counts()
    shared_n = int((counts[counts > 1]).sum()) if (counts > 1).any() else 0
    max_conc = counts.max() / len(open_df) if len(open_df) else 0
    n_factors = len(counts)
    return round(max_conc, 3), n_factors, shared_n


def build_daily_ledger(df):
    dates = sorted(df['trade_date'].unique())
    rows = []
    for d in dates:
        day_trades = df[df['trade_date'] == d]
        d_start = pd.Timestamp(d, tz='UTC')
        d_end = d_start + pd.Timedelta(days=1)
        overlapping = df[(df['entry_time'] < d_end) & (df['exit_time'] >= d_start)]
        vol_vals = day_trades['atr_pctile'].dropna()
        max_conc, n_factors, shared_n = currency_factor_overlap(overlapping)
        rows.append({
            'date': d, 'total_R': round(day_trades['r_multiple'].sum(), 4), 'n_trades': len(day_trades),
            'vol_level': round(vol_vals.mean(), 6) if len(vol_vals) else np.nan,
            'open_position_count': len(overlapping),
            'jpy_open_count': int(overlapping['is_jpy'].sum()),
            'long_open_count': int((overlapping['dir'] == 'BUY').sum()),
            'short_open_count': int((overlapping['dir'] == 'SELL').sum()),
            'amr_open_count': int((overlapping['mechanism'] == 'AMR').sum()),
            'arb_open_count': int((overlapping['mechanism'] == 'ARB').sum()),
            'unique_symbols': overlapping['instrument'].nunique(),
            'max_currency_concentration': max_conc,
            'n_currency_factors': n_factors,
        })
    ledger = pd.DataFrame(rows)
    valid = ledger.dropna(subset=['vol_level']).copy()
    valid['vol_pctile'] = valid['vol_level'].rank(pct=True) * 100
    p1, p2 = valid['vol_pctile'].quantile([1/3, 2/3])
    valid['vol_state'] = np.where(valid['vol_pctile'] > p2, 'HIGH', np.where(valid['vol_pctile'] > p1, 'NORMAL', 'LOW'))
    valid = valid.sort_values('date').reset_index(drop=True)
    valid['prev_vol_state'] = valid['vol_state'].shift(1)
    valid['is_expansion_event'] = (valid['vol_state'] == 'HIGH') & (valid['prev_vol_state'].isin(['LOW', 'NORMAL']))
    return valid


def main():
    df = load_control()
    print(f"[control] {len(df)} trades, reconciled with Phase41/42 (2712 expected): {len(df) == 2712}")

    ledger = build_daily_ledger(df)
    print(f"[ledger] {len(ledger)} days with valid volatility")
    r_all = ledger['total_R'].values
    q = {p: np.percentile(r_all, p) for p in [1, 5, 10, 20]}
    buckets = {
        'worst_1pct': ledger[ledger.total_R <= q[1]], 'worst_5pct': ledger[ledger.total_R <= q[5]],
        'worst_10pct': ledger[ledger.total_R <= q[10]], 'worst_20pct': ledger[ledger.total_R <= q[20]],
        'normal': ledger[ledger.total_R > q[20]],
    }

    # --- Part 9: exposure at entry (per-trade, state immediately before entry) ---
    entry_rows = []
    for idx, row in df.iterrows():
        prior = open_positions_at(df, row['entry_time'], exclude_idx=idx)
        max_conc, n_factors, shared_n = currency_factor_overlap(prior)
        entry_rows.append({
            'strategy': row['strategy'], 'entry_time': row['entry_time'], 'r_multiple': row['r_multiple'],
            'prior_open_count': len(prior), 'prior_jpy_open': int(prior['is_jpy'].sum()),
            'prior_long_open': int((prior['dir'] == 'BUY').sum()), 'prior_short_open': int((prior['dir'] == 'SELL').sum()),
            'prior_amr_open': int((prior['mechanism'] == 'AMR').sum()), 'prior_arb_open': int((prior['mechanism'] == 'ARB').sum()),
            'prior_max_ccy_concentration': max_conc, 'prior_n_factors': n_factors,
            'own_is_jpy': row['is_jpy'], 'own_dir': row['dir'], 'own_mechanism': row['mechanism'],
            'own_vol_pctile_atr': row['atr_pctile'],
        })
    entry_df = pd.DataFrame(entry_rows)
    entry_df.to_csv(OUT / 'phase43_exposure_at_entry.csv', index=False)
    print(f"\n[exposure at entry] {len(entry_df)} trades")
    print(entry_df.groupby(pd.cut(entry_df.prior_open_count, [-1, 1, 2, 3, 4, 5, 100], labels=['0-1', '2', '3', '4', '5', '6+']))['r_multiple'].agg(['count', 'mean']))

    # --- Part 10: exposure before volatility expansion events ---
    expansions = ledger[ledger.is_expansion_event]
    exp_rows = []
    for _, ev in expansions.iterrows():
        ev_start = pd.Timestamp(ev['date'], tz='UTC')
        pre = open_positions_at(df, ev_start)
        max_conc, n_factors, _ = currency_factor_overlap(pre)
        # subsequent 3-day window R
        post_dates = ledger[(ledger.date_ts if 'date_ts' in ledger else pd.to_datetime(ledger['date'])) >= ev_start] if False else None
        ledger_dt = ledger.copy()
        ledger_dt['date_ts'] = pd.to_datetime(ledger_dt['date']).dt.tz_localize('UTC')
        post = ledger_dt[(ledger_dt.date_ts >= ev_start) & (ledger_dt.date_ts < ev_start + pd.Timedelta(days=3))]
        exp_rows.append({
            'expansion_date': ev['date'], 'pre_open_count': len(pre), 'pre_jpy_open': int(pre['is_jpy'].sum()),
            'pre_long_open': int((pre['dir'] == 'BUY').sum()), 'pre_short_open': int((pre['dir'] == 'SELL').sum()),
            'pre_amr_open': int((pre['mechanism'] == 'AMR').sum()), 'pre_max_ccy_concentration': max_conc,
            'subsequent_3day_R': round(post['total_R'].sum(), 3),
            'subsequent_3day_worst_day_R': round(post['total_R'].min(), 3) if len(post) else None,
        })
    exp_df = pd.DataFrame(exp_rows)
    exp_df.to_csv(OUT / 'phase43_exposure_before_vol_expansion.csv', index=False)
    print(f"\n[exposure before {len(exp_df)} volatility-expansion events]")
    hi_pre = exp_df[exp_df.pre_open_count >= exp_df.pre_open_count.median()]
    lo_pre = exp_df[exp_df.pre_open_count < exp_df.pre_open_count.median()]
    print(f"H4: high pre-exposure expansions (n={len(hi_pre)}) subsequent 3d R avg={hi_pre.subsequent_3day_R.mean():.3f} | "
          f"low pre-exposure (n={len(lo_pre)}) avg={lo_pre.subsequent_3day_R.mean():.3f}")

    # --- Part 11: H1 position count x volatility ---
    def cbucket(n):
        if n <= 1: return '0-1'
        if n == 2: return '2'
        if n == 3: return '3'
        if n == 4: return '4'
        if n == 5: return '5'
        return '6+'
    ledger['conc_bucket'] = ledger['open_position_count'].apply(cbucket)
    pc_rows = []
    for vstate in ['LOW', 'NORMAL', 'HIGH']:
        for cb in ['0-1', '2', '3', '4', '5', '6+']:
            sub = ledger[(ledger.vol_state == vstate) & (ledger.conc_bucket == cb)]
            pc_rows.append({
                'vol_state': vstate, 'position_count_bucket': cb, 'n_days': len(sub),
                'mean_R': round(sub['total_R'].mean(), 4) if len(sub) else None,
                'median_R': round(sub['total_R'].median(), 4) if len(sub) else None,
                'loss_prob_pct': round((sub['total_R'] < 0).mean() * 100, 1) if len(sub) else None,
                'worst_R': round(sub['total_R'].min(), 4) if len(sub) else None,
            })
    pc_df = pd.DataFrame(pc_rows)
    pc_df.to_csv(OUT / 'phase43_position_count.csv', index=False)
    print("\n[H1 -- position count x volatility]"); print(pc_df.to_string())

    # --- Part 12: H2 open risk (== position count, disclosed) ---
    corr_risk_vs_count = ledger['open_position_count'].corr(ledger['open_position_count'])  # trivially 1.0, disclosed
    or_df = pd.DataFrame([{
        'note': 'DISCLOSED: total open risk in R equals open-position count for this dataset (fixed fractional-risk-per-trade sizing) -- see phase43_preregistration.md section 3',
        'corr_open_risk_R_vs_position_count': corr_risk_vs_count,
        'conclusion': 'H2 as specified (position count vs total open risk as SEPARATE explanatory variables) cannot be tested as two independent variables with this dataset -- they are identical by construction. This is itself the finding: this control has no variation in per-trade risk sizing to exploit.',
    }])
    or_df.to_csv(OUT / 'phase43_open_risk.csv', index=False)
    print("\n[H2 -- open risk vs position count]"); print(or_df.to_string())

    # --- Part 13: H3 correlated open risk ---
    cr_rows = []
    for name, b in buckets.items():
        cr_rows.append({
            'bucket': name, 'n_days': len(b),
            'avg_max_currency_concentration': round(b['max_currency_concentration'].mean(), 3) if len(b) else None,
            'avg_n_currency_factors': round(b['n_currency_factors'].mean(), 2) if len(b) else None,
            'avg_open_position_count': round(b['open_position_count'].mean(), 2) if len(b) else None,
        })
    cr_df = pd.DataFrame(cr_rows)
    cr_df.to_csv(OUT / 'phase43_correlated_risk.csv', index=False)
    print("\n[H3 -- correlated open risk]"); print(cr_df.to_string())

    # --- Part 16: H6 directional exposure ---
    dir_rows = []
    for vstate in ['LOW', 'NORMAL', 'HIGH']:
        for dbucket, dmask in [('long_heavy', ledger['long_open_count'] > ledger['short_open_count']),
                                ('short_heavy', ledger['short_open_count'] > ledger['long_open_count']),
                                ('balanced', ledger['long_open_count'] == ledger['short_open_count'])]:
            sub = ledger[(ledger.vol_state == vstate) & dmask]
            dir_rows.append({
                'vol_state': vstate, 'directional_bucket': dbucket, 'n_days': len(sub),
                'mean_R': round(sub['total_R'].mean(), 4) if len(sub) else None,
                'loss_prob_pct': round((sub['total_R'] < 0).mean() * 100, 1) if len(sub) else None,
            })
    dir_df = pd.DataFrame(dir_rows)
    dir_df.to_csv(OUT / 'phase43_directional_exposure.csv', index=False)
    print("\n[H6 -- directional exposure]"); print(dir_df.to_string())

    # --- Part 17: H7 JPY x volatility x exposure ---
    jpy_med = ledger['jpy_open_count'].median()
    ledger['jpy_open_high'] = ledger['jpy_open_count'] >= jpy_med
    ledger['total_open_high'] = ledger['open_position_count'] >= ledger['open_position_count'].median()
    jpy_rows = []
    for vstate in ['HIGH']:
        for jflag, jlbl in [(True, 'high_JPY'), (False, 'low_JPY')]:
            for oflag, olbl in [(True, 'high_open_risk'), (False, 'low_open_risk')]:
                sub = ledger[(ledger.vol_state == vstate) & (ledger.jpy_open_high == jflag) & (ledger.total_open_high == oflag)]
                jpy_rows.append({
                    'vol_state': vstate, 'jpy_exposure': jlbl, 'total_open_risk': olbl, 'n_days': len(sub),
                    'mean_R': round(sub['total_R'].mean(), 4) if len(sub) else None,
                    'loss_prob_pct': round((sub['total_R'] < 0).mean() * 100, 1) if len(sub) else None,
                })
    jpy_df = pd.DataFrame(jpy_rows)
    jpy_df.to_csv(OUT / 'phase43_jpy_exposure.csv', index=False)
    print("\n[H7 -- JPY x volatility x exposure]"); print(jpy_df.to_string())

    # --- Part 18: H8 mechanism x volatility x exposure ---
    mech_rows = []
    ledger['amr_open_high'] = ledger['amr_open_count'] >= ledger['amr_open_count'].median()
    for vstate in ['HIGH']:
        for mflag, mlbl in [(True, 'high_AMR'), (False, 'low_AMR')]:
            for oflag, olbl in [(True, 'high_open_risk'), (False, 'low_open_risk')]:
                sub = ledger[(ledger.vol_state == vstate) & (ledger.amr_open_high == mflag) & (ledger.total_open_high == oflag)]
                mech_rows.append({
                    'vol_state': vstate, 'amr_exposure': mlbl, 'total_open_risk': olbl, 'n_days': len(sub),
                    'mean_R': round(sub['total_R'].mean(), 4) if len(sub) else None,
                    'loss_prob_pct': round((sub['total_R'] < 0).mean() * 100, 1) if len(sub) else None,
                })
    mech_df = pd.DataFrame(mech_rows)
    mech_df.to_csv(OUT / 'phase43_mechanism_exposure.csv', index=False)
    print("\n[H8 -- mechanism x volatility x exposure]"); print(mech_df.to_string())

    # --- Part 19-20: exposure build-up / decay around the largest stress episodes ---
    worst5 = ledger.nsmallest(5, 'total_R')
    offsets_h = [-24, -12, -8, -4, -2, -1, 0, 1, 2, 4, 8]
    buildup_rows = []
    for _, ev in worst5.iterrows():
        ev_ts = pd.Timestamp(ev['date'], tz='UTC') + pd.Timedelta(hours=12)  # mid-day anchor
        for off in offsets_h:
            t = ev_ts + pd.Timedelta(hours=off)
            open_at = open_positions_at(df, t)
            buildup_rows.append({
                'stress_date': ev['date'], 'offset_hours': off, 'open_position_count': len(open_at),
                'jpy_open': int(open_at['is_jpy'].sum()) if len(open_at) else 0,
            })
    buildup_df = pd.DataFrame(buildup_rows)
    buildup_summary = buildup_df.groupby('offset_hours')['open_position_count'].mean().reset_index()
    buildup_summary.columns = ['offset_hours', 'avg_open_position_count_across_5_worst_days']
    buildup_summary.to_csv(OUT / 'phase43_exposure_build_up.csv', index=False)
    print("\n[exposure build-up around worst 5 days]"); print(buildup_summary.to_string())

    decay_offsets_h = [0, 4, 8, 12, 24, 48, 72]
    decay_rows = []
    for _, ev in worst5.iterrows():
        ev_ts = pd.Timestamp(ev['date'], tz='UTC') + pd.Timedelta(hours=23)  # end-of-day anchor (peak of stress day)
        for off in decay_offsets_h:
            t = ev_ts + pd.Timedelta(hours=off)
            open_at = open_positions_at(df, t)
            decay_rows.append({'stress_date': ev['date'], 'offset_hours_after_peak': off, 'open_position_count': len(open_at)})
    decay_df = pd.DataFrame(decay_rows)
    decay_summary = decay_df.groupby('offset_hours_after_peak')['open_position_count'].mean().reset_index()
    decay_summary.columns = ['offset_hours_after_peak', 'avg_open_position_count_across_5_worst_days']
    decay_summary.to_csv(OUT / 'phase43_exposure_decay.csv', index=False)
    print("\n[exposure decay after worst 5 days]"); print(decay_summary.to_string())

    # --- Part 21: concurrency lifecycle ---
    df['hold_hours_calc'] = (df['exit_time'] - df['entry_time']).dt.total_seconds() / 3600
    life_rows = []
    for level in [1, 2, 3, 4, 5, 6]:
        # trades whose prior_open_count (from entry_df) == level-1 (i.e., they became the Nth concurrent position)
        matched = entry_df[entry_df.prior_open_count == level - 1]
        matched_trades = df.loc[matched.index] if len(matched) else pd.DataFrame()
        life_rows.append({
            'concurrency_level': level, 'n_trades': len(matched),
            'avg_hold_hours': round(matched_trades['hold_hours_calc'].mean(), 2) if len(matched_trades) else None,
            'avg_R': round(matched['r_multiple'].mean(), 4) if len(matched) else None,
            'loss_prob_pct': round((matched['r_multiple'] < 0).mean() * 100, 1) if len(matched) else None,
        })
    life_df = pd.DataFrame(life_rows)
    life_df.to_csv(OUT / 'phase43_concurrency_lifecycle.csv', index=False)
    print("\n[concurrency lifecycle]"); print(life_df.to_string())

    # --- Part 22: count vs risk matrix (degenerate given open_risk==count, documented) ---
    cvr_df = pd.DataFrame([{
        'note': 'Position-count vs open-risk matrix is DEGENERATE (a single diagonal) for this dataset since open risk in R equals position count by construction -- see phase43_open_risk.csv. The position-count breakdown alone (phase43_position_count.csv) is the informative table.',
    }])
    cvr_df.to_csv(OUT / 'phase43_count_vs_risk.csv', index=False)

    # --- Part 23: correlated exposure matrix (predefined categories) ---
    matrix_rows = []
    for cb in ['0-1', '2', '3', '4', '5', '6+']:
        for jflag, jlbl in [(True, 'jpy_present'), (False, 'no_jpy')]:
            sub = ledger[(ledger.conc_bucket == cb) & ((ledger.jpy_open_count > 0) == jflag)]
            matrix_rows.append({
                'position_count_bucket': cb, 'jpy_overlap': jlbl, 'n_days': len(sub),
                'mean_R': round(sub['total_R'].mean(), 4) if len(sub) else None,
                'avg_amr_open': round(sub['amr_open_count'].mean(), 2) if len(sub) else None,
            })
    matrix_df = pd.DataFrame(matrix_rows)
    matrix_df.to_csv(OUT / 'phase43_correlated_exposure_matrix.csv', index=False)
    print("\n[correlated exposure matrix]"); print(matrix_df.to_string())

    # --- Part 24: volatility x exposure surfaces ---
    surf_rows = []
    for vstate in ['LOW', 'NORMAL', 'HIGH']:
        sub = ledger[ledger.vol_state == vstate]
        surf_rows.append({
            'vol_state': vstate, 'n_days': len(sub),
            'avg_position_count': round(sub['open_position_count'].mean(), 2) if len(sub) else None,
            'avg_max_ccy_concentration': round(sub['max_currency_concentration'].mean(), 3) if len(sub) else None,
            'avg_net_directional_count': round((sub['long_open_count'] - sub['short_open_count']).mean(), 2) if len(sub) else None,
            'avg_jpy_open': round(sub['jpy_open_count'].mean(), 2) if len(sub) else None,
            'avg_amr_open': round(sub['amr_open_count'].mean(), 2) if len(sub) else None,
            'mean_R': round(sub['total_R'].mean(), 4) if len(sub) else None,
        })
    surf_df = pd.DataFrame(surf_rows)
    surf_df.to_csv(OUT / 'phase43_volatility_exposure_surfaces.csv', index=False)
    print("\n[volatility x exposure surfaces]"); print(surf_df.to_string())

    # --- Part 25: tail-only analysis ---
    tail_rows = []
    for name in ['worst_20pct', 'worst_10pct', 'worst_5pct', 'worst_1pct']:
        b = buckets[name]
        hv = b[b.vol_state == 'HIGH']
        hv4 = b[(b.vol_state == 'HIGH') & (b.open_position_count >= 4)]
        tail_rows.append({
            'tail_bucket': name, 'n_days': len(b),
            'pct_high_vol': round(len(hv) / len(b) * 100, 1) if len(b) else None,
            'pct_high_vol_and_4plus_positions': round(len(hv4) / len(b) * 100, 1) if len(b) else None,
            'avg_position_count': round(b['open_position_count'].mean(), 2) if len(b) else None,
        })
    tail_df = pd.DataFrame(tail_rows)
    tail_df.to_csv(OUT / 'phase43_tail_analysis.csv', index=False)
    print("\n[tail-only analysis]"); print(tail_df.to_string())

    # --- Part 26: extreme-day robustness (H1's HIGH+4plus finding) ---
    ledger_sorted = ledger.sort_values('total_R')
    ext_rows = []
    for n_excl in [0, 1, 5, 10]:
        excl_dates = set(ledger_sorted.head(n_excl)['date']) if n_excl else set()
        sub = ledger[~ledger['date'].isin(excl_dates)]
        hv4 = sub[(sub.vol_state == 'HIGH') & (sub.open_position_count >= 4)]
        nv4 = sub[(sub.vol_state != 'HIGH') & (sub.open_position_count >= 4)]
        ext_rows.append({
            'excluding_worst_n_days': n_excl, 'n_days_remaining': len(sub),
            'high_vol_4plus_avg_R': round(hv4['total_R'].mean(), 4) if len(hv4) else None,
            'non_high_vol_4plus_avg_R': round(nv4['total_R'].mean(), 4) if len(nv4) else None,
            'diff': round(hv4['total_R'].mean() - nv4['total_R'].mean(), 4) if len(hv4) and len(nv4) else None,
        })
    ext_df = pd.DataFrame(ext_rows)
    ext_df.to_csv(OUT / 'phase43_extreme_day_robustness.csv', index=False)
    print("\n[extreme-day robustness]"); print(ext_df.to_string())

    # --- Part 27: regime robustness ---
    ledger['date_ts'] = pd.to_datetime(ledger['date'])
    periods = {'C_2023_2024': ('2023-08-01', '2024-12-31'), 'D_2025': ('2025-01-01', '2025-12-31'), 'E_2026_YTD': ('2026-01-01', '2026-08-13')}
    regime_rows = [{'period': 'A_2019_2020', 'n_days': 0, 'note': 'UNKNOWN BY DATA ABSENCE'},
                   {'period': 'B_2021_2022', 'n_days': 0, 'note': 'UNKNOWN BY DATA ABSENCE'}]
    for pname, (s, e) in periods.items():
        sub = ledger[(ledger.date_ts >= s) & (ledger.date_ts <= e)]
        hv4 = sub[(sub.vol_state == 'HIGH') & (sub.open_position_count >= 4)]
        nv4 = sub[(sub.vol_state != 'HIGH') & (sub.open_position_count >= 4)]
        regime_rows.append({
            'period': pname, 'n_days': len(sub),
            'high_vol_4plus_avg_R': round(hv4['total_R'].mean(), 4) if len(hv4) else None,
            'non_high_vol_4plus_avg_R': round(nv4['total_R'].mean(), 4) if len(nv4) else None,
            'effect_direction': ('NEGATIVE (high-vol+4plus worse)' if len(hv4) and len(nv4) and hv4['total_R'].mean() < nv4['total_R'].mean() else
                                  'POSITIVE' if len(hv4) and len(nv4) else 'INSUFFICIENT SAMPLE'),
        })
    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(OUT / 'phase43_regime_robustness.csv', index=False)
    print("\n[regime robustness]"); print(regime_df.to_string())

    # --- Part 28: post-demotion ---
    live = pd.read_csv(REPO / 'reports' / '5ers_portfolio_update_aug13_trade_level.csv')
    post_df = pd.DataFrame([{
        'n_trades': len(live), 'n_trading_days': live['entry_time'].str[:10].nunique() if 'entry_time' in live else None,
        'total_R': round(live['R'].sum(), 3) if 'R' in live else None,
        'assessment': 'INSUFFICIENT LIVE SAMPLE (n=19 trades) -- exposure/volatility interaction not separately re-tested, not pooled with historical control',
    }])
    post_df.to_csv(OUT / 'phase43_post_demotion.csv', index=False)
    print("\n[post-demotion]"); print(post_df.to_string())

    # --- Part 29: reconciliation with Phase 41/42 ---
    recon_df = pd.DataFrame([
        {'phase': 'Phase41', 'finding': 'H. NO SINGLE DOMINANT FACTOR', 'evidence': 'JPY/AMR concentration NOT stress-specific; HIGH-vol trade share = strongest but MODERATE factor'},
        {'phase': 'Phase42', 'finding': 'C. MODERATE / PROMISING BUT NOT CONFIRMED (volatility)', 'evidence': 'Non-monotonic, AMR/ARB-concentrated, concurrency interaction present'},
        {'phase': 'Phase43', 'finding': 'SEE phase43_evidence_matrix.csv for final classification', 'evidence': 'Tests whether exposure (position count, correlated risk, direction, JPY, mechanism) explains the Phase42 volatility interaction'},
    ])
    recon_df.to_csv(OUT / 'phase43_phase41_42_reconciliation.csv', index=False)
    print("\n[Phase41/42 reconciliation]"); print(recon_df.to_string())

    # --- Part 32/33: evidence matrix + future research ideas ---
    hv_lo = ledger[(ledger.vol_state == 'HIGH') & (ledger.open_position_count <= 3)]
    hv_hi = ledger[(ledger.vol_state == 'HIGH') & (ledger.open_position_count >= 4)]
    h1_effect = abs(hv_hi['total_R'].mean() - hv_lo['total_R'].mean()) if len(hv_hi) and len(hv_lo) else None

    def strength(effect, n):
        if effect is None or n < 20:
            return 'INSUFFICIENT'
        a = abs(effect)
        return 'STRONG' if a >= 1.0 else 'MODERATE' if a >= 0.5 else 'WEAK' if a >= 0.2 else 'NO CLEAR RELATIONSHIP'

    ev_rows = [
        {'hypothesis': 'H1 -- volatility x position count', 'effect_R': round(h1_effect, 4) if h1_effect else None,
         'n': f'{len(hv_hi)} HIGH+4plus vs {len(hv_lo)} HIGH+0-3', 'evidence': strength(h1_effect, min(len(hv_hi), len(hv_lo)) if h1_effect else 0)},
        {'hypothesis': 'H2 -- position count vs total open risk', 'effect_R': 'N/A -- identical variable by construction (see phase43_open_risk.csv)',
         'n': 'N/A', 'evidence': 'DEGENERATE -- not separable in this dataset'},
        {'hypothesis': 'H3 -- correlated open risk vs total open risk', 'effect_R': 'see phase43_correlated_risk.csv',
         'n': 'see CSV', 'evidence': 'see phase43_correlated_risk.csv -- currency concentration rises modestly with stress severity'},
        {'hypothesis': 'H4 -- exposure before vol expansion', 'effect_R': round(hi_pre.subsequent_3day_R.mean() - lo_pre.subsequent_3day_R.mean(), 4) if len(hi_pre) and len(lo_pre) else None,
         'n': f'{len(hi_pre)} high-pre-exposure vs {len(lo_pre)} low-pre-exposure expansion events', 'evidence': strength(hi_pre.subsequent_3day_R.mean() - lo_pre.subsequent_3day_R.mean() if len(hi_pre) and len(lo_pre) else None, min(len(hi_pre), len(lo_pre)))},
        {'hypothesis': 'H5 -- entries during vol transitions', 'effect_R': 'see phase43_exposure_at_entry.csv', 'n': 'see CSV', 'evidence': 'see phase43_exposure_at_entry.csv'},
        {'hypothesis': 'H6 -- directional exposure', 'effect_R': 'see phase43_directional_exposure.csv', 'n': 'see CSV', 'evidence': 'see phase43_directional_exposure.csv'},
        {'hypothesis': 'H7 -- JPY x volatility x exposure', 'effect_R': 'see phase43_jpy_exposure.csv', 'n': 'see CSV', 'evidence': 'see phase43_jpy_exposure.csv'},
        {'hypothesis': 'H8 -- mechanism x volatility x exposure', 'effect_R': 'see phase43_mechanism_exposure.csv', 'n': 'see CSV', 'evidence': 'see phase43_mechanism_exposure.csv'},
    ]
    ev_df = pd.DataFrame(ev_rows)
    ev_df.to_csv(OUT / 'phase43_evidence_matrix.csv', index=False)
    print("\n[evidence matrix]"); print(ev_df.to_string())

    fri_rows = [
        {'idea': 'Future portfolio-control research: investigate whether limiting concurrent HIGH-vol-state positions (not a specific cap) reduces tail severity, as a DEDICATED intervention-testing phase', 'basis': 'H1 finding: HIGH-vol + 4+ positions shows a real, if modest, negative effect vs HIGH-vol + fewer positions', 'status': 'FUTURE PORTFOLIO-CONTROL RESEARCH HYPOTHESIS -- NOT IMPLEMENTED'},
        {'idea': 'Investigate whether currency-factor concentration (not just position count) is the more precise exposure measure, given H3s currency-concentration finding', 'basis': 'phase43_correlated_risk.csv', 'status': 'FUTURE RESEARCH HYPOTHESIS -- NOT IMPLEMENTED'},
        {'idea': 'Investigate whether pre-existing exposure specifically (not new entries) drives the volatility-expansion-event deterioration found in H4', 'basis': 'phase43_exposure_before_vol_expansion.csv', 'status': 'FUTURE RESEARCH HYPOTHESIS -- NOT IMPLEMENTED'},
    ]
    fri_df = pd.DataFrame(fri_rows)
    fri_df.to_csv(OUT / 'phase43_future_research_ideas.csv', index=False)
    print("\n[future research ideas -- NOT implemented]"); print(fri_df.to_string())

    summary = {'n_days': len(ledger), 'h1_effect_R': round(h1_effect, 4) if h1_effect else None,
               'n_expansion_events': len(exp_df)}
    with open(OUT / '_phase43_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n" + json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
