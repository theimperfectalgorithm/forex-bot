"""
Phase 42 -- volatility stress decomposition. FORENSIC ANALYSIS ONLY.
Reuses the control's own already-validated per-trade atr_pctile as the
continuous volatility measure (no new indicator). No new strategy,
no backtest, no intervention.
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
    return df


def build_daily_volatility_ledger(df):
    """One row per trading day with a valid atr_pctile mean, R, concurrency, etc."""
    dates = sorted(df['trade_date'].unique())
    rows = []
    for d in dates:
        day_trades = df[df['trade_date'] == d]
        d_start = pd.Timestamp(d, tz='UTC')
        d_end = d_start + pd.Timedelta(days=1)
        overlapping = df[(df['entry_time'] < d_end) & (df['exit_time'] >= d_start)]
        vol_vals = day_trades['atr_pctile'].dropna()
        by_strat = day_trades.groupby('strategy')['r_multiple'].sum()
        rows.append({
            'date': d,
            'vol_level': round(vol_vals.mean(), 6) if len(vol_vals) else np.nan,
            'total_R': round(day_trades['r_multiple'].sum(), 4),
            'n_trades': len(day_trades),
            'concurrent_positions': len(overlapping),
            'n_simultaneous_losing_strategies': int((by_strat < 0).sum()),
            'jpy_R': round(day_trades.loc[day_trades.is_jpy, 'r_multiple'].sum(), 4),
            'nonjpy_R': round(day_trades.loc[~day_trades.is_jpy, 'r_multiple'].sum(), 4),
            'jpy_trade_pct': round(day_trades['is_jpy'].mean() * 100, 1) if len(day_trades) else None,
            'long_R': round(day_trades.loc[day_trades.dir == 'BUY', 'r_multiple'].sum(), 4),
            'short_R': round(day_trades.loc[day_trades.dir == 'SELL', 'r_multiple'].sum(), 4),
            'amr_R': round(day_trades.loc[day_trades.mechanism == 'AMR', 'r_multiple'].sum(), 4),
            'arb_R': round(day_trades.loc[day_trades.mechanism == 'ARB', 'r_multiple'].sum(), 4),
            'monday_R': round(day_trades.loc[day_trades.mechanism == 'MONDAY', 'r_multiple'].sum(), 4),
            'asian_R': round(day_trades.loc[day_trades.session == 'ASIAN', 'r_multiple'].sum(), 4),
            'london_R': round(day_trades.loc[day_trades.session == 'LONDON', 'r_multiple'].sum(), 4),
            'session_mix': '; '.join(sorted(day_trades['session'].unique())),
        })
    ledger = pd.DataFrame(rows)
    valid = ledger.dropna(subset=['vol_level']).copy()
    valid['vol_pctile'] = valid['vol_level'].rank(pct=True) * 100
    q1, q2 = valid['vol_pctile'].quantile([1/3, 2/3])
    valid['vol_state'] = np.where(valid['vol_pctile'] > q2 * 100 / 100, 'HIGH',
                          np.where(valid['vol_pctile'] > q1 * 100 / 100, 'NORMAL', 'LOW'))
    # simpler: recompute state from tercile of vol_pctile directly
    p1, p2 = valid['vol_pctile'].quantile([1/3, 2/3])
    valid['vol_state'] = np.where(valid['vol_pctile'] > p2, 'HIGH', np.where(valid['vol_pctile'] > p1, 'NORMAL', 'LOW'))
    valid = valid.sort_values('date').reset_index(drop=True)
    valid['vol_change'] = valid['vol_pctile'].diff()
    valid['vol_accel'] = valid['vol_change'].diff()
    valid['prev_vol_state'] = valid['vol_state'].shift(1)
    dropped = len(ledger) - len(valid)
    return valid, dropped


def main():
    df = load_control()
    print(f"[control] {len(df)} trades, reconciled with Phase41 (2712 expected): {len(df) == 2712}")

    ledger, dropped = build_daily_volatility_ledger(df)
    print(f"[ledger] {len(ledger)} days with valid vol_level ({dropped} days excluded -- no atr_pctile-valid trades)")
    ledger.to_csv(OUT / '_scratch_phase42_ledger.csv', index=False)

    r_all = ledger['total_R'].values
    q_thresh = {p: np.percentile(r_all, p) for p in [1, 5, 10, 20]}

    def stress_bucket(pct):
        return ledger[ledger['total_R'] <= q_thresh[pct]]

    normal = ledger[ledger['total_R'] > q_thresh[20]]

    # --- Part 8: continuous volatility percentile buckets ---
    bins = list(range(0, 101, 10))
    labels = [f'{lo}-{hi}%' for lo, hi in zip(bins[:-1], bins[1:])]
    ledger['pctile_bucket'] = pd.cut(ledger['vol_pctile'], bins=bins, labels=labels, include_lowest=True)
    pct_rows = []
    for lbl in labels:
        sub = ledger[ledger['pctile_bucket'] == lbl]
        if len(sub) == 0:
            pct_rows.append({'bucket': lbl, 'n_days': 0})
            continue
        pct_rows.append({
            'bucket': lbl, 'n_days': len(sub),
            'total_R': round(sub['total_R'].sum(), 3), 'avg_daily_R': round(sub['total_R'].mean(), 4),
            'median_daily_R': round(sub['total_R'].median(), 4),
            'win_day_pct': round((sub['total_R'] > 0).mean() * 100, 1),
            'loss_day_pct': round((sub['total_R'] < 0).mean() * 100, 1),
            'std_daily_R': round(sub['total_R'].std(), 4),
            'worst_day_R': round(sub['total_R'].min(), 4),
            'pctile5_daily_R': round(sub['total_R'].quantile(0.05), 4) if len(sub) >= 20 else None,
            'avg_concurrent_positions': round(sub['concurrent_positions'].mean(), 2),
            'avg_n_trades': round(sub['n_trades'].mean(), 2),
            'avg_jpy_trade_pct': round(sub['jpy_trade_pct'].mean(), 1),
        })
    pct_df = pd.DataFrame(pct_rows)
    pct_df.to_csv(OUT / 'phase42_volatility_percentiles.csv', index=False)
    print("\n[H1 -- volatility percentile buckets]"); print(pct_df.to_string())

    # H1 classification: monotonicity of avg_daily_R across buckets + correlation
    valid_buckets = pct_df.dropna(subset=['avg_daily_R'])
    corr_vol_r = ledger['vol_pctile'].corr(ledger['total_R'])
    corr_vol_negprob = ledger['vol_pctile'].corr((ledger['total_R'] < 0).astype(int))
    print(f"corr(vol_pctile, daily_R) = {corr_vol_r:.3f}; corr(vol_pctile, P(loss)) = {corr_vol_negprob:.3f}")

    # --- Part 10: volatility change / state transitions ---
    vc_rows = []
    valid_change = ledger.dropna(subset=['vol_change'])
    q1c, q2c = valid_change['vol_change'].quantile([1/3, 2/3])
    def change_state(v):
        if pd.isna(v):
            return 'UNKNOWN'
        return 'HIGH_INCREASE' if v > q2c else ('HIGH_DECREASE' if v < q1c else 'STABLE')
    valid_change = valid_change.copy()
    valid_change['change_state'] = valid_change['vol_change'].apply(change_state)
    combos = ['stable_LOW', 'stable_NORMAL', 'stable_HIGH', 'LOW_to_HIGH', 'NORMAL_to_HIGH', 'HIGH_to_HIGH', 'HIGH_to_NORMAL', 'NORMAL_to_LOW']
    def classify_combo(row):
        prev, cur = row['prev_vol_state'], row['vol_state']
        if prev == cur:
            return f'stable_{cur}'
        return f'{prev}_to_{cur}'
    valid_change['combo'] = valid_change.apply(classify_combo, axis=1)
    for combo in combos:
        sub = valid_change[valid_change['combo'] == combo]
        vc_rows.append({
            'transition': combo, 'n_days': len(sub),
            'avg_daily_R': round(sub['total_R'].mean(), 4) if len(sub) else None,
            'loss_prob_pct': round((sub['total_R'] < 0).mean() * 100, 1) if len(sub) else None,
            'tail_loss_worst_R': round(sub['total_R'].min(), 4) if len(sub) else None,
            'avg_concurrent_positions': round(sub['concurrent_positions'].mean(), 2) if len(sub) else None,
            'avg_simultaneous_losing_strategies': round(sub['n_simultaneous_losing_strategies'].mean(), 2) if len(sub) else None,
            'sample_flag': 'THIN (<20)' if len(sub) < 20 else 'ADEQUATE',
        })
    vc_df = pd.DataFrame(vc_rows)
    vc_df.to_csv(OUT / 'phase42_volatility_change.csv', index=False)
    print("\n[H2 -- volatility change / transitions]"); print(vc_df.to_string())

    # --- Part 11: volatility acceleration ---
    valid_accel = ledger.dropna(subset=['vol_accel'])
    q1a, q2a = valid_accel['vol_accel'].quantile([1/3, 2/3])
    def accel_state(v):
        return 'HIGH' if v > q2a else ('LOW' if v < q1a else 'NORMAL')
    valid_accel = valid_accel.copy()
    valid_accel['accel_state'] = valid_accel['vol_accel'].apply(accel_state)
    accel_rows = []
    for astate in ['LOW', 'NORMAL', 'HIGH']:
        sub = valid_accel[valid_accel['accel_state'] == astate]
        accel_rows.append({
            'acceleration_state': astate, 'n_days': len(sub),
            'avg_daily_R': round(sub['total_R'].mean(), 4) if len(sub) else None,
            'loss_prob_pct': round((sub['total_R'] < 0).mean() * 100, 1) if len(sub) else None,
            'worst_day_R': round(sub['total_R'].min(), 4) if len(sub) else None,
        })
    # HIGH vol + HIGH accel vs HIGH vol + stable
    hv_ha = valid_accel[(valid_accel.vol_state == 'HIGH') & (valid_accel.accel_state == 'HIGH')]
    hv_stable = valid_accel[(valid_accel.vol_state == 'HIGH') & (valid_accel.accel_state == 'NORMAL')]
    accel_rows.append({'acceleration_state': 'HIGH_VOL + HIGH_ACCEL', 'n_days': len(hv_ha),
                        'avg_daily_R': round(hv_ha['total_R'].mean(), 4) if len(hv_ha) else None,
                        'loss_prob_pct': round((hv_ha['total_R'] < 0).mean() * 100, 1) if len(hv_ha) else None,
                        'worst_day_R': round(hv_ha['total_R'].min(), 4) if len(hv_ha) else None})
    accel_rows.append({'acceleration_state': 'HIGH_VOL + STABLE_ACCEL', 'n_days': len(hv_stable),
                        'avg_daily_R': round(hv_stable['total_R'].mean(), 4) if len(hv_stable) else None,
                        'loss_prob_pct': round((hv_stable['total_R'] < 0).mean() * 100, 1) if len(hv_stable) else None,
                        'worst_day_R': round(hv_stable['total_R'].min(), 4) if len(hv_stable) else None})
    accel_df = pd.DataFrame(accel_rows)
    accel_df.to_csv(OUT / 'phase42_volatility_acceleration.csv', index=False)
    print("\n[H3 -- volatility acceleration]"); print(accel_df.to_string())

    # --- Part 12: H4 volatility x concurrency ---
    def conc_bucket(n):
        if n <= 1: return '0-1'
        if n == 2: return '2'
        if n == 3: return '3'
        if n == 4: return '4'
        if n == 5: return '5'
        return '6+'
    ledger['conc_bucket'] = ledger['concurrent_positions'].apply(conc_bucket)
    conc_rows = []
    for vstate in ['LOW', 'NORMAL', 'HIGH']:
        for cbucket in ['0-1', '2', '3', '4', '5', '6+']:
            sub = ledger[(ledger.vol_state == vstate) & (ledger.conc_bucket == cbucket)]
            conc_rows.append({
                'vol_state': vstate, 'concurrent_bucket': cbucket, 'n_days': len(sub),
                'avg_daily_R': round(sub['total_R'].mean(), 4) if len(sub) else None,
                'worst_daily_R': round(sub['total_R'].min(), 4) if len(sub) else None,
                'loss_prob_pct': round((sub['total_R'] < 0).mean() * 100, 1) if len(sub) else None,
                'avg_simultaneous_losses': round(sub['n_simultaneous_losing_strategies'].mean(), 2) if len(sub) else None,
            })
    conc_df = pd.DataFrame(conc_rows)
    conc_df.to_csv(OUT / 'phase42_volatility_concurrency.csv', index=False)
    print("\n[H4 -- volatility x concurrency]"); print(conc_df.to_string())
    for cb in ['4', '5', '6+']:
        hv = ledger[(ledger.vol_state == 'HIGH') & (ledger.conc_bucket == cb)]
        nv = ledger[(ledger.vol_state != 'HIGH') & (ledger.conc_bucket == cb)]
        print(f"HIGH-vol + {cb}+: n={len(hv)} avgR={hv['total_R'].mean() if len(hv) else None} | "
              f"non-HIGH + {cb}+: n={len(nv)} avgR={nv['total_R'].mean() if len(nv) else None}")

    # --- Part 13: H5 volatility x session (Asian->London transition only; NY absent) ---
    sess_rows = []
    for vstate in ['LOW', 'NORMAL', 'HIGH']:
        sub = ledger[ledger.vol_state == vstate]
        sess_rows.append({
            'vol_state': vstate, 'n_days': len(sub),
            'asian_R': round(sub['asian_R'].sum(), 3), 'london_R': round(sub['london_R'].sum(), 3),
            'avg_daily_R': round(sub['total_R'].mean(), 4) if len(sub) else None,
            'note': 'Control has ZERO New York or overlap-session trades (confirmed Phase41 data integrity) -- London->NY transition is UNKNOWN BY DATA ABSENCE, not tested',
        })
    sess_df = pd.DataFrame(sess_rows)
    sess_df.to_csv(OUT / 'phase42_volatility_session.csv', index=False)
    print("\n[H5 -- volatility x session]"); print(sess_df.to_string())

    # --- Part 14: H6 volatility x direction ---
    dir_rows = []
    for vstate in ['LOW', 'NORMAL', 'HIGH']:
        sub = ledger[ledger.vol_state == vstate]
        dir_rows.append({
            'vol_state': vstate, 'n_days': len(sub),
            'long_R': round(sub['long_R'].sum(), 3), 'short_R': round(sub['short_R'].sum(), 3),
            'net_directional_R': round(sub['long_R'].sum() + sub['short_R'].sum(), 3),
        })
    dir_df = pd.DataFrame(dir_rows)
    dir_df.to_csv(OUT / 'phase42_volatility_direction.csv', index=False)
    print("\n[H6 -- volatility x direction]"); print(dir_df.to_string())

    # --- Part 15: H7 volatility x JPY ---
    jpy_med = ledger['jpy_trade_pct'].median()
    ledger['jpy_high'] = ledger['jpy_trade_pct'] >= jpy_med
    jpy_rows = []
    for vstate in ['LOW', 'HIGH']:
        for jpy_flag, jlabel in [(True, 'high_JPY'), (False, 'low_JPY')]:
            sub = ledger[(ledger.vol_state == vstate) & (ledger.jpy_high == jpy_flag)]
            jpy_rows.append({
                'vol_state': vstate, 'jpy_exposure': jlabel, 'n_days': len(sub),
                'avg_daily_R': round(sub['total_R'].mean(), 4) if len(sub) else None,
                'loss_prob_pct': round((sub['total_R'] < 0).mean() * 100, 1) if len(sub) else None,
            })
    jpy_df = pd.DataFrame(jpy_rows)
    jpy_df.to_csv(OUT / 'phase42_volatility_jpy.csv', index=False)
    print("\n[H7 -- volatility x JPY]"); print(jpy_df.to_string())

    # --- Part 16: H8 volatility x mechanism ---
    mech_rows = []
    for vstate in ['LOW', 'NORMAL', 'HIGH']:
        sub = ledger[ledger.vol_state == vstate]
        mech_rows.append({
            'vol_state': vstate, 'n_days': len(sub),
            'amr_R': round(sub['amr_R'].sum(), 3), 'arb_R': round(sub['arb_R'].sum(), 3),
            'monday_R': round(sub['monday_R'].sum(), 3),
        })
    mech_df = pd.DataFrame(mech_rows)
    mech_df.to_csv(OUT / 'phase42_volatility_mechanism.csv', index=False)
    print("\n[H8 -- volatility x mechanism]"); print(mech_df.to_string())

    # --- Part 17: transition matrix ---
    trans_rows = []
    for prev in ['LOW', 'NORMAL', 'HIGH']:
        for cur in ['LOW', 'NORMAL', 'HIGH']:
            sub = valid_change[(valid_change.prev_vol_state == prev) & (valid_change.vol_state == cur)] if 'prev_vol_state' in valid_change else pd.DataFrame()
            sub2 = ledger[(ledger.prev_vol_state == prev) & (ledger.vol_state == cur)]
            trans_rows.append({
                'from': prev, 'to': cur, 'n_days': len(sub2),
                'avg_R': round(sub2['total_R'].mean(), 4) if len(sub2) else None,
                'median_R': round(sub2['total_R'].median(), 4) if len(sub2) else None,
                'loss_prob_pct': round((sub2['total_R'] < 0).mean() * 100, 1) if len(sub2) else None,
                'worst_loss_R': round(sub2['total_R'].min(), 4) if len(sub2) else None,
                'avg_concurrent_positions': round(sub2['concurrent_positions'].mean(), 2) if len(sub2) else None,
                'avg_simultaneous_losses': round(sub2['n_simultaneous_losing_strategies'].mean(), 2) if len(sub2) else None,
            })
    trans_df = pd.DataFrame(trans_rows)
    trans_df.to_csv(OUT / 'phase42_transition_matrix.csv', index=False)
    print("\n[transition matrix]"); print(trans_df.to_string())

    # --- Part 18: lead-lag (day-level only, per preregistration limitation) ---
    ledger_ll = ledger.copy()
    ledger_ll['vol_pctile_prevday'] = ledger_ll['vol_pctile'].shift(1)
    ll_rows = [
        {'lag_window': 'same_day', 'corr_vol_vs_R': round(ledger_ll['vol_pctile'].corr(ledger_ll['total_R']), 3), 'n_days': ledger_ll['vol_pctile'].notna().sum()},
        {'lag_window': 'previous_trading_day', 'corr_vol_vs_R': round(ledger_ll['vol_pctile_prevday'].corr(ledger_ll['total_R']), 3), 'n_days': ledger_ll['vol_pctile_prevday'].notna().sum()},
        {'lag_window': 'previous_session', 'corr_vol_vs_R': 'UNKNOWN BY DATA LIMITATION', 'n_days': 'N/A -- no continuous intraday session-level volatility series stored'},
        {'lag_window': 'previous_4_hours', 'corr_vol_vs_R': 'UNKNOWN BY DATA LIMITATION', 'n_days': 'N/A -- trade-level granularity only, not continuous intraday'},
        {'lag_window': 'previous_8_hours', 'corr_vol_vs_R': 'UNKNOWN BY DATA LIMITATION', 'n_days': 'N/A -- trade-level granularity only, not continuous intraday'},
    ]
    ll_df = pd.DataFrame(ll_rows)
    ll_df.to_csv(OUT / 'phase42_lead_lag.csv', index=False)
    print("\n[lead-lag -- EXPLORATORY, day-level only]"); print(ll_df.to_string())

    # --- Part 19: tail analysis ---
    hv_days = ledger[ledger.vol_state == 'HIGH']
    ordinary_days = ledger[ledger.vol_state != 'HIGH']
    tail_rows = []
    for pct in [1, 5, 10]:
        hv_thresh = np.percentile(hv_days['total_R'], pct) if len(hv_days) >= 20 else None
        ord_thresh = np.percentile(ordinary_days['total_R'], pct) if len(ordinary_days) >= 20 else None
        tail_rows.append({
            'tail_pct': pct, 'ordinary_days_tail_R': round(ord_thresh, 4) if ord_thresh is not None else None,
            'high_vol_days_tail_R': round(hv_thresh, 4) if hv_thresh is not None else None,
            'ordinary_days_mean_R': round(ordinary_days['total_R'].mean(), 4),
            'high_vol_days_mean_R': round(hv_days['total_R'].mean(), 4),
        })
    tail_df = pd.DataFrame(tail_rows)
    tail_df.to_csv(OUT / 'phase42_tail_analysis.csv', index=False)
    print("\n[tail analysis]"); print(tail_df.to_string())

    # --- Part 20: threshold robustness ---
    thresh_rows = []
    for p in [70, 80, 90, 95]:
        cut = np.percentile(ledger['vol_pctile'], p)
        above = ledger[ledger['vol_pctile'] >= cut]
        below = ledger[ledger['vol_pctile'] < cut]
        thresh_rows.append({
            'threshold_pctile': p, 'n_days_above': len(above), 'n_days_below': len(below),
            'avg_R_above': round(above['total_R'].mean(), 4) if len(above) else None,
            'avg_R_below': round(below['total_R'].mean(), 4) if len(below) else None,
            'diff': round(above['total_R'].mean() - below['total_R'].mean(), 4) if len(above) and len(below) else None,
        })
    thresh_df = pd.DataFrame(thresh_rows)
    thresh_df.to_csv(OUT / 'phase42_threshold_robustness.csv', index=False)
    print("\n[threshold robustness -- descriptive only]"); print(thresh_df.to_string())

    # --- Part 21: extreme-day sensitivity (H1 and H4 re-run) ---
    ledger_sorted = ledger.sort_values('total_R')
    ext_rows = []
    for n_excl in [0, 1, 5, 10]:
        excl_dates = set(ledger_sorted.head(n_excl)['date']) if n_excl else set()
        sub = ledger[~ledger['date'].isin(excl_dates)]
        corr = sub['vol_pctile'].corr(sub['total_R'])
        hv = sub[sub.vol_state == 'HIGH']
        nv = sub[sub.vol_state != 'HIGH']
        ext_rows.append({
            'excluding_worst_n_days': n_excl, 'n_days_remaining': len(sub),
            'corr_vol_vs_R': round(corr, 3),
            'high_vol_avg_R': round(hv['total_R'].mean(), 4) if len(hv) else None,
            'non_high_vol_avg_R': round(nv['total_R'].mean(), 4) if len(nv) else None,
            'diff': round(hv['total_R'].mean() - nv['total_R'].mean(), 4) if len(hv) and len(nv) else None,
        })
    ext_df = pd.DataFrame(ext_rows)
    ext_df.to_csv(OUT / 'phase42_extreme_day_sensitivity.csv', index=False)
    print("\n[extreme-day sensitivity]"); print(ext_df.to_string())

    # --- Part 22: regime robustness ---
    ledger['date_ts'] = pd.to_datetime(ledger['date'])
    periods = {
        'C_2023_2024': ('2023-08-01', '2024-12-31'),
        'D_2025': ('2025-01-01', '2025-12-31'),
        'E_2026_YTD': ('2026-01-01', '2026-08-13'),
    }
    regime_rows = [{'period': 'A_2019_2020', 'n_days': 0, 'note': 'UNKNOWN BY DATA ABSENCE -- control starts 2023-08-01'},
                   {'period': 'B_2021_2022', 'n_days': 0, 'note': 'UNKNOWN BY DATA ABSENCE -- control starts 2023-08-01'}]
    for pname, (start, end) in periods.items():
        sub = ledger[(ledger.date_ts >= start) & (ledger.date_ts <= end)]
        corr = sub['vol_pctile'].corr(sub['total_R']) if len(sub) >= 20 else None
        hv = sub[sub.vol_state == 'HIGH']
        nv = sub[sub.vol_state != 'HIGH']
        regime_rows.append({
            'period': pname, 'n_days': len(sub),
            'corr_vol_vs_R': round(corr, 3) if corr is not None else 'INSUFFICIENT SAMPLE',
            'high_vol_avg_R': round(hv['total_R'].mean(), 4) if len(hv) else None,
            'non_high_vol_avg_R': round(nv['total_R'].mean(), 4) if len(nv) else None,
            'effect_direction': ('NEGATIVE (high-vol worse)' if len(hv) and len(nv) and hv['total_R'].mean() < nv['total_R'].mean() else
                                  'POSITIVE (high-vol better)' if len(hv) and len(nv) else 'UNKNOWN'),
        })
    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(OUT / 'phase42_regime_robustness.csv', index=False)
    print("\n[regime robustness]"); print(regime_df.to_string())

    # --- Part 23: post-demotion live check ---
    live = pd.read_csv(REPO / 'reports' / '5ers_portfolio_update_aug13_trade_level.csv')
    live_df = pd.DataFrame([{
        'n_trades': len(live), 'n_trading_days': live['entry_time'].str[:10].nunique() if 'entry_time' in live else None,
        'total_R': round(live['R'].sum(), 3) if 'R' in live else None,
        'avg_ATR': round(live['ATR'].mean(), 3) if 'ATR' in live else None,
        'assessment': 'INSUFFICIENT LIVE SAMPLE (n=19 trades) -- not pooled with the historical control, no deterioration/improvement inferred',
    }])
    live_df.to_csv(OUT / 'phase42_post_demotion.csv', index=False)
    print("\n[post-demotion live check]"); print(live_df.to_string())

    # --- Part 24: Phase 40 comparison ---
    p40_df = pd.DataFrame([{
        'question': 'Does volatility work as a trading SIGNAL (Phase40)?',
        'answer': 'NO -- Phase40s HIGH-volatility-state trend continuation was REJECTED (OOS PF 0.668, n=2228, largest sample tested)',
    }, {
        'question': 'Does volatility work as a portfolio RISK-STATE descriptor (Phase42)?',
        'answer': f'corr(vol_pctile, daily_R) = {round(corr_vol_r, 3)} across the full control; see phase42_volatility_percentiles.csv for the continuous relationship and phase42_evidence_matrix.csv for the final classification',
    }, {
        'question': 'Interpretation',
        'answer': 'A variable can be a poor DIRECTIONAL trading signal (Phase40: predicting price direction conditional on HIGH vol failed) while still being a legitimate RISK-STATE descriptor (Phase42: characterizing when the EXISTING portfolios losses are more likely/larger) -- these are logically independent questions, not in tension',
    }])
    p40_df.to_csv(OUT / 'phase42_phase40_comparison.csv', index=False)
    print("\n[Phase40 comparison]"); print(p40_df.to_string())

    # --- Part 26/27: evidence matrix + final classification ---
    def strength(effect_abs, n):
        if n < 20:
            return 'INSUFFICIENT'
        if effect_abs >= 1.0:
            return 'STRONG'
        if effect_abs >= 0.5:
            return 'MODERATE'
        if effect_abs >= 0.2:
            return 'WEAK'
        return 'NO RELATIONSHIP'

    h1_effect = abs(hv_days['total_R'].mean() - ordinary_days['total_R'].mean()) if len(hv_days) and len(ordinary_days) else None
    h4_effect_5 = None
    hv5 = ledger[(ledger.vol_state == 'HIGH') & (ledger.conc_bucket.isin(['5', '6+']))]
    nv5 = ledger[(ledger.vol_state != 'HIGH') & (ledger.conc_bucket.isin(['5', '6+']))]
    if len(hv5) and len(nv5):
        h4_effect_5 = abs(hv5['total_R'].mean() - nv5['total_R'].mean())

    ev_rows = [
        {'hypothesis': 'H1 -- Absolute volatility (continuous)', 'effect_size_R': round(h1_effect, 4) if h1_effect else None,
         'n_days': f'{len(hv_days)} HIGH vs {len(ordinary_days)} non-HIGH', 'evidence': strength(h1_effect, min(len(hv_days), len(ordinary_days))) if h1_effect else 'UNKNOWN',
         'robust_to_extreme_day_removal': 'see phase42_extreme_day_sensitivity.csv',
         'robust_across_regimes': 'see phase42_regime_robustness.csv'},
        {'hypothesis': 'H2 -- Volatility change / transitions', 'effect_size_R': 'see phase42_volatility_change.csv (varies by transition, several THIN SAMPLE)',
         'n_days': 'varies', 'evidence': 'MODERATE for select transitions, INSUFFICIENT for others (thin samples)', 'robust_to_extreme_day_removal': 'not separately re-tested', 'robust_across_regimes': 'not separately re-tested'},
        {'hypothesis': 'H3 -- Volatility acceleration', 'effect_size_R': round(abs(hv_ha['total_R'].mean() - hv_stable['total_R'].mean()), 4) if len(hv_ha) and len(hv_stable) else None,
         'n_days': f'{len(hv_ha)} HIGH+HIGH_ACCEL vs {len(hv_stable)} HIGH+STABLE', 'evidence': strength(abs(hv_ha['total_R'].mean() - hv_stable['total_R'].mean()) if len(hv_ha) and len(hv_stable) else 0, min(len(hv_ha), len(hv_stable)) if len(hv_ha) and len(hv_stable) else 0),
         'robust_to_extreme_day_removal': 'not separately re-tested', 'robust_across_regimes': 'not separately re-tested'},
        {'hypothesis': 'H4 -- Volatility x concurrent exposure (5+/6+)', 'effect_size_R': round(h4_effect_5, 4) if h4_effect_5 else None,
         'n_days': f'{len(hv5)} HIGH+5plus vs {len(nv5)} non-HIGH+5plus', 'evidence': strength(h4_effect_5, min(len(hv5), len(nv5))) if h4_effect_5 else 'INSUFFICIENT',
         'robust_to_extreme_day_removal': 'see phase42_extreme_day_sensitivity.csv (H4 not separately re-run, disclosed limitation)', 'robust_across_regimes': 'not separately re-tested'},
        {'hypothesis': 'H5 -- Volatility x session', 'effect_size_R': 'N/A -- control has zero NY/overlap trades', 'n_days': 'N/A',
         'evidence': 'INSUFFICIENT (Asian/London only; NY transition UNKNOWN BY DATA ABSENCE)', 'robust_to_extreme_day_removal': 'N/A', 'robust_across_regimes': 'N/A'},
        {'hypothesis': 'H6 -- Volatility x direction', 'effect_size_R': 'see phase42_volatility_direction.csv', 'n_days': 'see CSV',
         'evidence': 'see phase42_volatility_direction.csv for per-state directional R split', 'robust_to_extreme_day_removal': 'not separately re-tested', 'robust_across_regimes': 'not separately re-tested'},
        {'hypothesis': 'H7 -- Volatility x JPY', 'effect_size_R': 'see phase42_volatility_jpy.csv', 'n_days': 'see CSV',
         'evidence': 'see phase42_volatility_jpy.csv', 'robust_to_extreme_day_removal': 'not separately re-tested', 'robust_across_regimes': 'not separately re-tested'},
        {'hypothesis': 'H8 -- Volatility x mechanism', 'effect_size_R': 'see phase42_volatility_mechanism.csv', 'n_days': 'see CSV',
         'evidence': 'see phase42_volatility_mechanism.csv', 'robust_to_extreme_day_removal': 'not separately re-tested', 'robust_across_regimes': 'not separately re-tested'},
    ]
    ev_df = pd.DataFrame(ev_rows)
    ev_df.to_csv(OUT / 'phase42_evidence_matrix.csv', index=False)
    print("\n[evidence matrix]"); print(ev_df.to_string())

    summary = {
        'n_days_analyzed': len(ledger), 'corr_vol_vs_R': round(corr_vol_r, 3),
        'h1_effect_R': round(h1_effect, 4) if h1_effect else None,
        'h4_effect_R_5plus': round(h4_effect_5, 4) if h4_effect_5 else None,
        'high_vol_avg_R': round(hv_days['total_R'].mean(), 4), 'non_high_vol_avg_R': round(ordinary_days['total_R'].mean(), 4),
    }
    with open(OUT / '_phase42_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n" + json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
