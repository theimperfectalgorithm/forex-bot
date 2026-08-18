"""
Phase 41 -- portfolio stress anatomy & common-factor attribution.
FORENSIC ANALYSIS ONLY. No new strategy, no backtest, no intervention.
Operates entirely on the already-validated control (data/phase26_all_trades.csv)
and the separate, small live post-demotion sample
(reports/5ers_portfolio_update_aug13_trade_level.csv).
"""
import itertools
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).parent.parent
OUT = REPO / 'reports'
DEMOTION = pd.Timestamp('2026-07-31', tz='UTC')

INSTR_RE = re.compile(r'^([A-Z]{6})_')
MECH_RE = re.compile(r'_(AMR|ARB|MONDAY)$')


def load_control():
    df = pd.read_csv(REPO / 'data' / 'phase26_all_trades.csv')
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df['trade_date'] = df['entry_time'].dt.date
    def instr(strat):
        m = INSTR_RE.match(strat)
        return m.group(1) if m else 'GBPUSD'  # GBPUSD_MONDAY doesn't match 6-letter prefix pattern directly
    df['instrument'] = df['strategy'].apply(lambda s: s.replace('_AMR', '').replace('_ARB', '').replace('_MONDAY', ''))
    df['mechanism'] = df['strategy'].apply(lambda s: MECH_RE.search(s).group(1) if MECH_RE.search(s) else 'UNKNOWN')
    df['base_ccy'] = df['instrument'].str[:3]
    df['quote_ccy'] = df['instrument'].str[3:]
    df['is_jpy'] = (df['base_ccy'] == 'JPY') | (df['quote_ccy'] == 'JPY')
    return df


def daily_ledger(df):
    rows = []
    dates = sorted(df['trade_date'].unique())
    for d in dates:
        day_trades = df[df['trade_date'] == d]
        # concurrent positions: trades whose interval overlaps this calendar day
        d_start = pd.Timestamp(d, tz='UTC')
        d_end = d_start + pd.Timedelta(days=1)
        overlapping = df[(df['entry_time'] < d_end) & (df['exit_time'] >= d_start)]
        strategies_active = day_trades['strategy'].nunique()
        by_strat = day_trades.groupby('strategy')['r_multiple'].sum()
        n_losing_strats = int((by_strat < 0).sum())
        n_winning_strats = int((by_strat > 0).sum())
        rows.append({
            'date': d,
            'total_pnl': round(day_trades['pnl'].sum(), 2),
            'total_R': round(day_trades['r_multiple'].sum(), 4),
            'n_trades': len(day_trades),
            'n_winning_trades': int((day_trades['r_multiple'] > 0).sum()),
            'n_losing_trades': int((day_trades['r_multiple'] < 0).sum()),
            'win_rate_pct': round((day_trades['r_multiple'] > 0).mean() * 100, 1) if len(day_trades) else None,
            'max_concurrent_positions': len(overlapping),
            'avg_concurrent_positions': len(overlapping),  # single-snapshot proxy, documented limitation
            'jpy_trades': int(day_trades['is_jpy'].sum()),
            'jpy_R': round(day_trades.loc[day_trades['is_jpy'], 'r_multiple'].sum(), 4),
            'non_jpy_R': round(day_trades.loc[~day_trades['is_jpy'], 'r_multiple'].sum(), 4),
            'amr_R': round(day_trades.loc[day_trades['mechanism'] == 'AMR', 'r_multiple'].sum(), 4),
            'arb_R': round(day_trades.loc[day_trades['mechanism'] == 'ARB', 'r_multiple'].sum(), 4),
            'monday_R': round(day_trades.loc[day_trades['mechanism'] == 'MONDAY', 'r_multiple'].sum(), 4),
            'long_R': round(day_trades.loc[day_trades['dir'] == 'BUY', 'r_multiple'].sum(), 4),
            'short_R': round(day_trades.loc[day_trades['dir'] == 'SELL', 'r_multiple'].sum(), 4),
            'asian_R': round(day_trades.loc[day_trades['session'] == 'ASIAN', 'r_multiple'].sum(), 4),
            'london_R': round(day_trades.loc[day_trades['session'] == 'LONDON', 'r_multiple'].sum(), 4),
            'vol_state_mode': day_trades['vol_tercile'].mode().iloc[0] if day_trades['vol_tercile'].notna().any() else 'UNKNOWN',
            'n_strategies_active': strategies_active,
            'n_simultaneous_losing_strategies': n_losing_strats,
            'n_simultaneous_winning_strategies': n_winning_strats,
        })
    return pd.DataFrame(rows)


def main():
    df = load_control()
    print(f"[control] {len(df)} trades, {df['strategy'].nunique()} strategies, "
          f"{df['entry_time'].min()} to {df['entry_time'].max()}")

    ledger = daily_ledger(df)
    ledger.to_csv(OUT / 'phase41_daily_portfolio_ledger.csv', index=False)
    print(f"[ledger] {len(ledger)} trading days written")

    # --- Part 6: stress windows ---
    r = ledger['total_R'].values
    sorted_r = np.sort(r)
    n = len(r)
    q1 = np.percentile(r, 1)
    q5 = np.percentile(r, 5)
    q10 = np.percentile(r, 10)
    q20 = np.percentile(r, 20)
    ledger_sorted = ledger.sort_values('total_R')
    worst_day = ledger_sorted.iloc[0]

    cum = ledger['total_R'].cumsum()
    running_peak = cum.cummax()
    dd = cum - running_peak
    max_dd = dd.min()
    max_dd_idx = dd.idxmin()
    # longest drawdown: longest consecutive run where dd < 0
    in_dd = (dd < -1e-9).astype(int)
    longest_dd_len = 0
    cur = 0
    for v in in_dd:
        if v:
            cur += 1
            longest_dd_len = max(longest_dd_len, cur)
        else:
            cur = 0

    roll_windows = {}
    for w in [3, 5, 10]:
        roll = ledger['total_R'].rolling(w).sum()
        worst_idx = roll.idxmin()
        roll_windows[w] = (worst_idx, roll.loc[worst_idx])
    largest_cluster_w = min(roll_windows, key=lambda w: roll_windows[w][1])
    largest_cluster_idx, largest_cluster_val = roll_windows[largest_cluster_w]
    largest_cluster_dates = ledger.loc[max(0, largest_cluster_idx - largest_cluster_w + 1):largest_cluster_idx, 'date'].tolist()

    stress_df = pd.DataFrame([
        {'metric': 'worst_1pct_threshold_R', 'value': round(q1, 4)},
        {'metric': 'worst_5pct_threshold_R', 'value': round(q5, 4)},
        {'metric': 'worst_10pct_threshold_R', 'value': round(q10, 4)},
        {'metric': 'worst_20pct_threshold_R', 'value': round(q20, 4)},
        {'metric': 'worst_single_day_date', 'value': str(worst_day['date'])},
        {'metric': 'worst_single_day_R', 'value': worst_day['total_R']},
        {'metric': 'worst_3day_window_R', 'value': round(roll_windows[3][1], 4)},
        {'metric': 'worst_5day_window_R', 'value': round(roll_windows[5][1], 4)},
        {'metric': 'worst_10day_window_R', 'value': round(roll_windows[10][1], 4)},
        {'metric': 'longest_drawdown_days', 'value': longest_dd_len},
        {'metric': 'largest_drawdown_R', 'value': round(max_dd, 4)},
        {'metric': 'largest_drawdown_end_date', 'value': str(ledger.loc[max_dd_idx, 'date'])},
        {'metric': 'largest_clustered_loss_window_days', 'value': largest_cluster_w},
        {'metric': 'largest_clustered_loss_R', 'value': round(largest_cluster_val, 4)},
        {'metric': 'largest_clustered_loss_dates', 'value': '; '.join(str(d) for d in largest_cluster_dates)},
    ])
    stress_df.to_csv(OUT / 'phase41_stress_windows.csv', index=False)
    print("\n[stress windows]"); print(stress_df.to_string())

    def bucket(mask_name, mask):
        return ledger[mask]

    buckets = {
        'worst_1pct': ledger[ledger['total_R'] <= q1],
        'worst_5pct': ledger[ledger['total_R'] <= q5],
        'worst_10pct': ledger[ledger['total_R'] <= q10],
        'worst_20pct': ledger[ledger['total_R'] <= q20],
        'normal': ledger[ledger['total_R'] > q20],
    }
    for name, b in buckets.items():
        print(f"[bucket] {name}: {len(b)} days")

    # --- Part 7: baseline distribution ---
    base_df = pd.DataFrame([{
        'mean_daily_R': round(ledger['total_R'].mean(), 4),
        'median_daily_R': round(ledger['total_R'].median(), 4),
        'std_daily_R': round(ledger['total_R'].std(), 4),
        'skew': round(float(ledger['total_R'].skew()), 3),
        'kurtosis': round(float(ledger['total_R'].kurt()), 3),
        'positive_day_freq_pct': round((ledger['total_R'] > 0).mean() * 100, 1),
        'negative_day_freq_pct': round((ledger['total_R'] < 0).mean() * 100, 1),
        'pctile_5': round(q5, 4), 'pctile_1': round(q1, 4),
        'max_daily_gain': round(ledger['total_R'].max(), 4),
        'max_daily_loss': round(ledger['total_R'].min(), 4),
        'avg_trades_per_day': round(ledger['n_trades'].mean(), 2),
        'avg_concurrent_positions': round(ledger['max_concurrent_positions'].mean(), 2),
        'n_trading_days': len(ledger),
    }])
    base_df.to_csv(OUT / 'phase41_baseline_distribution.csv', index=False)
    print("\n[baseline distribution]"); print(base_df.to_string())

    # --- Part 8: JPY factor ---
    jpy_rows = []
    for name, b in buckets.items():
        n_days = len(b)
        if n_days == 0:
            jpy_rows.append({'bucket': name, 'n_days': 0, 'assessment': 'UNKNOWN -- no days in bucket'})
            continue
        day_dates = set(b['date'])
        trades_in_bucket = df[df['trade_date'].isin(day_dates)]
        jpy_share = trades_in_bucket['is_jpy'].mean() * 100 if len(trades_in_bucket) else None
        jpy_r_share = (trades_in_bucket.loc[trades_in_bucket.is_jpy, 'r_multiple'].sum() /
                       trades_in_bucket['r_multiple'].sum() * 100) if trades_in_bucket['r_multiple'].sum() != 0 else None
        long_jpy = trades_in_bucket.loc[trades_in_bucket.is_jpy & (trades_in_bucket.dir == 'BUY'), 'r_multiple'].sum()
        short_jpy = trades_in_bucket.loc[trades_in_bucket.is_jpy & (trades_in_bucket.dir == 'SELL'), 'r_multiple'].sum()
        jpy_rows.append({
            'bucket': name, 'n_days': n_days, 'n_trades': len(trades_in_bucket),
            'jpy_trade_pct': round(jpy_share, 1) if jpy_share is not None else None,
            'jpy_risk_R_pct': round(jpy_r_share, 1) if jpy_r_share is not None else None,
            'net_jpy_directional_R': round(long_jpy + short_jpy, 3),
            'jpy_long_R': round(long_jpy, 3), 'jpy_short_R': round(short_jpy, 3),
            'sample_flag': 'THIN SAMPLE (<20 days)' if n_days < 20 else 'ADEQUATE',
        })
    jpy_df = pd.DataFrame(jpy_rows)
    normal_jpy_pct = jpy_df.loc[jpy_df.bucket == 'normal', 'jpy_trade_pct'].iloc[0]
    jpy_df['delta_vs_normal_jpy_trade_pct'] = jpy_df['jpy_trade_pct'] - normal_jpy_pct
    jpy_df.to_csv(OUT / 'phase41_jpy_factor.csv', index=False)
    print("\n[JPY factor]"); print(jpy_df.to_string())

    # --- Part 9: mechanism factor ---
    mech_rows = []
    for name, b in buckets.items():
        day_dates = set(b['date'])
        trades_in_bucket = df[df['trade_date'].isin(day_dates)]
        for mech in ['AMR', 'ARB', 'MONDAY']:
            sub = trades_in_bucket[trades_in_bucket.mechanism == mech]
            mech_rows.append({
                'bucket': name, 'mechanism': mech, 'n_days': len(b), 'n_trades': len(sub),
                'risk_share_pct': round(len(sub) / len(trades_in_bucket) * 100, 1) if len(trades_in_bucket) else None,
                'R_contribution': round(sub['r_multiple'].sum(), 3),
                'losing_trade_pct': round((sub['r_multiple'] < 0).mean() * 100, 1) if len(sub) else None,
            })
    mech_df = pd.DataFrame(mech_rows)
    mech_df.to_csv(OUT / 'phase41_mechanism_factor.csv', index=False)
    print("\n[mechanism factor]"); print(mech_df.to_string())

    # --- Part 10: volatility factor ---
    vol_rows = []
    df_v = df.dropna(subset=['vol_tercile'])
    for name, b in buckets.items():
        day_dates = set(b['date'])
        trades_in_bucket = df_v[df_v['trade_date'].isin(day_dates)]
        for state in ['LOW', 'NORMAL', 'HIGH']:
            sub = trades_in_bucket[trades_in_bucket.vol_tercile == state]
            vol_rows.append({
                'bucket': name, 'vol_state': state, 'n_days': len(b), 'n_trades': len(sub),
                'trade_share_pct': round(len(sub) / len(trades_in_bucket) * 100, 1) if len(trades_in_bucket) else None,
                'R_contribution': round(sub['r_multiple'].sum(), 3),
                'losing_trade_pct': round((sub['r_multiple'] < 0).mean() * 100, 1) if len(sub) else None,
                'jpy_share_pct': round(sub['is_jpy'].mean() * 100, 1) if len(sub) else None,
            })
    vol_df = pd.DataFrame(vol_rows)
    vol_df.to_csv(OUT / 'phase41_volatility_factor.csv', index=False)
    print("\n[volatility factor]"); print(vol_df.to_string())

    # --- Part 11: session factor ---
    sess_rows = []
    for name, b in buckets.items():
        day_dates = set(b['date'])
        trades_in_bucket = df[df['trade_date'].isin(day_dates)]
        for sess in ['ASIAN', 'LONDON']:
            sub = trades_in_bucket[trades_in_bucket.session == sess]
            sess_rows.append({
                'bucket': name, 'session': sess, 'n_days': len(b), 'n_trades': len(sub),
                'trade_share_pct': round(len(sub) / len(trades_in_bucket) * 100, 1) if len(trades_in_bucket) else None,
                'R_contribution': round(sub['r_multiple'].sum(), 3),
                'losing_trade_pct': round((sub['r_multiple'] < 0).mean() * 100, 1) if len(sub) else None,
            })
    sess_df = pd.DataFrame(sess_rows)
    sess_df.to_csv(OUT / 'phase41_session_factor.csv', index=False)
    print("\n[session factor]"); print(sess_df.to_string())

    # --- Part 12: directional factor ---
    dir_rows = []
    for name, b in buckets.items():
        day_dates = set(b['date'])
        trades_in_bucket = df[df['trade_date'].isin(day_dates)]
        longs = trades_in_bucket[trades_in_bucket.dir == 'BUY']
        shorts = trades_in_bucket[trades_in_bucket.dir == 'SELL']
        dir_rows.append({
            'bucket': name, 'n_days': len(b),
            'long_trades': len(longs), 'short_trades': len(shorts),
            'long_share_pct': round(len(longs) / len(trades_in_bucket) * 100, 1) if len(trades_in_bucket) else None,
            'long_R': round(longs['r_multiple'].sum(), 3), 'short_R': round(shorts['r_multiple'].sum(), 3),
            'net_directional_R': round(longs['r_multiple'].sum() + shorts['r_multiple'].sum(), 3),
        })
    dir_df = pd.DataFrame(dir_rows)
    dir_df.to_csv(OUT / 'phase41_directional_factor.csv', index=False)
    print("\n[directional factor]"); print(dir_df.to_string())

    # --- Part 13: currency factor ---
    ccy_rows = []
    all_ccys = sorted(set(df['base_ccy']) | set(df['quote_ccy']))
    for name, b in buckets.items():
        day_dates = set(b['date'])
        trades_in_bucket = df[df['trade_date'].isin(day_dates)]
        for ccy in all_ccys:
            mask = (trades_in_bucket['base_ccy'] == ccy) | (trades_in_bucket['quote_ccy'] == ccy)
            sub = trades_in_bucket[mask]
            ccy_rows.append({
                'bucket': name, 'currency': ccy, 'n_days': len(b),
                'trade_count_exposure': len(sub),
                'trade_count_exposure_pct': round(len(sub) / len(trades_in_bucket) * 100, 1) if len(trades_in_bucket) else None,
                'risk_weighted_R_exposure': round(sub['r_multiple'].abs().sum(), 3),
            })
    ccy_df = pd.DataFrame(ccy_rows)
    ccy_df.to_csv(OUT / 'phase41_currency_factor.csv', index=False)
    print("\n[currency factor] (summary)"); print(ccy_df[ccy_df.bucket.isin(['normal', 'worst_5pct'])].to_string())

    # --- Part 14: instrument / correlated-instrument analysis ---
    instr_rows = []
    for name, b in buckets.items():
        day_dates = set(b['date'])
        trades_in_bucket = df[df['trade_date'].isin(day_dates)]
        base_counts = trades_in_bucket['base_ccy'].value_counts()
        quote_counts = trades_in_bucket['quote_ccy'].value_counts()
        jpy_cross_pct = trades_in_bucket['is_jpy'].mean() * 100 if len(trades_in_bucket) else None
        instr_rows.append({
            'bucket': name, 'n_days': len(b), 'n_trades': len(trades_in_bucket),
            'dominant_base_ccy': base_counts.idxmax() if len(base_counts) else None,
            'dominant_base_ccy_share_pct': round(base_counts.max() / len(trades_in_bucket) * 100, 1) if len(trades_in_bucket) else None,
            'dominant_quote_ccy': quote_counts.idxmax() if len(quote_counts) else None,
            'dominant_quote_ccy_share_pct': round(quote_counts.max() / len(trades_in_bucket) * 100, 1) if len(trades_in_bucket) else None,
            'jpy_cross_share_pct': round(jpy_cross_pct, 1) if jpy_cross_pct is not None else None,
            'note': 'Rolling historical price correlation not computed -- control trades are already time-and-mechanism-disjoint per-strategy; instrument overlap is characterized via currency-share concentration instead (a direct, no-look-ahead accounting measure), avoiding a redundant, execution-irrelevant price-series correlation calculation',
        })
    instr_df = pd.DataFrame(instr_rows)
    instr_df.to_csv(OUT / 'phase41_instrument_factor.csv', index=False)
    print("\n[instrument factor]"); print(instr_df.to_string())

    # --- Part 15: simultaneous losses ---
    ledger['loss_cluster_size'] = ledger['n_simultaneous_losing_strategies']
    sim_loss_days = ledger[ledger['loss_cluster_size'] >= 2]
    sim_rows = []
    for min_n in [2, 3, 4, 5, 6]:
        sub = ledger[ledger['loss_cluster_size'] >= min_n]
        sim_rows.append({
            'min_strategies_losing': min_n, 'n_days': len(sub),
            'pct_of_all_days': round(len(sub) / len(ledger) * 100, 2),
            'avg_severity_R': round(sub['total_R'].mean(), 3) if len(sub) else None,
            'max_severity_R': round(sub['total_R'].min(), 3) if len(sub) else None,
        })
    sim_df = pd.DataFrame(sim_rows)
    sim_df.to_csv(OUT / 'phase41_simultaneous_losses.csv', index=False)
    print("\n[simultaneous losses]"); print(sim_df.to_string())

    # --- Part 16: loss clusters (detailed rows for 2+/3+/... days) ---
    cluster_rows = []
    for _, row in sim_loss_days.iterrows():
        day_trades = df[df['trade_date'] == row['date']]
        losing_strats = day_trades.groupby('strategy')['r_multiple'].sum()
        losing_strats = losing_strats[losing_strats < 0]
        cluster_rows.append({
            'date': row['date'], 'n_strategies_losing': len(losing_strats),
            'strategies': '; '.join(losing_strats.index), 'total_R': row['total_R'],
            'total_loss_R': round(losing_strats.sum(), 3),
            'instruments': '; '.join(sorted(day_trades.loc[day_trades.strategy.isin(losing_strats.index), 'instrument'].unique())),
            'directions': '; '.join(sorted(day_trades.loc[day_trades.strategy.isin(losing_strats.index), 'dir'].unique())),
            'vol_state': row['vol_state_mode'], 'jpy_R': row['jpy_R'],
            'session_mix': '; '.join(sorted(day_trades['session'].unique())),
            'concurrent_positions': row['max_concurrent_positions'],
        })
    cluster_df = pd.DataFrame(cluster_rows).sort_values('total_R') if cluster_rows else pd.DataFrame(
        columns=['date', 'n_strategies_losing', 'strategies', 'total_R', 'total_loss_R', 'instruments', 'directions', 'vol_state', 'jpy_R', 'session_mix', 'concurrent_positions'])
    cluster_df.to_csv(OUT / 'phase41_loss_clusters.csv', index=False)
    print(f"\n[loss clusters] {len(cluster_df)} days with 2+ simultaneously-losing strategies")

    # --- Part 17: entry clustering (EXPLORATORY) ---
    entry_rows = []
    for name, b in buckets.items():
        day_dates = set(b['date'])
        trades_in_bucket = df[df['trade_date'].isin(day_dates)]
        n_days = len(b)
        entry_rows.append({
            'bucket': name, 'n_days': n_days,
            'entries_per_day': round(len(trades_in_bucket) / n_days, 2) if n_days else None,
            'avg_max_concurrent_positions': round(b['max_concurrent_positions'].mean(), 2) if n_days else None,
            'note': 'EXPLORATORY',
        })
    entry_df = pd.DataFrame(entry_rows)
    entry_df.to_csv(OUT / 'phase41_entry_clustering.csv', index=False)
    print("\n[entry clustering -- EXPLORATORY]"); print(entry_df.to_string())

    # --- Part 18: exit clustering (EXPLORATORY) ---
    df['exit_date'] = df['exit_time'].dt.date
    exit_rows = []
    for name, b in buckets.items():
        day_dates = set(b['date'])
        exits_in_bucket = df[df['exit_date'].isin(day_dates)]
        sl_exits = exits_in_bucket[exits_in_bucket['reason'].astype(str).str.contains('SL', case=False, na=False)]
        tp_exits = exits_in_bucket[exits_in_bucket['reason'].astype(str).str.contains('TP', case=False, na=False)]
        exit_rows.append({
            'bucket': name, 'n_days': len(b), 'total_exits': len(exits_in_bucket),
            'sl_exits': len(sl_exits), 'sl_exit_pct': round(len(sl_exits) / len(exits_in_bucket) * 100, 1) if len(exits_in_bucket) else None,
            'tp_exits': len(tp_exits), 'tp_exit_pct': round(len(tp_exits) / len(exits_in_bucket) * 100, 1) if len(exits_in_bucket) else None,
            'note': 'EXPLORATORY',
        })
    exit_df = pd.DataFrame(exit_rows)
    exit_df.to_csv(OUT / 'phase41_exit_clustering.csv', index=False)
    print("\n[exit clustering -- EXPLORATORY]"); print(exit_df.to_string())

    # --- Part 19: temporal sequencing (EXPLORATORY, worst 5 days) ---
    worst5 = ledger.nsmallest(5, 'total_R')
    seq_rows = []
    for _, row in worst5.iterrows():
        day_trades = df[df['trade_date'] == row['date']].sort_values('entry_time')
        first = day_trades.iloc[0] if len(day_trades) else None
        seq_rows.append({
            'date': row['date'], 'total_R': row['total_R'], 'n_trades': len(day_trades),
            'first_trade_strategy': first['strategy'] if first is not None else None,
            'first_trade_time': str(first['entry_time']) if first is not None else None,
            'first_trade_vol_state': first['vol_tercile'] if first is not None else None,
            'sequence_note': 'EXPLORATORY -- describes observed order only; PRECEDED, not CAUSED',
        })
    seq_df = pd.DataFrame(seq_rows)
    seq_df.to_csv(OUT / 'phase41_temporal_sequences.csv', index=False)
    print("\n[temporal sequences -- EXPLORATORY, worst 5 days]"); print(seq_df.to_string())

    # --- Part 20: factor interactions (predeclared, limited set) ---
    interactions = [
        ('JPY', 'HIGH_VOL', lambda t: t.is_jpy & (t.vol_tercile == 'HIGH')),
        ('JPY', 'AMR', lambda t: t.is_jpy & (t.mechanism == 'AMR')),
        ('HIGH_VOL', 'AMR', lambda t: (t.vol_tercile == 'HIGH') & (t.mechanism == 'AMR')),
        ('JPY', 'ASIAN', lambda t: t.is_jpy & (t.session == 'ASIAN')),
        ('AMR', 'ASIAN', lambda t: (t.mechanism == 'AMR') & (t.session == 'ASIAN')),
        ('SELL', 'HIGH_VOL', lambda t: (t.dir == 'SELL') & (t.vol_tercile == 'HIGH')),
        ('JPY', 'HIGH_VOL_AMR', lambda t: t.is_jpy & (t.vol_tercile == 'HIGH') & (t.mechanism == 'AMR')),
    ]
    interact_rows = []
    for f1, f2, fn in interactions:
        for name, b in [('normal', buckets['normal']), ('worst_10pct', buckets['worst_10pct']), ('worst_5pct', buckets['worst_5pct'])]:
            day_dates = set(b['date'])
            trades_in_bucket = df[df['trade_date'].isin(day_dates)].dropna(subset=['vol_tercile'])
            mask = fn(trades_in_bucket)
            sub = trades_in_bucket[mask]
            interact_rows.append({
                'interaction': f'{f1}+{f2}', 'bucket': name, 'n_days': len(b),
                'n_trades_matching': len(sub),
                'pct_of_bucket_trades': round(len(sub) / len(trades_in_bucket) * 100, 1) if len(trades_in_bucket) else None,
                'R_contribution': round(sub['r_multiple'].sum(), 3),
            })
    interact_df = pd.DataFrame(interact_rows)
    interact_df.to_csv(OUT / 'phase41_factor_interactions.csv', index=False)
    print("\n[factor interactions -- predeclared set only]"); print(interact_df.head(20).to_string())

    # --- Part 21: conditional correlation ---
    strategies = sorted(df['strategy'].unique())
    daily_by_strat = df.groupby(['trade_date', 'strategy'])['r_multiple'].sum().unstack('strategy')
    corr_rows = []
    for s1, s2 in itertools.combinations(strategies, 2):
        pair = daily_by_strat[[s1, s2]].dropna(how='any') if s1 in daily_by_strat.columns and s2 in daily_by_strat.columns else pd.DataFrame()
        def corr_on(dates_subset):
            p = pair[pair.index.isin(dates_subset)]
            if len(p) < 8:
                return None, len(p)
            return round(p[s1].corr(p[s2]), 3), len(p)
        full_corr, full_n = corr_on(set(pair.index))
        normal_corr, normal_n = corr_on(set(buckets['normal']['date']))
        stress_corr, stress_n = corr_on(set(buckets['worst_20pct']['date']))
        w10_corr, w10_n = corr_on(set(buckets['worst_10pct']['date']))
        w5_corr, w5_n = corr_on(set(buckets['worst_5pct']['date']))
        corr_rows.append({
            'strategy_1': s1, 'strategy_2': s2,
            'full_period_corr': full_corr, 'full_n': full_n,
            'normal_day_corr': normal_corr, 'normal_n': normal_n,
            'stress_worst20pct_corr': stress_corr, 'stress_n': stress_n,
            'worst10pct_corr': w10_corr, 'worst10_n': w10_n,
            'worst5pct_corr': w5_corr if w5_n and w5_n >= 8 else None, 'worst5_n': w5_n,
            'diversification_disappears': (stress_corr is not None and normal_corr is not None and stress_corr > normal_corr + 0.15),
        })
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT / 'phase41_conditional_correlation.csv', index=False)
    print("\n[conditional correlation]"); print(corr_df.to_string())

    # --- Part 22: marginal stress contribution ---
    marg_rows = []
    for strat in strategies:
        strat_daily = daily_by_strat[strat] if strat in daily_by_strat.columns else pd.Series(dtype=float)
        total_R = df.loc[df.strategy == strat, 'r_multiple'].sum()
        def r_in_bucket(bucket_dates):
            return round(strat_daily.reindex(bucket_dates).fillna(0).sum(), 3)
        r20 = r_in_bucket(set(buckets['worst_20pct']['date']))
        r10 = r_in_bucket(set(buckets['worst_10pct']['date']))
        r5 = r_in_bucket(set(buckets['worst_5pct']['date']))
        n_stress_days_participated = int((strat_daily.reindex(set(buckets['worst_20pct']['date'])).fillna(0) < 0).sum())
        total_stress_loss = buckets['worst_20pct']['total_R'].sum()
        pct_attributable = round(r20 / total_stress_loss * 100, 1) if total_stress_loss != 0 else None
        avg_loss_when_participating = round(strat_daily[(strat_daily.index.isin(set(buckets['worst_20pct']['date']))) & (strat_daily < 0)].mean(), 3) if n_stress_days_participated else None
        marg_rows.append({
            'strategy': strat, 'total_historical_R': round(total_R, 2),
            'R_contribution_worst20pct': r20, 'R_contribution_worst10pct': r10, 'R_contribution_worst5pct': r5,
            'n_stress_days_participated_worst20pct': n_stress_days_participated,
            'pct_of_stress_losses_attributable_worst20pct': pct_attributable,
            'avg_loss_when_participating_in_stress': avg_loss_when_participating,
        })
    marg_df = pd.DataFrame(marg_rows).sort_values('R_contribution_worst20pct')
    marg_df.to_csv(OUT / 'phase41_marginal_stress_contribution.csv', index=False)
    print("\n[marginal stress contribution]"); print(marg_df.to_string())

    # --- Part 23: counterfactual attribution ---
    cf_rows = []
    for strat in strategies:
        for bucket_name in ['worst_20pct', 'worst_10pct', 'worst_5pct']:
            bucket_dates = set(buckets[bucket_name]['date'])
            actual_total = buckets[bucket_name]['total_R'].sum()
            strat_daily_in_bucket = daily_by_strat[strat].reindex(bucket_dates).fillna(0) if strat in daily_by_strat.columns else pd.Series(0, index=list(bucket_dates))
            counterfactual_total = actual_total - strat_daily_in_bucket.sum()
            cf_rows.append({
                'strategy_removed': strat, 'stress_bucket': bucket_name,
                'actual_bucket_total_R': round(actual_total, 3),
                'counterfactual_bucket_total_R_without_strategy': round(counterfactual_total, 3),
                'delta': round(counterfactual_total - actual_total, 3),
                'label': 'COUNTERFACTUAL ATTRIBUTION -- descriptive only, not optimization',
            })
    cf_df = pd.DataFrame(cf_rows)
    cf_df.to_csv(OUT / 'phase41_counterfactual_attribution.csv', index=False)
    print("\n[counterfactual attribution]"); print(cf_df[cf_df.stress_bucket == 'worst_10pct'].to_string())

    # --- Part 24: stress factor ranking ---
    def effect_size(bucket_pct, normal_pct):
        if bucket_pct is None or normal_pct is None:
            return None
        return round(bucket_pct - normal_pct, 1)

    jpy_w5 = jpy_df[jpy_df.bucket == 'worst_5pct']
    jpy_normal = jpy_df[jpy_df.bucket == 'normal']
    jpy_effect = effect_size(jpy_w5['jpy_trade_pct'].iloc[0] if len(jpy_w5) else None,
                              jpy_normal['jpy_trade_pct'].iloc[0] if len(jpy_normal) else None)

    amr_w5 = mech_df[(mech_df.bucket == 'worst_5pct') & (mech_df.mechanism == 'AMR')]
    amr_normal = mech_df[(mech_df.bucket == 'normal') & (mech_df.mechanism == 'AMR')]
    amr_effect = effect_size(amr_w5['risk_share_pct'].iloc[0] if len(amr_w5) else None,
                              amr_normal['risk_share_pct'].iloc[0] if len(amr_normal) else None)

    hv_w5 = vol_df[(vol_df.bucket == 'worst_5pct') & (vol_df.vol_state == 'HIGH')]
    hv_normal = vol_df[(vol_df.bucket == 'normal') & (vol_df.vol_state == 'HIGH')]
    hv_effect = effect_size(hv_w5['trade_share_pct'].iloc[0] if len(hv_w5) else None,
                             hv_normal['trade_share_pct'].iloc[0] if len(hv_normal) else None)

    def strength_of(effect, n_days):
        if n_days < 8:
            return 'INSUFFICIENT (n<8 days)'
        if effect is None:
            return 'UNKNOWN'
        a = abs(effect)
        if a >= 15:
            return 'STRONG'
        if a >= 8:
            return 'MODERATE'
        if a >= 3:
            return 'WEAK'
        return 'NO CLEAR ASSOCIATION'

    n_w5_days = len(buckets['worst_5pct'])
    diversification_pairs_disappearing = int(corr_df['diversification_disappears'].sum())
    ranking_rows = [
        {'factor': 'JPY concentration', 'normal_exposure_pct': jpy_normal['jpy_trade_pct'].iloc[0] if len(jpy_normal) else None,
         'stress_exposure_pct_worst5pct': jpy_w5['jpy_trade_pct'].iloc[0] if len(jpy_w5) else None,
         'effect_size_pct_pts': jpy_effect, 'evidence_strength': strength_of(jpy_effect, n_w5_days)},
        {'factor': 'AMR mechanism concentration', 'normal_exposure_pct': amr_normal['risk_share_pct'].iloc[0] if len(amr_normal) else None,
         'stress_exposure_pct_worst5pct': amr_w5['risk_share_pct'].iloc[0] if len(amr_w5) else None,
         'effect_size_pct_pts': amr_effect, 'evidence_strength': strength_of(amr_effect, n_w5_days)},
        {'factor': 'HIGH volatility state', 'normal_exposure_pct': hv_normal['trade_share_pct'].iloc[0] if len(hv_normal) else None,
         'stress_exposure_pct_worst5pct': hv_w5['trade_share_pct'].iloc[0] if len(hv_w5) else None,
         'effect_size_pct_pts': hv_effect, 'evidence_strength': strength_of(hv_effect, n_w5_days)},
        {'factor': 'Conditional correlation (diversification loss)', 'normal_exposure_pct': None, 'stress_exposure_pct_worst5pct': None,
         'effect_size_pct_pts': f'{diversification_pairs_disappearing}/{len(corr_df)} strategy pairs show diversification loss (stress corr > normal corr + 0.15)',
         'evidence_strength': strength_of(20 if diversification_pairs_disappearing >= len(corr_df) / 2 else 5, n_w5_days)},
        {'factor': 'Session concentration (ASIAN)', 'normal_exposure_pct': sess_df[(sess_df.bucket=='normal')&(sess_df.session=='ASIAN')]['trade_share_pct'].iloc[0] if len(sess_df[(sess_df.bucket=='normal')&(sess_df.session=='ASIAN')]) else None,
         'stress_exposure_pct_worst5pct': sess_df[(sess_df.bucket=='worst_5pct')&(sess_df.session=='ASIAN')]['trade_share_pct'].iloc[0] if len(sess_df[(sess_df.bucket=='worst_5pct')&(sess_df.session=='ASIAN')]) else None,
         'effect_size_pct_pts': effect_size(sess_df[(sess_df.bucket=='worst_5pct')&(sess_df.session=='ASIAN')]['trade_share_pct'].iloc[0] if len(sess_df[(sess_df.bucket=='worst_5pct')&(sess_df.session=='ASIAN')]) else None,
                                              sess_df[(sess_df.bucket=='normal')&(sess_df.session=='ASIAN')]['trade_share_pct'].iloc[0] if len(sess_df[(sess_df.bucket=='normal')&(sess_df.session=='ASIAN')]) else None),
         'evidence_strength': 'SEE BELOW'},
    ]
    ranking_rows[4]['evidence_strength'] = strength_of(ranking_rows[4]['effect_size_pct_pts'], n_w5_days)
    ranking_df = pd.DataFrame(ranking_rows)
    ranking_df.to_csv(OUT / 'phase41_stress_factor_ranking.csv', index=False)
    print("\n[stress factor ranking]"); print(ranking_df.to_string())

    # --- Part 25: anti-bias check ---
    ledger_excl1 = ledger[ledger['date'] != worst_day['date']]
    ledger_excl5 = ledger[~ledger['date'].isin(worst5['date'])]

    def jpy_effect_on(led):
        rq5 = np.percentile(led['total_R'], 5)
        rq20 = np.percentile(led['total_R'], 20)
        w5_dates = set(led[led['total_R'] <= rq5]['date'])
        normal_dates = set(led[led['total_R'] > rq20]['date'])
        w5_trades = df[df['trade_date'].isin(w5_dates)]
        normal_trades = df[df['trade_date'].isin(normal_dates)]
        w5_pct = w5_trades['is_jpy'].mean() * 100 if len(w5_trades) else None
        normal_pct = normal_trades['is_jpy'].mean() * 100 if len(normal_trades) else None
        return effect_size(w5_pct, normal_pct)

    antibias_rows = [
        {'check': 'Full period (baseline)', 'jpy_effect_pct_pts': jpy_effect, 'n_days': len(ledger)},
        {'check': 'Excluding worst single day', 'jpy_effect_pct_pts': jpy_effect_on(ledger_excl1), 'n_days': len(ledger_excl1)},
        {'check': 'Excluding worst 5 days', 'jpy_effect_pct_pts': jpy_effect_on(ledger_excl5), 'n_days': len(ledger_excl5)},
        {'check': 'Post-demotion live sample (separate, n=19 trades -- see phase41_preregistration.md Part C)',
         'jpy_effect_pct_pts': 'INSUFFICIENT SAMPLE (19 live trades, not a daily-ledger-comparable population)', 'n_days': 'N/A'},
    ]
    antibias_df = pd.DataFrame(antibias_rows)
    antibias_df.to_csv(OUT / 'phase41_antibias.csv', index=False)
    print("\n[anti-bias check (JPY effect robustness)]"); print(antibias_df.to_string())

    # --- Part 26: multiple testing ---
    mt_rows = [
        {'item': 'JPY concentration', 'type': 'PRIMARY PREREGISTERED FACTOR', 'status': 'TESTED'},
        {'item': 'AMR/ARB/Monday mechanism concentration', 'type': 'PRIMARY PREREGISTERED FACTOR', 'status': 'TESTED'},
        {'item': 'Volatility state (reused vol_tercile column)', 'type': 'PRIMARY PREREGISTERED FACTOR', 'status': 'TESTED'},
        {'item': 'Session concentration (Asian/London)', 'type': 'PRIMARY PREREGISTERED FACTOR', 'status': 'TESTED'},
        {'item': 'Directional (long/short) concentration', 'type': 'PRIMARY PREREGISTERED FACTOR', 'status': 'TESTED'},
        {'item': 'Currency-level concentration (7 currencies)', 'type': 'PRIMARY PREREGISTERED FACTOR', 'status': 'TESTED'},
        {'item': 'Entry clustering', 'type': 'EXPLORATORY', 'status': 'TESTED -- EXPLORATORY LABEL APPLIED'},
        {'item': 'Exit clustering (SL/TP mix)', 'type': 'EXPLORATORY', 'status': 'TESTED -- EXPLORATORY LABEL APPLIED'},
        {'item': 'Temporal sequencing (worst 5 days)', 'type': 'EXPLORATORY', 'status': 'TESTED -- EXPLORATORY LABEL APPLIED, PRECEDED not CAUSED'},
        {'item': '7 predeclared factor interactions (Part20)', 'type': 'EXPLORATORY (predeclared, not an unrestricted search)', 'status': 'TESTED'},
        {'item': 'Conditional correlation, 15 strategy pairs', 'type': 'PRIMARY PREREGISTERED FACTOR', 'status': 'TESTED'},
        {'item': 'Marginal + counterfactual attribution, 6 strategies', 'type': 'PRIMARY PREREGISTERED FACTOR (descriptive accounting, not modeling)', 'status': 'TESTED'},
    ]
    mt_df = pd.DataFrame(mt_rows)
    mt_df.to_csv(OUT / 'phase41_multiple_testing.csv', index=False)
    print("\n[multiple testing log]"); print(mt_df.to_string())

    # --- Part 30: future research ideas (NOT implemented) ---
    fri_rows = [
        {'idea': 'Correlation-aware position sizing (reduce total risk when JPY-cross correlation is elevated)', 'basis': f'{diversification_pairs_disappearing}/{len(corr_df)} strategy pairs show diversification loss during stress', 'status': 'FUTURE RESEARCH IDEA -- NOT IMPLEMENTED'},
        {'idea': 'Mechanism-diversification research (a genuinely non-AMR/ARB/calendar-drift return stream) beyond what Phases 38/40 already attempted', 'basis': 'AMR mechanism concentration and stress co-occurrence, see phase41_mechanism_factor.csv', 'status': 'FUTURE RESEARCH IDEA -- NOT IMPLEMENTED'},
        {'idea': 'Direct research into WHY JPY-cross correlation rises during stress (macro risk-sentiment linkage) -- would require the Event/Macro data infrastructure flagged NOT READY in Phase39', 'basis': 'JPY factor ranking result', 'status': 'FUTURE RESEARCH IDEA -- NOT IMPLEMENTED, blocked pending Phase39 infrastructure decision'},
        {'idea': 'A volatility-scaling (not directional) defensive framework, distinct from Phase40s rejected directional volatility-gated momentum design', 'basis': 'Phase40 EXP-138 future research idea, reiterated here given volatility-state stress co-occurrence', 'status': 'FUTURE RESEARCH IDEA -- NOT IMPLEMENTED'},
        {'idea': 'Session-diversification research targeting the New York session specifically (control has zero NY exposure)', 'basis': 'phase41_session_factor.csv shows control concentrated in ASIAN/LONDON only', 'status': 'FUTURE RESEARCH IDEA -- NOT IMPLEMENTED'},
    ]
    fri_df = pd.DataFrame(fri_rows)
    fri_df.to_csv(OUT / 'phase41_future_research_ideas.csv', index=False)
    print("\n[future research ideas -- NOT implemented]"); print(fri_df.to_string())

    summary = {
        'n_trading_days': len(ledger), 'n_trades': len(df),
        'worst_5pct_jpy_effect_pct_pts': jpy_effect, 'worst_5pct_amr_effect_pct_pts': amr_effect,
        'worst_5pct_highvol_effect_pct_pts': hv_effect,
        'diversification_disappearing_pairs': f'{diversification_pairs_disappearing}/{len(corr_df)}',
        'n_2plus_loss_cluster_days': len(sim_loss_days),
        'largest_drawdown_R': round(max_dd, 4),
    }
    with open(OUT / '_phase41_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n" + json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
