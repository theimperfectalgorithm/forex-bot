"""
Phase 49 -- build the daily portfolio dataset used by all downstream
stress-mechanism analyses. Diagnostic only, no live change.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).parent.parent
MECH_RE = re.compile(r'_(AMR|ARB|MONDAY)$')


def load_control():
    df = pd.read_csv(REPO / 'data' / 'phase26_all_trades.csv')
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df['trade_date'] = df['entry_time'].dt.date
    df['instrument'] = df['strategy'].apply(lambda s: s.replace('_AMR', '').replace('_ARB', '').replace('_MONDAY', ''))
    df['mechanism'] = df['strategy'].apply(lambda s: MECH_RE.search(s).group(1) if MECH_RE.search(s) else 'UNKNOWN')
    df['is_jpy'] = df['instrument'].str.contains('JPY')
    return df


def build_daily_dataset(df):
    dates = sorted(df['trade_date'].unique())
    rows = []
    for d in dates:
        day = df[df['trade_date'] == d]
        d_start = pd.Timestamp(d, tz='UTC')
        d_end = d_start + pd.Timedelta(days=1)
        overlapping = df[(df['entry_time'] < d_end) & (df['exit_time'] >= d_start)]
        vol_vals = day['atr_pctile'].dropna()
        n_trades = len(day)
        rows.append({
            'date': d, 'total_R': round(day['r_multiple'].sum(), 4), 'n_trades': n_trades,
            'n_entries': n_trades, 'n_exits': len(df[df['exit_time'].dt.date == d]),
            'max_concurrent': len(overlapping), 'avg_concurrent': len(overlapping),
            'jpy_share_pct': round(day['is_jpy'].mean() * 100, 1) if n_trades else None,
            'amr_share_pct': round((day['mechanism'] == 'AMR').mean() * 100, 1) if n_trades else None,
            'arb_share_pct': round((day['mechanism'] == 'ARB').mean() * 100, 1) if n_trades else None,
            'monday_share_pct': round((day['mechanism'] == 'MONDAY').mean() * 100, 1) if n_trades else None,
            'long_share_pct': round((day['dir'] == 'BUY').mean() * 100, 1) if n_trades else None,
            'short_share_pct': round((day['dir'] == 'SELL').mean() * 100, 1) if n_trades else None,
            'asian_share_pct': round((day['session'] == 'ASIAN').mean() * 100, 1) if n_trades else None,
            'london_share_pct': round((day['session'] == 'LONDON').mean() * 100, 1) if n_trades else None,
            'ny_share_pct': round((~day['session'].isin(['ASIAN', 'LONDON'])).mean() * 100, 1) if n_trades else 0.0,
            'vol_level': round(vol_vals.mean(), 6) if len(vol_vals) else np.nan,
            'n_strategies_active': day['strategy'].nunique(),
            'active_strategies': '; '.join(sorted(day['strategy'].unique())),
            'n_simultaneous_jpy': int((overlapping['is_jpy']).sum()),
            'n_simultaneous_amr': int((overlapping['mechanism'] == 'AMR').sum()),
            'gbpjpy_amr_active': int((day['strategy'] == 'GBPJPY_AMR').sum()),
        })
    ledger = pd.DataFrame(rows)
    valid = ledger.dropna(subset=['vol_level']).copy()
    valid['vol_pctile'] = valid['vol_level'].rank(pct=True) * 100
    p1, p2 = valid['vol_pctile'].quantile([1/3, 2/3])
    valid['vol_state'] = np.where(valid['vol_pctile'] > p2, 'HIGH', np.where(valid['vol_pctile'] > p1, 'NORMAL', 'LOW'))
    valid = valid.sort_values('date').reset_index(drop=True)
    valid['prev_vol_state'] = valid['vol_state'].shift(1)
    valid['vol_transition'] = valid['prev_vol_state'].fillna('NONE') + '_to_' + valid['vol_state']
    return valid
