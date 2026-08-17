"""
Phase 37 Track A -- standardized validation of AUDUSD Monday LONG, using the
EXACT definition from src/phase30_nonjpy_calendar_screen.py::drift_cell().
No optimization. No parameter search. No modification to the candidate.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

sys.path.insert(0, str(Path(__file__).parent))
from phase31_factor_regime_map import load_hist  # noqa: E402
from research_data_validator import ValidationReport, validate_column_count_consistency  # noqa: E402

REPO = Path(__file__).parent.parent
OUT = REPO / 'reports'
RNG = np.random.default_rng(20260819)

# --- exact, unmodified constants from src/phase30_nonjpy_calendar_screen.py ---
COST = 0.00018
DATA_START = datetime(2023, 1, 1)
DATA_END = datetime(2026, 8, 14)
IS_END = datetime(2025, 1, 1, tzinfo=timezone.utc)


def pull_daily(symbol, start=DATA_START, end=DATA_END):
    if not mt5.initialize():
        raise RuntimeError("MT5 initialize() failed -- STOP")
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_D1, start, end)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No D1 data for {symbol} -- STOP")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    assert df['time'].is_monotonic_increasing, "timestamps not monotonic -- STOP"
    assert df['time'].duplicated().sum() == 0, "duplicate candles -- STOP"
    assert (df[['open', 'high', 'low', 'close']] > 0).all().all(), "non-positive OHLC -- STOP"
    df['dow'] = df['time'].dt.day_name()
    return df[['time', 'dow', 'open', 'high', 'low', 'close']]


def monday_long_trades(df, atr_window=14, cost=COST):
    """Reproduces drift_cell(df, 'Monday', 'LONG', cost, df) verbatim, but
    returns the per-trade table (not just aggregate stats), needed for the
    downstream OOS-sub-half / regime / correlation / MC analyses."""
    mon = df[df['dow'] == 'Monday'].copy()
    if len(mon) == 0:
        return pd.DataFrame()
    raw_move = mon['close'] - mon['open']
    net_move = raw_move - cost
    atr = df['high'].sub(df['low']).rolling(atr_window).mean()
    mon_atr = atr.reindex(mon.index)
    r_multiple = (net_move / mon_atr).replace([np.inf, -np.inf], np.nan)
    out = pd.DataFrame({
        'entry_time': mon['time'], 'entry_price': mon['open'], 'exit_price': mon['close'],
        'atr_at_entry': mon_atr, 'r_multiple': r_multiple,
    }).dropna(subset=['r_multiple'])
    return out.reset_index(drop=True)


def edge_metrics(trades):
    if len(trades) == 0:
        return {'trades': 0, 'win_rate_pct': None, 'pf': None, 'expectancy_R': None, 'total_R': None}
    r = trades['r_multiple']
    wins, losses = r[r > 0], r[r < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else np.nan
    return {'trades': len(trades), 'win_rate_pct': round((r > 0).mean() * 100, 1),
            'pf': round(pf, 3) if pf == pf else None, 'expectancy_R': round(r.mean(), 4),
            'total_R': round(r.sum(), 2)}


def max_streak(r_series):
    s = ms = 0
    for v in r_series:
        if v < 0:
            s += 1; ms = max(ms, s)
        else:
            s = 0
    return ms


def dd_of(r_series):
    cum = np.cumsum(r_series)
    return float((cum - np.maximum.accumulate(cum)).min())


def main():
    # Part 2: data integrity check on the historical control (reused across the phase)
    hist_path = REPO / 'data' / 'phase26_all_trades.csv'
    r = ValidationReport(path=str(hist_path))
    validate_column_count_consistency(hist_path, r)
    print(f"[validate] {r.summary()}")

    df_full = pull_daily('AUDUSD', datetime(2019, 1, 1), DATA_END)  # extended for Part 10 historical regimes
    data_start_utc = DATA_START.replace(tzinfo=timezone.utc)
    data_end_utc = DATA_END.replace(tzinfo=timezone.utc)
    df_2326 = df_full[(df_full['time'] >= data_start_utc) & (df_full['time'] <= data_end_utc)].reset_index(drop=True)
    print(f"[data] AUDUSD D1 bars 2019-2026: {len(df_full)}, 2023-2026 subset: {len(df_2326)}")

    trades_all = monday_long_trades(df_2326)
    is_trades = trades_all[trades_all.entry_time < IS_END]
    oos_trades = trades_all[trades_all.entry_time >= IS_END]

    # --- A3/A4: Reproduction + OOS sub-half ---
    m_is, m_oos = edge_metrics(is_trades), edge_metrics(oos_trades)
    print(f"\n[reproduction] IS: {json.dumps(m_is)}")
    print(f"[reproduction] OOS: {json.dumps(m_oos)}")
    original = {'oos_trades': 84, 'oos_pf': 3.070, 'is_t_stat': 1.65, 'oos_t_stat': 4.15}
    trade_count_match = (m_oos['trades'] == original['oos_trades'])
    pf_tolerance_ok = m_oos['pf'] is not None and abs(m_oos['pf'] - original['oos_pf']) / original['oos_pf'] <= 0.05
    print(f"[reproduction] trade_count_match={trade_count_match} pf_within_5pct={pf_tolerance_ok}")

    repro_df = pd.DataFrame([{
        'metric': 'oos_trades', 'reproduced': m_oos['trades'], 'original': original['oos_trades'], 'match': trade_count_match},
        {'metric': 'oos_pf', 'reproduced': m_oos['pf'], 'original': original['oos_pf'], 'match': pf_tolerance_ok},
        {'metric': 'oos_expectancy_R', 'reproduced': m_oos['expectancy_R'], 'original': 'NOT DIRECTLY COMPARABLE (original reported mean_R at unrounded precision + t-stat, not expectancy_R by this exact label)', 'match': 'N/A'},
        {'metric': 'oos_win_rate_pct', 'reproduced': m_oos['win_rate_pct'], 'original': 'NOT AVAILABLE in original registry (only PF/t-stat/mean_R recorded)', 'match': 'N/A'},
        {'metric': 'oos_total_R', 'reproduced': m_oos['total_R'], 'original': 'NOT AVAILABLE in original registry', 'match': 'N/A'},
        {'metric': 'oos_max_drawdown_R', 'reproduced': round(dd_of(oos_trades['r_multiple'].values), 2) if len(oos_trades) else None, 'original': 'NOT AVAILABLE in original registry', 'match': 'N/A'},
        {'metric': 'oos_max_losing_streak', 'reproduced': max_streak(oos_trades['r_multiple'].tolist()), 'original': 'NOT AVAILABLE in original registry', 'match': 'N/A'},
    ])
    repro_df.to_csv(OUT / 'phase37_audusd_reproduction.csv', index=False)
    print(repro_df.to_string())

    if not (trade_count_match and pf_tolerance_ok):
        print("\n*** REPRODUCTION TOLERANCE NOT MET -- per preregistration, investigate before proceeding ***")

    # OOS sub-half
    mid = oos_trades['entry_time'].median()
    h1 = oos_trades[oos_trades.entry_time < mid]
    h2 = oos_trades[oos_trades.entry_time >= mid]
    m_h1, m_h2 = edge_metrics(h1), edge_metrics(h2)
    sign_consistent = (m_h1['expectancy_R'] or 0) * (m_h2['expectancy_R'] or 0) > 0
    n_total_oos = m_oos['trades']
    consistency_verdict = ('PASS' if sign_consistent else
                            ('WARNING (n<40, per frozen preregistration)' if n_total_oos < 40 else 'FAIL'))
    oos_cons_df = pd.DataFrame([{
        'oos_h1_trades': m_h1['trades'], 'oos_h1_expectancy_R': m_h1['expectancy_R'], 'oos_h1_pf': m_h1['pf'],
        'oos_h2_trades': m_h2['trades'], 'oos_h2_expectancy_R': m_h2['expectancy_R'], 'oos_h2_pf': m_h2['pf'],
        'sign_consistent': sign_consistent, 'total_oos_trades': n_total_oos, 'verdict': consistency_verdict,
    }])
    oos_cons_df.to_csv(OUT / 'phase37_audusd_oos_consistency.csv', index=False)
    print("\n[oos consistency]"); print(oos_cons_df.to_string())

    # --- A5: parameter robustness (ATR window only, with the disclosed limitation) ---
    param_rows = []
    for label, window in [('-20% (ATR11)', 11), ('baseline (ATR14)', 14), ('+20% (ATR17)', 17)]:
        t_all = monday_long_trades(df_2326, atr_window=window)
        t_oos = t_all[t_all.entry_time >= IS_END]
        m = edge_metrics(t_oos)
        param_rows.append({'perturbation': label, **m})
    param_df = pd.DataFrame(param_rows)
    base_exp = param_df.loc[param_df.perturbation.str.contains('baseline'), 'expectancy_R'].iloc[0]
    signs = np.sign(param_df['expectancy_R'].astype(float))
    sign_reversal = signs.nunique() > 1
    param_df.to_csv(OUT / 'phase37_audusd_parameter_robustness.csv', index=False)
    print("\n[parameter robustness]"); print(param_df.to_string())
    print(f"sign_reversal={sign_reversal} (NOTE: this parameter only rescales R, does not change trade selection -- see preregistration A5 limitation)")

    # --- A6: cost stress ---
    cost_rows = []
    for mult, label in [(1.0, 'normal'), (1.5, '1.5x'), (2.0, '2.0x')]:
        t_all = monday_long_trades(df_2326, cost=COST * mult)
        t_oos = t_all[t_all.entry_time >= IS_END]
        m = edge_metrics(t_oos)
        cost_rows.append({'cost_multiplier': label, **m})
    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(OUT / 'phase37_audusd_cost_stress.csv', index=False)
    print("\n[cost stress]"); print(cost_df.to_string())

    # --- A7: regime (ATR terciles fixed on IS only) ---
    regime_rows = []
    is_atr = is_trades['atr_at_entry'].dropna()
    if len(is_atr) >= 15:
        q1, q2 = is_atr.quantile([1/3, 2/3])
        def regime_of(a):
            if pd.isna(a):
                return None
            return 'LOW' if a <= q1 else ('NORMAL' if a <= q2 else 'HIGH')
        oos_r = oos_trades.copy()
        oos_r['regime'] = oos_r['atr_at_entry'].apply(regime_of)
        for regime_val, sub in oos_r.groupby('regime'):
            m = edge_metrics(sub)
            cls = ('UNKNOWN' if m['trades'] < 10 else
                   'STRONG' if (regime_val == 'HIGH' and m['expectancy_R'] and m['expectancy_R'] > 0) else
                   'WEAK' if (regime_val == 'HIGH' and m['expectancy_R'] and m['expectancy_R'] < 0) else 'NEUTRAL')
            regime_rows.append({'regime': regime_val, **m, 'max_dd_R': round(dd_of(sub['r_multiple'].values), 2),
                                 'max_losing_streak': max_streak(sub['r_multiple'].tolist()),
                                 'classification': cls if regime_val == 'HIGH' else ''})
    else:
        regime_rows.append({'regime': 'ALL', 'note': 'insufficient IS trades for terciles', 'classification': 'UNKNOWN'})
    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(OUT / 'phase37_audusd_regime_analysis.csv', index=False)
    print("\n[regime]"); print(regime_df.to_string())

    # --- A8: historical regime analysis (2019-2026, real data, unmodified mechanics) ---
    periods = {
        'A_2019_2020': (datetime(2019, 1, 1, tzinfo=timezone.utc), datetime(2020, 12, 31, tzinfo=timezone.utc)),
        'B_2021_2022': (datetime(2021, 1, 1, tzinfo=timezone.utc), datetime(2022, 12, 31, tzinfo=timezone.utc)),
        'C_2023_2024': (datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2024, 12, 31, tzinfo=timezone.utc)),
        'D_2025': (datetime(2025, 1, 1, tzinfo=timezone.utc), datetime(2025, 12, 31, tzinfo=timezone.utc)),
        'E_2026_YTD': (datetime(2026, 1, 1, tzinfo=timezone.utc), DATA_END.replace(tzinfo=timezone.utc)),
    }
    trades_full = monday_long_trades(df_full)
    hist_regime_rows = []
    for pname, (start, end) in periods.items():
        sub = trades_full[(trades_full.entry_time >= start) & (trades_full.entry_time <= end)]
        m = edge_metrics(sub)
        hist_regime_rows.append({'period': pname, **m,
                                  'max_dd_R': round(dd_of(sub['r_multiple'].values), 2) if len(sub) else None,
                                  'candidate_level_evidence': 'YES (this is the actual candidate mechanics, not a price proxy)'})
    hist_regime_df = pd.DataFrame(hist_regime_rows)
    print("\n[historical regime]"); print(hist_regime_df.to_string())

    # --- A9: drawdown correlation (OOS-window-matched control) ---
    hist = load_hist().sort_values('entry_time').reset_index(drop=True)
    hist['trade_date'] = hist['entry_time'].dt.date
    daily_control = hist.groupby('trade_date')['r_multiple'].sum().rename('control_R')
    oos_start_date = pd.Timestamp('2025-01-01').date()
    daily_control_oos = daily_control[daily_control.index >= oos_start_date]

    oos_c = oos_trades.copy()
    oos_c['trade_date'] = oos_c['entry_time'].dt.date
    daily_cand = oos_c.groupby('trade_date')['r_multiple'].sum().rename('candidate_R')

    cum = daily_control_oos.cumsum()
    dd = cum - cum.cummax()
    dd_thresh = dd.quantile(0.10)
    dd_days = set(dd[dd <= dd_thresh].index)

    merged = pd.concat([daily_control_oos, daily_cand], axis=1).dropna()
    merged['is_dd'] = merged.index.isin(dd_days)
    normal_corr = merged.loc[~merged.is_dd, ['control_R', 'candidate_R']].corr().iloc[0, 1] if (~merged.is_dd).sum() > 5 else None
    dd_corr = merged.loc[merged.is_dd, ['control_R', 'candidate_R']].corr().iloc[0, 1] if merged.is_dd.sum() >= 8 else None
    n_dd_overlap = int(merged.is_dd.sum())

    if dd_corr is None:
        div_class = 'UNKNOWN (insufficient drawdown-day overlap, <8 days)'
    elif normal_corr is not None and dd_corr <= normal_corr:
        div_class = 'STRONG DIVERSIFIER'
    elif normal_corr is not None and dd_corr <= normal_corr + 0.15:
        div_class = 'NEUTRAL'
    else:
        div_class = 'CORRELATED'

    dd_corr_df = pd.DataFrame([{
        'overlapping_days': len(merged), 'normal_day_corr': round(normal_corr, 3) if normal_corr is not None else None,
        'n_drawdown_days_overlap': n_dd_overlap,
        'drawdown_day_corr': round(dd_corr, 3) if dd_corr is not None else None,
        'classification': div_class,
    }])
    dd_corr_df.to_csv(OUT / 'phase37_audusd_drawdown_correlation.csv', index=False)
    print("\n[drawdown correlation]"); print(dd_corr_df.to_string())

    # --- A10: portfolio integration ---
    control_std = daily_control_oos.std()
    port_rows = []
    for weight in [0.5, 1.0]:
        scaled_cand = daily_cand * weight
        combined = pd.concat([daily_control_oos, scaled_cand], axis=1).fillna(0).sum(axis=1)
        def metrics(series):
            c = series.cumsum()
            ddser = c - c.cummax()
            s = ms = 0
            for v in series:
                if v < 0: s += 1; ms = max(ms, s)
                else: s = 0
            return {'total_R': round(series.sum(), 2), 'max_dd': round(ddser.min(), 2), 'max_streak_days': ms}
        mc, mx = metrics(daily_control_oos), metrics(combined)
        port_rows.append({'weight': weight, 'control_total_R': mc['total_R'], 'combined_total_R': mx['total_R'],
                           'control_max_dd': mc['max_dd'], 'combined_max_dd': mx['max_dd'],
                           'control_max_streak_days': mc['max_streak_days'], 'combined_max_streak_days': mx['max_streak_days']})
    port_df = pd.DataFrame(port_rows)
    port_df.to_csv(OUT / 'phase37_audusd_portfolio_integration.csv', index=False)
    print("\n[portfolio integration]"); print(port_df.to_string())

    # --- A11: Monte Carlo ---
    r_arr = oos_trades['r_multiple'].values
    mc_totals, mc_dds, mc_streaks = [], [], []
    for _ in range(10000):
        shuf = RNG.permutation(r_arr)
        cum = np.cumsum(shuf)
        dd_ = (cum - np.maximum.accumulate(cum)).min()
        s = ms = 0
        for v in shuf:
            if v < 0: s += 1; ms = max(ms, s)
            else: s = 0
        mc_totals.append(shuf.sum()); mc_dds.append(dd_); mc_streaks.append(ms)
    mc_dds = np.array(mc_dds); mc_streaks = np.array(mc_streaks)
    actual_dd = dd_of(r_arr)
    actual_streak = max_streak(r_arr.tolist())
    mc_df = pd.DataFrame([{
        'n_sims': 10000, 'oos_trades': len(r_arr), 'data_type': 'SIMULATED (trade-order reshuffle of the candidate\'s own 84 OOS trades)',
        'actual_max_dd_R': round(actual_dd, 2), 'mc_dd_median': round(np.median(mc_dds), 2),
        'mc_dd_p95': round(np.percentile(mc_dds, 95), 2), 'mc_dd_p99': round(np.percentile(mc_dds, 99), 2),
        'actual_dd_percentile_in_mc': round(float((mc_dds < actual_dd).mean() * 100), 1),
        'actual_max_streak': actual_streak, 'mc_streak_p95': round(np.percentile(mc_streaks, 95), 1),
        'mc_streak_p99': round(np.percentile(mc_streaks, 99), 1),
    }])
    mc_df.to_csv(OUT / 'phase37_audusd_monte_carlo.csv', index=False)
    print("\n[Monte Carlo]"); print(mc_df.to_string())

    # --- A14: sample size ---
    sample_rows = [
        {'metric': 'OOS trades (aggregate PF/expectancy point estimate)', 'n': m_oos['trades'],
         'assessment': 'STATISTICALLY INFORMATIVE for a point estimate (n>=30)' if m_oos['trades'] >= 30 else 'OBSERVED ONLY'},
        {'metric': 'OOS sub-half (n per half)', 'n': m_h1['trades'],
         'assessment': 'UNDERPOWERED for a confident sub-half sign test (n<40 total OOS)' if n_total_oos < 40 else 'ADEQUATE'},
        {'metric': 'HIGH-vol regime bucket', 'n': next((row['trades'] for row in regime_rows if row.get('regime') == 'HIGH'), 0),
         'assessment': 'see phase37_audusd_regime_analysis.csv for the 10-trade floor result'},
        {'metric': 'Drawdown-correlation overlap days', 'n': n_dd_overlap,
         'assessment': 'STATISTICALLY INFORMATIVE (n>=8)' if n_dd_overlap >= 8 else 'UNKNOWN -- below the 8-day floor'},
        {'metric': 'Historical regime periods with adequate sample (>=10 trades)',
         'n': int((hist_regime_df['trades'].fillna(0) >= 10).sum()),
         'assessment': f"{int((hist_regime_df['trades'].fillna(0) >= 10).sum())} of {len(hist_regime_df)} characterized periods have >=10 trades"},
    ]
    sample_df = pd.DataFrame(sample_rows)
    sample_df.to_csv(OUT / 'phase37_audusd_sample_size.csv', index=False)
    print("\n[sample size]"); print(sample_df.to_string())

    hist_regime_df.to_csv(OUT / '_scratch_hist_regime.csv', index=False)

    summary = {
        'reproduction_match': bool(trade_count_match and pf_tolerance_ok),
        'oos_consistency_verdict': consistency_verdict,
        'parameter_sign_reversal': bool(sign_reversal),
        'cost_stress_1_5x_pf': cost_df.loc[cost_df.cost_multiplier == '1.5x', 'pf'].iloc[0],
        'high_vol_classification': next((row['classification'] for row in regime_rows if row.get('regime') == 'HIGH'), 'UNKNOWN'),
        'drawdown_correlation_classification': div_class,
        'n_oos_trades': m_oos['trades'],
    }
    with open(OUT / '_phase37_track_a_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n" + json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
