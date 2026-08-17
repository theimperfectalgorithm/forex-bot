"""
Phase 38 H2 -- Asian-range breakout continuation (London open -> NY close),
per the frozen definition in reports/phase38_preregistration.md.
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
RNG = np.random.default_rng(20260902)

COST = 0.00018
PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD']
DATA_START = datetime(2019, 1, 1, tzinfo=timezone.utc)
DATA_END = datetime(2026, 8, 14, tzinfo=timezone.utc)
IS_END = datetime(2025, 1, 1, tzinfo=timezone.utc)
TRAIN_START = datetime(2023, 1, 1, tzinfo=timezone.utc)

ASIAN_START_H, ASIAN_END_H = 0, 7    # 00:00-07:00 UTC (frozen)
LONDON_OPEN_H = 7
NY_CLOSE_H = 22


def pull_h1(symbol, start, end):
    if not mt5.initialize():
        raise RuntimeError("MT5 initialize() failed -- STOP")
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start, end)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No H1 data for {symbol} -- STOP")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    assert df['time'].is_monotonic_increasing, f"{symbol}: timestamps not monotonic -- STOP"
    assert df['time'].duplicated().sum() == 0, f"{symbol}: duplicate candles -- STOP"
    assert (df[['open', 'high', 'low', 'close']] > 0).all().all(), f"{symbol}: non-positive OHLC -- STOP"
    df['date'] = df['time'].dt.date
    df['hour'] = df['time'].dt.hour
    return df


def asian_breakout_trades(df, asian_end_h=ASIAN_END_H, cost=COST):
    trades = []
    for date, day in df.groupby('date'):
        asian = day[(day.hour >= ASIAN_START_H) & (day.hour < asian_end_h)]
        if len(asian) < 4:
            continue
        a_high, a_low = asian['high'].max(), asian['low'].min()
        london_bar = day[day.hour == LONDON_OPEN_H]
        if len(london_bar) == 0:
            continue
        lb = london_bar.iloc[0]
        broke_up = lb['high'] > a_high
        broke_down = lb['low'] < a_low
        if broke_up and not broke_down:
            direction = 1; entry_price = a_high; stop = a_low
        elif broke_down and not broke_up:
            direction = -1; entry_price = a_low; stop = a_high
        elif broke_up and broke_down:
            # amended tie-break (frozen 2026-08-18, before any result under this rule): close position decides
            if lb['close'] > lb['open']:
                direction = 1; entry_price = a_high; stop = a_low
            else:
                direction = -1; entry_price = a_low; stop = a_high
        else:
            continue  # inside range through the London-open bar -> no trade

        path = day[(day.hour >= LONDON_OPEN_H) & (day.hour <= NY_CLOSE_H)]
        if len(path) == 0:
            continue
        exit_price = None
        exit_reason = 'time'
        for _, bar in path.iterrows():
            if direction == 1 and bar['low'] <= stop:
                exit_price = stop; exit_reason = 'stop'; break
            if direction == -1 and bar['high'] >= stop:
                exit_price = stop; exit_reason = 'stop'; break
        if exit_price is None:
            exit_price = path.iloc[-1]['close']
        raw_move = direction * (exit_price - entry_price)
        net_move = raw_move - cost
        stop_dist = abs(entry_price - stop)
        r_multiple = net_move / stop_dist if stop_dist > 0 else np.nan
        trades.append({'date': date, 'entry_time': london_bar['time'].iloc[0], 'direction': direction,
                        'entry_price': entry_price, 'stop': stop, 'exit_price': exit_price,
                        'exit_reason': exit_reason, 'r_multiple': r_multiple})
    tdf = pd.DataFrame(trades).dropna(subset=['r_multiple']) if trades else pd.DataFrame()
    return tdf.reset_index(drop=True)


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
    hist_path = REPO / 'data' / 'phase26_all_trades.csv'
    r = ValidationReport(path=str(hist_path))
    validate_column_count_consistency(hist_path, r)
    print(f"[validate] {r.summary()}")

    h1_frames = {sym: pull_h1(sym, DATA_START, DATA_END) for sym in PAIRS}
    for sym, df in h1_frames.items():
        print(f"[data] {sym}: {len(df)} H1 bars")

    all_trades = pd.concat([asian_breakout_trades(df).assign(symbol=sym) for sym, df in h1_frames.items()], ignore_index=True)
    trades_2326 = all_trades[(all_trades.entry_time >= TRAIN_START) & (all_trades.entry_time <= DATA_END)]
    is_ = trades_2326[trades_2326.entry_time < IS_END]
    oos = trades_2326[trades_2326.entry_time >= IS_END]
    m_is, m_oos = edge_metrics(is_), edge_metrics(oos)
    print(f"\n[H2 edge] IS: {json.dumps(m_is)}")
    print(f"[H2 edge] OOS: {json.dumps(m_oos)}")

    edge_df = pd.DataFrame([
        {'split': 'IS (2023-01-01 to 2025-01-01)', **m_is},
        {'split': 'OOS (2025-01-01 to 2026-08-14)', **m_oos,
         'max_dd_R': round(dd_of(oos['r_multiple'].values), 2) if len(oos) else None,
         'max_losing_streak': max_streak(oos['r_multiple'].tolist()) if len(oos) else None},
    ])
    edge_df.to_csv(OUT / 'phase38_h2_session_oos.csv', index=False)
    print(edge_df.to_string())

    gate1_pass = m_oos['pf'] is not None and m_oos['pf'] > 1.0
    print(f"\nGate1 (credible OOS edge, PF>1.0): {'PASS' if gate1_pass else 'FAIL'}")

    if len(oos) >= 4:
        mid = oos['entry_time'].median()
        h1s = oos[oos.entry_time < mid]
        h2s = oos[oos.entry_time >= mid]
        m_h1, m_h2 = edge_metrics(h1s), edge_metrics(h2s)
        sign_consistent = (m_h1['expectancy_R'] or 0) * (m_h2['expectancy_R'] or 0) > 0
        verdict = 'PASS' if sign_consistent else ('WARNING (n<40)' if m_oos['trades'] < 40 else 'FAIL')
    else:
        m_h1 = m_h2 = {'trades': 0, 'expectancy_R': None, 'pf': None}
        sign_consistent = None
        verdict = 'UNKNOWN (insufficient OOS trades to split)'
    oos_cons_df = pd.DataFrame([{
        'oos_h1_trades': m_h1['trades'], 'oos_h1_expectancy_R': m_h1['expectancy_R'], 'oos_h1_pf': m_h1['pf'],
        'oos_h2_trades': m_h2['trades'], 'oos_h2_expectancy_R': m_h2['expectancy_R'], 'oos_h2_pf': m_h2['pf'],
        'sign_consistent': sign_consistent, 'total_oos_trades': m_oos['trades'], 'verdict': verdict,
    }])
    oos_cons_df.to_csv(OUT / 'phase38_h2_oos_consistency.csv', index=False)
    print("\n[H2 OOS consistency]"); print(oos_cons_df.to_string())

    # Parameter robustness: Asian window length 5.6h/7h/8.4h -> asian_end_h in {6 (=6h),7,8}
    param_rows = []
    for label, end_h in [('-20% (5.6h~5h)', 5), ('baseline (7h)', 7), ('+20% (8.4h~8h)', 8)]:
        t_all = pd.concat([asian_breakout_trades(df, asian_end_h=end_h).assign(symbol=sym) for sym, df in h1_frames.items()], ignore_index=True)
        if 'entry_time' in t_all.columns:
            t_all = t_all[(t_all.entry_time >= TRAIN_START) & (t_all.entry_time <= DATA_END)]
        t_oos = t_all[t_all.entry_time >= IS_END] if len(t_all) else t_all
        m = edge_metrics(t_oos)
        param_rows.append({'perturbation': label, **m})
    param_df = pd.DataFrame(param_rows)
    exps = [row['expectancy_R'] for row in param_rows if row['expectancy_R'] is not None]
    sign_reversal = len(set(np.sign(exps))) > 1 if exps else None
    param_df.to_csv(OUT / 'phase38_h2_parameter_robustness.csv', index=False)
    print("\n[H2 parameter robustness]"); print(param_df.to_string())
    print(f"sign_reversal={sign_reversal}")

    cost_rows = []
    for mult, label in [(1.0, 'normal'), (1.5, '1.5x'), (2.0, '2.0x')]:
        t_all = pd.concat([asian_breakout_trades(df, cost=COST * mult).assign(symbol=sym) for sym, df in h1_frames.items()], ignore_index=True)
        if 'entry_time' in t_all.columns:
            t_all = t_all[(t_all.entry_time >= TRAIN_START) & (t_all.entry_time <= DATA_END)]
        t_oos = t_all[t_all.entry_time >= IS_END] if len(t_all) else t_all
        m = edge_metrics(t_oos)
        cost_rows.append({'cost_multiplier': label, **m})
    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(OUT / 'phase38_h2_cost_stress.csv', index=False)
    print("\n[H2 cost stress]"); print(cost_df.to_string())

    regime_rows = []
    is_vol = (is_['entry_price'] - is_['stop']).abs()
    if len(is_) >= 15:
        q1, q2 = is_vol.quantile([1/3, 2/3])
        oos_r = oos.copy()
        oos_r['stopdist'] = (oos_r['entry_price'] - oos_r['stop']).abs()
        oos_r['regime'] = oos_r['stopdist'].apply(lambda a: 'LOW' if a <= q1 else ('NORMAL' if a <= q2 else 'HIGH'))
        for regime_val, sub in oos_r.groupby('regime'):
            m = edge_metrics(sub)
            cls = ('UNKNOWN' if m['trades'] < 10 else
                   'STRONG' if (regime_val == 'HIGH' and m['expectancy_R'] and m['expectancy_R'] > 0) else
                   'WEAK' if (regime_val == 'HIGH' and m['expectancy_R'] and m['expectancy_R'] < 0) else 'NEUTRAL')
            regime_rows.append({'regime': regime_val, **m, 'classification': cls if regime_val == 'HIGH' else ''})
    else:
        regime_rows.append({'regime': 'ALL', 'trades': len(is_), 'classification': 'UNKNOWN (insufficient IS sample for terciles)'})
    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(OUT / 'phase38_h2_regime_analysis.csv', index=False)
    print("\n[H2 regime]"); print(regime_df.to_string())

    periods = {
        'A_2019_2020': (datetime(2019, 1, 1, tzinfo=timezone.utc), datetime(2020, 12, 31, tzinfo=timezone.utc)),
        'B_2021_2022': (datetime(2021, 1, 1, tzinfo=timezone.utc), datetime(2022, 12, 31, tzinfo=timezone.utc)),
        'C_2023_2024': (datetime(2023, 1, 1, tzinfo=timezone.utc), datetime(2024, 12, 31, tzinfo=timezone.utc)),
        'D_2025': (datetime(2025, 1, 1, tzinfo=timezone.utc), datetime(2025, 12, 31, tzinfo=timezone.utc)),
        'E_2026_YTD': (datetime(2026, 1, 1, tzinfo=timezone.utc), DATA_END),
    }
    hist_rows = []
    for pname, (start, end) in periods.items():
        sub = all_trades[(all_trades.entry_time >= start) & (all_trades.entry_time <= end)]
        m = edge_metrics(sub)
        hist_rows.append({'period': pname, **m})
    hist_df = pd.DataFrame(hist_rows)
    print("\n[H2 historical regime]"); print(hist_df.to_string())

    hist = load_hist().sort_values('entry_time').reset_index(drop=True)
    hist['trade_date'] = hist['entry_time'].dt.date
    daily_control = hist.groupby('trade_date')['r_multiple'].sum().rename('control_R')
    oos_start_date = pd.Timestamp('2025-01-01').date()
    daily_control_oos = daily_control[daily_control.index >= oos_start_date]

    oos_c = oos.copy()
    oos_c['trade_date'] = oos_c['entry_time'].dt.date
    daily_cand = oos_c.groupby('trade_date')['r_multiple'].sum().rename('candidate_R')

    cum = daily_control_oos.cumsum()
    dd = cum - cum.cummax()
    dd_thresh = dd.quantile(0.10)
    dd_days = set(dd[dd <= dd_thresh].index)

    merged = pd.concat([daily_control_oos, daily_cand], axis=1).dropna()
    merged['is_dd'] = merged.index.isin(dd_days)
    normal_corr = merged.loc[~merged.is_dd, ['control_R', 'candidate_R']].corr().iloc[0, 1] if (~merged.is_dd).sum() > 5 else None
    n_dd_overlap = int(merged.is_dd.sum())
    dd_corr = merged.loc[merged.is_dd, ['control_R', 'candidate_R']].corr().iloc[0, 1] if n_dd_overlap >= 8 else None

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
        'n_drawdown_days_overlap': n_dd_overlap, 'drawdown_day_corr': round(dd_corr, 3) if dd_corr is not None else None,
        'classification': div_class,
    }])
    dd_corr_df.to_csv(OUT / 'phase38_h2_drawdown_correlation.csv', index=False)
    print("\n[H2 drawdown correlation]"); print(dd_corr_df.to_string())

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
    port_df.to_csv(OUT / 'phase38_h2_portfolio_integration.csv', index=False)
    print("\n[H2 portfolio integration]"); print(port_df.to_string())

    if len(oos) >= 10:
        r_arr = oos['r_multiple'].values
        mc_dds, mc_streaks = [], []
        for _ in range(10000):
            shuf = RNG.permutation(r_arr)
            cum = np.cumsum(shuf)
            dd_ = (cum - np.maximum.accumulate(cum)).min()
            s = ms = 0
            for v in shuf:
                if v < 0: s += 1; ms = max(ms, s)
                else: s = 0
            mc_dds.append(dd_); mc_streaks.append(ms)
        mc_dds = np.array(mc_dds); mc_streaks = np.array(mc_streaks)
        actual_dd = dd_of(r_arr); actual_streak = max_streak(r_arr.tolist())
        mc_df = pd.DataFrame([{
            'n_sims': 10000, 'oos_trades': len(r_arr), 'data_type': 'SIMULATED (trade-order reshuffle)',
            'actual_max_dd_R': round(actual_dd, 2), 'mc_dd_median': round(np.median(mc_dds), 2),
            'mc_dd_p95': round(np.percentile(mc_dds, 95), 2), 'mc_dd_p99': round(np.percentile(mc_dds, 99), 2),
            'actual_dd_percentile_in_mc': round(float((mc_dds < actual_dd).mean() * 100), 1),
            'actual_max_streak': actual_streak, 'mc_streak_p95': round(np.percentile(mc_streaks, 95), 1),
        }])
    else:
        mc_df = pd.DataFrame([{'n_sims': 0, 'oos_trades': len(oos), 'data_type': 'UNKNOWN -- insufficient OOS trades (n<10)'}])
    mc_df.to_csv(OUT / 'phase38_h2_monte_carlo.csv', index=False)
    print("\n[H2 Monte Carlo]"); print(mc_df.to_string())

    summary = {
        'oos_trades': m_oos['trades'], 'oos_pf': m_oos['pf'], 'oos_expectancy_R': m_oos['expectancy_R'],
        'gate1_edge_pass': gate1_pass, 'oos_consistency_verdict': verdict,
        'parameter_sign_reversal': sign_reversal,
        'cost_stress_2x_pf': cost_df.loc[cost_df.cost_multiplier == '2.0x', 'pf'].iloc[0],
        'high_vol_classification': next((row['classification'] for row in regime_rows if row.get('regime') == 'HIGH'), 'UNKNOWN'),
        'drawdown_correlation_classification': div_class,
    }
    with open(OUT / '_phase38_h2_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n" + json.dumps(summary, indent=2, default=str))
    hist_df.to_csv(OUT / '_scratch_h2_hist_regime.csv', index=False)


if __name__ == '__main__':
    main()
