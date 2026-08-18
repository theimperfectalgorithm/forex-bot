"""
Phase 40 -- HIGH-volatility-state trend continuation, per the frozen
definition in reports/phase40_preregistration.md. ONE hypothesis, no
tournament, no optimization.
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
RNG = np.random.default_rng(20260919)

COST = 0.00018
PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDCAD']
DATA_START = datetime(2019, 1, 1, tzinfo=timezone.utc)
DATA_END = datetime(2026, 8, 14, tzinfo=timezone.utc)

TRAIN_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
TRAIN_END = datetime(2024, 8, 31, tzinfo=timezone.utc)
VAL_START = datetime(2024, 9, 1, tzinfo=timezone.utc)
VAL_END = datetime(2025, 4, 30, tzinfo=timezone.utc)
OOS_START = datetime(2025, 5, 1, tzinfo=timezone.utc)
OOS_END = DATA_END

NY_ENTRY_START_H, NY_ENTRY_END_H = 13, 20   # entries allowed in this window
NY_CLOSE_H = 21


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
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return df, tr


def train_thresholds(df, tr, atr_window):
    atr = tr.rolling(atr_window).mean()
    norm_atr = atr / df['close']
    train_mask = (df['time'] >= TRAIN_START) & (df['time'] <= TRAIN_END)
    train_vals = norm_atr[train_mask].dropna()
    q1, q2 = train_vals.quantile([1/3, 2/3])
    return norm_atr, atr, q1, q2


def high_vol_trades(df, tr, atr_window=14, cost=COST):
    norm_atr, atr, q1, q2 = train_thresholds(df, tr, atr_window)
    d = df.copy()
    d['norm_atr'] = norm_atr
    d['atr'] = atr
    # one-bar lag: the state used for entry at bar i is bar i-1's own state (no leakage)
    d['prior_norm_atr'] = d['norm_atr'].shift(1)
    d['prior_atr'] = d['atr'].shift(1)
    d['prior_open'] = d['open'].shift(1)
    d['prior_close'] = d['close'].shift(1)
    d['prior_state_high'] = d['prior_norm_atr'] > q2

    entry_mask = (d['hour'] >= NY_ENTRY_START_H) & (d['hour'] <= NY_ENTRY_END_H) & d['prior_state_high']
    trades = []
    for date, day in d[entry_mask].groupby('date'):
        for _, bar in day.iterrows():
            direction = 1 if bar['prior_close'] > bar['prior_open'] else -1
            entry_price = bar['open']
            stop_dist = bar['prior_atr']
            if pd.isna(stop_dist) or stop_dist <= 0:
                continue
            stop = entry_price - direction * stop_dist

            same_day = d[(d['date'] == date) & (d['hour'] >= bar['hour']) & (d['hour'] <= NY_CLOSE_H)]
            exit_price, exit_reason = None, 'time'
            for _, pb in same_day.iterrows():
                if direction == 1 and pb['low'] <= stop:
                    exit_price = stop; exit_reason = 'stop'; break
                if direction == -1 and pb['high'] >= stop:
                    exit_price = stop; exit_reason = 'stop'; break
            if exit_price is None:
                if len(same_day) == 0:
                    continue
                exit_price = same_day.iloc[-1]['close']
            raw_move = direction * (exit_price - entry_price)
            net_move = raw_move - cost
            r_multiple = net_move / stop_dist
            trades.append({'date': date, 'entry_time': bar['time'], 'direction': direction,
                            'entry_price': entry_price, 'stop': stop, 'exit_price': exit_price,
                            'exit_reason': exit_reason, 'r_multiple': r_multiple})
    tdf = pd.DataFrame(trades).dropna(subset=['r_multiple']) if trades else pd.DataFrame()
    return tdf.reset_index(drop=True), q1, q2


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

    raw = {}
    for sym in PAIRS:
        df, tr = pull_h1(sym, DATA_START, DATA_END)
        raw[sym] = (df, tr)
        print(f"[data] {sym}: {len(df)} H1 bars")

    all_trades_list = []
    thresholds = {}
    for sym, (df, tr) in raw.items():
        tdf, q1, q2 = high_vol_trades(df, tr)
        tdf['symbol'] = sym
        all_trades_list.append(tdf)
        thresholds[sym] = (q1, q2)
        print(f"[thresholds] {sym}: TRAIN q1={q1:.6f} q2={q2:.6f}")
    all_trades = pd.concat(all_trades_list, ignore_index=True) if all_trades_list else pd.DataFrame()

    is_ = all_trades[(all_trades.entry_time >= TRAIN_START) & (all_trades.entry_time <= TRAIN_END)]
    val_ = all_trades[(all_trades.entry_time >= VAL_START) & (all_trades.entry_time <= VAL_END)]
    oos = all_trades[(all_trades.entry_time >= OOS_START) & (all_trades.entry_time <= OOS_END)]
    m_is, m_val, m_oos = edge_metrics(is_), edge_metrics(val_), edge_metrics(oos)
    print(f"\n[reproduction] TRAIN: {json.dumps(m_is)}")
    print(f"[reproduction] VALIDATION: {json.dumps(m_val)}")
    print(f"[reproduction] OOS: {json.dumps(m_oos)}")

    repro_df = pd.DataFrame([
        {'split': 'TRAIN (2023-01-01 to 2024-08-31)', **m_is},
        {'split': 'VALIDATION (2024-09-01 to 2025-04-30)', **m_val},
        {'split': 'OOS (2025-05-01 to 2026-08-14)', **m_oos,
         'max_dd_R': round(dd_of(oos['r_multiple'].values), 2) if len(oos) else None,
         'max_losing_streak': max_streak(oos['r_multiple'].tolist()) if len(oos) else None},
    ])
    repro_df.to_csv(OUT / 'phase40_reproduction.csv', index=False)
    print(repro_df.to_string())

    oos_df_out = pd.DataFrame([{'split': 'OOS', **m_oos,
                                 'max_dd_R': round(dd_of(oos['r_multiple'].values), 2) if len(oos) else None,
                                 'max_losing_streak': max_streak(oos['r_multiple'].tolist()) if len(oos) else None}])
    oos_df_out.to_csv(OUT / 'phase40_oos_results.csv', index=False)

    gate1_pass = m_oos['pf'] is not None and m_oos['pf'] > 1.0
    print(f"\nGate1 (credible OOS edge, PF>1.0): {'PASS' if gate1_pass else 'FAIL'}")

    # OOS sub-half
    if len(oos) >= 4:
        mid = oos['entry_time'].median()
        h1 = oos[oos.entry_time < mid]
        h2 = oos[oos.entry_time >= mid]
        m_h1, m_h2 = edge_metrics(h1), edge_metrics(h2)
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
    oos_cons_df.to_csv(OUT / 'phase40_oos_consistency.csv', index=False)
    print("\n[OOS consistency]"); print(oos_cons_df.to_string())

    # Parameter robustness: ATR window 11/14/17
    param_rows = []
    for label, window in [('-20% (ATR11)', 11), ('baseline (ATR14)', 14), ('+20% (ATR17)', 17)]:
        t_list = []
        for sym, (df, tr) in raw.items():
            tdf, _, _ = high_vol_trades(df, tr, atr_window=window)
            t_list.append(tdf)
        t_all = pd.concat(t_list, ignore_index=True) if t_list else pd.DataFrame()
        t_oos = t_all[(t_all.entry_time >= OOS_START) & (t_all.entry_time <= OOS_END)] if len(t_all) else t_all
        m = edge_metrics(t_oos)
        param_rows.append({'perturbation': label, **m})
    param_df = pd.DataFrame(param_rows)
    exps = [row['expectancy_R'] for row in param_rows if row['expectancy_R'] is not None]
    sign_reversal = len(set(np.sign(exps))) > 1 if exps else None
    param_df.to_csv(OUT / 'phase40_parameter_robustness.csv', index=False)
    print("\n[parameter robustness]"); print(param_df.to_string())
    print(f"sign_reversal={sign_reversal}")

    # Cost stress
    cost_rows = []
    for mult, label in [(1.0, 'normal'), (1.5, '1.5x'), (2.0, '2.0x')]:
        t_list = []
        for sym, (df, tr) in raw.items():
            tdf, _, _ = high_vol_trades(df, tr, cost=COST * mult)
            t_list.append(tdf)
        t_all = pd.concat(t_list, ignore_index=True) if t_list else pd.DataFrame()
        t_oos = t_all[(t_all.entry_time >= OOS_START) & (t_all.entry_time <= OOS_END)] if len(t_all) else t_all
        m = edge_metrics(t_oos)
        cost_rows.append({'cost_multiplier': label, **m})
    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(OUT / 'phase40_cost_stress.csv', index=False)
    print("\n[cost stress]"); print(cost_df.to_string())

    # Volatility regime table (this IS the strategy's own gate -- report trade distribution by TRAIN tercile bucket, all OOS trades are HIGH by construction; also report what LOW/NORMAL bars would have looked like as a no-trade diagnostic)
    regime_rows = [{'regime': 'HIGH (the only state this strategy trades)', **m_oos,
                     'max_dd_R': round(dd_of(oos['r_multiple'].values), 2) if len(oos) else None,
                     'max_losing_streak': max_streak(oos['r_multiple'].tolist()) if len(oos) else None,
                     'classification': 'STRONG' if (m_oos['expectancy_R'] and m_oos['expectancy_R'] > 0) else ('WEAK' if m_oos['trades'] else 'UNKNOWN')}]
    regime_rows.append({'regime': 'LOW/NORMAL (not traded by design -- diagnostic only, per Part16)', 'trades': 'N/A BY DESIGN', 'note': 'This candidate only ever trades in HIGH state; LOW/NORMAL performance is not applicable to this hypothesis'})
    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(OUT / 'phase40_volatility_regimes.csv', index=False)
    print("\n[volatility regimes]"); print(regime_df.to_string())

    # Volatility transition diagnostic (Part 17) -- classify each OOS trade's prior-prior state to see high->high (persistent) vs low/normal->high (transition)
    trans_rows = []
    for sym, (df, tr) in raw.items():
        norm_atr, atr, q1, q2 = train_thresholds(df, tr, 14)
        d = df.copy()
        d['norm_atr'] = norm_atr
        d['state'] = np.where(d['norm_atr'] > q2, 'HIGH', np.where(d['norm_atr'] > q1, 'NORMAL', 'LOW'))
        d['prev_state'] = d['state'].shift(1)
        # transitions INTO high state (i.e. bar i-1 became high, having been not-high at i-2)
        oos_sym = oos[oos.symbol == sym] if len(oos) else oos
        for _, row in oos_sym.iterrows():
            match = d[d['time'] == row['entry_time']]
            if len(match) == 0:
                continue
            idx = match.index[0]
            if idx < 2:
                continue
            prior_state = d.loc[idx - 1, 'state']  # the HIGH state that gated this trade
            prior_prior_state = d.loc[idx - 2, 'state']
            transition = 'PERSISTENT_HIGH' if prior_prior_state == 'HIGH' else f'TRANSITION_{prior_prior_state}_TO_HIGH'
            trans_rows.append({'symbol': sym, 'entry_time': row['entry_time'], 'r_multiple': row['r_multiple'], 'transition_type': transition})
    trans_df = pd.DataFrame(trans_rows)
    if len(trans_df):
        trans_summary = trans_df.groupby('transition_type')['r_multiple'].agg(['count', 'mean', 'sum']).reset_index()
        trans_summary.columns = ['transition_type', 'trades', 'expectancy_R', 'total_R']
        trans_summary['note'] = 'DIAGNOSTIC ONLY per Part17 -- not used to create a replacement strategy'
    else:
        trans_summary = pd.DataFrame([{'transition_type': 'N/A', 'trades': 0, 'expectancy_R': None, 'total_R': None, 'note': 'insufficient data'}])
    trans_summary.to_csv(OUT / 'phase40_volatility_transitions.csv', index=False)
    print("\n[volatility transitions -- DIAGNOSTIC ONLY]"); print(trans_summary.to_string())

    # Historical regime (2019-2026)
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
    hist_df.to_csv(OUT / 'phase40_historical_regimes.csv', index=False)
    print("\n[historical regime]"); print(hist_df.to_string())

    # HIGH-vol gate (Part 19) -- since this candidate ONLY trades HIGH-vol bars, its OOS result IS the HIGH-vol result
    if m_oos['trades'] == 0:
        hv_class = 'D. INSUFFICIENT HIGH-VOLATILITY OBSERVATIONS'
    elif m_oos['trades'] < 10:
        hv_class = 'D. INSUFFICIENT HIGH-VOLATILITY OBSERVATIONS (n<10)'
    elif m_oos['expectancy_R'] and m_oos['expectancy_R'] > 0.05:
        hv_class = 'A. PERFORMS POSITIVELY IN HIGH VOLATILITY'
    elif m_oos['expectancy_R'] and m_oos['expectancy_R'] > -0.05:
        hv_class = 'B. NEUTRAL IN HIGH VOLATILITY'
    else:
        hv_class = 'C. MATERIALLY DETERIORATES IN HIGH VOLATILITY'
    hv_df = pd.DataFrame([{'gate': 'HIGH-volatility behaviour (this candidate trades ONLY in HIGH state, so OOS result = HIGH-vol result)',
                            **m_oos, 'classification': hv_class}])
    hv_df.to_csv(OUT / 'phase40_high_volatility.csv', index=False)
    print("\n[HIGH-volatility gate]"); print(hv_df.to_string())

    # Drawdown correlation
    hist = load_hist().sort_values('entry_time').reset_index(drop=True)
    hist['trade_date'] = hist['entry_time'].dt.date
    daily_control = hist.groupby('trade_date')['r_multiple'].sum().rename('control_R')
    oos_start_date = pd.Timestamp('2025-05-01').date()
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
    dd_corr_df.to_csv(OUT / 'phase40_drawdown_correlation.csv', index=False)
    print("\n[drawdown correlation]"); print(dd_corr_df.to_string())

    # Mechanism diversification (qualitative, per Part 21)
    mech_df = pd.DataFrame([
        {'compared_to': 'AMR', 'shared_signal_mechanism': 'None (AMR=Asian-range mean-reversion; Phase40=NY volatility-gated momentum)', 'shared_session': 'No (Asian vs NY)', 'shared_instruments': 'No (AMR is JPY-cross; Phase40 is non-JPY)', 'classification': 'STRONGLY DISTINCT'},
        {'compared_to': 'ARB', 'shared_signal_mechanism': 'Not fully documented in this ledger (predates confirmatory framework)', 'shared_session': 'UNKNOWN', 'shared_instruments': 'Partial overlap possible (ARB is JPY-cross)', 'classification': 'UNKNOWN'},
        {'compared_to': 'GBPUSD Monday', 'shared_signal_mechanism': 'None (calendar drift vs volatility-gated momentum)', 'shared_session': 'No (full-session vs NY-only)', 'shared_instruments': 'Partial (GBPUSD overlaps GBPUSD, but different mechanism)', 'classification': 'MEANINGFULLY DISTINCT'},
        {'compared_to': 'Existing 6-strategy control (aggregate)', 'shared_signal_mechanism': 'None of the 6 live strategies use volatility-state gating as an entry condition', 'shared_session': 'Partial (control has zero confirmed NY-session exposure per Phase31)', 'shared_instruments': 'Partial (non-JPY vs control JPY-heavy)', 'classification': 'MEANINGFULLY DISTINCT structurally -- see drawdown-correlation result above for the decisive test'},
    ])
    mech_df.to_csv(OUT / 'phase40_mechanism_diversification.csv', index=False)
    print("\n[mechanism diversification]"); print(mech_df.to_string())

    # JPY exposure
    jpy_df = pd.DataFrame([{
        'candidate_jpy_exposure': '0% (universe is EURUSD/GBPUSD/AUDUSD/USDCAD, zero JPY legs)',
        'control_jpy_exposure': 'HIGH (4 of 6 live strategies are JPY-linked -- AMR x4 pairs incl. GBPJPY/EURJPY/AUDJPY/CADJPY, ARB incl. GBPJPY/CADJPY)',
        'combined_jpy_exposure_at_1x_weight': 'Unchanged from control (candidate adds zero JPY-linked risk)',
        'risk_weighted_jpy_exposure': 'Structurally reduces relative JPY concentration if deployed, since candidate risk is 100% non-JPY',
        'note': 'Per Part22, non-JPY exposure alone does NOT constitute diversification -- see drawdown_correlation.csv for the decisive test',
    }])
    jpy_df.to_csv(OUT / 'phase40_jpy_exposure.csv', index=False)
    print("\n[JPY exposure]"); print(jpy_df.to_string())

    # Session diversification
    sess_df = pd.DataFrame([{
        'candidate_session': 'New York (13:00-21:00 UTC-server-hour)',
        'control_session_exposure': 'Predominantly Asian (AMR) and London-linked (ARB, GBPUSD Monday full-session) per Phase31 factor map',
        'audusd_monday_long_session': 'Monday full session (D1 open-to-close, not session-scoped)',
        'phase38_h1_session': 'session-independent (weekly rebalance)',
        'phase38_h2_session': 'Asian range -> London open trigger -> NY close (multi-session)',
        'assessment': 'Candidate is the FIRST hypothesis in this ledger to trade EXCLUSIVELY within the New York session with no Asian/London signal dependency -- structurally adds session diversity, per Phase39s finding that NY was the most thinly-tested session',
    }])
    sess_df.to_csv(OUT / 'phase40_session_diversification.csv', index=False)
    print("\n[session diversification]"); print(sess_df.to_string())

    # Portfolio integration
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
    port_df.to_csv(OUT / 'phase40_portfolio_integration.csv', index=False)
    print("\n[portfolio integration]"); print(port_df.to_string())

    # Monte Carlo
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
    mc_df.to_csv(OUT / 'phase40_monte_carlo.csv', index=False)
    print("\n[Monte Carlo]"); print(mc_df.to_string())

    # Sample size
    sample_rows = [
        {'metric': 'OOS trades', 'n': m_oos['trades'],
         'assessment': 'STATISTICALLY INFORMATIVE (n>=30)' if m_oos['trades'] >= 30 else 'OBSERVED ONLY (n<30)'},
        {'metric': 'OOS sub-half (n per half)', 'n': m_h1['trades'],
         'assessment': 'UNDERPOWERED for a confident sign test (n<40 total OOS)' if m_oos['trades'] < 40 else 'ADEQUATE'},
        {'metric': 'Drawdown-correlation overlap days', 'n': n_dd_overlap,
         'assessment': 'STATISTICALLY INFORMATIVE (n>=8)' if n_dd_overlap >= 8 else 'UNKNOWN -- below the 8-day floor'},
        {'metric': 'Historical regime periods with >=10 trades', 'n': int((hist_df['trades'].fillna(0) >= 10).sum()),
         'assessment': f"{int((hist_df['trades'].fillna(0) >= 10).sum())} of {len(hist_df)} periods have >=10 trades"},
    ]
    sample_df = pd.DataFrame(sample_rows)
    sample_df.to_csv(OUT / 'phase40_sample_size.csv', index=False)
    print("\n[sample size]"); print(sample_df.to_string())

    summary = {
        'oos_trades': m_oos['trades'], 'oos_pf': m_oos['pf'], 'oos_expectancy_R': m_oos['expectancy_R'],
        'gate1_edge_pass': gate1_pass, 'oos_consistency_verdict': verdict,
        'parameter_sign_reversal': sign_reversal,
        'cost_stress_2x_pf': cost_df.loc[cost_df.cost_multiplier == '2.0x', 'pf'].iloc[0],
        'high_vol_classification': hv_class,
        'drawdown_correlation_classification': div_class,
    }
    with open(OUT / '_phase40_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n" + json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
