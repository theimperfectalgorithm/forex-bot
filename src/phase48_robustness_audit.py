"""
Phase 48 -- six-strategy parameter & cost robustness audit. Extends the
Phase47 validated reproduction harness with a bar-by-bar trade-outcome
resolver (needed because parameter perturbation changes SL/TP distances,
which Phase47's signal-only reproduction did not resolve to a realized R).
No live strategy code/YAML/execution logic touched. No optimization.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from research_data_validator import ValidationReport, validate_column_count_consistency  # noqa: E402

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

REPO = Path(__file__).parent.parent
OUT = REPO / 'reports'
RNG = np.random.default_rng(20261115)

DATA_START = pd.Timestamp('2023-08-01', tz='UTC')
DATA_END = pd.Timestamp('2026-08-13', tz='UTC')
TRAIN_START = DATA_START
TRAIN_END = pd.Timestamp('2024-08-31', tz='UTC')
OOS_START = pd.Timestamp('2025-05-01', tz='UTC')
OOS_END = DATA_END
BASE_COST_PIPS = 1.0

STRATS = {
    'AUDJPY_AMR': ('AMR', 'AUDJPY', REPO / 'pairs' / 'AUDJPY_asianrev.yaml'),
    'CADJPY_AMR': ('AMR', 'CADJPY', REPO / 'pairs' / 'CADJPY_asianrev.yaml'),
    'EURJPY_AMR': ('AMR', 'EURJPY', REPO / 'pairs' / 'EURJPY_asianrev.yaml'),
    'GBPJPY_AMR': ('AMR', 'GBPJPY', REPO / 'pairs' / 'GBPJPY_asianrev.yaml'),
    'CADJPY_ARB': ('ARB', 'CADJPY', REPO / 'pairs' / 'CADJPY_asianrange.yaml'),
    'GBPUSD_MONDAY': ('MON', 'GBPUSD', REPO / 'pairs' / 'GBPUSD_monday.yaml'),
}
PERTURBABLE = {
    'AMR': ['z_threshold', 'sl_multiplier'],
    'ARB': ['tp_multiplier', 'min_range_pips'],
    'MON': ['sl_atr_mult', 'tp_atr_mult'],
}


STRATEGY_DEFAULTS = {
    # from strategies/asian_hours_reversion.py module-level constants
    'AMR': {'z_threshold': 2.0, 'sl_multiplier': 1.5, 'entry_end_hour': 4},
    # from strategies/asian_range_breakout.py -- MIN_ASIAN_RANGE_PIPS default
    'ARB': {'tp_multiplier': 1.5, 'min_range_pips': 10, 'h4_filter': True},
    # from strategies/monday_drift.py module-level constants
    'MON': {'sl_atr_mult': 1.25, 'tp_atr_mult': 1.0},
}


def load_yaml(p, mech):
    with open(p, encoding='utf-8-sig') as f:
        cfg = yaml.safe_load(f)
    # merge the strategy's own coded defaults for any key the YAML doesn't override
    # (e.g. CADJPY_asianrange.yaml has no min_range_pips -- it uses the source's
    # own MIN_ASIAN_RANGE_PIPS default) -- disclosed, not a silent substitution
    merged = {**STRATEGY_DEFAULTS[mech], **cfg}
    return merged


def pull(symbol, timeframe):
    if not mt5.initialize():
        raise RuntimeError("MT5 init failed")
    rates = mt5.copy_rates_range(symbol, timeframe, DATA_START, DATA_END)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    return df.reset_index(drop=True)


def pip_of(pair):
    return 0.1 if pair.startswith('XAU') else (0.01 if 'JPY' in pair else 0.0001)


# ---------------- AMR: M15 z-score reversion, time exit 07:00 UTC ----------------
def sim_amr(pair, m15, cfg, cost_pips):
    pip = pip_of(pair)
    z_thr, sl_mult, end_hour = cfg['z_threshold'], cfg['sl_multiplier'], cfg['entry_end_hour']
    trades, last_date = [], None
    for i in range(20, len(m15) - 1):
        bt = m15.loc[i, 'time']
        if bt.hour >= end_hour or last_date == bt.date():
            continue
        w = m15.loc[i - 19:i, 'close'].values
        sma, std = float(np.mean(w)), float(np.std(w, ddof=1))
        if std <= 0:
            continue
        close = float(w[-1])
        z = (close - sma) / std
        tp_pips = abs(sma - close) / pip
        if abs(z) < z_thr or tp_pips < 3.0:
            continue
        direction = 1 if z <= -z_thr else -1
        sl_pips = round(tp_pips * sl_mult, 1)
        last_date = bt.date()
        entry = close
        sl = entry - direction * sl_pips * pip
        tp = entry + direction * tp_pips * pip
        exit_price, reason = None, 'time'
        for j in range(i + 1, len(m15)):
            bar = m15.loc[j]
            if bar['time'].date() != bt.date() or bar['time'].hour >= 7:
                break
            if direction == 1:
                if bar['low'] <= sl: exit_price, reason = sl, 'SL'; break
                if bar['high'] >= tp: exit_price, reason = tp, 'TP'; break
            else:
                if bar['high'] >= sl: exit_price, reason = sl, 'SL'; break
                if bar['low'] <= tp: exit_price, reason = tp, 'TP'; break
        if exit_price is None:
            future = m15[(m15.time.dt.date == bt.date()) & (m15.time.dt.hour >= 7)]
            exit_price = future.iloc[0]['open'] if len(future) else m15.loc[min(i + 20, len(m15) - 1), 'close']
        raw_pips = direction * (exit_price - entry) / pip
        net_pips = raw_pips - cost_pips
        r = net_pips / sl_pips if sl_pips > 0 else np.nan
        trades.append({'entry_time': bt, 'trade_date': bt.date(), 'direction': direction, 'r_multiple': r, 'exit_reason': reason})
    return pd.DataFrame(trades)


# ---------------- ARB: H4 trend + Asian range breakout at 07:00-08:30 ----------------
def sim_arb(pair, h4, h1, cfg, cost_pips):
    pip = pip_of(pair)
    use_h4 = cfg.get('h4_filter', True)
    tp_mult = cfg.get('tp_multiplier', 1.5)
    min_range = cfg.get('min_range_pips', 15)

    h4 = h4.copy()
    h4['sma50'] = h4['close'].rolling(50).mean()
    h4['sma200'] = h4['close'].rolling(200).mean()
    h4['trend'] = np.sign(h4['sma50'] - h4['sma200'])
    h4_by_date = h4.set_index(h4['time'].dt.date)

    h1 = h1.copy()
    h1['date'] = h1['time'].dt.date
    h1['hour'] = h1['time'].dt.hour
    trades = []
    for d, day in h1.groupby('date'):
        asian = day[day.hour < 7]
        if len(asian) < 2:
            continue
        high, low = asian['high'].max(), asian['low'].min()
        range_pips = (high - low) / pip
        if range_pips < min_range:
            continue
        trend = 0
        if use_h4:
            prior = h4_by_date[h4_by_date.index <= d]
            if len(prior) == 0 or pd.isna(prior['trend'].iloc[-1]):
                continue
            trend = prior['trend'].iloc[-1]
            if trend == 0:
                continue
        overlap = day[(day.hour >= 7) & (day.hour <= 8)]
        entered = False
        for idx, bar in overlap.iterrows():
            direction = None
            if bar['close'] > high and (not use_h4 or trend > 0):
                direction = 1
            elif bar['close'] < low and (not use_h4 or trend < 0):
                direction = -1
            if direction:
                entry = bar['close']
                sl_pips, tp_pips = range_pips, range_pips * tp_mult
                sl = entry - direction * sl_pips * pip
                tp = entry + direction * tp_pips * pip
                future = h1[h1.index > idx].head(120)
                exit_price, reason = None, 'time'
                for _, fb in future.iterrows():
                    if direction == 1:
                        if fb['low'] <= sl: exit_price, reason = sl, 'SL'; break
                        if fb['high'] >= tp: exit_price, reason = tp, 'TP'; break
                    else:
                        if fb['high'] >= sl: exit_price, reason = sl, 'SL'; break
                        if fb['low'] <= tp: exit_price, reason = tp, 'TP'; break
                if exit_price is None:
                    exit_price = future.iloc[-1]['close'] if len(future) else entry
                raw_pips = direction * (exit_price - entry) / pip
                net_pips = raw_pips - cost_pips
                r = net_pips / sl_pips if sl_pips > 0 else np.nan
                trades.append({'entry_time': bar['time'], 'trade_date': d, 'direction': direction, 'r_multiple': r, 'exit_reason': reason})
                entered = True
                break
        if entered:
            continue
    return pd.DataFrame(trades)


# ---------------- Monday: ATR20d SL/TP, time exit 21:00 UTC Monday ----------------
def sim_monday(pair, h1, cfg, cost_pips):
    pip = 0.0001
    sl_mult, tp_mult = cfg.get('sl_atr_mult', 1.25), cfg.get('tp_atr_mult', 1.0)
    h1i = h1.set_index('time')
    daily = pd.DataFrame({'H': h1i['high'].resample('1D').max(), 'L': h1i['low'].resample('1D').min(), 'C': h1i['close'].resample('1D').last()}).dropna()
    pc = daily['C'].shift(1)
    tr = np.maximum.reduce([(daily['H'] - daily['L']).to_numpy(), (daily['H'] - pc).abs().to_numpy(), (daily['L'] - pc).abs().to_numpy()])
    daily['atr20d'] = pd.Series(tr, index=daily.index).rolling(20).mean()

    h1r = h1.reset_index(drop=True)
    trades = []
    for i, bar in h1r.iterrows():
        bt = bar['time']
        if bt.weekday() != 0 or bt.hour != 0:
            continue
        prior = daily[daily.index < bt.normalize()]
        if len(prior) < 20 or pd.isna(prior['atr20d'].iloc[-1]):
            continue
        atr_pips = prior['atr20d'].iloc[-1] / pip
        if atr_pips <= 0:
            continue
        entry = bar['close']
        sl_pips, tp_pips = atr_pips * sl_mult, atr_pips * tp_mult
        sl, tp = entry - sl_pips * pip, entry + tp_pips * pip
        exit_price, reason = None, 'time'
        for j in range(i + 1, min(i + 25, len(h1r))):
            fb = h1r.loc[j]
            if fb['time'].date() != bt.date():
                break
            if fb['low'] <= sl: exit_price, reason = sl, 'SL'; break
            if fb['high'] >= tp: exit_price, reason = tp, 'TP'; break
            if fb['time'].hour >= 21: exit_price, reason = fb['close'], 'time'; break
        if exit_price is None:
            same_day = h1r[(h1r.time.dt.date == bt.date())]
            exit_price = same_day.iloc[-1]['close'] if len(same_day) else entry
        raw_pips = (exit_price - entry) / pip
        net_pips = raw_pips - cost_pips
        r = net_pips / sl_pips if sl_pips > 0 else np.nan
        trades.append({'entry_time': bt, 'trade_date': bt.date(), 'direction': 1, 'r_multiple': r, 'exit_reason': reason})
    return pd.DataFrame(trades)


def edge_metrics(sub):
    if len(sub) == 0:
        return {'trades': 0, 'win_rate_pct': None, 'pf': None, 'expectancy_R': None, 'total_R': None}
    r = sub['r_multiple'].dropna()
    wins, losses = r[r > 0], r[r < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else np.nan
    return {'trades': len(r), 'win_rate_pct': round((r > 0).mean() * 100, 1),
            'pf': round(pf, 3) if pf == pf else None, 'expectancy_R': round(r.mean(), 4), 'total_R': round(r.sum(), 2)}


def max_streak(r):
    s = ms = 0
    for v in r:
        if v < 0: s += 1; ms = max(ms, s)
        else: s = 0
    return ms


def dd_of(r):
    if len(r) == 0:
        return None
    cum = np.cumsum(r)
    return float((cum - np.maximum.accumulate(cum)).min())


def run_strategy(strat, mech, pair, cfg, cost_pips, data_cache):
    if mech == 'AMR':
        return sim_amr(pair, data_cache[(pair, 'M15')], cfg, cost_pips)
    if mech == 'ARB':
        return sim_arb(pair, data_cache[(pair, 'H4')], data_cache[(pair, 'H1')], cfg, cost_pips)
    return sim_monday(pair, data_cache[(pair, 'H1')], cfg, cost_pips)


def main():
    hist_path = REPO / 'data' / 'phase26_all_trades.csv'
    r = ValidationReport(path=str(hist_path))
    validate_column_count_consistency(hist_path, r)
    print(f"[validate] {r.summary()}")
    hist = pd.read_csv(hist_path)
    hist['entry_time'] = pd.to_datetime(hist['entry_time'])
    hist['trade_date'] = hist['entry_time'].dt.date

    # pull data once per pair/timeframe, reuse across all perturbations
    data_cache = {}
    for strat, (mech, pair, path) in STRATS.items():
        if (pair, 'M15') not in data_cache and mech == 'AMR':
            data_cache[(pair, 'M15')] = pull(pair, mt5.TIMEFRAME_M15)
        if (pair, 'H1') not in data_cache:
            data_cache[(pair, 'H1')] = pull(pair, mt5.TIMEFRAME_H1)
        if (pair, 'H4') not in data_cache and mech == 'ARB':
            data_cache[(pair, 'H4')] = pull(pair, mt5.TIMEFRAME_H4)
    print(f"[data] pulled {len(data_cache)} pair/timeframe series")

    cfgs = {s: load_yaml(p, m) for s, (m, pair, p) in STRATS.items()}

    # --- baseline reproduction + OOS baseline ---
    baseline_trades = {}
    baserepro_rows, oos_rows = [], []
    for strat, (mech, pair, path) in STRATS.items():
        full = run_strategy(strat, mech, pair, cfgs[strat], BASE_COST_PIPS, data_cache)
        baseline_trades[strat] = full
        oos = full[(full.entry_time >= OOS_START) & (full.entry_time <= OOS_END)]
        m = edge_metrics(oos)
        oos_rows.append({'strategy': strat, **m, 'max_dd_R': round(dd_of(oos['r_multiple'].dropna().values), 2) if len(oos) else None,
                          'max_losing_streak': max_streak(oos['r_multiple'].dropna().tolist())})
        n_hist = len(hist[hist.strategy == strat])
        baserepro_rows.append({'strategy': strat, 'phase47_known_trades': n_hist, 'phase48_simulated_total_trades': len(full),
                                'phase48_simulated_oos_trades': len(oos), 'consistent_with_phase47': abs(len(full) - n_hist) / n_hist < 0.10 if n_hist else None})
    pd.DataFrame(baserepro_rows).to_csv(OUT / 'phase48_baseline_reproduction.csv', index=False)
    oos_df = pd.DataFrame(oos_rows)
    oos_df.to_csv(OUT / 'phase48_oos_baseline.csv', index=False)
    print("\n[OOS baseline]"); print(oos_df.to_string())

    # --- parameter inventory (carried forward from Phase47) ---
    pinv_rows = []
    for strat, (mech, pair, path) in STRATS.items():
        for p in PERTURBABLE[mech]:
            pinv_rows.append({'strategy': strat, 'parameter': p, 'category': 'continuous', 'baseline_value': cfgs[strat].get(p)})
    pd.DataFrame(pinv_rows).to_csv(OUT / 'phase48_parameter_inventory.csv', index=False)

    # --- parameter perturbation (one-factor-at-a-time) ---
    pert_rows = []
    for strat, (mech, pair, path) in STRATS.items():
        for p in PERTURBABLE[mech]:
            for label, pct in [('-20%', -0.20), ('baseline', 0.0), ('+20%', 0.20)]:
                cfg2 = dict(cfgs[strat])
                cfg2[p] = round(cfgs[strat][p] * (1 + pct), 6)
                full = run_strategy(strat, mech, pair, cfg2, BASE_COST_PIPS, data_cache)
                oos = full[(full.entry_time >= OOS_START) & (full.entry_time <= OOS_END)]
                m = edge_metrics(oos)
                pert_rows.append({'strategy': strat, 'parameter': p, 'perturbation': label, 'value': cfg2[p], **m})
    pert_df = pd.DataFrame(pert_rows)
    pert_df.to_csv(OUT / 'phase48_parameter_perturbation.csv', index=False)
    print("\n[parameter perturbation]"); print(pert_df.to_string())

    # --- parameter stability + plateau classification ---
    stab_rows, plateau_rows = [], []
    for strat in STRATS:
        for p in pert_df[pert_df.strategy == strat]['parameter'].unique():
            sub = pert_df[(pert_df.strategy == strat) & (pert_df.parameter == p)].set_index('perturbation')
            base_exp = sub.loc['baseline', 'expectancy_R'] if 'baseline' in sub.index else None
            exps = sub['expectancy_R'].dropna()
            sign_rev = len(set(np.sign(exps))) > 1 if len(exps) > 1 else False
            if base_exp is not None and base_exp != 0 and not sign_rev:
                pct_changes = [(abs(sub.loc[l, 'expectancy_R'] - base_exp) / abs(base_exp) * 100) for l in ['-20%', '+20%'] if l in sub.index and pd.notna(sub.loc[l, 'expectancy_R'])]
                max_pct = max(pct_changes) if pct_changes else None
            else:
                max_pct = None
            sensitivity = ('SIGN REVERSAL' if sign_rev else
                            'STABLE' if max_pct is not None and max_pct < 15 else
                            'MODERATELY SENSITIVE' if max_pct is not None and max_pct < 30 else
                            'HIGHLY SENSITIVE' if max_pct is not None else 'INSUFFICIENT DATA')
            stab_rows.append({'strategy': strat, 'parameter': p, 'baseline_expectancy_R': base_exp, 'max_pct_change': round(max_pct, 1) if max_pct is not None else None, 'sensitivity': sensitivity, 'sign_reversal': sign_rev})
        strat_stab = [r['sensitivity'] for r in stab_rows if r['strategy'] == strat]
        if any(s == 'SIGN REVERSAL' for s in strat_stab):
            plateau = 'D. SIGN REVERSAL'
        elif any(s == 'INSUFFICIENT DATA' for s in strat_stab):
            plateau = 'E. INSUFFICIENT DATA'
        elif all(s == 'STABLE' for s in strat_stab):
            plateau = 'A. BROAD PLATEAU'
        elif any(s == 'HIGHLY SENSITIVE' for s in strat_stab):
            plateau = 'C. NARROW PEAK'
        else:
            plateau = 'B. MODERATE STABILITY'
        plateau_rows.append({'strategy': strat, 'plateau_classification': plateau})
    stab_df = pd.DataFrame(stab_rows)
    stab_df.to_csv(OUT / 'phase48_parameter_stability.csv', index=False)
    plateau_df = pd.DataFrame(plateau_rows)
    plateau_df.to_csv(OUT / 'phase48_parameter_plateau.csv', index=False)
    print("\n[parameter stability]"); print(stab_df.to_string())
    print("\n[parameter plateau]"); print(plateau_df.to_string())

    # --- cost stress ---
    cost_rows = []
    for strat, (mech, pair, path) in STRATS.items():
        for mult, label in [(1.0, 'baseline'), (2.0, '2x')]:
            full = run_strategy(strat, mech, pair, cfgs[strat], BASE_COST_PIPS * mult, data_cache)
            oos = full[(full.entry_time >= OOS_START) & (full.entry_time <= OOS_END)]
            m = edge_metrics(oos)
            cost_rows.append({'strategy': strat, 'cost_multiplier': label, **m})
    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(OUT / 'phase48_cost_stress.csv', index=False)
    print("\n[cost stress]"); print(cost_df.to_string())

    # --- OOS sub-half ---
    subhalf_rows = []
    for strat, (mech, pair, path) in STRATS.items():
        full = baseline_trades[strat]
        oos = full[(full.entry_time >= OOS_START) & (full.entry_time <= OOS_END)]
        if len(oos) < 4:
            subhalf_rows.append({'strategy': strat, 'verdict': 'INSUFFICIENT SAMPLE'})
            continue
        mid = oos['entry_time'].median()
        h1s, h2s = oos[oos.entry_time < mid], oos[oos.entry_time >= mid]
        m1, m2 = edge_metrics(h1s), edge_metrics(h2s)
        consistent = (m1['expectancy_R'] or 0) * (m2['expectancy_R'] or 0) > 0
        verdict = 'PASS' if consistent else ('WARNING (n<40)' if len(oos) < 40 else 'FAIL')
        subhalf_rows.append({'strategy': strat, 'h1_trades': m1['trades'], 'h1_expectancy_R': m1['expectancy_R'], 'h1_pf': m1['pf'],
                              'h2_trades': m2['trades'], 'h2_expectancy_R': m2['expectancy_R'], 'h2_pf': m2['pf'], 'verdict': verdict})
    pd.DataFrame(subhalf_rows).to_csv(OUT / 'phase48_oos_subhalf.csv', index=False)

    # --- regime robustness ---
    periods = {'C_2023_2024': (pd.Timestamp('2023-08-01', tz='UTC'), pd.Timestamp('2024-12-31', tz='UTC')),
               'D_2025': (pd.Timestamp('2025-01-01', tz='UTC'), pd.Timestamp('2025-12-31', tz='UTC')),
               'E_2026_YTD': (pd.Timestamp('2026-01-01', tz='UTC'), DATA_END)}
    regime_rows = []
    for strat in STRATS:
        full = baseline_trades[strat]
        for pname, (s, e) in periods.items():
            sub = full[(full.entry_time >= s) & (full.entry_time <= e)]
            regime_rows.append({'strategy': strat, 'period': pname, **edge_metrics(sub)})
    pd.DataFrame(regime_rows).to_csv(OUT / 'phase48_regime_robustness.csv', index=False)

    # --- volatility behavior (join with historical vol_tercile by pair+date) ---
    vol_map = hist.dropna(subset=['vol_tercile'])[['strategy', 'trade_date', 'vol_tercile']].drop_duplicates()
    vol_rows = []
    for strat in STRATS:
        full = baseline_trades[strat]
        oos = full[(full.entry_time >= OOS_START) & (full.entry_time <= OOS_END)].copy()
        vmap_s = vol_map[vol_map.strategy == strat].set_index('trade_date')['vol_tercile']
        oos['vol_state'] = oos['trade_date'].map(vmap_s)
        for state in ['LOW', 'NORMAL', 'HIGH']:
            ssub = oos[oos.vol_state == state]
            vol_rows.append({'strategy': strat, 'vol_state': state, **edge_metrics(ssub)})
    pd.DataFrame(vol_rows).to_csv(OUT / 'phase48_volatility_behavior.csv', index=False)

    # --- drawdown correlation (reuse Phase46 exact methodology) ---
    hist_all = hist.copy()
    daily_control = hist_all.groupby('trade_date')['r_multiple'].sum().rename('control_R')
    oos_start_date = OOS_START.date()
    daily_control_oos = daily_control[daily_control.index >= oos_start_date]
    cum = daily_control_oos.cumsum(); dd = cum - cum.cummax()
    dd_thresh = dd.quantile(0.10); dd_days = set(dd[dd <= dd_thresh].index)
    ddcorr_rows = []
    for strat in STRATS:
        full = baseline_trades[strat]
        oos = full[(full.entry_time >= OOS_START)].copy()
        daily_s = oos.groupby('trade_date')['r_multiple'].sum().rename('strategy_R')
        hist_strat_daily = hist_all[hist_all.strategy == strat].groupby('trade_date')['r_multiple'].sum()
        control_excl = (daily_control_oos - hist_strat_daily.reindex(daily_control_oos.index).fillna(0)).rename('control_R')
        merged = pd.concat([control_excl, daily_s], axis=1).dropna()
        merged['is_dd'] = merged.index.isin(dd_days)
        normal_corr = merged.loc[~merged.is_dd, ['control_R', 'strategy_R']].corr().iloc[0, 1] if (~merged.is_dd).sum() > 5 else None
        n_dd = int(merged.is_dd.sum())
        dd_corr = merged.loc[merged.is_dd, ['control_R', 'strategy_R']].corr().iloc[0, 1] if n_dd >= 8 else None
        cls = ('UNKNOWN (n<8)' if dd_corr is None else 'STRONG DIVERSIFIER' if normal_corr is not None and dd_corr <= normal_corr else
               'NEUTRAL' if normal_corr is not None and dd_corr <= normal_corr + 0.15 else 'CORRELATED')
        ddcorr_rows.append({'strategy': strat, 'normal_day_corr': round(normal_corr, 3) if normal_corr is not None else None,
                             'n_dd_overlap': n_dd, 'dd_day_corr': round(dd_corr, 3) if dd_corr is not None else None, 'classification': cls})
    pd.DataFrame(ddcorr_rows).to_csv(OUT / 'phase48_drawdown_correlation.csv', index=False)
    print("\n[drawdown correlation]"); print(pd.DataFrame(ddcorr_rows).to_string())

    # --- portfolio robustness + leave-one-out ---
    def daily_of(trades_df):
        return trades_df.groupby('trade_date')['r_multiple'].sum()

    full_daily = {s: daily_of(baseline_trades[s][(baseline_trades[s].entry_time >= OOS_START)]) for s in STRATS}
    all_dates = sorted(set().union(*[d.index for d in full_daily.values()]))
    port_baseline = pd.Series(0.0, index=all_dates)
    for s, d in full_daily.items():
        port_baseline = port_baseline.add(d.reindex(all_dates).fillna(0), fill_value=0)

    def port_metrics(series):
        c = series.cumsum(); ddser = c - c.cummax()
        return {'total_R': round(series.sum(), 2), 'max_dd': round(ddser.min(), 2), 'n_days': len(series)}

    port_rows = [{'scenario': 'FULL_PORTFOLIO_BASELINE', **port_metrics(port_baseline)}]
    loo_rows = [{'configuration': 'FULL_SIX_STRATEGY', **port_metrics(port_baseline)}]
    for s in STRATS:
        without = port_baseline - full_daily[s].reindex(all_dates).fillna(0)
        loo_rows.append({'configuration': f'WITHOUT_{s}', **port_metrics(without)})
    pd.DataFrame(loo_rows).to_csv(OUT / 'phase48_leave_one_out.csv', index=False)

    # portfolio with each strategy's +/-20% perturbation substituted one at a time
    for strat, (mech, pair, path) in STRATS.items():
        for p in PERTURBABLE[mech]:
            for label, pct in [('-20%', -0.20), ('+20%', 0.20)]:
                cfg2 = dict(cfgs[strat]); cfg2[p] = round(cfgs[strat][p] * (1 + pct), 6)
                full2 = run_strategy(strat, mech, pair, cfg2, BASE_COST_PIPS, data_cache)
                oos2 = full2[(full2.entry_time >= OOS_START) & (full2.entry_time <= OOS_END)]
                d2 = daily_of(oos2)
                port2 = port_baseline - full_daily[strat].reindex(all_dates).fillna(0) + d2.reindex(all_dates).fillna(0)
                port_rows.append({'scenario': f'{strat}_{p}_{label}', **port_metrics(port2)})
    port_df = pd.DataFrame(port_rows)
    port_df.to_csv(OUT / 'phase48_portfolio_robustness.csv', index=False)
    print("\n[portfolio robustness] (sample)"); print(port_df.head(10).to_string())

    # --- Monte Carlo ---
    mc_rows = []
    for strat in STRATS:
        full = baseline_trades[strat]
        oos = full[(full.entry_time >= OOS_START)]
        r_arr = oos['r_multiple'].dropna().values
        if len(r_arr) < 10:
            mc_rows.append({'strategy': strat, 'n_sims': 0, 'note': 'insufficient trades'}); continue
        mc_dds = []
        for _ in range(10000):
            shuf = RNG.permutation(r_arr)
            cum = np.cumsum(shuf)
            mc_dds.append((cum - np.maximum.accumulate(cum)).min())
        mc_dds = np.array(mc_dds)
        actual_dd = dd_of(r_arr)
        mc_rows.append({'strategy': strat, 'n_sims': 10000, 'n_trades': len(r_arr), 'actual_max_dd_R': round(actual_dd, 2),
                         'mc_dd_median': round(np.median(mc_dds), 2), 'mc_dd_p95': round(np.percentile(mc_dds, 95), 2),
                         'actual_dd_percentile_in_mc': round(float((mc_dds < actual_dd).mean() * 100), 1)})
    pd.DataFrame(mc_rows).to_csv(OUT / 'phase48_monte_carlo.csv', index=False)

    summary = {'strategies': list(STRATS.keys()), 'oos_pf': {s: oos_df[oos_df.strategy == s]['pf'].iloc[0] for s in STRATS}}
    with open(OUT / '_phase48_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n" + json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
