"""
Phase 33 continuation: parameter sensitivity (+-20%, frozen in the
preregistration), portfolio correlation/drawdown-correlation gate, and
portfolio integration test using the ACTUAL OOS trade streams (real
backtested trades, not synthetic archetypes -- unlike Phase 32).
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

sys.path.insert(0, str(Path(__file__).parent))
from phase31_factor_regime_map import load_hist, RISK_PCT, CURRENT_SIX  # noqa: E402
from phase33_strategy_discovery import (  # noqa: E402
    pull, backtest_xauusd, backtest_usdcad, split_periods, edge_metrics, max_streak, VAL_END,
)

REPO = Path(__file__).parent.parent
OUT = REPO / 'reports'
RNG = np.random.default_rng(20260816)


def backtest_xauusd_perturbed(df, pctile_mult):
    """Re-runs Candidate 1 with the volatility-contraction percentile
    threshold perturbed by pctile_mult (e.g. 0.8 or 1.2 for +-20%)."""
    import phase33_strategy_discovery as p33
    orig = p33.backtest_xauusd.__code__
    # simplest safe approach: monkey-patch the quantile call via a wrapper
    df = df.copy().reset_index(drop=True)
    df['hour'] = df['time'].dt.hour
    df['tr'] = np.maximum(df['high'] - df['low'],
                           np.maximum((df['high'] - df['close'].shift(1)).abs(),
                                      (df['low'] - df['close'].shift(1)).abs()))
    df['atr14'] = df['tr'].rolling(14).mean()
    cost = 0.35
    prelondon = df[df['hour'].isin([3, 4, 5, 6])].copy()
    prelondon['date'] = prelondon['time'].dt.date
    range_by_day = prelondon.groupby('date').agg(range_high=('high', 'max'), range_low=('low', 'min'))
    range_by_day['range_width'] = range_by_day['range_high'] - range_by_day['range_low']
    range_by_day = range_by_day.sort_index()
    q = (1 / 3) * pctile_mult
    q = min(max(q, 0.01), 0.99)
    range_by_day['thresh'] = range_by_day['range_width'].rolling(30, min_periods=15).quantile(q).shift(1)

    london_bars = df[df['hour'] == 7].copy()
    london_bars['date'] = london_bars['time'].dt.date
    trades = []
    for _, lb in london_bars.iterrows():
        d = lb['date']
        if d not in range_by_day.index:
            continue
        row = range_by_day.loc[d]
        if pd.isna(row['thresh']) or pd.isna(row['range_width']) or row['range_width'] >= row['thresh']:
            continue
        window = df[(df['time'].dt.date == d) & (df['hour'] >= 7) & (df['hour'] <= 20)]
        entry_idx = direction = None
        for idx, bar in window.iterrows():
            if bar['close'] > row['range_high']:
                entry_idx, direction = idx, 'BUY'
                break
            if bar['close'] < row['range_low']:
                entry_idx, direction = idx, 'SELL'
                break
        if entry_idx is None:
            continue
        entry_price = df.loc[entry_idx, 'close']
        sl = row['range_low'] if direction == 'BUY' else row['range_high']
        sl_dist = abs(entry_price - sl)
        if sl_dist <= 0:
            continue
        tp = entry_price + 2.0 * sl_dist if direction == 'BUY' else entry_price - 2.0 * sl_dist
        future = df.loc[entry_idx + 1: entry_idx + 1 + 200]
        exit_price = exit_time = None
        for _, fb in future.iterrows():
            if direction == 'BUY':
                if fb['low'] <= sl:
                    exit_price, exit_time = sl, fb['time']; break
                if fb['high'] >= tp:
                    exit_price, exit_time = tp, fb['time']; break
            else:
                if fb['high'] >= sl:
                    exit_price, exit_time = sl, fb['time']; break
                if fb['low'] <= tp:
                    exit_price, exit_time = tp, fb['time']; break
        if exit_price is None:
            continue
        raw_pnl = (exit_price - entry_price) if direction == 'BUY' else (entry_price - exit_price)
        r_multiple = (raw_pnl - cost) / sl_dist
        trades.append({'entry_time': df.loc[entry_idx, 'time'], 'r_multiple': r_multiple})
    return pd.DataFrame(trades)


def backtest_usdcad_perturbed(df, er_mult):
    df = df.copy().reset_index(drop=True)
    df['tr'] = np.maximum(df['high'] - df['low'],
                           np.maximum((df['high'] - df['close'].shift(1)).abs(),
                                      (df['low'] - df['close'].shift(1)).abs()))
    df['atr20'] = df['tr'].rolling(20).mean()
    df['hi20'] = df['high'].rolling(20).max().shift(1)
    df['lo20'] = df['low'].rolling(20).min().shift(1)
    net_disp = (df['close'] - df['close'].shift(20)).abs()
    sum_abs = df['close'].diff().abs().rolling(20).sum()
    df['er'] = net_disp / sum_abs.replace(0, np.nan)
    threshold = min(max(0.35 * er_mult, 0.05), 0.95)
    cost = 0.00020
    trades = []
    in_pos_until = None
    for i in range(20, len(df) - 1):
        if in_pos_until is not None and df.loc[i, 'time'] <= in_pos_until:
            continue
        row = df.loc[i]
        if pd.isna(row['hi20']) or pd.isna(row['er']) or pd.isna(row['atr20']) or row['atr20'] <= 0:
            continue
        direction = None
        if row['close'] > row['hi20'] and row['er'] > threshold:
            direction = 'BUY'
        elif row['close'] < row['lo20'] and row['er'] > threshold:
            direction = 'SELL'
        if direction is None:
            continue
        entry_price = df.loc[i + 1, 'open']
        sl_dist = 1.5 * row['atr20']
        tp_dist = 3.0 * row['atr20']
        sl = entry_price - sl_dist if direction == 'BUY' else entry_price + sl_dist
        tp = entry_price + tp_dist if direction == 'BUY' else entry_price - tp_dist
        future = df.loc[i + 2: i + 2 + 60]
        exit_price = exit_time = None
        for _, fb in future.iterrows():
            if direction == 'BUY':
                if fb['low'] <= sl:
                    exit_price, exit_time = sl, fb['time']; break
                if fb['high'] >= tp:
                    exit_price, exit_time = tp, fb['time']; break
            else:
                if fb['high'] >= sl:
                    exit_price, exit_time = sl, fb['time']; break
                if fb['low'] <= tp:
                    exit_price, exit_time = tp, fb['time']; break
        if exit_price is None:
            continue
        raw_pnl = (exit_price - entry_price) if direction == 'BUY' else (entry_price - exit_price)
        r_multiple = (raw_pnl - cost) / sl_dist
        trades.append({'entry_time': df.loc[i + 1, 'time'], 'r_multiple': r_multiple})
        in_pos_until = exit_time
    return pd.DataFrame(trades)


def oos_metrics(trades_df):
    if len(trades_df) == 0:
        return {'trades': 0, 'pf': None, 'expectancy_R': None}
    trades_df = trades_df.copy()
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'], utc=True)
    oos = trades_df[trades_df.entry_time >= VAL_END]
    return edge_metrics(oos)


def main():
    print("[sensitivity] pulling data")
    xau_df = pull('XAUUSD', mt5.TIMEFRAME_H1)
    cad_df = pull('USDCAD', mt5.TIMEFRAME_H4)

    sens_rows = []
    for mult, label in [(0.8, '-20%'), (1.0, 'baseline'), (1.2, '+20%')]:
        t = backtest_xauusd_perturbed(xau_df, mult)
        m = oos_metrics(t)
        sens_rows.append({'candidate_id': 'EXP-125_XAUUSD_LONDON_VOL_EXPANSION', 'perturbation': label,
                           'oos_trades': m['trades'], 'oos_pf': m['pf'], 'oos_expectancy_R': m['expectancy_R']})
    for mult, label in [(0.8, '-20%'), (1.0, 'baseline'), (1.2, '+20%')]:
        t = backtest_usdcad_perturbed(cad_df, mult)
        m = oos_metrics(t)
        sens_rows.append({'candidate_id': 'EXP-126_USDCAD_MOMENTUM_CONTINUATION', 'perturbation': label,
                           'oos_trades': m['trades'], 'oos_pf': m['pf'], 'oos_expectancy_R': m['expectancy_R']})
    sens_df = pd.DataFrame(sens_rows)
    print("\n=== parameter sensitivity (+-20%) ===")
    print(sens_df.to_string())

    # ---- portfolio correlation / drawdown correlation ----
    validate_hist = load_hist()
    validate_hist = validate_hist.sort_values('entry_time').reset_index(drop=True)
    validate_hist['trade_date'] = validate_hist['entry_time'].dt.date
    daily_control = validate_hist.groupby('trade_date')['r_multiple'].sum().rename('control_R')
    cum = daily_control.cumsum()
    dd = cum - cum.cummax()
    dd_threshold = dd.quantile(0.10)
    dd_days = set(dd[dd <= dd_threshold].index)

    corr_rows = []
    portfolio_rows = []
    for cand_id, path in [('EXP-125_XAUUSD_LONDON_VOL_EXPANSION', OUT / '_scratch_oos_EXP-125_XAUUSD_LONDON_VOL_EXPANSION.csv'),
                           ('EXP-126_USDCAD_MOMENTUM_CONTINUATION', OUT / '_scratch_oos_EXP-126_USDCAD_MOMENTUM_CONTINUATION.csv')]:
        if not path.exists():
            continue
        oos_trades = pd.read_csv(path, parse_dates=['entry_time'])
        oos_trades['trade_date'] = oos_trades['entry_time'].dt.date
        daily_cand = oos_trades.groupby('trade_date')['r_multiple'].sum().rename('candidate_R')

        merged = pd.concat([daily_control, daily_cand], axis=1).dropna()
        merged['is_dd'] = merged.index.isin(dd_days)
        normal_corr = merged.loc[~merged.is_dd, ['control_R', 'candidate_R']].corr().iloc[0, 1] if (~merged.is_dd).sum() > 5 else None
        dd_corr = merged.loc[merged.is_dd, ['control_R', 'candidate_R']].corr().iloc[0, 1] if merged.is_dd.sum() > 5 else None
        normal_ok = normal_corr is not None and normal_corr == normal_corr
        dd_ok = dd_corr is not None and dd_corr == dd_corr
        corr_rows.append({'candidate_id': cand_id, 'overlapping_days': len(merged),
                           'normal_day_corr': round(normal_corr, 3) if normal_ok else None,
                           'drawdown_day_corr': round(dd_corr, 3) if dd_ok else None,
                           'n_drawdown_day_overlap': int(merged.is_dd.sum()),
                           'diversification_gate': (
                               'FAIL -- E. POOR DRAWDOWN DIVERSIFICATION' if (normal_ok and dd_ok and dd_corr > normal_corr + 0.15)
                               else 'INSUFFICIENT OVERLAP' if (not normal_ok or not dd_ok)
                               else 'PASS')})

        # ---- portfolio integration: control vs control+candidate, 0.5x/1.0x ----
        control_std = daily_control.std()
        for weight in [0.5, 1.0]:
            scaled_cand = daily_cand * weight
            combined = pd.concat([daily_control, scaled_cand], axis=1).fillna(0).sum(axis=1)
            control_only = daily_control
            def metrics(series):
                c = series.cumsum()
                ddser = c - c.cummax()
                s = ms = 0
                for v in series:
                    if v < 0:
                        s += 1; ms = max(ms, s)
                    else:
                        s = 0
                return {'total_R': round(series.sum(), 2), 'max_dd': round(ddser.min(), 2), 'max_streak_days': ms}
            mc, mx = metrics(control_only), metrics(combined)
            portfolio_rows.append({'candidate_id': cand_id, 'weight': weight,
                                    'control_total_R': mc['total_R'], 'combined_total_R': mx['total_R'],
                                    'control_max_dd': mc['max_dd'], 'combined_max_dd': mx['max_dd'],
                                    'control_max_streak_days': mc['max_streak_days'], 'combined_max_streak_days': mx['max_streak_days']})

    corr_df = pd.DataFrame(corr_rows)
    portfolio_df = pd.DataFrame(portfolio_rows)
    print("\n=== drawdown correlation gate ===")
    print(corr_df.to_string())
    print("\n=== portfolio integration ===")
    print(portfolio_df.to_string())

    sens_df.to_csv(OUT / '_scratch_sensitivity.csv', index=False)
    corr_df.to_csv(OUT / 'phase33_drawdown_correlation.csv', index=False)
    portfolio_df.to_csv(OUT / 'phase33_portfolio_integration.csv', index=False)


if __name__ == '__main__':
    main()
