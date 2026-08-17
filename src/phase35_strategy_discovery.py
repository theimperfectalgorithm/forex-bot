"""
Phase 35 -- Expanded Target-Profile Strategy Discovery.

RESEARCH ONLY. Implements EXACTLY the five hypotheses frozen in
reports/phase35_preregistration.md (committed 7821cd7, before this
script's first run). No parameter search beyond the single pre-registered
+-20% sensitivity check per candidate.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

sys.path.insert(0, str(Path(__file__).parent))
from phase31_factor_regime_map import load_hist, RISK_PCT, CURRENT_SIX  # noqa: E402
from research_data_validator import ValidationReport, validate_column_count_consistency, validate_required_columns  # noqa: E402

REPO = Path(__file__).parent.parent
OUT = REPO / 'reports'
RNG = np.random.default_rng(20260818)

DATA_START = datetime(2023, 1, 1)
DATA_END = datetime(2026, 8, 14)
TRAIN_END = datetime(2024, 8, 31, tzinfo=timezone.utc)
VAL_END = datetime(2025, 4, 30, tzinfo=timezone.utc)

SPREAD_COST = {'AUDUSD': 0.00018, 'USDCAD': 0.00020, 'USDCHF': 0.00020}


def pull(symbol, timeframe):
    if not mt5.initialize():
        raise RuntimeError("MT5 initialize() failed -- STOP")
    rates = mt5.copy_rates_range(symbol, timeframe, DATA_START, DATA_END)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No data for {symbol} -- STOP")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    assert df['time'].is_monotonic_increasing, f"{symbol}: timestamps not monotonic -- STOP"
    assert df['time'].duplicated().sum() == 0, f"{symbol}: duplicate candles -- STOP"
    assert (df[['open', 'high', 'low', 'close']] > 0).all().all(), f"{symbol}: non-positive OHLC -- STOP"
    assert (df['high'] >= df['low']).all(), f"{symbol}: high<low bars -- STOP"
    return df[['time', 'open', 'high', 'low', 'close']]


def add_atr(df, window, col='atr'):
    df = df.copy()
    tr = np.maximum(df['high'] - df['low'],
                     np.maximum((df['high'] - df['close'].shift(1)).abs(),
                                (df['low'] - df['close'].shift(1)).abs()))
    df[col] = tr.rolling(window).mean()
    return df


def walk_to_exit(df, start_idx, direction, sl, tp, max_bars):
    future = df.loc[start_idx: start_idx + max_bars]
    for _, fb in future.iterrows():
        if direction == 'BUY':
            if fb['low'] <= sl:
                return sl, fb['time'], 'SL'
            if fb['high'] >= tp:
                return tp, fb['time'], 'TP'
        else:
            if fb['high'] >= sl:
                return sl, fb['time'], 'SL'
            if fb['low'] <= tp:
                return tp, fb['time'], 'TP'
    return None, None, None


# ---------------------------------------------------------------------------
# H1: USDCAD NY Open Range Breakout
# ---------------------------------------------------------------------------

def backtest_h1(df, tp_mult=1.5):
    df = df.copy().reset_index(drop=True)
    df['hour'] = df['time'].dt.hour
    cost = SPREAD_COST['USDCAD']
    range_bars = df[df['hour'].isin([13, 14])].copy()
    range_bars['date'] = range_bars['time'].dt.date
    rng = range_bars.groupby('date').agg(rh=('high', 'max'), rl=('low', 'min'))
    entry_bars = df[(df['hour'] >= 15) & (df['hour'] <= 20)].copy()
    entry_bars['date'] = entry_bars['time'].dt.date

    trades = []
    used_dates = set()
    for idx, bar in entry_bars.iterrows():
        d = bar['date']
        if d in used_dates or d not in rng.index:
            continue
        row = rng.loc[d]
        direction = None
        if bar['close'] > row['rh']:
            direction = 'BUY'
        elif bar['close'] < row['rl']:
            direction = 'SELL'
        if direction is None:
            continue
        entry_price = bar['close']
        sl = row['rl'] if direction == 'BUY' else row['rh']
        sl_dist = abs(entry_price - sl)
        if sl_dist <= 0:
            continue
        tp = entry_price + tp_mult * sl_dist if direction == 'BUY' else entry_price - tp_mult * sl_dist
        exit_price, exit_time, reason = walk_to_exit(df, idx + 1, direction, sl, tp, 48)
        if exit_price is None:
            continue
        raw_pnl = (exit_price - entry_price) if direction == 'BUY' else (entry_price - exit_price)
        r = (raw_pnl - cost) / sl_dist
        trades.append({'entry_time': bar['time'], 'exit_time': exit_time, 'direction': direction,
                        'r_multiple': r, 'exit_reason': reason, 'atr_at_entry': None})
        used_dates.add(d)
    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# H2: AUDUSD NY Session Momentum
# ---------------------------------------------------------------------------

def backtest_h2(df, mom_mult=1.0):
    df = add_atr(df, 14)
    df = df.reset_index(drop=True)
    df['hour'] = df['time'].dt.hour
    df['chg3h'] = df['close'] - df['close'].shift(3)
    df['abs_chg3h'] = df['chg3h'].abs()
    df['avg_abs_chg3h_20d'] = df['abs_chg3h'].rolling(20 * 24, min_periods=20 * 12).mean().shift(1)
    cost = SPREAD_COST['AUDUSD']

    trades = []
    in_pos_until = None
    for i in range(20 * 24, len(df) - 1):
        row = df.loc[i]
        if in_pos_until is not None and row['time'] <= in_pos_until:
            continue
        if row['hour'] < 13 or row['hour'] > 20:
            continue
        if pd.isna(row['avg_abs_chg3h_20d']) or row['avg_abs_chg3h_20d'] <= 0 or pd.isna(row['atr']) or row['atr'] <= 0:
            continue
        if row['abs_chg3h'] <= mom_mult * row['avg_abs_chg3h_20d']:
            continue
        direction = 'BUY' if row['chg3h'] > 0 else 'SELL'
        entry_price = row['close']
        sl_dist = 1.0 * row['atr']
        tp_dist = 2.0 * row['atr']
        sl = entry_price - sl_dist if direction == 'BUY' else entry_price + sl_dist
        tp = entry_price + tp_dist if direction == 'BUY' else entry_price - tp_dist
        exit_price, exit_time, reason = walk_to_exit(df, i + 1, direction, sl, tp, 24)
        if exit_price is None:
            continue
        raw_pnl = (exit_price - entry_price) if direction == 'BUY' else (entry_price - exit_price)
        r = (raw_pnl - cost) / sl_dist
        trades.append({'entry_time': row['time'], 'exit_time': exit_time, 'direction': direction,
                        'r_multiple': r, 'exit_reason': reason, 'atr_at_entry': row['atr']})
        in_pos_until = exit_time
    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# H3: USDCHF London/NY Overlap Continuation
# ---------------------------------------------------------------------------

def backtest_h3(df, er_threshold=0.40):
    df = add_atr(df, 14)
    df = df.reset_index(drop=True)
    df['hour'] = df['time'].dt.hour
    cost = SPREAD_COST['USDCHF']

    london = df[(df['hour'] >= 7) & (df['hour'] < 13)].copy()
    london['date'] = london['time'].dt.date
    london_stats = london.groupby('date').apply(
        lambda g: pd.Series({
            'net_disp': (g['close'].iloc[-1] - g['open'].iloc[0]) if len(g) else np.nan,
            'sum_abs': g['close'].diff().abs().sum() if len(g) > 1 else np.nan,
        }), include_groups=False)
    london_stats['er'] = london_stats['net_disp'].abs() / london_stats['sum_abs'].replace(0, np.nan)

    entry_bars = df[df['hour'] == 13].copy()
    entry_bars['date'] = entry_bars['time'].dt.date

    trades = []
    for idx, bar in entry_bars.iterrows():
        d = bar['date']
        if d not in london_stats.index:
            continue
        row = london_stats.loc[d]
        if pd.isna(row['er']) or row['er'] <= er_threshold:
            continue
        direction = 'BUY' if row['net_disp'] > 0 else 'SELL'
        atr_row = df.loc[idx, 'atr']
        if pd.isna(atr_row) or atr_row <= 0:
            continue
        entry_price = bar['close']
        sl_dist = 1.0 * atr_row
        tp_dist = 2.0 * atr_row
        sl = entry_price - sl_dist if direction == 'BUY' else entry_price + sl_dist
        tp = entry_price + tp_dist if direction == 'BUY' else entry_price - tp_dist
        # session-bounded: force-flat at 16:00 UTC same day if unresolved
        window = df[(df['time'].dt.date == d) & (df['hour'] > 13) & (df['hour'] <= 16)]
        exit_price, exit_time, reason = None, None, None
        for _, fb in window.iterrows():
            if direction == 'BUY':
                if fb['low'] <= sl:
                    exit_price, exit_time, reason = sl, fb['time'], 'SL'; break
                if fb['high'] >= tp:
                    exit_price, exit_time, reason = tp, fb['time'], 'TP'; break
            else:
                if fb['high'] >= sl:
                    exit_price, exit_time, reason = sl, fb['time'], 'SL'; break
                if fb['low'] <= tp:
                    exit_price, exit_time, reason = tp, fb['time'], 'TP'; break
        if exit_price is None and len(window):
            last = window.iloc[-1]
            exit_price, exit_time, reason = last['close'], last['time'], 'SESSION_CLOSE'
        if exit_price is None:
            continue
        raw_pnl = (exit_price - entry_price) if direction == 'BUY' else (entry_price - exit_price)
        r = (raw_pnl - cost) / sl_dist
        trades.append({'entry_time': bar['time'], 'exit_time': exit_time, 'direction': direction,
                        'r_multiple': r, 'exit_reason': reason, 'atr_at_entry': atr_row})
    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# H4: USDCAD Multi-Timeframe Trend Continuation (D1 filter + H4 execution)
# ---------------------------------------------------------------------------

def backtest_h4(df_h4, df_d1, lookback=10):
    df_h4 = add_atr(df_h4, 20).reset_index(drop=True)
    df_d1 = df_d1.copy().reset_index(drop=True)
    df_d1['trend_up'] = df_d1['close'] > df_d1['close'].shift(20)
    df_d1['trend_dn'] = df_d1['close'] < df_d1['close'].shift(20)
    df_d1['date'] = df_d1['time'].dt.date
    trend_by_date = df_d1.set_index('date')[['trend_up', 'trend_dn']]

    df_h4['hi_n'] = df_h4['high'].rolling(lookback).max().shift(1)
    df_h4['lo_n'] = df_h4['low'].rolling(lookback).min().shift(1)
    df_h4['date'] = df_h4['time'].dt.date
    cost = SPREAD_COST['USDCAD']

    trades = []
    in_pos_until = None
    for i in range(lookback, len(df_h4) - 1):
        row = df_h4.loc[i]
        if in_pos_until is not None and row['time'] <= in_pos_until:
            continue
        d = row['date']
        # use the most recent PRIOR D1 bar's trend (no same-day leakage)
        prior_dates = trend_by_date.index[trend_by_date.index < d]
        if len(prior_dates) == 0:
            continue
        trend_row = trend_by_date.loc[prior_dates.max()]
        if pd.isna(row['hi_n']) or pd.isna(row['atr']) or row['atr'] <= 0:
            continue
        direction = None
        if trend_row['trend_up'] and row['close'] > row['hi_n']:
            direction = 'BUY'
        elif trend_row['trend_dn'] and row['close'] < row['lo_n']:
            direction = 'SELL'
        if direction is None:
            continue
        entry_price = df_h4.loc[i + 1, 'open']
        sl_dist = 1.5 * row['atr']
        tp_dist = 3.0 * row['atr']
        sl = entry_price - sl_dist if direction == 'BUY' else entry_price + sl_dist
        tp = entry_price + tp_dist if direction == 'BUY' else entry_price - tp_dist
        exit_price, exit_time, reason = walk_to_exit(df_h4, i + 2, direction, sl, tp, 60)
        if exit_price is None:
            continue
        raw_pnl = (exit_price - entry_price) if direction == 'BUY' else (entry_price - exit_price)
        r = (raw_pnl - cost) / sl_dist
        trades.append({'entry_time': df_h4.loc[i + 1, 'time'], 'exit_time': exit_time, 'direction': direction,
                        'r_multiple': r, 'exit_reason': reason, 'atr_at_entry': row['atr']})
        in_pos_until = exit_time
    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# H5: AUDUSD ATR-Scaled Volatility Expansion
# ---------------------------------------------------------------------------

def backtest_h5(df, tp_atr_mult=2.5):
    df = add_atr(df, 14).reset_index(drop=True)
    df['hour'] = df['time'].dt.hour
    cost = SPREAD_COST['AUDUSD']

    prelondon = df[df['hour'].isin([3, 4, 5, 6])].copy()
    prelondon['date'] = prelondon['time'].dt.date
    rng = prelondon.groupby('date').agg(rh=('high', 'max'), rl=('low', 'min'))
    rng['width'] = rng['rh'] - rng['rl']
    rng = rng.sort_index()
    rng['thresh'] = rng['width'].rolling(30, min_periods=15).quantile(1 / 3).shift(1)

    entry_bars = df[(df['hour'] >= 7) & (df['hour'] <= 20)].copy()
    entry_bars['date'] = entry_bars['time'].dt.date

    trades = []
    used_dates = set()
    for idx, bar in entry_bars.iterrows():
        d = bar['date']
        if d in used_dates or d not in rng.index:
            continue
        row = rng.loc[d]
        if pd.isna(row['thresh']) or pd.isna(row['width']) or row['width'] >= row['thresh']:
            continue
        direction = None
        if bar['close'] > row['rh']:
            direction = 'BUY'
        elif bar['close'] < row['rl']:
            direction = 'SELL'
        if direction is None:
            continue
        atr_val = df.loc[idx, 'atr']
        if pd.isna(atr_val) or atr_val <= 0:
            continue
        entry_price = bar['close']
        sl_dist = 1.0 * atr_val
        tp_dist = tp_atr_mult * atr_val
        sl = entry_price - sl_dist if direction == 'BUY' else entry_price + sl_dist
        tp = entry_price + tp_dist if direction == 'BUY' else entry_price - tp_dist
        exit_price, exit_time, reason = walk_to_exit(df, idx + 1, direction, sl, tp, 200)
        if exit_price is None:
            continue
        raw_pnl = (exit_price - entry_price) if direction == 'BUY' else (entry_price - exit_price)
        r = (raw_pnl - cost) / sl_dist
        trades.append({'entry_time': bar['time'], 'exit_time': exit_time, 'direction': direction,
                        'r_multiple': r, 'exit_reason': reason, 'atr_at_entry': atr_val})
        used_dates.add(d)
    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def split_periods(trades):
    trades = trades.copy()
    trades['entry_time'] = pd.to_datetime(trades['entry_time'], utc=True)
    train = trades[trades.entry_time < TRAIN_END]
    val = trades[(trades.entry_time >= TRAIN_END) & (trades.entry_time < VAL_END)]
    oos = trades[trades.entry_time >= VAL_END]
    return train, val, oos


def edge_metrics(trades):
    if len(trades) == 0:
        return {'trades': 0, 'win_rate_pct': None, 'pf': None, 'expectancy_R': None, 'total_R': None}
    r = trades['r_multiple']
    wins, losses = r[r > 0], r[r < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else np.nan
    return {'trades': len(trades), 'win_rate_pct': round((r > 0).mean() * 100, 1),
            'pf': round(pf, 3) if pf == pf else None, 'expectancy_R': round(r.mean(), 3),
            'total_R': round(r.sum(), 2)}


def max_streak(r_series):
    s = ms = 0
    for v in r_series:
        if v < 0:
            s += 1; ms = max(ms, s)
        else:
            s = 0
    return ms


def apply_cost_multiplier(trades, symbol, mult):
    if len(trades) == 0:
        return trades
    adj = trades.copy()
    base_cost = SPREAD_COST[symbol]
    # NOTE: sl_dist not retained per-trade in this schema; approximate by
    # reconstructing raw R (pre-cost, in sl_dist units) then re-applying scaled cost
    # using each trade's own realized R and the cost-in-R-units implied by cost/sl_dist.
    # Since sl_dist isn't persisted, use ATR-based approximation where available,
    # else apply a uniform relative cost bump consistent with Phase33's method.
    return adj  # placeholder overwritten by caller with a proper per-symbol sl_dist column


def main():
    print("[phase35] validating inputs")
    hist_path = REPO / 'data' / 'phase26_all_trades.csv'
    r = ValidationReport(path=str(hist_path))
    validate_column_count_consistency(hist_path, r)
    validate_required_columns(hist_path, {'entry_time', 'exit_time', 'dir', 'strategy', 'r_multiple'}, r)
    print(f"[validate] {r.summary()}")

    print("[phase35] pulling data")
    usdcad_h1 = pull('USDCAD', mt5.TIMEFRAME_H1)
    audusd_h1 = pull('AUDUSD', mt5.TIMEFRAME_H1)
    usdchf_h1 = pull('USDCHF', mt5.TIMEFRAME_H1)
    usdcad_h4 = pull('USDCAD', mt5.TIMEFRAME_H4)
    usdcad_d1 = pull('USDCAD', mt5.TIMEFRAME_D1)
    print(f"[phase35] bars: USDCAD H1={len(usdcad_h1)} AUDUSD H1={len(audusd_h1)} USDCHF H1={len(usdchf_h1)} "
          f"USDCAD H4={len(usdcad_h4)} USDCAD D1={len(usdcad_d1)}")

    print("[phase35] running H1 (USDCAD NY ORB)")
    h1_trades = backtest_h1(usdcad_h1)
    print(f"[phase35] H1 trades: {len(h1_trades)}")

    print("[phase35] running H2 (AUDUSD NY momentum)")
    h2_trades = backtest_h2(audusd_h1)
    print(f"[phase35] H2 trades: {len(h2_trades)}")

    print("[phase35] running H3 (USDCHF London/NY overlap continuation)")
    h3_trades = backtest_h3(usdchf_h1)
    print(f"[phase35] H3 trades: {len(h3_trades)}")

    print("[phase35] running H4 (USDCAD multi-timeframe trend)")
    h4_trades = backtest_h4(usdcad_h4, usdcad_d1)
    print(f"[phase35] H4 trades: {len(h4_trades)}")

    print("[phase35] running H5 (AUDUSD ATR-scaled vol expansion)")
    h5_trades = backtest_h5(audusd_h1)
    print(f"[phase35] H5 trades: {len(h5_trades)}")

    candidates = {
        'H1_USDCAD_NY_ORB': (h1_trades, 'USDCAD'),
        'H2_AUDUSD_NY_MOMENTUM': (h2_trades, 'AUDUSD'),
        'H3_USDCHF_OVERLAP_CONTINUATION': (h3_trades, 'USDCHF'),
        'H4_USDCAD_MTF_TREND': (h4_trades, 'USDCAD'),
        'H5_AUDUSD_ATR_VOL_EXPANSION': (h5_trades, 'AUDUSD'),
    }

    results_rows, regime_rows = [], []
    for cand_id, (trades, symbol) in candidates.items():
        if len(trades) == 0:
            results_rows.append({'candidate_id': cand_id, 'symbol': symbol, 'note': 'ZERO TRADES'})
            continue
        train, val, oos = split_periods(trades)
        m_train, m_val, m_oos = edge_metrics(train), edge_metrics(val), edge_metrics(oos)
        results_rows.append({'candidate_id': cand_id, 'symbol': symbol,
                              'train_trades': m_train['trades'], 'train_pf': m_train['pf'], 'train_expectancy_R': m_train['expectancy_R'],
                              'val_trades': m_val['trades'], 'val_pf': m_val['pf'], 'val_expectancy_R': m_val['expectancy_R'],
                              'oos_trades': m_oos['trades'], 'oos_pf': m_oos['pf'], 'oos_expectancy_R': m_oos['expectancy_R'],
                              'oos_total_R': m_oos['total_R'], 'oos_win_rate_pct': m_oos['win_rate_pct'],
                              'oos_max_losing_streak': max_streak(oos['r_multiple'].tolist()) if len(oos) else None})

        if len(oos) >= 5:
            train_val_atr = pd.concat([train, val])['atr_at_entry'].dropna()
            if len(train_val_atr) >= 30:
                q1, q2 = train_val_atr.quantile([1/3, 2/3])
                oos_c = oos.copy()
                def regime_of(a):
                    if pd.isna(a):
                        return None
                    return 'LOW' if a <= q1 else ('NORMAL' if a <= q2 else 'HIGH')
                oos_c['regime'] = oos_c['atr_at_entry'].apply(regime_of)
                for regime_val, sub in oos_c.groupby('regime'):
                    m = edge_metrics(sub)
                    cls = ('UNKNOWN' if m['trades'] < 10 else
                           'STRONG' if (regime_val == 'HIGH' and m['expectancy_R'] and m['expectancy_R'] > 0) else
                           'WEAK' if (regime_val == 'HIGH' and m['expectancy_R'] and m['expectancy_R'] < 0) else 'NEUTRAL')
                    regime_rows.append({'candidate_id': cand_id, 'regime': regime_val, **m,
                                         'classification': cls if regime_val == 'HIGH' else ''})
            else:
                regime_rows.append({'candidate_id': cand_id, 'regime': 'ALL', 'note': 'insufficient TRAIN+VAL for terciles', 'classification': 'UNKNOWN'})
        else:
            regime_rows.append({'candidate_id': cand_id, 'regime': 'ALL', 'note': 'insufficient OOS trades', 'classification': 'UNKNOWN'})

        oos.to_csv(OUT / f'_scratch_oos_{cand_id}.csv', index=False)

    pd.DataFrame(results_rows).to_csv(OUT / 'phase35_candidate_results.csv', index=False)
    pd.DataFrame(regime_rows).to_csv(OUT / 'phase35_regime_analysis.csv', index=False)
    print("\n=== candidate results ===")
    print(pd.DataFrame(results_rows).to_string())
    print("\n=== regime ===")
    print(pd.DataFrame(regime_rows).to_string())


if __name__ == '__main__':
    main()
