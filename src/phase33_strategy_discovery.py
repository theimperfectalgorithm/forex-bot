"""
Phase 33 -- Target-Profile Strategy Discovery & Pre-Registered Validation.

RESEARCH ONLY. No live strategy/parameter/risk/portfolio-weight change.
No candidate deployed.

Implements EXACTLY the two candidates frozen in reports/phase33_preregistration.md
(committed 8bcd30e, before this script's first run) -- no parameter search,
no additional candidates. Every gate/threshold below is copied verbatim from
that frozen document; none were adjusted after seeing results.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import MetaTrader5 as mt5

sys.path.insert(0, str(Path(__file__).parent))
from phase31_factor_regime_map import CURRENT_SIX, load_hist, validate_inputs, RISK_PCT  # noqa: E402

REPO = Path(__file__).parent.parent
OUT = REPO / 'reports'
RNG = np.random.default_rng(20260816)

DATA_START = datetime(2023, 1, 1)
DATA_END = datetime(2026, 8, 14)
TRAIN_END = datetime(2024, 8, 31, tzinfo=timezone.utc)
VAL_END = datetime(2025, 4, 30, tzinfo=timezone.utc)
# OOS = VAL_END .. DATA_END

SPREAD_COST = {'XAUUSD': 0.35, 'USDCAD': 0.00020}


def pull(symbol, timeframe):
    if not mt5.initialize():
        raise RuntimeError("MT5 initialize() failed -- STOP per data-integrity policy")
    rates = mt5.copy_rates_range(symbol, timeframe, DATA_START, DATA_END)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No data for {symbol} -- STOP, do not silently proceed")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    # data-integrity checks (Part 6): duplicates, monotonic time, no negative/zero OHLC
    assert df['time'].is_monotonic_increasing, f"{symbol}: timestamps not monotonic -- STOP"
    assert df['time'].duplicated().sum() == 0, f"{symbol}: duplicate candles found -- STOP"
    assert (df[['open', 'high', 'low', 'close']] > 0).all().all(), f"{symbol}: non-positive OHLC -- STOP"
    assert (df['high'] >= df['low']).all(), f"{symbol}: high<low bars found -- STOP"
    return df[['time', 'open', 'high', 'low', 'close']]


# ---------------------------------------------------------------------------
# Candidate 1: XAUUSD London volatility-expansion breakout
# ---------------------------------------------------------------------------

def backtest_xauusd(df):
    df = df.copy().reset_index(drop=True)
    df['hour'] = df['time'].dt.hour
    df['tr'] = np.maximum(df['high'] - df['low'],
                           np.maximum((df['high'] - df['close'].shift(1)).abs(),
                                      (df['low'] - df['close'].shift(1)).abs()))
    df['atr14'] = df['tr'].rolling(14).mean()

    trades = []
    cost = SPREAD_COST['XAUUSD']

    # pre-London 4h range = hours 3,4,5,6 UTC; London open = hour 7
    prelondon = df[df['hour'].isin([3, 4, 5, 6])].copy()
    prelondon['date'] = prelondon['time'].dt.date
    range_by_day = prelondon.groupby('date').agg(range_high=('high', 'max'), range_low=('low', 'min'))
    range_by_day['range_width'] = range_by_day['range_high'] - range_by_day['range_low']

    # 30-day rolling 33rd percentile of range width, using only PAST data (no leakage)
    range_by_day = range_by_day.sort_index()
    range_by_day['range_pctile_threshold'] = range_by_day['range_width'].rolling(30, min_periods=15).quantile(1/3).shift(1)

    london_bars = df[df['hour'] == 7].copy()
    london_bars['date'] = london_bars['time'].dt.date

    for _, lb in london_bars.iterrows():
        d = lb['date']
        if d not in range_by_day.index:
            continue
        row = range_by_day.loc[d]
        if pd.isna(row['range_pctile_threshold']) or pd.isna(row['range_width']):
            continue
        if row['range_width'] >= row['range_pctile_threshold']:
            continue  # precondition (contraction) not met

        # find first H1 close beyond the pre-London range, starting at London open bar
        window = df[(df['time'].dt.date == d) & (df['hour'] >= 7) & (df['hour'] <= 20)]
        entry_idx = None
        direction = None
        for idx, bar in window.iterrows():
            if bar['close'] > row['range_high']:
                entry_idx = idx
                direction = 'BUY'
                break
            if bar['close'] < row['range_low']:
                entry_idx = idx
                direction = 'SELL'
                break
        if entry_idx is None:
            continue

        entry_price = df.loc[entry_idx, 'close']
        sl = row['range_low'] if direction == 'BUY' else row['range_high']
        sl_dist = abs(entry_price - sl)
        if sl_dist <= 0:
            continue
        tp = entry_price + 2.0 * sl_dist if direction == 'BUY' else entry_price - 2.0 * sl_dist

        # walk forward from entry_idx+1 to find SL/TP hit (SL-first on ambiguous bars, project convention)
        future = df.loc[entry_idx + 1: entry_idx + 1 + 200]  # cap search window (~8 days of H1)
        exit_price, exit_time, reason = None, None, None
        for _, fb in future.iterrows():
            if direction == 'BUY':
                if fb['low'] <= sl:
                    exit_price, exit_time, reason = sl, fb['time'], 'SL'
                    break
                if fb['high'] >= tp:
                    exit_price, exit_time, reason = tp, fb['time'], 'TP'
                    break
            else:
                if fb['high'] >= sl:
                    exit_price, exit_time, reason = sl, fb['time'], 'SL'
                    break
                if fb['low'] <= tp:
                    exit_price, exit_time, reason = tp, fb['time'], 'TP'
                    break
        if exit_price is None:
            continue  # never resolved within window -- excluded, not force-closed (documented limitation)

        raw_pnl = (exit_price - entry_price) if direction == 'BUY' else (entry_price - exit_price)
        net_pnl = raw_pnl - cost
        r_multiple = net_pnl / sl_dist
        trades.append({'entry_time': df.loc[entry_idx, 'time'], 'exit_time': exit_time,
                        'direction': direction, 'entry_price': entry_price, 'exit_price': exit_price,
                        'sl_dist': sl_dist, 'r_multiple': r_multiple, 'exit_reason': reason,
                        'atr_at_entry': df.loc[entry_idx, 'atr14']})
    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# Candidate 2: USDCAD H4 momentum continuation
# ---------------------------------------------------------------------------

def backtest_usdcad(df):
    df = df.copy().reset_index(drop=True)
    df['tr'] = np.maximum(df['high'] - df['low'],
                           np.maximum((df['high'] - df['close'].shift(1)).abs(),
                                      (df['low'] - df['close'].shift(1)).abs()))
    df['atr20'] = df['tr'].rolling(20).mean()
    df['hi20'] = df['high'].rolling(20).max().shift(1)  # prior 20-bar high, excluding current bar
    df['lo20'] = df['low'].rolling(20).min().shift(1)
    net_disp = (df['close'] - df['close'].shift(20)).abs()
    sum_abs_moves = df['close'].diff().abs().rolling(20).sum()
    df['efficiency_ratio'] = net_disp / sum_abs_moves.replace(0, np.nan)

    cost = SPREAD_COST['USDCAD']
    trades = []
    in_position_until = None
    for i in range(20, len(df) - 1):
        if in_position_until is not None and df.loc[i, 'time'] <= in_position_until:
            continue
        row = df.loc[i]
        if pd.isna(row['hi20']) or pd.isna(row['efficiency_ratio']) or pd.isna(row['atr20']) or row['atr20'] <= 0:
            continue
        direction = None
        if row['close'] > row['hi20'] and row['efficiency_ratio'] > 0.35:
            direction = 'BUY'
        elif row['close'] < row['lo20'] and row['efficiency_ratio'] > 0.35:
            direction = 'SELL'
        if direction is None:
            continue

        entry_price = df.loc[i + 1, 'open']
        sl_dist = 1.5 * row['atr20']
        tp_dist = 3.0 * row['atr20']
        sl = entry_price - sl_dist if direction == 'BUY' else entry_price + sl_dist
        tp = entry_price + tp_dist if direction == 'BUY' else entry_price - tp_dist

        future = df.loc[i + 2: i + 2 + 60]  # cap ~10 trading days of H4 bars
        exit_price, exit_time, reason = None, None, None
        for _, fb in future.iterrows():
            if direction == 'BUY':
                if fb['low'] <= sl:
                    exit_price, exit_time, reason = sl, fb['time'], 'SL'
                    break
                if fb['high'] >= tp:
                    exit_price, exit_time, reason = tp, fb['time'], 'TP'
                    break
            else:
                if fb['high'] >= sl:
                    exit_price, exit_time, reason = sl, fb['time'], 'SL'
                    break
                if fb['low'] <= tp:
                    exit_price, exit_time, reason = tp, fb['time'], 'TP'
                    break
        if exit_price is None:
            continue

        raw_pnl = (exit_price - entry_price) if direction == 'BUY' else (entry_price - exit_price)
        net_pnl = raw_pnl - cost
        r_multiple = net_pnl / sl_dist
        trades.append({'entry_time': df.loc[i + 1, 'time'], 'exit_time': exit_time,
                        'direction': direction, 'entry_price': entry_price, 'exit_price': exit_price,
                        'sl_dist': sl_dist, 'r_multiple': r_multiple, 'exit_reason': reason,
                        'atr_at_entry': row['atr20']})
        in_position_until = exit_time
    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# Shared analysis utilities
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
            s += 1
            ms = max(ms, s)
        else:
            s = 0
    return ms


def apply_cost_multiplier(trades, symbol, mult):
    """Re-derive R at a higher cost assumption without re-running the full
    backtest (SL/TP triggers are unaffected by cost -- only realized PnL is)."""
    extra_cost = SPREAD_COST[symbol] * (mult - 1.0)
    adj = trades.copy()
    raw_pnl = adj['r_multiple'] * adj['sl_dist'] + SPREAD_COST[symbol]  # back out raw pnl
    adj_pnl = raw_pnl - SPREAD_COST[symbol] * mult
    adj['r_multiple'] = adj_pnl / adj['sl_dist']
    return adj


def main():
    validate_inputs()
    print("[phase33] pulling candidate data (MetaQuotes-Demo feed, disclosed limitation per preregistration)")
    xau_df = pull('XAUUSD', mt5.TIMEFRAME_H1)
    cad_df = pull('USDCAD', mt5.TIMEFRAME_H4)
    print(f"[phase33] XAUUSD H1 bars: {len(xau_df)}, USDCAD H4 bars: {len(cad_df)}")

    print("[phase33] running Candidate 1: XAUUSD_LONDON_VOL_EXPANSION")
    xau_trades = backtest_xauusd(xau_df)
    print(f"[phase33] Candidate 1 total trades generated: {len(xau_trades)}")

    print("[phase33] running Candidate 2: USDCAD_MOMENTUM_CONTINUATION")
    cad_trades = backtest_usdcad(cad_df)
    print(f"[phase33] Candidate 2 total trades generated: {len(cad_trades)}")

    candidates = {'EXP-125_XAUUSD_LONDON_VOL_EXPANSION': (xau_trades, 'XAUUSD'),
                  'EXP-126_USDCAD_MOMENTUM_CONTINUATION': (cad_trades, 'USDCAD')}

    results_rows = []
    robustness_rows = []
    cost_rows = []
    regime_rows = []
    mc_rows = []
    for cand_id, (trades, symbol) in candidates.items():
        if len(trades) == 0:
            results_rows.append({'candidate_id': cand_id, 'symbol': symbol, 'note': 'ZERO TRADES GENERATED -- cannot evaluate'})
            continue
        train, val, oos = split_periods(trades)
        m_train, m_val, m_oos = edge_metrics(train), edge_metrics(val), edge_metrics(oos)
        results_rows.append({'candidate_id': cand_id, 'symbol': symbol,
                              'train_trades': m_train['trades'], 'train_pf': m_train['pf'], 'train_expectancy_R': m_train['expectancy_R'],
                              'val_trades': m_val['trades'], 'val_pf': m_val['pf'], 'val_expectancy_R': m_val['expectancy_R'],
                              'oos_trades': m_oos['trades'], 'oos_pf': m_oos['pf'], 'oos_expectancy_R': m_oos['expectancy_R'],
                              'oos_total_R': m_oos['total_R'], 'oos_win_rate_pct': m_oos['win_rate_pct'],
                              'oos_max_losing_streak': max_streak(oos['r_multiple'].tolist()) if len(oos) else None})

        # --- robustness: OOS sub-halves ---
        if len(oos) >= 10:
            mid = oos['entry_time'].median()
            oos_h1 = oos[oos.entry_time < mid]
            oos_h2 = oos[oos.entry_time >= mid]
            m_h1, m_h2 = edge_metrics(oos_h1), edge_metrics(oos_h2)
            sign_consistent = (m_h1['expectancy_R'] or 0) * (m_h2['expectancy_R'] or 0) > 0
        else:
            m_h1 = m_h2 = {'expectancy_R': None}
            sign_consistent = None

        # --- robustness: Monte Carlo reshuffle of OOS trade order ---
        if len(oos) >= 10:
            r_arr = oos['r_multiple'].values
            mc_dd = []
            for _ in range(10000):
                shuf = RNG.permutation(r_arr)
                cum = np.cumsum(shuf)
                mc_dd.append((cum - np.maximum.accumulate(cum)).min())
            mc_dd = np.array(mc_dd)
            actual_cum = np.cumsum(r_arr)
            actual_dd = (actual_cum - np.maximum.accumulate(actual_cum)).min()
            mc_rows.append({'candidate_id': cand_id, 'n_sims': 10000, 'oos_trades': len(oos),
                             'actual_oos_max_dd_R': round(actual_dd, 2),
                             'mc_dd_p5': round(np.percentile(mc_dd, 5), 2), 'mc_dd_p50': round(np.percentile(mc_dd, 50), 2),
                             'mc_dd_p95': round(np.percentile(mc_dd, 95), 2),
                             'actual_dd_percentile_in_mc': round(float((mc_dd < actual_dd).mean() * 100), 1)})
        else:
            mc_rows.append({'candidate_id': cand_id, 'note': 'insufficient OOS trades for MC (<10)'})

        robustness_rows.append({
            'candidate_id': cand_id,
            'oos_h1_expectancy_R': m_h1['expectancy_R'], 'oos_h2_expectancy_R': m_h2['expectancy_R'],
            'oos_subhalf_sign_consistent': sign_consistent,
            'param_sensitivity_note': 'see phase33_strategy_discovery.md -- +-20% threshold perturbation results',
        })

        # --- cost stress ---
        for mult, label in [(1.0, 'normal'), (1.5, '1.5x'), (2.0, '2.0x')]:
            adj_oos = apply_cost_multiplier(oos, symbol, mult) if mult != 1.0 else oos
            m_adj = edge_metrics(adj_oos)
            cost_rows.append({'candidate_id': cand_id, 'cost_multiplier': label, 'oos_pf': m_adj['pf'],
                               'oos_expectancy_R': m_adj['expectancy_R'], 'oos_trades': m_adj['trades']})

        # --- HIGH-vol regime gate (own-instrument ATR terciles) ---
        oos_c = oos.copy()
        train_val_atr = pd.concat([train, val])['atr_at_entry'].dropna()
        if len(train_val_atr) >= 30:
            q1, q2 = train_val_atr.quantile([1/3, 2/3])  # terciles fixed from TRAIN+VAL only, no leakage into OOS
            def regime_of(a):
                if pd.isna(a):
                    return None
                return 'LOW' if a <= q1 else ('NORMAL' if a <= q2 else 'HIGH')
            oos_c['regime'] = oos_c['atr_at_entry'].apply(regime_of)
            for regime_val, sub in oos_c.groupby('regime'):
                m = edge_metrics(sub)
                classification = ('UNKNOWN' if m['trades'] < 10 else
                                   'STRONG HIGH-VOL COMPATIBILITY' if (regime_val == 'HIGH' and m['expectancy_R'] and m['expectancy_R'] > 0) else
                                   'WEAK' if (regime_val == 'HIGH' and m['expectancy_R'] and m['expectancy_R'] < 0) else
                                   'NEUTRAL')
                regime_rows.append({'candidate_id': cand_id, 'regime': regime_val, **m,
                                     'classification': classification if regime_val == 'HIGH' else ''})
        else:
            regime_rows.append({'candidate_id': cand_id, 'regime': 'ALL', 'note': 'insufficient TRAIN+VAL trades to fix ATR terciles',
                                 'classification': 'UNKNOWN'})

    pd.DataFrame(results_rows).to_csv(OUT / 'phase33_candidate_results.csv', index=False)
    pd.DataFrame(robustness_rows).to_csv(OUT / 'phase33_robustness_results.csv', index=False)
    pd.DataFrame(cost_rows).to_csv(OUT / 'phase33_cost_stress.csv', index=False)
    pd.DataFrame(regime_rows).to_csv(OUT / 'phase33_regime_results.csv', index=False)
    pd.DataFrame(mc_rows).to_csv(OUT / 'phase33_monte_carlo.csv', index=False)

    print("\n=== candidate results ===")
    print(pd.DataFrame(results_rows).to_string())
    print("\n=== robustness ===")
    print(pd.DataFrame(robustness_rows).to_string())
    print("\n=== cost stress ===")
    print(pd.DataFrame(cost_rows).to_string())
    print("\n=== regime ===")
    print(pd.DataFrame(regime_rows).to_string())
    print("\n=== Monte Carlo ===")
    print(pd.DataFrame(mc_rows).to_string())

    # save raw OOS trade streams for the portfolio-integration step
    for cand_id, (trades, symbol) in candidates.items():
        if len(trades):
            _, _, oos = split_periods(trades)
            oos.to_csv(OUT / f'_scratch_oos_{cand_id}.csv', index=False)


if __name__ == '__main__':
    main()
