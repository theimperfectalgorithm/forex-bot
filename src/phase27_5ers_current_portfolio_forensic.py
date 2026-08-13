"""
Phase 27 -- Final forensic investigation of the ACTUAL current 5ers portfolio,
using the real production trade export (reports/5ers_trade_export.csv,
generated on the VPS from C:\\forex-bot-5ers\\data\\{trades_log.csv,
journal/events.jsonl}).

DIAGNOSTIC ONLY. No strategy modification, no optimization, no deployment.
Read-only against every input file.

Historical reference population: data/phase26_all_trades.csv (2,712 trades,
the exact frozen-parameter reconstruction of the current 6-strategy book
used in the prior current_6_strategy_revalidation phase, EXP-105..111).
"""
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).parent.parent
EXPORT_CSV = REPO / 'reports' / '5ers_trade_export.csv'
HIST_TRADES_CSV = REPO / 'data' / 'phase26_all_trades.csv'
HIST_SUMMARY_CSV = REPO / 'reports' / 'current_6_strategy_revalidation.csv'

DEMOTION_DATE = datetime(2026, 7, 31, tzinfo=timezone.utc)
CURRENT_SIX = {'AUDJPY_AMR', 'CADJPY_AMR', 'EURJPY_AMR', 'GBPJPY_AMR', 'CADJPY_ARB', 'GBPUSD_MONDAY'}
JPY_STRATEGIES = {'AUDJPY_AMR', 'CADJPY_AMR', 'EURJPY_AMR', 'GBPJPY_AMR'}

RNG = np.random.default_rng(20260813)
N_MC = 20000

OUT = REPO / 'reports'


def load_export():
    df = pd.read_csv(EXPORT_CSV, dtype=str)
    print(f"[integrity] rows={len(df)} unique_trade_id={df['trade_id'].nunique()}")
    status_counts = df['status'].value_counts().to_dict()
    print(f"[integrity] status_counts={status_counts}")
    assert len(df) == 70, f"expected 70 rows, got {len(df)}"
    assert df['trade_id'].nunique() == 35, f"expected 35 unique tickets, got {df['trade_id'].nunique()}"
    assert status_counts.get('OPEN') == 35 and status_counts.get('CLOSED') == 35, "expected 35 OPEN / 35 CLOSED"
    assert (df['account'] == '5ERS').all(), "not all rows tagged account=5ERS"
    assert df['strategy'].notna().all() and (df['strategy'] != '').all(), "missing strategy attribution found"
    return df


def normalize_strategy(raw: str) -> str:
    """Export 'key' normalization used strategy_from_key(): PAIR_STRAT
    (e.g. GBPJPY_ARB, AUDJPY_AMR). GBPUSD Monday key comes through as
    GBPUSD_MON or GBPUSD_MONDAY depending on the journal's key string --
    normalize both."""
    s = raw.strip()
    if s in ('GBPUSD_MON', 'GBPUSD_MONDAY'):
        return 'GBPUSD_MONDAY'
    return s


def build_closed(df: pd.DataFrame) -> pd.DataFrame:
    closed = df[df['status'] == 'CLOSED'].copy()
    assert closed['trade_id'].nunique() == len(closed), "duplicate CLOSED trade_id found"
    closed['strategy_norm'] = closed['strategy'].apply(normalize_strategy)

    num_cols = ['entry_price', 'exit_price', 'lots', 'risk_percent', 'initial_risk',
                'profit', 'R', 'stop_loss', 'take_profit', 'spread', 'ATR', 'holding_time']
    for c in num_cols:
        closed[c] = pd.to_numeric(closed[c], errors='coerce')

    for c in ['signal_time', 'entry_time', 'exit_time']:
        closed[c + '_dt'] = pd.to_datetime(closed[c], errors='coerce', utc=True)

    closed['is_current_six'] = closed['strategy_norm'].isin(CURRENT_SIX)
    closed['pre_demotion'] = closed['demotion_status'] == 'PRE_DEMOTION'
    closed['post_demotion'] = closed['demotion_status'] == 'POST_DEMOTION'
    closed['is_win'] = closed['profit'] > 0
    closed['is_loss'] = closed['profit'] < 0

    # KNOWN BUG (PROJECT_REPORT.md, 2026-08-08 fix): agent_execution.place_trade()
    # logged 0.0 as entry_price for some trades before the fix -- confirmed present
    # in this export (most pre-Aug-8 rows have entry_price==0.0). SL/TP/PnL/R were
    # never affected, only the recorded entry_price -- so entry_price - stop_loss
    # is NOT a usable stop-distance proxy for those rows. Instead, derive the
    # intended SL distance in pips from initial_risk / (lots * pip_value_usd),
    # which uses only fields unaffected by the bug.
    closed['entry_price_valid'] = closed['entry_price'] > 0
    pip_val = closed['symbol'].apply(_pip_value_usd)
    closed['sl_pips_implied'] = closed['initial_risk'] / (closed['lots'] * pip_val)
    closed['spread_over_sl_pct'] = np.where(
        closed['sl_pips_implied'] > 0,
        closed['spread'] / closed['sl_pips_implied'] * 100.0,
        np.nan)
    closed = closed.sort_values('entry_time_dt').reset_index(drop=True)
    return closed


def _pip_value_usd(pair: str) -> float:
    if not pair:
        return 10.0
    if pair.upper() == 'XAUUSD':
        return 10.0
    if pair.upper().endswith('JPY'):
        return 6.7
    return 10.0


def r_recompute_check(closed: pd.DataFrame):
    """Independent R recheck: profit / initial_risk, compared to exported R."""
    recompute = closed['profit'] / closed['initial_risk']
    diff = (recompute - closed['R']).abs()
    mismatches = closed[diff > 0.02].copy()
    mismatches['recomputed_R'] = recompute[diff > 0.02]
    return mismatches[['trade_id', 'strategy_norm', 'profit', 'initial_risk', 'R', 'recomputed_R']]


def account_metrics(sub: pd.DataFrame) -> dict:
    n = len(sub)
    if n == 0:
        return {'trades': 0}
    wins = sub[sub['is_win']]
    losses = sub[sub['is_loss']]
    gross_win = wins['profit'].sum()
    gross_loss = losses['profit'].sum()
    pf = (gross_win / abs(gross_loss)) if gross_loss != 0 else np.inf
    total_r = sub['R'].sum()
    avg_r = sub['R'].mean()
    med_r = sub['R'].median()
    exp = avg_r
    win_rate = len(wins) / n * 100
    avg_win = wins['profit'].mean() if len(wins) else np.nan
    avg_loss = losses['profit'].mean() if len(losses) else np.nan
    payoff = (avg_win / abs(avg_loss)) if (avg_loss and not np.isnan(avg_loss) and avg_loss != 0) else np.nan

    # equity curve / drawdown in R, chronological
    ordered = sub.sort_values('entry_time_dt')
    cum_r = ordered['R'].cumsum()
    running_max = cum_r.cummax()
    dd = cum_r - running_max
    max_dd_r = dd.min() if len(dd) else 0
    current_dd_r = dd.iloc[-1] if len(dd) else 0

    # losing streaks
    streak = 0
    max_streak = 0
    cur_streak = 0
    outcomes = ordered['is_loss'].tolist()
    for i, is_l in enumerate(outcomes):
        if is_l:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    # current streak = trailing losses at the end
    for is_l in reversed(outcomes):
        if is_l:
            cur_streak += 1
        else:
            break

    win_streak = 0
    max_win_streak = 0
    for is_w in ordered['is_win'].tolist():
        if is_w:
            win_streak += 1
            max_win_streak = max(max_win_streak, win_streak)
        else:
            win_streak = 0

    exit_counts = sub['exit_reason'].value_counts(normalize=True) * 100

    return {
        'trades': n,
        'wins': len(wins),
        'losses': len(losses),
        'win_rate_pct': round(win_rate, 1),
        'total_pnl': round(sub['profit'].sum(), 2),
        'total_R': round(total_r, 2),
        'avg_R': round(avg_r, 3),
        'median_R': round(med_r, 3),
        'expectancy_R': round(exp, 3),
        'profit_factor': round(pf, 3) if np.isfinite(pf) else 'INF',
        'avg_win_usd': round(avg_win, 2) if not np.isnan(avg_win) else NA_STR,
        'avg_loss_usd': round(avg_loss, 2) if not np.isnan(avg_loss) else NA_STR,
        'payoff_ratio': round(payoff, 2) if payoff == payoff else NA_STR,
        'max_losing_streak': max_streak,
        'current_losing_streak': cur_streak,
        'max_winning_streak': max_win_streak,
        'max_drawdown_R': round(max_dd_r, 2),
        'current_drawdown_R': round(current_dd_r, 2),
        'largest_single_loss_usd': round(losses['profit'].min(), 2) if len(losses) else NA_STR,
        'largest_single_win_usd': round(wins['profit'].max(), 2) if len(wins) else NA_STR,
        'pct_SL': round(exit_counts.get('SL', 0), 1),
        'pct_TP': round(exit_counts.get('TP', 0), 1),
        'pct_scheduled_exit': round(exit_counts.get('SCHEDULED_STRATEGY_EXIT', 0), 1),
        'avg_holding_hours': round(sub['holding_time'].mean(), 2) if sub['holding_time'].notna().any() else NA_STR,
        'median_holding_hours': round(sub['holding_time'].median(), 2) if sub['holding_time'].notna().any() else NA_STR,
    }


NA_STR = 'NOT_AVAILABLE'


def strategy_breakdown(closed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strat, sub in closed.groupby('strategy_norm'):
        m = account_metrics(sub)
        m['strategy'] = strat
        m['is_current_six'] = strat in CURRENT_SIX
        rows.append(m)
    return pd.DataFrame(rows)


def directional_breakdown(closed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strat, direction), sub in closed.groupby(['strategy_norm', 'direction']):
        m = account_metrics(sub)
        m['strategy'] = strat
        m['direction'] = direction
        rows.append(m)
    return pd.DataFrame(rows)


def load_historical():
    hist = pd.read_csv(HIST_TRADES_CSV, parse_dates=['entry_time', 'exit_time'])
    summ = pd.read_csv(HIST_SUMMARY_CSV, engine='python', on_bad_lines='warn')
    return hist, summ


def monte_carlo_pooled(hist: pd.DataFrame, live_trade_count: int, n_sims=N_MC):
    """Pooled resampling: draw live_trade_count trades with replacement
    from the full historical pool (not preserving per-strategy weights)."""
    r_pool = hist['r_multiple'].dropna().values
    pfs, wrs, total_rs, maxdds = [], [], [], []
    for _ in range(n_sims):
        draw = RNG.choice(r_pool, size=live_trade_count, replace=True)
        wins = draw[draw > 0].sum()
        losses = draw[draw < 0].sum()
        pf = wins / abs(losses) if losses != 0 else np.inf
        wr = (draw > 0).mean() * 100
        cum = np.cumsum(draw)
        dd = (cum - np.maximum.accumulate(cum)).min()
        pfs.append(pf if np.isfinite(pf) else np.nan)
        wrs.append(wr)
        total_rs.append(draw.sum())
        maxdds.append(dd)
    return np.array(pfs), np.array(wrs), np.array(total_rs), np.array(maxdds)


def monte_carlo_strategy_aware(hist: pd.DataFrame, live_closed: pd.DataFrame, n_sims=N_MC):
    """Strategy-aware resampling: draw the SAME number of trades per
    strategy as actually occurred live, each from that strategy's own
    historical R-multiple pool -- preserves live strategy-frequency mix."""
    live_counts = live_closed[live_closed['is_current_six']]['strategy_norm'].value_counts().to_dict()
    pools = {s: hist[hist['strategy'] == s]['r_multiple'].dropna().values for s in live_counts}

    pfs, wrs, total_rs, maxdds, streaks = [], [], [], [], []
    for _ in range(n_sims):
        draw = []
        for s, n in live_counts.items():
            if len(pools[s]) == 0 or n == 0:
                continue
            draw.append(RNG.choice(pools[s], size=n, replace=True))
        if not draw:
            continue
        draw = np.concatenate(draw)
        RNG.shuffle(draw)
        wins = draw[draw > 0].sum()
        losses = draw[draw < 0].sum()
        pf = wins / abs(losses) if losses != 0 else np.inf
        wr = (draw > 0).mean() * 100
        cum = np.cumsum(draw)
        dd = (cum - np.maximum.accumulate(cum)).min()
        # losing streak
        streak = 0
        maxs = 0
        for v in draw:
            if v < 0:
                streak += 1
                maxs = max(maxs, streak)
            else:
                streak = 0
        pfs.append(pf if np.isfinite(pf) else np.nan)
        wrs.append(wr)
        total_rs.append(draw.sum())
        maxdds.append(dd)
        streaks.append(maxs)
    return (np.array(pfs), np.array(wrs), np.array(total_rs), np.array(maxdds), np.array(streaks), live_counts)


def percentile_rank(value, dist):
    dist = dist[~np.isnan(dist)]
    if len(dist) == 0 or value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    return float((dist < value).mean() * 100)


def main():
    df = load_export()
    closed = build_closed(df)
    closed.to_csv(OUT / '_scratch_closed_debug.csv', index=False)  # temp debug, removed below

    r_mismatch = r_recompute_check(closed)

    n_entry_price_bad = int((~closed['entry_price_valid']).sum())
    print(f"[execution] entry_price==0.0 (known fill-price logging bug) affects "
          f"{n_entry_price_bad}/{len(closed)} CLOSED trades")

    spread_buckets = pd.cut(closed['spread_over_sl_pct'],
                             bins=[0, 10, 20, 30, 40, np.inf],
                             labels=['<10%', '10-20%', '20-30%', '30-40%', '>40%'])
    closed['spread_over_sl_bucket'] = spread_buckets
    spread_bucket_perf = closed.groupby('spread_over_sl_bucket', observed=True).agg(
        trades=('trade_id', 'count'),
        win_rate=('is_win', lambda x: round(x.mean() * 100, 1)),
        avg_R=('R', lambda x: round(x.mean(), 3)))
    print("\n=== spread/SL bucket performance (implied-SL basis) ===")
    print(spread_bucket_perf.to_string())

    hist, summ = load_historical()

    # ---- Phase 3: account performance (current six, post-demotion only vs all) ----
    all_closed = closed
    current_six_all = closed[closed['is_current_six']]
    pre_demo = closed[closed['pre_demotion']]
    post_demo_current_six = closed[closed['is_current_six'] & (closed['post_demotion'] | closed['demotion_status'].str.contains('N/A'))]

    metrics_A = account_metrics(all_closed)
    metrics_B = account_metrics(current_six_all)
    metrics_C = account_metrics(pre_demo)
    metrics_D = account_metrics(post_demo_current_six)

    strat_table = strategy_breakdown(closed)
    dir_table = directional_breakdown(closed[closed['is_current_six']])

    # ---- exit reason ----
    exit_rows = []
    for (strat, reason), sub in closed[closed['is_current_six']].groupby(['strategy_norm', 'exit_reason']):
        exit_rows.append({
            'strategy': strat, 'exit_reason': reason, 'count': len(sub),
            'win_rate_pct': round((sub['profit'] > 0).mean() * 100, 1),
            'avg_R': round(sub['R'].mean(), 3),
            'avg_holding_hours': round(sub['holding_time'].mean(), 2) if sub['holding_time'].notna().any() else NA_STR,
        })
    exit_table = pd.DataFrame(exit_rows)

    # ---- volatility regime (ATR-based terciles from THIS live sample, since
    # we don't have this account's own historical ATR distribution to bucket
    # against -- documented explicitly as a live-sample-relative regime, not
    # the project's historical regime definition) ----
    live_atr = current_six_all['ATR'].dropna()
    regime_rows = []
    if len(live_atr) >= 6:
        q1, q2 = live_atr.quantile([1/3, 2/3])
        def regime_of(atr):
            if pd.isna(atr):
                return NA_STR
            if atr <= q1:
                return 'LOW'
            if atr <= q2:
                return 'NORMAL'
            return 'HIGH'
        current_six_all = current_six_all.copy()
        current_six_all['regime'] = current_six_all['ATR'].apply(regime_of)
        for (strat, regime), sub in current_six_all.groupby(['strategy_norm', 'regime']):
            regime_rows.append({
                'strategy': strat, 'regime': regime, 'trades': len(sub),
                'win_rate_pct': round((sub['profit'] > 0).mean() * 100, 1),
                'avg_R': round(sub['R'].mean(), 3),
                'total_R': round(sub['R'].sum(), 3),
            })
    regime_table = pd.DataFrame(regime_rows) if regime_rows else pd.DataFrame(
        [{'strategy': NA_STR, 'regime': NA_STR, 'trades': 0, 'win_rate_pct': NA_STR, 'avg_R': NA_STR, 'total_R': NA_STR,
          'note': 'ATR sample too small (<6 non-null values) to build terciles'}])

    # ---- JPY concentration / clustering ----
    jpy_trades = current_six_all[current_six_all['strategy_norm'].isin(JPY_STRATEGIES)]
    non_jpy_trades = current_six_all[~current_six_all['strategy_norm'].isin(JPY_STRATEGIES)]
    jpy_by_day = jpy_trades.copy()
    jpy_by_day['trade_date'] = jpy_by_day['entry_time_dt'].dt.date
    daily_jpy_strats = jpy_by_day.groupby('trade_date')['strategy_norm'].nunique()
    daily_jpy_losses = jpy_by_day[jpy_by_day['is_loss']].groupby('trade_date')['strategy_norm'].nunique()

    corr_rows = []
    for d, n_strats in daily_jpy_strats.items():
        n_losing = daily_jpy_losses.get(d, 0)
        corr_rows.append({'date': str(d), 'jpy_strategies_active': n_strats,
                           'jpy_strategies_losing': n_losing})
    corr_table = pd.DataFrame(corr_rows)

    multi_jpy_days = (daily_jpy_strats >= 2).sum()
    multi_jpy_losing_days = (daily_jpy_losses >= 2).sum()

    pct_trades_jpy = len(jpy_trades) / max(len(current_six_all), 1) * 100
    pct_risk_jpy = jpy_trades['initial_risk'].sum() / max(current_six_all['initial_risk'].sum(), 1e-9) * 100
    total_losing_r_all = current_six_all[current_six_all['is_loss']]['R'].sum()
    jpy_losing_r = jpy_trades[jpy_trades['is_loss']]['R'].sum()
    pct_losing_r_jpy = (jpy_losing_r / total_losing_r_all * 100) if total_losing_r_all != 0 else np.nan

    # ---- drawdown attribution (post-demotion current-six population) ----
    attrib_rows = []
    total_loss_usd = post_demo_current_six[post_demo_current_six['is_loss']]['profit'].sum()
    total_loss_r = post_demo_current_six[post_demo_current_six['is_loss']]['R'].sum()
    for strat in sorted(CURRENT_SIX) + (['GBPJPY_ARB (pre-demotion only)'] if len(pre_demo) else []):
        if strat.startswith('GBPJPY_ARB'):
            sub = pre_demo
            strat_label = 'GBPJPY_ARB (pre-demotion)'
        else:
            sub = post_demo_current_six[post_demo_current_six['strategy_norm'] == strat]
            strat_label = strat
        losses = sub[sub['is_loss']]
        attrib_rows.append({
            'strategy': strat_label,
            'trades': len(sub),
            'wins': int((sub['profit'] > 0).sum()),
            'losses': len(losses),
            'dollar_contribution': round(sub['profit'].sum(), 2),
            'R_contribution': round(sub['R'].sum(), 2),
            'pct_of_total_loss_usd': round(losses['profit'].sum() / total_loss_usd * 100, 1) if total_loss_usd != 0 else NA_STR,
            'pct_of_total_loss_R': round(losses['R'].sum() / total_loss_r * 100, 1) if total_loss_r != 0 else NA_STR,
        })
    attrib_table = pd.DataFrame(attrib_rows)

    # ---- Monte Carlo ----
    live_n_current_six = len(post_demo_current_six)
    pooled_pf, pooled_wr, pooled_totalr, pooled_dd = monte_carlo_pooled(hist, live_n_current_six)
    (sa_pf, sa_wr, sa_totalr, sa_dd, sa_streak, sa_counts) = monte_carlo_strategy_aware(hist, post_demo_current_six)

    live_pf = metrics_D['profit_factor'] if metrics_D['profit_factor'] != 'INF' else np.inf
    live_wr = metrics_D['win_rate_pct']
    live_streak = metrics_D['max_losing_streak']
    live_dd_r = metrics_D['max_drawdown_R']

    mc_rows = [
        {'method': 'pooled', 'metric': 'PF', 'p1': np.nanpercentile(pooled_pf, 1), 'p5': np.nanpercentile(pooled_pf, 5),
         'p10': np.nanpercentile(pooled_pf, 10), 'p25': np.nanpercentile(pooled_pf, 25), 'p50': np.nanpercentile(pooled_pf, 50),
         'p75': np.nanpercentile(pooled_pf, 75), 'p90': np.nanpercentile(pooled_pf, 90), 'p95': np.nanpercentile(pooled_pf, 95),
         'p99': np.nanpercentile(pooled_pf, 99), 'observed': live_pf, 'observed_percentile': percentile_rank(live_pf, pooled_pf)},
        {'method': 'pooled', 'metric': 'win_rate', 'p1': np.nanpercentile(pooled_wr, 1), 'p5': np.nanpercentile(pooled_wr, 5),
         'p10': np.nanpercentile(pooled_wr, 10), 'p25': np.nanpercentile(pooled_wr, 25), 'p50': np.nanpercentile(pooled_wr, 50),
         'p75': np.nanpercentile(pooled_wr, 75), 'p90': np.nanpercentile(pooled_wr, 90), 'p95': np.nanpercentile(pooled_wr, 95),
         'p99': np.nanpercentile(pooled_wr, 99), 'observed': live_wr, 'observed_percentile': percentile_rank(live_wr, pooled_wr)},
        {'method': 'pooled', 'metric': 'max_drawdown_R', 'p1': np.nanpercentile(pooled_dd, 1), 'p5': np.nanpercentile(pooled_dd, 5),
         'p10': np.nanpercentile(pooled_dd, 10), 'p25': np.nanpercentile(pooled_dd, 25), 'p50': np.nanpercentile(pooled_dd, 50),
         'p75': np.nanpercentile(pooled_dd, 75), 'p90': np.nanpercentile(pooled_dd, 90), 'p95': np.nanpercentile(pooled_dd, 95),
         'p99': np.nanpercentile(pooled_dd, 99), 'observed': live_dd_r, 'observed_percentile': percentile_rank(live_dd_r, pooled_dd)},
        {'method': 'strategy_aware', 'metric': 'PF', 'p1': np.nanpercentile(sa_pf, 1), 'p5': np.nanpercentile(sa_pf, 5),
         'p10': np.nanpercentile(sa_pf, 10), 'p25': np.nanpercentile(sa_pf, 25), 'p50': np.nanpercentile(sa_pf, 50),
         'p75': np.nanpercentile(sa_pf, 75), 'p90': np.nanpercentile(sa_pf, 90), 'p95': np.nanpercentile(sa_pf, 95),
         'p99': np.nanpercentile(sa_pf, 99), 'observed': live_pf, 'observed_percentile': percentile_rank(live_pf, sa_pf)},
        {'method': 'strategy_aware', 'metric': 'win_rate', 'p1': np.nanpercentile(sa_wr, 1), 'p5': np.nanpercentile(sa_wr, 5),
         'p10': np.nanpercentile(sa_wr, 10), 'p25': np.nanpercentile(sa_wr, 25), 'p50': np.nanpercentile(sa_wr, 50),
         'p75': np.nanpercentile(sa_wr, 75), 'p90': np.nanpercentile(sa_wr, 90), 'p95': np.nanpercentile(sa_wr, 95),
         'p99': np.nanpercentile(sa_wr, 99), 'observed': live_wr, 'observed_percentile': percentile_rank(live_wr, sa_wr)},
        {'method': 'strategy_aware', 'metric': 'max_drawdown_R', 'p1': np.nanpercentile(sa_dd, 1), 'p5': np.nanpercentile(sa_dd, 5),
         'p10': np.nanpercentile(sa_dd, 10), 'p25': np.nanpercentile(sa_dd, 25), 'p50': np.nanpercentile(sa_dd, 50),
         'p75': np.nanpercentile(sa_dd, 75), 'p90': np.nanpercentile(sa_dd, 90), 'p95': np.nanpercentile(sa_dd, 95),
         'p99': np.nanpercentile(sa_dd, 99), 'observed': live_dd_r, 'observed_percentile': percentile_rank(live_dd_r, sa_dd)},
        {'method': 'strategy_aware', 'metric': 'max_losing_streak', 'p1': np.nanpercentile(sa_streak, 1), 'p5': np.nanpercentile(sa_streak, 5),
         'p10': np.nanpercentile(sa_streak, 10), 'p25': np.nanpercentile(sa_streak, 25), 'p50': np.nanpercentile(sa_streak, 50),
         'p75': np.nanpercentile(sa_streak, 75), 'p90': np.nanpercentile(sa_streak, 90), 'p95': np.nanpercentile(sa_streak, 95),
         'p99': np.nanpercentile(sa_streak, 99), 'observed': live_streak, 'observed_percentile': percentile_rank(live_streak, sa_streak)},
    ]
    mc_table = pd.DataFrame(mc_rows)

    # ---- write CSV deliverables ----
    trade_level_cols = ['trade_id', 'strategy_norm', 'symbol', 'direction', 'signal_time', 'entry_time', 'exit_time',
                         'holding_time', 'entry_price', 'exit_price', 'stop_loss', 'take_profit', 'lots',
                         'risk_percent', 'initial_risk', 'profit', 'R', 'exit_reason', 'raw_exit_reason',
                         'spread', 'ATR', 'strategy_reason', 'demotion_status', 'r_source', 'match_method']
    closed[trade_level_cols].rename(columns={'strategy_norm': 'strategy'}).to_csv(
        OUT / '5ers_current_portfolio_forensic_trade_level.csv', index=False)

    strat_table.to_csv(OUT / '5ers_current_portfolio_forensic_strategy_summary.csv', index=False)
    attrib_table.to_csv(OUT / '5ers_current_portfolio_forensic_drawdown_attribution.csv', index=False)
    mc_table.to_csv(OUT / '5ers_current_portfolio_forensic_monte_carlo.csv', index=False)
    regime_table.to_csv(OUT / '5ers_current_portfolio_forensic_regime_analysis.csv', index=False)
    corr_table.to_csv(OUT / '5ers_current_portfolio_forensic_correlation.csv', index=False)

    (OUT / '_scratch_closed_debug.csv').unlink(missing_ok=True)

    # ---- dump key numbers to a JSON the report-writer reads ----
    summary_blob = {
        'metrics_A_all_closed_all_time': metrics_A,
        'metrics_B_current_six_all_time': metrics_B,
        'metrics_C_pre_demotion': metrics_C,
        'metrics_D_post_demotion_current_six': metrics_D,
        'r_mismatch_rows': r_mismatch.to_dict(orient='records'),
        'pct_trades_jpy': round(pct_trades_jpy, 1),
        'pct_risk_jpy': round(pct_risk_jpy, 1),
        'pct_losing_r_jpy': round(pct_losing_r_jpy, 1) if pct_losing_r_jpy == pct_losing_r_jpy else NA_STR,
        'multi_jpy_active_days': int(multi_jpy_days),
        'multi_jpy_losing_days': int(multi_jpy_losing_days),
        'jpy_active_days_total': int(len(daily_jpy_strats)),
        'sa_mc_live_counts': {k: int(v) for k, v in sa_counts.items()},
        'live_n_current_six_closed': live_n_current_six,
        'pre_demo_trade_ids': pre_demo['trade_id'].tolist(),
        'pre_demo_strategies': pre_demo['strategy_norm'].unique().tolist(),
        'date_range': [str(closed['entry_time_dt'].min()), str(closed['entry_time_dt'].max())],
    }
    with open(OUT / '_phase27_summary_blob.json', 'w') as f:
        json.dump(summary_blob, f, indent=2, default=str)

    print(json.dumps(summary_blob, indent=2, default=str))
    print("\n=== strategy table ===")
    print(strat_table.to_string())
    print("\n=== directional table ===")
    print(dir_table.to_string())
    print("\n=== exit table ===")
    print(exit_table.to_string())
    print("\n=== attribution table ===")
    print(attrib_table.to_string())
    print("\n=== MC table ===")
    print(mc_table.to_string())
    print("\n=== R mismatches ===")
    print(r_mismatch.to_string())


if __name__ == '__main__':
    main()
