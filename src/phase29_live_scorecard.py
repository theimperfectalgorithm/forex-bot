"""
Phase 29 -- Live Strategy Scorecard analysis engine.

Builds the four data windows (entire live history / pre-demotion /
post-demotion / 2026-08-09 onward) per current-six strategy from the
fresh production export, computes live metrics, bootstrap CIs, BUY/SELL
and regime splits, and a portfolio-level concentration/correlation
summary. DIAGNOSTIC ONLY -- no strategy/parameter/risk/config changes.

Historical reference: reports/current_6_strategy_revalidation.csv
(pre-live acceptance criteria, EXP-096..104) and
data/phase26_all_trades.csv (2,712-trade historical population,
EXP-105..111) for bootstrap/resampling comparisons.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import phase27_5ers_current_portfolio_forensic as p27

REPO = Path(__file__).parent.parent
OUT = REPO / 'reports'

DEMOTION_DATE = datetime(2026, 7, 31, tzinfo=timezone.utc)
RECENT_START = datetime(2026, 8, 9, tzinfo=timezone.utc)
NEW_TICKET = 588709831
RNG = np.random.default_rng(20260814)
N_BOOT = 10000


def bootstrap_expectancy_ci(r_values, n_boot=N_BOOT, ci=90):
    """Percentile bootstrap CI on mean R (expectancy). Returns (lo, hi, pct_draws_positive)."""
    r_values = np.asarray(r_values, dtype=float)
    r_values = r_values[~np.isnan(r_values)]
    if len(r_values) < 3:
        return None, None, None
    boots = np.array([RNG.choice(r_values, size=len(r_values), replace=True).mean()
                       for _ in range(n_boot)])
    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    lo, hi = np.percentile(boots, [lo_q, hi_q])
    pct_positive = float((boots > 0).mean() * 100)
    return round(lo, 3), round(hi, 3), round(pct_positive, 1)


def min_sample_for_direction(hist_expectancy, hist_std_approx=1.0):
    """Rough sample size to detect a mean-R shift of the historical
    expectancy's own magnitude at ~80% power, two-sided alpha=0.05,
    using a normal approximation (n ~ 16 * sigma^2 / delta^2). This is
    an approximation for planning purposes only, not a formal power
    study on the actual live/historical R distributions."""
    if hist_expectancy is None or hist_expectancy == 0:
        return None
    delta = abs(hist_expectancy)
    n = 16 * (hist_std_approx ** 2) / (delta ** 2)
    return int(np.ceil(n))


def main():
    df = p27.load_export(expect_rows=72, expect_tickets=36, require_ticket=NEW_TICKET)
    closed = p27.build_closed(df)
    assert len(closed) == 36, f"expected 36 CLOSED, got {len(closed)}"
    print(f"[integrity] CLOSED trades: {len(closed)}")

    current_six = closed[closed['is_current_six']].copy()
    print(f"[integrity] current-six trades (any date): {len(current_six)}")

    windows = {
        'entire_history': current_six,
        'pre_demotion': current_six[current_six['entry_time_dt'] < DEMOTION_DATE],
        'post_demotion': current_six[current_six['entry_time_dt'] >= DEMOTION_DATE],
        'recent_aug9_13': current_six[current_six['entry_time_dt'] >= RECENT_START],
    }
    for name, w in windows.items():
        print(f"[window] {name}: {len(w)} trades")

    hist, summ = p27.load_historical()
    hist_by_strat = {row['strategy']: row for _, row in summ.iterrows()}

    strategies = ['GBPJPY_AMR', 'EURJPY_AMR', 'AUDJPY_AMR', 'CADJPY_AMR', 'CADJPY_ARB', 'GBPUSD_MONDAY']

    baseline_rows = []
    for strat in strategies:
        for wname, w in windows.items():
            sub = w[w['strategy_norm'] == strat]
            m = p27.account_metrics(sub)
            m['strategy'] = strat
            m['window'] = wname
            open_ct = len(closed[(closed['strategy_norm'] == strat)]) if False else None
            baseline_rows.append(m)

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_df.to_csv(OUT / 'live_validation_baseline.csv', index=False)
    print("\n=== baseline by strategy x window ===")
    print(baseline_df[['strategy', 'window', 'trades', 'wins', 'losses', 'win_rate_pct',
                        'profit_factor', 'expectancy_R', 'total_R', 'max_losing_streak',
                        'total_pnl']].to_string())

    # ---- directional + bootstrap CI, post_demotion window ----
    dir_rows = []
    boot_rows = []
    for strat in strategies:
        sub = windows['post_demotion'][windows['post_demotion']['strategy_norm'] == strat]
        for direction in ['BUY', 'SELL']:
            dsub = sub[sub['direction'] == direction]
            if len(dsub) == 0:
                continue
            m = p27.account_metrics(dsub)
            m['strategy'] = strat
            m['direction'] = direction
            dir_rows.append(m)

        lo, hi, pct_pos = bootstrap_expectancy_ci(sub['R'].values)
        hist_row = hist_by_strat.get(strat, {})
        hist_exp = hist_row.get('historical_expectancy')
        hist_exp = float(hist_exp) if hist_exp not in (None, '') else None
        min_n = min_sample_for_direction(hist_exp) if hist_exp else None
        boot_rows.append({
            'strategy': strat, 'live_n': len(sub),
            'live_expectancy_R': round(sub['R'].mean(), 3) if len(sub) else None,
            'boot_ci90_lo': lo, 'boot_ci90_hi': hi, 'boot_pct_draws_positive': pct_pos,
            'historical_expectancy_R': hist_exp,
            'approx_min_n_for_detection': min_n,
        })
    dir_df = pd.DataFrame(dir_rows)
    boot_df = pd.DataFrame(boot_rows)
    print("\n=== directional, post_demotion ===")
    print(dir_df[['strategy', 'direction', 'trades', 'wins', 'losses', 'win_rate_pct',
                  'profit_factor', 'total_R', 'expectancy_R']].to_string())
    print("\n=== bootstrap expectancy CI, post_demotion (90% CI, 10k resamples) ===")
    print(boot_df.to_string())

    # ---- regime (ATR tercile), post_demotion, using live-sample terciles ----
    pd_win = windows['post_demotion']
    live_atr = pd_win['ATR'].dropna()
    regime_rows = []
    if len(live_atr) >= 6:
        q1, q2 = live_atr.quantile([1/3, 2/3])
        def regime_of(a):
            if pd.isna(a):
                return p27.NA_STR
            return 'LOW' if a <= q1 else ('NORMAL' if a <= q2 else 'HIGH')
        pd_win = pd_win.copy()
        pd_win['regime'] = pd_win['ATR'].apply(regime_of)
        for (strat, regime), sub in pd_win.groupby(['strategy_norm', 'regime']):
            regime_rows.append({'strategy': strat, 'regime': regime, 'trades': len(sub),
                                 'win_rate_pct': round((sub['profit'] > 0).mean() * 100, 1),
                                 'avg_R': round(sub['R'].mean(), 3), 'total_R': round(sub['R'].sum(), 3)})
    regime_df = pd.DataFrame(regime_rows)
    print("\n=== regime, post_demotion ===")
    print(regime_df.to_string())

    # ---- execution: PRE_FIX vs POST_FIX, post_demotion ----
    exec_rows = []
    for status in ['PRE_FIX', 'POST_FIX']:
        sub = pd_win[pd_win['entry_fix_status'] == status]
        exec_rows.append({
            'entry_fix_status': status, 'trades': len(sub),
            'win_rate_pct': round((sub['profit'] > 0).mean() * 100, 1) if len(sub) else p27.NA_STR,
            'avg_R': round(sub['R'].mean(), 3) if len(sub) else p27.NA_STR,
            'avg_spread_pips': round(sub['spread'].mean(), 2) if sub['spread'].notna().any() else p27.NA_STR,
            'avg_spread_over_sl_pct': round(sub['spread_over_sl_pct'].mean(), 1) if sub['spread_over_sl_pct'].notna().any() else p27.NA_STR,
        })
    exec_df = pd.DataFrame(exec_rows)
    print("\n=== execution quality by fix status, post_demotion ===")
    print(exec_df.to_string())

    # ---- portfolio-level: JPY concentration, correlation, factor concentration ----
    jpy_trades = pd_win[pd_win['strategy_norm'].isin(p27.JPY_STRATEGIES)]
    pct_trades_jpy = len(jpy_trades) / max(len(pd_win), 1) * 100
    pct_risk_jpy = jpy_trades['initial_risk'].sum() / max(pd_win['initial_risk'].sum(), 1e-9) * 100

    by_day = pd_win.copy()
    by_day['trade_date'] = by_day['entry_time_dt'].dt.date
    daily = by_day.groupby('trade_date').agg(
        trades=('trade_id', 'count'),
        strategies_active=('strategy_norm', 'nunique'),
        strategies_losing=('is_loss', lambda x: (by_day.loc[x.index].groupby('strategy_norm')['is_loss'].any()).sum()),
        total_R=('R', 'sum'))
    multi_strat_days = int((daily['strategies_active'] >= 2).sum())
    multi_strat_losing_days = int((daily['strategies_losing'] >= 2).sum())

    portfolio_metrics = p27.account_metrics(pd_win)

    print(f"\n=== portfolio concentration, post_demotion ===")
    print(f"pct_trades_jpy={pct_trades_jpy:.1f}  pct_risk_jpy={pct_risk_jpy:.1f}  "
          f"multi_strategy_days={multi_strat_days}/{len(daily)}  "
          f"multi_strategy_losing_days={multi_strat_losing_days}")
    print(f"portfolio total_R={portfolio_metrics['total_R']}  PF={portfolio_metrics['profit_factor']}  "
          f"max_streak={portfolio_metrics['max_losing_streak']}  max_dd_R={portfolio_metrics['max_drawdown_R']}")

    # ---- write CSVs ----
    dir_df.to_csv(OUT / '_scratch_dir.csv', index=False)
    boot_df.to_csv(OUT / '_scratch_boot.csv', index=False)
    regime_df.to_csv(OUT / '_scratch_regime.csv', index=False)
    exec_df.to_csv(OUT / '_scratch_exec.csv', index=False)
    daily.to_csv(OUT / '_scratch_daily.csv')

    summary = {
        'windows_n': {k: len(v) for k, v in windows.items()},
        'portfolio_post_demotion': portfolio_metrics,
        'pct_trades_jpy': round(pct_trades_jpy, 1), 'pct_risk_jpy': round(pct_risk_jpy, 1),
        'multi_strategy_days': multi_strat_days, 'multi_strategy_losing_days': multi_strat_losing_days,
        'jpy_active_days_total': len(daily),
    }
    with open(OUT / '_phase29_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n" + json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
