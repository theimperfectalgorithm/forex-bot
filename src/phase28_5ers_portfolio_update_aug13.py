"""
Phase 28 -- Evidence UPDATE to the phase-27 current-5ers-portfolio forensic
investigation, using the fresh production export (72 rows / 36 tickets,
latest exit 2026-08-13T19:12:09Z, includes the new CADJPY ARB SELL loss
ticket 588709831).

DIAGNOSTIC ONLY. No strategy modification, no optimization, no deployment.
Reuses phase27's helper functions (import, not copy) for identical
methodology; adds:
  - Period A (repro of the phase-27 snapshot) / B (full fresh data) / C
    (2026-08-09 onward, the recent deterioration window) splits
  - PRE_FIX / POST_FIX entry-price classification per
    reports/entry_price_logging_audit.md, used to scope execution-quality
    claims to trustworthy data only
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import phase27_5ers_current_portfolio_forensic as p27

REPO = Path(__file__).parent.parent
OUT = REPO / 'reports'

# Previous snapshot cutoff, stated as "~07:00 UTC" -- the actual last trade in
# that snapshot exited at 07:00:05 UTC (5 seconds past a bare 07:00:00 cutoff),
# so the boundary is set a full minute later to unambiguously include it while
# still excluding the new ticket's 19:12 UTC exit.
PERIOD_A_END = datetime(2026, 8, 13, 7, 1, 0, tzinfo=timezone.utc)
PERIOD_B_START = datetime(2026, 7, 31, 0, 0, 0, tzinfo=timezone.utc)  # demotion date
PERIOD_C_START = datetime(2026, 8, 9, 0, 0, 0, tzinfo=timezone.utc)   # recent-deterioration window

NEW_TICKET = 588709831

# The previous report's reproduced numbers (population D: post-demotion,
# current-six, from the 70-row/35-ticket snapshot) -- used as the Period-A
# reproduction check target, hardcoded from reports/5ers_current_portfolio_forensic_analysis.md
PREVIOUS_METRICS_D = {
    'trades': 32, 'wins': 12, 'losses': 20, 'win_rate_pct': 37.5,
    'total_pnl': -80.20, 'total_R': -6.43, 'expectancy_R': -0.201,
    'profit_factor': 0.513, 'max_losing_streak': 9, 'max_drawdown_R': -8.71,
}


def main():
    df = p27.load_export(expect_rows=72, expect_tickets=36, require_ticket=NEW_TICKET)
    closed = p27.build_closed(df)
    print(f"\n[integrity] CLOSED trades: {len(closed)} (expected 36)")
    assert len(closed) == 36, f"expected 36 CLOSED trades, got {len(closed)}"

    r_mismatch = p27.r_recompute_check(closed)
    print(f"[integrity] R-recompute mismatches: {len(r_mismatch)}")

    fix_counts = closed['entry_fix_status'].value_counts().to_dict()
    print(f"[execution] entry_fix_status counts: {fix_counts}")
    new_trade_row = closed[closed['trade_id'] == str(NEW_TICKET)]
    print(f"[execution] new ticket {NEW_TICKET} entry_fix_status: "
          f"{new_trade_row['entry_fix_status'].iloc[0] if len(new_trade_row) else 'NOT FOUND'}")

    # ---- populations ----
    current_six_post_demo = closed[
        closed['is_current_six'] &
        (closed['post_demotion'] | closed['demotion_status'].str.contains('N/A'))
    ].copy()

    # -- Reproduction population: EXACTLY the previous report's population D
    # definition (current-six, non-PRE_DEMOTION-labeled, no additional date
    # floor), restricted to trades already CLOSED by the previous snapshot's
    # cutoff (by exit_time, since the new ticket entered before 07:00 UTC on
    # 08-13 but didn't close until 19:12 -- it was still OPEN at snapshot time).
    repro_population = current_six_post_demo[current_six_post_demo['exit_time_dt'] < PERIOD_A_END]
    metrics_repro = p27.account_metrics(repro_population)

    print("\n=== REPRODUCTION CHECK (previous report's population D definition) ===")
    print(json.dumps(metrics_repro, indent=2, default=str))
    print("\n=== vs PREVIOUS REPORT ===")
    mismatches = []
    for k, v in PREVIOUS_METRICS_D.items():
        got = metrics_repro.get(k)
        ok = (got == v) if not isinstance(v, float) else (isinstance(got, (int, float)) and abs(got - v) < 0.02)
        print(f"  {k}: previous={v} reproduced={got} {'OK' if ok else 'MISMATCH'}")
        if not ok:
            mismatches.append(k)
    print(f"\n[repro] {'ALL MATCH -- proceeding' if not mismatches else 'MISMATCHES: ' + str(mismatches)}")

    # IMPORTANT CORRECTION FOUND: the previous report's "current six" population
    # was never actually date-floored at the 2026-07-31 demotion -- only
    # GBPJPY_ARB/XAUUSD_ARB carry a PRE_DEMOTION label (they're the only
    # entries in DEMOTED_STRATEGIES); the current six's own trades from BEFORE
    # 07-31 (predating the risk_scale 1.0->0.5 cut) were silently included.
    pre_0731_current_six = current_six_post_demo[current_six_post_demo['entry_time_dt'] < PERIOD_B_START]
    print(f"\n[correction] {len(pre_0731_current_six)} current-six trades predate the 2026-07-31 demotion/"
          f"risk_scale cut and were included in the previous report's population without a date floor:")
    print(pre_0731_current_six[['trade_id', 'strategy_norm', 'entry_time']].to_string())

    # -- User-specified periods A/B/C: current-six trades with entry_time on
    # or after the 2026-07-31 demotion (the risk_scale-0.5, 6-slot regime),
    # per the literal instruction. This is the corrected, properly-scoped set.
    strict_current_six = current_six_post_demo[current_six_post_demo['entry_time_dt'] >= PERIOD_B_START]
    period_A = strict_current_six[strict_current_six['exit_time_dt'] < PERIOD_A_END]
    period_B = strict_current_six  # entry_time already >= 07-31; no upper bound = latest
    period_C = strict_current_six[strict_current_six['entry_time_dt'] >= PERIOD_C_START]

    metrics_A = p27.account_metrics(period_A)
    metrics_B = p27.account_metrics(period_B)
    metrics_C = p27.account_metrics(period_C)

    print("\n=== PERIOD A (strict: entry>=07-31, closed before 08-13 07:00) ===")
    print(json.dumps(metrics_A, indent=2, default=str))
    print("\n=== PERIOD B (strict: entry>=07-31, latest data) ===")
    print(json.dumps(metrics_B, indent=2, default=str))
    print("\n=== PERIOD C (2026-08-09 onward) ===")
    print(json.dumps(metrics_C, indent=2, default=str))

    # ---- strategy breakdown, periods A/B/C ----
    strat_A = p27.strategy_breakdown(period_A).assign(period='A')
    strat_B = p27.strategy_breakdown(period_B).assign(period='B')
    strat_C = p27.strategy_breakdown(period_C).assign(period='C')
    strat_table = pd.concat([strat_A, strat_B, strat_C], ignore_index=True)
    print("\n=== strategy breakdown by period ===")
    print(strat_table[['period', 'strategy', 'trades', 'wins', 'losses', 'win_rate_pct',
                        'profit_factor', 'expectancy_R', 'total_R', 'max_losing_streak',
                        'total_pnl']].to_string())

    # ---- CADJPY ARB specific (new loss) ----
    cadjpy_arb_B = period_B[period_B['strategy_norm'] == 'CADJPY_ARB'].sort_values('entry_time_dt')
    print("\n=== CADJPY ARB, Period B, chronological ===")
    print(cadjpy_arb_B[['trade_id', 'entry_time', 'direction', 'profit', 'R', 'exit_reason']].to_string())

    # ---- directional breakdown, Period B ----
    dir_B = p27.directional_breakdown(period_B)
    print("\n=== directional breakdown, Period B ===")
    print(dir_B[['strategy', 'direction', 'trades', 'wins', 'losses', 'win_rate_pct',
                 'profit_factor', 'total_R', 'expectancy_R']].to_string())

    # ---- exit reason, Period B ----
    exit_rows = []
    for (strat, reason), sub in period_B.groupby(['strategy_norm', 'exit_reason']):
        exit_rows.append({
            'strategy': strat, 'exit_reason': reason, 'count': len(sub),
            'win_rate_pct': round((sub['profit'] > 0).mean() * 100, 1),
            'avg_R': round(sub['R'].mean(), 3),
            'avg_holding_hours': round(sub['holding_time'].mean(), 2) if sub['holding_time'].notna().any() else p27.NA_STR,
        })
    exit_table = pd.DataFrame(exit_rows)
    print("\n=== exit reason, Period B ===")
    print(exit_table.to_string())

    # ---- regime (ATR tercile), Period B ----
    live_atr = period_B['ATR'].dropna()
    regime_rows = []
    if len(live_atr) >= 6:
        q1, q2 = live_atr.quantile([1 / 3, 2 / 3])
        def regime_of(atr):
            if pd.isna(atr):
                return p27.NA_STR
            if atr <= q1:
                return 'LOW'
            if atr <= q2:
                return 'NORMAL'
            return 'HIGH'
        period_B = period_B.copy()
        period_B['regime'] = period_B['ATR'].apply(regime_of)
        for (strat, regime), sub in period_B.groupby(['strategy_norm', 'regime']):
            regime_rows.append({
                'strategy': strat, 'regime': regime, 'trades': len(sub),
                'win_rate_pct': round((sub['profit'] > 0).mean() * 100, 1),
                'avg_R': round(sub['R'].mean(), 3), 'total_R': round(sub['R'].sum(), 3),
            })
    regime_table = pd.DataFrame(regime_rows)
    print("\n=== regime (ATR tercile), Period B ===")
    print(regime_table.to_string())

    # ---- JPY concentration + daily clustering, Period B ----
    jpy_trades = period_B[period_B['strategy_norm'].isin(p27.JPY_STRATEGIES)]
    pct_trades_jpy = len(jpy_trades) / max(len(period_B), 1) * 100
    pct_risk_jpy = jpy_trades['initial_risk'].sum() / max(period_B['initial_risk'].sum(), 1e-9) * 100
    total_losing_r_all = period_B[period_B['is_loss']]['R'].sum()
    jpy_losing_r = jpy_trades[jpy_trades['is_loss']]['R'].sum()
    pct_losing_r_jpy = (jpy_losing_r / total_losing_r_all * 100) if total_losing_r_all != 0 else np.nan

    jpy_by_day = jpy_trades.copy()
    jpy_by_day['trade_date'] = jpy_by_day['entry_time_dt'].dt.date
    daily_jpy_strats = jpy_by_day.groupby('trade_date')['strategy_norm'].nunique()
    daily_jpy_losses = jpy_by_day[jpy_by_day['is_loss']].groupby('trade_date')['strategy_norm'].nunique()
    daily_jpy_r = jpy_by_day.groupby('trade_date')['R'].sum()

    portfolio_by_day = period_B.copy()
    portfolio_by_day['trade_date'] = portfolio_by_day['entry_time_dt'].dt.date
    daily_portfolio_r = portfolio_by_day.groupby('trade_date')['R'].sum()

    corr_rows = []
    for d, n_strats in daily_jpy_strats.items():
        corr_rows.append({
            'date': str(d), 'jpy_strategies_active': n_strats,
            'jpy_strategies_losing': daily_jpy_losses.get(d, 0),
            'jpy_total_R': round(daily_jpy_r.get(d, 0), 3),
            'portfolio_total_R': round(daily_portfolio_r.get(d, 0), 3),
        })
    corr_table = pd.DataFrame(corr_rows)
    multi_jpy_days = int((daily_jpy_strats >= 2).sum())
    multi_jpy_losing_days = int((daily_jpy_losses >= 2).sum())

    print(f"\n=== JPY concentration, Period B ===")
    print(f"pct_trades_jpy={pct_trades_jpy:.1f}  pct_risk_jpy={pct_risk_jpy:.1f}  "
          f"pct_losing_r_jpy={pct_losing_r_jpy:.1f}  multi_jpy_days={multi_jpy_days}/{len(daily_jpy_strats)}  "
          f"multi_jpy_losing_days={multi_jpy_losing_days}")
    print(corr_table.to_string())

    # ---- execution quality: POST_FIX-only entry-price-anchored checks ----
    postfix_B = period_B[period_B['entry_fix_status'] == 'POST_FIX']
    prefix_B = period_B[period_B['entry_fix_status'] == 'PRE_FIX']
    print(f"\n=== execution quality, Period B: PRE_FIX={len(prefix_B)} POST_FIX={len(postfix_B)} "
          f"UNKNOWN={len(period_B[period_B['entry_fix_status']=='UNKNOWN'])} ===")

    spread_buckets = pd.cut(period_B['spread_over_sl_pct'], bins=[0, 10, 20, 30, 40, np.inf],
                             labels=['<10%', '10-20%', '20-30%', '30-40%', '>40%'])
    period_B_sb = period_B.assign(spread_over_sl_bucket=spread_buckets)
    spread_bucket_perf = period_B_sb.groupby('spread_over_sl_bucket', observed=True).agg(
        trades=('trade_id', 'count'), win_rate=('is_win', lambda x: round(x.mean() * 100, 1)),
        avg_R=('R', lambda x: round(x.mean(), 3)))
    print(spread_bucket_perf.to_string())

    # ---- Monte Carlo, Period B population ----
    hist, summ = p27.load_historical()
    live_n_B = len(period_B)
    pooled_pf, pooled_wr, pooled_totalr, pooled_dd = p27.monte_carlo_pooled(hist, live_n_B)
    sa_pf, sa_wr, sa_totalr, sa_dd, sa_streak, sa_counts = p27.monte_carlo_strategy_aware(hist, period_B)

    live_pf = metrics_B['profit_factor'] if metrics_B['profit_factor'] != 'INF' else np.inf
    live_wr = metrics_B['win_rate_pct']
    live_streak = metrics_B['max_losing_streak']
    live_dd_r = metrics_B['max_drawdown_R']

    mc_rows = []
    for method, arrs in [('pooled', (pooled_pf, pooled_wr, pooled_dd, None)),
                          ('strategy_aware', (sa_pf, sa_wr, sa_dd, sa_streak))]:
        pf_arr, wr_arr, dd_arr, streak_arr = arrs
        for metric, arr, obs in [('PF', pf_arr, live_pf), ('win_rate', wr_arr, live_wr),
                                  ('max_drawdown_R', dd_arr, live_dd_r)]:
            mc_rows.append({
                'method': method, 'metric': metric,
                **{f'p{q}': np.nanpercentile(arr, q) for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]},
                'observed': obs, 'observed_percentile': p27.percentile_rank(obs, arr),
            })
        if streak_arr is not None:
            mc_rows.append({
                'method': method, 'metric': 'max_losing_streak',
                **{f'p{q}': np.nanpercentile(streak_arr, q) for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]},
                'observed': live_streak, 'observed_percentile': p27.percentile_rank(live_streak, streak_arr),
            })
    mc_table = pd.DataFrame(mc_rows)
    print("\n=== Monte Carlo, Period B ===")
    print(mc_table.to_string())

    # ---- Period C deep dive (2026-08-09 onward) ----
    period_C_sorted = period_C.sort_values('entry_time_dt')
    print(f"\n=== Period C (2026-08-09 onward) trade list ===")
    print(period_C_sorted[['trade_id', 'entry_time', 'strategy_norm', 'direction', 'profit', 'R',
                            'exit_reason', 'ATR', 'spread']].to_string())

    period_C_by_day = period_C.copy()
    period_C_by_day['trade_date'] = period_C_by_day['entry_time_dt'].dt.date
    print("\n=== Period C by day ===")
    print(period_C_by_day.groupby('trade_date').agg(
        trades=('trade_id', 'count'), losses=('is_loss', 'sum'), total_R=('R', 'sum'),
        strategies=('strategy_norm', lambda x: list(x))).to_string())

    # ---- write CSVs ----
    trade_cols = ['trade_id', 'strategy_norm', 'symbol', 'direction', 'entry_time', 'exit_time',
                  'holding_time', 'entry_price', 'exit_price', 'profit', 'R', 'exit_reason',
                  'spread', 'ATR', 'demotion_status', 'entry_fix_status']
    period_B.rename(columns={'strategy_norm': 'strategy'})[
        [c if c != 'strategy_norm' else 'strategy' for c in trade_cols]
    ].to_csv(OUT / '5ers_portfolio_update_aug13_trade_level.csv', index=False)

    strat_table.to_csv(OUT / '5ers_portfolio_update_aug13_strategy_by_period.csv', index=False)
    dir_B.to_csv(OUT / '5ers_portfolio_update_aug13_directional.csv', index=False)
    exit_table.to_csv(OUT / '5ers_portfolio_update_aug13_exit_reason.csv', index=False)
    regime_table.to_csv(OUT / '5ers_portfolio_update_aug13_regime.csv', index=False)
    corr_table.to_csv(OUT / '5ers_portfolio_update_aug13_jpy_correlation.csv', index=False)
    mc_table.to_csv(OUT / '5ers_portfolio_update_aug13_monte_carlo.csv', index=False)

    summary_blob = {
        'metrics_A_repro': metrics_A, 'metrics_B_full': metrics_B, 'metrics_C_recent': metrics_C,
        'repro_mismatches': mismatches, 'r_mismatch_count': len(r_mismatch),
        'entry_fix_status_counts': fix_counts,
        'new_ticket_fix_status': new_trade_row['entry_fix_status'].iloc[0] if len(new_trade_row) else None,
        'pct_trades_jpy': round(pct_trades_jpy, 1), 'pct_risk_jpy': round(pct_risk_jpy, 1),
        'pct_losing_r_jpy': round(pct_losing_r_jpy, 1) if pct_losing_r_jpy == pct_losing_r_jpy else p27.NA_STR,
        'multi_jpy_days': multi_jpy_days, 'multi_jpy_losing_days': multi_jpy_losing_days,
        'jpy_active_days_total': len(daily_jpy_strats),
        'date_range_B': [str(period_B['entry_time_dt'].min()), str(period_B['entry_time_dt'].max())],
    }
    with open(OUT / '_phase28_summary_blob.json', 'w') as f:
        json.dump(summary_blob, f, indent=2, default=str)
    print("\n" + json.dumps(summary_blob, indent=2, default=str))


if __name__ == '__main__':
    main()
