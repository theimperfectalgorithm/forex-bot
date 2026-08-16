"""
Phase 31 -- Portfolio Factor & Regime Map.

DIAGNOSTIC ONLY. No strategy/parameter/risk/config modification.

Primary data source: data/phase26_all_trades.csv (2,712 trades, the
frozen-parameter historical reconstruction of the current 6-strategy book,
EXP-105..111) -- carries session/dow/hold_hours/vol_tercile/trend_tercile/
r_multiple/dir per trade, which is exactly the metadata this phase needs and
is REUSED, not re-derived, since it was already validated in prior phases.
Cross-checked against reports/5ers_trade_export.csv (live production, 72
rows/36 tickets) for the live-specific views. Both sources pass
src/research_data_validator.py before any analysis proceeds (STOP-on-failure
enforced, not skipped).

Regime definitions (vol_tercile, trend_tercile) are REUSED from the
project's existing phase20/21 regime methodology already baked into
phase26_all_trades.csv -- no new regime model is invented here, per
explicit instruction.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from research_data_validator import (  # noqa: E402
    ResearchDataError, ValidationReport, validate_column_count_consistency,
    validate_required_columns, validate_row_count, validate_lifecycle_pairing,
)

REPO = Path(__file__).parent.parent
OUT = REPO / 'reports'
HIST_CSV = REPO / 'data' / 'phase26_all_trades.csv'
LIVE_CSV = REPO / 'reports' / '5ers_trade_export.csv'

CURRENT_SIX = ['GBPJPY_AMR', 'EURJPY_AMR', 'AUDJPY_AMR', 'CADJPY_AMR', 'CADJPY_ARB', 'GBPUSD_MONDAY']

# strategy -> (instrument, base_ccy, quote_ccy, family, config_session)
STRATEGY_META = {
    'GBPJPY_AMR':    ('GBPJPY', 'GBP', 'JPY', 'mean_reversion (asian_hours_reversion)', 'Asian 00:00-07:00 server'),
    'EURJPY_AMR':    ('EURJPY', 'EUR', 'JPY', 'mean_reversion (asian_hours_reversion)', 'Asian 00:00-07:00 server'),
    'AUDJPY_AMR':    ('AUDJPY', 'AUD', 'JPY', 'mean_reversion (asian_hours_reversion)', 'Asian 00:00-07:00 server'),
    'CADJPY_AMR':    ('CADJPY', 'CAD', 'JPY', 'mean_reversion (asian_hours_reversion)', 'Asian 00:00-07:00 server'),
    'CADJPY_ARB':    ('CADJPY', 'CAD', 'JPY', 'asian_range_breakout', 'breakout 07:00-09:00 server'),
    'GBPUSD_MONDAY': ('GBPUSD', 'GBP', 'USD', 'calendar_drift (monday_drift)', 'Monday 00:00->21:00 server (long-only)'),
}
RISK_PCT = {
    'GBPJPY_AMR': 0.0025, 'EURJPY_AMR': 0.0025, 'AUDJPY_AMR': 0.0025,
    'CADJPY_AMR': 0.0025, 'CADJPY_ARB': 0.005, 'GBPUSD_MONDAY': 0.0025,
}


def validate_inputs():
    r1 = ValidationReport(path=str(HIST_CSV))
    validate_column_count_consistency(HIST_CSV, r1)
    validate_required_columns(HIST_CSV, {'entry_time', 'exit_time', 'dir', 'strategy', 'r_multiple',
                                          'session', 'dow', 'hold_hours', 'vol_tercile', 'trend_tercile'}, r1)
    validate_row_count(HIST_CSV, min_rows=100, report=r1)
    print(f"[validate] {r1.summary()}")

    r2 = ValidationReport(path=str(LIVE_CSV))
    validate_column_count_consistency(LIVE_CSV, r2)
    validate_lifecycle_pairing(LIVE_CSV, 'trade_id', 'status', report=r2)
    print(f"[validate] {r2.summary()}")


def load_hist():
    df = pd.read_csv(HIST_CSV, parse_dates=['entry_time', 'exit_time'])
    df = df[df['strategy'].isin(CURRENT_SIX)].copy()
    df['instrument'] = df['strategy'].map(lambda s: STRATEGY_META[s][0])
    df['base_ccy'] = df['strategy'].map(lambda s: STRATEGY_META[s][1])
    df['quote_ccy'] = df['strategy'].map(lambda s: STRATEGY_META[s][2])
    df['family'] = df['strategy'].map(lambda s: STRATEGY_META[s][3])
    df['risk_weight'] = df['strategy'].map(RISK_PCT)
    df['is_win'] = df['r_multiple'] > 0
    df['is_loss'] = df['r_multiple'] < 0
    # directional factor exposure: BUY = long base / short quote; SELL = short base / long quote
    df['long_ccy'] = np.where(df['dir'] == 'BUY', df['base_ccy'], df['quote_ccy'])
    df['short_ccy'] = np.where(df['dir'] == 'BUY', df['quote_ccy'], df['base_ccy'])
    return df


def load_live():
    df = pd.read_csv(LIVE_CSV, dtype=str)
    closed = df[df['status'] == 'CLOSED'].copy()
    for c in ['entry_price', 'exit_price', 'R', 'profit', 'holding_time', 'spread', 'ATR']:
        closed[c] = pd.to_numeric(closed[c], errors='coerce')
    closed['entry_time_dt'] = pd.to_datetime(closed['entry_time'], errors='coerce', utc=True)
    def norm(s):
        return 'GBPUSD_MONDAY' if s in ('GBPUSD_MON', 'GBPUSD_MONDAY') else s
    closed['strategy_norm'] = closed['strategy'].apply(norm)
    return closed[closed['strategy_norm'].isin(CURRENT_SIX)]


def account_metrics(sub):
    n = len(sub)
    if n == 0:
        return {'trades': 0, 'win_rate_pct': None, 'pf': None, 'expectancy_R': None, 'total_R': None, 'max_streak': None}
    r = sub['r_multiple'] if 'r_multiple' in sub.columns else sub['R']
    wins = r[r > 0]
    losses = r[r < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else np.nan
    ordered = sub.sort_values('entry_time')
    ro = ordered['r_multiple'] if 'r_multiple' in ordered.columns else ordered['R']
    streak = maxs = 0
    for v in ro:
        if v < 0:
            streak += 1
            maxs = max(maxs, streak)
        else:
            streak = 0
    return {'trades': n, 'win_rate_pct': round((r > 0).mean() * 100, 1),
            'pf': round(pf, 3) if pf == pf else None, 'expectancy_R': round(r.mean(), 3),
            'total_R': round(r.sum(), 3), 'max_streak': maxs}


def main():
    validate_inputs()
    hist = load_hist()
    live = load_live()
    print(f"[data] historical current-six trades: {len(hist)}; live current-six CLOSED trades: {len(live)}")

    # ---- PART 3: currency factor map ----
    rows = []
    for strat, (instr, base, quote, family, sess) in STRATEGY_META.items():
        for direction in ['BUY', 'SELL']:
            hsub = hist[(hist['strategy'] == strat) & (hist['dir'] == direction)]
            lsub = live[(live['strategy_norm'] == strat) & (live['direction'] == direction)]
            if len(hsub) == 0 and len(lsub) == 0:
                continue
            long_ccy = base if direction == 'BUY' else quote
            short_ccy = quote if direction == 'BUY' else base
            rows.append({
                'strategy': strat, 'instrument': instr, 'direction': direction,
                'base_currency': base, 'quote_currency': quote,
                'currency_exposure': f'long {long_ccy} / short {short_ccy}',
                'trade_count': len(hsub), 'risk_weight': RISK_PCT[strat],
                'historical_R': round(hsub['r_multiple'].sum(), 2) if len(hsub) else None,
                'live_R': round(lsub['R'].sum(), 2) if len(lsub) else 0.0,
                'notes': '',
            })
    factor_map = pd.DataFrame(rows)
    factor_map.to_csv(OUT / 'portfolio_currency_factor_map.csv', index=False)
    print("\n=== currency factor map ===")
    print(factor_map.to_string())

    # ---- PART 4: strategy family map ----
    fam_rows = []
    for strat, (instr, base, quote, family, sess) in STRATEGY_META.items():
        hsub = hist[hist['strategy'] == strat]
        m = account_metrics(hsub)
        fam_rows.append({'strategy': strat, 'instrument': instr, 'family': family,
                          'session': sess, 'jpy_exposure': quote == 'JPY' or base == 'JPY',
                          **m})
    family_map = pd.DataFrame(fam_rows)
    family_map.to_csv(OUT / 'portfolio_strategy_family_map.csv', index=False)
    print("\n=== strategy family map ===")
    print(family_map.to_string())

    # ---- PART 5: session exposure ----
    sess_rows = []
    for (strat, sess), sub in hist.groupby(['strategy', 'session']):
        sess_rows.append({'strategy': strat, 'session': sess, 'trades': len(sub),
                           'pct_of_strategy_trades': round(len(sub) / len(hist[hist.strategy == strat]) * 100, 1),
                           'risk_weight': RISK_PCT[strat], 'avg_hold_hours': round(sub['hold_hours'].mean(), 2)})
    session_df = pd.DataFrame(sess_rows)
    session_df.to_csv(OUT / 'portfolio_session_exposure.csv', index=False)
    print("\n=== session exposure ===")
    print(session_df.to_string())

    # ---- PART 6: trade overlap (historical, using entry/exit intervals) ----
    hist_sorted = hist.sort_values('entry_time').reset_index(drop=True)
    overlap_counts = []
    for i, row in hist_sorted.iterrows():
        others = hist_sorted[(hist_sorted['entry_time'] < row['exit_time']) &
                              (hist_sorted['exit_time'] > row['entry_time']) &
                              (hist_sorted.index != i)]
        overlap_counts.append(others['strategy'].nunique())
    hist_sorted['n_other_strategies_open'] = overlap_counts
    avg_overlap = hist_sorted['n_other_strategies_open'].mean()
    max_overlap = hist_sorted['n_other_strategies_open'].max()

    hist_sorted['trade_date'] = hist_sorted['entry_time'].dt.date
    daily = hist_sorted.groupby('trade_date').agg(
        strategies_active=('strategy', 'nunique'),
        trades=('strategy', 'count'),
        losses=('is_loss', 'sum'),
        total_R=('r_multiple', 'sum'))
    multi_entry_days = int((daily['strategies_active'] >= 2).sum())
    days_2plus_losses = int((daily['losses'] >= 2).sum())
    days_3plus_losses = int((daily['losses'] >= 3).sum())

    overlap_summary = pd.DataFrame([{
        'avg_simultaneous_strategies': round(avg_overlap, 2), 'max_simultaneous_strategies': int(max_overlap),
        'total_days': len(daily), 'multi_strategy_entry_days': multi_entry_days,
        'days_2plus_losses': days_2plus_losses, 'days_3plus_losses': days_3plus_losses,
        'pct_days_2plus_losses': round(days_2plus_losses / len(daily) * 100, 2),
        'pct_days_3plus_losses': round(days_3plus_losses / len(daily) * 100, 2),
    }])
    overlap_summary.to_csv(OUT / 'portfolio_trade_overlap.csv', index=False)
    print("\n=== trade overlap summary ===")
    print(overlap_summary.to_string())

    daily.to_csv(OUT / '_scratch_daily_hist.csv')

    # ---- PART 7: return correlation ----
    daily_by_strat = hist_sorted.groupby(['trade_date', 'strategy'])['r_multiple'].sum().unstack('strategy')
    corr_pearson = daily_by_strat.corr(method='pearson')
    corr_spearman = daily_by_strat.corr(method='spearman')
    corr_pearson.to_csv(OUT / '_scratch_corr_pearson.csv')
    corr_spearman.to_csv(OUT / '_scratch_corr_spearman.csv')
    print("\n=== daily R Pearson correlation (missing-strategy-days = NaN, NOT zero-filled) ===")
    print(corr_pearson.round(2).to_string())

    weekly_by_strat = hist_sorted.set_index('entry_time').groupby('strategy')['r_multiple'].resample('W').sum().unstack('strategy')
    corr_weekly = weekly_by_strat.corr(method='pearson')

    # trade-level correlation is not meaningful (different strategies rarely
    # share an exact trade timestamp) -- explicitly documented as such,
    # daily/weekly used instead per the instruction's own suggested views
    trade_level_note = "NOT MEANINGFUL -- trades across strategies almost never share exact timestamps; daily/weekly aggregation used instead (see B methodology note in the master report)"

    corr_rows = []
    for view, mat in [('daily_pearson', corr_pearson), ('daily_spearman', corr_spearman), ('weekly_pearson', corr_weekly)]:
        for s1 in CURRENT_SIX:
            for s2 in CURRENT_SIX:
                if s1 >= s2 or s1 not in mat.columns or s2 not in mat.columns:
                    continue
                corr_rows.append({'view': view, 'strategy_1': s1, 'strategy_2': s2,
                                   'correlation': round(mat.loc[s1, s2], 3) if pd.notna(mat.loc[s1, s2]) else None})
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT / 'portfolio_return_correlation.csv', index=False)
    print("\n=== correlation summary (daily pearson) ===")
    print(corr_df[corr_df.view == 'daily_pearson'].sort_values('correlation', ascending=False).to_string())

    # ---- PART 8: drawdown correlation ----
    portfolio_daily_R = daily['total_R']
    cum = portfolio_daily_R.cumsum()
    running_max = cum.cummax()
    dd = cum - running_max
    threshold = dd.quantile(0.10)  # worst 10% of days by drawdown depth
    dd_days = dd[dd <= threshold].index
    normal_days = dd[dd > threshold].index

    dd_period_daily = daily_by_strat.loc[daily_by_strat.index.isin(dd_days)]
    normal_period_daily = daily_by_strat.loc[daily_by_strat.index.isin(normal_days)]
    corr_dd = dd_period_daily.corr(method='pearson')
    corr_normal = normal_period_daily.corr(method='pearson')

    dd_rows = []
    for s1 in CURRENT_SIX:
        for s2 in CURRENT_SIX:
            if s1 >= s2 or s1 not in corr_dd.columns or s2 not in corr_dd.columns:
                continue
            dd_rows.append({
                'strategy_1': s1, 'strategy_2': s2,
                'correlation_drawdown_days': round(corr_dd.loc[s1, s2], 3) if pd.notna(corr_dd.loc[s1, s2]) else None,
                'correlation_normal_days': round(corr_normal.loc[s1, s2], 3) if pd.notna(corr_normal.loc[s1, s2]) else None,
                'n_drawdown_days': int(dd_period_daily[[s1, s2]].dropna(how='all').shape[0]),
            })
    dd_corr_df = pd.DataFrame(dd_rows)
    dd_corr_df.to_csv(OUT / 'portfolio_drawdown_factor_analysis.csv', index=False)
    print(f"\n=== drawdown-day correlation (worst-decile days by portfolio DD, n={len(dd_days)} of {len(daily)}) vs normal days ===")
    print(dd_corr_df.to_string())

    # ---- PART 9/10: regime matrix (reusing existing vol_tercile/trend_tercile) ----
    regime_rows = []
    for strat in CURRENT_SIX:
        sub = hist[hist.strategy == strat]
        for regime_col, regime_label in [('vol_tercile', 'volatility'), ('trend_tercile', 'trend')]:
            for regime_val, rsub in sub.groupby(regime_col):
                if pd.isna(regime_val):
                    continue
                m = account_metrics(rsub)
                regime_rows.append({'strategy': strat, 'regime_type': regime_label, 'regime_value': regime_val, **m})
    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(OUT / 'portfolio_regime_matrix.csv', index=False)
    print("\n=== regime matrix (volatility) ===")
    print(regime_df[regime_df.regime_type == 'volatility'].to_string())

    # regime coincidence: which regimes have multiple strategies net-negative
    coincidence_rows = []
    for regime_col, regime_label in [('vol_tercile', 'volatility'), ('trend_tercile', 'trend')]:
        for regime_val in hist[regime_col].dropna().unique():
            n_negative = 0
            combined_R = 0
            for strat in CURRENT_SIX:
                sub = hist[(hist.strategy == strat) & (hist[regime_col] == regime_val)]
                if len(sub) == 0:
                    continue
                total_r = sub['r_multiple'].sum()
                combined_R += total_r
                if total_r < 0:
                    n_negative += 1
            coincidence_rows.append({'regime_type': regime_label, 'regime_value': regime_val,
                                      'strategies_net_negative': n_negative, 'combined_R': round(combined_R, 2)})
    coincidence_df = pd.DataFrame(coincidence_rows)
    print("\n=== regime coincidence (how many of 6 strategies net-negative) ===")
    print(coincidence_df.to_string())

    # ---- PART 12: directional factor map ----
    dir_rows = []
    for strat in CURRENT_SIX:
        for direction in ['BUY', 'SELL']:
            sub = hist[(hist.strategy == strat) & (hist.dir == direction)]
            if len(sub) == 0:
                continue
            m = account_metrics(sub)
            dir_rows.append({'strategy': strat, 'direction': direction, **m})
    dir_df = pd.DataFrame(dir_rows)
    print("\n=== directional factor map ===")
    print(dir_df.to_string())

    # ---- PART 13: holding period buckets ----
    def bucket(h):
        if h < 2:
            return '<2h'
        if h < 6:
            return '2-6h'
        if h < 12:
            return '6-12h'
        if h < 24:
            return '12-24h'
        return '>24h'
    hist['hold_bucket'] = hist['hold_hours'].apply(bucket)
    hold_rows = []
    for (strat, b), sub in hist.groupby(['strategy', 'hold_bucket']):
        hold_rows.append({'strategy': strat, 'hold_bucket': b, 'trades': len(sub),
                           'total_R': round(sub['r_multiple'].sum(), 2),
                           'win_rate_pct': round((sub.r_multiple > 0).mean() * 100, 1)})
    hold_df = pd.DataFrame(hold_rows)
    print("\n=== holding period buckets ===")
    print(hold_df.to_string())

    # ---- PART 14: risk concentration decomposition ----
    total_risk_weighted_trades = sum(len(hist[hist.strategy == s]) * RISK_PCT[s] for s in CURRENT_SIX)
    risk_by_strategy = {s: round(len(hist[hist.strategy == s]) * RISK_PCT[s] / total_risk_weighted_trades * 100, 1) for s in CURRENT_SIX}
    risk_by_currency = {}
    for s in CURRENT_SIX:
        base, quote = STRATEGY_META[s][1], STRATEGY_META[s][2]
        w = len(hist[hist.strategy == s]) * RISK_PCT[s]
        for ccy in {base, quote}:
            risk_by_currency[ccy] = risk_by_currency.get(ccy, 0) + w
    risk_by_currency = {k: round(v / total_risk_weighted_trades * 100, 1) for k, v in risk_by_currency.items()}
    risk_by_family = {}
    for s in CURRENT_SIX:
        fam = STRATEGY_META[s][3].split(' (')[0]
        w = len(hist[hist.strategy == s]) * RISK_PCT[s]
        risk_by_family[fam] = risk_by_family.get(fam, 0) + w
    risk_by_family = {k: round(v / total_risk_weighted_trades * 100, 1) for k, v in risk_by_family.items()}

    print("\n=== risk concentration (trade-count x risk-weight proxy, NOT true $ risk since MAX_LOT caps aren't modeled here) ===")
    print("by strategy:", risk_by_strategy)
    print("by currency (note: a trade counts toward BOTH its base and quote currency, so this does not sum to 100%):", risk_by_currency)
    print("by family:", risk_by_family)

    # ---- PART 15: effective diversification (HHI-based, mathematically justified) ----
    weights = np.array(list(risk_by_strategy.values())) / 100
    hhi = (weights ** 2).sum()
    effective_n_strategy = 1 / hhi if hhi > 0 else None

    corr_vals = corr_pearson.values[np.triu_indices_from(corr_pearson.values, k=1)]
    corr_vals = corr_vals[~np.isnan(corr_vals)]
    avg_pairwise_corr = float(np.mean(corr_vals)) if len(corr_vals) else None

    print(f"\n=== effective diversification ===")
    print(f"Effective N (strategy risk-weight HHI, 1/sum(w_i^2)): {effective_n_strategy:.2f} of 6 nominal strategies")
    print(f"Average pairwise daily-R correlation across the 15 strategy pairs: {avg_pairwise_corr:.3f}" if avg_pairwise_corr else "N/A")

    summary = {
        'avg_overlap': round(avg_overlap, 2), 'max_overlap': int(max_overlap),
        'multi_strategy_entry_days': multi_entry_days, 'total_days': len(daily),
        'days_2plus_losses': days_2plus_losses, 'days_3plus_losses': days_3plus_losses,
        'risk_by_strategy': risk_by_strategy, 'risk_by_currency': risk_by_currency, 'risk_by_family': risk_by_family,
        'effective_n_strategy': round(effective_n_strategy, 2) if effective_n_strategy else None,
        'avg_pairwise_corr': round(avg_pairwise_corr, 3) if avg_pairwise_corr else None,
        'regime_coincidence': coincidence_df.to_dict(orient='records'),
    }
    with open(OUT / '_phase31_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # persist a few more scratch tables for report-writing
    dir_df.to_csv(OUT / '_scratch_directional.csv', index=False)
    hold_df.to_csv(OUT / '_scratch_holding.csv', index=False)
    coincidence_df.to_csv(OUT / '_scratch_coincidence.csv', index=False)

    print("\n" + json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
