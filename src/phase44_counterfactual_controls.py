"""
Phase 44 -- portfolio-control counterfactual validation. HISTORICAL
COUNTERFACTUAL RESEARCH ONLY. No live change, no deployment, no
optimization. Exactly 5 frozen controls (A baseline + B/C/D/E),
thresholds taken from Phase42/43's already-published findings.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).parent.parent
OUT = REPO / 'reports'
RNG = np.random.default_rng(20261001)
MECH_RE = re.compile(r'_(AMR|ARB|MONDAY)$')


def load_control():
    df = pd.read_csv(REPO / 'data' / 'phase26_all_trades.csv')
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    df['trade_date'] = df['entry_time'].dt.date
    df['instrument'] = df['strategy'].apply(lambda s: s.replace('_AMR', '').replace('_ARB', '').replace('_MONDAY', ''))
    df['mechanism'] = df['strategy'].apply(lambda s: MECH_RE.search(s).group(1) if MECH_RE.search(s) else 'UNKNOWN')
    df['base_ccy'] = df['instrument'].str[:3]
    df['quote_ccy'] = df['instrument'].str[3:]
    df['is_jpy'] = (df['base_ccy'] == 'JPY') | (df['quote_ccy'] == 'JPY')
    return df.sort_values('entry_time').reset_index(drop=True)


def open_positions_at(df, ts, exclude_idx=None):
    mask = (df['entry_time'] <= ts) & (df['exit_time'] > ts)
    if exclude_idx is not None:
        mask &= (df.index != exclude_idx)
    return df[mask]


def build_daily_vol_ledger(df):
    dates = sorted(df['trade_date'].unique())
    rows = []
    for d in dates:
        day_trades = df[df['trade_date'] == d]
        vol_vals = day_trades['atr_pctile'].dropna()
        rows.append({'date': d, 'vol_level': round(vol_vals.mean(), 6) if len(vol_vals) else np.nan})
    ledger = pd.DataFrame(rows)
    valid = ledger.dropna(subset=['vol_level']).copy()
    valid['vol_pctile'] = valid['vol_level'].rank(pct=True) * 100
    p1, p2 = valid['vol_pctile'].quantile([1/3, 2/3])
    valid['vol_state'] = np.where(valid['vol_pctile'] > p2, 'HIGH', np.where(valid['vol_pctile'] > p1, 'NORMAL', 'LOW'))
    valid = valid.sort_values('date').reset_index(drop=True)
    valid['prev_vol_state'] = valid['vol_state'].shift(1)
    valid['is_high_to_normal'] = (valid['vol_state'] == 'NORMAL') & (valid['prev_vol_state'] == 'HIGH')
    return valid.set_index('date')


def suppress(df, vol_ledger, rule_fn, label):
    """rule_fn(row, prior_open_df, day_row) -> bool (True = suppress)"""
    keep_mask = []
    suppressed_rows = []
    for idx, row in df.iterrows():
        d = row['trade_date']
        day_row = vol_ledger.loc[d] if d in vol_ledger.index else None
        prior = open_positions_at(df, row['entry_time'], exclude_idx=idx)
        do_suppress = rule_fn(row, prior, day_row, idx)
        keep_mask.append(not do_suppress)
        if do_suppress:
            suppressed_rows.append({
                'control': label, 'strategy': row['strategy'], 'symbol': row['instrument'],
                'direction': row['dir'], 'original_R': row['r_multiple'],
                'vol_state': day_row['vol_state'] if day_row is not None else 'UNKNOWN',
                'prior_concurrency': len(prior), 'session': row['session'],
                'classification': 'WINNER' if row['r_multiple'] > 0.05 else ('LOSER' if row['r_multiple'] < -0.05 else 'NEAR_ZERO'),
            })
    kept = df[np.array(keep_mask)].copy()
    return kept, pd.DataFrame(suppressed_rows)


def daily_R(trades_df):
    return trades_df.groupby('trade_date')['r_multiple'].sum().sort_index()


def metrics(daily_r_series, trades_df):
    if len(daily_r_series) == 0:
        return {k: None for k in ['total_R', 'max_dd_R', 'worst_day_R', 'worst_5day_R', 'worst_10day_R',
                                   'pctile95_daily_loss', 'pctile99_daily_loss', 'downside_dev', 'dd_duration_days',
                                   'recovery_duration_days', 'pf', 'trade_count']}
    r = daily_r_series.values
    cum = np.cumsum(r)
    running_peak = np.maximum.accumulate(cum)
    dd = cum - running_peak
    max_dd = dd.min()
    max_dd_idx = int(np.argmin(dd))
    # drawdown duration: consecutive days in_dd ending at max_dd_idx (or longest run overall)
    in_dd = (dd < -1e-9).astype(int)
    longest = cur = 0
    for v in in_dd:
        if v:
            cur += 1; longest = max(longest, cur)
        else:
            cur = 0
    # recovery duration from the max-dd point to when cum returns to the prior peak
    recovery = None
    peak_before = running_peak[max_dd_idx]
    for j in range(max_dd_idx, len(cum)):
        if cum[j] >= peak_before:
            recovery = j - max_dd_idx
            break
    roll3 = pd.Series(r).rolling(3).sum().min()
    roll5 = pd.Series(r).rolling(5).sum().min()
    roll10 = pd.Series(r).rolling(10).sum().min()
    downside = r[r < 0]
    downside_dev = np.std(downside) if len(downside) > 1 else (abs(downside[0]) if len(downside) == 1 else 0)
    wins = trades_df.loc[trades_df.r_multiple > 0, 'r_multiple'].sum()
    losses = trades_df.loc[trades_df.r_multiple < 0, 'r_multiple'].sum()
    pf = wins / abs(losses) if losses != 0 else np.nan
    return {
        'total_R': round(r.sum(), 3), 'max_dd_R': round(max_dd, 3), 'worst_day_R': round(r.min(), 4),
        'worst_5day_R': round(roll5, 4), 'worst_10day_R': round(roll10, 4),
        'pctile95_daily_loss': round(np.percentile(r, 5), 4), 'pctile99_daily_loss': round(np.percentile(r, 1), 4),
        'downside_dev': round(downside_dev, 4), 'dd_duration_days': longest,
        'recovery_duration_days': recovery, 'pf': round(pf, 3) if pf == pf else None,
        'trade_count': len(trades_df),
    }


def main():
    df = load_control()
    print(f"[control] {len(df)} trades, reconciled: {len(df) == 2712}")
    vol_ledger = build_daily_vol_ledger(df)

    # --- Control A: baseline ---
    a_daily = daily_R(df)
    a_metrics = metrics(a_daily, df)
    a_df = pd.DataFrame([{'control': 'A_baseline', **a_metrics}])
    a_df.to_csv(OUT / 'phase44_baseline.csv', index=False)
    print("\n[Control A -- baseline]"); print(a_df.to_string())

    # --- Control B: HIGH-vol day, alternate-suppress every 2nd new entry ---
    high_counter = {}
    def rule_b(row, prior, day_row, idx):
        if day_row is None or day_row['vol_state'] != 'HIGH':
            return False
        d = row['trade_date']
        high_counter[d] = high_counter.get(d, 0) + 1
        return (high_counter[d] % 2 == 0)  # suppress every 2nd entry that day
    b_kept, b_suppressed = suppress(df, vol_ledger, rule_b, 'B_high_vol_50pct')
    b_daily = daily_R(b_kept)
    b_metrics = metrics(b_daily, b_kept)
    b_df = pd.DataFrame([{'control': 'B_high_vol_50pct', **b_metrics}])
    b_df.to_csv(OUT / 'phase44_high_vol_control.csv', index=False)
    print("\n[Control B -- HIGH-vol 50% alternating suppression]"); print(b_df.to_string())
    print(f"  suppressed {len(b_suppressed)} trades ({len(b_suppressed)/len(df)*100:.1f}%)")

    # --- Control C: HIGH-vol AND concurrency>=4 ---
    def rule_c(row, prior, day_row, idx):
        return day_row is not None and day_row['vol_state'] == 'HIGH' and len(prior) >= 4
    c_kept, c_suppressed = suppress(df, vol_ledger, rule_c, 'C_high_vol_concurrency4')
    c_daily = daily_R(c_kept)
    c_metrics = metrics(c_daily, c_kept)
    c_df = pd.DataFrame([{'control': 'C_high_vol_concurrency4', **c_metrics}])
    c_df.to_csv(OUT / 'phase44_high_vol_concurrency_control.csv', index=False)
    print("\n[Control C -- HIGH-vol + concurrency>=4]"); print(c_df.to_string())
    print(f"  suppressed {len(c_suppressed)} trades ({len(c_suppressed)/len(df)*100:.1f}%)")

    # --- Control D: HIGH_to_NORMAL transition day ---
    def rule_d(row, prior, day_row, idx):
        return day_row is not None and bool(day_row['is_high_to_normal'])
    d_kept, d_suppressed = suppress(df, vol_ledger, rule_d, 'D_transition_high_to_normal')
    d_daily = daily_R(d_kept)
    d_metrics = metrics(d_daily, d_kept)
    d_df = pd.DataFrame([{'control': 'D_transition_high_to_normal', **d_metrics}])
    d_df.to_csv(OUT / 'phase44_transition_control.csv', index=False)
    print("\n[Control D -- HIGH_to_NORMAL transition]"); print(d_df.to_string())
    print(f"  suppressed {len(d_suppressed)} trades ({len(d_suppressed)/len(df)*100:.1f}%)")

    # --- Control E: concurrency>=5, exposure-agnostic ---
    def rule_e(row, prior, day_row, idx):
        return len(prior) >= 5
    e_kept, e_suppressed = suppress(df, vol_ledger, rule_e, 'E_concurrency5_agnostic')
    e_daily = daily_R(e_kept)
    e_metrics = metrics(e_daily, e_kept)
    e_df = pd.DataFrame([{'control': 'E_concurrency5_agnostic', **e_metrics}])
    e_df.to_csv(OUT / 'phase44_defensive_control.csv', index=False)
    print("\n[Control E -- concurrency>=5]"); print(e_df.to_string())
    print(f"  suppressed {len(e_suppressed)} trades ({len(e_suppressed)/len(df)*100:.1f}%)")

    # --- suppressed trades combined ---
    all_suppressed = pd.concat([b_suppressed, c_suppressed, d_suppressed, e_suppressed], ignore_index=True)
    all_suppressed.to_csv(OUT / 'phase44_suppressed_trades.csv', index=False)
    print(f"\n[suppressed trades total] {len(all_suppressed)} rows across all 4 controls")
    for ctrl in all_suppressed['control'].unique():
        sub = all_suppressed[all_suppressed.control == ctrl]
        win_pct = (sub['classification'] == 'WINNER').mean() * 100
        print(f"  {ctrl}: {len(sub)} suppressed, {win_pct:.1f}% were historical winners")

    # --- stress comparison (Phase41 stress windows, worst 1/5/10/20%) ---
    q = {p: np.percentile(a_daily.values, p) for p in [1, 5, 10, 20]}
    stress_rows = []
    controls_data = {'A_baseline': (a_daily, df), 'B_high_vol_50pct': (b_daily, b_kept),
                      'C_high_vol_concurrency4': (c_daily, c_kept), 'D_transition_high_to_normal': (d_daily, d_kept),
                      'E_concurrency5_agnostic': (e_daily, e_kept)}
    for name, (daily, trades) in controls_data.items():
        for pct in [1, 5, 10, 20]:
            stress_dates = a_daily[a_daily <= q[pct]].index  # stress windows FIXED from baseline, per Part20
            sub = daily.reindex(stress_dates).fillna(0)
            stress_rows.append({
                'control': name, 'stress_bucket': f'worst_{pct}pct', 'n_stress_days': len(stress_dates),
                'stress_period_R': round(sub.sum(), 3),
            })
    stress_df = pd.DataFrame(stress_rows)
    stress_df.to_csv(OUT / 'phase44_stress_comparison.csv', index=False)
    print("\n[stress-period comparison]"); print(stress_df[stress_df.stress_bucket == 'worst_5pct'].to_string())

    # --- trade-off analysis ---
    tradeoff_rows = []
    for name, m in [('B_high_vol_50pct', b_metrics), ('C_high_vol_concurrency4', c_metrics),
                     ('D_transition_high_to_normal', d_metrics), ('E_concurrency5_agnostic', e_metrics)]:
        # NOTE: max_dd_R and worst_5day_R are both negative numbers; a "reduction" (improvement)
        # means the control's magnitude is SMALLER (closer to zero) than baseline's, so the
        # comparison must be done on absolute values, not raw signed values.
        dd_reduction = (abs(a_metrics['max_dd_R']) - abs(m['max_dd_R'])) / abs(a_metrics['max_dd_R']) * 100 if a_metrics['max_dd_R'] else None
        r_reduction = (a_metrics['total_R'] - m['total_R']) / abs(a_metrics['total_R']) * 100 if a_metrics['total_R'] else None
        trade_reduction = (a_metrics['trade_count'] - m['trade_count']) / a_metrics['trade_count'] * 100
        worst5_reduction = (abs(a_metrics['worst_5day_R']) - abs(m['worst_5day_R'])) / abs(a_metrics['worst_5day_R']) * 100 if a_metrics['worst_5day_R'] else None
        tradeoff_rows.append({
            'control': name, 'pct_dd_reduction': round(dd_reduction, 1) if dd_reduction is not None else None,
            'pct_total_R_reduction': round(r_reduction, 1) if r_reduction is not None else None,
            'pct_trade_reduction': round(trade_reduction, 1),
            'pct_worst5day_reduction': round(worst5_reduction, 1) if worst5_reduction is not None else None,
            'tradeoff_ratio_dd_per_R_sacrificed': round(dd_reduction / r_reduction, 2) if dd_reduction and r_reduction and r_reduction != 0 else None,
        })
    tradeoff_df = pd.DataFrame(tradeoff_rows)
    tradeoff_df.to_csv(OUT / 'phase44_tradeoff_analysis.csv', index=False)
    print("\n[trade-off analysis]"); print(tradeoff_df.to_string())

    # --- regime robustness ---
    periods = {'C_2023_2024': ('2023-08-01', '2024-12-31'), 'D_2025': ('2025-01-01', '2025-12-31'), 'E_2026_YTD': ('2026-01-01', '2026-08-13')}
    regime_rows = [{'period': 'A_2019_2020', 'note': 'UNKNOWN BY DATA ABSENCE'}, {'period': 'B_2021_2022', 'note': 'UNKNOWN BY DATA ABSENCE'}]
    for pname, (s, e) in periods.items():
        s_ts, e_ts = pd.Timestamp(s), pd.Timestamp(e)
        for name, (daily, trades) in controls_data.items():
            idx_dt = pd.to_datetime(daily.index)
            sub = daily[(idx_dt >= s_ts) & (idx_dt <= e_ts)]
            regime_rows.append({'period': pname, 'control': name, 'n_days': len(sub), 'total_R': round(sub.sum(), 3),
                                 'max_dd_R': round((sub.cumsum() - sub.cumsum().cummax()).min(), 3) if len(sub) else None})
    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(OUT / 'phase44_regime_robustness.csv', index=False)
    print("\n[regime robustness] (sample)"); print(regime_df[regime_df.get('period') == 'E_2026_YTD'].to_string() if 'period' in regime_df else regime_df.tail(5).to_string())

    # --- extreme-day robustness ---
    a_sorted = a_daily.sort_values()
    ext_rows = []
    for n_excl in [0, 1, 5, 10]:
        excl_dates = set(a_sorted.head(n_excl).index) if n_excl else set()
        for name, (daily, trades) in controls_data.items():
            sub = daily[~daily.index.isin(excl_dates)]
            m = metrics(sub, trades[~trades['trade_date'].isin(excl_dates)])
            ext_rows.append({'excluding_worst_n_days': n_excl, 'control': name, 'total_R': m['total_R'], 'max_dd_R': m['max_dd_R']})
    ext_df = pd.DataFrame(ext_rows)
    ext_df.to_csv(OUT / 'phase44_extreme_day_robustness.csv', index=False)
    print("\n[extreme-day robustness] (n_excl=5)"); print(ext_df[ext_df.excluding_worst_n_days == 5].to_string())

    # --- cost sensitivity (disclosed limitation) ---
    cost_df = pd.DataFrame([{
        'note': 'Cost sensitivity (baseline vs 2x cost) is NOT COMPUTABLE from this dataset -- r_multiple/pnl do not separately expose the cost component per trade, and Phase44 does not re-simulate trades (only suppresses/retains historical ones). Disclosed limitation per preregistration section 3, not fabricated.',
        'status': 'UNKNOWN BY DATA LIMITATION',
    }])
    cost_df.to_csv(OUT / 'phase44_cost_sensitivity.csv', index=False)
    print("\n[cost sensitivity]"); print(cost_df.to_string())

    # --- Monte Carlo ---
    mc_rows = []
    for name, (daily, trades) in controls_data.items():
        r_arr = trades['r_multiple'].values
        if len(r_arr) < 10:
            mc_rows.append({'control': name, 'n_sims': 0, 'note': 'insufficient trades'})
            continue
        mc_dds = []
        for _ in range(10000):
            shuf = RNG.permutation(r_arr)
            cum = np.cumsum(shuf)
            mc_dds.append((cum - np.maximum.accumulate(cum)).min())
        mc_dds = np.array(mc_dds)
        actual_dd = metrics(daily, trades)['max_dd_R']
        mc_rows.append({
            'control': name, 'n_sims': 10000, 'n_trades': len(r_arr), 'data_type': 'SIMULATED (trade-order reshuffle)',
            'actual_max_dd_R': actual_dd, 'mc_dd_median': round(np.median(mc_dds), 3),
            'mc_dd_p95': round(np.percentile(mc_dds, 95), 3),
            'actual_dd_percentile_in_mc': round(float((mc_dds < actual_dd).mean() * 100), 1),
        })
    mc_df = pd.DataFrame(mc_rows)
    mc_df.to_csv(OUT / 'phase44_monte_carlo.csv', index=False)
    print("\n[Monte Carlo]"); print(mc_df.to_string())

    # --- false-positive check ---
    fp_df = pd.DataFrame([{
        'disclosure': 'ALL Phase44 findings are IN-SAMPLE COUNTERFACTUAL EVIDENCE. Controls C/D/E were constructed using thresholds/definitions derived from Phases 42-43s analysis of this EXACT SAME historical sample. This is not out-of-sample validation and must not be presented as production validity.',
        'label': 'IN-SAMPLE COUNTERFACTUAL EVIDENCE',
    }])
    fp_df.to_csv(OUT / 'phase44_false_positive_check.csv', index=False)

    # --- evidence matrix + classification ---
    def classify(name, m, tradeoff_row, suppressed_sub):
        dd_red = tradeoff_row['pct_dd_reduction']
        r_red = tradeoff_row['pct_total_R_reduction']
        win_pct_suppressed = (suppressed_sub['classification'] == 'WINNER').mean() * 100 if len(suppressed_sub) else None
        if dd_red is None or dd_red <= 0:
            return 'C. REJECTED -- NO MEANINGFUL BENEFIT', f'No drawdown reduction observed (dd_red={dd_red})'
        if r_red is not None and r_red > 0 and dd_red < r_red:
            return 'D. REJECTED -- EXCESSIVE RETURN SACRIFICE', f'Return sacrificed ({r_red}%) exceeds drawdown reduction ({dd_red}%)'
        if win_pct_suppressed is not None and win_pct_suppressed >= 45:
            return 'C. REJECTED -- NO MEANINGFUL BENEFIT', f'{win_pct_suppressed:.1f}% of suppressed trades were historical winners -- control removes activity broadly, not selectively bad trades'
        return 'B. MIXED / INSUFFICIENT', 'Some drawdown improvement with acceptable return sacrifice, but not clearing all 5 pre-registered success criteria (see master report)'

    ev_rows = []
    for name, tradeoff_row in zip(['B_high_vol_50pct', 'C_high_vol_concurrency4', 'D_transition_high_to_normal', 'E_concurrency5_agnostic'],
                                    tradeoff_rows):
        sup_sub = all_suppressed[all_suppressed.control == name]
        m = {'B_high_vol_50pct': b_metrics, 'C_high_vol_concurrency4': c_metrics,
             'D_transition_high_to_normal': d_metrics, 'E_concurrency5_agnostic': e_metrics}[name]
        cls, reason = classify(name, m, tradeoff_row, sup_sub)
        ev_rows.append({'control': name, 'pct_dd_reduction': tradeoff_row['pct_dd_reduction'],
                         'pct_total_R_reduction': tradeoff_row['pct_total_R_reduction'],
                         'pct_suppressed_winners': round((sup_sub['classification'] == 'WINNER').mean() * 100, 1) if len(sup_sub) else None,
                         'final_classification': cls, 'reason': reason})
    ev_df = pd.DataFrame(ev_rows)
    ev_df.to_csv(OUT / 'phase44_evidence_matrix.csv', index=False)
    print("\n[evidence matrix]"); print(ev_df.to_string())

    # --- future validation / research ideas ---
    fv_rows = [
        {'item': 'Any control reaching A. HISTORICALLY PROMISING would require a genuinely out-of-sample walk-forward or paper-trading validation before any live consideration', 'status': 'FUTURE VALIDATION REQUIREMENT -- NOT IMPLEMENTED'},
        {'item': 'Investigate a milder (e.g. 25%) HIGH-vol suppression fraction as a separate, independently preregistered future phase, if Control B shows a directionally promising but insufficiently large effect', 'status': 'FUTURE VALIDATION CANDIDATE -- NOT TESTED (would require new preregistration)'},
        {'item': 'Investigate whether suppressed-winner trades cluster in specific strategies, informing a more selective (not exposure-agnostic) future control design', 'status': 'FUTURE RESEARCH IDEA -- NOT IMPLEMENTED'},
    ]
    fv_df = pd.DataFrame(fv_rows)
    fv_df.to_csv(OUT / 'phase44_future_validation.csv', index=False)
    print("\n[future validation / research ideas]"); print(fv_df.to_string())

    summary = {'baseline_total_R': a_metrics['total_R'], 'baseline_max_dd': a_metrics['max_dd_R'],
               'n_suppressed_B': len(b_suppressed), 'n_suppressed_C': len(c_suppressed),
               'n_suppressed_D': len(d_suppressed), 'n_suppressed_E': len(e_suppressed)}
    with open(OUT / '_phase44_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n" + json.dumps(summary, indent=2, default=str))


if __name__ == '__main__':
    main()
