"""
Forex Bot - Phase 22: AUDJPY AMR Confirmatory Filter Experiment
========================================================================
ONE controlled confirmatory experiment on AUDJPY AMR ONLY, testing two
pre-registered, frozen candidate filters against the exact existing
baseline: does volatility filtering (Model A) or removing SELL trades
(Model B) improve AUDJPY AMR out-of-sample? Model C (A+B combined) is
secondary/exploratory only, per instructions, not used to pick a winner.

Does NOT modify AUDJPY AMR or any other strategy. Does NOT change the
demo account. All three filters are frozen BEFORE the OOS split is
examined -- no threshold, stop, target, session, or entry logic is
searched anywhere in this file.

Baseline: existing live AUDJPY AMR parameters (z_thr=2.0, sl_mult=1.5,
end_hour=4), unchanged reconstruction from phase20/21.

Requirements: MetaTrader 5 OPEN and LOGGED IN (or CSV cache).
Output: reports/phase22_confirmatory_log.txt, data/phase22_audjpy_trades.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy_matrix_backtest import run_sim, windowed_atr, REPO_ROOT
from phase3b_amr_jpy_refine import signals_amr_v
from phase21_amr_regime_mechanism import fetch

PAIR = 'AUDJPY'
Z_THR, SL_MULT, END_HOUR = 2.0, 1.5, 4     # frozen live AUDJPY AMR params, unchanged
SPREAD_NORMAL = 2.0                         # frozen live spread assumption, unchanged
RISK_PCT = 0.0025                           # frozen live risk, unchanged
ATR_PCTILE_THRESHOLD = 0.75                 # frozen BEFORE this experiment (phase20/21's HIGH boundary)

LOG: list[str] = []


def say(msg=''):
    print(msg)
    LOG.append(str(msg))


def build_full_trades() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (m15, trades_df) -- trades_df has one row per baseline
    AUDJPY AMR trade with entry-time ATR percentile (frozen, unchanged
    definition from phase20/21) attached."""
    m15 = fetch(PAIR, 'M15')
    pip = 0.01
    highs, lows, closes = m15['High'].to_numpy(), m15['Low'].to_numpy(), m15['Close'].to_numpy()
    atr = windowed_atr(highs, lows, closes, 14, 66) / pip
    atr_pctile = pd.Series(atr, index=m15.index).rank(pct=True)

    cands = signals_amr_v(m15, pip, SPREAD_NORMAL, Z_THR, SL_MULT, END_HOUR)
    idx_map = {t: i for i, t in enumerate(m15.index)}

    tdf, _ = run_sim(m15, cands, pip, SPREAD_NORMAL, RISK_PCT)
    tdf = tdf.copy()
    entry_idx = tdf['entry_time'].map(idx_map)
    tdf['atr_pctile'] = entry_idx.map(pd.Series(atr_pctile.to_numpy()))
    with np.errstate(divide='ignore', invalid='ignore'):
        tdf['r_multiple'] = np.where(tdf['sl_pips'] > 0, tdf['pips'] / tdf['sl_pips'], np.nan)
    tdf['year'] = tdf['entry_time'].dt.year
    tdf['month'] = tdf['entry_time'].dt.to_period('M')
    tdf['quarter'] = tdf['entry_time'].dt.to_period('Q')

    mfes, maes = [], []
    for _, row in tdf.iterrows():
        win = m15.loc[row['entry_time']:row['exit_time']]
        a = atr[idx_map.get(row['entry_time'], 0)] if row['entry_time'] in idx_map else np.nan
        if win.empty or not a or np.isnan(a) or a <= 0:
            mfes.append(np.nan); maes.append(np.nan); continue
        if row['dir'] == 'BUY':
            mfe = (win['High'].max() - row['entry']) / pip / a
            mae = (row['entry'] - win['Low'].min()) / pip / a
        else:
            mfe = (row['entry'] - win['Low'].min()) / pip / a
            mae = (win['High'].max() - row['entry']) / pip / a
        mfes.append(mfe); maes.append(mae)
    tdf['mfe_atr'] = mfes
    tdf['mae_atr'] = maes
    return m15, tdf


def run_variant(m15, cands, spread_mult=1.0, delay_bars=0) -> pd.DataFrame:
    """Re-simulate a (possibly filtered) candidate list under cost stress."""
    pip = 0.01
    if delay_bars:
        n = len(m15)
        cands = [(min(i + delay_bars, n - 1), d, sl, tp) for i, d, sl, tp in cands]
    tdf, _ = run_sim(m15, cands, pip, SPREAD_NORMAL * spread_mult, RISK_PCT)
    return tdf


def summarize(tdf: pd.DataFrame) -> dict:
    if tdf.empty:
        return dict(n=0)
    wins = tdf[tdf.pnl > 0]['pnl'].sum()
    losses = -tdf[tdf.pnl < 0]['pnl'].sum()
    pf = wins / losses if losses > 0 else np.nan
    cum = tdf['pnl'].cumsum()
    dd = (cum - cum.cummax()).min()
    is_loss = (tdf['pnl'] < 0).to_numpy()
    max_streak, cur = 0, 0
    for x in is_loss:
        cur = cur + 1 if x else 0
        max_streak = max(max_streak, cur)
    r10 = tdf['pnl'].rolling(10).sum().min() if len(tdf) >= 10 else np.nan
    return dict(n=len(tdf), win_rate=float((tdf.pnl > 0).mean()),
                expectancy=float(tdf['pnl'].mean()), avg_r=float(tdf['r_multiple'].mean()),
                median_r=float(tdf['r_multiple'].median()), pf=float(pf),
                total_r=float(tdf['r_multiple'].sum()), max_dd=float(dd),
                max_losing_streak=int(max_streak), worst_10trade_seq=float(r10) if not np.isnan(r10) else np.nan,
                avg_mfe=float(tdf['mfe_atr'].mean()) if 'mfe_atr' in tdf else np.nan,
                avg_mae=float(tdf['mae_atr'].mean()) if 'mae_atr' in tdf else np.nan)


def bootstrap_ci_diff(a: pd.Series, b: pd.Series, col='pnl', n_boot=2000, seed=53):
    rng = np.random.default_rng(seed)
    a_arr, b_arr = a[col].to_numpy(), b[col].to_numpy()
    if len(a_arr) < 10 or len(b_arr) < 10:
        return dict(mean_diff=np.nan, ci_low=np.nan, ci_high=np.nan, pct_above_zero=np.nan)
    diffs = np.array([
        rng.choice(a_arr, size=len(a_arr), replace=True).mean() -
        rng.choice(b_arr, size=len(b_arr), replace=True).mean()
        for _ in range(n_boot)
    ])
    return dict(mean_diff=float(diffs.mean()), ci_low=float(np.percentile(diffs, 2.5)),
                ci_high=float(np.percentile(diffs, 97.5)), pct_above_zero=float((diffs > 0).mean()))


def main():
    say('=' * 90)
    say('PHASE 22 -- AUDJPY AMR CONFIRMATORY FILTER EXPERIMENT (research only, no live change)')
    say('=' * 90)
    say('PART 1 -- FROZEN BASELINE: existing live AUDJPY AMR (z_thr=2.0, sl_mult=1.5, end_hour=4,')
    say(f'  spread={SPREAD_NORMAL}pips, risk={RISK_PCT:.2%}). Entry/exit/stop/target/session unchanged.')
    say('PART 2 -- FROZEN CANDIDATE FILTERS (pre-registered before any OOS result is examined):')
    say(f'  MODEL A (volatility filter): entry-time ATR percentile < {ATR_PCTILE_THRESHOLD} (the pre-')
    say('    existing HIGH-regime boundary from phase20/21 -- NOT searched here).')
    say('  MODEL B (BUY-only): exclude SELL trades, BUY entry unchanged.')
    say('  MODEL C (secondary/exploratory only): A AND B combined -- NOT used to pick a winner.')

    m15, tdf = build_full_trades()
    tdf.to_csv(REPO_ROOT / 'data' / 'phase22_audjpy_trades.csv', index=False)
    say(f'\nTotal baseline trades reconstructed: {len(tdf)} '
        f'({(tdf.dir=="BUY").sum()} BUY / {(tdf.dir=="SELL").sum()} SELL)')
    say(f'Date range: {tdf["entry_time"].min()} to {tdf["entry_time"].max()}')

    # ---- PART 3: strict chronological 3-way split (frozen before any result examined) ----
    say('\n' + '=' * 90)
    say('PART 3 -- DATA SPLIT (strict chronological thirds, frozen before results examined)')
    say('=' * 90)
    t0, t1 = tdf['entry_time'].min(), tdf['entry_time'].max()
    span = t1 - t0
    train_end = t0 + span / 3
    val_end = t0 + 2 * span / 3
    say(f'  TRAIN/IS:    {t0.date()} to {train_end.date()}')
    say(f'  VALIDATION:  {train_end.date()} to {val_end.date()}')
    say(f'  FINAL OOS:   {val_end.date()} to {t1.date()}')

    def split_mask(df):
        return dict(TRAIN=df['entry_time'] < train_end,
                    VALIDATION=(df['entry_time'] >= train_end) & (df['entry_time'] < val_end),
                    OOS=df['entry_time'] >= val_end)

    masks = split_mask(tdf)
    for period, mask in masks.items():
        say(f'  {period}: {mask.sum()} trades')

    model_A = tdf[tdf['atr_pctile'] < ATR_PCTILE_THRESHOLD]
    model_B = tdf[tdf['dir'] == 'BUY']
    model_C = tdf[(tdf['atr_pctile'] < ATR_PCTILE_THRESHOLD) & (tdf['dir'] == 'BUY')]
    masks_A = split_mask(model_A)
    masks_B = split_mask(model_B)
    masks_C = split_mask(model_C)

    # ---- PART 5/6: primary comparison, by period ----
    say('\n' + '=' * 90)
    say('PART 5/6 -- PRIMARY COMPARISON: BASELINE vs MODEL A vs MODEL B, by period (OOS is decisive)')
    say('=' * 90)
    results = {}
    for name, df, msk in [('BASELINE', tdf, masks), ('MODEL_A', model_A, masks_A),
                           ('MODEL_B', model_B, masks_B), ('MODEL_C_secondary', model_C, masks_C)]:
        say(f'\n-- {name} --')
        rows = []
        for period in ['TRAIN', 'VALIDATION', 'OOS']:
            s = summarize(df[msk[period]])
            rows.append(dict(period=period, **s))
        out = pd.DataFrame(rows)
        say(out.to_string(index=False))
        results[name] = out

    # ---- PART 7: walk-forward (6-month rolling windows across full history) ----
    say('\n' + '=' * 90)
    say('PART 7 -- WALK-FORWARD (6-month rolling windows, frozen rules, no re-fitting per window)')
    say('=' * 90)
    window_starts = pd.date_range(t0, t1 - pd.Timedelta(days=180), freq='90D')
    for name, df in [('BASELINE', tdf), ('MODEL_A', model_A), ('MODEL_B', model_B)]:
        say(f'\n-- {name} --')
        rows = []
        for ws in window_starts:
            we = ws + pd.Timedelta(days=180)
            sub = df[(df['entry_time'] >= ws) & (df['entry_time'] < we)]
            s = summarize(sub)
            rows.append(dict(window_start=ws.date(), window_end=we.date(), **s))
        say(pd.DataFrame(rows).to_string(index=False))

    # ---- PART 8: cost stress ----
    say('\n' + '=' * 90)
    say('PART 8 -- COST STRESS (re-simulated, not re-fit)')
    say('=' * 90)
    pip = 0.01
    cands_all = signals_amr_v(m15, pip, SPREAD_NORMAL, Z_THR, SL_MULT, END_HOUR)

    # recompute atr_pctile array aligned to m15 for filtering candidates directly
    highs, lows, closes = m15['High'].to_numpy(), m15['Low'].to_numpy(), m15['Close'].to_numpy()
    atr = windowed_atr(highs, lows, closes, 14, 66) / pip
    atr_pctile_series = pd.Series(atr, index=m15.index).rank(pct=True)

    cands_A = [c for c in cands_all if atr_pctile_series.iloc[c[0]] < ATR_PCTILE_THRESHOLD]
    cands_B = [c for c in cands_all if c[1] == 'BUY']

    for name, cands in [('BASELINE', cands_all), ('MODEL_A', cands_A), ('MODEL_B', cands_B)]:
        say(f'\n-- {name} --')
        rows = []
        for label, mult, delay in [('normal', 1.0, 0), ('1.5x_spread', 1.5, 0),
                                    ('2x_spread', 2.0, 0), ('1bar_delay', 1.0, 1)]:
            t = run_variant(m15, cands, spread_mult=mult, delay_bars=delay)
            if not t.empty:
                with np.errstate(divide='ignore', invalid='ignore'):
                    t['r_multiple'] = np.where(t['sl_pips'] > 0, t['pips'] / t['sl_pips'], np.nan)
                t['mfe_atr'] = np.nan; t['mae_atr'] = np.nan
            rows.append(dict(scenario=label, **summarize(t)))
        say(pd.DataFrame(rows).to_string(index=False))

    # ---- PART 9: statistical comparison (OOS only, the decisive period) ----
    say('\n' + '=' * 90)
    say('PART 9 -- STATISTICAL COMPARISON (bootstrap, FINAL OOS period only)')
    say('=' * 90)
    base_oos = tdf[masks['OOS']]
    a_oos = model_A[masks_A['OOS']]
    b_oos = model_B[masks_B['OOS']]
    for name, sub in [('MODEL_A', a_oos), ('MODEL_B', b_oos)]:
        ci = bootstrap_ci_diff(sub, base_oos, col='pnl')
        say(f'{name} - BASELINE expectancy diff (OOS): mean_diff={ci["mean_diff"]:+.2f}  '
            f'95% CI=[{ci["ci_low"]:+.2f}, {ci["ci_high"]:+.2f}]  P({name}>BASELINE)={ci["pct_above_zero"]:.3f}  '
            f'(n_{name}={len(sub)}, n_BASELINE={len(base_oos)})')

    # ---- PART 10: trade-count / opportunity impact ----
    say('\n' + '=' * 90)
    say('PART 10 -- TRADE-COUNT / OPPORTUNITY IMPACT (full history)')
    say('=' * 90)
    base_n, base_r, base_profit = len(tdf), tdf['r_multiple'].sum(), tdf[tdf.pnl > 0]['pnl'].sum()
    for name, df in [('MODEL_A', model_A), ('MODEL_B', model_B), ('MODEL_C', model_C)]:
        n_ret = 100 * len(df) / base_n
        r_ret = 100 * df['r_multiple'].sum() / base_r if base_r else np.nan
        profit_ret = 100 * df[df.pnl > 0]['pnl'].sum() / base_profit if base_profit else np.nan
        say(f'{name}: trades retained={n_ret:.1f}%  total-R retained={r_ret:.1f}%  '
            f'gross-profit retained={profit_ret:.1f}%  opportunity reduction={100-n_ret:.1f}%')

    # ---- PART 11: year consistency ----
    say('\n' + '=' * 90)
    say('PART 11 -- YEAR CONSISTENCY')
    say('=' * 90)
    for name, df in [('BASELINE', tdf), ('MODEL_A', model_A), ('MODEL_B', model_B)]:
        say(f'\n-- {name} --')
        rows = [dict(year=yr, **summarize(df[df.year == yr])) for yr in [2024, 2025, 2026] if (df.year == yr).sum() > 0]
        say(pd.DataFrame(rows).to_string(index=False))

    # ---- PART 12: drawdown / worst-period analysis ----
    say('\n' + '=' * 90)
    say('PART 12 -- DRAWDOWN / WORST-PERIOD ANALYSIS')
    say('=' * 90)
    for name, df in [('BASELINE', tdf), ('MODEL_A', model_A), ('MODEL_B', model_B)]:
        worst_month = df.groupby('month')['pnl'].sum().min() if not df.empty else np.nan
        worst_quarter = df.groupby('quarter')['pnl'].sum().min() if not df.empty else np.nan
        s = summarize(df)
        say(f'{name}: max_dd={s.get("max_dd", np.nan):.2f}  max_losing_streak={s.get("max_losing_streak", np.nan)}  '
            f'worst_10trade_seq={s.get("worst_10trade_seq", np.nan):.2f}  worst_month={worst_month:.2f}  '
            f'worst_quarter={worst_quarter:.2f}')

    # ---- PART 13: regime robustness ----
    say('\n' + '=' * 90)
    say('PART 13 -- REGIME ROBUSTNESS (reporting only, no re-optimization)')
    say('=' * 90)
    say(f'MODEL A by construction only contains trades with ATR percentile < {ATR_PCTILE_THRESHOLD} -- '
        f'min={model_A["atr_pctile"].min():.3f}, max={model_A["atr_pctile"].max():.3f} (sanity check).')
    excluded_sell = tdf[tdf.dir == 'SELL']
    say('MODEL B: BUY (retained) vs SELL (excluded) full-history comparison:')
    say(pd.DataFrame([dict(group='BUY (retained)', **summarize(model_B)),
                       dict(group='SELL (excluded)', **summarize(excluded_sell))]).to_string(index=False))

    report_path = REPO_ROOT / 'reports' / 'phase22_confirmatory_log.txt'
    report_path.write_text('\n'.join(LOG), encoding='utf-8')
    say(f'\nFull log written to {report_path}')


if __name__ == '__main__':
    main()
