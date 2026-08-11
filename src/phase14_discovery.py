"""
Forex Bot - Phase 14: Research Discovery Phase 1
===================================================
Per user instruction: discover repeatable intraday FX phenomena BEFORE
building/optimizing another strategy. This script is descriptive/
exploratory only -- it does NOT generate trade signals, does NOT compute
PF/drawdown, and does NOT modify any live strategy (AMR included).

Pipeline for every question: phenomenon -> descriptive stats -> conditional
distribution -> hypothesis -> baseline check -> (NOT strategy, that's a
later phase pending user review).

Five families, each its own section below:
  1. Volatility regimes (compression -> expansion transitions)
  2. Conditional session behavior (Asia -> London -> NY)
  3. Market regime classification (does regime affect AMR's own trades?)
  4. Intraday price distribution after standardized ATR moves
  5. Intraday seasonality by hour

Requirements: MetaTrader 5 OPEN and LOGGED IN (or CSV cache in
data/historical/, see core/data_loader.py).
Output: reports/phase14_discovery_log.txt (full console log),
        data/phase14_*.csv (structured results per family)
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import data_loader
from strategy_matrix_backtest import windowed_atr, REPO_ROOT

PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD', 'GBPJPY', 'EURJPY', 'CADJPY']
MONTHS = 36

# server-hour session convention, matching this project's existing usage
ASIAN_START, ASIAN_END   = 0, 7
LONDON_START, LONDON_END = 7, 16
NY_START, NY_END         = 12, 21

PIP = {p: 0.01 if p.endswith('JPY') else 0.0001 for p in PAIRS}

LOG_LINES: list[str] = []


def say(msg=''):
    print(msg)
    LOG_LINES.append(str(msg))


def fetch(pair: str, tf: str) -> pd.DataFrame:
    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=MONTHS * 30 + 30)
    return data_loader.get_bars(pair, tf, date_from, date_to)


def add_session_col(df: pd.DataFrame) -> pd.DataFrame:
    h = df.index.hour
    sess = np.full(len(df), '', dtype=object)
    sess[(h >= ASIAN_START) & (h < ASIAN_END)] = 'ASIAN'
    sess[(h >= LONDON_START) & (h < LONDON_END)] = 'LONDON'
    sess[(h >= NY_START) & (h < NY_END)] = 'NY'
    out = df.copy()
    out['session'] = sess
    out['date'] = out.index.date
    return out


# ═══════════════════════════════════════════════════════════════════════
# FAMILY 1 -- VOLATILITY REGIMES
# ═══════════════════════════════════════════════════════════════════════

def family1_volatility_regimes(pair: str, h1: pd.DataFrame) -> dict:
    """Does today's session range-percentile predict tomorrow's/next
    session's range-percentile? (compression -> expansion persistence)"""
    pip = PIP[pair]
    df = add_session_col(h1)
    highs, lows, closes = df['High'].to_numpy(), df['Low'].to_numpy(), df['Close'].to_numpy()
    atr = windowed_atr(highs, lows, closes, 14, 66) / pip
    df['atr'] = atr

    results = {}
    for sess_name in ['ASIAN', 'LONDON', 'NY']:
        sub = df[df['session'] == sess_name]
        if sub.empty:
            continue
        daily_range = sub.groupby('date').apply(
            lambda g: (g['High'].max() - g['Low'].min()) / pip, include_groups=False)
        daily_range = daily_range.dropna()
        if len(daily_range) < 60:
            continue
        pctile = daily_range.rank(pct=True)
        # next session's range percentile, same series shifted by 1 (1 trading day ahead)
        next_pctile = pctile.shift(-1)
        valid = pctile.notna() & next_pctile.notna()
        x, y = pctile[valid].to_numpy(), next_pctile[valid].to_numpy()
        if len(x) < 60:
            continue
        corr = np.corrcoef(x, y)[0, 1]

        # compression -> expansion: bottom-quartile range day, what fraction
        # of the NEXT day's range is top-half?
        low_q = x <= 0.25
        n_low = low_q.sum()
        if n_low >= 20:
            p_expand_after_compress = (y[low_q] >= 0.5).mean()
        else:
            p_expand_after_compress = np.nan
        p_expand_baseline = (y >= 0.5).mean()

        # consecutive low-range persistence: 2 compressed days in a row -> 3rd day expansion?
        two_low = low_q[:-1] & low_q[1:] if len(low_q) > 1 else np.array([])
        results[sess_name] = dict(
            n_days=len(x), autocorr=corr,
            p_expand_after_compress=p_expand_after_compress,
            p_expand_baseline=p_expand_baseline,
            n_compressed_days=int(n_low),
        )
    return results


# ═══════════════════════════════════════════════════════════════════════
# FAMILY 2 -- CONDITIONAL SESSION BEHAVIOR (Asia -> London -> NY)
# ═══════════════════════════════════════════════════════════════════════

def _session_daily_metrics(df: pd.DataFrame, sess_name: str, pip: float) -> pd.DataFrame:
    sub = df[df['session'] == sess_name]
    if sub.empty:
        return pd.DataFrame()
    g = sub.groupby('date')
    rng = (g['High'].max() - g['Low'].min()) / pip
    ret = (g['Close'].last() - g['Open'].first()) / pip
    close_loc = g.apply(lambda x: (x['Close'].iloc[-1] - x['Low'].min()) /
                         max(x['High'].max() - x['Low'].min(), 1e-9), include_groups=False)
    out = pd.DataFrame({'range': rng, 'ret': ret, 'close_loc': close_loc})
    out['range_pctile'] = out['range'].rank(pct=True)
    out['direction'] = np.sign(out['ret'])
    return out


def family2_conditional_sessions(pair: str, h1: pd.DataFrame) -> dict:
    pip = PIP[pair]
    df = add_session_col(h1)
    asia = _session_daily_metrics(df, 'ASIAN', pip)
    london = _session_daily_metrics(df, 'LONDON', pip)
    ny = _session_daily_metrics(df, 'NY', pip)

    out = {}
    # Asia range percentile -> London range percentile
    j = asia[['range_pctile']].join(london[['range_pctile']], lsuffix='_asia', rsuffix='_london', how='inner')
    if len(j) >= 60:
        out['asia_range_to_london_range_corr'] = np.corrcoef(
            j['range_pctile_asia'], j['range_pctile_london'])[0, 1]
        out['n_asia_london'] = len(j)

    # Asia direction -> London direction (same-sign probability)
    j2 = asia[['direction']].join(london[['direction']], lsuffix='_asia', rsuffix='_london', how='inner')
    j2 = j2[(j2['direction_asia'] != 0) & (j2['direction_london'] != 0)]
    if len(j2) >= 60:
        out['p_london_follows_asia_direction'] = (j2['direction_asia'] == j2['direction_london']).mean()
        out['n_direction_pairs'] = len(j2)

    # London range percentile -> NY continuation (NY range top-half probability)
    j3 = london[['range_pctile']].join(ny[['range_pctile']], lsuffix='_london', rsuffix='_ny', how='inner')
    if len(j3) >= 60:
        hi = j3['range_pctile_london'] >= 0.75
        if hi.sum() >= 20:
            out['p_ny_expand_after_london_expand'] = (j3.loc[hi, 'range_pctile_ny'] >= 0.5).mean()
        out['p_ny_expand_baseline'] = (j3['range_pctile_ny'] >= 0.5).mean()

    # London close location -> NY return sign
    j4 = london[['close_loc']].join(ny[['ret']], how='inner')
    if len(j4) >= 60:
        near_high = j4['close_loc'] >= 0.8
        near_low = j4['close_loc'] <= 0.2
        if near_high.sum() >= 20:
            out['p_ny_up_after_london_close_near_high'] = (j4.loc[near_high, 'ret'] > 0).mean()
        if near_low.sum() >= 20:
            out['p_ny_up_after_london_close_near_low'] = (j4.loc[near_low, 'ret'] > 0).mean()
        out['p_ny_up_baseline'] = (j4['ret'] > 0).mean()

    return out


# ═══════════════════════════════════════════════════════════════════════
# FAMILY 3 -- REGIME CLASSIFICATION (does regime predict AMR's own edge?)
# ═══════════════════════════════════════════════════════════════════════

def efficiency_ratio(closes: np.ndarray, window: int) -> np.ndarray:
    """Kaufman efficiency ratio: |net move| / sum(|bar-to-bar moves|) over
    a rolling window. 1.0 = pure trend, ~0 = pure noise/mean-reversion."""
    net = np.abs(closes - np.roll(closes, window))
    net[:window] = np.nan
    diffs = np.abs(np.diff(closes, prepend=closes[0]))
    roll_sum = pd.Series(diffs).rolling(window).sum().to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        er = net / roll_sum
    er[:window] = np.nan
    return er


def family3_regime_classification(pair: str, h1: pd.DataFrame) -> dict:
    """Classify each bar's regime via ATR percentile x efficiency ratio,
    then check whether AMR's own asian-hours mean-reversion signal
    direction (BUY on low, SELL on high, i.e. the AMR thesis) tends to be
    correct more often in mean-reverting vs trending regimes. This reuses
    only the price data, NOT the live AMR class -- read-only research."""
    pip = PIP[pair]
    closes = h1['Close'].to_numpy()
    highs, lows = h1['High'].to_numpy(), h1['Low'].to_numpy()
    atr = windowed_atr(highs, lows, closes, 14, 66) / pip
    atr_pctile = pd.Series(atr).rank(pct=True).to_numpy()
    er = efficiency_ratio(closes, 20)

    df = add_session_col(h1)
    df['atr_pctile'] = atr_pctile
    df['er'] = er
    asian = df[df['session'] == 'ASIAN'].dropna(subset=['er', 'atr_pctile'])
    if len(asian) < 200:
        return {}

    # AMR thesis check: within the Asian session, does a low-ER (mean-
    # reverting) hour followed by an extreme close within that hour see
    # more reversion over the next 4 bars than a high-ER (trending) hour?
    fwd4 = (pd.Series(closes).shift(-4) - pd.Series(closes)).to_numpy() / pip
    asian = asian.copy()
    asian['fwd4'] = pd.Series(fwd4, index=df.index)[asian.index]
    low_er = asian['er'] <= asian['er'].quantile(0.33)
    high_er = asian['er'] >= asian['er'].quantile(0.67)

    bar_ret = (pd.Series(closes) - pd.Series(closes).shift(1)).to_numpy() / pip
    asian['bar_ret'] = pd.Series(bar_ret, index=df.index)[asian.index]
    up_move = asian['bar_ret'] > 0

    # reversion rate: after an up-move bar, is fwd4 negative (mean-reverting)?
    rev_low_er = (asian.loc[low_er & up_move, 'fwd4'] < 0).mean() if (low_er & up_move).sum() >= 20 else np.nan
    rev_high_er = (asian.loc[high_er & up_move, 'fwd4'] < 0).mean() if (high_er & up_move).sum() >= 20 else np.nan
    rev_baseline = (asian.loc[up_move, 'fwd4'] < 0).mean() if up_move.sum() >= 20 else np.nan

    return dict(
        n_asian_bars=len(asian),
        reversion_rate_low_er=rev_low_er,
        reversion_rate_high_er=rev_high_er,
        reversion_rate_baseline=rev_baseline,
        n_low_er_up=int((low_er & up_move).sum()),
        n_high_er_up=int((high_er & up_move).sum()),
    )


# ═══════════════════════════════════════════════════════════════════════
# FAMILY 4 -- INTRADAY PRICE DISTRIBUTION AFTER STANDARDIZED MOVES
# ═══════════════════════════════════════════════════════════════════════

def family4_post_move_distribution(pair: str, m15: pd.DataFrame) -> dict:
    pip = PIP[pair]
    closes = m15['Close'].to_numpy()
    highs, lows = m15['High'].to_numpy(), m15['Low'].to_numpy()
    atr = windowed_atr(highs, lows, closes, 14, 66) / pip
    move = (pd.Series(closes).diff()).to_numpy() / pip  # 1-bar (15min) move in pips
    move_atr = move / np.where(atr > 0, atr, np.nan)

    horizons = {'15m': 1, '30m': 2, '60m': 4, '120m': 8}
    thresholds = [0.5, 1.0, 1.5]
    out = {}
    for thr in thresholds:
        up = move_atr >= thr
        down = move_atr <= -thr
        for label, n_bars in horizons.items():
            fwd = (pd.Series(closes).shift(-n_bars) - pd.Series(closes)).to_numpy() / pip
            fwd_atr = fwd / np.where(atr > 0, atr, np.nan)
            up_valid = up & ~np.isnan(fwd_atr)
            down_valid = down & ~np.isnan(fwd_atr)
            if up_valid.sum() >= 30:
                out[f'up_{thr}atr_{label}_continuation_rate'] = float((fwd_atr[up_valid] > 0).mean())
                out[f'up_{thr}atr_{label}_mean_fwd_atr'] = float(np.nanmean(fwd_atr[up_valid]))
                out[f'up_{thr}atr_{label}_n'] = int(up_valid.sum())
            if down_valid.sum() >= 30:
                out[f'down_{thr}atr_{label}_continuation_rate'] = float((fwd_atr[down_valid] < 0).mean())
                out[f'down_{thr}atr_{label}_mean_fwd_atr'] = float(np.nanmean(fwd_atr[down_valid]))
                out[f'down_{thr}atr_{label}_n'] = int(down_valid.sum())
    return out


# ═══════════════════════════════════════════════════════════════════════
# FAMILY 5 -- INTRADAY SEASONALITY BY HOUR
# ═══════════════════════════════════════════════════════════════════════

def family5_seasonality(pair: str, h1: pd.DataFrame) -> pd.DataFrame:
    pip = PIP[pair]
    df = h1.copy()
    df['ret'] = (df['Close'] - df['Open']) / pip
    df['range'] = (df['High'] - df['Low']) / pip
    df['hour'] = df.index.hour
    df['year'] = df.index.year

    rows = []
    for hr, g in df.groupby('hour'):
        if len(g) < 100:
            continue
        rows.append(dict(
            hour=hr, n=len(g), mean_ret=g['ret'].mean(), median_ret=g['ret'].median(),
            std_ret=g['ret'].std(), mean_range=g['range'].mean(),
            p_positive=(g['ret'] > 0).mean(),
            p05=g['ret'].quantile(0.05), p95=g['ret'].quantile(0.95),
        ))
    hourly = pd.DataFrame(rows)

    # year-consistency: for the single best/worst hour by |mean_ret|, is
    # the sign consistent across years?
    year_consistency = {}
    if not hourly.empty:
        best_hr = hourly.loc[hourly['mean_ret'].abs().idxmax(), 'hour']
        by_year = df[df['hour'] == best_hr].groupby('year')['ret'].mean()
        year_consistency = dict(
            best_hour=int(best_hr),
            years_same_sign=int((np.sign(by_year) == np.sign(by_year.mean())).sum()),
            n_years=len(by_year),
            per_year=by_year.to_dict(),
        )
    return hourly, year_consistency


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    say('=' * 90)
    say('PHASE 14 -- RESEARCH DISCOVERY PHASE 1 (descriptive only, no strategies)')
    say(f'Pairs: {PAIRS}   Months: {MONTHS}   Run: {datetime.now(timezone.utc).isoformat()}')
    say('=' * 90)

    all_f1, all_f2, all_f3, all_f4, all_f5_hourly = [], [], [], [], []

    for pair in PAIRS:
        say(f'\n--- {pair} ---')
        try:
            h1 = fetch(pair, 'H1')
            m15 = fetch(pair, 'M15')
        except Exception as e:
            say(f'  SKIP ({e})')
            continue
        if len(h1) < 500 or len(m15) < 2000:
            say(f'  SKIP (insufficient data: H1={len(h1)}, M15={len(m15)})')
            continue

        r1 = family1_volatility_regimes(pair, h1)
        for sess, v in r1.items():
            all_f1.append(dict(pair=pair, session=sess, **v))
        say(f'  F1 volatility-regime: {r1}')

        r2 = family2_conditional_sessions(pair, h1)
        all_f2.append(dict(pair=pair, **r2))
        say(f'  F2 conditional-session: {r2}')

        r3 = family3_regime_classification(pair, h1)
        if r3:
            all_f3.append(dict(pair=pair, **r3))
        say(f'  F3 regime-classification: {r3}')

        r4 = family4_post_move_distribution(pair, m15)
        all_f4.append(dict(pair=pair, **r4))
        say(f'  F4 post-move (1.0 ATR / 60m): up={r4.get("up_1.0atr_60m_continuation_rate")}, '
            f'down={r4.get("down_1.0atr_60m_continuation_rate")}')

        hourly, yc = family5_seasonality(pair, h1)
        hourly['pair'] = pair
        all_f5_hourly.append(hourly)
        say(f'  F5 seasonality best-hour: {yc}')

    out_dir = REPO_ROOT / 'data'
    pd.DataFrame(all_f1).to_csv(out_dir / 'phase14_family1_volatility.csv', index=False)
    pd.DataFrame(all_f2).to_csv(out_dir / 'phase14_family2_conditional_session.csv', index=False)
    pd.DataFrame(all_f3).to_csv(out_dir / 'phase14_family3_regime.csv', index=False)
    pd.DataFrame(all_f4).to_csv(out_dir / 'phase14_family4_postmove.csv', index=False)
    pd.concat(all_f5_hourly, ignore_index=True).to_csv(out_dir / 'phase14_family5_seasonality.csv', index=False)

    say('\n' + '=' * 90)
    say('Wrote structured results to data/phase14_family{1..5}_*.csv')
    say('=' * 90)

    report_path = REPO_ROOT / 'reports' / 'phase14_discovery_log.txt'
    report_path.write_text('\n'.join(LOG_LINES), encoding='utf-8')
    say(f'\nFull log written to {report_path}')


if __name__ == '__main__':
    main()
