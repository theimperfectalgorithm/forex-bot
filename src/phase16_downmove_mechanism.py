"""
Forex Bot - Phase 16: Down-Move Reversion — MECHANISM Research
===================================================================
Follow-up to phase 15 (down-move reversion baseline: phenomenon GENUINE,
naive 1:1 trading implementation FAILED). This phase does NOT touch the
trading implementation at all -- it investigates WHY some >=1.0 ATR M15
down-moves reverse and others continue, via 6 pre-specified, economically
motivated explanatory variables. No parameter search, no filter
combination, no strategy. No existing strategy (ARB/AMR/Monday-drift) is
read, imported, or modified.

Event definition (frozen, unchanged from phase 15): M15 close-to-close
move <= -1.0 x ATR(Wilder14, 66-bar window). Outcome split (frozen,
unchanged from phase 15's own continuation/reversal definition, 4-bar/
60-min horizon): REVERSAL = fwd_atr >= 0 (net higher 60 min later),
CONTINUATION = fwd_atr < 0.

Requirements: MetaTrader 5 OPEN and LOGGED IN (or CSV cache).
Output: reports/phase16_mechanism_log.txt, data/phase16_events.csv (the
full per-event table, for any follow-up slicing without re-fetching).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy_matrix_backtest import REPO_ROOT
from phase15_downmove_reversion_baseline import (
    PAIRS, PIP, ASIAN, LONDON, OVERLAP, NY,
    fetch_m15, session_of_hour, forward_extreme, PairData,
)

THR = 1.0          # frozen event threshold, unchanged from phase15
HORIZON = 4         # frozen outcome horizon (60 min), unchanged from phase15

LOG: list[str] = []


def say(msg=''):
    print(msg)
    LOG.append(str(msg))


# ═══════════════════════════════════════════════════════════════════════
# BUILD PER-EVENT TABLE: one row per >=1.0 ATR down-move, with outcome +
# all Hypothesis 1-6 explanatory variables computed from data available
# STRICTLY BEFORE/AT the event bar (no lookahead).
# ═══════════════════════════════════════════════════════════════════════

def asian_third(hour: int) -> str:
    if 0 <= hour <= 2:
        return 'EARLY_ASIAN'
    if 3 <= hour <= 4:
        return 'MID_ASIAN'
    if 5 <= hour <= 6:
        return 'LATE_ASIAN'
    return 'NOT_ASIAN'


def build_events(pdz: PairData) -> pd.DataFrame:
    n = len(pdz.closes)
    closes, highs, lows, atr, pip = pdz.closes, pdz.highs, pdz.lows, pdz.atr, pdz.pip
    df = pdz.df
    dates = pdz.idx.date
    hours = pdz.idx.hour

    # outcome (frozen, from phase15)
    fwd_close = np.roll(closes, -HORIZON); fwd_close[-HORIZON:] = np.nan
    fwd_atr = (fwd_close - closes) / pip / np.where(atr > 0, atr, np.nan)
    down = (pdz.move_atr <= -THR) & ~np.isnan(fwd_atr)

    if down.sum() == 0:
        return pd.DataFrame()

    # ---- session-so-far running high/low (H1: broader market location) ----
    date_arr = pd.Series(dates)
    sess_arr = pd.Series(pdz.session)
    sess_key = date_arr.astype(str) + '_' + sess_arr.astype(str)
    highs_s = pd.Series(highs)
    lows_s = pd.Series(lows)
    sess_high_so_far = highs_s.groupby(sess_key).cummax().to_numpy()
    sess_low_so_far = lows_s.groupby(sess_key).cummin().to_numpy()

    # ---- daily running high/low (position within current daily range so far) ----
    day_high_so_far = highs_s.groupby(date_arr).cummax().to_numpy()
    day_low_so_far = lows_s.groupby(date_arr).cummin().to_numpy()

    # ---- previous COMPLETE calendar day's H/L (shift by 1 day via daily resample) ----
    daily_hl = df.resample('1D').agg({'High': 'max', 'Low': 'min'}).dropna()
    daily_hl.index = daily_hl.index.date  # tz-naive plain date keys, matches `dates` below
    prev_day_high_map = daily_hl['High'].shift(1)
    prev_day_low_map = daily_hl['Low'].shift(1)
    day_index = pd.Series(dates, index=pdz.idx)
    prev_high_series = day_index.map(prev_day_high_map)
    prev_low_series = day_index.map(prev_day_low_map)
    prev_day_high = prev_high_series.to_numpy()
    prev_day_low = prev_low_series.to_numpy()

    # ---- recent H1-equivalent (4h = 16 bars) / H4-equivalent (24h = 96 bars) extremes ----
    recent_h1_high = highs_s.rolling(16, min_periods=16).max().to_numpy()
    recent_h1_low = lows_s.rolling(16, min_periods=16).min().to_numpy()
    recent_h4_high = highs_s.rolling(96, min_periods=96).max().to_numpy()
    recent_h4_low = lows_s.rolling(96, min_periods=96).min().to_numpy()

    # ---- H2: prior directional pressure over pre-specified windows ----
    closes_s = pd.Series(closes)

    def prior_return_atr(win):
        prior_close = closes_s.shift(win).to_numpy()
        return (closes - prior_close) / pip / np.where(atr > 0, atr, np.nan)

    def prior_persistence(win):
        # fraction of up bars in the trailing `win` bars before (not incl.) event bar
        diffs = closes_s.diff()
        up = (diffs > 0).astype(float)
        return up.shift(1).rolling(win, min_periods=win).mean().to_numpy()

    ret_1h_atr = prior_return_atr(4)
    ret_4h_atr = prior_return_atr(16)
    ret_8h_atr = prior_return_atr(32)
    pers_1h = prior_persistence(4)
    pers_4h = prior_persistence(16)
    pers_8h = prior_persistence(32)

    # previous session return: prior session's own close-to-close return (last session of same type)
    sess_close = closes_s.groupby(sess_key).last()
    sess_open = closes_s.groupby(sess_key).first()  # approx: first close of session as proxy open
    sess_ret_map = (sess_close - sess_open)
    sess_order = pd.Series(sess_key.to_numpy()).drop_duplicates().reset_index(drop=True)
    prev_sess_ret_by_key = {}
    keys_list = sess_order.tolist()
    for idx_k, k in enumerate(keys_list):
        if idx_k == 0:
            continue
        prev_key = keys_list[idx_k - 1]
        prev_sess_ret_by_key[k] = sess_ret_map.get(prev_key, np.nan)
    prev_session_ret_pips = sess_key.map(prev_sess_ret_by_key).to_numpy() / pip

    # previous day return
    daily_close = df['Close'].resample('1D').last()
    daily_open = df['Open'].resample('1D').first()
    daily_ret = (daily_close - daily_open)
    daily_ret.index = daily_ret.index.date
    prev_day_ret_map = daily_ret.shift(1)
    prev_day_ret_pips = day_index.map(prev_day_ret_map).to_numpy() / pip

    # ---- H3: volatility transition ----
    vol_pctile_pre = pdz.vol_pctile  # already computed on trailing ATR
    event_range_atr = (highs - lows) / pip / np.where(atr > 0, atr, np.nan)
    # realized vol pre/post (std of 1-bar returns), 8-bar windows
    bar_ret = closes_s.diff().to_numpy() / pip
    realized_vol_pre = pd.Series(bar_ret).rolling(8, min_periods=8).std().shift(1).to_numpy()
    realized_vol_post = pd.Series(bar_ret).shift(-8).rolling(8, min_periods=8).std().to_numpy()
    # align post-window: rolling on the shifted series computes trailing std ending at i-8..i,
    # which after shift(-8) represents bars i+1..i+8 -- acceptable approx, no lookahead into i itself
    with np.errstate(divide='ignore', invalid='ignore'):
        vol_expansion_ratio = realized_vol_post / realized_vol_pre
    atr_pctile_series = pd.Series(vol_pctile_pre)
    vol_pctile_post = atr_pctile_series.shift(-8).to_numpy()  # ATR percentile ~8 bars later
    vol_pctile_change = vol_pctile_post - vol_pctile_pre

    # ---- H4: session location (already have pdz.session); Asian thirds ----
    asian_sub = np.array([asian_third(h) for h in hours])

    # ---- H5: range break vs internal move (20-bar trailing range boundary, excl. event bar) ----
    range_low_20 = lows_s.shift(1).rolling(20, min_periods=20).min().to_numpy()
    is_range_break = lows < range_low_20  # event bar's low breaches the established 20-bar low

    # ---- H6: distance from session VWAP ----
    if 'tick_volume' in df.columns:
        vol = df['tick_volume'].to_numpy().astype(float)
    else:
        vol = np.ones(n)
    typical = (highs + lows + closes) / 3.0
    pv = typical * vol
    cum_pv = pd.Series(pv).groupby(sess_key).cumsum().to_numpy()
    cum_vol = pd.Series(vol).groupby(sess_key).cumsum().to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        session_vwap = cum_pv / cum_vol
    dist_from_vwap_atr = (closes - session_vwap) / pip / np.where(atr > 0, atr, np.nan)

    idxs = np.where(down)[0]
    rows = pd.DataFrame({
        'pair': pdz.pair, 'i': idxs, 'time': pdz.idx[idxs],
        'year': pdz.year[idxs], 'dow': pdz.dow[idxs], 'session': pdz.session[idxs],
        'asian_third': asian_sub[idxs],
        'fwd_atr': fwd_atr[idxs],
        'outcome': np.where(fwd_atr[idxs] >= 0, 'REVERSAL', 'CONTINUATION'),
        # H1
        'dist_from_sess_high_atr': (sess_high_so_far[idxs] - closes[idxs]) / pip / atr[idxs],
        'dist_from_sess_low_atr': (closes[idxs] - sess_low_so_far[idxs]) / pip / atr[idxs],
        'pos_in_day_range': (closes[idxs] - day_low_so_far[idxs]) / np.maximum(day_high_so_far[idxs] - day_low_so_far[idxs], 1e-9),
        'pos_in_prevday_range': (closes[idxs] - prev_day_low[idxs]) / np.maximum(prev_day_high[idxs] - prev_day_low[idxs], 1e-9),
        'dist_from_prevday_high_atr': (prev_day_high[idxs] - closes[idxs]) / pip / atr[idxs],
        'dist_from_prevday_low_atr': (closes[idxs] - prev_day_low[idxs]) / pip / atr[idxs],
        'dist_from_recent_h1_high_atr': (recent_h1_high[idxs] - closes[idxs]) / pip / atr[idxs],
        'dist_from_recent_h1_low_atr': (closes[idxs] - recent_h1_low[idxs]) / pip / atr[idxs],
        'dist_from_recent_h4_high_atr': (recent_h4_high[idxs] - closes[idxs]) / pip / atr[idxs],
        'dist_from_recent_h4_low_atr': (closes[idxs] - recent_h4_low[idxs]) / pip / atr[idxs],
        # H2
        'ret_1h_atr': ret_1h_atr[idxs], 'ret_4h_atr': ret_4h_atr[idxs], 'ret_8h_atr': ret_8h_atr[idxs],
        'persistence_1h': pers_1h[idxs], 'persistence_4h': pers_4h[idxs], 'persistence_8h': pers_8h[idxs],
        'prev_session_ret_pips': prev_session_ret_pips[idxs],
        'prev_day_ret_pips': prev_day_ret_pips[idxs],
        # H3
        'atr_pctile_pre': vol_pctile_pre[idxs],
        'event_range_atr': event_range_atr[idxs],
        'realized_vol_pre': realized_vol_pre[idxs], 'realized_vol_post': realized_vol_post[idxs],
        'vol_expansion_ratio': vol_expansion_ratio[idxs],
        'vol_pctile_change': vol_pctile_change[idxs],
        # H5
        'is_range_break': is_range_break[idxs],
        # H6
        'dist_from_vwap_atr': dist_from_vwap_atr[idxs],
    })
    return rows


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS HELPERS
# ═══════════════════════════════════════════════════════════════════════

def compare_groups(events: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    sub = events.dropna(subset=[col])
    rows = []
    for outcome in ['REVERSAL', 'CONTINUATION']:
        s = sub[sub.outcome == outcome][col]
        if len(s) < 20:
            continue
        rows.append(dict(variable=label, outcome=outcome, n=len(s), mean=s.mean(), median=s.median(),
                          std=s.std(), p25=s.quantile(0.25), p75=s.quantile(0.75)))
    out = pd.DataFrame(rows)
    if len(out) == 2:
        rev_mean = out.loc[out.outcome == 'REVERSAL', 'mean'].values[0]
        cont_mean = out.loc[out.outcome == 'CONTINUATION', 'mean'].values[0]
        pooled_std = sub[col].std()
        effect_size = (rev_mean - cont_mean) / pooled_std if pooled_std > 0 else np.nan
        out['effect_size_cohens_d'] = effect_size
    return out


def say_comparison(events: pd.DataFrame, col: str, label: str):
    out = compare_groups(events, col, label)
    if out.empty:
        say(f'  {label}: insufficient data')
        return out
    say(f'  {label}:')
    say('   ' + out.to_string(index=False).replace('\n', '\n   '))
    return out


def per_pair_year_check(events: pd.DataFrame, col: str, label: str):
    """Verify the reversal-vs-continuation mean difference in `col` has the
    same sign across pairs and across 2025 vs 2026 specifically."""
    sub = events.dropna(subset=[col])
    pair_signs = []
    for pair, g in sub.groupby('pair'):
        rev = g[g.outcome == 'REVERSAL'][col]
        cont = g[g.outcome == 'CONTINUATION'][col]
        if len(rev) >= 15 and len(cont) >= 15:
            pair_signs.append(np.sign(rev.mean() - cont.mean()))
    year_signs = {}
    for yr in [2025, 2026]:
        g = sub[sub.year == yr]
        rev = g[g.outcome == 'REVERSAL'][col]
        cont = g[g.outcome == 'CONTINUATION'][col]
        if len(rev) >= 15 and len(cont) >= 15:
            year_signs[yr] = np.sign(rev.mean() - cont.mean())
    if pair_signs:
        agree = max(pair_signs.count(1), pair_signs.count(-1))
        say(f'    Cross-pair sign agreement: {agree}/{len(pair_signs)} pairs agree on direction.')
    if len(year_signs) == 2:
        same = year_signs.get(2025) == year_signs.get(2026)
        say(f'    2025 vs 2026 sign agreement: {"SAME" if same else "DIFFERENT"} '
            f'(2025={year_signs.get(2025)}, 2026={year_signs.get(2026)}).')
    return pair_signs, year_signs


def main():
    say('=' * 90)
    say('PHASE 16 -- DOWN-MOVE REVERSION: MECHANISM RESEARCH (why does it reverse?)')
    say(f'Event (frozen): M15 close-to-close move <= -{THR}xATR(14,66). Outcome (frozen, {HORIZON}-bar/60min):')
    say('  REVERSAL = fwd_atr >= 0.  CONTINUATION = fwd_atr < 0.')
    say('No parameter search. No filter combination. No strategy. AMR untouched.')
    say('=' * 90)

    all_events = []
    for pair in PAIRS:
        try:
            m15 = fetch_m15(pair)
        except Exception as e:
            say(f'{pair}: SKIP ({e})')
            continue
        if len(m15) < 3000:
            say(f'{pair}: SKIP (insufficient data)')
            continue
        pdz = PairData(pair, m15)
        ev = build_events(pdz)
        all_events.append(ev)
        say(f'{pair}: {len(ev)} down-move events '
            f'({(ev.outcome=="REVERSAL").sum()} reversal / {(ev.outcome=="CONTINUATION").sum()} continuation)')

    events = pd.concat(all_events, ignore_index=True)
    say(f'\nTotal events pooled: {len(events)} '
        f'({(events.outcome=="REVERSAL").sum()} reversal / {(events.outcome=="CONTINUATION").sum()} continuation)')

    out_dir = REPO_ROOT / 'data'
    events.to_csv(out_dir / 'phase16_events.csv', index=False)

    say('\n' + '=' * 90)
    say('HYPOTHESIS 1 -- BROADER MARKET LOCATION')
    say('=' * 90)
    h1_cols = ['dist_from_sess_high_atr', 'dist_from_sess_low_atr', 'pos_in_day_range',
               'pos_in_prevday_range', 'dist_from_prevday_high_atr', 'dist_from_prevday_low_atr',
               'dist_from_recent_h1_high_atr', 'dist_from_recent_h1_low_atr',
               'dist_from_recent_h4_high_atr', 'dist_from_recent_h4_low_atr']
    for col in h1_cols:
        say_comparison(events, col, col)
        per_pair_year_check(events, col, col)

    say('\n' + '=' * 90)
    say('HYPOTHESIS 2 -- PRIOR DIRECTIONAL PRESSURE')
    say('=' * 90)
    h2_cols = ['ret_1h_atr', 'ret_4h_atr', 'ret_8h_atr', 'persistence_1h', 'persistence_4h',
               'persistence_8h', 'prev_session_ret_pips', 'prev_day_ret_pips']
    for col in h2_cols:
        say_comparison(events, col, col)
        per_pair_year_check(events, col, col)

    say('\n' + '=' * 90)
    say('HYPOTHESIS 3 -- VOLATILITY TRANSITION')
    say('=' * 90)
    h3_cols = ['atr_pctile_pre', 'event_range_atr', 'realized_vol_pre', 'realized_vol_post',
               'vol_expansion_ratio', 'vol_pctile_change']
    for col in h3_cols:
        say_comparison(events, col, col)
        per_pair_year_check(events, col, col)

    say('\n' + '=' * 90)
    say('HYPOTHESIS 4 -- SESSION LOCATION')
    say('=' * 90)
    sess_tab = events.groupby('session')['outcome'].apply(lambda s: (s == 'REVERSAL').mean()).reindex(
        [ASIAN, LONDON, OVERLAP, NY])
    say('Reversal rate by session:')
    say(sess_tab.to_string())
    asian_events = events[events.session == ASIAN]
    asian_tab = asian_events.groupby('asian_third')['outcome'].apply(lambda s: (s == 'REVERSAL').mean()).reindex(
        ['EARLY_ASIAN', 'MID_ASIAN', 'LATE_ASIAN'])
    asian_n = asian_events.groupby('asian_third').size().reindex(['EARLY_ASIAN', 'MID_ASIAN', 'LATE_ASIAN'])
    say('\nReversal rate within Asian session, by pre-defined third:')
    say(pd.DataFrame({'reversal_rate': asian_tab, 'n': asian_n}).to_string())

    say('\n' + '=' * 90)
    say('HYPOTHESIS 5 -- RANGE BREAK VS INTERNAL MOVE')
    say('=' * 90)
    rb_tab = events.groupby('is_range_break')['outcome'].apply(lambda s: (s == 'REVERSAL').mean())
    rb_n = events.groupby('is_range_break').size()
    say(pd.DataFrame({'reversal_rate': rb_tab, 'n': rb_n}).to_string())
    say('(is_range_break = event bar low breached the trailing 20-bar low, established BEFORE the event bar)')

    say('\n' + '=' * 90)
    say('HYPOTHESIS 6 -- DISTANCE FROM SESSION VWAP')
    say('=' * 90)
    say_comparison(events, 'dist_from_vwap_atr', 'dist_from_vwap_atr')
    per_pair_year_check(events, 'dist_from_vwap_atr', 'dist_from_vwap_atr')

    report_path = REPO_ROOT / 'reports' / 'phase16_mechanism_log.txt'
    report_path.write_text('\n'.join(LOG), encoding='utf-8')
    say(f'\nFull log written to {report_path}')
    say('Full per-event table written to data/phase16_events.csv')


if __name__ == '__main__':
    main()
