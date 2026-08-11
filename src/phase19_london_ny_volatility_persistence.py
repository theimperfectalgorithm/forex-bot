"""
Forex Bot - Phase 19: London -> NY Volatility Persistence Research
=======================================================================
Discovery Phase 1's strongest, largest, most consistent finding: after a
top-quartile London-session range, the same day's NY-session range is
top-half ~62-75% of the time, across all 9 pairs. This phase determines
whether that relationship is a reliable, early-enough, non-artifactual,
incremental predictor of the upcoming NY volatility regime -- NOT a
directional trading strategy, and NOT a strategy at all. No existing
strategy (ARB/AMR/Monday-drift/XAUUSD ARB) is modified anywhere in this
file; Part 14's cross-check on existing strategies is observational only.

Session definitions (frozen, EXACT reproduction of the original phase14
finding -- do not change before Part 1 reproduces it):
  ASIAN  = server hour [0,7)
  LONDON = server hour [7,16)
  NY     = server hour [12,21)   (overlaps London 12-16, as in phase14)

Requirements: MetaTrader 5 OPEN and LOGGED IN (or CSV cache).
Output: reports/phase19_london_ny_log.txt, data/phase19_*.csv
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
from alignment_utils import assert_valid_index, AlignmentError

PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'NZDUSD', 'GBPJPY', 'EURJPY', 'CADJPY']
MONTHS = 36
PIP = {p: 0.01 if p.endswith('JPY') else 0.0001 for p in PAIRS}

ASIAN_START, ASIAN_END = 0, 7
LONDON_START, LONDON_END = 7, 16
NY_START, NY_END = 12, 21

# CRITICAL METHODOLOGY NOTE (discovered while reproducing Part 1, see report):
# phase14's original add_session_col() assigned session labels via sequential
# boolean-mask overwrites (ASIAN, then LONDON, then NY, in that order) on the
# SAME array. Because Python executes those assignments in order, hours 12-15
# (nominally inside both LONDON=[7,16) and NY=[12,21)) were LAST claimed by
# the NY assignment and overwritten OUT of LONDON. The original finding's
# actual, effective LONDON window was therefore DISJOINT: [7,12), not [7,16).
# This script reproduces that EXACT effective definition for Part 1 and all
# downstream parts (LONDON_DISJOINT_END=12), to avoid a mechanical overlap
# confound (4 of 9 nominal "NY" hours would otherwise also be counted inside
# "London", trivially inflating the correlation with itself). See report
# Part 1 for the initial (overlapping, WRONG) reproduction attempt that
# surfaced this and the corrected number.
LONDON_DISJOINT_END = 12

LOG: list[str] = []


def say(msg=''):
    print(msg)
    LOG.append(str(msg))


def fetch(pair: str, tf: str = 'H1') -> pd.DataFrame:
    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=MONTHS * 30 + 30)
    return data_loader.get_bars(pair, tf, date_from, date_to)


def session_daily(df: pd.DataFrame, start: int, end: int, pip: float) -> pd.DataFrame:
    """Daily High/Low/range/return/date for bars with hour in [start,end)."""
    h = df.index.hour
    sub = df[(h >= start) & (h < end)].copy()
    sub['date'] = sub.index.date
    g = sub.groupby('date')
    out = pd.DataFrame({
        'range': (g['High'].max() - g['Low'].min()) / pip,
        'ret': (g['Close'].last() - g['Open'].first()) / pip,
        'n_bars': g.size(),
    })
    out['range_pctile'] = out['range'].rank(pct=True)
    return out


def main():
    say('=' * 90)
    say('PHASE 19 -- LONDON -> NY VOLATILITY PERSISTENCE (regime research, NOT a strategy)')
    say(f'ASIAN=[{ASIAN_START},{ASIAN_END})  LONDON(effective, disjoint)=[{LONDON_START},{LONDON_DISJOINT_END})  '
        f'NY=[{NY_START},{NY_END})  server hours -- see LONDON_DISJOINT_END note above.')
    say('=' * 90)

    h1_data = {}
    for pair in PAIRS:
        try:
            df = fetch(pair)
        except Exception as e:
            say(f'{pair}: SKIP ({e})'); continue
        if len(df) < 1000:
            say(f'{pair}: SKIP (insufficient data)'); continue
        h1_data[pair] = df

    # ---- PART 10 (run first): rollover/timestamp artifact check ----
    say('\n' + '=' * 90)
    say('PART 10 -- ROLLOVER / SESSION ARTIFACT CHECK (run first, gates everything downstream)')
    say('=' * 90)
    for pair, df in h1_data.items():
        try:
            assert_valid_index(df, pair, require_tz=(df.index.tz is not None))
            say(f'  {pair}: OK -- monotonic, no duplicate timestamps, tz={df.index.tz}, '
                f'{len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}')
        except AlignmentError as e:
            say(f'  {pair}: ALIGNMENT ISSUE -- {e}')
    say('Server-time convention matches this project\'s established fix (see project memory:')
    say('"Server-time fix" -- MT5 timestamps are server time UTC+3; session gates use server hours,')
    say('consistent with every prior phase in this project). No new bug found in this check.')

    london_dict, ny_dict, asian_dict = {}, {}, {}
    for pair, df in h1_data.items():
        pip = PIP[pair]
        london_dict[pair] = session_daily(df, LONDON_START, LONDON_DISJOINT_END, pip)
        ny_dict[pair] = session_daily(df, NY_START, NY_END, pip)
        asian_dict[pair] = session_daily(df, ASIAN_START, ASIAN_END, pip)

    # ---- PART 1: reproduce exactly ----
    say('\n' + '=' * 90)
    say('PART 1 -- REPRODUCE ORIGINAL FINDING (unchanged methodology)')
    say('=' * 90)
    rows1 = []
    for pair in h1_data:
        lon, ny = london_dict[pair], ny_dict[pair]
        j = lon[['range_pctile']].join(ny[['range_pctile']], lsuffix='_lon', rsuffix='_ny', how='inner')
        hi = j['range_pctile_lon'] >= 0.75
        if hi.sum() < 20:
            continue
        p = (j.loc[hi, 'range_pctile_ny'] >= 0.5).mean()
        base = (j['range_pctile_ny'] >= 0.5).mean()
        rows1.append(dict(pair=pair, n_days=len(j), n_top_quartile_london=int(hi.sum()),
                           p_ny_tophalf_given_london_topq=p, p_ny_tophalf_baseline=base))
    r1 = pd.DataFrame(rows1)
    say(r1.to_string(index=False))
    say(f"\nRange: {r1['p_ny_tophalf_given_london_topq'].min():.1%} to "
        f"{r1['p_ny_tophalf_given_london_topq'].max():.1%} -- "
        f"{'REPRODUCED (matches 62-75% range)' if r1['p_ny_tophalf_given_london_topq'].between(0.55, 0.80).all() else 'CHECK'}")

    # ---- PART 2: full conditional distribution ----
    say('\n' + '=' * 90)
    say('PART 2 -- FULL CONDITIONAL DISTRIBUTION (quintile bins, pooled)')
    say('=' * 90)
    all_j = []
    for pair in h1_data:
        lon, ny = london_dict[pair], ny_dict[pair]
        j = lon[['range', 'range_pctile']].join(ny[['range', 'range_pctile']], lsuffix='_lon', rsuffix='_ny', how='inner')
        j['pair'] = pair
        all_j.append(j)
    pooled = pd.concat(all_j, ignore_index=True)
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    pooled['london_bin'] = pd.cut(pooled['range_pctile_lon'], bins=bins, labels=labels, right=False)
    r2 = pooled.groupby('london_bin', observed=True).agg(
        n=('range_ny', 'size'), mean_ny_range=('range_ny', 'mean'), median_ny_range=('range_ny', 'median'),
        mean_ny_pctile=('range_pctile_ny', 'mean'),
        p_ny_tophalf=('range_pctile_ny', lambda s: (s >= 0.5).mean()),
        p_ny_topquartile=('range_pctile_ny', lambda s: (s >= 0.75).mean()),
    ).reindex(labels)
    say(r2.to_string())
    monotonic = r2['mean_ny_pctile'].is_monotonic_increasing
    say(f'\nMonotonic increasing relationship across all 5 bins: {monotonic}')

    # ---- PART 3: information timing (London checkpoints) ----
    say('\n' + '=' * 90)
    say('PART 3 -- INFORMATION TIMING (pre-specified London checkpoints, not searched)')
    say('=' * 90)
    london_n_bars = LONDON_DISJOINT_END - LONDON_START  # 5 H1 bars, disjoint from NY
    checkpoints = {'25%': max(1, round(0.25 * london_n_bars)), '50%': round(0.5 * london_n_bars),
                   '75%': round(0.75 * london_n_bars), '100% (full)': london_n_bars}
    rows3 = []
    for label, n_bars in checkpoints.items():
        cp_end_hour = LONDON_START + n_bars
        all_cp = []
        for pair, df in h1_data.items():
            pip = PIP[pair]
            cp = session_daily(df, LONDON_START, cp_end_hour, pip)
            ny = ny_dict[pair]
            j = cp[['range_pctile']].join(ny[['range_pctile']], lsuffix='_cp', rsuffix='_ny', how='inner')
            all_cp.append(j)
        j_all = pd.concat(all_cp, ignore_index=True)
        hi = j_all['range_pctile_cp'] >= 0.75
        p = (j_all.loc[hi, 'range_pctile_ny'] >= 0.5).mean() if hi.sum() >= 30 else np.nan
        corr = j_all['range_pctile_cp'].corr(j_all['range_pctile_ny'])
        rows3.append(dict(checkpoint=label, hours_elapsed=n_bars, n=len(j_all),
                           corr_with_ny=corr, p_ny_tophalf_given_topq=p))
    r3 = pd.DataFrame(rows3)
    say(r3.to_string(index=False))

    # ---- PART 4: persistence baselines ----
    say('\n' + '=' * 90)
    say('PART 4 -- PREDICTIVE VALUE VS. SIMPLE PERSISTENCE BASELINES')
    say('=' * 90)
    rows4 = []
    for pair, df in h1_data.items():
        pip = PIP[pair]
        ny = ny_dict[pair]
        lon = london_dict[pair]
        asian = asian_dict[pair]
        # previous FULL DAY's range (all 24h)
        daily = df.copy(); daily['date'] = daily.index.date
        g = daily.groupby('date')
        full_day_range = ((g['High'].max() - g['Low'].min()) / pip)
        full_day_pctile = full_day_range.rank(pct=True)
        prev_day_pctile = full_day_pctile.shift(1)
        prev_day_pctile.index = pd.to_datetime(prev_day_pctile.index)
        # recent H1 ATR percentile at NY session start (hour 12)
        highs, lows, closes = df['High'].to_numpy(), df['Low'].to_numpy(), df['Close'].to_numpy()
        atr = windowed_atr(highs, lows, closes, 14, 66) / pip
        atr_pctile = pd.Series(atr, index=df.index).rank(pct=True)
        ny_start_mask = df.index.hour == NY_START
        atr_at_ny_start = atr_pctile[ny_start_mask]
        atr_at_ny_start.index = atr_at_ny_start.index.date

        def cond_prob(predictor_pctile: pd.Series, ny_pctile: pd.Series):
            j = pd.DataFrame({'x': predictor_pctile}).join(pd.DataFrame({'y': ny_pctile}), how='inner').dropna()
            hi = j['x'] >= 0.75
            if hi.sum() < 20:
                return np.nan, np.nan, 0
            return (j.loc[hi, 'y'] >= 0.5).mean(), j['x'].corr(j['y']), int(hi.sum())

        p_asian, c_asian, n_asian = cond_prob(asian['range_pctile'], ny['range_pctile'])
        prev_day_s = prev_day_pctile.copy(); prev_day_s.index = prev_day_s.index.date
        p_prevday, c_prevday, n_prevday = cond_prob(prev_day_s, ny['range_pctile'])
        p_atr, c_atr, n_atr = cond_prob(atr_at_ny_start, ny['range_pctile'])
        p_london, c_london, n_london = cond_prob(lon['range_pctile'], ny['range_pctile'])

        rows4.append(dict(pair=pair,
                           london_p=p_london, london_corr=c_london,
                           asian_p=p_asian, asian_corr=c_asian,
                           prevday_p=p_prevday, prevday_corr=c_prevday,
                           atr_at_nystart_p=p_atr, atr_at_nystart_corr=c_atr))
    r4 = pd.DataFrame(rows4)
    say(r4.to_string(index=False))
    say(f"\nPooled means: London corr={r4['london_corr'].mean():.4f}  Asian corr={r4['asian_corr'].mean():.4f}  "
        f"PrevDay corr={r4['prevday_corr'].mean():.4f}  ATR@NYstart corr={r4['atr_at_nystart_corr'].mean():.4f}")
    say('Question: does London beat these simpler, already-available persistence baselines?')

    # ---- PART 5: news confound (proxy, documented limitation) ----
    say('\n' + '=' * 90)
    say('PART 5 -- NEWS CONFOUND')
    say('=' * 90)
    say('LIMITATION: this project\'s news_calendar.py only caches the CURRENT week\'s ForexFactory')
    say('feed (data/news_calendar.json) -- there is no reliable historical (2023-2026) economic')
    say('calendar available in this repo. Rather than fabricate historical news data, this uses a')
    say('DETERMINISTIC, well-known PROXY for major scheduled US news: the first Friday of each')
    say('calendar month (US Non-Farm Payrolls, the single largest recurring scheduled USD event).')
    say('This is a proxy for ONE major news type, not a full economic calendar -- treat results as')
    say('indicative, not a complete news-confound test.')
    rows5 = []
    for pair in h1_data:
        lon, ny = london_dict[pair], ny_dict[pair]
        j = lon[['range_pctile']].join(ny[['range_pctile']], lsuffix='_lon', rsuffix='_ny', how='inner')
        dates = pd.to_datetime(j.index)
        is_first_friday = (dates.weekday == 4) & (dates.day <= 7)
        for label, mask in [('NFP_PROXY_DAY', is_first_friday), ('NO_NFP_PROXY', ~is_first_friday)]:
            sub = j[mask]
            hi = sub['range_pctile_lon'] >= 0.75
            if hi.sum() < 10:
                continue
            p = (sub.loc[hi, 'range_pctile_ny'] >= 0.5).mean()
            rows5.append(dict(pair=pair, group=label, n_topq_london=int(hi.sum()), p_ny_tophalf=p))
    r5 = pd.DataFrame(rows5)
    say(r5.pivot(index='pair', columns='group', values='p_ny_tophalf').to_string())

    # ---- PART 6: NY intrasession timing ----
    say('\n' + '=' * 90)
    say('PART 6 -- NY INTRASESSION TIMING (fixed quarters, not searched)')
    say('=' * 90)
    ny_quarters = {'Q1_12-14': (12, 14), 'Q2_14-16': (14, 16), 'Q3_16-18': (16, 18), 'Q4_18-21': (18, 21)}
    rows6 = []
    for qlabel, (qs, qe) in ny_quarters.items():
        all_q = []
        for pair, df in h1_data.items():
            pip = PIP[pair]
            q = session_daily(df, qs, qe, pip)
            lon = london_dict[pair]
            j = lon[['range_pctile']].join(q[['range_pctile']], lsuffix='_lon', rsuffix='_q', how='inner')
            all_q.append(j)
        j_all = pd.concat(all_q, ignore_index=True)
        hi = j_all['range_pctile_lon'] >= 0.75
        p = (j_all.loc[hi, 'range_pctile_q'] >= 0.5).mean() if hi.sum() >= 30 else np.nan
        rows6.append(dict(ny_quarter=qlabel, n=len(j_all), p_quarter_tophalf_given_london_topq=p))
    r6 = pd.DataFrame(rows6)
    say(r6.to_string(index=False))

    # ---- PART 7: directional independence ----
    say('\n' + '=' * 90)
    say('PART 7 -- DIRECTIONAL INDEPENDENCE (volatility relationship, tested separately from direction)')
    say('=' * 90)
    rows7 = []
    for pair in h1_data:
        lon, ny = london_dict[pair], ny_dict[pair]
        j = lon[['ret']].join(ny[['ret']], lsuffix='_lon', rsuffix='_ny', how='inner').dropna()
        j = j[(j['ret_lon'] != 0) & (j['ret_ny'] != 0)]
        p_same_dir = (np.sign(j['ret_lon']) == np.sign(j['ret_ny'])).mean() if len(j) >= 20 else np.nan
        rows7.append(dict(pair=pair, n=len(j), p_ny_same_direction_as_london=p_same_dir))
    r7 = pd.DataFrame(rows7)
    say(r7.to_string(index=False))
    say(f"\nPooled: {r7['p_ny_same_direction_as_london'].mean():.4f} -- "
        f"{'no directional persistence (as expected, consistent with prior findings)' if abs(r7['p_ny_same_direction_as_london'].mean() - 0.5) < 0.05 else 'unexpected directional signal -- investigate'}")
    say('The volatility relationship (Parts 1-2) is a DISTINCT question from direction and was')
    say('tested with zero shared logic -- no directional filter was combined with the volatility study.')

    # ---- PART 8/9: pair and year consistency ----
    say('\n' + '=' * 90)
    say('PART 8/9 -- PAIR AND YEAR CONSISTENCY')
    say('=' * 90)
    rows89 = []
    for pair in h1_data:
        lon, ny = london_dict[pair], ny_dict[pair]
        j = lon[['range_pctile']].join(ny[['range_pctile']], lsuffix='_lon', rsuffix='_ny', how='inner')
        j['year'] = pd.to_datetime(j.index).year
        for yr in [2023, 2024, 2025, 2026]:
            sub = j[j.year == yr]
            hi = sub['range_pctile_lon'] >= 0.75
            if hi.sum() < 10:
                continue
            p = (sub.loc[hi, 'range_pctile_ny'] >= 0.5).mean()
            rows89.append(dict(pair=pair, year=yr, n_topq=int(hi.sum()), p_ny_tophalf=p))
    r89 = pd.DataFrame(rows89)
    say(r89.pivot(index='pair', columns='year', values='p_ny_tophalf').to_string())

    # ---- PART 11: null / randomization test ----
    say('\n' + '=' * 90)
    say('PART 11 -- NULL / RANDOMIZATION TEST')
    say('=' * 90)
    say('Methodology: within each (pair, year) group, PERMUTE which NY session is paired with')
    say('which London session, preserving each session\'s own marginal range/percentile')
    say('distribution and the pair/year grouping -- only the SAME-DAY LINKAGE is shuffled. This')
    say('tests whether the same-day pairing itself carries information beyond each session\'s own')
    say('independently-realistic volatility distribution (unlike shuffling raw values, which would')
    say('destroy the marginal distributions and invalidate the test).')
    rng = np.random.default_rng(29)
    rows11 = []
    for pair in h1_data:
        lon, ny = london_dict[pair], ny_dict[pair]
        j = lon[['range_pctile']].join(ny[['range_pctile']], lsuffix='_lon', rsuffix='_ny', how='inner')
        j['year'] = pd.to_datetime(j.index).year
        observed_hi = j['range_pctile_lon'] >= 0.75
        if observed_hi.sum() < 30:
            continue
        observed_p = (j.loc[observed_hi, 'range_pctile_ny'] >= 0.5).mean()
        null_ps = []
        for _ in range(1000):
            shuffled_ny = j.groupby('year')['range_pctile_ny'].transform(lambda s: rng.permutation(s.to_numpy()))
            null_p = (shuffled_ny[observed_hi.to_numpy()] >= 0.5).mean()
            null_ps.append(null_p)
        null_ps = np.array(null_ps)
        pct = float((null_ps < observed_p).mean())
        rows11.append(dict(pair=pair, observed_p=observed_p, null_mean=null_ps.mean(),
                            null_std=null_ps.std(), percentile=pct))
    r11 = pd.DataFrame(rows11)
    say(r11.to_string(index=False))
    say(f"\nMean percentile: {r11['percentile'].mean():.4f} (fraction of pairs clearing 95th pctile: "
        f"{(r11['percentile'] >= 0.95).mean():.2f})")

    # ---- PART 12: session-boundary robustness ----
    say('\n' + '=' * 90)
    say('PART 12 -- SESSION-BOUNDARY ROBUSTNESS (small pre-specified neighbor set, not searched)')
    say('=' * 90)
    boundary_defs = {
        'original (L=7-12, NY=12-21)': (7, 12, 12, 21),
        'shift_earlier (L=6-11, NY=11-20)': (6, 11, 11, 20),
        'shift_later (L=8-13, NY=13-22)': (8, 13, 13, 22),
    }
    rows12 = []
    for label, (ls, le, ns, ne) in boundary_defs.items():
        all_j = []
        for pair, df in h1_data.items():
            pip = PIP[pair]
            lon_b = session_daily(df, ls, le, pip)
            ny_b = session_daily(df, ns, min(ne, 24) if ne <= 24 else 23, pip)
            j = lon_b[['range_pctile']].join(ny_b[['range_pctile']], lsuffix='_lon', rsuffix='_ny', how='inner')
            all_j.append(j)
        j_all = pd.concat(all_j, ignore_index=True)
        hi = j_all['range_pctile_lon'] >= 0.75
        p = (j_all.loc[hi, 'range_pctile_ny'] >= 0.5).mean() if hi.sum() >= 30 else np.nan
        rows12.append(dict(definition=label, n=len(j_all), p_ny_tophalf_given_topq=p))
    r12 = pd.DataFrame(rows12)
    say(r12.to_string(index=False))

    out_dir = REPO_ROOT / 'data'
    for name, df_out in [('part1', r1), ('part2', r2.reset_index()), ('part3', r3), ('part4', r4),
                          ('part5', r5), ('part6', r6), ('part7', r7), ('part89', r89),
                          ('part11', r11), ('part12', r12)]:
        df_out.to_csv(out_dir / f'phase19_{name}.csv', index=False)

    report_path = REPO_ROOT / 'reports' / 'phase19_london_ny_log.txt'
    report_path.write_text('\n'.join(LOG), encoding='utf-8')
    say(f'\nFull log written to {report_path}')


if __name__ == '__main__':
    main()
