"""
Forex Bot - Phase 17: Down-Move Volatility Predictability (information-timing)
==================================================================================
Follow-up to phase 16 (mechanism: post-event volatility transition is the
strongest signal, d=-0.233, but that measurement uses bars AFTER the
event -- explanatory, not necessarily usable at entry time). This phase
asks: can the same mechanism be detected EARLY ENOUGH (at or before T0,
the event bar's own close) to have predictive value?

T0 = the close of the M15 bar that confirms the >=1.0 ATR down-move
(frozen event definition, unchanged from phase15/16). Every feature below
is explicitly timestamped and tiered:
  TIER 0 = known strictly BEFORE the event bar opens (data through bar i-1)
  TIER 1 = known AT T0 (uses the event bar's own OHLC, still <= T0)
  TIER 2 = known only after the FIRST post-event candle (bar i+1)
  TIER 3 = known only after MULTIPLE future candles (bars i+2 onward)
Only Tier 0/1 findings can support a trading hypothesis that enters at T0.
Tier 2/3 findings explain the mechanism but are NOT claimed as predictive.

No strategy is built or optimized here. No existing strategy (ARB/AMR/
Monday-drift/XAUUSD ARB) is read, imported, or modified.

Data limitation (documented, not worked around): core/data_loader.py only
supports M15/H1/H4 -- there is no M1/M5 feed available, so Part 2's
"first 5-minute interval" cannot be tested and is explicitly skipped
rather than approximated.

Requirements: MetaTrader 5 OPEN and LOGGED IN (or CSV cache).
Output: reports/phase17_predictability_log.txt, data/phase17_events.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy_matrix_backtest import REPO_ROOT
from phase15_downmove_reversion_baseline import PAIRS, ASIAN, LONDON, OVERLAP, NY, fetch_m15
from phase16_downmove_mechanism import PairData, THR, HORIZON, asian_third

LOG: list[str] = []


def say(msg=''):
    print(msg)
    LOG.append(str(msg))


# ═══════════════════════════════════════════════════════════════════════
# BUILD EVENT TABLE WITH EXPLICIT TIER LABELS
# ═══════════════════════════════════════════════════════════════════════

def build_tiered_events(pdz: PairData) -> pd.DataFrame:
    n = len(pdz.closes)
    closes, highs, lows, opens, atr, pip = (pdz.closes, pdz.highs, pdz.lows,
                                             pdz.df['Open'].to_numpy(), pdz.atr, pdz.pip)

    fwd_close = np.roll(closes, -HORIZON); fwd_close[-HORIZON:] = np.nan
    fwd_atr = (fwd_close - closes) / pip / np.where(atr > 0, atr, np.nan)
    down = (pdz.move_atr <= -THR) & ~np.isnan(fwd_atr)
    idxs = np.where(down)[0]
    idxs = idxs[(idxs >= 40) & (idxs < n - HORIZON - 1)]  # need i-1..i-32 history and i+1..i+HORIZON future
    if len(idxs) == 0:
        return pd.DataFrame()

    bar_ret = pd.Series(closes).diff().to_numpy() / pip  # bar_ret[i] = close[i]-close[i-1], pips

    def realized_vol(end_excl_i, window):
        """std of bar_ret over [end_excl_i-window .. end_excl_i-1] -- STRICTLY before end_excl_i."""
        out = np.full(n, np.nan)
        s = pd.Series(bar_ret)
        roll = s.rolling(window).std()
        # roll[j] = std of bar_ret[j-window+1..j]; we want std ending at end_excl_i-1 = j
        return roll.shift(0).to_numpy()  # caller passes already-shifted index

    roll_std_series = pd.Series(bar_ret).rolling(8).std()   # roll_std[j] = std(bar_ret[j-7..j])
    roll_std_4 = pd.Series(bar_ret).rolling(4).std()
    roll_std_2 = pd.Series(bar_ret).rolling(2).std()

    session_high_so_far = pd.Series(highs).groupby(
        pd.Series(pdz.session) + '_' + pd.Series(pdz.idx.date).astype(str)).cummax().to_numpy()
    session_low_so_far = pd.Series(lows).groupby(
        pd.Series(pdz.session) + '_' + pd.Series(pdz.idx.date).astype(str)).cummin().to_numpy()

    if 'tick_volume' in pdz.df.columns:
        vol = pdz.df['tick_volume'].to_numpy().astype(float)
    else:
        vol = np.ones(n)
    typical = (highs + lows + closes) / 3.0
    sess_key = pd.Series(pdz.session) + '_' + pd.Series(pdz.idx.date).astype(str)
    cum_pv = pd.Series(typical * vol).groupby(sess_key).cumsum().to_numpy()
    cum_vol = pd.Series(vol).groupby(sess_key).cumsum().to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        session_vwap = cum_pv / cum_vol

    atr_pctile_full = pdz.vol_pctile  # atr[i]'s own percentile (Tier 1, uses bar i)
    atr_series = pd.Series(atr)

    rows = []
    for i in idxs:
        a_prev = atr[i - 1]          # Tier 0 denominator: last bar known BEFORE the event bar opens
        a_event = atr[i]             # Tier 1 denominator: includes the event bar itself
        if a_prev <= 0 or np.isnan(a_prev) or a_event <= 0 or np.isnan(a_event):
            continue

        # ---- TIER 0: strictly before bar i (data through bar i-1 only) ----
        atr_pctile_pre_T0 = atr_series.iloc[:i].rank(pct=True).iloc[-1] if i >= 30 else np.nan
        realized_vol_8_T0 = roll_std_series.iloc[i - 1]     # std of bar_ret[i-8..i-1]
        realized_vol_4_T0 = roll_std_4.iloc[i - 1]
        realized_vol_2_T0 = roll_std_2.iloc[i - 1]
        vol_older = roll_std_series.iloc[i - 5] if i >= 5 else np.nan   # std of bar_ret[i-13..i-6] approx
        with np.errstate(divide='ignore', invalid='ignore'):
            vol_slope = (realized_vol_8_T0 - vol_older) / vol_older if vol_older and not np.isnan(vol_older) else np.nan
            vol_acceleration = vol_slope  # same quantity, kept as a separate named field per Part 3 spec
        prev_candle_range_atr_T0 = (highs[i - 1] - lows[i - 1]) / pip / a_prev
        # session volatility state: session-so-far realized vol (up to bar i-1) vs that session's
        # historical average realized vol for the SAME pair (computed once, below, outside the loop)

        # ---- TIER 1: known at T0 (uses the event bar i's own OHLC) ----
        event_range_atr = (highs[i] - lows[i]) / pip / a_prev   # normalized by a_prev to avoid embedding the event bar in its own denominator
        event_body = abs(closes[i] - opens[i])
        event_range_raw = max(highs[i] - lows[i], 1e-9)
        body_pct = event_body / event_range_raw
        upper_wick_pct = (highs[i] - max(closes[i], opens[i])) / event_range_raw
        lower_wick_pct = (min(closes[i], opens[i]) - lows[i]) / event_range_raw
        close_location = (closes[i] - lows[i]) / event_range_raw   # 0 = closed at low, 1 = closed at high
        move_atr_T1 = pdz.move_atr[i]
        dist_from_sess_high_T1 = (session_high_so_far[i] - closes[i]) / pip / a_prev
        dist_from_vwap_T1 = (closes[i] - session_vwap[i]) / pip / a_prev if not np.isnan(session_vwap[i]) else np.nan
        atr_pctile_at_T0 = atr_pctile_full[i]   # Tier 1 version (includes event bar's own TR)

        # ---- TIER 2: first post-event candle (bar i+1) only ----
        b1_range = (highs[i + 1] - lows[i + 1]) / pip / a_prev
        b1_body = abs(closes[i + 1] - opens[i + 1]) / pip / a_prev
        b1_direction = np.sign(closes[i + 1] - opens[i + 1])
        b1_close_loc = (closes[i + 1] - lows[i + 1]) / max(highs[i + 1] - lows[i + 1], 1e-9)
        b1_vol_vs_event = b1_range / max(event_range_atr, 1e-9)
        b1_vol_vs_pre = b1_range / max(realized_vol_8_T0 / a_prev, 1e-9) if realized_vol_8_T0 and not np.isnan(realized_vol_8_T0) else np.nan

        # ---- TIER 3: multiple future candles (bars i+1..i+HORIZON), same as phase16 ----
        post_window = bar_ret[i + 1:i + 1 + HORIZON]
        realized_vol_post_T3 = np.std(post_window) if len(post_window) == HORIZON else np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            vol_expansion_ratio_T3 = realized_vol_post_T3 / (realized_vol_8_T0 if realized_vol_8_T0 else np.nan)
        atr_pctile_post_T3 = atr_pctile_full[min(i + HORIZON, n - 1)]
        vol_pctile_change_T3 = atr_pctile_post_T3 - atr_pctile_at_T0

        # horizon sweep (Part 2): realized vol over first 1/2/3/4 bars after i (15/30/45/60 min)
        horizon_vols = {}
        for h in [1, 2, 3, 4]:
            w = bar_ret[i + 1:i + 1 + h]
            horizon_vols[f'vol_first_{h*15}m'] = np.std(w) if len(w) == h and h > 1 else (abs(w[0]) if len(w) == 1 else np.nan)

        rows.append(dict(
            pair=pdz.pair, i=i, time=pdz.idx[i], year=pdz.year[i], dow=pdz.dow[i], session=pdz.session[i],
            asian_third=asian_third(pdz.idx.hour[i]), fwd_atr=fwd_atr[i],
            outcome=('REVERSAL' if fwd_atr[i] >= 0 else 'CONTINUATION'),
            # Tier 0
            atr_pctile_pre_T0=atr_pctile_pre_T0, realized_vol_8_T0=realized_vol_8_T0,
            realized_vol_4_T0=realized_vol_4_T0, realized_vol_2_T0=realized_vol_2_T0,
            vol_slope_T0=vol_slope, vol_acceleration_T0=vol_acceleration,
            prev_candle_range_atr_T0=prev_candle_range_atr_T0,
            # Tier 1
            event_range_atr_T1=event_range_atr, body_pct_T1=body_pct,
            upper_wick_pct_T1=upper_wick_pct, lower_wick_pct_T1=lower_wick_pct,
            close_location_T1=close_location, move_atr_T1=move_atr_T1,
            dist_from_sess_high_T1=dist_from_sess_high_T1, dist_from_vwap_T1=dist_from_vwap_T1,
            atr_pctile_at_T0_T1=atr_pctile_at_T0,
            # Tier 2
            b1_range_T2=b1_range, b1_body_T2=b1_body, b1_direction_T2=b1_direction,
            b1_close_loc_T2=b1_close_loc, b1_vol_vs_event_T2=b1_vol_vs_event, b1_vol_vs_pre_T2=b1_vol_vs_pre,
            # Tier 3
            realized_vol_post_T3=realized_vol_post_T3, vol_expansion_ratio_T3=vol_expansion_ratio_T3,
            vol_pctile_change_T3=vol_pctile_change_T3,
            **horizon_vols,
        ))
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# ANALYSIS HELPERS (mirrors phase16's style for continuity)
# ═══════════════════════════════════════════════════════════════════════

def compare(events: pd.DataFrame, col: str) -> dict:
    sub = events.dropna(subset=[col])
    rev = sub[sub.outcome == 'REVERSAL'][col]
    cont = sub[sub.outcome == 'CONTINUATION'][col]
    if len(rev) < 20 or len(cont) < 20:
        return dict(variable=col, n_rev=len(rev), n_cont=len(cont), effect_size=np.nan)
    pooled_std = sub[col].std()
    d = (rev.mean() - cont.mean()) / pooled_std if pooled_std > 0 else np.nan
    return dict(variable=col, n_rev=len(rev), n_cont=len(cont),
                rev_mean=rev.mean(), cont_mean=cont.mean(), effect_size=d)


def pair_year_agreement(events: pd.DataFrame, col: str):
    sub = events.dropna(subset=[col])
    signs = []
    for pair, g in sub.groupby('pair'):
        rev, cont = g[g.outcome == 'REVERSAL'][col], g[g.outcome == 'CONTINUATION'][col]
        if len(rev) >= 15 and len(cont) >= 15:
            signs.append(np.sign(rev.mean() - cont.mean()))
    agree = max(signs.count(1.0), signs.count(-1.0)) if signs else 0
    year_signs = {}
    for yr in [2023, 2024, 2025, 2026]:
        g = sub[sub.year == yr]
        rev, cont = g[g.outcome == 'REVERSAL'][col], g[g.outcome == 'CONTINUATION'][col]
        if len(rev) >= 15 and len(cont) >= 15:
            year_signs[yr] = np.sign(rev.mean() - cont.mean())
    return agree, len(signs), year_signs


def report_variable(events: pd.DataFrame, col: str, tier: str):
    r = compare(events, col)
    agree, npairs, year_signs = pair_year_agreement(events, col)
    say(f"  [{tier}] {col}: n_rev={r.get('n_rev')} n_cont={r.get('n_cont')} "
        f"rev_mean={r.get('rev_mean', np.nan):.4f} cont_mean={r.get('cont_mean', np.nan):.4f} "
        f"d={r.get('effect_size', np.nan):+.4f} | pairs {agree}/{npairs} agree | years: {year_signs}")
    return dict(tier=tier, **r, pair_agree=f'{agree}/{npairs}', year_signs=str(year_signs))


def main():
    say('=' * 90)
    say('PHASE 17 -- DOWN-MOVE VOLATILITY PREDICTABILITY (information-timing)')
    say('T0 = close of the M15 bar confirming the >=1.0xATR down-move (frozen, unchanged).')
    say('TIER 0 = strictly before bar i. TIER 1 = at T0 (uses bar i itself). TIER 2 = first')
    say('post-event candle (bar i+1). TIER 3 = multiple future candles. Only Tier 0/1 can')
    say('support a trading hypothesis that enters AT T0. No strategy, no optimization.')
    say('Data limitation: core/data_loader.py supports M15/H1/H4 only -- no M1/M5 feed, so')
    say('the "first 5-minute interval" requested in Part 2 cannot be tested and is skipped')
    say('rather than approximated. Finest available granularity is M15.')
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
        ev = build_tiered_events(pdz)
        all_events.append(ev)
        say(f'{pair}: {len(ev)} events')

    events = pd.concat(all_events, ignore_index=True)
    say(f'\nTotal events pooled: {len(events)}')
    out_dir = REPO_ROOT / 'data'
    events.to_csv(out_dir / 'phase17_events.csv', index=False)

    say('\n' + '=' * 90)
    say('PART 2 -- EARLY VOLATILITY RESPONSE: when does separation first appear?')
    say('=' * 90)
    for h in [1, 2, 3, 4]:
        col = f'vol_first_{h*15}m'
        report_variable(events, col, f'TIER3 (uses bar i+1..i+{h})')

    say('\n' + '=' * 90)
    say('PART 3 -- PRE-EVENT INFORMATION (TIER 0: strictly before the event bar)')
    say('=' * 90)
    t0_cols = ['atr_pctile_pre_T0', 'realized_vol_8_T0', 'realized_vol_4_T0', 'realized_vol_2_T0',
               'vol_slope_T0', 'prev_candle_range_atr_T0']
    for col in t0_cols:
        report_variable(events, col, 'TIER 0')

    say('\n' + '=' * 90)
    say('PART 4 -- EVENT-CANDLE STRUCTURE (TIER 1: uses the event bar itself, still <= T0)')
    say('=' * 90)
    t1_cols = ['event_range_atr_T1', 'body_pct_T1', 'upper_wick_pct_T1', 'lower_wick_pct_T1',
               'close_location_T1', 'move_atr_T1', 'dist_from_sess_high_T1', 'dist_from_vwap_T1',
               'atr_pctile_at_T0_T1']
    for col in t1_cols:
        report_variable(events, col, 'TIER 1')

    say('\n' + '=' * 90)
    say('PART 5 -- FIRST POST-EVENT CANDLE (TIER 2: known only after bar i+1)')
    say('=' * 90)
    t2_cols = ['b1_range_T2', 'b1_body_T2', 'b1_direction_T2', 'b1_close_loc_T2',
               'b1_vol_vs_event_T2', 'b1_vol_vs_pre_T2']
    for col in t2_cols:
        report_variable(events, col, 'TIER 2')

    say('\n' + '=' * 90)
    say('TIER 3 (for reference / continuity with phase 16 -- NOT usable for T0 entry)')
    say('=' * 90)
    t3_cols = ['realized_vol_post_T3', 'vol_expansion_ratio_T3', 'vol_pctile_change_T3']
    for col in t3_cols:
        report_variable(events, col, 'TIER 3')

    say('\n' + '=' * 90)
    say('PART 9 -- ASIAN ROLLOVER ARTIFACT SENSITIVITY (exclude hours 0-2)')
    say('=' * 90)
    clean = events[events.asian_third != 'EARLY_ASIAN']
    say(f'Excluding EARLY_ASIAN (hours 0-2): {len(events) - len(clean)} events removed, {len(clean)} remain.')
    say('Re-running the Tier 3 volatility-transition finding on the reduced sample:')
    for col in t3_cols:
        report_variable(clean, col, 'TIER 3 (ex-EARLY_ASIAN)')

    say('\n' + '=' * 90)
    say('PART 10 -- NULL / RANDOMIZATION TEST on the earliest Tier-3 separation point')
    say('=' * 90)
    say('Methodology: bootstrap (2000 draws) a random sample of non-event bars matched on')
    say('pre-event volatility tercile (atr_pctile_pre_T0), same size as the event group, and')
    say('compute the mean of vol_first_15m for each draw -- builds a null distribution of what')
    say('a volatility-matched random bar produces at the earliest horizon tested. Report the')
    say('percentile of the REAL reversal-vs-continuation gap within that null.')
    rng = np.random.default_rng(23)
    null_rows = []
    for pair, g in events.groupby('pair'):
        g2 = g.dropna(subset=['vol_first_15m', 'atr_pctile_pre_T0'])
        if len(g2) < 100:
            continue
        tercile = np.where(g2['atr_pctile_pre_T0'] <= 1/3, 'LOW',
                   np.where(g2['atr_pctile_pre_T0'] >= 2/3, 'HIGH', 'MID'))
        g2 = g2.assign(tercile=tercile)
        observed_gap = (g2[g2.outcome == 'REVERSAL']['vol_first_15m'].mean() -
                         g2[g2.outcome == 'CONTINUATION']['vol_first_15m'].mean())
        pools = {t: g2[g2.tercile == t]['vol_first_15m'].to_numpy() for t in ['LOW', 'MID', 'HIGH']}
        n_rev = (g2.outcome == 'REVERSAL').sum()
        rev_tercile_counts = g2[g2.outcome == 'REVERSAL']['tercile'].value_counts()
        cont_tercile_counts = g2[g2.outcome == 'CONTINUATION']['tercile'].value_counts()
        null_gaps = []
        for _ in range(2000):
            rev_sample, cont_sample = [], []
            for t, cnt in rev_tercile_counts.items():
                if len(pools.get(t, [])) > 0:
                    rev_sample.append(rng.choice(pools[t], size=int(cnt), replace=True))
            for t, cnt in cont_tercile_counts.items():
                if len(pools.get(t, [])) > 0:
                    cont_sample.append(rng.choice(pools[t], size=int(cnt), replace=True))
            if rev_sample and cont_sample:
                null_gaps.append(np.concatenate(rev_sample).mean() - np.concatenate(cont_sample).mean())
        null_gaps = np.array(null_gaps)
        pct = float((null_gaps < observed_gap).mean()) if len(null_gaps) else np.nan
        null_rows.append(dict(pair=pair, observed_gap=observed_gap, null_mean=null_gaps.mean(),
                               null_std=null_gaps.std(), percentile=pct))
    null_df = pd.DataFrame(null_rows)
    say(null_df.to_string(index=False))
    say(f"\nMean percentile: {null_df['percentile'].mean():.4f}")

    report_path = REPO_ROOT / 'reports' / 'phase17_predictability_log.txt'
    report_path.write_text('\n'.join(LOG), encoding='utf-8')
    say(f'\nFull log written to {report_path}')
    say('Full per-event table written to data/phase17_events.csv')


if __name__ == '__main__':
    main()
