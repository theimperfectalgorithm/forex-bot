"""
Forex Bot - Phase 19b: Existing-Strategy Observational Regime Check (Part 14)
=================================================================================
OBSERVATIONAL ONLY. Does NOT modify ARB, AMR, Monday Drift, or XAUUSD ARB.
Reconstructs each of the 8 live demo strategies from their exact frozen
live parameters (same reconstruction as phase13_nzdjpy_portfolio_analysis.py)
and checks whether each strategy's own trade P&L differs on days when that
day's London session (using phase19's corrected, disjoint 7-12 definition)
was top-quartile range vs. not.

CRITICAL TIMING CAVEAT (stated here and in the report): ARB entries occur
at server hours 7-8 (the very start of London), AMR entries occur at
server hours 0-4 (before London begins), and Monday Drift enters at
server hour 0 Monday (before London begins). None of these 8 strategies'
ENTRY decisions could use "full London session range" as an input -- that
information does not exist yet at their entry time. This script therefore
does NOT test or imply a usable real-time filter; it only asks the
retrospective/observational question of whether completed trade P&L
happens to co-vary with that day's later-revealed regime label.

Requirements: MetaTrader 5 OPEN and LOGGED IN (or CSV cache).
Output: reports/phase19b_regime_check_log.txt
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
from strategy_matrix_backtest import run_sim, REPO_ROOT
from phase2_meanrev_arb_search import signals_arb_p
from phase3b_amr_jpy_refine import signals_amr_v
from phase8_monday_validation import signals_monday

MONTHS = 36
LOG: list[str] = []


def say(msg=''):
    print(msg)
    LOG.append(str(msg))


def fetch(pair, tf):
    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=MONTHS * 30 + 30)
    return data_loader.get_bars(pair, tf, date_from, date_to)


def pip_of(pair):
    return 0.01 if (pair.endswith('JPY') or pair == 'XAUUSD') else 0.0001


def london_topquartile_flags(pair: str, h1: pd.DataFrame) -> pd.Series:
    """Same corrected, disjoint definition as phase19: London=[7,12)."""
    pip = pip_of(pair)
    h = h1.index.hour
    sub = h1[(h >= 7) & (h < 12)].copy()
    sub['date'] = sub.index.date
    g = sub.groupby('date')
    rng = (g['High'].max() - g['Low'].min()) / pip
    pctile = rng.rank(pct=True)
    return (pctile >= 0.75)   # boolean Series indexed by date


def regime_check(label, pair, tdf, topq_by_date):
    if tdf.empty:
        say(f'  {label}: no trades reconstructed -- skipped')
        return
    tdf = tdf.copy()
    tdf['date'] = pd.to_datetime(tdf['entry_time']).dt.date
    tdf['topq'] = tdf['date'].map(topq_by_date)
    tdf = tdf.dropna(subset=['topq'])
    tdf['topq'] = tdf['topq'].astype(bool)
    hi = tdf[tdf.topq]
    lo = tdf[~tdf.topq]
    say(f'  {label}: n_total={len(tdf)}  n_topqLondon_days={len(hi)}  n_other_days={len(lo)}')
    if len(hi) >= 15 and len(lo) >= 15:
        say(f'    mean pnl | topqLondon={hi["pnl"].mean():+.2f}  other={lo["pnl"].mean():+.2f}  '
            f'win_rate | topqLondon={( hi["pnl"]>0).mean():.3f}  other={(lo["pnl"]>0).mean():.3f}')
    else:
        say('    sample too small for a reliable split (need >=15 trades in each bucket) -- not reported as a conclusion')


def main():
    say('=' * 90)
    say('PHASE 19b -- EXISTING-STRATEGY OBSERVATIONAL REGIME CHECK (Part 14, OBSERVATIONAL ONLY)')
    say('=' * 90)
    say('TIMING CAVEAT: ARB enters at server hours 7-8 (London START), AMR enters at server')
    say('hours 0-4 (BEFORE London), Monday Drift enters at server hour 0 Monday (BEFORE London).')
    say('None of these entries could use "completed London session range" as an input -- it does')
    say('not exist yet. This is a retrospective/observational correlation only, NOT a usable')
    say('real-time filter, and is not being proposed as one.')
    say('No strategy logic was read for the purpose of modifying it, and nothing here changes')
    say('any live/demo configuration.')

    # ---- ARB family: GBPJPY, CADJPY (tp_mult=2.0, use_h4=False), XAUUSD (tp_mult=1.5, min_range=30) ----
    say('\n-- ARB family --')
    for pair, tp_mult, use_h4, min_range in [('GBPJPY', 2.0, False, 10), ('CADJPY', 2.0, False, 10),
                                              ('XAUUSD', 1.5, False, 30)]:
        try:
            h1 = fetch(pair, 'H1')
            h4 = fetch(pair, 'H4')
        except Exception as e:
            say(f'  {pair}_ARB: SKIP ({e})'); continue
        pip = pip_of(pair)
        cands = signals_arb_p(h1, h4, pip, tp_mult, use_h4, min_range=min_range)
        tdf, _ = run_sim(h1, cands, pip, 2.0, 0.005)
        topq = london_topquartile_flags(pair, h1)
        regime_check(f'{pair}_ARB', pair, tdf, topq)

    # ---- AMR family: GBPJPY(2.5,1.25,4) EURJPY(2.0,1.5,6) AUDJPY(2.0,1.5,4) CADJPY(2.0,1.5,4) ----
    say('\n-- AMR family --')
    for pair, z_thr, sl_mult, end_hour in [('GBPJPY', 2.5, 1.25, 4), ('EURJPY', 2.0, 1.5, 6),
                                            ('AUDJPY', 2.0, 1.5, 4), ('CADJPY', 2.0, 1.5, 4)]:
        try:
            m15 = fetch(pair, 'M15')
            h1 = fetch(pair, 'H1')
        except Exception as e:
            say(f'  {pair}_AMR: SKIP ({e})'); continue
        pip = pip_of(pair)
        cands = signals_amr_v(m15, pip, 2.0, z_thr, sl_mult, end_hour)
        tdf, _ = run_sim(m15, cands, pip, 2.0, 0.0025)
        topq = london_topquartile_flags(pair, h1)
        regime_check(f'{pair}_AMR', pair, tdf, topq)

    # ---- Monday Drift: GBPUSD ----
    say('\n-- Monday Drift --')
    pair = 'GBPUSD'
    try:
        h1 = fetch(pair, 'H1')
        cands = signals_monday(h1, 1.25, 1.0)
        tdf, _ = run_sim(h1, cands, pip_of(pair), 1.2, 0.0025, time_exit_hour=21)
        topq = london_topquartile_flags(pair, h1)
        regime_check(f'{pair}_MONDAY', pair, tdf, topq)
    except Exception as e:
        say(f'  {pair}_MONDAY: SKIP ({e})')

    report_path = REPO_ROOT / 'reports' / 'phase19b_regime_check_log.txt'
    report_path.write_text('\n'.join(LOG), encoding='utf-8')
    say(f'\nFull log written to {report_path}')


if __name__ == '__main__':
    main()
