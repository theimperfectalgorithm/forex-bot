"""
Forex Bot - Phase 13b: CRITICAL BUG FIX -- USDJPY/NZDJPY bar misalignment
==============================================================================
Phase 13's Part 1 alignment check found that build_usdjpy_proxy() /
signals_xmomentum() index the USDJPY proxy array by BAR POSITION, not
by TIMESTAMP. NZDJPY and USDJPY H1 series differ in length (a 6-bar
NZDJPY-only stretch on 2023-12-25, two USDJPY-only bars on 2024-12-25
and 2025-01-01) -- once the FIRST length divergence occurs, every
subsequent positional index is permanently offset between the two
series. Measured: 16,287 of 19,352 overlapping positions (84%) have
MISMATCHED timestamps between the two arrays from 2023-12-25 onward --
essentially the entire dataset after the first ~5 months.

This means: for the vast majority of the frozen candidate's history,
usdjpy_move[i] (intended to be "USDJPY's move as of the SAME calendar
hour as NZDJPY bar i") was actually reading USDJPY's move from a
DIFFERENT hour, off by a roughly constant few-hour offset. This is a
DATA-PLUMBING BUG, not a strategy-rule change: the frozen rule is
"compare against USDJPY's move at the same time" -- fixing the
implementation to actually do that is a bug fix, not an optimization,
exactly analogous to this project's own prior "server-time" bug fix
(commit e75d680) where a similar implementation/intent mismatch
silently broke several live strategies.

This script rebuilds the USDJPY proxy using TIMESTAMP-based alignment
(pandas .reindex(), not positional array indexing) and re-runs the
frozen candidate's core validation (IS/OOS, permutation test, CADJPY
replication) to determine whether the original finding survives
correction. Same frozen parameters throughout (check_hour=15,
thr_atr=1.25, sl_atr=1.5, tp_atr=2.5) -- nothing about the STRATEGY
RULE changes, only the correctness of the USDJPY input it reads.

Requirements: MetaTrader 5 OPEN and LOGGED IN.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy_matrix_backtest import (
    MONTHS, IS_MONTHS, RISK_PCT_DEFAULT, windowed_atr, run_sim, metrics,
    REPO_ROOT,
)
from phase10_jpy_london_ny import fetch_h1, LONDON_START, NY_END
from phase12_nzdjpy_validation_gate import (
    CHECK_HOUR, THR_ATR, SL_ATR, TP_ATR, TIME_EXIT, SPREAD_BASELINE, PIP,
    permutation_test,
)
from research_ledger import log_experiment, next_experiment_id

REPORT = REPO_ROOT / 'reports' / 'phase13b_alignment_fix_report.md'
log_rows = []
def say(s=''):
    print(s)
    log_rows.append(str(s))


def build_usdjpy_proxy_aligned(usd: pd.DataFrame, pip: float) -> pd.Series:
    """TIMESTAMP-indexed version of build_usdjpy_proxy(). Returns a
    pandas Series of 'move_pips' indexed by USDJPY's own bar timestamps
    -- callers must .reindex() this onto the TRADED pair's index (never
    assume positional correspondence between two different symbols'
    arrays again)."""
    opens = usd['Open']; closes = usd['Close']
    hours = usd.index.hour
    dates = usd.index.normalize()
    day_open = pd.Series(np.where(hours == LONDON_START, opens.to_numpy(), np.nan),
                         index=usd.index).groupby(dates).transform(
                             lambda s: s.ffill())
    move = (closes - day_open) / pip
    return move


def signals_xmomentum_aligned(h1: pd.DataFrame, pip: float, spread: float,
                              usdjpy_move_ts: pd.Series, check_hour: int,
                              thr_atr: float, sl_atr: float, tp_atr: float) -> list:
    """Same rule as the frozen signals_xmomentum, but usdjpy_move is
    joined onto h1's index BY TIMESTAMP (reindex), never by position."""
    aligned_move = usdjpy_move_ts.reindex(h1.index)
    highs = h1['High'].to_numpy(); lows = h1['Low'].to_numpy()
    closes = h1['Close'].to_numpy()
    atr = windowed_atr(highs, lows, closes, 14, 66) / pip
    hours = h1.index.hour
    move_arr = aligned_move.to_numpy()

    # day-ATR-at-London-open, also timestamp-safe (built on h1's own array,
    # no cross-symbol join needed for this part)
    dates = h1.index.normalize()
    day_atr = pd.Series(np.where(hours == LONDON_START, atr, np.nan),
                        index=h1.index).groupby(dates).transform(lambda s: s.ffill()).to_numpy()

    min_tp = max(3.0, 2.0 * spread)
    out = []
    for i in range(66, len(h1)):
        if hours[i] != check_hour:
            continue
        mv = move_arr[i]
        a0 = day_atr[i]
        if np.isnan(mv) or np.isnan(a0) or a0 <= 0 or np.isnan(atr[i]) or atr[i] <= 0:
            continue
        if abs(mv) < thr_atr * a0:
            continue
        sig = 'BUY' if mv > 0 else 'SELL'
        sl_pips, tp_pips = sl_atr * atr[i], tp_atr * atr[i]
        if tp_pips >= min_tp:
            out.append((i, sig, sl_pips, tp_pips))
    return out


def main():
    now = pd.Timestamp.now(tz='UTC')
    oos_start = now - pd.DateOffset(months=MONTHS - IS_MONTHS)
    is_start  = now - pd.DateOffset(months=MONTHS)

    say('=' * 78)
    say('PHASE 13b: ALIGNMENT BUG FIX + RE-VERIFICATION')
    say('=' * 78)

    nzd = fetch_h1('NZDJPY')
    usd = fetch_h1('USDJPY')
    cad = fetch_h1('CADJPY')

    say(f'\nNZDJPY bars: {len(nzd)}   USDJPY bars: {len(usd)}   CADJPY bars: {len(cad)}')
    say('Rebuilding USDJPY proxy with TIMESTAMP-based alignment (pandas '
       'reindex, not positional array indexing) ...')

    usd_move_ts = build_usdjpy_proxy_aligned(usd, 0.01)

    # sanity: confirm the aligned version has NO positional-mismatch
    # possibility by construction (it's a reindex, always timestamp-correct)
    reindexed_onto_nzd = usd_move_ts.reindex(nzd.index)
    n_missing = reindexed_onto_nzd.isna().sum()
    say(f'  Reindexed onto NZDJPY''s {len(nzd)} timestamps: '
       f'{n_missing} bars have no USDJPY match (correctly NaN -> skipped, '
       f'vs. the old bug which would have silently used a WRONG bar''s '
       f'value there instead of skipping)')

    # ---- corrected NZDJPY signal + validation ----
    say('\n' + '-' * 78)
    say('CORRECTED NZDJPY candidate (frozen params, timestamp-aligned proxy)')
    say('-' * 78)
    cands_fixed = signals_xmomentum_aligned(nzd, PIP, SPREAD_BASELINE, usd_move_ts,
                                            CHECK_HOUR, THR_ATR, SL_ATR, TP_ATR)
    say(f'  Signal count: {len(cands_fixed)}  (was 506 under the buggy alignment)')
    t_fix, e_fix = run_sim(nzd, cands_fixed, PIP, SPREAD_BASELINE, RISK_PCT_DEFAULT,
                           time_exit_hour=TIME_EXIT)
    mi_fix = metrics(t_fix, e_fix, is_start, oos_start)
    mo_fix = metrics(t_fix, e_fix, oos_start, now)
    say(f"  IS:  n={mi_fix.get('trades',0):>3} PF={mi_fix.get('pf',0):<6} "
       f"DD={mi_fix.get('max_dd_pct',0):>5.2f}% pm={mi_fix.get('prof_months_pct',0):>5.1f}%  "
       f"P&L=${mi_fix.get('pnl',0):>+9,.2f}")
    say(f"  OOS: n={mo_fix.get('trades',0):>3} PF={mo_fix.get('pf',0):<6} "
       f"P&L=${mo_fix.get('pnl',0):>+9,.2f}  DD={mo_fix.get('max_dd_pct',0):.2f}%")

    perm_fix = permutation_test(nzd, cands_fixed, PIP, SPREAD_BASELINE, n_perm=1000, seed=11)
    say(f"  Permutation test: real PF={perm_fix['real_pf']}, beats "
       f"{perm_fix['real_pf_percentile_vs_null']:.1f}% of {perm_fix['n_perm']} "
       f"random-direction shuffles (null mean PF={perm_fix['null_pf_mean']})")

    say('\n  COMPARISON -- buggy (phase 12) vs corrected (this script):')
    say(f"    IS PF:   1.49 (buggy)  ->  {mi_fix.get('pf',0)} (corrected)")
    say(f"    OOS PF:  1.20 (buggy)  ->  {mo_fix.get('pf',0)} (corrected)")
    say(f"    OOS P&L: +$7,731 (buggy)  ->  ${mo_fix.get('pnl',0):+,.2f} (corrected)")
    say(f"    Permutation percentile: 100.0% (buggy)  ->  "
       f"{perm_fix['real_pf_percentile_vs_null']:.1f}% (corrected)")

    # ---- corrected CADJPY replication ----
    say('\n' + '-' * 78)
    say('CORRECTED CADJPY replication (same frozen params, timestamp-aligned)')
    say('-' * 78)
    cad_cands_fixed = signals_xmomentum_aligned(cad, PIP, SPREAD_BASELINE, usd_move_ts,
                                                CHECK_HOUR, THR_ATR, SL_ATR, TP_ATR)
    t_cadf, e_cadf = run_sim(cad, cad_cands_fixed, PIP, SPREAD_BASELINE, RISK_PCT_DEFAULT,
                             time_exit_hour=TIME_EXIT)
    mi_cadf = metrics(t_cadf, e_cadf, is_start, oos_start)
    mo_cadf = metrics(t_cadf, e_cadf, oos_start, now)
    say(f"  IS:  n={mi_cadf.get('trades',0):>3} PF={mi_cadf.get('pf',0):<6} "
       f"DD={mi_cadf.get('max_dd_pct',0):>5.2f}%  P&L=${mi_cadf.get('pnl',0):>+9,.2f}")
    say(f"  OOS: n={mo_cadf.get('trades',0):>3} PF={mo_cadf.get('pf',0):<6} "
       f"P&L=${mo_cadf.get('pnl',0):>+9,.2f}")

    # ---- verdict ----
    say('\n' + '=' * 78)
    say('VERDICT')
    say('=' * 78)
    is_ok = (mi_fix.get('trades',0) >= 30 and mi_fix.get('pf',0) > 1.3
            and mi_fix.get('max_dd_pct',99) < 8 and mi_fix.get('prof_months_pct',0) >= 60)
    oos_ok = mo_fix.get('trades',0) >= 10 and mo_fix.get('pnl',0) > 0 and mo_fix.get('max_dd_pct',99) < 8
    perm_ok = perm_fix['real_pf_percentile_vs_null'] >= 90
    survives = is_ok and oos_ok and perm_ok
    say(f'  Corrected result {"SURVIVES" if survives else "DOES NOT SURVIVE"} '
       f'the alignment fix at the standard IS/OOS/permutation bar.')
    if not survives:
        say('  This means the original phase-10/10b/12 finding was, to a '
           'material degree, an artifact of comparing NZDJPY against the '
           'WRONG USDJPY bar for ~84% of the dataset -- not a real edge. '
           'The frozen NZDJPY candidate should be REJECTED, and phases 3-10 '
           'of the originally-requested portfolio/factor analysis (which '
           'assume a real signal worth analyzing) are moot and were NOT run '
           'further on the corrected signal.')
    else:
        say('  The signal survives correction -- reduced in magnitude from '
           'the buggy version but still real. The Part 3-10 portfolio/factor '
           'analysis should be RE-RUN using this corrected signal before any '
           'of phase 13''s numbers are trusted.')

    eid = next_experiment_id()
    log_experiment(eid, family='nzdjpy_alignment_bugfix',
                   hypothesis='CRITICAL BUG: usdjpy proxy was positionally '
                             'indexed, not timestamp-aligned -- 84% of bars '
                             'mismatched from 2023-12-25 onward',
                   pair='NZDJPY', session=f'server_hour_{CHECK_HOUR}',
                   dataset_start=nzd.index[0].date(), dataset_end=nzd.index[-1].date(),
                   is_metrics=mi_fix, oos_metrics=mo_fix,
                   decision=('INVESTIGATE' if survives else 'REJECT'),
                   status=('PROMISING BUT INSUFFICIENT' if survives else 'FAILED'),
                   notes=(f'Corrected IS PF={mi_fix.get("pf",0)} (was 1.49 buggy), '
                         f'OOS PF={mo_fix.get("pf",0)} (was 1.20 buggy), '
                         f'permutation percentile={perm_fix["real_pf_percentile_vs_null"]:.1f}% '
                         f'(was 100.0% buggy). CADJPY corrected: IS PF='
                         f'{mi_cadf.get("pf",0)}, OOS PF={mo_cadf.get("pf",0)}. '
                         f'{"Signal survives fix." if survives else "Signal DOES NOT survive fix -- original finding was largely a data-alignment artifact."} '
                         f'SUPERSEDES EXP-030/031/032/033.'))
    say(f'\nLogged as {eid} (supersedes EXP-030/031/032/033)')

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n'.join(log_rows), encoding='utf-8')
    say(f'\nSaved: {REPORT}')
    return survives


if __name__ == '__main__':
    main()
