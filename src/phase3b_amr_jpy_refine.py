"""
Forex Bot - Phase 3b: AMR-JPY refinement (in-sample selection only)
====================================================================
Phase 3 found Asian-hours M15 mean reversion is the one family that is
consistently profitable on the JPY crosses (all 6 variants OOS-positive,
5/6 IS-positive) but below the PF > 1.3 in-sample bar (IS PF 1.07-1.12).

This script runs a small refinement grid on GBPJPY / EURJPY / AUDJPY:
    sl_mult      in {1.0, 1.25, 1.5}   (tighter stops -- MR losers trend)
    entry window in {00-04, 00-06}     (deep-Tokyo only vs full Asian)
    z threshold  in {2.0, 2.5}
Selection is by IN-SAMPLE profit factor among variants meeting the IS
criteria; the winner's OOS stats are then reported for confirmation.

METHOD CAVEAT (disclosed): the FAMILY itself was noticed partly because
of its phase-3 OOS strength, so the OOS window here is not pristine.
Whatever passes must still be treated as a forward-test candidate on
demo, not an activation candidate.

Output: data/phase3b_results.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy_matrix_backtest import (
    SPREAD_PIPS, RISK_PCT_DEFAULT, MONTHS, IS_MONTHS,
    CRIT_PF, CRIT_MAX_DD, CRIT_PROF_MONTHS,
    run_sim, metrics, REPO_ROOT,
)
from phase3_session_structure_search import fetch_tf

OUT_CSV = REPO_ROOT / 'data' / 'phase3b_results.csv'
PAIRS = ['GBPJPY', 'EURJPY', 'AUDJPY']


def signals_amr_v(m15, pip, spread, z_thr, sl_mult, end_hour):
    closes = m15['Close'].to_numpy()
    s = pd.Series(closes)
    sma20 = s.rolling(20).mean().to_numpy()
    std20 = s.rolling(20).std().to_numpy()
    hours = m15.index.hour
    min_tp = max(3.0, 2.5 * spread)
    out = []
    for i in range(20, len(m15)):
        if hours[i] >= end_hour:
            continue
        if np.isnan(std20[i]) or std20[i] <= 0:
            continue
        z = (closes[i] - sma20[i]) / std20[i]
        if z <= -z_thr:
            tp_pips = (sma20[i] - closes[i]) / pip
            if tp_pips >= min_tp:
                out.append((i, 'BUY', tp_pips * sl_mult, tp_pips))
        elif z >= z_thr:
            tp_pips = (closes[i] - sma20[i]) / pip
            if tp_pips >= min_tp:
                out.append((i, 'SELL', tp_pips * sl_mult, tp_pips))
    return out


def main():
    now = pd.Timestamp.now(tz='UTC')
    oos_start = now - pd.DateOffset(months=MONTHS - IS_MONTHS)
    is_start  = now - pd.DateOffset(months=MONTHS)

    rows = []
    for pair in PAIRS:
        print(f'Fetching {pair} M15 ...')
        m15 = fetch_tf(pair, 'M15')
        pip = 0.01
        for z_thr in (2.0, 2.5):
            for sl_mult in (1.0, 1.25, 1.5):
                for end_hour in (4, 6):
                    label = f'amr(z{z_thr},sl{sl_mult},h<{end_hour})'
                    cands = signals_amr_v(m15, pip, SPREAD_PIPS[pair],
                                          z_thr, sl_mult, end_hour)
                    tdf, eq = run_sim(m15, cands, pip, SPREAD_PIPS[pair],
                                      RISK_PCT_DEFAULT, time_exit_hour=7)
                    m_is  = metrics(tdf, eq, is_start, oos_start)
                    m_oos = metrics(tdf, eq, oos_start, now)
                    is_pass = (
                        m_is.get('trades', 0) >= 30
                        and m_is.get('pf', 0) > CRIT_PF
                        and m_is.get('max_dd_pct', 99) < CRIT_MAX_DD
                        and m_is.get('prof_months_pct', 0) >= CRIT_PROF_MONTHS
                    )
                    rows.append({
                        'run': label, 'pair': pair,
                        'is_trades': m_is.get('trades', 0),
                        'is_pf': m_is.get('pf', 0),
                        'is_pnl': m_is.get('pnl', 0),
                        'is_dd': m_is.get('max_dd_pct', 0),
                        'is_prof_months': m_is.get('prof_months_pct', 0),
                        'oos_trades': m_oos.get('trades', 0),
                        'oos_pf': m_oos.get('pf', 0),
                        'oos_pnl': m_oos.get('pnl', 0),
                        'oos_dd': m_oos.get('max_dd_pct', 0),
                        'IS_PASS': is_pass,
                    })
                    r = rows[-1]
                    tag = 'is-ok' if is_pass else 'fail '
                    print(f"[{tag}] {label:<26} {pair}  "
                          f"IS: n={r['is_trades']:>4} PF={r['is_pf']:<5} "
                          f"DD={r['is_dd']:>5.2f}% pm={r['is_prof_months']:>5.1f}%  "
                          f"OOS: n={r['oos_trades']:>4} PF={r['oos_pf']:<5} "
                          f"P&L=${r['oos_pnl']:>+9,.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f'\nSaved: {OUT_CSV}')

    print('\n--- In-sample selection (best IS PF per pair) ---')
    for pair in PAIRS:
        sub = df[df['pair'] == pair].sort_values('is_pf', ascending=False)
        best = sub.iloc[0]
        verdict = 'meets IS criteria' if best['IS_PASS'] else 'still below IS bar'
        print(f"{pair}: {best['run']}  IS PF={best['is_pf']} "
              f"DD={best['is_dd']}% pm={best['is_prof_months']}%  "
              f"-> {verdict}; OOS PF={best['oos_pf']} "
              f"P&L=${best['oos_pnl']:+,.2f}")


if __name__ == '__main__':
    main()
