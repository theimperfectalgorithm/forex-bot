"""
Forex Bot - Phase 10b: cross-asset JPY momentum robustness probe
===================================================================
Phase 10 found ONE full pass out of 52 configurations: cross-asset JPY
momentum (USDJPY proxy) on NZDJPY at check_hour=14, thr_atr=1.25 (IS PF
1.38/DD 7.66%/64%pm, OOS PF 1.15/+$5,218/DD 4.56%). But 3 of its 4
immediate neighbors in the original grid had similar profit factors
(1.17-1.39) and POSITIVE-leaning OOS, just failing the DD<8% cut --
that pattern (threading between two criteria, not comfortably clearing
both) is the same fragility signature that made MDS-GBPUSD's phase-1
"pass" turn out to be noise (see PROJECT_REPORT.md section 4). This
probe checks: is the NZDJPY result parameter-stable (real), or a lucky
cell (noise)? Also checks whether the mechanic generalizes to the other
5 JPY crosses at all, the way ARB/AMR replicating onto CADJPY was
treated as the strongest genuineness evidence for THOSE edges.

Selection is on IS only (as always); OOS is reported for confirmation,
never used to pick the winning cell.

Requirements: MetaTrader 5 OPEN and LOGGED IN.
Output: data/phase10b_results.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy_matrix_backtest import MONTHS, IS_MONTHS, REPO_ROOT
from phase10_jpy_london_ny import (
    ALL_PAIRS, SIGNAL_PAIRS, NY_END, fetch_h1, build_usdjpy_proxy,
    signals_xmomentum, evaluate,
)

OUT_CSV = REPO_ROOT / 'data' / 'phase10b_results.csv'


def main():
    now = pd.Timestamp.now(tz='UTC')
    oos_start = now - pd.DateOffset(months=MONTHS - IS_MONTHS)
    is_start  = now - pd.DateOffset(months=MONTHS)

    print('======== PHASE 10b: xmo robustness probe ========\n')
    data = {}
    for p in ALL_PAIRS:
        print(f'Fetching {p} H1 ...')
        data[p] = fetch_h1(p)
    print()

    usdjpy_move, usdjpy_atr = build_usdjpy_proxy(data['USDJPY'], 0.01)

    rows = []
    for pair in SIGNAL_PAIRS:
        for check_hour in (13, 14, 15):
            for thr_atr in (1.0, 1.25, 1.5):
                label = f'xmo(ch{check_hour},thr{thr_atr})'
                cands = signals_xmomentum(
                    data[pair], 0.01, 2.0, usdjpy_move, usdjpy_atr,
                    check_hour, thr_atr, sl_atr=1.5, tp_atr=2.5)
                evaluate(label, pair, data[pair], cands, NY_END,
                         is_start, oos_start, now, rows)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f'\nSaved: {OUT_CSV}')
    print(f"Full passes: {int(df['PASS'].sum())} / {len(df)}")
    if df['PASS'].any():
        print(df[df['PASS']].to_string(index=False))
    print('\n--- NZDJPY neighborhood (the phase-10 hit) ---')
    print(df[df['pair'] == 'NZDJPY'].to_string(index=False))


if __name__ == '__main__':
    main()
