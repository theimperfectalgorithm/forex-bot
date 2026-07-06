"""
Forex Bot - Phase 8: Monday-drift strategy validation (GBPUSD)
===============================================================
Phase 7 found GBPUSD rises on Mondays (+0.13%/day IS t=+3.33, +0.21%/day
OOS t=+4.00, positive every year 2023-2026, day-specific). This converts
the raw anomaly into a BOUNDED strategy and validates it with the
standard harness criteria:

  Entry : BUY GBPUSD at the close of Monday's 00:00-01:00 H1 bar.
  SL    : k x ATR20d (rolling 20-day daily ATR, in pips) -- bounded risk.
  TP    : either 1.0 x ATR20d, or none (rides to the time exit).
  Exit  : 21:00 UTC Monday (flat before the daily rollover; also keeps
          the trade clear of most Mon-evening positioning).
  One trade per week; risk 0.25%. EURUSD variant included as a weaker-
  prior control.

Output: data/phase8_monday_results.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategy_matrix_backtest import (
    SPREAD_PIPS, MONTHS, IS_MONTHS, run_sim, metrics, REPO_ROOT,
)
from phase3_session_structure_search import fetch_tf

OUT_CSV = REPO_ROOT / 'data' / 'phase8_monday_results.csv'
PIP = 0.0001


def daily_atr_pips(h1: pd.DataFrame, period: int = 20) -> pd.Series:
    d = pd.DataFrame({
        'High': h1['High'].resample('1D').max(),
        'Low': h1['Low'].resample('1D').min(),
        'Close': h1['Close'].resample('1D').last(),
    }).dropna()
    pc = d['Close'].shift(1)
    tr = pd.concat([d['High'] - d['Low'], (d['High'] - pc).abs(),
                    (d['Low'] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean() / PIP


def signals_monday(h1, sl_mult, tp_mult):
    """BUY at the close of Monday's 00:00 H1 bar. sl/tp in ATR20d units;
    tp_mult=None -> effectively no TP (time exit does the work)."""
    atr_d = daily_atr_pips(h1)
    hours = h1.index.hour
    wday = h1.index.weekday
    dates = h1.index.normalize()
    out = []
    for i in range(len(h1)):
        if wday[i] != 0 or hours[i] != 0:
            continue
        a = atr_d.asof(dates[i] - pd.Timedelta(days=1))
        if pd.isna(a) or a <= 0:
            continue
        sl = sl_mult * a
        tp = (tp_mult * a) if tp_mult else 5.0 * a   # 5xATR ~ unreachable
        out.append((i, 'BUY', sl, tp))
    return out


def main():
    now = pd.Timestamp.now(tz='UTC')
    oos_start = now - pd.DateOffset(months=MONTHS - IS_MONTHS)
    is_start = now - pd.DateOffset(months=MONTHS)

    rows = []
    for pair in ('GBPUSD', 'EURUSD'):
        h1 = fetch_tf(pair, 'H1')
        for sl_mult in (0.75, 1.25):
            for tp_mult in (1.0, None):
                cands = signals_monday(h1, sl_mult, tp_mult)
                tdf, eq = run_sim(h1, cands, PIP, SPREAD_PIPS[pair], 0.0025,
                                  time_exit_hour=21)
                mi = metrics(tdf, eq, is_start, oos_start)
                mo = metrics(tdf, eq, oos_start, now)
                is_pass = (mi.get('trades', 0) >= 30 and mi.get('pf', 0) > 1.3
                           and mi.get('max_dd_pct', 99) < 8
                           and mi.get('prof_months_pct', 0) >= 60)
                passed = (is_pass and mo.get('trades', 0) >= 10
                          and mo.get('pnl', 0) > 0
                          and mo.get('max_dd_pct', 99) < 8)
                lbl = f"monday(sl{sl_mult},tp{tp_mult or 'none'})"
                rows.append({'run': lbl, 'pair': pair, **{
                    'is_trades': mi.get('trades', 0), 'is_pf': mi.get('pf', 0),
                    'is_pnl': mi.get('pnl', 0), 'is_dd': mi.get('max_dd_pct', 0),
                    'is_pm': mi.get('prof_months_pct', 0),
                    'oos_trades': mo.get('trades', 0), 'oos_pf': mo.get('pf', 0),
                    'oos_pnl': mo.get('pnl', 0), 'oos_dd': mo.get('max_dd_pct', 0),
                    'IS_PASS': is_pass, 'PASS': passed}})
                r = rows[-1]
                tag = 'PASS' if passed else ('is-ok' if is_pass else 'fail ')
                print(f"[{tag}] {lbl:<24} {pair}  "
                      f"IS: n={r['is_trades']:>3} PF={r['is_pf']:<5} "
                      f"DD={r['is_dd']:>5.2f}% pm={r['is_pm']:>5.1f}%  "
                      f"OOS: n={r['oos_trades']:>3} PF={r['oos_pf']:<5} "
                      f"P&L=${r['oos_pnl']:>+8,.0f} DD={r['oos_dd']:.2f}%")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    print(f"Full passes: {int(df['PASS'].sum())} / {len(df)}")


if __name__ == '__main__':
    main()
