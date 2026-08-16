"""
Phase 30B -- Non-JPY diversification research: calendar/day-of-week drift
screen across the candidate universe (EURUSD, GBPUSD, AUDUSD, USDCAD,
USDCHF, XAUUSD).

HYPOTHESIS (declared before evaluating results, per B5 discovery standard):
GBPUSD Monday Drift (already live, phase 7/8, OOS PF 2.929) demonstrates
that a day-of-week open-to-close drift effect exists and is tradeable on at
least one major pair. This screen tests whether a SIMILAR mechanism
(calendar/drift family, per B4 -- explicitly not a copy of AMR's
mean-reversion) exists on the other days of the week and/or other pairs in
the non-JPY candidate universe. This is the same methodology class already
proven to work once in this project (phase 7's calendar screen), applied
to unexplored (day, pair) cells -- not a new, unrelated strategy family
invented for this task.

METHODOLOGY: H1 data pulled live via this session's MT5 connection
(MetaQuotes-Demo broker feed -- NOT the 5ers broker's own price history;
documented explicitly as a limitation, since this session has no access to
5ers historical data). For each (pair, day-of-week) cell: enter long (and
separately, short) at that day's first H1 bar open (00:00 server-equivalent
per this broker's timestamps), exit at a fixed holding horizon, using the
exact same chronological IS (first ~24mo) / OOS (remaining ~last 12-20mo)
split convention as every other phase in this project. NO shuffling.

COST MODEL: a conservative flat spread assumption per pair (documented
per-pair below), applied as an immediate cost against every trade -- not
optimized, not tuned per cell.

MULTIPLE-TESTING NOTE: this screen tests 6 pairs x 5 weekdays x 2
directions = 60 cells. This is EXPLORATORY, not confirmatory -- any cell
that appears promising must be treated as a single held-out hypothesis to
re-test on fresh data before being called confirmatory (see the research
report's multiple-testing section). No cell's parameters are tuned after
seeing its own result.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import MetaTrader5 as mt5

REPO = Path(__file__).parent.parent
OUT = REPO / 'reports'

CANDIDATE_PAIRS = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDCAD', 'USDCHF', 'XAUUSD']
# Conservative flat per-trade spread cost assumption (round-trip, in price
# terms), documented per pair -- not tuned, deliberately on the wide side of
# typical retail spreads for these majors.
SPREAD_COST = {
    'EURUSD': 0.00015, 'GBPUSD': 0.00020, 'AUDUSD': 0.00018,
    'USDCAD': 0.00020, 'USDCHF': 0.00020, 'XAUUSD': 0.35,
}
HOLD_DAYS = 1  # single trading-day drift, matching Monday Drift's own design
DATA_START = datetime(2023, 1, 1)
DATA_END = datetime(2026, 8, 14)
IS_END = datetime(2025, 1, 1, tzinfo=timezone.utc)  # ~first 24 months = IS/discovery; rest = OOS


def pull_daily(symbol: str) -> pd.DataFrame:
    if not mt5.initialize():
        raise RuntimeError("MT5 initialize() failed -- cannot pull candidate data")
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_D1, DATA_START, DATA_END)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No D1 data returned for {symbol} -- STOP per data-gap policy")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df['dow'] = df['time'].dt.day_name()
    return df[['time', 'dow', 'open', 'high', 'low', 'close']]


def drift_cell(df: pd.DataFrame, day: str, direction: str, cost: float, period_df: pd.DataFrame) -> dict:
    """Long/short the open-to-close move of every bar matching `day`,
    within period_df's date range. R defined relative to the day's own
    ATR(14) at entry (computed on the full series to avoid a lookback gap,
    then filtered to the period) so trades are risk-normalized like every
    other strategy in this project."""
    sub = period_df[period_df['dow'] == day].copy()
    if len(sub) == 0:
        return None
    raw_move = (sub['close'] - sub['open']) if direction == 'LONG' else (sub['open'] - sub['close'])
    net_move = raw_move - cost
    atr = df['high'].sub(df['low']).rolling(14).mean()
    sub_atr = atr.reindex(sub.index)
    r_multiple = (net_move / sub_atr).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r_multiple) < 5:
        return None
    wins = r_multiple[r_multiple > 0]
    losses = r_multiple[r_multiple < 0]
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else np.nan
    mean_r = r_multiple.mean()
    std_r = r_multiple.std()
    t_stat = (mean_r / (std_r / np.sqrt(len(r_multiple)))) if std_r > 0 else np.nan
    return {
        'trades': len(r_multiple), 'win_rate_pct': round((r_multiple > 0).mean() * 100, 1),
        'pf': round(pf, 3) if pf == pf else None, 'mean_R': round(mean_r, 4),
        't_stat': round(t_stat, 2) if t_stat == t_stat else None,
    }


def main():
    registry_rows = []
    exp_id = 121

    for symbol in CANDIDATE_PAIRS:
        df = pull_daily(symbol)
        is_df = df[df['time'] < IS_END]
        oos_df = df[df['time'] >= IS_END]
        cost = SPREAD_COST[symbol]
        print(f"\n=== {symbol}  (IS: {len(is_df)} bars, OOS: {len(oos_df)} bars) ===")

        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            for direction in ['LONG', 'SHORT']:
                is_res = drift_cell(df, day, direction, cost, is_df)
                oos_res = drift_cell(df, day, direction, cost, oos_df)
                if is_res is None or oos_res is None:
                    continue
                print(f"  {day:9s} {direction:5s}  IS: n={is_res['trades']:2d} t={is_res['t_stat']:>6} "
                      f"PF={is_res['pf']}   OOS: n={oos_res['trades']:2d} t={oos_res['t_stat']:>6} PF={oos_res['pf']}")
                registry_rows.append({
                    'experiment_id': f'EXP-{exp_id}',
                    'instrument': symbol, 'strategy_family': 'calendar_drift',
                    'hypothesis': f'{day} {direction.lower()} open-to-close drift, analogous to the validated GBPUSD Monday Drift mechanism',
                    'timeframe': 'D1', 'session': f'{day} full session',
                    'train_period': f'{DATA_START.date()} to {IS_END.date()}',
                    'validation_period': 'N/A -- single IS/OOS split, no separate validation fold (small-universe exploratory screen)',
                    'oos_period': f'{IS_END.date()} to {DATA_END.date()}',
                    'trade_count': oos_res['trades'],
                    'oos_pf': oos_res['pf'], 'oos_expectancy': oos_res['mean_R'],
                    'is_t_stat': is_res['t_stat'], 'oos_t_stat': oos_res['t_stat'],
                    'is_pf': is_res['pf'], 'is_trades': is_res['trades'],
                })
                exp_id += 1

    reg = pd.DataFrame(registry_rows)
    reg.to_csv(OUT / '_scratch_calendar_screen.csv', index=False)

    # Screening bar: both IS and OOS t-stat magnitude >= 2.0 (roughly p<0.05
    # two-sided), same sign, AND OOS PF > 1.0 -- mirrors the bar phase 7 used
    # to first flag GBPUSD Monday before its dedicated phase-8 validation.
    reg['passes_screen'] = (
        (reg['is_t_stat'].abs() >= 2.0) & (reg['oos_t_stat'].abs() >= 2.0) &
        (np.sign(reg['is_t_stat']) == np.sign(reg['oos_t_stat'])) &
        (reg['oos_pf'] > 1.0)
    )
    survivors = reg[reg['passes_screen']]
    print(f"\n=== SCREEN RESULT: {len(reg)} cells tested, {len(survivors)} survive the IS+OOS t>=2.0 same-sign bar ===")
    print(survivors.to_string() if len(survivors) else "(none)")

    reg.to_csv(OUT / '_scratch_calendar_screen_final.csv', index=False)
    summary = {'cells_tested': len(reg), 'survivors': len(survivors),
               'survivor_cells': survivors[['instrument', 'hypothesis']].to_dict(orient='records') if len(survivors) else []}
    print(json.dumps(summary, indent=2, default=str))
    with open(OUT / '_phase30_screen_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == '__main__':
    main()
