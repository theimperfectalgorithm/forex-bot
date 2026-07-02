"""
Forex Bot -- NY Open Breakout Backtest + Optimization (EURUSD, H1)

Strategy under test (src/../strategies/breakout/ny_open_breakout.py):
  1. London range = High/Low of H1 bars in a London window (grid: 08:00-12:00,
     08:00-13:00, or 07:00-13:00 UTC).
  2. H4 trend filter: Close > SMA50 > SMA200 (bullish) or Close < SMA50 <
     SMA200 (bearish), AND |SMA50-SMA200| >= a minimum pip threshold (grid:
     20/30/40 pips) so flat H4 charts don't count as trending. Evaluated
     using only H4 bars that have FULLY CLOSED before NY open (13:00 UTC) --
     H4 bars are 4h candles aligned to 00/04/08/12/16/20 UTC, so the 12:00
     bar is still forming at 13:00 and is correctly excluded.
  3. NY-open breakout: scan the 13:00 and 14:00 H1 bars (approximating the
     13:00-14:30 NY-open window at H1 resolution) for the first close
     beyond the London range, in the H4 trend direction only.
  4. SL = SL_MULTIPLIER x London range beyond entry (grid: 1.0x/1.5x/2.0x).
     TP = 2x the SL distance (2:1 reward:risk).
  5. EOD close ~17:30 UTC if neither SL nor TP is hit. H1 data can't
     express a 17:30 mid-bar exit exactly, so the 17:00-18:00 bar's
     Open/Close midpoint is used as the closest honest approximation.

Data: core/data_loader.get_bars() -- CSV-backed on this Mac (see
scripts/export_historical_data.py), identical MT5 path on the VPS.

Protocol (train/forward split is a hard boundary -- no parameter in the
grid is ever chosen by looking at forward-test results):
  1. Grid search (3 range windows x 3 SL multipliers x 3 H4 thresholds =
     27 configs) on TRAIN data only (2020-01 to 2022-06).
  2. Apply pass criteria, rank ALL 27 by profit factor desc / max DD asc.
  3. Take the top 3 by that ranking and run them ONCE on FORWARD data
     (2022-07 to 2024-12) -- untouched until this point.
  4. Walk-forward: split the full 2020-2024 range into 6 equal calendar
     periods and report P&L/win-rate per period for the top 3 configs.
  5. Print an honest verdict.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from core import data_loader

# ── SETTINGS ────────────────────────────────────────────────────────────────

PAIR             = 'EURUSD'
PIP_SIZE         = 0.0001
LOT_SIZE         = 0.1
PIP_VALUE        = 1.00        # USD per pip at 0.1 lots on EURUSD
START_BALANCE    = 10_000.00

H4_SMA_FAST      = 50
H4_SMA_SLOW      = 200
H4_BAR_HOURS     = 4

NY_ENTRY_HOURS   = [13, 14]    # H1 bars scanned for the breakout (13:00-14:30 approx.)
NY_ENTRY_START_HOUR = 13
EOD_HOUR         = 17          # 17:00-18:00 H1 bar approximates the 17:30 EOD close
MIN_LONDON_RANGE_PIPS = 5.0    # skip degenerate near-zero ranges (not part of the grid)
RISK_REWARD      = 2.0         # TP = RR x SL distance

RANGE_WINDOWS = {
    '08:00-12:00': (8, 12),
    '08:00-13:00': (8, 13),
    '07:00-13:00': (7, 13),
}
SL_MULTIPLIERS = [1.0, 1.5, 2.0]
H4_THRESHOLDS  = [20, 30, 40]

DATA_START   = datetime(2020, 1, 1, tzinfo=timezone.utc)
DATA_END     = datetime(2024, 12, 31, 23, 59, tzinfo=timezone.utc)
TRAIN_START  = datetime(2020, 1, 1, tzinfo=timezone.utc)
TRAIN_END    = datetime(2022, 6, 30, 23, 59, tzinfo=timezone.utc)
FORWARD_START = datetime(2022, 7, 1, tzinfo=timezone.utc)
FORWARD_END   = datetime(2024, 12, 31, 23, 59, tzinfo=timezone.utc)

PASS_WIN_RATE_MIN     = 40.0
PASS_WIN_RATE_MAX     = 50.0
PASS_MAX_DD           = 4.0
PASS_MIN_PROFIT_FACTOR = 1.2
PASS_MIN_TRADES       = 60
PASS_MIN_PROFITABLE_MONTHS_PCT = 60.0


# ── 1. DATA FETCH ───────────────────────────────────────────────────────────

def fetch_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Fetching {PAIR} H1 + H4  {DATA_START.date()} to {DATA_END.date()} ...")
    h1 = data_loader.get_bars(PAIR, 'H1', DATA_START, DATA_END)
    h4 = data_loader.get_bars(PAIR, 'H4', DATA_START, DATA_END)
    print(f"  H1: {len(h1):,} bars  ({h1.index[0]} to {h1.index[-1]})")
    print(f"  H4: {len(h4):,} bars  ({h4.index[0]} to {h4.index[-1]})\n")
    return h1, h4


# ── 2. H4 TREND (computed once, threshold applied at lookup time) ───────────

def compute_h4_trend(h4: pd.DataFrame) -> pd.DataFrame:
    df = h4.copy()
    df['SMA50']  = df['Close'].rolling(H4_SMA_FAST).mean()
    df['SMA200'] = df['Close'].rolling(H4_SMA_SLOW).mean()
    df = df.dropna(subset=['SMA200']).copy()

    df['diff_pips'] = (df['SMA50'] - df['SMA200']).abs() / PIP_SIZE
    df['trend'] = 0
    bull = (df['SMA50'] > df['SMA200']) & (df['Close'] > df['SMA50'])
    bear = (df['SMA50'] < df['SMA200']) & (df['Close'] < df['SMA50'])
    df.loc[bull, 'trend'] = 1
    df.loc[bear, 'trend'] = -1

    # A bar "closes" H4_BAR_HOURS after it opens -- only use bars that have
    # fully closed before the lookup timestamp to avoid look-ahead bias.
    df['close_time'] = df.index + pd.Timedelta(hours=H4_BAR_HOURS)
    return df[['trend', 'diff_pips', 'close_time']]


def get_h4_trend_at(h4_trend: pd.DataFrame, close_times: np.ndarray, ts: pd.Timestamp):
    idx = np.searchsorted(close_times, ts.to_datetime64(), side='right') - 1
    if idx < 0:
        return 0, 0.0
    row = h4_trend.iloc[idx]
    return int(row['trend']), float(row['diff_pips'])


# ── 3. LONDON RANGE (per range-window variant, computed once each) ──────────

def compute_london_ranges(h1: pd.DataFrame, start_hour: int, end_hour: int) -> dict:
    window = h1[(h1.index.hour >= start_hour) & (h1.index.hour < end_hour)]
    ranges = {}
    for day, grp in window.groupby(window.index.date):
        if grp.empty:
            continue
        high = float(grp['High'].max())
        low  = float(grp['Low'].min())
        ranges[day] = {
            'high': high, 'low': low,
            'range_pips': (high - low) / PIP_SIZE,
        }
    return ranges


# ── 4. BACKTEST ENGINE ───────────────────────────────────────────────────────

def run_backtest(h1_by_day: dict, h4_trend: pd.DataFrame, close_times: np.ndarray,
                 london_ranges: dict, sl_multiplier: float, h4_threshold_pips: float,
                 trading_days: list) -> tuple[list, list]:
    trades  = []
    balance = START_BALANCE
    equity  = [START_BALANCE]

    for day in trading_days:
        rng = london_ranges.get(day)
        if rng is None or rng['range_pips'] < MIN_LONDON_RANGE_PIPS:
            continue

        day_bars = h1_by_day.get(day)
        if day_bars is None or day_bars.empty:
            continue

        ny_open_ts = pd.Timestamp(day, tz='UTC') + pd.Timedelta(hours=NY_ENTRY_START_HOUR)
        trend, diff_pips = get_h4_trend_at(h4_trend, close_times, ny_open_ts)
        if trend == 0 or diff_pips < h4_threshold_pips:
            continue

        ny_bars = day_bars[day_bars.index.hour.isin(NY_ENTRY_HOURS)]
        if ny_bars.empty:
            continue

        signal = None
        for ts, bar in ny_bars.iterrows():
            buy_break  = bar['High'] > rng['high']
            sell_break = bar['Low']  < rng['low']
            if buy_break and sell_break:
                break  # ambiguous bar -- skip the day
            if trend == 1 and buy_break:
                signal = {'direction': 'BUY', 'entry': rng['high'], 'trigger_ts': ts}
                break
            if trend == -1 and sell_break:
                signal = {'direction': 'SELL', 'entry': rng['low'], 'trigger_ts': ts}
                break
        if signal is None:
            continue

        sl_pips = rng['range_pips'] * sl_multiplier
        tp_pips = sl_pips * RISK_REWARD
        entry   = signal['entry']
        if signal['direction'] == 'BUY':
            sl_price = entry - sl_pips * PIP_SIZE
            tp_price = entry + tp_pips * PIP_SIZE
        else:
            sl_price = entry + sl_pips * PIP_SIZE
            tp_price = entry - tp_pips * PIP_SIZE

        scan_bars = day_bars[(day_bars.index >= signal['trigger_ts']) &
                             (day_bars.index.hour <= EOD_HOUR)]

        exit_price = exit_reason = exit_ts = None
        for ts, bar in scan_bars.iterrows():
            if signal['direction'] == 'BUY':
                sl_hit = bar['Low']  <= sl_price
                tp_hit = bar['High'] >= tp_price
            else:
                sl_hit = bar['High'] >= sl_price
                tp_hit = bar['Low']  <= tp_price

            if sl_hit:                       # conservative tie-break: SL wins
                exit_price, exit_reason = sl_price, 'SL'
            elif tp_hit:
                exit_price, exit_reason = tp_price, 'TP'
            if exit_price is not None:
                exit_ts = ts
                break

        if exit_price is None:
            eod_bar = day_bars[day_bars.index.hour == EOD_HOUR]
            if not eod_bar.empty:
                b = eod_bar.iloc[0]
                exit_price = (b['Open'] + b['Close']) / 2   # ~17:30 approximation
                exit_ts    = eod_bar.index[0]
            elif not scan_bars.empty:
                exit_price = scan_bars.iloc[-1]['Close']
                exit_ts    = scan_bars.index[-1]
            else:
                exit_price = day_bars.iloc[-1]['Close']
                exit_ts    = day_bars.index[-1]
            exit_reason = 'EOD'

        pips = ((exit_price - entry) if signal['direction'] == 'BUY'
                else (entry - exit_price)) / PIP_SIZE
        pnl  = round(pips * PIP_VALUE, 2)
        balance = round(balance + pnl, 2)

        trades.append({
            'Date'        : str(day),
            'Direction'   : signal['direction'],
            'Entry Time'  : signal['trigger_ts'].strftime('%H:%M'),
            'Exit Time'   : exit_ts.strftime('%H:%M'),
            'Entry Price' : round(entry, 5),
            'Exit Price'  : round(exit_price, 5),
            'SL'          : round(sl_price, 5),
            'TP'          : round(tp_price, 5),
            'Range Pips'  : round(rng['range_pips'], 1),
            'SL Pips'     : round(sl_pips, 1),
            'TP Pips'     : round(tp_pips, 1),
            'Pips'        : round(pips, 1),
            'P&L (USD)'   : pnl,
            'Balance'     : balance,
            'Exit Reason' : exit_reason,
            'Result'      : 'WIN' if pnl > 0 else ('LOSS' if pnl < 0 else 'BE'),
        })
        equity.append(balance)

    return trades, equity


# ── 5. STATS + PASS CRITERIA ─────────────────────────────────────────────────

def compute_stats(trades: list, equity: list) -> dict:
    if not trades:
        return {'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0, 'pnl': 0.0,
                'profit_factor': 0.0, 'max_dd': 0.0, 'profitable_months_pct': 0.0,
                'months_tested': 0, 'avg_win': 0.0, 'avg_loss': 0.0,
                'final_balance': START_BALANCE}

    df     = pd.DataFrame(trades)
    wins   = df[df['P&L (USD)'] > 0]
    losses = df[df['P&L (USD)'] < 0]

    gross_win  = wins['P&L (USD)'].sum()
    gross_loss = abs(losses['P&L (USD)'].sum())
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float('inf')

    peak, max_dd = START_BALANCE, 0.0
    for e in equity:
        peak   = max(peak, e)
        max_dd = max(max_dd, (peak - e) / peak * 100)

    df['ym'] = pd.to_datetime(df['Date']).dt.to_period('M')
    monthly  = df.groupby('ym')['P&L (USD)'].sum()
    months_tested      = len(monthly)
    profitable_months  = int((monthly > 0).sum())
    profitable_months_pct = (profitable_months / months_tested * 100) if months_tested else 0.0

    return {
        'trades'        : len(df),
        'wins'          : len(wins),
        'losses'        : len(losses),
        'win_rate'      : round(len(wins) / len(df) * 100, 1),
        'pnl'           : round(df['P&L (USD)'].sum(), 2),
        'profit_factor' : (round(profit_factor, 2) if profit_factor != float('inf')
                          else profit_factor),
        'max_dd'        : round(max_dd, 2),
        'profitable_months_pct': round(profitable_months_pct, 1),
        'months_tested' : months_tested,
        'avg_win'       : round(wins['P&L (USD)'].mean(), 2) if not wins.empty else 0.0,
        'avg_loss'      : round(losses['P&L (USD)'].mean(), 2) if not losses.empty else 0.0,
        'final_balance' : equity[-1],
    }


def passes_criteria(s: dict) -> tuple[bool, list]:
    reasons = []
    if not (PASS_WIN_RATE_MIN <= s['win_rate'] <= PASS_WIN_RATE_MAX):
        reasons.append(f"win rate {s['win_rate']}% outside {PASS_WIN_RATE_MIN:.0f}-{PASS_WIN_RATE_MAX:.0f}%")
    if not (s['max_dd'] < PASS_MAX_DD):
        reasons.append(f"max DD {s['max_dd']}% >= {PASS_MAX_DD:.0f}%")
    pf = s['profit_factor']
    if not (pf > PASS_MIN_PROFIT_FACTOR):
        reasons.append(f"profit factor {pf} <= {PASS_MIN_PROFIT_FACTOR}")
    if not (s['trades'] >= PASS_MIN_TRADES):
        reasons.append(f"only {s['trades']} trades (< {PASS_MIN_TRADES})")
    if not (s['profitable_months_pct'] >= PASS_MIN_PROFITABLE_MONTHS_PCT):
        reasons.append(f"only {s['profitable_months_pct']}% profitable months "
                       f"(< {PASS_MIN_PROFITABLE_MONTHS_PCT:.0f}%)")
    return len(reasons) == 0, reasons


def rank_key(result: dict):
    pf = result['stats']['profit_factor']
    pf_sort = 1e9 if pf == float('inf') else pf
    return (-pf_sort, result['stats']['max_dd'])


# ── 6. REPORTING ──────────────────────────────────────────────────────────

def print_grid_table(results: list) -> None:
    w = 108
    print("=" * w)
    print("  TRAINING GRID SEARCH -- 27 configs  |  EURUSD H1  |  2020-01 to 2022-06")
    print("=" * w)
    print(f"  {'Range window':<13} {'SL x':>5} {'H4 thr':>7}  {'Trades':>7} {'Win%':>6} "
          f"{'PF':>6} {'MaxDD':>7} {'ProfMo%':>8} {'P&L':>10}  Pass?")
    print("-" * w)
    for r in sorted(results, key=rank_key):
        s = r['stats']
        pf_str = 'inf' if s['profit_factor'] == float('inf') else f"{s['profit_factor']:.2f}"
        print(f"  {r['range_window']:<13} {r['sl_mult']:>4.1f}x {r['h4_threshold']:>6.0f}p  "
              f"{s['trades']:>7} {s['win_rate']:>5.1f}% {pf_str:>6} {s['max_dd']:>6.1f}% "
              f"{s['profitable_months_pct']:>7.1f}% ${s['pnl']:>+8,.2f}  "
              f"{'PASS' if r['passed'] else 'fail'}")
    print("=" * w)
    n_pass = sum(1 for r in results if r['passed'])
    print(f"  {n_pass}/{len(results)} configs passed all training criteria.\n")


def print_top3(results: list) -> list:
    top3 = sorted(results, key=rank_key)[:3]
    print("TOP 3 TRAINING CONFIGS (by profit factor desc, max DD asc tiebreak):\n")
    for i, r in enumerate(top3, 1):
        s = r['stats']
        pf_str = 'inf' if s['profit_factor'] == float('inf') else f"{s['profit_factor']:.2f}"
        print(f"  #{i}  range={r['range_window']}  SL={r['sl_mult']}x  H4_thr={r['h4_threshold']}p")
        print(f"      trades={s['trades']}  win_rate={s['win_rate']}%  PF={pf_str}  "
              f"max_dd={s['max_dd']}%  profitable_months={s['profitable_months_pct']}%  "
              f"P&L=${s['pnl']:+,.2f}")
        if not r['passed']:
            print(f"      DID NOT PASS training criteria: {'; '.join(r['fail_reasons'])}")
        print()
    return top3


def sub_periods(start: datetime, end: datetime, n: int) -> list:
    total_days = (end - start).days
    step = total_days / n
    bounds = []
    for i in range(n):
        p_start = start + pd.Timedelta(days=round(step * i))
        p_end   = start + pd.Timedelta(days=round(step * (i + 1))) - pd.Timedelta(seconds=1)
        bounds.append((p_start, p_end))
    return bounds


def print_walk_forward(top3: list, h1_by_day: dict, h4_trend: pd.DataFrame,
                       close_times: np.ndarray, ranges_by_variant: dict,
                       all_days: list) -> None:
    periods = sub_periods(DATA_START, DATA_END, 6)
    w = 100
    print("=" * w)
    print("  WALK-FORWARD -- 6 equal sub-periods across the FULL 2020-2024 dataset")
    print("=" * w)

    for i, r in enumerate(top3, 1):
        print(f"\n  Config #{i}: range={r['range_window']}  SL={r['sl_mult']}x  "
              f"H4_thr={r['h4_threshold']}p")
        print(f"  {'Period':<26} {'Trades':>7} {'Win%':>6} {'P&L':>10}   Flag")
        print("  " + "-" * (w - 2))
        ranges = ranges_by_variant[r['range_window']]
        any_negative = False
        for p_start, p_end in periods:
            period_days = [d for d in all_days if p_start.date() <= d <= p_end.date()]
            trades, equity = run_backtest(h1_by_day, h4_trend, close_times, ranges,
                                          r['sl_mult'], r['h4_threshold'], period_days)
            s = compute_stats(trades, equity)
            flag = ''
            if s['trades'] > 0 and s['pnl'] < 0:
                flag = '<<< NEGATIVE'
                any_negative = True
            label = f"{p_start.date()} to {p_end.date()}"
            print(f"  {label:<26} {s['trades']:>7} {s['win_rate']:>5.1f}% "
                  f"${s['pnl']:>+8,.2f}   {flag}")
        if not any_negative:
            print("  No negative sub-periods.")
    print()


def print_forward_test(top3: list, h1_by_day: dict, h4_trend: pd.DataFrame,
                       close_times: np.ndarray, ranges_by_variant: dict,
                       forward_days: list) -> list:
    w = 100
    print("=" * w)
    print("  FORWARD TEST -- top 3 training configs, run ONCE on unseen 2022-07 to 2024-12 data")
    print("=" * w)
    forward_results = []
    for i, r in enumerate(top3, 1):
        ranges = ranges_by_variant[r['range_window']]
        trades, equity = run_backtest(h1_by_day, h4_trend, close_times, ranges,
                                      r['sl_mult'], r['h4_threshold'], forward_days)
        s = compute_stats(trades, equity)
        passed, reasons = passes_criteria(s)
        forward_results.append({**r, 'forward_stats': s, 'forward_passed': passed,
                                'forward_fail_reasons': reasons})
        pf_str = 'inf' if s['profit_factor'] == float('inf') else f"{s['profit_factor']:.2f}"
        print(f"\n  #{i}  range={r['range_window']}  SL={r['sl_mult']}x  H4_thr={r['h4_threshold']}p")
        print(f"      trades={s['trades']}  win_rate={s['win_rate']}%  PF={pf_str}  "
              f"max_dd={s['max_dd']}%  profitable_months={s['profitable_months_pct']}%  "
              f"P&L=${s['pnl']:+,.2f}")
        print(f"      {'SURVIVES' if passed else 'DOES NOT SURVIVE'} forward test "
              f"against the same pass criteria"
              + ('' if passed else f":  {'; '.join(reasons)}"))
    print()
    return forward_results


def print_comparison(forward_results: list) -> None:
    w = 118
    print("=" * w)
    print("  TRAIN vs FORWARD TEST -- TOP 3 CONFIGS")
    print("=" * w)
    print(f"  {'Config':<30} {'Trades':>7} {'Win%':>6} {'PF':>6} {'MaxDD':>7} "
          f"{'ProfMo%':>8} {'P&L':>10}   Split")
    print("-" * w)
    for r in forward_results:
        label = f"range={r['range_window']} SL={r['sl_mult']}x thr={r['h4_threshold']}p"
        for split_name, s in [('TRAIN  ', r['stats']), ('FORWARD', r['forward_stats'])]:
            pf_str = 'inf' if s['profit_factor'] == float('inf') else f"{s['profit_factor']:.2f}"
            print(f"  {label:<30} {s['trades']:>7} {s['win_rate']:>5.1f}% {pf_str:>6} "
                  f"{s['max_dd']:>6.1f}% {s['profitable_months_pct']:>7.1f}% "
                  f"${s['pnl']:>+8,.2f}   {split_name}")
        print()
    print("=" * w)


def print_verdict(forward_results: list) -> None:
    survivors = [r for r in forward_results if r['forward_passed']]
    print()
    print("=" * 90)
    print("  HONEST VERDICT")
    print("=" * 90)
    if not survivors:
        print("  NO training-ranked config passes the same criteria out-of-sample.")
        print("  Whatever edge shows up on 2020-2022 training data does not carry")
        print("  forward cleanly to 2022-2024 -- treat NY-open breakout on EURUSD as")
        print("  NOT validated for live trading in this form.")
    else:
        print(f"  {len(survivors)}/3 top-ranked configs pass the SAME pass criteria on")
        print("  unseen 2022-2024 data:")
        for r in survivors:
            print(f"    - range={r['range_window']}  SL={r['sl_mult']}x  "
                  f"H4_thr={r['h4_threshold']}p")
        print("  Treat this as tentative evidence of a durable edge, not proof --")
        print("  one clean out-of-sample pass on a single pair/period is a signal to")
        print("  keep testing (more pairs, live paper trading), not a green light to")
        print("  go live at size.")
    print("=" * 90)


# ── MAIN ──────────────────────────────────────────────────────────────────

def main() -> None:
    h1, h4 = fetch_data()

    h1_by_day = {d: g for d, g in h1.groupby(h1.index.date)}
    all_days  = sorted(d for d in h1_by_day if pd.Timestamp(d).weekday() < 5)
    train_days   = [d for d in all_days if TRAIN_START.date() <= d <= TRAIN_END.date()]
    forward_days = [d for d in all_days if FORWARD_START.date() <= d <= FORWARD_END.date()]

    print(f"Trading days -- train: {len(train_days)}  forward: {len(forward_days)}  "
          f"total: {len(all_days)}\n")

    h4_trend    = compute_h4_trend(h4)
    close_times = h4_trend['close_time'].values

    ranges_by_variant = {
        label: compute_london_ranges(h1, s, e)
        for label, (s, e) in RANGE_WINDOWS.items()
    }

    # -- 1. Grid search on TRAIN data only --------------------------------
    results = []
    for rw_label in RANGE_WINDOWS:
        ranges = ranges_by_variant[rw_label]
        for sl_mult in SL_MULTIPLIERS:
            for thr in H4_THRESHOLDS:
                trades, equity = run_backtest(h1_by_day, h4_trend, close_times,
                                              ranges, sl_mult, thr, train_days)
                stats = compute_stats(trades, equity)
                passed, reasons = passes_criteria(stats)
                results.append({
                    'range_window': rw_label, 'sl_mult': sl_mult, 'h4_threshold': thr,
                    'stats': stats, 'passed': passed, 'fail_reasons': reasons,
                })

    print_grid_table(results)
    top3 = print_top3(results)

    # -- 2. Forward test top 3 (run once) ----------------------------------
    forward_results = print_forward_test(top3, h1_by_day, h4_trend, close_times,
                                         ranges_by_variant, forward_days)

    # -- 3. Walk-forward across the full 2020-2024 range -------------------
    print_walk_forward(top3, h1_by_day, h4_trend, close_times, ranges_by_variant, all_days)

    # -- 4. Final comparison + verdict --------------------------------------
    print_comparison(forward_results)
    print_verdict(forward_results)


if __name__ == '__main__':
    main()
