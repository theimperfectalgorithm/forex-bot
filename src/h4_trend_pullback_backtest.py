"""
Forex Bot -- H4 Trend Pullback Backtest + Optimization (GBPJPY, H1)
      plus a correlation study against LondonBreakout on the same pair.

RERUN NOTE: this replaces the first run's same-day EOD-close engine.
That run's own diagnostic was that the daily 17:30 UTC close was
truncating winners before they could reach the 2:1 TP (only 21% of
trades hit TP; 27% got cut at EOD averaging near-breakeven) while losers
still ate the full stop. This version removes the same-day forced close
entirely -- positions now run to their natural SL/TP across multiple
days, with a Friday-only close at 20:00 UTC (mirroring the live bot's
own EOD->Friday-close migration) as the only forced exit. See the
exit-reason breakdown printed by this script for the direct before/after
comparison.

Strategy under test (strategies/trend_following/h4_trend_pullback.py):
  1. H4 trend: SMA50 vs SMA200 on H4 -- bullish if 50 above 200, bearish
     if 50 below 200, AND the two SMAs must be separated by at least
     H4_THRESHOLD_PIPS (grid: 20/30/50p), otherwise neutral (no trade).
     Only H4 bars that have FULLY CLOSED before the signal bar are used
     (H4 bars are 4h candles aligned to 00/04/08/12/16/20 UTC).
  2. Pullback + reclaim on H1, using an EMA(period) (grid: 20/50/100):
       bullish trend -> BUY  when a bar's Low  <= EMA + depth_pips  AND
                         that same bar's Close  > EMA
       bearish trend -> SELL when a bar's High >= EMA - depth_pips  AND
                         that same bar's Close  < EMA
     (depth_pips grid: 5/10/15 -- how close the wick must get to the EMA
     to count as a "touch"; the bar's Close confirms the reclaim.)
  3. Session filter: only bars in SESSION_START_HOUR-SESSION_END_HOUR UTC
     (08:00-21:00) are eligible ENTRY bars. Unlike the first run, this is
     now the FULL stated window -- the old ENTRY_CUTOFF_HOUR=17 narrowing
     existed only because entries after ~16:00 had no time left before
     the old 17:30 EOD; with EOD removed, that reasoning no longer
     applies. Once a position is open, it is monitored on EVERY
     subsequent H1 bar regardless of session hours (SL/TP can trigger
     any time the market is open), not just during the entry window.
  4. SL = the swing low (BUY) / swing high (SELL) over the last
     SL_LOOKBACK H1 bars (grid: 3/5/8), ending at the signal bar.
     TP = 2x the SL distance (2:1 reward:risk).
  5. NO same-day forced close. Positions run to natural SL/TP across
     multiple days. The ONLY forced exit is Friday at 20:00 UTC (weekend
     gap protection) -- checked BEFORE that bar's own SL/TP, so it
     preempts same-bar price action, and closed at that bar's Open since
     20:00 falls exactly on an H1 boundary (no approximation needed,
     unlike the old 17:30 mid-bar case).
  6. Only one OPEN POSITION at a time per pair -- generalizes the first
     run's "one trade per day" now that a position can span multiple
     days (there's no longer a clean daily boundary to count against).

Data: core/data_loader.get_bars() -- CSV-backed on this Mac, identical
MT5 path on the VPS.

Protocol (train/forward split is a hard boundary -- the grid is never
chosen by looking at forward-test results):
  1. Grid search (3 EMA periods x 3 pullback depths x 3 H4 thresholds x
     3 SL lookbacks = 81 configs) on TRAIN data only (2020-01 to 2022-06).
  2. Apply pass criteria, rank ALL 81 by profit factor desc / max DD asc.
  3. Take the top 3 by that ranking and run them ONCE on FORWARD data
     (2022-07 to 2024-12) -- untouched until this point.
  4. Walk-forward: split the full 2020-2024 range into 6 equal calendar
     periods, report P&L/win-rate per period for the top 3 configs.
  5. Correlation study: re-implement LondonBreakout's Asian-range-breakout
     logic (H1-approximated -- the live class uses M15; only H1 CSV data
     is in scope for this task) on the same GBPJPY data, and compare its
     daily P&L series against the #1-ranked h4_trend_pullback config.
     (LondonBreakout's own logic is UNCHANGED from the first run -- only
     h4_trend_pullback's engine changed.)
  6. Print an honest verdict, plus the SL/TP/FRIDAY_CLOSE exit-reason
     breakdown for direct comparison against the first run's numbers.
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

PAIR             = 'GBPJPY'
PIP_SIZE         = 0.01        # JPY pair
LOT_SIZE         = 0.1
PIP_VALUE        = 0.67        # approx. USD/pip at 0.1 lots for a JPY-quoted pair
START_BALANCE    = 10_000.00

H4_SMA_FAST      = 50
H4_SMA_SLOW      = 200
H4_BAR_HOURS     = 4

SESSION_START_HOUR = 8         # London/NY session filter: 08:00-21:00 UTC
SESSION_END_HOUR   = 21        # full window now -- no EOD-driven narrowing
FRIDAY_CLOSE_HOUR  = 20        # weekend gap protection -- forced exit at
                                # Friday >= 20:00 UTC, mirrors the live
                                # bot's is_friday_close_time() exactly

RISK_REWARD      = 2.0

EMA_PERIODS      = [20, 50, 100]
PULLBACK_DEPTHS  = [5, 10, 15]      # pips
H4_THRESHOLDS    = [20, 30, 50]     # pips between SMA50/SMA200
SL_LOOKBACKS     = [3, 5, 8]        # H1 bars

DATA_START    = datetime(2020, 1, 1, tzinfo=timezone.utc)
DATA_END      = datetime(2024, 12, 31, 23, 59, tzinfo=timezone.utc)
TRAIN_START   = datetime(2020, 1, 1, tzinfo=timezone.utc)
TRAIN_END     = datetime(2022, 6, 30, 23, 59, tzinfo=timezone.utc)
FORWARD_START = datetime(2022, 7, 1, tzinfo=timezone.utc)
FORWARD_END   = datetime(2024, 12, 31, 23, 59, tzinfo=timezone.utc)

PASS_WIN_RATE_MIN = 40.0
PASS_WIN_RATE_MAX = 55.0
PASS_MAX_DD       = 5.0
PASS_MIN_PROFIT_FACTOR = 1.2
PASS_MIN_TRADES  = 50
PASS_MIN_PROFITABLE_MONTHS_PCT = 60.0

# -- LondonBreakout (H1-approximated) comparison constants -- matches the
#    production defaults in strategies/london_breakout.py verbatim
LB_ASIAN_END_HOUR       = 7
LB_BREAKOUT_START_HOUR  = 8
LB_BREAKOUT_END_HOUR    = 22
LB_MIN_ASIAN_RANGE_PIPS = 10
LB_MAX_OVERSHOOT_PIPS   = 20
LB_EOD_HOUR             = 17


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
    """Pure SMA50-vs-SMA200 trend (no Close condition) -- per this task's spec."""
    df = h4.copy()
    df['SMA50']  = df['Close'].rolling(H4_SMA_FAST).mean()
    df['SMA200'] = df['Close'].rolling(H4_SMA_SLOW).mean()
    df = df.dropna(subset=['SMA200']).copy()

    df['diff_pips'] = (df['SMA50'] - df['SMA200']).abs() / PIP_SIZE
    df['trend'] = 0
    df.loc[df['SMA50'] > df['SMA200'], 'trend'] =  1
    df.loc[df['SMA50'] < df['SMA200'], 'trend'] = -1

    df['close_time'] = df.index + pd.Timedelta(hours=H4_BAR_HOURS)
    return df[['trend', 'diff_pips', 'close_time']]


def compute_h4_trend_lb(h4: pd.DataFrame) -> pd.DataFrame:
    """LondonBreakout's actual production trend test: Close > SMA50 > SMA200."""
    df = h4.copy()
    df['SMA50']  = df['Close'].rolling(H4_SMA_FAST).mean()
    df['SMA200'] = df['Close'].rolling(H4_SMA_SLOW).mean()
    df = df.dropna(subset=['SMA200']).copy()

    df['trend'] = 0
    bull = (df['Close'] > df['SMA50']) & (df['SMA50'] > df['SMA200'])
    bear = (df['Close'] < df['SMA50']) & (df['SMA50'] < df['SMA200'])
    df.loc[bull, 'trend'] =  1
    df.loc[bear, 'trend'] = -1
    df['diff_pips'] = 0.0   # unused -- LondonBreakout has no threshold, kept for API parity

    df['close_time'] = df.index + pd.Timedelta(hours=H4_BAR_HOURS)
    return df[['trend', 'diff_pips', 'close_time']]


def get_h4_trend_at(h4_trend: pd.DataFrame, close_times: np.ndarray, ts: pd.Timestamp):
    idx = np.searchsorted(close_times, ts.to_datetime64(), side='right') - 1
    if idx < 0:
        return 0, 0.0
    row = h4_trend.iloc[idx]
    return int(row['trend']), float(row['diff_pips'])


# ── 3. INDICATORS (EMA + swing low/high, each computed once per variant) ────

def compute_emas(h1: pd.DataFrame) -> dict:
    return {p: h1['Close'].ewm(span=p, adjust=False).mean() for p in EMA_PERIODS}


def compute_swings(h1: pd.DataFrame) -> tuple[dict, dict]:
    lows  = {n: h1['Low'].rolling(window=n, min_periods=n).min()  for n in SL_LOOKBACKS}
    highs = {n: h1['High'].rolling(window=n, min_periods=n).max() for n in SL_LOOKBACKS}
    return lows, highs


# ── 4. BACKTEST ENGINE -- H4 Trend Pullback ──────────────────────────────────
#
# Continuous multi-day walk (replaces the old day-by-day loop). At most one
# open position at a time: while flat, only bars inside the session window
# are checked for a new entry; while a position is open, EVERY subsequent
# bar (any hour, any day) is checked for SL/TP or the Friday 20:00 UTC
# forced close, until it resolves.

def run_backtest(h1: pd.DataFrame, ema_series: pd.Series, swing_low: pd.Series,
                 swing_high: pd.Series, h4_trend: pd.DataFrame, close_times: np.ndarray,
                 depth_pips: float, h4_threshold_pips: float,
                 start_ts: datetime, end_ts: datetime) -> tuple[list, list]:
    trades  = []
    balance = START_BALANCE
    equity  = [START_BALANCE]

    bars = h1[(h1.index >= start_ts) & (h1.index <= end_ts)]
    open_pos = None   # dict: direction, entry, sl, tp, entry_ts

    for ts, bar in bars.iterrows():
        if open_pos is not None:
            direction = open_pos['direction']
            sl_price  = open_pos['sl']
            tp_price  = open_pos['tp']

            if ts.weekday() == 4 and ts.hour >= FRIDAY_CLOSE_HOUR:
                # Weekend gap protection: forced exit at 20:00 UTC exactly,
                # checked before this bar's own price action so it can't be
                # preempted by a same-bar SL/TP that would only occur later
                # within the hour.
                exit_price, exit_reason = bar['Open'], 'FRIDAY_CLOSE'
            else:
                exit_price = exit_reason = None
                if direction == 'BUY':
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
                entry = open_pos['entry']
                pips = ((exit_price - entry) if direction == 'BUY'
                        else (entry - exit_price)) / PIP_SIZE
                pnl  = round(pips * PIP_VALUE, 2)
                balance = round(balance + pnl, 2)

                trades.append({
                    'Date'        : str(ts.date()),   # realized (exit) date
                    'Direction'   : direction,
                    'Entry Time'  : open_pos['entry_ts'].strftime('%Y-%m-%d %H:%M'),
                    'Exit Time'   : ts.strftime('%Y-%m-%d %H:%M'),
                    'Entry Price' : round(entry, 5),
                    'Exit Price'  : round(exit_price, 5),
                    'SL'          : round(sl_price, 5),
                    'TP'          : round(tp_price, 5),
                    'Pips'        : round(pips, 1),
                    'P&L (USD)'   : pnl,
                    'Balance'     : balance,
                    'Exit Reason' : exit_reason,
                    'Result'      : 'WIN' if pnl > 0 else ('LOSS' if pnl < 0 else 'BE'),
                    'Hold Hours'  : round((ts - open_pos['entry_ts']).total_seconds() / 3600, 1),
                })
                equity.append(balance)
                open_pos = None
            continue   # closed or still open -- either way, no new entry on this bar

        # -- flat: only consider a new entry during the session window --
        if not (SESSION_START_HOUR <= ts.hour < SESSION_END_HOUR):
            continue

        trend, diff_pips = get_h4_trend_at(h4_trend, close_times, ts)
        if trend == 0 or diff_pips < h4_threshold_pips:
            continue

        ema_val = ema_series.loc[ts]
        if np.isnan(ema_val):
            continue

        low, high, close = bar['Low'], bar['High'], bar['Close']

        if trend == 1:
            if not (low <= ema_val + depth_pips * PIP_SIZE and close > ema_val):
                continue
            direction = 'BUY'
        else:
            if not (high >= ema_val - depth_pips * PIP_SIZE and close < ema_val):
                continue
            direction = 'SELL'

        entry   = close
        sw_low  = swing_low.loc[ts]
        sw_high = swing_high.loc[ts]

        if direction == 'BUY':
            if np.isnan(sw_low) or sw_low >= entry:
                continue
            sl_price = sw_low
            tp_price = entry + RISK_REWARD * (entry - sl_price)
        else:
            if np.isnan(sw_high) or sw_high <= entry:
                continue
            sl_price = sw_high
            tp_price = entry - RISK_REWARD * (sl_price - entry)

        open_pos = {'direction': direction, 'entry': entry, 'sl': sl_price,
                    'tp': tp_price, 'entry_ts': ts}

    # Period boundary: force-close any position still open when the window
    # ends. Should be rare -- Friday closes happen every week, so this only
    # fires if the window itself ends before the next Friday 20:00 does.
    if open_pos is not None and len(bars) > 0:
        last_ts, last_bar = bars.index[-1], bars.iloc[-1]
        direction, entry = open_pos['direction'], open_pos['entry']
        exit_price = last_bar['Close']
        pips = ((exit_price - entry) if direction == 'BUY'
                else (entry - exit_price)) / PIP_SIZE
        pnl  = round(pips * PIP_VALUE, 2)
        balance = round(balance + pnl, 2)
        trades.append({
            'Date'        : str(last_ts.date()),
            'Direction'   : direction,
            'Entry Time'  : open_pos['entry_ts'].strftime('%Y-%m-%d %H:%M'),
            'Exit Time'   : last_ts.strftime('%Y-%m-%d %H:%M'),
            'Entry Price' : round(entry, 5),
            'Exit Price'  : round(exit_price, 5),
            'SL'          : round(open_pos['sl'], 5),
            'TP'          : round(open_pos['tp'], 5),
            'Pips'        : round(pips, 1),
            'P&L (USD)'   : pnl,
            'Balance'     : balance,
            'Exit Reason' : 'END_OF_DATA',
            'Result'      : 'WIN' if pnl > 0 else ('LOSS' if pnl < 0 else 'BE'),
            'Hold Hours'  : round((last_ts - open_pos['entry_ts']).total_seconds() / 3600, 1),
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
    if not (s['profit_factor'] > PASS_MIN_PROFIT_FACTOR):
        reasons.append(f"profit factor {s['profit_factor']} <= {PASS_MIN_PROFIT_FACTOR}")
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


def sub_periods(start: datetime, end: datetime, n: int) -> list:
    total_days = (end - start).days
    step = total_days / n
    bounds = []
    for i in range(n):
        p_start = start + pd.Timedelta(days=round(step * i))
        p_end   = start + pd.Timedelta(days=round(step * (i + 1))) - pd.Timedelta(seconds=1)
        bounds.append((p_start, p_end))
    return bounds


# ── 6. LONDONBREAKOUT (H1-approximated) -- comparison baseline ──────────────

def compute_asian_ranges_h1(h1: pd.DataFrame) -> dict:
    window = h1[h1.index.hour < LB_ASIAN_END_HOUR]
    ranges = {}
    for day, grp in window.groupby(window.index.date):
        if grp.empty:
            continue
        high = float(grp['High'].max())
        low  = float(grp['Low'].min())
        ranges[day] = {'high': high, 'low': low, 'range_pips': (high - low) / PIP_SIZE}
    return ranges


def run_london_breakout_h1(h1_by_day: dict, h4_trend_lb: pd.DataFrame, close_times_lb: np.ndarray,
                           asian_ranges: dict, trading_days: list) -> tuple[list, list]:
    trades, balance, equity = [], START_BALANCE, [START_BALANCE]

    for day in trading_days:
        rng = asian_ranges.get(day)
        if rng is None or rng['range_pips'] < LB_MIN_ASIAN_RANGE_PIPS:
            continue
        day_bars = h1_by_day.get(day)
        if day_bars is None or day_bars.empty:
            continue

        breakout_open_ts = pd.Timestamp(day, tz='UTC') + pd.Timedelta(hours=LB_BREAKOUT_START_HOUR)
        trend, _ = get_h4_trend_at(h4_trend_lb, close_times_lb, breakout_open_ts)
        if trend == 0:
            continue

        scan = day_bars[(day_bars.index.hour >= LB_BREAKOUT_START_HOUR) &
                        (day_bars.index.hour < LB_BREAKOUT_END_HOUR)]
        signal = None
        for ts, bar in scan.iterrows():
            buy_break  = bar['Close'] > rng['high']
            sell_break = bar['Close'] < rng['low']
            if trend == 1 and buy_break:
                if (bar['Close'] - rng['high']) / PIP_SIZE > LB_MAX_OVERSHOOT_PIPS:
                    break
                signal = {'direction': 'BUY', 'entry': rng['high'], 'trigger_ts': ts}
                break
            if trend == -1 and sell_break:
                if (rng['low'] - bar['Close']) / PIP_SIZE > LB_MAX_OVERSHOOT_PIPS:
                    break
                signal = {'direction': 'SELL', 'entry': rng['low'], 'trigger_ts': ts}
                break
        if signal is None:
            continue

        sl_pips, tp_pips = rng['range_pips'] * 0.50, rng['range_pips'] * 1.00
        entry = signal['entry']
        if signal['direction'] == 'BUY':
            sl_price, tp_price = entry - sl_pips * PIP_SIZE, entry + tp_pips * PIP_SIZE
        else:
            sl_price, tp_price = entry + sl_pips * PIP_SIZE, entry - tp_pips * PIP_SIZE

        exit_scan = day_bars[(day_bars.index > signal['trigger_ts']) &
                             (day_bars.index.hour <= LB_EOD_HOUR)]
        exit_price = exit_reason = exit_ts = None
        for ts, bar in exit_scan.iterrows():
            if signal['direction'] == 'BUY':
                sl_hit, tp_hit = bar['Low'] <= sl_price, bar['High'] >= tp_price
            else:
                sl_hit, tp_hit = bar['High'] >= sl_price, bar['Low'] <= tp_price
            if sl_hit:
                exit_price, exit_reason = sl_price, 'SL'
            elif tp_hit:
                exit_price, exit_reason = tp_price, 'TP'
            if exit_price is not None:
                exit_ts = ts
                break

        if exit_price is None:
            eod_bar = day_bars[day_bars.index.hour == LB_EOD_HOUR]
            if not eod_bar.empty:
                b = eod_bar.iloc[0]
                exit_price, exit_ts = (b['Open'] + b['Close']) / 2, eod_bar.index[0]
            elif not exit_scan.empty:
                exit_price, exit_ts = exit_scan.iloc[-1]['Close'], exit_scan.index[-1]
            else:
                exit_price, exit_ts = day_bars.iloc[-1]['Close'], day_bars.index[-1]
            exit_reason = 'EOD'

        pips = ((exit_price - entry) if signal['direction'] == 'BUY'
                else (entry - exit_price)) / PIP_SIZE
        pnl  = round(pips * PIP_VALUE, 2)
        balance = round(balance + pnl, 2)
        trades.append({'Date': str(day), 'Direction': signal['direction'],
                       'P&L (USD)': pnl, 'Balance': balance, 'Exit Reason': exit_reason})
        equity.append(balance)

    return trades, equity


def daily_pnl_series(trades: list, all_days: list) -> pd.Series:
    idx = pd.to_datetime(sorted(all_days))
    s = pd.Series(0.0, index=idx)
    if trades:
        df = pd.DataFrame(trades)
        g  = df.groupby('Date')['P&L (USD)'].sum()
        g.index = pd.to_datetime(g.index)
        s.update(g)
    return s


# ── 7. REPORTING ──────────────────────────────────────────────────────────

def print_exit_breakdown(trades: list, label: str) -> None:
    """
    % of trades hitting SL vs TP vs FRIDAY_CLOSE (vs END_OF_DATA, if any)
    -- the direct before/after comparison against the first (same-day EOD)
    run's diagnosis: 52% SL / 27% EOD (avg +$3.65, near-breakeven) / 21% TP.
    """
    w = 78
    print("-" * w)
    print(f"  EXIT REASON BREAKDOWN -- {label}")
    print("-" * w)
    if not trades:
        print("  no trades")
        print("-" * w + "\n")
        return

    df    = pd.DataFrame(trades)
    total = len(df)
    print(f"  {'Reason':<14} {'Count':>7} {'% of trades':>12} {'Avg P&L':>10} {'Avg hold (h)':>13}")
    for reason in ['SL', 'TP', 'FRIDAY_CLOSE', 'END_OF_DATA']:
        sub = df[df['Exit Reason'] == reason]
        n = len(sub)
        if n == 0 and reason == 'END_OF_DATA':
            continue   # only show this row if it actually happened
        pct      = n / total * 100 if total else 0.0
        avg_pnl  = sub['P&L (USD)'].mean() if n else 0.0
        avg_hold = sub['Hold Hours'].mean() if n else 0.0
        print(f"  {reason:<14} {n:>7} {pct:>11.1f}% ${avg_pnl:>+9.2f} {avg_hold:>13.1f}")
    print(f"  {'TOTAL':<14} {total:>7}")
    print("-" * w + "\n")


def print_grid_summary(results: list) -> None:
    w = 110
    print("=" * w)
    print(f"  TRAINING GRID SEARCH -- {len(results)} configs  |  GBPJPY H1  |  2020-01 to 2022-06")
    print("=" * w)
    n_pass = sum(1 for r in results if r['passed'])
    print(f"  {n_pass}/{len(results)} configs passed all training criteria.\n")

    print(f"  {'EMA':>5} {'Depth':>6} {'H4thr':>6} {'SLbk':>5}  {'Trades':>7} {'Win%':>6} "
          f"{'PF':>6} {'MaxDD':>7} {'ProfMo%':>8} {'P&L':>10}  Pass?")
    print("-" * w)
    ranked = sorted(results, key=rank_key)
    for r in ranked[:15]:
        s = r['stats']
        pf_str = 'inf' if s['profit_factor'] == float('inf') else f"{s['profit_factor']:.2f}"
        print(f"  {r['ema']:>4}p {r['depth']:>5}p {r['h4_threshold']:>5}p {r['sl_lookback']:>4}b  "
              f"{s['trades']:>7} {s['win_rate']:>5.1f}% {pf_str:>6} {s['max_dd']:>6.1f}% "
              f"{s['profitable_months_pct']:>7.1f}% ${s['pnl']:>+8,.2f}  "
              f"{'PASS' if r['passed'] else 'fail'}")
    print(f"  ... ({len(ranked) - 15} more configs not shown, full grid ranked by "
          f"the same key)" if len(ranked) > 15 else '')
    print("=" * w + "\n")


def print_top3(results: list) -> list:
    top3 = sorted(results, key=rank_key)[:3]
    print("TOP 3 TRAINING CONFIGS (by profit factor desc, max DD asc tiebreak):\n")
    for i, r in enumerate(top3, 1):
        s = r['stats']
        pf_str = 'inf' if s['profit_factor'] == float('inf') else f"{s['profit_factor']:.2f}"
        print(f"  #{i}  EMA={r['ema']}  depth={r['depth']}p  H4_thr={r['h4_threshold']}p  "
              f"SL_lookback={r['sl_lookback']}bars")
        print(f"      trades={s['trades']}  win_rate={s['win_rate']}%  PF={pf_str}  "
              f"max_dd={s['max_dd']}%  profitable_months={s['profitable_months_pct']}%  "
              f"P&L=${s['pnl']:+,.2f}")
        if not r['passed']:
            print(f"      DID NOT PASS training criteria: {'; '.join(r['fail_reasons'])}")
        print()
    return top3


def print_forward_test(top3: list, h1: pd.DataFrame, emas: dict, swing_lows: dict,
                       swing_highs: dict, h4_trend: pd.DataFrame, close_times: np.ndarray
                       ) -> list:
    w = 100
    print("=" * w)
    print("  FORWARD TEST -- top 3 training configs, run ONCE on unseen 2022-07 to 2024-12 data")
    print("=" * w)
    forward_results = []
    for i, r in enumerate(top3, 1):
        trades, equity = run_backtest(h1, emas[r['ema']], swing_lows[r['sl_lookback']],
                                      swing_highs[r['sl_lookback']], h4_trend, close_times,
                                      r['depth'], r['h4_threshold'], FORWARD_START, FORWARD_END)
        s = compute_stats(trades, equity)
        passed, reasons = passes_criteria(s)
        forward_results.append({**r, 'forward_stats': s, 'forward_passed': passed,
                                'forward_fail_reasons': reasons, 'forward_trades': trades})
        pf_str = 'inf' if s['profit_factor'] == float('inf') else f"{s['profit_factor']:.2f}"
        print(f"\n  #{i}  EMA={r['ema']}  depth={r['depth']}p  H4_thr={r['h4_threshold']}p  "
              f"SL_lookback={r['sl_lookback']}bars")
        print(f"      trades={s['trades']}  win_rate={s['win_rate']}%  PF={pf_str}  "
              f"max_dd={s['max_dd']}%  profitable_months={s['profitable_months_pct']}%  "
              f"P&L=${s['pnl']:+,.2f}")
        print(f"      {'SURVIVES' if passed else 'DOES NOT SURVIVE'} forward test "
              f"against the same pass criteria"
              + ('' if passed else f":  {'; '.join(reasons)}"))
    print()
    return forward_results


def print_walk_forward(top3: list, h1: pd.DataFrame, emas: dict, swing_lows: dict,
                       swing_highs: dict, h4_trend: pd.DataFrame, close_times: np.ndarray
                       ) -> None:
    periods = sub_periods(DATA_START, DATA_END, 6)
    w = 100
    print("=" * w)
    print("  WALK-FORWARD -- 6 equal sub-periods across the FULL 2020-2024 dataset")
    print("=" * w)

    for i, r in enumerate(top3, 1):
        print(f"\n  Config #{i}: EMA={r['ema']}  depth={r['depth']}p  "
              f"H4_thr={r['h4_threshold']}p  SL_lookback={r['sl_lookback']}bars")
        print(f"  {'Period':<26} {'Trades':>7} {'Win%':>6} {'P&L':>10}   Flag")
        print("  " + "-" * (w - 2))
        any_negative = False
        for p_start, p_end in periods:
            trades, equity = run_backtest(h1, emas[r['ema']], swing_lows[r['sl_lookback']],
                                          swing_highs[r['sl_lookback']], h4_trend, close_times,
                                          r['depth'], r['h4_threshold'], p_start, p_end)
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


def print_comparison(forward_results: list) -> None:
    w = 118
    print("=" * w)
    print("  TRAIN vs FORWARD TEST -- TOP 3 CONFIGS")
    print("=" * w)
    print(f"  {'Config':<42} {'Trades':>7} {'Win%':>6} {'PF':>6} {'MaxDD':>7} "
          f"{'ProfMo%':>8} {'P&L':>10}   Split")
    print("-" * w)
    for r in forward_results:
        label = f"EMA={r['ema']} depth={r['depth']}p thr={r['h4_threshold']}p sl={r['sl_lookback']}b"
        for split_name, s in [('TRAIN  ', r['stats']), ('FORWARD', r['forward_stats'])]:
            pf_str = 'inf' if s['profit_factor'] == float('inf') else f"{s['profit_factor']:.2f}"
            print(f"  {label:<42} {s['trades']:>7} {s['win_rate']:>5.1f}% {pf_str:>6} "
                  f"{s['max_dd']:>6.1f}% {s['profitable_months_pct']:>7.1f}% "
                  f"${s['pnl']:>+8,.2f}   {split_name}")
        print()
    print("=" * w)


def print_correlation(top3: list, h1: pd.DataFrame, h1_by_day: dict, emas: dict,
                      swing_lows: dict, swing_highs: dict, h4_trend: pd.DataFrame,
                      close_times: np.ndarray, h4: pd.DataFrame, all_days: list) -> None:
    w = 90
    print("=" * w)
    print("  CORRELATION vs LONDONBREAKOUT (H1-approximated) -- full 2020-2024, GBPJPY")
    print("=" * w)

    best = top3[0]
    pb_trades, pb_equity = run_backtest(h1, emas[best['ema']], swing_lows[best['sl_lookback']],
                                        swing_highs[best['sl_lookback']], h4_trend, close_times,
                                        best['depth'], best['h4_threshold'], DATA_START, DATA_END)
    pb_stats = compute_stats(pb_trades, pb_equity)

    h4_trend_lb = compute_h4_trend_lb(h4)
    close_times_lb = h4_trend_lb['close_time'].values
    asian_ranges = compute_asian_ranges_h1(h1)
    lb_trades, lb_equity = run_london_breakout_h1(h1_by_day, h4_trend_lb, close_times_lb,
                                                  asian_ranges, all_days)
    lb_stats = compute_stats(lb_trades, lb_equity)

    print(f"\n  H4 Trend Pullback (best config #1): trades={pb_stats['trades']}  "
          f"win_rate={pb_stats['win_rate']}%  P&L=${pb_stats['pnl']:+,.2f}  "
          f"max_dd={pb_stats['max_dd']}%")
    print(f"  LondonBreakout (H1-approx)         : trades={lb_stats['trades']}  "
          f"win_rate={lb_stats['win_rate']}%  P&L=${lb_stats['pnl']:+,.2f}  "
          f"max_dd={lb_stats['max_dd']}%\n")

    pnl_pb = daily_pnl_series(pb_trades, all_days)
    pnl_lb = daily_pnl_series(lb_trades, all_days)
    corr = pnl_pb.corr(pnl_lb)

    both_traded = (pnl_pb != 0) & (pnl_lb != 0)
    n_both = int(both_traded.sum())
    if n_both > 0:
        same_direction = (np.sign(pnl_pb[both_traded]) == np.sign(pnl_lb[both_traded])).sum()
        agree_pct = same_direction / n_both * 100
    else:
        agree_pct = 0.0

    print(f"  Daily P&L correlation (Pearson): {corr:.3f}")
    print(f"  Days both strategies traded    : {n_both}")
    print(f"  Win/loss agreement on those days: {agree_pct:.1f}%")
    print()
    if pd.isna(corr):
        print("  Not enough overlapping trade days to compute a meaningful correlation.")
    elif corr >= 0.4:
        print("  MEANINGFULLY CORRELATED -- the two strategies tend to win and lose on")
        print("  similar days (shared dependence on the same H4 trend filter and UTC")
        print("  session structure). Running both adds trade frequency but does NOT")
        print("  meaningfully diversify drawdown risk.")
    elif corr <= -0.15:
        print("  NEGATIVELY CORRELATED -- combining them would likely smooth the")
        print("  combined equity curve and reduce drawdown versus either alone.")
    else:
        print("  WEAKLY CORRELATED -- some diversification benefit from running both,")
        print("  but it is not strong. Treat as a mild, not primary, reason to combine them.")
    print("=" * w + "\n")


def print_verdict(forward_results: list, correlation_note: str = "") -> None:
    survivors = [r for r in forward_results if r['forward_passed']]
    print()
    print("=" * 90)
    print("  HONEST VERDICT")
    print("=" * 90)
    if not survivors:
        print("  NO training-ranked config passes the same criteria out-of-sample.")
        print("  Whatever edge shows up on 2020-2022 training data does not carry")
        print("  forward cleanly to 2022-2024 -- treat H4 trend pullback on GBPJPY as")
        print("  NOT validated for live trading in this form.")
    else:
        print(f"  {len(survivors)}/3 top-ranked configs pass the SAME pass criteria on")
        print("  unseen 2022-2024 data:")
        for r in survivors:
            print(f"    - EMA={r['ema']}  depth={r['depth']}p  H4_thr={r['h4_threshold']}p  "
                  f"SL_lookback={r['sl_lookback']}bars")
        print("  Treat this as tentative evidence of a durable edge, not proof -- one")
        print("  clean out-of-sample pass on a single pair/period is a signal to keep")
        print("  testing (more pairs, live paper trading), not a green light to go")
        print("  live at size.")
    print("=" * 90)


# ── MAIN ──────────────────────────────────────────────────────────────────

def main() -> None:
    h1, h4 = fetch_data()

    # h1_by_day / all_days are still needed for the (unchanged) LondonBreakout
    # comparison engine and for daily_pnl_series() in the correlation section.
    # h4_trend_pullback's own engine now walks h1 directly by timestamp range.
    h1_by_day = {d: g for d, g in h1.groupby(h1.index.date)}
    all_days  = sorted(d for d in h1_by_day if pd.Timestamp(d).weekday() < 5)
    train_days   = [d for d in all_days if TRAIN_START.date() <= d <= TRAIN_END.date()]
    forward_days = [d for d in all_days if FORWARD_START.date() <= d <= FORWARD_END.date()]

    print(f"Trading days -- train: {len(train_days)}  forward: {len(forward_days)}  "
          f"total: {len(all_days)}\n")

    h4_trend    = compute_h4_trend(h4)
    close_times = h4_trend['close_time'].values

    emas = compute_emas(h1)
    swing_lows, swing_highs = compute_swings(h1)

    # -- 1. Grid search on TRAIN data only ---------------------------------
    results = []
    n_configs = len(EMA_PERIODS) * len(PULLBACK_DEPTHS) * len(H4_THRESHOLDS) * len(SL_LOOKBACKS)
    print(f"Running grid search: {n_configs} configs on {TRAIN_START.date()} to {TRAIN_END.date()} ...")
    for ema_p in EMA_PERIODS:
        for depth in PULLBACK_DEPTHS:
            for thr in H4_THRESHOLDS:
                for sl_lb in SL_LOOKBACKS:
                    trades, equity = run_backtest(
                        h1, emas[ema_p], swing_lows[sl_lb], swing_highs[sl_lb],
                        h4_trend, close_times, depth, thr, TRAIN_START, TRAIN_END)
                    stats = compute_stats(trades, equity)
                    passed, reasons = passes_criteria(stats)
                    results.append({
                        'ema': ema_p, 'depth': depth, 'h4_threshold': thr, 'sl_lookback': sl_lb,
                        'stats': stats, 'passed': passed, 'fail_reasons': reasons,
                        'trades': trades,
                    })
    print("Grid search complete.\n")

    print_grid_summary(results)
    top3 = print_top3(results)

    print_exit_breakdown(top3[0]['trades'],
                         "#1 TRAINING config (2020-01 to 2022-06) -- compare against "
                         "the first run's 52% SL / 27% EOD / 21% TP")

    # -- 2. Forward test top 3 (run once) ----------------------------------
    forward_results = print_forward_test(top3, h1, emas, swing_lows, swing_highs,
                                         h4_trend, close_times)

    print_exit_breakdown(forward_results[0]['forward_trades'],
                         "#1 config, FORWARD TEST (2022-07 to 2024-12)")

    # -- 3. Walk-forward across the full 2020-2024 range --------------------
    print_walk_forward(top3, h1, emas, swing_lows, swing_highs, h4_trend, close_times)

    # -- 4. Correlation vs LondonBreakout ------------------------------------
    print_correlation(top3, h1, h1_by_day, emas, swing_lows, swing_highs, h4_trend,
                      close_times, h4, all_days)

    # -- 5. Final comparison + verdict --------------------------------------
    print_comparison(forward_results)
    print_verdict(forward_results)


if __name__ == '__main__':
    main()
