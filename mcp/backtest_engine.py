"""
Generic range-breakout backtest engine for the MCP server's run_backtest
tool. This is a NEW file -- it does not modify any existing bot file.

Supports the 3 registered strategies that share a common shape (H4 trend
filter + session-range breakout + pip-multiple SL/TP): london_breakout,
ny_open_breakout, asian_range_breakout. Default parameters for each are
read directly from that strategy module's own constants (imported, not
hand-duplicated) so they stay in sync with the real strategy code.

Other registered strategies (sma_ema_combined, h4_trend_pullback,
mean_reversion, and all stubs) have fundamentally different entry logic
-- crossover state machines, pullback+reclaim, RSI mean reversion -- that
this generic range-breakout walk can't faithfully represent. run()
returns a clear "not supported" response for those rather than a
misleading generic approximation. For a rigorous validation of any
strategy, use the dedicated src/*_backtest.py research scripts; this
engine is intentionally a lighter-weight tool for quick interactive
queries from Claude.

Read-only: only reads OHLC bars via core.data_loader.get_bars(). Never
touches trades_log.csv, daily_state.json, or places any order.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import data_loader

START_BALANCE = 10_000.00


def _pip_size(pair: str) -> float:
    return 0.01 if 'JPY' in pair else 0.0001


def _pip_value(pair: str) -> float:
    """Approx USD/pip at 0.1 lots -- matches the convention used across
    this repo's other backtest scripts (src/*_backtest.py)."""
    return 0.67 if 'JPY' in pair else 1.00


def _preset(strategy: str) -> dict | None:
    """Default backtest parameters, sourced from each strategy module's
    own constants."""
    if strategy == 'london_breakout':
        from strategies import london_breakout as m
        return {
            'range_start_hour': 0, 'range_end_hour': m.ASIAN_END_HOUR,
            'breakout_start_hour': 8, 'breakout_end_hour': 22,
            'min_range_pips': m.MIN_ASIAN_RANGE_PIPS,
            'sl_multiplier': 0.5, 'tp_multiplier': 1.0,
            'h4_mode': 'close_beyond_sma', 'h4_threshold_pips': 0,
            'eod_hour': 17,
        }
    if strategy == 'ny_open_breakout':
        from strategies import ny_open_breakout as m
        return {
            'range_start_hour': m.LONDON_RANGE_START_HOUR,
            'range_end_hour': m.LONDON_RANGE_END_HOUR,
            'breakout_start_hour': m.NY_ENTRY_START_HOUR,
            'breakout_end_hour': m.NY_ENTRY_END_HOUR,
            'min_range_pips': m.MIN_LONDON_RANGE_PIPS,
            'sl_multiplier': m.SL_MULTIPLIER,
            'tp_multiplier': m.SL_MULTIPLIER * m.RISK_REWARD,
            'h4_mode': 'sma_sign_threshold', 'h4_threshold_pips': m.H4_THRESHOLD_PIPS,
            'eod_hour': 17,
        }
    if strategy == 'asian_range_breakout':
        from strategies import asian_range_breakout as m
        return {
            'range_start_hour': 0, 'range_end_hour': m.ASIAN_END_HOUR,
            'breakout_start_hour': m.OVERLAP_START_HOUR,
            'breakout_end_hour': m.OVERLAP_END_HOUR,
            'min_range_pips': m.MIN_ASIAN_RANGE_PIPS,
            'sl_multiplier': 1.0, 'tp_multiplier': m.TP_MULTIPLIER,
            'h4_mode': 'sma_sign', 'h4_threshold_pips': 0,
            'eod_hour': 20,
        }
    return None


SUPPORTED_STRATEGIES = ['london_breakout', 'ny_open_breakout', 'asian_range_breakout']


def _compute_h4_trend(h4: pd.DataFrame, pip_size: float, mode: str,
                      threshold_pips: float) -> pd.DataFrame:
    df = h4.copy()
    df['SMA50']  = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    df = df.dropna(subset=['SMA200']).copy()
    df['trend'] = 0

    if mode == 'close_beyond_sma':
        bull = (df['Close'] > df['SMA50']) & (df['SMA50'] > df['SMA200'])
        bear = (df['Close'] < df['SMA50']) & (df['SMA50'] < df['SMA200'])
    elif mode == 'sma_sign_threshold':
        diff_pips = (df['SMA50'] - df['SMA200']).abs() / pip_size
        bull = (df['SMA50'] > df['SMA200']) & (diff_pips >= threshold_pips)
        bear = (df['SMA50'] < df['SMA200']) & (diff_pips >= threshold_pips)
    else:  # 'sma_sign'
        bull = df['SMA50'] > df['SMA200']
        bear = df['SMA50'] < df['SMA200']

    df.loc[bull, 'trend'] = 1
    df.loc[bear, 'trend'] = -1
    # H4 bars close 4h after they open -- only use bars fully closed
    # before the lookup timestamp (avoids look-ahead bias).
    df['close_time'] = df.index + pd.Timedelta(hours=4)
    return df[['trend', 'close_time']]


def _get_h4_trend_at(h4_trend: pd.DataFrame, close_times: np.ndarray,
                     ts: pd.Timestamp) -> int:
    idx = np.searchsorted(close_times, ts.to_datetime64(), side='right') - 1
    if idx < 0:
        return 0
    return int(h4_trend.iloc[idx]['trend'])


def _run_range_breakout(h1_by_day: dict, trading_days: list, h4_trend: pd.DataFrame,
                        close_times: np.ndarray, pip_size: float, pip_value: float,
                        p: dict) -> tuple[list, list]:
    trades, balance, equity = [], START_BALANCE, [START_BALANCE]

    for day in trading_days:
        day_bars = h1_by_day.get(day)
        if day_bars is None or day_bars.empty:
            continue

        range_bars = day_bars[(day_bars.index.hour >= p['range_start_hour']) &
                              (day_bars.index.hour < p['range_end_hour'])]
        if range_bars.empty:
            continue
        r_high = float(range_bars['High'].max())
        r_low  = float(range_bars['Low'].min())
        range_pips = (r_high - r_low) / pip_size
        if range_pips < p['min_range_pips']:
            continue

        breakout_ts = pd.Timestamp(day, tz='UTC') + pd.Timedelta(hours=p['breakout_start_hour'])
        trend = _get_h4_trend_at(h4_trend, close_times, breakout_ts)
        if trend == 0:
            continue

        scan = day_bars[(day_bars.index.hour >= p['breakout_start_hour']) &
                        (day_bars.index.hour < p['breakout_end_hour'])]
        signal = None
        for ts, bar in scan.iterrows():
            if trend == 1 and bar['Close'] > r_high:
                signal = {'direction': 'BUY', 'entry': r_high, 'trigger_ts': ts}
                break
            if trend == -1 and bar['Close'] < r_low:
                signal = {'direction': 'SELL', 'entry': r_low, 'trigger_ts': ts}
                break
        if signal is None:
            continue

        sl_pips = range_pips * p['sl_multiplier']
        tp_pips = range_pips * p['tp_multiplier']
        entry = signal['entry']
        if signal['direction'] == 'BUY':
            sl_price, tp_price = entry - sl_pips * pip_size, entry + tp_pips * pip_size
        else:
            sl_price, tp_price = entry + sl_pips * pip_size, entry - tp_pips * pip_size

        exit_scan = day_bars[(day_bars.index > signal['trigger_ts']) &
                             (day_bars.index.hour <= p['eod_hour'])]
        exit_price = exit_reason = None
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
                break

        if exit_price is None:
            eod_bar = day_bars[day_bars.index.hour == p['eod_hour']]
            if not eod_bar.empty:
                b = eod_bar.iloc[0]
                exit_price = (b['Open'] + b['Close']) / 2
            elif not exit_scan.empty:
                exit_price = exit_scan.iloc[-1]['Close']
            else:
                exit_price = day_bars.iloc[-1]['Close']
            exit_reason = 'EOD'

        pips = ((exit_price - entry) if signal['direction'] == 'BUY'
                else (entry - exit_price)) / pip_size
        pnl  = round(pips * pip_value, 2)
        balance = round(balance + pnl, 2)

        trades.append({
            'date': str(day), 'direction': signal['direction'],
            'entry': round(entry, 5), 'exit': round(exit_price, 5),
            'pnl': pnl, 'exit_reason': exit_reason,
        })
        equity.append(balance)

    return trades, equity


def _compute_stats(trades: list, equity: list) -> dict:
    if not trades:
        return {'trades': 0, 'win_rate': 0.0, 'pnl': 0.0, 'profit_factor': 0.0,
                'max_drawdown_pct': 0.0, 'final_balance': START_BALANCE}

    df = pd.DataFrame(trades)
    wins   = df[df['pnl'] > 0]
    losses = df[df['pnl'] < 0]
    gross_win  = wins['pnl'].sum()
    gross_loss = abs(losses['pnl'].sum())
    pf = (gross_win / gross_loss) if gross_loss > 0 else float('inf')

    peak, max_dd = equity[0], 0.0
    for e in equity:
        peak   = max(peak, e)
        max_dd = max(max_dd, (peak - e) / peak * 100)

    return {
        'trades'          : len(df),
        'win_rate'        : round(len(wins) / len(df) * 100, 1),
        'pnl'             : round(df['pnl'].sum(), 2),
        'profit_factor'   : (round(pf, 2) if pf != float('inf') else 'inf'),
        'max_drawdown_pct': round(max_dd, 2),
        'final_balance'   : equity[-1],
    }


def run(strategy: str, pair: str, start_date: str, end_date: str,
       config: dict | None = None) -> dict:
    """
    Run a quick range-breakout backtest. Returns a dict with 'success'
    and either 'stats'/'trades' or 'error'.
    """
    preset = _preset(strategy)
    if preset is None:
        return {
            'success': False,
            'error': (f"No backtest engine available for strategy '{strategy}' via "
                     f"the MCP server. Supported: {SUPPORTED_STRATEGIES}. Other "
                     f"registered strategies have fundamentally different entry "
                     f"logic that this generic range-breakout engine can't "
                     f"faithfully represent -- use the dedicated "
                     f"src/*_backtest.py scripts for those."),
            'supported_strategies': SUPPORTED_STRATEGIES,
        }

    config = config or {}
    p = {**preset, **config}

    try:
        start = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        end   = datetime.strptime(end_date, '%Y-%m-%d').replace(
            hour=23, minute=59, second=59, tzinfo=timezone.utc)
    except ValueError as e:
        return {'success': False, 'error': f'invalid date format (expected YYYY-MM-DD): {e}'}

    pip_size  = _pip_size(pair)
    pip_value = _pip_value(pair)

    try:
        h1 = data_loader.get_bars(pair, 'H1', start, end)
        h4 = data_loader.get_bars(pair, 'H4', start, end)
    except Exception as e:
        return {'success': False, 'error': f'data fetch failed: {e}'}

    if h1.empty or h4.empty:
        return {'success': False, 'error': 'no data returned for this pair/date range'}

    h4_trend    = _compute_h4_trend(h4, pip_size, p['h4_mode'], p['h4_threshold_pips'])
    close_times = h4_trend['close_time'].values

    h1_by_day     = {d: g for d, g in h1.groupby(h1.index.date)}
    trading_days  = sorted(d for d in h1_by_day if pd.Timestamp(d).weekday() < 5)

    trades, equity = _run_range_breakout(h1_by_day, trading_days, h4_trend,
                                         close_times, pip_size, pip_value, p)
    stats = _compute_stats(trades, equity)

    return {
        'success'    : True,
        'strategy'   : strategy,
        'pair'       : pair,
        'start_date' : start_date,
        'end_date'   : end_date,
        'config_used': p,
        'stats'      : stats,
        'trades'     : trades[:50],   # cap payload size; stats cover the full set
        'trade_count': len(trades),
    }
