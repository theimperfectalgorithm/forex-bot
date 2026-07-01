"""
data_loader -- single source of truth for historical OHLC bars used by
every backtest script.

get_bars() hides the platform difference: MT5 is Windows-only, so on any
machine without it (e.g. Mac) this transparently falls back to the CSV
files scripts/export_historical_data.py produces on the VPS.

Returned shape is identical either way: a DataFrame with a UTC
DatetimeIndex named 'time' and columns Open, High, Low, Close,
tick_volume -- the same shape every backtest script's own fetch function
already built by hand from MT5's raw rates array.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

REPO_ROOT      = Path(__file__).parent.parent
HISTORICAL_DIR = REPO_ROOT / 'data' / 'historical'

TIMEFRAMES = ('M15', 'H1', 'H4')

_mt5_timeframe_consts = None  # populated lazily -- needs the mt5 module loaded


def _mt5_timeframe_const(timeframe: str) -> int:
    global _mt5_timeframe_consts
    if _mt5_timeframe_consts is None:
        _mt5_timeframe_consts = {
            'M15': mt5.TIMEFRAME_M15,
            'H1' : mt5.TIMEFRAME_H1,
            'H4' : mt5.TIMEFRAME_H4,
        }
    return _mt5_timeframe_consts[timeframe]


def _mt5_ready() -> bool:
    if not MT5_AVAILABLE:
        return False
    return mt5.terminal_info() is not None or mt5.initialize()


def _from_mt5(pair: str, timeframe: str, start_date: datetime,
             end_date: datetime) -> pd.DataFrame:
    rates = mt5.copy_rates_range(pair, _mt5_timeframe_const(timeframe),
                                 start_date, end_date)
    if rates is None or len(rates) == 0:
        raise ValueError(
            f"get_bars: MT5 returned no data for {pair} {timeframe} "
            f"({start_date.date()} to {end_date.date()}) -- {mt5.last_error()}"
        )
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High',
                       'low': 'Low', 'close': 'Close'}, inplace=True)
    return df[['Open', 'High', 'Low', 'Close', 'tick_volume']]


def _from_csv(pair: str, timeframe: str, start_date: datetime,
             end_date: datetime) -> pd.DataFrame:
    csv_path = HISTORICAL_DIR / f"{pair}_{timeframe}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"get_bars: no CSV at {csv_path} -- MT5 isn't available on this "
            f"machine and no offline data has been exported. Run "
            f"scripts/export_historical_data.py on the VPS, then copy "
            f"data/historical/ here."
        )

    df = pd.read_csv(csv_path, parse_dates=['datetime'])
    df.rename(columns={'datetime': 'time', 'open': 'Open', 'high': 'High',
                       'low': 'Low', 'close': 'Close'}, inplace=True)
    if df['time'].dt.tz is None:
        df['time'] = df['time'].dt.tz_localize('UTC')
    df.set_index('time', inplace=True)

    df = df.loc[(df.index >= start_date) & (df.index <= end_date)]
    if df.empty:
        raise ValueError(
            f"get_bars: no rows for {pair} {timeframe} between "
            f"{start_date.date()} and {end_date.date()} in "
            f"{csv_path.relative_to(REPO_ROOT)} -- the CSV may not cover "
            f"this range"
        )
    return df[['Open', 'High', 'Low', 'Close', 'tick_volume']]


def get_bars(pair: str, timeframe: str, start_date: datetime,
            end_date: datetime) -> pd.DataFrame:
    """
    Historical OHLC bars for `pair` at `timeframe` ('M15', 'H1', or 'H4')
    between start_date and end_date (UTC-aware datetimes, inclusive).

    Returns a DataFrame with a UTC DatetimeIndex named 'time' and columns
    Open, High, Low, Close, tick_volume -- regardless of which backend
    served it:
      - MT5 installed and reachable (Windows/VPS) -> live fetch
      - otherwise (e.g. Mac)                       -> CSV files in
        data/historical/, produced by scripts/export_historical_data.py
    """
    if timeframe not in TIMEFRAMES:
        raise ValueError(
            f"get_bars: timeframe must be one of {TIMEFRAMES}, got {timeframe!r}"
        )

    if _mt5_ready():
        return _from_mt5(pair, timeframe, start_date, end_date)
    return _from_csv(pair, timeframe, start_date, end_date)
