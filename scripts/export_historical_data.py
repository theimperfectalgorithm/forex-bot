"""
Export historical OHLC data from MT5 to CSV files for offline backtesting
on machines without MT5 (e.g. Mac).

Run this on the Windows/VPS machine, with the MetaTrader 5 terminal OPEN
and LOGGED IN. It has no effect on Mac -- MT5 is Windows-only, so there is
no live data to export from there.

Output: data/historical/{PAIR}_{TIMEFRAME}.csv
Columns: datetime, open, high, low, close, tick_volume

Usage:
    python scripts/export_historical_data.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

import pandas as pd

REPO_ROOT  = Path(__file__).parent.parent
OUTPUT_DIR = REPO_ROOT / 'data' / 'historical'

PAIRS       = ["GBPJPY", "EURJPY", "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "NZDUSD"]
TIMEFRAMES  = ['M15', 'H1', 'H4']
START_DATE  = datetime(2020, 1, 1, tzinfo=timezone.utc)


def connect() -> None:
    if not MT5_AVAILABLE:
        print("ERROR: MetaTrader5 package not installed. This script exports "
              "live MT5 history, so it must run on the Windows/VPS machine "
              "with MetaTrader 5 installed and logged in.")
        sys.exit(1)
    if not mt5.initialize():
        print(f"ERROR: MT5 initialize() failed -- {mt5.last_error()}")
        print("Make sure MetaTrader 5 is open and logged in.")
        sys.exit(1)
    info    = mt5.terminal_info()
    account = mt5.account_info()
    print(f"Connected : {info.name}")
    print(f"Account   : {account.login}  ({account.server})\n")


def export_pair_timeframe(pair: str, tf_name: str, tf_const: int,
                          date_to: datetime) -> int:
    rates = mt5.copy_rates_range(pair, tf_const, START_DATE, date_to)
    if rates is None or len(rates) == 0:
        print(f"  [-] {pair:<8} {tf_name:<4}: no data returned "
              f"-- {mt5.last_error()}")
        return 0

    df = pd.DataFrame(rates)
    df['datetime'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df = df[['datetime', 'open', 'high', 'low', 'close', 'tick_volume']]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{pair}_{tf_name}.csv"
    df.to_csv(out_path, index=False)

    print(f"  [+] {pair:<8} {tf_name:<4}: {len(df):>7,} bars  "
          f"({df['datetime'].iloc[0].date()} to {df['datetime'].iloc[-1].date()})  "
          f"-> {out_path.relative_to(REPO_ROOT)}")
    return len(df)


def main() -> None:
    connect()

    tf_consts = {
        'M15': mt5.TIMEFRAME_M15,
        'H1' : mt5.TIMEFRAME_H1,
        'H4' : mt5.TIMEFRAME_H4,
    }
    date_to = datetime.now(timezone.utc)

    print(f"Exporting {len(PAIRS)} pairs x {len(TIMEFRAMES)} timeframes, "
          f"{START_DATE.date()} to {date_to.date()}")
    print(f"Note: MT5 history depth (especially for M15) depends on your "
          f"broker -- shorter timeframes may not go back all the way to "
          f"{START_DATE.date()}.\n")

    summary = {}
    for pair in PAIRS:
        for tf_name in TIMEFRAMES:
            count = export_pair_timeframe(pair, tf_name, tf_consts[tf_name], date_to)
            summary[(pair, tf_name)] = count

    mt5.shutdown()

    print("\n" + "=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)
    for (pair, tf_name), count in summary.items():
        status = "OK" if count > 0 else "EMPTY"
        print(f"  {pair:<8} {tf_name:<4}  {count:>10,} bars   [{status}]")
    print("=" * 60)

    total = sum(summary.values())
    empty = [f"{p} {t}" for (p, t), c in summary.items() if c == 0]
    print(f"\nTotal bars exported: {total:,}")
    if empty:
        print(f"Empty (no data returned): {', '.join(empty)}")


if __name__ == '__main__':
    main()
