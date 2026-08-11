"""
AUDJPY AMR BUY-only -- Prospective Forward Validation Tracker
====================================================================
DATA COLLECTION ONLY. No research, no optimization, no parameter search.

Frozen candidate: existing AUDJPY AMR (signals_amr_v, z_thr=2.0,
sl_mult=1.5, end_hour=4, spread=2.0 pips, risk=0.25%), BUY-only vs
ORIGINAL (both directions), evaluated side-by-side on every qualifying
signal from the frozen start timestamp forward. Identical entry price,
spread, slippage, stop, target, holding period, and execution
assumptions on both paths -- the ONLY difference is whether SELL is
permitted. Code reference: this is exactly Model B from
src/phase22_audjpy_amr_confirmatory.py, unmodified.

START_TIME = 2026-08-11 09:45 UTC (frozen, per user instruction -- do
not include any bar before this timestamp in the prospective dataset,
though earlier bars ARE used as rolling-indicator lookback context,
exactly as a live strategy would use historical bars to compute its
current indicator values -- this is not "future information", it's the
same backward-looking computation AMR always does).

Run this script periodically (manually or via a scheduled task) to pull
new bars and update the prospective trade log. It is idempotent: each
run only appends genuinely new, previously-unrecorded signals/closes.

Immutability: data/audjpy_amr_forward_trades.csv is APPEND-ONLY. Once a
trade row is written with a final outcome, it is never edited or
deleted by this script. Corrections (if ever needed) must be made by a
human via a separate audit entry in data/audjpy_amr_forward_audit_log.jsonl,
never by silently rewriting the trades CSV.

Requirements: MetaTrader 5 OPEN and LOGGED IN (or CSV cache).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import data_loader
from strategy_matrix_backtest import windowed_atr, REPO_ROOT
from phase3b_amr_jpy_refine import signals_amr_v

PAIR = 'AUDJPY'
PIP = 0.01
Z_THR, SL_MULT, END_HOUR = 2.0, 1.5, 4   # frozen live AUDJPY AMR params, unchanged
SPREAD_PIPS = 2.0                         # frozen, unchanged
RISK_PCT = 0.0025                         # frozen, unchanged (informational only -- P&L tracked in R, not USD)
HORIZON_BARS_LOOKBACK_FOR_CONTEXT_DAYS = 60   # historical bars fetched purely as rolling-indicator context

START_TIME = pd.Timestamp('2026-08-11 09:45:00', tz='UTC')   # FROZEN, do not change
STRATEGY_VERSION = 'phase22_model_B_buy_only@55e301e353ef271b00a766cee34b294bf66edc81'

DATA_DIR = REPO_ROOT / 'data'
TRADES_LOG = DATA_DIR / 'audjpy_amr_forward_trades.csv'
STATE_FILE = DATA_DIR / 'audjpy_amr_forward_state.json'
AUDIT_LOG = DATA_DIR / 'audjpy_amr_forward_audit_log.jsonl'

TRADE_COLUMNS = [
    'signal_timestamp', 'entry_timestamp', 'direction', 'entry_price',
    'original_amr_eligible', 'buy_only_eligible',
    'sl_price', 'tp_price', 'sl_pips', 'tp_pips',
    'exit_timestamp', 'exit_reason', 'exit_price',
    'r_result', 'pnl_pips', 'mfe_atr', 'mae_atr',
    'spread_pips', 'slippage_pips', 'data_source', 'strategy_version',
    'recorded_at',
]


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {'last_processed_bar_time': None, 'open_positions': []}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def load_trades_log() -> pd.DataFrame:
    if TRADES_LOG.exists():
        return pd.read_csv(TRADES_LOG, parse_dates=['signal_timestamp', 'entry_timestamp', 'exit_timestamp'])
    return pd.DataFrame(columns=TRADE_COLUMNS)


def append_trade(row: dict):
    """Append-only. Never rewrites an existing row."""
    df = pd.DataFrame([row])
    header = not TRADES_LOG.exists()
    df.to_csv(TRADES_LOG, mode='a', header=header, index=False)


def audit_log_entry(action: str, detail: dict):
    entry = dict(timestamp=datetime.now(timezone.utc).isoformat(), action=action, detail=detail)
    with open(AUDIT_LOG, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, default=str) + '\n')


def main():
    print('=' * 90)
    print('AUDJPY AMR BUY-only -- PROSPECTIVE FORWARD VALIDATION TRACKER (data collection only)')
    print(f'Frozen start: {START_TIME}   Strategy version: {STRATEGY_VERSION}')
    print('=' * 90)

    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=HORIZON_BARS_LOOKBACK_FOR_CONTEXT_DAYS + 5)
    m15 = data_loader.get_bars(PAIR, 'M15', date_from, date_to)
    print(f'Fetched {len(m15)} bars, {m15.index[0]} to {m15.index[-1]} (context lookback + prospective window).')

    highs, lows, closes = m15['High'].to_numpy(), m15['Low'].to_numpy(), m15['Close'].to_numpy()
    atr = windowed_atr(highs, lows, closes, 14, 66) / PIP

    all_cands = signals_amr_v(m15, PIP, SPREAD_PIPS, Z_THR, SL_MULT, END_HOUR)
    prospective_cands = [(i, d, sl, tp) for (i, d, sl, tp) in all_cands if m15.index[i] >= START_TIME]
    print(f'Total signals in fetched window: {len(all_cands)}.  Prospective (>= START_TIME): {len(prospective_cands)}.')

    state = load_state()
    trades_log = load_trades_log()
    already_recorded_signals = set(pd.to_datetime(trades_log['signal_timestamp'])) if not trades_log.empty else set()

    new_closed, new_opened = 0, 0
    HORIZON = 4  # frozen AMR max holding period (bars), unchanged from all prior phases
    n = len(m15)

    for i, d, sl_pips, tp_pips in prospective_cands:
        sig_ts = m15.index[i]
        if sig_ts in already_recorded_signals:
            continue  # idempotent: already logged (closed) in a prior run

        entry_ts = sig_ts  # AMR/run_sim convention: entry at signal-bar close
        entry_px = closes[i]
        if d == 'BUY':
            entry_px_exec = entry_px + SPREAD_PIPS * PIP
            sl_px = entry_px_exec - sl_pips * PIP
            tp_px = entry_px_exec + tp_pips * PIP
        else:
            entry_px_exec = entry_px - SPREAD_PIPS * PIP
            sl_px = entry_px_exec + sl_pips * PIP
            tp_px = entry_px_exec - tp_pips * PIP

        a_entry = atr[i]
        exit_px, exit_reason, exit_ts = None, None, None
        mfe, mae = 0.0, 0.0
        for b in range(i + 1, min(i + 1 + HORIZON, n)):
            lo, hi = lows[b], highs[b]
            if d == 'BUY':
                mfe = max(mfe, (hi - entry_px_exec) / PIP / max(a_entry, 1e-9))
                mae = min(mae, (lo - entry_px_exec) / PIP / max(a_entry, 1e-9))
                if lo <= sl_px:
                    exit_px, exit_reason, exit_ts = sl_px, 'SL', m15.index[b]; break
                if hi >= tp_px:
                    exit_px, exit_reason, exit_ts = tp_px, 'TP', m15.index[b]; break
            else:
                mfe = max(mfe, (entry_px_exec - lo) / PIP / max(a_entry, 1e-9))
                mae = min(mae, (entry_px_exec - hi) / PIP / max(a_entry, 1e-9))
                if hi >= sl_px:
                    exit_px, exit_reason, exit_ts = sl_px, 'SL', m15.index[b]; break
                if lo <= tp_px:
                    exit_px, exit_reason, exit_ts = tp_px, 'TP', m15.index[b]; break

        last_seen_bar = i + HORIZON
        if exit_px is None and last_seen_bar < n:
            # horizon fully elapsed within fetched data -> time exit, resolved
            last_b = min(i + HORIZON, n - 1)
            exit_px, exit_reason, exit_ts = closes[last_b], 'TIME', m15.index[last_b]

        if exit_px is None:
            # still within its holding window and we don't have enough future
            # bars yet -- leave OPEN, do not record a final row this run
            new_opened += 1
            continue

        pnl_pips = (exit_px - entry_px_exec) / PIP if d == 'BUY' else (entry_px_exec - exit_px) / PIP
        r_result = pnl_pips / sl_pips if sl_pips > 0 else np.nan

        row = dict(
            signal_timestamp=sig_ts, entry_timestamp=entry_ts, direction=d, entry_price=entry_px_exec,
            original_amr_eligible=True, buy_only_eligible=(d == 'BUY'),
            sl_price=sl_px, tp_price=tp_px, sl_pips=sl_pips, tp_pips=tp_pips,
            exit_timestamp=exit_ts, exit_reason=exit_reason, exit_price=exit_px,
            r_result=r_result, pnl_pips=pnl_pips, mfe_atr=mfe, mae_atr=mae,
            spread_pips=SPREAD_PIPS, slippage_pips=0.0, data_source='MT5/data_loader',
            strategy_version=STRATEGY_VERSION, recorded_at=datetime.now(timezone.utc).isoformat(),
        )
        append_trade(row)
        new_closed += 1

    state['last_processed_bar_time'] = str(m15.index[-1])
    state['last_run_at'] = datetime.now(timezone.utc).isoformat()
    save_state(state)

    if new_closed or new_opened:
        audit_log_entry('tracker_run', dict(new_closed=new_closed, new_opened_pending=new_opened,
                                             last_bar=str(m15.index[-1])))

    print(f'\nNewly closed & recorded this run: {new_closed}')
    print(f'Newly opened & still pending (not yet recorded, will resolve on a future run): {new_opened}')

    trades_log = load_trades_log()
    print(f'Cumulative prospective trades recorded to date: {len(trades_log)}')
    if not trades_log.empty:
        print(trades_log[['signal_timestamp', 'direction', 'exit_reason', 'r_result']].to_string(index=False))


if __name__ == '__main__':
    main()
