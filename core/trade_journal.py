"""
trade_journal -- append-only JSONL journal of every SIGNAL, REJECTION,
ENTRY and EXIT, with ML-ready context features. Built 2026-07-15 as the
data foundation for the Strategy Health Monitor (core/health_monitor.py)
and, later, meta-labeling (training a small model on our own signals'
context to size/skip them -- parked until enough rows accumulate).

Design decisions:
  - JSONL (one JSON object per line) in data/journal/events.jsonl:
    append-only, schema-flexible (new fields never break old rows),
    loads straight into pandas: pd.read_json(path, lines=True).
  - REJECTED signals are first-class records ("what we didn't take" is
    the counterfactual half of a future training set).
  - Every event carries both real-UTC and server timestamps plus the
    server offset (MT5 bar coordinates are server time -- see
    PROJECT_REPORT.md section 2.3).
  - Journaling must NEVER break trading: every public function is
    wrapped so failures log a warning and return silently.
  - The legacy data/trades_log.csv is untouched (backward compat).

Event kinds and their extra fields (all events also carry ts_utc,
ts_server, server_offset_h, kind):
  signal    : key, symbol, direction, sl_pips, tp_pips, signal_price,
              strategy_reason (the strategy's own explanation string --
              contains z-score/ATR/etc. for later feature parsing)
  rejection : signal fields + stage ('daily_limit'|'risk'|'execution')
              + reject_reason + account snapshot + market context
  entry     : signal fields + ticket, lots, fill_price, slippage_pips,
              risk_pct_config, risk_usd_intended
              + account snapshot + market context
  exit      : ticket, key, symbol, direction, exit_price, exit_reason,
              pnl (gross), gross_pnl, commission, swap, fee, net_pnl,
              entry_time, exit_time, hold_hours

Account snapshot: balance, equity, open_positions, open_risk_usd.
Market context: spread_pips, atr14_h1_pips, hour_server, dow_server,
minutes_to_next_high_news (from core.news_calendar's cached feed).
MAE/MFE are intentionally NOT captured live (would need tick tracking);
they can be reconstructed offline from bar history via entry/exit times.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

from core.mt5_time import observed_server_utc_offset_hours
from core.runtime_paths import data_dir

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

JOURNAL_DIR  = data_dir() / 'journal'
JOURNAL_FILE = JOURNAL_DIR / 'events.jsonl'

log = logging.getLogger('JOURNAL')


def _server_offset_hours() -> int:
    """Return the observed server offset, with a live-only DST fallback."""
    if MT5_AVAILABLE:
        observed = observed_server_utc_offset_hours(mt5)
        if observed is not None:
            return observed
    return 3 if 3 <= datetime.now(timezone.utc).month <= 10 else 2


def _pip_size(symbol: str) -> float:
    return (0.1 if symbol.startswith('XAU')
            else 0.01 if 'JPY' in symbol else 0.0001)


def snapshot_account() -> dict:
    """Balance/equity/open-position count/aggregate entry->SL risk."""
    out = {'balance': None, 'equity': None,
           'open_positions': None, 'open_risk_usd': None}
    try:
        if not (MT5_AVAILABLE and mt5.initialize()):
            return out
        a = mt5.account_info()
        if a:
            out['balance'] = round(a.balance, 2)
            out['equity']  = round(a.equity, 2)
        positions = mt5.positions_get() or []
        out['open_positions'] = len(positions)
        risk = 0.0
        for p in positions:
            info = mt5.symbol_info(p.symbol)
            if p.sl and info and info.trade_tick_size:
                risk += (abs(p.price_open - p.sl) / info.trade_tick_size
                         * info.trade_tick_value * p.volume)
        out['open_risk_usd'] = round(risk, 2)
    except Exception as e:
        log.warning(f'journal snapshot_account failed: {e}')
    return out


def market_context(symbol: str) -> dict:
    """Spread, H1 ATR(14), server hour/day-of-week, minutes to the next
    high-impact news event touching this symbol's currencies."""
    out = {'spread_pips': None, 'atr14_h1_pips': None,
           'hour_server': None, 'dow_server': None,
           'minutes_to_next_high_news': None}
    try:
        off = _server_offset_hours()
        srv = datetime.now(timezone.utc) + timedelta(hours=off)
        out['hour_server'] = srv.hour
        out['dow_server']  = srv.weekday()
        if not (MT5_AVAILABLE and mt5.initialize()):
            return out
        pip = _pip_size(symbol)
        tick = mt5.symbol_info_tick(symbol)
        if tick and tick.ask and tick.bid:
            out['spread_pips'] = round((tick.ask - tick.bid) / pip, 2)
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 1, 15)
        if rates is not None and len(rates) == 15:
            trs = []
            for i in range(1, 15):
                h, l, pc = rates[i]['high'], rates[i]['low'], rates[i-1]['close']
                trs.append(max(h - l, abs(h - pc), abs(l - pc)))
            out['atr14_h1_pips'] = round(sum(trs) / len(trs) / pip, 1)
    except Exception as e:
        log.warning(f'journal market_context({symbol}) failed: {e}')
    try:
        from core.news_calendar import _get_events
        curs = {symbol[:3].upper(), symbol[3:6].upper()}
        now = datetime.now(timezone.utc)
        deltas = []
        for ev in _get_events():
            if ev['currency'] in curs:
                t = datetime.fromisoformat(ev['time_utc'])
                if t >= now:
                    deltas.append((t - now).total_seconds() / 60)
        if deltas:
            out['minutes_to_next_high_news'] = round(min(deltas))
    except Exception:
        pass
    return out


def log_event(kind: str, data: dict):
    """Append one event. NEVER raises into the caller."""
    try:
        JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
        off = _server_offset_hours()
        now = datetime.now(timezone.utc)
        row = {'ts_utc': now.isoformat(timespec='seconds'),
               'ts_server': (now + timedelta(hours=off)).isoformat(timespec='seconds'),
               'server_offset_h': off,
               'kind': kind, **data}
        with open(JOURNAL_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(row, default=str) + '\n')
    except Exception as e:
        log.warning(f'journal write failed ({kind}): {e}')


# -- convenience wrappers (all fail-safe via log_event) ---------------------

def log_signal(key, symbol, direction, sl_pips, tp_pips, signal_price,
               strategy_reason):
    log_event('signal', {
        'key': key, 'symbol': symbol, 'direction': direction,
        'sl_pips': sl_pips, 'tp_pips': tp_pips,
        'signal_price': signal_price, 'strategy_reason': strategy_reason})


def log_rejection(key, symbol, direction, sl_pips, tp_pips, signal_price,
                  strategy_reason, stage, reject_reason):
    log_event('rejection', {
        'key': key, 'symbol': symbol, 'direction': direction,
        'sl_pips': sl_pips, 'tp_pips': tp_pips,
        'signal_price': signal_price, 'strategy_reason': strategy_reason,
        'stage': stage, 'reject_reason': reject_reason,
        **snapshot_account(), **market_context(symbol)})


def log_entry(key, symbol, direction, sl_pips, tp_pips, signal_price,
              strategy_reason, ticket, lots, fill_price,
              risk_pct_config=None):
    pip = _pip_size(symbol)
    slip = None
    try:
        if signal_price:
            slip = round((fill_price - signal_price) / pip
                         * (1 if direction == 'BUY' else -1), 2)
    except Exception:
        pass
    acct = snapshot_account()
    risk_usd = None
    try:
        if risk_pct_config and acct.get('balance'):
            risk_usd = round(acct['balance'] * risk_pct_config, 2)
    except Exception:
        pass
    log_event('entry', {
        'key': key, 'symbol': symbol, 'direction': direction,
        'sl_pips': sl_pips, 'tp_pips': tp_pips,
        'signal_price': signal_price, 'strategy_reason': strategy_reason,
        'ticket': ticket, 'lots': lots, 'fill_price': fill_price,
        'slippage_pips': slip, 'risk_pct_config': risk_pct_config,
        'risk_usd_intended': risk_usd,
        **acct, **market_context(symbol)})


def compute_hold_hours(entry_time, exit_time):
    """Return elapsed hours for two normalized ISO-8601 timestamps."""
    try:
        if entry_time and exit_time:
            return round((datetime.fromisoformat(str(exit_time))
                          - datetime.fromisoformat(str(entry_time))
                          ).total_seconds() / 3600, 2)
    except Exception:
        return None
    return None


def log_exit(trade: dict):
    """trade = the orchestrator's open_trades dict after close detection."""
    hold = compute_hold_hours(trade.get('entry_time'), trade.get('exit_time'))
    log_event('exit', {
        'key': trade.get('strategy_key', trade.get('symbol')),
        'symbol': trade.get('symbol'),
        'direction': trade.get('direction'),
        'ticket': trade.get('ticket'),
        'exit_price': trade.get('exit_price'),
        'exit_reason': trade.get('exit_reason'),
        # ``pnl`` remains the long-standing gross field used by analytics.
        'pnl': trade.get('exit_pnl'),
        'gross_pnl': trade.get('gross_pnl', trade.get('exit_pnl')),
        'commission': trade.get('commission'),
        'swap': trade.get('swap'),
        'fee': trade.get('fee'),
        'net_pnl': trade.get('net_pnl'),
        'accounting_coverage': trade.get('accounting_coverage',
                                         'complete' if trade.get('net_pnl') is not None else 'incomplete'),
        'entry_time': trade.get('entry_time'),
        'exit_time': trade.get('exit_time'),
        'hold_hours': hold})
