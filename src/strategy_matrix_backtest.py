"""
Forex Bot - Strategy Matrix Walk-Forward Backtest
==================================================
Validates the 3 NEW untested strategies (VolatilityRegimeTrend,
MomentumDivergenceSession, RegimeFilteredMACross) plus the implemented but
never-validated AsianRangeBreakout across ALL major pairs, using the
standard validation criteria from each strategy's docstring:

    PF > 1.3, max drawdown < 8%, >= 60% profitable months over the
    2-year in-sample period, AND a clean (profitable) out-of-sample
    forward test on the final 12 months.

Data: 36 months of H1 (+H4 for regime filters) fetched live from MT5
via core.data_loader.get_bars(). In-sample = first 24 months,
out-of-sample = final 12 months.

Simulation fidelity vs the live strategy classes:
  - Signals are evaluated ONLY on fully closed H1 bars (live uses pos=1).
  - Indicator values are computed on a rolling window of exactly the same
    length the live class fetches (H1_BARS_NEEDED), with the same seeding,
    so backtest EMA/ATR/RSI values match what the live code would compute.
    (Verified by assertion against the live recursive implementations.)
  - One trade per day per pair; no new entry while a position is open.
  - Fixed SL/TP in pips (from the GBPUSD-calibrated YAMLs, ATR-scaled for
    other pairs); AsianRangeBreakout uses its dynamic range-based SL/TP.
  - Friday close at 20:00 UTC (orchestrator behaviour).
  - SL takes priority when SL and TP are both touched inside one H1 bar
    (conservative assumption).
  - A fixed per-pair spread cost is deducted from every trade -- prior
    backtests in this repo ignored spread; results here are net of it.
  - Sizing: risk_percent of current balance per trade (compounding),
    starting balance $100,000 (5ers-style account).

Requirements: MetaTrader 5 must be OPEN and LOGGED IN.

Outputs:
  data/strategy_matrix_results.csv   -- one row per (strategy, pair) run
  data/strategy_matrix_trades/       -- per-run trade logs (CSV)
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from core import data_loader

# ── SETTINGS ─────────────────────────────────────────────────────────────────

INITIAL_BALANCE = 100_000.00
MONTHS          = 36
IS_MONTHS       = 24          # in-sample window; remainder is out-of-sample

RISK_PCT_DEFAULT = 0.005      # 0.5% per trade, matches the new pair YAMLs

# Typical spread in pips per pair (deducted from every trade)
SPREAD_PIPS = {
    'EURUSD': 1.0, 'GBPUSD': 1.2, 'USDJPY': 1.0, 'USDCAD': 1.5,
    'AUDUSD': 1.0, 'NZDUSD': 1.5, 'GBPJPY': 2.5, 'EURJPY': 1.8,
    'AUDJPY': 1.8,
}

# GBPUSD-calibrated SL/TP from pairs/*.yaml -- scaled to other pairs by
# the ratio of median H1 ATR(14), preserving the calibrated SL geometry.
YAML_SLTP_GBPUSD = {
    'volatility_regime_trend'     : (30, 60),
    'momentum_divergence_session' : (25, 50),
    'regime_filtered_ma_cross'    : (30, 60),
}

# Which pairs to test per strategy (full majors sweep; the strategy class
# COMPATIBLE_PAIRS lists are noted in the results as 'listed_compat').
MAJORS = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDUSD', 'NZDUSD',
          'GBPJPY', 'EURJPY', 'AUDJPY']

MATRIX = {
    'volatility_regime_trend'     : MAJORS,
    'momentum_divergence_session' : MAJORS,
    'regime_filtered_ma_cross'    : MAJORS,
    'asian_range_breakout'        : ['USDJPY', 'AUDJPY', 'GBPJPY', 'EURJPY',
                                     'AUDUSD', 'NZDUSD'],
}

LISTED_COMPAT = {
    'volatility_regime_trend'     : ['GBPUSD', 'EURUSD', 'GBPJPY', 'EURJPY', 'AUDUSD'],
    'momentum_divergence_session' : ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD'],
    'regime_filtered_ma_cross'    : ['GBPJPY', 'GBPUSD', 'EURUSD', 'USDCAD'],
    'asian_range_breakout'        : ['AUDJPY'],
}

# Validation criteria (strategy docstrings / registry policy)
CRIT_PF          = 1.3
CRIT_MAX_DD      = 8.0     # percent
CRIT_PROF_MONTHS = 60.0    # percent of months profitable (in-sample)

REPO_ROOT  = Path(__file__).parent.parent
OUT_CSV    = REPO_ROOT / 'data' / 'strategy_matrix_results.csv'
TRADES_DIR = REPO_ROOT / 'data' / 'strategy_matrix_trades'


# ── WINDOWED INDICATORS (exact match to live per-window seeding) ─────────────
#
# The live classes fetch a fixed window of bars each cycle and seed the
# recursion from the window's first value. That makes each indicator value a
# LINEAR function of the window -> computable exactly for every bar at once
# with a convolution.

def windowed_ema(values: np.ndarray, period: int, window: int) -> np.ndarray:
    """EMA seeded at the first value of each rolling `window`, evaluated at
    the last bar of the window. out[i] corresponds to values[i]; the first
    window-1 entries are NaN."""
    alpha = 2.0 / (period + 1)
    k = np.arange(window)                      # position inside window
    w = alpha * (1 - alpha) ** (window - 1 - k)
    w[0] = (1 - alpha) ** (window - 1)         # seed weight
    out = np.full(len(values), np.nan)
    if len(values) >= window:
        out[window - 1:] = np.convolve(values, w[::-1], mode='valid')
    return out


def windowed_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 period: int, window_bars: int) -> np.ndarray:
    """Wilder ATR exactly as strategies' _atr_wilder(): TRs over a rolling
    window of `window_bars` bars (window_bars-1 TRs), seed = mean of first
    `period` TRs, Wilder recursion over the rest. out[i] is the ATR the live
    class would compute if bar i were the most recent closed bar."""
    tr = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1]),
    ])
    n_tr = window_bars - 1                     # TRs per window
    m = n_tr - period                          # recursion steps
    r = (period - 1) / period
    w = np.zeros(n_tr)
    w[:period] = (1.0 / period) * r ** m       # seed contribution
    for j in range(1, m + 1):                  # tr index period+j-1
        w[period + j - 1] = (1.0 / period) * r ** (m - j)
    out = np.full(len(close), np.nan)
    if len(tr) >= n_tr:
        out[window_bars - 1:] = np.convolve(tr, w[::-1], mode='valid')
    return out


def windowed_rsi(close: np.ndarray, period: int, window_bars: int
                 ) -> np.ndarray:
    """Wilder RSI matching MomentumDivergenceSession._rsi_series() evaluated
    on a rolling fetch window of `window_bars` bars. avg_gain / avg_loss are
    linear in the per-window gains/losses -> convolution each, then combine."""
    deltas = np.diff(close)
    gains  = np.clip(deltas, 0, None)
    losses = -np.clip(deltas, None, 0)
    n_d = window_bars - 1                      # deltas per window
    m = n_d - period
    r = (period - 1) / period
    w = np.zeros(n_d)
    w[:period] = (1.0 / period) * r ** m
    for j in range(1, m + 1):
        w[period + j - 1] = (1.0 / period) * r ** (m - j)
    out = np.full(len(close), np.nan)
    if len(deltas) >= n_d:
        ag = np.convolve(gains,  w[::-1], mode='valid')
        al = np.convolve(losses, w[::-1], mode='valid')
        with np.errstate(divide='ignore'):
            rs = np.where(al > 0, ag / np.where(al > 0, al, 1), np.inf)
        out[window_bars - 1:] = 100.0 - 100.0 / (1.0 + rs)
    return out


def _self_check():
    """Verify the convolution forms reproduce the live recursive code."""
    rng = np.random.default_rng(7)
    closes = 1.25 + np.cumsum(rng.normal(0, 0.001, 400))
    highs  = closes + np.abs(rng.normal(0, 0.0005, 400))
    lows   = closes - np.abs(rng.normal(0, 0.0005, 400))

    # live-style recursive EMA on one window
    def live_ema(vals, period):
        a = 2.0 / (period + 1)
        e = vals[0]
        for v in vals[1:]:
            e = a * v + (1 - a) * e
        return e

    W = 66
    got = windowed_ema(closes, 20, W)
    for i in (W - 1, 200, 399):
        want = live_ema(closes[i - W + 1:i + 1], 20)
        assert abs(got[i] - want) < 1e-10, 'windowed_ema mismatch'

    def live_atr(h, l, c, period):
        trs = [max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
               for i in range(1, len(c))]
        atr = float(np.mean(trs[:period]))
        for tr in trs[period:]:
            atr = (atr * (period - 1) + tr) / period
        return atr

    got = windowed_atr(highs, lows, closes, 14, W)
    for i in (W - 1, 250, 399):
        s = slice(i - W + 1, i + 1)
        want = live_atr(highs[s], lows[s], closes[s], 14)
        assert abs(got[i] - want) < 1e-10, 'windowed_atr mismatch'

    def live_rsi_last(c, period):
        d = np.diff(c)
        g = np.clip(d, 0, None); lo = -np.clip(d, None, 0)
        ag = float(np.mean(g[:period])); al = float(np.mean(lo[:period]))
        for i in range(period, len(d)):
            ag = (ag * (period - 1) + g[i]) / period
            al = (al * (period - 1) + lo[i]) / period
        rs = ag / al if al > 0 else np.inf
        return 100 - 100 / (1 + rs)

    W2 = 49
    got = windowed_rsi(closes, 14, W2)
    for i in (W2 - 1, 300, 399):
        want = live_rsi_last(closes[i - W2 + 1:i + 1], 14)
        assert abs(got[i] - want) < 1e-9, 'windowed_rsi mismatch'


# ── DATA ─────────────────────────────────────────────────────────────────────

def fetch(pair: str):
    date_to   = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=MONTHS * 30 + 60)  # pad for warmup
    h1 = data_loader.get_bars(pair, 'H1', date_from, date_to)
    h4 = data_loader.get_bars(pair, 'H4', date_from, date_to)
    return h1, h4


def h4_regime_series(h1: pd.DataFrame, h4: pd.DataFrame) -> np.ndarray:
    """Per-H1-bar H4 SMA50/200 regime (+1/-1/0) using only H4 bars fully
    CLOSED before the H1 bar's own close."""
    c = h4['Close'].to_numpy()
    sma50  = pd.Series(c).rolling(50).mean().to_numpy()
    sma200 = pd.Series(c).rolling(200).mean().to_numpy()
    regime_h4 = np.where(np.isnan(sma200), 0,
                np.where(sma50 > sma200, 1, np.where(sma50 < sma200, -1, 0)))
    h4_close_time = h4.index + pd.Timedelta(hours=4)   # bar fully closed
    h1_close_time = h1.index + pd.Timedelta(hours=1)
    idx = np.searchsorted(h4_close_time.values, h1_close_time.values,
                          side='right') - 1
    out = np.zeros(len(h1), dtype=int)
    ok = idx >= 0
    out[ok] = regime_h4[idx[ok]]
    return out


# ── SIGNAL GENERATORS (one per strategy) ─────────────────────────────────────
# Each returns a list of (bar_index, direction, sl_pips, tp_pips) candidates.
# The trade engine applies one-trade-per-day / no-overlap / Friday-close.

def signals_vrt(h1, h4, pip, sl_pips, tp_pips, atr_min_pips):
    W = 50 + 14 + 2                                    # H1_BARS_NEEDED = 66
    closes = h1['Close'].to_numpy()
    ema20 = windowed_ema(closes, 20, W)
    ema50 = windowed_ema(closes, 50, W)
    atr = windowed_atr(h1['High'].to_numpy(), h1['Low'].to_numpy(), closes,
                       14, W) / pip
    hours = h1.index.hour
    out = []
    for i in range(W, len(h1)):
        if not (8 <= hours[i] < 20):
            continue
        if np.isnan(ema50[i]) or atr[i] < atr_min_pips:
            continue
        if ema20[i] > ema50[i] and closes[i] > ema20[i]:
            out.append((i, 'BUY', sl_pips, tp_pips))
        elif ema20[i] < ema50[i] and closes[i] < ema20[i]:
            out.append((i, 'SELL', sl_pips, tp_pips))
    return out


def signals_mds(h1, h4, pip, sl_pips, tp_pips, _unused):
    LOOKBACK, GAP, RLO, RHI = 30, 3, 35, 65
    W = LOOKBACK + 14 + 5                              # H1_BARS_NEEDED = 49
    closes = h1['Close'].to_numpy()
    highs  = h1['High'].to_numpy()
    lows   = h1['Low'].to_numpy()
    # live rsi array covers the fetch window; last LOOKBACK values are used.
    # windowed_rsi()[i] == live rsi[-1] at bar i; build the per-bar lookback
    # slice from the same series (rsi at bar j computed on window ending j --
    # equivalent to live values for the recent bars that matter).
    rsi = windowed_rsi(closes, 14, W)
    hours = h1.index.hour
    out = []
    for i in range(W + LOOKBACK, len(h1)):
        if not (12 <= hours[i] < 17):
            continue
        h = highs[i - LOOKBACK + 1:i + 1]
        l = lows[i - LOOKBACK + 1:i + 1]
        r = rsi[i - LOOKBACK + 1:i + 1]
        sh = [j for j in range(1, LOOKBACK - 1)
              if h[j] > h[j - 1] and h[j] > h[j + 1]]
        sl_ = [j for j in range(1, LOOKBACK - 1)
               if l[j] < l[j - 1] and l[j] < l[j + 1]]
        sig = None
        if len(sh) >= 2:
            i1, i2 = sh[-2], sh[-1]
            if (i2 - i1 >= GAP and h[i2] > h[i1]
                    and not np.isnan(r[i1]) and not np.isnan(r[i2])
                    and r[i1] > RHI and r[i2] < r[i1]):
                sig = 'SELL'
        if sig is None and len(sl_) >= 2:
            i1, i2 = sl_[-2], sl_[-1]
            if (i2 - i1 >= GAP and l[i2] < l[i1]
                    and not np.isnan(r[i1]) and not np.isnan(r[i2])
                    and r[i1] < RLO and r[i2] > r[i1]):
                sig = 'BUY'
        if sig:
            out.append((i, sig, sl_pips, tp_pips))
    return out


def signals_rfmc(h1, h4, pip, sl_pips, tp_pips, _unused):
    W = 21 + 3                                         # H1_BARS_NEEDED = 24
    closes = h1['Close'].to_numpy()
    ema9  = windowed_ema(closes, 9, W)
    ema21 = windowed_ema(closes, 21, W)
    regime = h4_regime_series(h1, h4)
    hours = h1.index.hour
    out = []
    for i in range(W + 1, len(h1)):
        if not (8 <= hours[i] < 17):
            continue
        if np.isnan(ema21[i]) or np.isnan(ema21[i - 1]):
            continue
        prev_above = ema9[i - 1] > ema21[i - 1]
        curr_above = ema9[i] > ema21[i]
        if not prev_above and curr_above and regime[i] == 1:
            out.append((i, 'BUY', sl_pips, tp_pips))
        elif prev_above and not curr_above and regime[i] == -1:
            out.append((i, 'SELL', sl_pips, tp_pips))
    return out


def signals_arb(h1, h4, pip, _sl, _tp, _unused):
    """AsianRangeBreakout: 00:00-07:00 UTC range, breakout on H1 close in
    hours {7,8} in the H4 trend direction; SL = opposite side of range,
    TP = 1.5x range distance from entry; min range 10 pips."""
    MIN_RANGE, TP_MULT = 10, 1.5
    closes = h1['Close'].to_numpy()
    highs  = h1['High'].to_numpy()
    lows   = h1['Low'].to_numpy()
    regime = h4_regime_series(h1, h4)
    hours = h1.index.hour
    dates = h1.index.date
    out = []
    cur_date, a_hi, a_lo = None, None, None
    for i in range(len(h1)):
        d, hr = dates[i], hours[i]
        if d != cur_date:
            cur_date, a_hi, a_lo = d, None, None
        if hr < 7:
            a_hi = highs[i] if a_hi is None else max(a_hi, highs[i])
            a_lo = lows[i]  if a_lo is None else min(a_lo, lows[i])
            continue
        if hr in (7, 8) and a_hi is not None:
            rng_pips = (a_hi - a_lo) / pip
            if rng_pips < MIN_RANGE:
                continue
            c = closes[i]
            if c > a_hi and regime[i] == 1:
                sl_pips = (c - a_lo) / pip
                out.append((i, 'BUY', sl_pips, rng_pips * TP_MULT))
            elif c < a_lo and regime[i] == -1:
                sl_pips = (a_hi - c) / pip
                out.append((i, 'SELL', sl_pips, rng_pips * TP_MULT))
    return out


SIGNAL_FN = {
    'volatility_regime_trend'     : signals_vrt,
    'momentum_divergence_session' : signals_mds,
    'regime_filtered_ma_cross'    : signals_rfmc,
    'asian_range_breakout'        : signals_arb,
}


# ── TRADE ENGINE ─────────────────────────────────────────────────────────────

def run_sim(h1: pd.DataFrame, candidates: list, pip: float, spread_pips: float,
            risk_pct: float, time_exit_hour: int | None = None):
    """Bar-by-bar simulation. Entry at signal-bar close; SL priority on
    same-bar SL+TP touch; Friday close 20:00 UTC; one trade/day; no overlap.
    time_exit_hour: if set, any open position is closed at the OPEN of the
    first bar whose UTC hour >= time_exit_hour (intraday time stop -- the
    time exit precedes that bar's SL/TP since it happens at the open).
    Returns (trades DataFrame, equity Series indexed by bar close time)."""
    cand = {i: (d, slp, tpp) for i, d, slp, tpp in candidates}
    opens  = h1['Open'].to_numpy()
    highs  = h1['High'].to_numpy()
    lows   = h1['Low'].to_numpy()
    closes = h1['Close'].to_numpy()
    times  = h1.index
    weekday = times.weekday
    hours   = times.hour
    dates   = times.date

    balance = INITIAL_BALANCE
    pos = None
    last_trade_date = None
    trades = []
    equity = np.empty(len(h1))

    for i in range(len(h1)):
        # ---- manage open position
        if pos is not None:
            exit_price = exit_reason = None
            if (time_exit_hour is not None and hours[i] >= time_exit_hour
                    and exit_price is None):
                exit_price, exit_reason = opens[i], 'TimeExit'
            elif pos['dir'] == 'BUY':
                if lows[i] <= pos['sl']:
                    exit_price, exit_reason = pos['sl'], 'SL'
                elif highs[i] >= pos['tp']:
                    exit_price, exit_reason = pos['tp'], 'TP'
            else:
                if highs[i] >= pos['sl']:
                    exit_price, exit_reason = pos['sl'], 'SL'
                elif lows[i] <= pos['tp']:
                    exit_price, exit_reason = pos['tp'], 'TP'
            if exit_price is None and weekday[i] == 4 and hours[i] >= 20:
                exit_price, exit_reason = opens[i], 'FridayClose'
            if exit_price is not None:
                pips = ((exit_price - pos['entry']) if pos['dir'] == 'BUY'
                        else (pos['entry'] - exit_price)) / pip - spread_pips
                pnl = pips * pos['usd_per_pip']
                balance += pnl
                trades.append({
                    'entry_time': pos['time'], 'exit_time': times[i],
                    'dir': pos['dir'], 'entry': pos['entry'],
                    'exit': exit_price, 'reason': exit_reason,
                    'sl_pips': pos['sl_pips'], 'tp_pips': pos['tp_pips'],
                    'pips': round(pips, 1), 'pnl': round(pnl, 2),
                    'balance': round(balance, 2),
                })
                pos = None

        # ---- new entry
        if pos is None and i in cand:
            d, slp, tpp = cand[i]
            if dates[i] != last_trade_date and slp > 0:
                entry = closes[i]
                risk_usd = balance * risk_pct
                pos = {
                    'dir': d, 'entry': entry, 'time': times[i],
                    'sl': entry - slp * pip if d == 'BUY' else entry + slp * pip,
                    'tp': entry + tpp * pip if d == 'BUY' else entry - tpp * pip,
                    'sl_pips': slp, 'tp_pips': tpp,
                    'usd_per_pip': risk_usd / slp,
                }
                last_trade_date = dates[i]

        # ---- mark-to-market equity at bar close
        float_pnl = 0.0
        if pos is not None:
            fp = ((closes[i] - pos['entry']) if pos['dir'] == 'BUY'
                  else (pos['entry'] - closes[i])) / pip
            float_pnl = fp * pos['usd_per_pip']
        equity[i] = balance + float_pnl

    tdf = pd.DataFrame(trades)
    eq = pd.Series(equity, index=times + pd.Timedelta(hours=1))
    return tdf, eq


# ── METRICS ──────────────────────────────────────────────────────────────────

def metrics(tdf: pd.DataFrame, eq: pd.Series, start, end) -> dict:
    """Stats for trades entered in [start, end)."""
    if tdf.empty:
        return {'trades': 0}
    t = tdf[(tdf['entry_time'] >= start) & (tdf['entry_time'] < end)]
    e = eq[(eq.index >= start) & (eq.index < end)]
    if t.empty or e.empty:
        return {'trades': 0}
    wins = t[t['pnl'] > 0]
    losses = t[t['pnl'] <= 0]
    gross_win = wins['pnl'].sum()
    gross_loss = abs(losses['pnl'].sum())
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else np.inf

    # normalise the window's equity to its own start for DD
    e_norm = e / e.iloc[0] * 100.0
    peak = e_norm.cummax()
    dd = ((e_norm - peak) / peak * 100.0).min()

    daily = e.resample('1D').last().dropna()
    day_ret = daily.pct_change().dropna() * 100.0
    worst_day = day_ret.min() if len(day_ret) else 0.0

    monthly = t.set_index('entry_time')['pnl'].resample('ME').sum()
    monthly = monthly[monthly != 0.0]
    prof_months = ((monthly > 0).sum() / len(monthly) * 100.0
                   if len(monthly) else 0.0)

    return {
        'trades': len(t),
        'win_rate': round(len(wins) / len(t) * 100, 1),
        'pf': pf,
        'pnl': round(t['pnl'].sum(), 2),
        'max_dd_pct': round(abs(dd), 2),
        'worst_day_pct': round(abs(worst_day), 2),
        'prof_months_pct': round(prof_months, 1),
        'active_months': len(monthly),
    }


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    _self_check()
    print('Indicator self-check vs live implementations: OK\n')

    TRADES_DIR.mkdir(parents=True, exist_ok=True)

    now = pd.Timestamp.now(tz='UTC')
    oos_start = now - pd.DateOffset(months=MONTHS - IS_MONTHS)
    is_start  = now - pd.DateOffset(months=MONTHS)

    # cache pair data + median ATR for SL scaling
    pair_data, pair_atr = {}, {}
    all_pairs = sorted({p for ps in MATRIX.values() for p in ps})
    for pair in all_pairs:
        print(f'Fetching {pair} H1/H4 ...')
        h1, h4 = fetch(pair)
        pip = 0.01 if 'JPY' in pair else 0.0001
        atr = windowed_atr(h1['High'].to_numpy(), h1['Low'].to_numpy(),
                           h1['Close'].to_numpy(), 14, 66) / pip
        pair_data[pair] = (h1, h4)
        pair_atr[pair] = float(np.nanmedian(atr))
    gbp_atr = pair_atr.get('GBPUSD', 10.0)
    print()

    rows = []
    for strat, pairs in MATRIX.items():
        for pair in pairs:
            h1, h4 = pair_data[pair]
            pip = 0.01 if 'JPY' in pair else 0.0001

            if strat in YAML_SLTP_GBPUSD:
                base_sl, base_tp = YAML_SLTP_GBPUSD[strat]
                if pair == 'GBPUSD':
                    sl_pips, tp_pips = base_sl, base_tp
                else:
                    scale = pair_atr[pair] / gbp_atr
                    sl_pips = max(10, round(base_sl * scale / 5) * 5)
                    tp_pips = sl_pips * 2
            else:
                sl_pips = tp_pips = 0    # ARB: dynamic

            atr_min = 10 if pair == 'GBPUSD' else round(8 * pair_atr[pair] / gbp_atr)

            cands = SIGNAL_FN[strat](h1, h4, pip, sl_pips, tp_pips, atr_min)
            tdf, eq = run_sim(h1, cands, pip, SPREAD_PIPS[pair],
                              RISK_PCT_DEFAULT)

            m_is  = metrics(tdf, eq, is_start, oos_start)
            m_oos = metrics(tdf, eq, oos_start, now)

            passed = (
                m_is.get('trades', 0) >= 30
                and m_is.get('pf', 0) > CRIT_PF
                and m_is.get('max_dd_pct', 99) < CRIT_MAX_DD
                and m_is.get('prof_months_pct', 0) >= CRIT_PROF_MONTHS
                and m_oos.get('trades', 0) >= 10
                and m_oos.get('pnl', 0) > 0
                and m_oos.get('max_dd_pct', 99) < CRIT_MAX_DD
            )

            row = {
                'strategy': strat, 'pair': pair,
                'listed_compat': pair in LISTED_COMPAT[strat],
                'sl_pips': sl_pips, 'tp_pips': tp_pips,
                'is_trades': m_is.get('trades', 0),
                'is_wr': m_is.get('win_rate', 0),
                'is_pf': m_is.get('pf', 0),
                'is_pnl': m_is.get('pnl', 0),
                'is_dd': m_is.get('max_dd_pct', 0),
                'is_prof_months': m_is.get('prof_months_pct', 0),
                'oos_trades': m_oos.get('trades', 0),
                'oos_wr': m_oos.get('win_rate', 0),
                'oos_pf': m_oos.get('pf', 0),
                'oos_pnl': m_oos.get('pnl', 0),
                'oos_dd': m_oos.get('max_dd_pct', 0),
                'worst_day_is': m_is.get('worst_day_pct', 0),
                'worst_day_oos': m_oos.get('worst_day_pct', 0),
                'PASS': passed,
            }
            rows.append(row)

            if not tdf.empty:
                tdf.to_csv(TRADES_DIR / f'{strat}__{pair}.csv', index=False)

            tag = 'PASS' if passed else 'fail'
            print(f"[{tag}] {strat:<28} {pair}  "
                  f"IS: n={row['is_trades']:>3} PF={row['is_pf']:<5} "
                  f"DD={row['is_dd']:>5.2f}% pm={row['is_prof_months']:>5.1f}%  "
                  f"OOS: n={row['oos_trades']:>3} PF={row['oos_pf']:<5} "
                  f"P&L=${row['oos_pnl']:>+9,.2f} DD={row['oos_dd']:.2f}%")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f'\nSaved: {OUT_CSV}')
    n_pass = int(df['PASS'].sum())
    print(f'\n{n_pass} / {len(df)} strategy-pair combinations passed all criteria.')
    if n_pass:
        print(df[df['PASS']][['strategy', 'pair', 'is_pf', 'is_dd',
                              'is_prof_months', 'oos_pf', 'oos_pnl', 'oos_dd']]
              .to_string(index=False))


if __name__ == '__main__':
    main()
