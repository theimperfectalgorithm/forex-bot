"""
Trade-sequence Monte Carlo -- reusable for any strategy's trade list.

Distinct from the PORTFOLIO-level block-bootstrap Monte Carlo already in
phase6/7/9 (which resamples BLOCKS of daily P&L to simulate a whole
account's path against prop-firm rules). This module does the classic
single-strategy robustness check the master prompt asks for: shuffle
the ORDER of a strategy's own historical trades many times (the trades
themselves stay real -- only their sequence is randomized) and measure
how much the resulting equity path could plausibly have varied. A
strategy whose live-viable conclusion depends on the EXACT order its
historical trades happened to occur in is not robust.

Usage:
    from monte_carlo import run_monte_carlo
    result = run_monte_carlo(trades_df, initial_balance=100_000)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def run_monte_carlo(tdf: pd.DataFrame, initial_balance: float = 100_000.0,
                    n_runs: int = 2000, seed: int = 7) -> dict:
    """
    tdf: a trades DataFrame with a 'pnl' column (as produced by
    strategy_matrix_backtest.run_sim). Shuffles trade ORDER (not
    values) n_runs times, replays each as a simple sequential equity
    curve, and reports the distribution of outcomes.

    Returns a dict with the distribution of max drawdown and losing
    streaks across the shuffles, plus risk-of-ruin (fraction of runs
    breaching -20% of starting balance at any point along the path).

    NOTE on final P&L: it is deliberately NOT reported as a percentile
    band here. Summing the same set of trades in a different order
    still produces the identical total (addition is order-invariant),
    so "final P&L confidence interval" from pure reshuffling is
    mathematically always a single point (the historical sum) -- it
    would be reported with entirely fake-looking precision. What
    genuinely varies with trade order is the PATH those trades take to
    get there: how deep the drawdown gets and how long the losing
    streaks run, which is what actually matters for whether an account
    survives to collect that total. (A true P&L *range* would require
    resampling WITH replacement / varying trade count, which is a
    different question -- bootstrap resampling, not reordering -- and
    is intentionally out of scope for this module; the portfolio-level
    block-bootstrap in phase6/7/9 already covers that at the account
    level.)
    """
    if tdf.empty or 'pnl' not in tdf.columns:
        return {'n_trades': 0}

    pnls = tdf['pnl'].to_numpy()
    n = len(pnls)
    rng = np.random.default_rng(seed)

    max_dds    = np.empty(n_runs)
    worst_streaks = np.empty(n_runs, dtype=int)
    ruin_count = 0
    ruin_threshold = -0.20 * initial_balance

    for i in range(n_runs):
        order = rng.permutation(n)
        seq = pnls[order]
        equity = initial_balance + np.cumsum(seq)
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak)
        max_dds[i] = dd.min()
        if dd.min() <= ruin_threshold:
            ruin_count += 1
        # longest consecutive losing-trade streak in this shuffle
        losing = seq <= 0
        streak = cur = 0
        for L in losing:
            cur = cur + 1 if L else 0
            streak = max(streak, cur)
        worst_streaks[i] = streak

    pct = lambda arr, p: round(float(np.percentile(arr, p)), 2)
    return {
        'n_trades': n,
        'n_runs': n_runs,
        'historical_final_pnl': round(float(pnls.sum()), 2),
        'mc_max_dd_p50_usd': pct(max_dds, 50),
        'mc_max_dd_p95_usd': pct(max_dds, 5),   # 5th pct of dd = worst 5%
        'mc_max_dd_p50_pct': round(pct(max_dds, 50) / initial_balance * 100, 2),
        'mc_max_dd_worst_pct': round(pct(max_dds, 5) / initial_balance * 100, 2),
        'worst_losing_streak_p50': int(np.percentile(worst_streaks, 50)),
        'worst_losing_streak_p95': int(np.percentile(worst_streaks, 95)),
        'risk_of_ruin_pct': round(ruin_count / n_runs * 100, 2),
    }
