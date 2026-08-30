"""
health_monitor -- the "backpropagation loop" applied at the SYSTEM level:
compare each strategy's LIVE results against its BACKTEST expectation,
compute how statistically abnormal any gap is, and grade it.

No machine learning, deliberately: pure binomial statistics on win rate,
fully auditable. Runs inside the daily report (main_agent step_report)
and standalone: `python -m core.health_monitor` from the repo root.

Grades per strategy key:
  GREEN : live win rate consistent with the backtest expectation
  AMBER : P(observing this WR or worse | backtest WR) < 10%  -> watch
  RED   : P < 2% with n >= 10                                -> act
          (recommendation is logged loudly; auto-pause is deliberately
          NOT implemented in v1 -- pausing stays a human decision, per
          the project's standing checkpoint rules)

Expectations come from the walk-forward validations (see
PROJECT_REPORT.md section 3); win rates are the conservative of IS/OOS.

Data source: data/journal/events.jsonl (entry+exit rows joined on
ticket). The journal starts 2026-07-15; earlier trades are not graded.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from core.runtime_paths import data_dir

JOURNAL_FILE = data_dir() / 'journal' / 'events.jsonl'

# expected win rates from the validation record (conservative pick)
EXPECTED_WR = {
    'GBPJPY@arb': 0.48, 'CADJPY@arb': 0.46, 'XAUUSD@arb': 0.45,
    'GBPJPY@amr': 0.62, 'EURJPY@amr': 0.58, 'AUDJPY@amr': 0.58,
    'CADJPY@amr': 0.58, 'GBPUSD@mon': 0.62,
}
DEFAULT_WR = 0.50

AMBER_P = 0.10
RED_P   = 0.02
RED_MIN_N = 10


def _binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p). Exact; n is small here."""
    total = 0.0
    for i in range(k + 1):
        total += math.comb(n, i) * p**i * (1 - p)**(n - i)
    return min(1.0, total)


def load_closed_trades() -> list:
    """Join entry/exit journal events by ticket -> one dict per closed
    trade with key, pnl, and (when available) entry context."""
    if not JOURNAL_FILE.exists():
        return []
    entries, exits = {}, {}
    with open(JOURNAL_FILE, encoding='utf-8') as f:
        for line in f:
            try:
                ev = json.loads(line)
            except Exception:
                continue
            t = ev.get('ticket')
            if ev.get('kind') == 'entry' and t:
                entries[t] = ev
            elif ev.get('kind') == 'exit' and t:
                exits[t] = ev
    out = []
    for t, ex in exits.items():
        row = {'ticket': t, 'key': ex.get('key'), 'pnl': ex.get('pnl'),
               'exit_reason': ex.get('exit_reason'),
               'hold_hours': ex.get('hold_hours')}
        if t in entries:
            row['entry'] = entries[t]
        if row['pnl'] is not None:
            out.append(row)
    return out


def run(log: logging.Logger | None = None) -> list:
    """Grade every strategy key with >= 3 closed journaled trades.
    Returns the report rows; logs a summary (WARNING for AMBER/RED)."""
    log = log or logging.getLogger('HEALTH')
    trades = load_closed_trades()
    by_key = {}
    for tr in trades:
        by_key.setdefault(tr['key'], []).append(tr)

    rows = []
    for key, ts in sorted(by_key.items()):
        n = len(ts)
        if n < 3:
            continue
        wins = sum(1 for t in ts if (t['pnl'] or 0) > 0)
        net = sum(t['pnl'] or 0 for t in ts)
        exp = EXPECTED_WR.get(key, DEFAULT_WR)
        p = _binom_cdf(wins, n, exp)   # P(this many wins or fewer)
        if p < RED_P and n >= RED_MIN_N:
            grade = 'RED'
        elif p < AMBER_P:
            grade = 'AMBER'
        else:
            grade = 'GREEN'
        rows.append({'key': key, 'n': n, 'wins': wins,
                     'wr': round(wins / n * 100, 1),
                     'expected_wr': round(exp * 100, 1),
                     'net': round(net, 2), 'p_value': round(p, 4),
                     'grade': grade})
        line = (f"HEALTH {grade:5} {key:12} n={n:>3} "
                f"WR={wins}/{n} ({wins/n*100:.0f}% vs exp {exp*100:.0f}%) "
                f"net=${net:+,.2f}  p={p:.3f}")
        if grade == 'RED':
            log.warning(line + "  -> RECOMMEND PAUSE (human decision "
                               "per standing checkpoint rules)")
        elif grade == 'AMBER':
            log.warning(line)
        else:
            log.info(line)

    if not rows:
        log.info("HEALTH: no strategy has >= 3 journaled closed trades yet "
                 "(journal started 2026-07-15)")
    return rows


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s  %(levelname)-8s  %(message)s')
    run()
