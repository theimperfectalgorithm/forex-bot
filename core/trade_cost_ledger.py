"""Append-only, ticket-keyed accounting sidecar for closed MT5 positions.

The legacy trades_log.csv intentionally remains gross-P&L-only for backward
compatibility.  This ledger stores the MT5 accounting fields needed to
reconcile account balance without rewriting historical CSV rows.
"""

from __future__ import annotations

import json
import errno
import logging
import math
import os
from pathlib import Path
import time

from core.runtime_paths import data_dir


LEDGER_FILE = data_dir() / 'accounting' / 'trade_costs.jsonl'
REQUIRED_FIELDS = frozenset({'ticket', 'gross_pnl', 'commission', 'swap', 'fee', 'net_pnl'})
_LOCK_SUFFIX = '.lock'
LOCK_STALE_AFTER_SECONDS = 300
log = logging.getLogger('TRADE_COST_LEDGER')


def _number(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def aggregate_position_deals(deals, ticket: int) -> dict:
    """Aggregate every supplied deal belonging to one MT5 position exactly once."""
    position_deals = []
    seen_ids = set()
    for deal in (deals or []):
        if getattr(deal, 'position_id', None) != ticket:
            continue
        deal_id = getattr(deal, 'ticket', None)
        if deal_id is None:
            deal_id = getattr(deal, 'deal', None)
        # If MT5 exposes no identifier, do not invent one: count the record.
        if deal_id not in (None, 0, ''):
            try:
                key = str(deal_id)
            except Exception:
                key = None
            if key is not None:
                if key in seen_ids:
                    continue
                seen_ids.add(key)
        position_deals.append(deal)
    gross = sum(_number(getattr(d, 'profit', 0.0)) for d in position_deals)
    commission = sum(_number(getattr(d, 'commission', 0.0)) for d in position_deals)
    swap = sum(_number(getattr(d, 'swap', 0.0)) for d in position_deals)
    fee = sum(_number(getattr(d, 'fee', 0.0)) for d in position_deals)
    return {
        'ticket': int(ticket),
        'gross_pnl': round(gross, 2),
        'commission': round(commission, 2),
        'swap': round(swap, 2),
        'fee': round(fee, 2),
        'net_pnl': round(gross + commission + swap + fee, 2),
        'deal_count': len(position_deals),
    }


def load_cost_ledger(path: Path | None = None) -> dict[str, dict]:
    """Return the last complete accounting record per ticket; malformed/incomplete rows are ignored."""
    path = path or LEDGER_FILE
    records: dict[str, dict] = {}
    if not path.exists():
        return records
    try:
        with open(path, encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    ticket = str(record['ticket'])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
                if _is_complete(record):
                    records[ticket] = record
    except OSError:
        return records
    return records


def _is_complete(record: dict) -> bool:
    if not isinstance(record, dict) or not REQUIRED_FIELDS.issubset(record):
        return False
    try:
        return bool(str(record['ticket'])) and all(
            isinstance(record[field], (int, float)) and not isinstance(record[field], bool)
            and math.isfinite(record[field])
            for field in REQUIRED_FIELDS - {'ticket'})
    except (KeyError, TypeError):
        return False


class _ExclusiveFileLock:
    def __init__(self, path: Path, timeout: float = 5.0):
        self.path, self.timeout, self.fd = path, timeout, None

    def __enter__(self):
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(self.fd, str(os.getpid()).encode('ascii'))
                except Exception:
                    os.close(self.fd)
                    self.fd = None
                    self.path.unlink(missing_ok=True)
                    raise
                return self
            except FileExistsError:
                self._recover_stale_lock()
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'ledger lock timeout: {self.path}')
                time.sleep(0.01)

    @staticmethod
    def _process_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError as exc:
            # ESRCH/EINVAL (and Windows ERROR_INVALID_PARAMETER) mean dead or
            # invalid PID; permission-denied/other states are kept active.
            if exc.errno in (errno.ESRCH, errno.EINVAL) or getattr(exc, 'winerror', None) == 87:
                return False
            return True
        except ValueError:
            return False

    def _recover_stale_lock(self):
        """Remove only old locks whose recorded owner is no longer alive.

        Five minutes is deliberately far longer than this append operation;
        age alone is insufficient, so the owner PID must also be dead.
        Malformed lock files are retained because they are not demonstrably
        abandoned.
        """
        try:
            age = time.time() - self.path.stat().st_mtime
            if age <= LOCK_STALE_AFTER_SECONDS:
                return
            raw = self.path.read_text(encoding='ascii').strip()
            pid = int(raw)
            if not self._process_alive(pid):
                self.path.unlink()
        except (FileNotFoundError, OSError, ValueError):
            return

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.fd is not None:
                os.close(self.fd)
        finally:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


def append_cost_record(record: dict, path: Path | None = None) -> bool:
    """Append one ticket record once. Returns False when it already exists or cannot be saved."""
    path = path or LEDGER_FILE
    ticket = str(record.get('ticket') or '') if isinstance(record, dict) else ''
    if not ticket or not _is_complete(record):
        log.error('Cost-ledger write rejected for ticket=%s; accounting coverage is incomplete', ticket or '<unknown>')
        return False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _ExclusiveFileLock(path.with_name(path.name + _LOCK_SUFFIX)):
            if ticket in load_cost_ledger(path):
                return False
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, sort_keys=True) + '\n')
                f.flush()
        return True
    except (OSError, TimeoutError, TypeError):
        log.error('Cost-ledger write failed for ticket=%s; accounting coverage is incomplete', ticket,
                  exc_info=True)
        return False
