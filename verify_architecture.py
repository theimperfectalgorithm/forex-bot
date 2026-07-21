"""
Verification script for the modular pair-agnostic architecture
(strategies/, pairs/, core/). Run from repo root:

    python verify_architecture.py

7 checks, all must pass:
  1. pair_manager.get_active_pairs() returns GBPJPY, EURJPY, EURUSD only
  2. USDJPY is NOT in active pairs (active: false)
  3. generate_signal() with mock data returns BUY/SELL/None for each active pair
  4. Flipping USDJPY to active: true makes it appear, but generate_signal()
     raises NotImplementedError (expected stub behavior)
  5. USDJPY flipped back to active: false
  6. All 3 active pairs' session windows match core.session_filter predicates
  7. locked_pairs allowlist correctly intersects active pairs (prop-firm
     clone isolation), and is a no-op when absent
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.pair_manager import get_active_pairs
from core import session_filter as sf

PASS = "[PASS]"
FAIL = "[FAIL]"

PAIRS_DIR = Path(__file__).parent / 'pairs'
USDJPY_YAML = PAIRS_DIR / 'USDJPY.yaml'

results = []

def check(label: str, condition: bool, detail: str = ''):
    mark = PASS if condition else FAIL
    print(f"  {mark}  {label}" + (f"  -- {detail}" if detail else ""))
    results.append(condition)


def section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


# ---------------------------------------------------------------------------
# CHECK 1 + 2 -- active pairs are exactly GBPJPY, EURJPY, EURUSD; USDJPY excluded
# ---------------------------------------------------------------------------
section("CHECK 1+2 -- Active pairs scan")

active = get_active_pairs()
active_names = sorted(name for name, _inst, _cfg in active)
# The 2026-07-05 portfolio deployment (Book B+): london_breakout and
# sma_ema_combined are DEACTIVATED (failed walk-forward audits); the
# active book is asian_range_breakout on GBPJPY/CADJPY/XAUUSD ('@arb'
# cache keys) + asian_hours_reversion on GBPJPY/EURJPY/AUDJPY/CADJPY
# ('@amr' keys). GBPJPY and CADJPY each appear twice (two strategies).
expected = sorted(['GBPJPY', 'GBPJPY', 'CADJPY', 'CADJPY',
                   'EURJPY', 'AUDJPY', 'XAUUSD', 'GBPUSD'])

check("Active pairs == Book B+ (ARB x3 + AMR x4 + Monday drift)",
      active_names == expected,
      f"got {active_names}")
check("USDJPY NOT in active pairs (active: false)",
      'USDJPY' not in active_names,
      f"active list: {active_names}")

# ---------------------------------------------------------------------------
# CHECK 3 -- generate_signal() with mock data, no errors, valid return value
# ---------------------------------------------------------------------------
section("CHECK 3 -- generate_signal() with mock data (BUY/SELL/None, no errors)")

for name, inst, cfg in active:
    try:
        result = inst.generate_signal(name, 1.10000, 1)   # mock price + bullish H4 trend
        valid = result in ('BUY', 'SELL', None)
        check(f"{name}.generate_signal(mock) -> {result!r}", valid)
    except Exception as e:
        check(f"{name}.generate_signal(mock) raised unexpected error", False, str(e))

# ---------------------------------------------------------------------------
# CHECK 4 -- flip USDJPY to active: true, confirm it appears + stub raises
# ---------------------------------------------------------------------------
section("CHECK 4 -- USDJPY active:true -> appears, generate_signal() raises NotImplementedError")

original_yaml = USDJPY_YAML.read_text(encoding='utf-8')
try:
    flipped_yaml = original_yaml.replace('active: false', 'active: true')
    USDJPY_YAML.write_text(flipped_yaml, encoding='utf-8')

    active_with_usdjpy = get_active_pairs()
    names_with_usdjpy = sorted(name for name, _inst, _cfg in active_with_usdjpy)
    check("USDJPY now appears in active pairs",
          'USDJPY' in names_with_usdjpy,
          f"active list: {names_with_usdjpy}")

    usdjpy_entry = next((t for t in active_with_usdjpy if t[0] == 'USDJPY'), None)
    if usdjpy_entry is None:
        check("USDJPY strategy instance loaded", False, "not found in active pairs after flip")
    else:
        _name, usdjpy_inst, _cfg = usdjpy_entry
        try:
            usdjpy_inst.generate_signal('USDJPY', 150.00, 1)
            check("USDJPY.generate_signal() raises NotImplementedError", False,
                  "no exception was raised -- stub should not return a real signal")
        except NotImplementedError as e:
            check("USDJPY.generate_signal() raises NotImplementedError", True, str(e))
        except Exception as e:
            check("USDJPY.generate_signal() raises NotImplementedError", False,
                  f"raised wrong exception type {type(e).__name__}: {e}")
finally:
    # ------------------------------------------------------------------
    # CHECK 5 -- flip USDJPY back to active: false
    # ------------------------------------------------------------------
    section("CHECK 5 -- USDJPY flipped back to active: false")
    USDJPY_YAML.write_text(original_yaml, encoding='utf-8')
    restored = get_active_pairs()
    restored_names = sorted(name for name, _inst, _cfg in restored)
    check("USDJPY removed again after restoring YAML",
          'USDJPY' not in restored_names,
          f"active list: {restored_names}")
    check("YAML file content restored exactly",
          USDJPY_YAML.read_text(encoding='utf-8') == original_yaml)

# ---------------------------------------------------------------------------
# CHECK 6 -- active pairs' session windows match core.session_filter predicates
# ---------------------------------------------------------------------------
section("CHECK 6 -- session_filter consistency for all 3 active pairs")

# 09:00 UTC -- inside London window (08:00-13:00)
london_sample = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)
# 13:30 UTC -- inside London/NY overlap window (12:00-16:00)
overlap_sample = datetime(2026, 1, 5, 13, 30, tzinfo=timezone.utc)

for name, inst, cfg in active:
    try:
        windows = inst.get_session_windows()
    except Exception as e:
        check(f"{name}.get_session_windows()", False, str(e))
        continue

    if cfg['strategy'] == 'london_breakout':
        london_window_present = any(w['name'] == 'london' for w in windows)
        matches_filter = sf.london_session(london_sample)
        check(f"{name}: get_session_windows() includes 'london' AND "
              f"session_filter.london_session(09:00 UTC) == True",
              london_window_present and matches_filter)
    elif cfg['strategy'] == 'sma_ema_combined':
        overlap_window_present = any(w['name'] == 'london_ny_overlap' for w in windows)
        matches_filter = sf.london_ny_overlap(overlap_sample)
        check(f"{name}: get_session_windows() includes 'london_ny_overlap' AND "
              f"session_filter.london_ny_overlap(13:30 UTC) == True",
              overlap_window_present and matches_filter)
    elif cfg['strategy'] == 'monday_drift':
        monday_window_present = any(w['name'] == 'monday' for w in windows)
        check(f"{name} (monday_drift): get_session_windows() includes 'monday'",
              monday_window_present)
    elif cfg['strategy'] in ('asian_range_breakout', 'asian_hours_reversion'):
        # 03:00 UTC -- inside the Asian session window (00:00-07:00)
        asian_sample = datetime(2026, 1, 5, 3, 0, tzinfo=timezone.utc)
        asian_window_present = any(w['name'] == 'asian' for w in windows)
        matches_filter = sf.asian_session(asian_sample)
        check(f"{name} ({cfg['strategy']}): get_session_windows() includes "
              f"'asian' AND session_filter.asian_session(03:00 UTC) == True",
              asian_window_present and matches_filter)
    else:
        check(f"{name}: unexpected strategy type {cfg['strategy']!r} in check 6", False)

# ---------------------------------------------------------------------------
# CHECK 7 -- locked_pairs allowlist (prop-firm clone isolation) intersects
# correctly, and is fully backward-compatible when absent
# ---------------------------------------------------------------------------
section("CHECK 7 -- locked_pairs allowlist intersection")

locked_active = get_active_pairs(locked_pairs={('GBPJPY', 'asian_range_breakout')})
locked_names = sorted(name for name, _inst, _cfg in locked_active)
check("locked_pairs={GBPJPY/asian_range_breakout} -> only that pair survives",
      locked_names == ['GBPJPY'],
      f"got {locked_names}")

unrestricted = get_active_pairs(locked_pairs=None)
unrestricted_names = sorted(name for name, _inst, _cfg in unrestricted)
check("locked_pairs=None -> unrestricted, matches CHECK 1's full active set",
      unrestricted_names == active_names,
      f"got {unrestricted_names}")

# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
section("VERIFICATION SUMMARY")
total  = len(results)
passed = sum(results)
print(f"  {passed} / {total} checks passed")
if passed == total:
    print(f"\n  ALL CHECKS PASSED -- architecture verified.")
    sys.exit(0)
else:
    print(f"\n  {total - passed} CHECK(S) FAILED -- see [FAIL] lines above.")
    sys.exit(1)
