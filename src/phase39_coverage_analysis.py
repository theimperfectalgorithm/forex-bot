"""
Phase 39 -- coverage/duplication/multiple-testing/stop-list analysis over
reports/phase39_fx_research_inventory.csv. No backtest, no simulation --
pure reconciliation and classification of already-committed results.
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).parent.parent
OUT = REPO / 'reports'

with open(OUT / 'phase39_fx_research_inventory.csv', newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

SCREEN = [r for r in rows if 'screen' in r.get('notes', '').lower()]
CONFIRM = [r for r in rows if r not in SCREEN]
print(f"total rows={len(rows)} screen={len(SCREEN)} confirmatory={len(CONFIRM)}")

# --- Part 4: family coverage ---
fam_rows = []
fam_counts = Counter(r['strategy_family'] for r in rows)
for fam, n in fam_counts.most_common():
    sub = [r for r in rows if r['strategy_family'] == fam]
    is_screen = all(r in SCREEN for r in sub)
    fam_rows.append({
        'strategy_family': fam, 'hypothesis_count': n,
        'pct_of_total': round(n / len(rows) * 100, 1),
        'type': 'EXPLORATORY SCREEN (Phase30 calendar/drift, 60 cells across day-of-week x direction x instrument)' if is_screen and fam == 'calendar_drift' and n > 1 else
                ('MIXED (screen + confirmatory)' if fam == 'calendar_drift' else 'CONFIRMATORY (preregistered)'),
        'unique_instruments': len(set(x['instrument'] for x in sub)),
        'final_classifications': '; '.join(sorted(set(x['final_classification'][:60] for x in sub)))[:300],
        'classification_confidence': 'HIGH (explicit strategy_family field in source ledger)',
    })
with open(OUT / 'phase39_fx_family_coverage.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(fam_rows[0].keys()))
    w.writeheader(); w.writerows(fam_rows)
print("wrote phase39_fx_family_coverage.csv")

# --- Part 5: session coverage ---
def session_bucket(s):
    s = (s or '').lower()
    if 'asian' in s and 'session-independent' not in s:
        return 'Asian'
    if 'london' in s and 'overlap' not in s and 'ny' not in s:
        return 'London'
    if 'overlap' in s:
        return 'London/NY overlap'
    if 'new york' in s or s.startswith('ny'):
        return 'New York'
    if 'multi-session' in s or 'monday full session' in s or 'unrestricted' in s:
        return 'multi-session'
    if 'session-independent' in s:
        return 'session-independent'
    return 'UNKNOWN'

sess_counter = defaultdict(lambda: {'hyp': 0, 'mechanisms': set(), 'confirmatory': 0})
for r in rows:
    b = session_bucket(r['session'])
    sess_counter[b]['hyp'] += 1
    sess_counter[b]['mechanisms'].add(r['strategy_family'])
    if r in CONFIRM:
        sess_counter[b]['confirmatory'] += 1

sess_rows = []
for b, d in sess_counter.items():
    sess_rows.append({
        'session_bucket': b, 'hypothesis_count': d['hyp'], 'confirmatory_count': d['confirmatory'],
        'unique_mechanisms_tested': len(d['mechanisms']), 'mechanisms': '; '.join(sorted(d['mechanisms'])),
        'genuinely_unexplored': 'NO' if d['confirmatory'] > 0 else ('PARTIAL -- only exploratory screen coverage' if d['hyp'] > 0 else 'YES'),
    })
with open(OUT / 'phase39_fx_session_coverage.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(sess_rows[0].keys()))
    w.writeheader(); w.writerows(sess_rows)
print("wrote phase39_fx_session_coverage.csv")

# --- Part 6: instrument coverage ---
def instr_bucket(instr):
    instr = (instr or '')
    if 'SYNTHETIC' in instr:
        return 'SYNTHETIC (cross-sectional basket)'
    if '/' in instr:
        return instr  # multi-instrument confirmatory row, keep as-is
    return instr

instr_counter = defaultdict(lambda: {'hyp': 0, 'confirmatory': 0, 'families': set()})
for r in rows:
    key = instr_bucket(r['instrument'])
    instr_counter[key]['hyp'] += 1
    instr_counter[key]['families'].add(r['strategy_family'])
    if r in CONFIRM:
        instr_counter[key]['confirmatory'] += 1

instr_rows = []
for instr, d in sorted(instr_counter.items(), key=lambda x: -x[1]['hyp']):
    jpy = 'YES' if 'JPY' in instr else ('MIXED' if 'JPY' in instr else 'NO')
    instr_rows.append({
        'instrument': instr, 'hypothesis_count': d['hyp'], 'confirmatory_count': d['confirmatory'],
        'unique_families_tested': len(d['families']), 'is_jpy': jpy,
        'asset_class': 'FX cross (JPY-linked)' if 'JPY' in instr else ('Commodity (metals)' if 'XAU' in instr else ('Synthetic basket' if 'SYNTHETIC' in instr else 'FX major')),
    })
with open(OUT / 'phase39_fx_instrument_coverage.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(instr_rows[0].keys()))
    w.writeheader(); w.writerows(instr_rows)
print("wrote phase39_fx_instrument_coverage.csv")

# --- Part 7: mechanism coverage (confirmatory only, to avoid pooling exploratory screen cells into fake significance) ---
mech_counter = defaultdict(lambda: {'hyp': 0, 'instr': set(), 'sess': set(), 'edge': 0, 'robust': 0, 'qualified': 0})
for r in CONFIRM:
    m = r['strategy_family']
    mech_counter[m]['hyp'] += 1
    mech_counter[m]['instr'].add(r['instrument'])
    mech_counter[m]['sess'].add(session_bucket(r['session']))
    cls = r['final_classification']
    if not cls.startswith('A.') and not cls.startswith('B. REJECTED -- NO'):
        mech_counter[m]['edge'] += 1  # cleared Gate1 at least once
    if cls.startswith('J.') or cls.startswith('H.') or cls.startswith('I.'):
        mech_counter[m]['qualified'] += 1

mech_rows = []
for m, d in mech_counter.items():
    mech_rows.append({
        'mechanism': m, 'confirmatory_hypotheses': d['hyp'], 'unique_instruments': len(d['instr']),
        'unique_sessions': len(d['sess']), 'initial_edge_count': d['edge'],
        'robustness_pass_count': 0, 'portfolio_qualified_count': d['qualified'],
        'note': 'Confirmatory-only count (excludes the 60-cell exploratory calendar/drift screen to avoid pooled significance)',
    })
with open(OUT / 'phase39_fx_mechanism_coverage.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(mech_rows[0].keys()))
    w.writeheader(); w.writerows(mech_rows)
print("wrote phase39_fx_mechanism_coverage.csv")

# --- Part 8: structural duplication ---
# Return-driver grouping per the frozen preregistration definition (Part E)
driver_groups = {
    'calendar_drift': 'day-of-week open-to-close drift',
    'volatility_contraction_expansion_breakout': 'volatility-state-change breakout',
    'trend_momentum_continuation': 'trend/momentum continuation',
    'new_york_open_range_breakout': 'single-session range breakout',
    'new_york_session_momentum': 'single-session momentum continuation',
    'london_ny_overlap_continuation': 'single-window continuation',
    'multi_timeframe_trend_continuation': 'trend/momentum continuation',
    'atr_scaled_volatility_expansion': 'volatility-state-change breakout',
    'cross_sectional_relative_momentum': 'cross-instrument relative ranking',
    'session_transition_breakout_continuation': 'session-transition range breakout',
}
driver_counter = defaultdict(list)
for r in CONFIRM:
    driver_counter[driver_groups.get(r['strategy_family'], 'UNKNOWN')].append(r['strategy_family'] + '/' + r['instrument'])

dup_rows = []
for driver, members in driver_counter.items():
    cls = 'A. GENUINELY DISTINCT' if len(set(members)) == len(members) and len(members) <= 1 else \
          ('C. NEAR-DUPLICATES (same return driver)' if len(members) > 1 else 'A. GENUINELY DISTINCT')
    dup_rows.append({
        'return_driver_group': driver, 'member_hypotheses': '; '.join(members), 'raw_count': len(members),
        'classification': cls,
    })
with open(OUT / 'phase39_structural_duplication.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(dup_rows[0].keys()))
    w.writeheader(); w.writerows(dup_rows)
print("wrote phase39_structural_duplication.csv")

raw_count = len(CONFIRM)
distinct_drivers = len(driver_counter)
print(f"\nRAW CONFIRMATORY HYPOTHESIS COUNT: {raw_count}")
print(f"ESTIMATED DISTINCT RESEARCH CONCEPT COUNT (by return driver): {distinct_drivers}")
for driver, members in driver_counter.items():
    print(f"  {driver}: {len(members)} variant(s) -- {members}")
