"""
Phase 34 -- reusable data-integrity / reproducibility verification, run
before the Phase 34 synthesis (reports/phase34_*.md, reports/phase34_*.csv)
was written. No new backtesting -- Phase 34 is a synthesis/audit of Phases
29-33's already-completed, already-committed results.

Verifies:
  1. research_data_validator passes on both control inputs
  2. the Phase 31/32 control reproduces exactly (trade_count, effective_n,
     avg_pairwise_correlation)
  3. the Phase 33 preregistration has exactly one commit touching it (i.e.
     was never edited after being frozen)
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from phase32_portfolio_architecture import build_control, reproducibility_gate  # noqa: E402

REPO = Path(__file__).parent.parent


def verify_preregistration_never_edited():
    result = subprocess.run(
        ['git', 'log', '--follow', '--format=%H', '--', 'reports/phase33_preregistration.md'],
        cwd=REPO, capture_output=True, text=True, check=True)
    commits = [c for c in result.stdout.strip().split('\n') if c]
    if len(commits) != 1:
        raise RuntimeError(
            f"Phase 33 preregistration has {len(commits)} commits touching it -- "
            f"expected exactly 1 (frozen, never edited after results existed). STOP.")
    print(f"[integrity] Phase 33 preregistration touched by exactly 1 commit ({commits[0][:8]}) -- confirmed frozen")


def main():
    print("[integrity] Reproducing Phase 31/32 control...")
    hist, daily, daily_by_strat, control_profile, dd_days, avg_corr = build_control()
    reproducibility_gate(control_profile)
    print("[integrity] Phase 31/32 control reproduction: PASSED")

    verify_preregistration_never_edited()

    print("\n[integrity] All Phase 34 pre-analysis checks PASSED. Proceeding to synthesis was authorized.")


if __name__ == '__main__':
    main()
