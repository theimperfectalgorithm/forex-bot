"""
Phase 39 -- reusable, reproducible priority-score computation over the 3
in-scope return-stream classes (Event/Macro, Volatility, Index-based),
using the exact weights frozen in reports/phase39_preregistration.md Part N.
No strategy is designed or backtested here.
"""
import csv

WEIGHTS = {
    'independence': 0.25, 'dd_div': 0.20, 'highvol': 0.15, 'mechanism': 0.15,
    'dataq': 0.10, 'research': 0.05, 'cost': 0.05, 'overfit': 0.05,
}

# 0-3 component scores, derived from this phase's own audits:
#   independence/dd_div/highvol/mechanism <- phase39_portfolio_relevance.csv (gap6/gap2/gap1, mechanism diversity assessed qualitatively)
#   dataq <- phase39_data_quality_matrix.csv (letter grade -> 0-3)
#   research <- inverse of phase39_research_cost.csv
#   cost <- inverse of phase39_research_cost.csv (execution feasibility)
#   overfit <- inverse of phase39_overfitting_risk.csv
CLASSES = {
    'Event/Macro-conditioned': dict(independence=3, dd_div=3, highvol=2, mechanism=3, dataq=0, research=0.5, cost=0.5, overfit=0.5),
    'Volatility-conditioned (self-calculated)': dict(independence=2, dd_div=2, highvol=3, mechanism=2, dataq=2.5, research=2.5, cost=2.5, overfit=2),
    'Index-based': dict(independence=3, dd_div=2, highvol=1, mechanism=3, dataq=2, research=1.5, cost=1.5, overfit=2),
}

FIELDNAMES = [
    'rank', 'class_name', 'portfolio_independence_score', 'drawdown_diversification_score',
    'high_vol_compatibility_score', 'mechanism_diversity_score', 'data_quality_score',
    'researchability_score', 'cost_execution_feasibility_score', 'overfitting_risk_score_inverse',
    'weighted_priority_score_pct', 'label',
]
LABEL = ('RESEARCH-PRIORITY ASSESSMENT ONLY -- not a profitability score, '
         'not a claim the class will diversify the portfolio or be profitable')


def compute_score(c):
    raw = sum(c[k] * WEIGHTS[k] for k in WEIGHTS)
    return round(raw / 3 * 100, 1)


def main():
    scored = [(name, compute_score(c), c) for name, c in CLASSES.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    rows = []
    for rank, (name, score, c) in enumerate(scored, start=1):
        rows.append({
            'rank': rank, 'class_name': name,
            'portfolio_independence_score': c['independence'], 'drawdown_diversification_score': c['dd_div'],
            'high_vol_compatibility_score': c['highvol'], 'mechanism_diversity_score': c['mechanism'],
            'data_quality_score': c['dataq'], 'researchability_score': c['research'],
            'cost_execution_feasibility_score': c['cost'], 'overfitting_risk_score_inverse': c['overfit'],
            'weighted_priority_score_pct': score, 'label': LABEL,
        })
    out = 'reports/phase39_return_stream_priority.csv'
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out}")
    for r in rows:
        print(f"{r['rank']} {r['class_name']:<45} {r['weighted_priority_score_pct']}")


if __name__ == '__main__':
    main()
