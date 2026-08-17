"""Phase 37 Track B -- reusable, reproducible priority-score computation.

Reads the frozen weights from reports/phase37_preregistration.md (weights below
mirror that document exactly) and the per-class 0-3 component scores (derived
from reports/phase37_portfolio_gap_mapping.csv, phase37_data_availability.csv,
and phase37_overfitting_risk.csv) to (re)produce
reports/phase37_return_stream_priorities.csv.

This is a RESEARCH-PRIORITY ASSESSMENT ONLY -- not a profitability score, not
a claim that any class will diversify the portfolio.
"""
import csv

WEIGHTS = {
    "independence": 0.25,
    "dd_div": 0.20,
    "highvol": 0.15,
    "mechanism": 0.15,
    "dataq": 0.10,
    "research": 0.05,
    "cost": 0.05,
    "overfit": 0.05,
}

# Component scores (0-3 scale), derived from the Phase 37 Track B source CSVs.
CLASSES = {
    "Event/macro-conditioned systems": dict(independence=2.5, dd_div=3, highvol=3, mechanism=3, dataq=0.5, research=1, cost=2, overfit=1),
    "Index-based return streams": dict(independence=3, dd_div=1, highvol=2.5, mechanism=3, dataq=0.5, research=2.5, cost=2, overfit=2),
    "Volatility-conditioned systems": dict(independence=2.5, dd_div=2, highvol=3, mechanism=2, dataq=0.5, research=1, cost=2, overfit=2),
    "Cross-sectional FX": dict(independence=2.5, dd_div=1, highvol=2, mechanism=2, dataq=3, research=2, cost=2, overfit=2),
    "Multi-asset momentum": dict(independence=3, dd_div=1, highvol=2.5, mechanism=3, dataq=0.5, research=1, cost=1, overfit=1.5),
    "Commodity-based return streams": dict(independence=2, dd_div=1, highvol=2, mechanism=2, dataq=2, research=3, cost=3, overfit=2),
    "Session-specific structures": dict(independence=2, dd_div=1, highvol=2, mechanism=2, dataq=2.5, research=2.5, cost=2.5, overfit=2),
    "Relative-value / spread structures": dict(independence=2.5, dd_div=1, highvol=0, mechanism=3, dataq=2.5, research=2, cost=2, overfit=1.5),
    "Cross-asset relationships": dict(independence=3, dd_div=1, highvol=0, mechanism=3, dataq=0.5, research=2, cost=2, overfit=2),
    "Other structurally distinct mechanisms": dict(independence=0, dd_div=0, highvol=0, mechanism=2, dataq=0.5, research=0, cost=0, overfit=0),
}

FIELDNAMES = [
    "rank", "class_name",
    "portfolio_independence_score", "drawdown_diversification_score",
    "high_vol_compatibility_score", "mechanism_diversity_score",
    "data_quality_score", "researchability_score",
    "cost_execution_feasibility_score", "overfitting_risk_score_inverse",
    "weighted_priority_score_pct", "label",
]

LABEL = ("RESEARCH-PRIORITY ASSESSMENT ONLY -- not a profitability score, "
         "not a claim the class will diversify the portfolio")


def compute_score(components: dict) -> float:
    raw = sum(components[k] * WEIGHTS[k] for k in WEIGHTS)
    return round(raw / 3 * 100, 1)


def main():
    scored = [(name, compute_score(c), c) for name, c in CLASSES.items()]
    scored.sort(key=lambda x: x[1], reverse=True)

    rows = []
    for rank, (name, score, c) in enumerate(scored, start=1):
        rows.append({
            "rank": rank,
            "class_name": name,
            "portfolio_independence_score": c["independence"],
            "drawdown_diversification_score": c["dd_div"],
            "high_vol_compatibility_score": c["highvol"],
            "mechanism_diversity_score": c["mechanism"],
            "data_quality_score": c["dataq"],
            "researchability_score": c["research"],
            "cost_execution_feasibility_score": c["cost"],
            "overfitting_risk_score_inverse": c["overfit"],
            "weighted_priority_score_pct": score,
            "label": LABEL,
        })

    out_path = "reports/phase37_return_stream_priorities.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_path}")
    for r in rows:
        print(f"{r['rank']:>2} {r['class_name']:<40} {r['weighted_priority_score_pct']}")


if __name__ == "__main__":
    main()
