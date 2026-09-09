from cases.case10_procurement_optimization import (
    base_problem, INCUMBENT_BY_CATEGORY, DEMAND, OFFERS, COST_OF_CAPITAL_RATE, CONCENTRATION_CAP,
)
from nuri.procurement import solve_procurement, incumbent_cost, cost_composition, effective_unit_cost

problem = base_problem()

baseline_cost, baseline_breakdown = incumbent_cost(problem, INCUMBENT_BY_CATEGORY)
best_single = solve_procurement(base_problem(concentration_cap=None, single_source_only=True))
recommended = solve_procurement(base_problem())

print("=== A. Baseline (current, naive single-incumbent-source per category) ===")
print(f"Total: N{baseline_cost:,.0f}")
for cat, cost in baseline_breakdown.items():
    print(f"  {cat}: N{cost:,.0f}")

print()
print("=== B. Best possible if the concentration/risk policy didn't exist ===")
print(f"Total: N{best_single.total_cost:,.0f}")
for cat, suppliers in best_single.suppliers_used_by_category.items():
    print(f"  {cat}: {[(s, round(q)) for s, q in suppliers]}")

print()
print(f"=== C. Recommended (risk-aware: {CONCENTRATION_CAP:.0%} concentration cap, multi-sourcing allowed) ===")
print(f"Total: N{recommended.total_cost:,.0f}")
for cat, suppliers in recommended.suppliers_used_by_category.items():
    print(f"  {cat}: {[(s, round(q)) for s, q in suppliers]}")
print(f"Budget binding: {recommended.budget_binding}")

print()
print("=== Decomposition ===")
addressable_gap = baseline_cost - best_single.total_cost
diversification_cost = recommended.total_cost - best_single.total_cost
net_savings = baseline_cost - recommended.total_cost
print(f"A (baseline):                            N{baseline_cost:,.0f}")
print(f"B (best, ignoring risk policy):          N{best_single.total_cost:,.0f}")
print(f"C (recommended, risk-aware):             N{recommended.total_cost:,.0f}")
print(f"Total addressable gap (A - B):           N{addressable_gap:,.0f}")
print(f"Cost of the {CONCENTRATION_CAP:.0%} diversification policy (C - B): N{diversification_cost:,.0f}")
print(f"Net realistic savings (A - C):           N{net_savings:,.0f}  ({net_savings/baseline_cost:.1%} of baseline spend)")

print()
print("=== Why the winning packaging_materials supplier wins, despite worse freight/rejection ===")
for offer in OFFERS:
    if offer.category == "packaging_materials":
        comp = cost_composition(offer, COST_OF_CAPITAL_RATE)
        total = effective_unit_cost(offer, COST_OF_CAPITAL_RATE)
        print(
            f"  {offer.supplier}: price=N{comp['price']:.2f}  freight_effect=+N{comp['freight_effect']:.2f}  "
            f"rejection_effect=+N{comp['rejection_effect']:.2f}  financing_benefit=N{comp['financing_benefit']:.2f}  "
            f"-> effective=N{total:.2f}/unit"
        )
