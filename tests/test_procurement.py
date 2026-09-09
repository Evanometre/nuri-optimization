import pytest

from cases.case10_procurement_optimization import base_problem, INCUMBENT_BY_CATEGORY, DEMAND, OFFERS, COST_OF_CAPITAL_RATE
from nuri.procurement import solve_procurement, incumbent_cost, effective_unit_cost, cost_composition


def test_baseline_is_worse_than_best_single_source():
    baseline_cost, _ = incumbent_cost(base_problem(), INCUMBENT_BY_CATEGORY)
    best_single = solve_procurement(base_problem(concentration_cap=None, single_source_only=True))
    assert best_single.success
    assert best_single.total_cost < baseline_cost


def test_recommended_sits_between_best_single_source_and_baseline():
    # Diversification (the concentration cap) can only cost money relative to
    # the unconstrained best-single-source plan, never save money -- so
    # best_single <= recommended <= baseline should always hold.
    baseline_cost, _ = incumbent_cost(base_problem(), INCUMBENT_BY_CATEGORY)
    best_single = solve_procurement(base_problem(concentration_cap=None, single_source_only=True))
    recommended = solve_procurement(base_problem())

    assert best_single.total_cost <= recommended.total_cost
    assert recommended.total_cost <= baseline_cost


def test_two_categories_genuinely_require_multi_sourcing():
    # raw_materials and logistics_services: the alternative supplier's
    # capacity alone is below full demand, so even "best single source"
    # can't single-source there -- it's forced back to the incumbent alone.
    best_single = solve_procurement(base_problem(concentration_cap=None, single_source_only=True))
    assert len(best_single.suppliers_used_by_category["raw_materials"]) == 1
    assert best_single.suppliers_used_by_category["raw_materials"][0][0] == "S3_incumbent"
    assert len(best_single.suppliers_used_by_category["logistics_services"]) == 1


def test_concentration_cap_is_infeasible_below_50_percent_with_two_suppliers():
    # With exactly 2 suppliers per category, a cap c needs 2c >= 1, i.e.
    # c >= 0.5, to even be feasible. This is a real modeling constraint, not
    # an implementation bug -- pin it so nobody "fixes" the cap back to a
    # number that silently breaks the case.
    infeasible = solve_procurement(base_problem(concentration_cap=0.4))
    assert not infeasible.success


def test_recommended_respects_the_concentration_cap():
    recommended = solve_procurement(base_problem())
    for category, demand in DEMAND.items():
        cap = 0.60 * demand
        for supplier, qty in recommended.suppliers_used_by_category[category]:
            assert qty <= cap + 1e-3


def test_budget_is_not_binding():
    recommended = solve_procurement(base_problem())
    assert not recommended.budget_binding


def test_cheaper_supplier_wins_despite_worse_freight_and_rejection():
    # The packaging_materials alternative has WORSE freight and WORSE
    # rejection cost than the incumbent, yet a lower effective cost overall
    # -- the case's central "don't just look at freight/quality in isolation"
    # point, verified numerically rather than just asserted.
    incumbent = next(o for o in OFFERS if o.supplier == "S1_incumbent")
    alt = next(o for o in OFFERS if o.supplier == "S2_alt")

    incumbent_comp = cost_composition(incumbent, COST_OF_CAPITAL_RATE)
    alt_comp = cost_composition(alt, COST_OF_CAPITAL_RATE)

    assert alt_comp["freight_effect"] > incumbent_comp["freight_effect"]
    assert alt_comp["rejection_effect"] > incumbent_comp["rejection_effect"]
    assert effective_unit_cost(alt, COST_OF_CAPITAL_RATE) < effective_unit_cost(incumbent, COST_OF_CAPITAL_RATE)


def test_diversification_cost_is_non_negative():
    best_single = solve_procurement(base_problem(concentration_cap=None, single_source_only=True))
    recommended = solve_procurement(base_problem())
    diversification_cost = recommended.total_cost - best_single.total_cost
    assert diversification_cost >= -1e-3
