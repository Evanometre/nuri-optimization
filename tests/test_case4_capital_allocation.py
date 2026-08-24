import pytest

from cases.case4_capital_allocation import capital_allocation_problem
from nuri.lp import solve_lp
from nuri.ilp import solve_ilp


def test_lp_relaxation_uses_full_budget():
    result = solve_lp(capital_allocation_problem())

    assert result.success
    assert result.objective_value == pytest.approx(692_500, abs=1)
    assert result.utilization["cash"] == pytest.approx(1.0, abs=1e-6)
    # B and C sit at their demand ceilings, not held back by cash.
    assert result.quantities["product_b"] == pytest.approx(80, abs=1e-6)
    assert result.quantities["product_c"] == pytest.approx(150, abs=1e-6)


def test_lp_product_a_is_a_reduced_cost_tie():
    # Product A's return per naira of cash exactly matches cash's shadow price,
    # so the LP is indifferent to it -- reduced cost should be ~0, not negative.
    result = solve_lp(capital_allocation_problem())

    assert result.reduced_costs["product_a"] == pytest.approx(0, abs=1)


def test_ilp_respects_whole_units_and_stays_within_budget():
    result = solve_ilp(capital_allocation_problem())

    assert result.success
    assert result.objective_value <= 692_500  # can't beat the LP relaxation bound
    total_cash = sum(
        capital_allocation_problem().products[name]["cash"] * qty
        for name, qty in result.quantities.items()
    )
    assert total_cash <= 2_000_000
    for name, qty in result.quantities.items():
        assert qty == pytest.approx(round(qty), abs=1e-6)  # whole units only
