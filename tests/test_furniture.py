import pytest

from cases.furniture import furniture_problem
from nuri.lp import solve_lp
from nuri.ilp import solve_ilp

EXPECTED_QUANTITIES = {"tables": 20, "chairs": 40, "shelves": 0}
EXPECTED_PROFIT = 1_300_000


def test_lp_matches_known_optimum():
    result = solve_lp(furniture_problem())

    assert result.success
    assert result.objective_value == pytest.approx(EXPECTED_PROFIT, abs=1)
    for name, expected_qty in EXPECTED_QUANTITIES.items():
        assert result.quantities[name] == pytest.approx(expected_qty, abs=1e-6)


def test_ilp_matches_lp_here():
    # This particular case happens to have an integer optimum, so LP and ILP
    # should agree exactly. That won't always be true for other cases.
    lp_result = solve_lp(furniture_problem())
    ilp_result = solve_ilp(furniture_problem())

    assert ilp_result.success
    assert ilp_result.objective_value == lp_result.objective_value


def test_all_resources_binding():
    result = solve_lp(furniture_problem())

    assert set(result.binding_constraints) == {"wood", "machine", "labour"}
