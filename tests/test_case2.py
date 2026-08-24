import pytest

from cases.case2_generic import case2_problem
from nuri.lp import solve_lp
from nuri.ilp import solve_ilp


def test_lp_matches_hand_solved_optimum():
    result = solve_lp(case2_problem())

    assert result.success
    assert result.objective_value == pytest.approx(1560, abs=1)
    assert result.quantities["x"] == pytest.approx(2, abs=1e-6)
    assert result.quantities["y"] == pytest.approx(3, abs=1e-6)
    assert set(result.binding_constraints) == {"c1", "c2"}


def test_ilp_matches_lp_here():
    lp_result = solve_lp(case2_problem())
    ilp_result = solve_ilp(case2_problem())

    assert ilp_result.success
    assert ilp_result.objective_value == pytest.approx(lp_result.objective_value, abs=1)
