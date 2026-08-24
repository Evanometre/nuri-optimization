import pytest

from cases.case3_generic import case3_problem
from nuri.lp import solve_lp
from nuri.ilp import solve_ilp


def test_lp_relaxation():
    result = solve_lp(case3_problem())

    assert result.success
    assert result.objective_value == pytest.approx(23.5, abs=1e-6)
    assert result.quantities["x"] == pytest.approx(3.25, abs=1e-6)
    assert result.quantities["y"] == pytest.approx(3.5, abs=1e-6)


def test_ilp_beats_naive_rounding():
    # Rounding the LP relaxation (3.25, 3.5) down to (3, 3) gives z=21.
    # The true integer optimum is (4, 2) with z=22 -- only branch-and-bound finds it.
    result = solve_ilp(case3_problem())

    assert result.success
    assert result.objective_value == pytest.approx(22, abs=1e-6)
    assert result.quantities["x"] == pytest.approx(4, abs=1e-6)
    assert result.quantities["y"] == pytest.approx(2, abs=1e-6)
