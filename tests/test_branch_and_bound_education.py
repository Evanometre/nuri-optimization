import pytest

from nuri.education.branch_and_bound import solve_branch_and_bound
from nuri.education.simplex import solve_simplex_general
from nuri.ilp import solve_ilp

from cases.furniture import furniture_problem
from cases.case2_generic import case2_problem
from cases.case3_generic import case3_problem
from cases.case4_capital_allocation import capital_allocation_problem


def test_matches_ortools_objective_on_every_ilp_case():
    cases = [
        furniture_problem(),
        case2_problem(),
        case3_problem(),
        capital_allocation_problem(),
    ]
    for problem in cases:
        _, my_obj, status, _ = solve_branch_and_bound(problem)
        ortools_result = solve_ilp(problem)

        assert status == "optimal"
        assert my_obj == pytest.approx(ortools_result.objective_value, abs=1)


def test_case3_beats_naive_rounding_via_real_branching():
    # The LP relaxation optimum is (3.25, 3.5); rounding down gives (3, 3) at
    # z=21, worse than the true integer optimum (4, 2) at z=22. My B&B must
    # find (4, 2) through actual branching, not by accident.
    quantities, obj, status, nodes_explored = solve_branch_and_bound(case3_problem())

    assert status == "optimal"
    assert obj == pytest.approx(22, abs=1e-6)
    assert quantities["x"] == pytest.approx(4, abs=1e-6)
    assert quantities["y"] == pytest.approx(2, abs=1e-6)
    assert nodes_explored > 1  # confirms it actually branched, didn't get lucky at the root


def test_trivially_integer_relaxations_need_no_branching():
    # Furniture's LP relaxation is already integer, so B&B should solve it at
    # the root node without ever branching.
    _, _, status, nodes_explored = solve_branch_and_bound(furniture_problem())
    assert status == "optimal"
    assert nodes_explored == 1


def test_infeasible_ilp_is_detected():
    x, obj, status = solve_simplex_general([1], [([1], "<=", 2), ([1], ">=", 5)])
    assert status == "infeasible"
    assert x is None
