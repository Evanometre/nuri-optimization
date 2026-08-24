import pytest

from nuri.education.simplex import solve_simplex, solve_simplex_for_problem
from nuri.lp import solve_lp

from cases.furniture import furniture_problem
from cases.case2_generic import case2_problem
from cases.case3_generic import case3_problem
from cases.case4_capital_allocation import capital_allocation_problem


def test_matches_highs_objective_on_every_lp_case():
    cases = [
        furniture_problem(),
        case2_problem(),
        case3_problem(),
        capital_allocation_problem(),
    ]
    for problem in cases:
        _, my_obj, status = solve_simplex_for_problem(problem)
        highs_result = solve_lp(problem)

        assert status == "optimal"
        assert my_obj == pytest.approx(highs_result.objective_value, rel=1e-6)


def test_matches_highs_quantities_where_the_optimum_is_unique():
    # Furniture, case2, and case3 all have a single optimal vertex, so the
    # exact production quantities should match HiGHS too (case4 doesn't --
    # it has a tied/degenerate optimum, see the dedicated test below).
    for problem in (furniture_problem(), case2_problem(), case3_problem()):
        my_quantities, _, _ = solve_simplex_for_problem(problem)
        highs_result = solve_lp(problem)

        for name in problem.product_names():
            assert my_quantities[name] == pytest.approx(highs_result.quantities[name], abs=1e-6)


def test_case4_lands_on_a_different_but_equally_optimal_vertex():
    # product_a's reduced cost is exactly 0 (see the capital allocation case
    # investigation) -- the optimum isn't unique. My simplex (Bland's rule)
    # and HiGHS (its own pivoting) are both allowed to land on different
    # vertices of the same optimal face, as long as the objective matches.
    problem = capital_allocation_problem()
    my_quantities, my_obj, _ = solve_simplex_for_problem(problem)
    highs_result = solve_lp(problem)

    assert my_obj == pytest.approx(highs_result.objective_value, rel=1e-6)
    assert my_quantities != pytest.approx(highs_result.quantities)  # genuinely a different vertex


def test_unbounded_problem_is_detected():
    # max x, no constraints at all -> unbounded above.
    x, obj, status = solve_simplex(c=[1], A=[[0]], b=[0])
    assert status == "unbounded"
    assert x is None


def test_bland_rule_terminates_on_a_known_degenerate_case():
    # Beale's classic cycling example: without an anti-cycling rule, naive
    # simplex can loop forever on this problem. Bland's rule must terminate.
    c = [0.75, -150, 0.02, -6]
    A = [
        [0.25, -60, -0.04, 9],
        [0.5, -90, -0.02, 3],
        [0, 0, 1, 0],
    ]
    b = [0, 0, 1]
    x, obj, status = solve_simplex(c, A, b, max_iterations=200)
    assert status == "optimal"
