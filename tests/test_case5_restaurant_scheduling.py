import pytest

from cases.case5_restaurant_scheduling import (
    EMPLOYEES,
    restaurant_scheduling_problem,
    fully_staffed_employees,
)
from nuri.scheduling import compute_overtime_cost, solve_schedule


def test_original_8_employees_infeasible_at_36h():
    # Saturday evening needs 8, but only 7 of the 8 employees are even
    # available that day (B works Mon-Fri only) -- no schedule can cover it.
    result = solve_schedule(restaurant_scheduling_problem(max_hours=36))
    assert result.success is False


def test_original_8_employees_infeasible_even_with_saturday_relaxed():
    # Relaxing Saturday evening to the max available (7) still isn't enough --
    # the real bottleneck is total weekly hour capacity, not one day's headcount.
    problem = restaurant_scheduling_problem(
        max_hours=36, requirement_overrides={("Sat", "evening"): 7}
    )
    result = solve_schedule(problem)
    assert result.success is False


def test_original_8_employees_infeasible_at_42h_too():
    problem = restaurant_scheduling_problem(
        max_hours=42, requirement_overrides={("Sat", "evening"): 7}
    )
    result = solve_schedule(problem)
    assert result.success is False


def test_true_feasibility_threshold_is_54_hours():
    problem_53 = restaurant_scheduling_problem(
        max_hours=53, requirement_overrides={("Sat", "evening"): 7}
    )
    problem_54 = restaurant_scheduling_problem(
        max_hours=54, requirement_overrides={("Sat", "evening"): 7}
    )
    assert solve_schedule(problem_53).success is False
    assert solve_schedule(problem_54).success is True


def test_three_weekend_hires_makes_36h_feasible():
    # 1 extra weekend-only hire isn't enough; it takes 3 to close the gap at a
    # real 36h/week cap, with the original Saturday evening requirement of 8.
    problem = restaurant_scheduling_problem(max_hours=36, employees=fully_staffed_employees())
    result = solve_schedule(problem)

    assert result.success
    assert result.total_hours == pytest.approx(390, abs=1)
    assert not result.unused_employees
    # Fully saturated: every one of the 14 day/period requirements is binding.
    assert len(result.binding_requirements) == 14


def test_two_weekend_hires_not_enough():
    employees = dict(fully_staffed_employees())
    del employees["Weekend3"]
    problem = restaurant_scheduling_problem(max_hours=36, employees=employees)
    result = solve_schedule(problem)
    assert result.success is False


def test_weekdays_alone_already_use_most_of_the_original_teams_capacity():
    # Even with Saturday AND Sunday fully removed, the original 8 need 252 of
    # their 288 total available hours (8 x 36h) just for Mon-Fri -- only 36h
    # of slack exists in the whole week before any weekend demand is added.
    problem = restaurant_scheduling_problem(
        max_hours=36,
        requirement_overrides={
            ("Sat", "morning"): 0, ("Sat", "evening"): 0,
            ("Sun", "morning"): 0, ("Sun", "evening"): 0,
        },
    )
    result = solve_schedule(problem)
    assert result.success
    assert result.total_hours == pytest.approx(252, abs=1)


def test_saturday_only_hires_never_fix_it_regardless_of_count():
    # The shortfall isn't Saturday-specific -- it's spread across the whole
    # week -- so hires who can only work Saturday can never close it, no
    # matter how many are added.
    for n in (1, 2, 10, 20):
        employees = dict(EMPLOYEES)
        for i in range(n):
            employees[f"Sat{i + 1}"] = ["Sat"]
        problem = restaurant_scheduling_problem(max_hours=36, employees=employees)
        assert solve_schedule(problem).success is False, f"n={n} unexpectedly feasible"


def test_54h_cap_route_is_more_expensive_than_hiring_with_overtime_premium():
    # Raising the cap to 54h (with Saturday evening relaxed to 7, since that's
    # the only way it's feasible at all) is more expensive than the 3-hire
    # solution once a standard 1.5x overtime premium applies beyond 36h.
    problem = restaurant_scheduling_problem(
        max_hours=54, requirement_overrides={("Sat", "evening"): 7}, with_wages=True
    )
    result = solve_schedule(problem, minimize="hours")
    assert result.success

    _, _, total_with_overtime = compute_overtime_cost(problem, result.assignments)
    assert total_with_overtime > 229_500  # more expensive than the 3-hire solution


def test_cost_and_hours_minimizing_schedules_reach_same_total():
    # The schedule is fully saturated (zero slack everywhere), so total
    # weekday/weekend hours -- and therefore total cost -- are pinned down by
    # the requirements themselves; only the employee-to-shift assignment
    # differs between the two objectives.
    problem = restaurant_scheduling_problem(
        max_hours=36, employees=fully_staffed_employees(), with_wages=True
    )

    by_hours = solve_schedule(problem, minimize="hours")
    by_cost = solve_schedule(problem, minimize="cost")

    assert by_hours.success and by_cost.success
    assert by_hours.total_hours == pytest.approx(by_cost.total_hours, abs=1)
    assert by_hours.total_cost == pytest.approx(by_cost.total_cost, abs=1)
    assert by_cost.total_cost == pytest.approx(229_500, abs=500)
