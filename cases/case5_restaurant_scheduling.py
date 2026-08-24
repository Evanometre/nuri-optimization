from nuri.scheduling import SchedulingProblem

EMPLOYEES = {
    "A": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
    "B": ["Mon", "Tue", "Wed", "Thu", "Fri"],
    "C": ["Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    "D": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    "E": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    "F": ["Wed", "Thu", "Fri", "Sat", "Sun"],
    "G": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
    "H": ["Fri", "Sat", "Sun"],
}

STAFFING_REQUIREMENTS = {
    "Mon": {"morning": 3, "evening": 4},
    "Tue": {"morning": 3, "evening": 4},
    "Wed": {"morning": 3, "evening": 4},
    "Thu": {"morning": 4, "evening": 5},
    "Fri": {"morning": 5, "evening": 7},
    "Sat": {"morning": 6, "evening": 8},
    "Sun": {"morning": 4, "evening": 5},
}


# Assumed wage rates for the cost-minimizing (v2) version -- adjust to real figures
# once the owner provides them. Weekend carries a 50% premium.
WEEKDAY_HOURLY_WAGE = 500
WEEKEND_HOURLY_WAGE = 750


def restaurant_scheduling_problem(
    max_hours=36,
    employees=None,
    requirement_overrides=None,
    with_wages=False,
):
    requirements = {day: dict(periods) for day, periods in STAFFING_REQUIREMENTS.items()}
    for (day, period), value in (requirement_overrides or {}).items():
        requirements[day][period] = value

    return SchedulingProblem(
        employees=employees if employees is not None else EMPLOYEES,
        staffing_requirements=requirements,
        max_hours_per_employee=max_hours,
        weekday_hourly_wage=WEEKDAY_HOURLY_WAGE if with_wages else None,
        weekend_hourly_wage=WEEKEND_HOURLY_WAGE if with_wages else None,
    )


def fully_staffed_employees():
    """Original 8 plus the 3 weekend-only hires found necessary to make the
    schedule feasible at a real 36h/week cap (see case5 investigation)."""
    employees = dict(EMPLOYEES)
    for i in range(3):
        employees[f"Weekend{i + 1}"] = ["Fri", "Sat", "Sun"]
    return employees
