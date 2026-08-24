from dataclasses import dataclass, field

from ortools.linear_solver import pywraplp

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
WEEKEND_DAYS = {"Sat", "Sun"}

# A "full" shift covers both periods in the same day with one person.
SHIFT_HOURS = {"morning": 6, "evening": 6, "full": 12}
SHIFT_COVERS = {"morning": ["morning"], "evening": ["evening"], "full": ["morning", "evening"]}


@dataclass
class SchedulingProblem:
    employees: dict  # {employee: [available_days]}
    staffing_requirements: dict  # {day: {"morning": n, "evening": n}}
    max_hours_per_employee: int = 36
    max_working_days_per_employee: int = 6  # implies >=1 day off
    weekday_hourly_wage: float = None
    weekend_hourly_wage: float = None

    def shift_cost(self, day, shift):
        wage = self.weekend_hourly_wage if day in WEEKEND_DAYS else self.weekday_hourly_wage
        return SHIFT_HOURS[shift] * wage


@dataclass
class ScheduleResult:
    success: bool
    assignments: list = field(default_factory=list)  # (employee, day, shift)
    total_hours: float = 0
    total_cost: float = None
    hours_by_employee: dict = field(default_factory=dict)
    cost_by_employee: dict = field(default_factory=dict)
    days_worked_by_employee: dict = field(default_factory=dict)
    unused_employees: list = field(default_factory=list)
    binding_requirements: list = field(default_factory=list)  # (day, period) exactly met
    slack_requirements: dict = field(default_factory=dict)  # (day, period) -> extra staff beyond minimum

    def explain(self, currency="N"):
        if not self.success:
            return "No feasible schedule found."

        lines = [f"Minimum total labour hours: {self.total_hours:.0f}"]
        if self.total_cost is not None:
            lines.append(f"Total labour cost: {currency}{self.total_cost:,.2f}")

        lines.append("\nHours worked per employee:")
        for emp, hrs in self.hours_by_employee.items():
            days = self.days_worked_by_employee[emp]
            cost_note = (
                f", {currency}{self.cost_by_employee[emp]:,.2f}"
                if self.cost_by_employee
                else ""
            )
            lines.append(f"  {emp}: {hrs:.0f}h over {days} day(s){cost_note}")

        if self.unused_employees:
            lines.append(f"\nEmployees not needed at all: {', '.join(self.unused_employees)}")
        else:
            lines.append("\nAll employees were used at least once.")

        if self.binding_requirements:
            lines.append("\nStaffing requirements met with zero slack (binding):")
            for day, period in self.binding_requirements:
                lines.append(f"  {day} {period}")

        return "\n".join(lines)


def _build_model(problem):
    """Shared constraint-building for both the hours- and cost-minimizing solves."""
    solver = pywraplp.Solver.CreateSolver("SCIP")

    employees = list(problem.employees.keys())
    x = {}
    for e in employees:
        available_days = set(problem.employees[e])
        for d in DAYS:
            if d not in available_days:
                continue
            for s in SHIFT_HOURS:
                x[(e, d, s)] = solver.BoolVar(f"x_{e}_{d}_{s}")

    def var(e, d, s):
        return x.get((e, d, s))

    for d in DAYS:
        for period in ("morning", "evening"):
            required = problem.staffing_requirements[d][period]
            covering_vars = [
                var(e, d, s)
                for e in employees
                for s in SHIFT_HOURS
                if period in SHIFT_COVERS[s] and var(e, d, s) is not None
            ]
            solver.Add(sum(covering_vars) >= required)

    for e in employees:
        for d in DAYS:
            day_vars = [var(e, d, s) for s in SHIFT_HOURS if var(e, d, s) is not None]
            if day_vars:
                solver.Add(sum(day_vars) <= 1)

    for e in employees:
        emp_vars_hours = [
            SHIFT_HOURS[s] * var(e, d, s)
            for d in DAYS
            for s in SHIFT_HOURS
            if var(e, d, s) is not None
        ]
        emp_vars_days = [
            var(e, d, s) for d in DAYS for s in SHIFT_HOURS if var(e, d, s) is not None
        ]
        if emp_vars_hours:
            solver.Add(sum(emp_vars_hours) <= problem.max_hours_per_employee)
        if emp_vars_days:
            solver.Add(sum(emp_vars_days) <= problem.max_working_days_per_employee)

    return solver, x, employees


def _extract_result(problem, x, employees):
    assignments = [
        (e, d, s) for (e, d, s), var_ in x.items() if var_.solution_value() > 0.5
    ]

    hours_by_employee = {e: 0 for e in employees}
    days_worked_by_employee = {e: 0 for e in employees}
    for e, d, s in assignments:
        hours_by_employee[e] += SHIFT_HOURS[s]
        days_worked_by_employee[e] += 1

    has_wages = problem.weekday_hourly_wage is not None and problem.weekend_hourly_wage is not None
    cost_by_employee = {}
    total_cost = None
    if has_wages:
        cost_by_employee = {e: 0 for e in employees}
        for e, d, s in assignments:
            cost_by_employee[e] += problem.shift_cost(d, s)
        total_cost = sum(cost_by_employee.values())

    unused_employees = [e for e in employees if hours_by_employee[e] == 0]
    total_hours = sum(hours_by_employee.values())

    binding_requirements = []
    slack_requirements = {}
    for d in DAYS:
        for period in ("morning", "evening"):
            required = problem.staffing_requirements[d][period]
            covering = sum(
                1
                for (e, dd, s) in assignments
                if dd == d and period in SHIFT_COVERS[s]
            )
            slack = covering - required
            slack_requirements[(d, period)] = slack
            if slack == 0:
                binding_requirements.append((d, period))

    return ScheduleResult(
        success=True,
        assignments=assignments,
        total_hours=total_hours,
        total_cost=total_cost,
        hours_by_employee=hours_by_employee,
        cost_by_employee=cost_by_employee,
        days_worked_by_employee=days_worked_by_employee,
        unused_employees=unused_employees,
        binding_requirements=binding_requirements,
        slack_requirements=slack_requirements,
    )


def compute_overtime_cost(problem, assignments, overtime_threshold=36, overtime_multiplier=1.5):
    """Pay per employee: normal day-based rate for the first `overtime_threshold`
    hours (Mon-Sun order), `overtime_multiplier`x that day's rate beyond it.
    Requires weekday_hourly_wage/weekend_hourly_wage to be set on the problem.
    """
    by_employee = {}
    for e, d, s in assignments:
        by_employee.setdefault(e, []).append((d, s))

    cost_by_employee = {}
    overtime_hours_by_employee = {}
    for e, shifts in by_employee.items():
        shifts_sorted = sorted(shifts, key=lambda ds: DAYS.index(ds[0]))
        cum_hours = 0
        cost = 0
        overtime_hours = 0
        for d, s in shifts_sorted:
            hrs = SHIFT_HOURS[s]
            base_rate = problem.weekend_hourly_wage if d in WEEKEND_DAYS else problem.weekday_hourly_wage
            normal_hrs = max(0, min(hrs, overtime_threshold - cum_hours))
            ot_hrs = hrs - normal_hrs
            cost += normal_hrs * base_rate + ot_hrs * base_rate * overtime_multiplier
            overtime_hours += ot_hrs
            cum_hours += hrs
        cost_by_employee[e] = cost
        overtime_hours_by_employee[e] = overtime_hours

    return cost_by_employee, overtime_hours_by_employee, sum(cost_by_employee.values())


def solve_schedule(problem, minimize="hours"):
    """minimize: "hours" (default) or "cost" (requires weekday/weekend wages set)."""
    solver, x, employees = _build_model(problem)

    if minimize == "hours":
        solver.Minimize(sum(SHIFT_HOURS[s] * var_ for (e, d, s), var_ in x.items()))
    elif minimize == "cost":
        if problem.weekday_hourly_wage is None or problem.weekend_hourly_wage is None:
            raise ValueError("minimize='cost' requires weekday_hourly_wage and weekend_hourly_wage")
        solver.Minimize(
            sum(problem.shift_cost(d, s) * var_ for (e, d, s), var_ in x.items())
        )
    else:
        raise ValueError(f"Unknown minimize target: {minimize!r}")

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        return ScheduleResult(success=False)

    return _extract_result(problem, x, employees)
