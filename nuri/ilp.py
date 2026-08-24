from ortools.linear_solver import pywraplp

from nuri.results import OptimizationResult, compute_utilization
from nuri.lp import solve_lp


def solve_ilp(problem):
    """Solve a ProductMixProblem as an integer program (whole units only) via OR-Tools/SCIP."""
    names = problem.product_names()
    solver = pywraplp.Solver.CreateSolver("SCIP")

    variables = {}
    for name, cap in zip(names, problem.demand_caps()):
        upper = cap if cap is not None else solver.infinity()
        variables[name] = solver.IntVar(0, upper, name)

    for resource, usages, limit in zip(
        problem.resource_names(), problem.resource_matrix(), problem.resource_limits()
    ):
        solver.Add(
            sum(usage * variables[name] for usage, name in zip(usages, names)) <= limit
        )

    solver.Maximize(
        sum(coef * variables[name] for coef, name in zip(problem.objective_coefficients(), names))
    )

    status = solver.Solve()

    if status != pywraplp.Solver.OPTIMAL:
        return OptimizationResult(success=False, quantities={}, objective_value=0)

    quantities = {name: variables[name].solution_value() for name in names}
    objective_value = solver.Objective().Value()
    utilization, binding = compute_utilization(problem, quantities)

    # LP duality doesn't hold exactly once integrality is required, but the LP
    # relaxation's shadow prices are still a useful directional guide.
    relaxation = solve_lp(problem)

    return OptimizationResult(
        success=True,
        quantities=quantities,
        objective_value=objective_value,
        utilization=utilization,
        binding_constraints=binding,
        shadow_prices=relaxation.shadow_prices if relaxation.success else {},
        reduced_costs=relaxation.reduced_costs if relaxation.success else {},
        shadow_prices_are_approximate=True,
        decision_label=problem.decision_label,
        unit_label=problem.unit_label,
    )
