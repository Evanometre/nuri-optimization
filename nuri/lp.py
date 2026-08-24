from scipy.optimize import linprog

from nuri.results import (
    OptimizationResult,
    compute_utilization,
    compute_shadow_prices_and_reduced_costs,
)


def solve_lp(problem):
    """Solve a ProductMixProblem as a continuous LP (maximize profit) via HiGHS."""
    names = problem.product_names()

    # linprog minimizes, so negate profit to maximize
    objective = [-c for c in problem.objective_coefficients()]

    A_ub = problem.resource_matrix()
    b_ub = problem.resource_limits()

    bounds = [(0, cap) for cap in problem.demand_caps()]

    result = linprog(
        objective,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        return OptimizationResult(success=False, quantities={}, objective_value=0)

    quantities = dict(zip(names, result.x))
    objective_value = -result.fun
    utilization, binding = compute_utilization(problem, quantities)

    # ineqlin.marginals are duals of the minimized (negated) objective, so negate
    # back to get shadow prices in terms of the original maximization.
    shadow_price_list = [-m for m in result.ineqlin.marginals]
    shadow_prices, reduced_costs = compute_shadow_prices_and_reduced_costs(
        problem, shadow_price_list
    )

    return OptimizationResult(
        success=True,
        quantities=quantities,
        objective_value=objective_value,
        utilization=utilization,
        binding_constraints=binding,
        shadow_prices=shadow_prices,
        reduced_costs=reduced_costs,
        decision_label=problem.decision_label,
        unit_label=problem.unit_label,
    )
