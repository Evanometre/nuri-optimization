"""From-scratch Branch & Bound for integer programs, built on top of
nuri.education.simplex (our own Simplex, not HiGHS) for every LP relaxation
it solves. Learning/validation module -- real ILPs should go through
nuri/ilp.py (OR-Tools).

At each node: solve the LP relaxation under the node's variable bounds. If
infeasible, or its objective can't beat the best integer solution found so
far, prune. If the relaxation happens to be integer already, it's a
candidate solution. Otherwise pick a fractional variable and branch into
x_i <= floor(v) and x_i >= ceil(v) -- together these throw away exactly the
fractional region around v while keeping every integer point on both sides.
"""

import math

from nuri.education.simplex import solve_simplex_general


def _is_integer(value, tol=1e-6):
    return abs(value - round(value)) <= tol


def solve_branch_and_bound(problem, tol=1e-6, node_limit=10000):
    names = problem.product_names()
    n = len(names)
    c = problem.objective_coefficients()

    resource_rows = [
        (list(row), "<=", limit)
        for row, limit in zip(problem.resource_matrix(), problem.resource_limits())
    ]

    initial_bounds = {
        i: (0, cap) for i, cap in enumerate(problem.demand_caps())
    }

    def bound_rows(bounds):
        rows = []
        for i, (lo, hi) in bounds.items():
            if hi is not None:
                unit = [1 if j == i else 0 for j in range(n)]
                rows.append((unit, "<=", hi))
            if lo > 0:
                unit = [1 if j == i else 0 for j in range(n)]
                rows.append((unit, ">=", lo))
        return rows

    best_x = None
    best_obj = float("-inf")
    nodes_explored = 0

    def recurse(bounds):
        nonlocal best_x, best_obj, nodes_explored
        nodes_explored += 1
        if nodes_explored > node_limit:
            return

        constraints = resource_rows + bound_rows(bounds)
        x, obj, status = solve_simplex_general(c, constraints)

        if status != "optimal":
            return  # infeasible relaxation -> prune
        if obj <= best_obj + tol:
            return  # can't beat the incumbent even with fractional slack -> prune

        fractional = [i for i in range(n) if not _is_integer(x[i])]
        if not fractional:
            best_x, best_obj = x, obj
            return

        branch_var = fractional[0]
        v = x[branch_var]
        lo, hi = bounds[branch_var]

        left_bounds = dict(bounds)
        left_bounds[branch_var] = (lo, math.floor(v))
        recurse(left_bounds)

        right_bounds = dict(bounds)
        right_bounds[branch_var] = (math.ceil(v), hi)
        recurse(right_bounds)

    recurse(initial_bounds)

    if best_x is None:
        return None, None, "infeasible", nodes_explored

    quantities = dict(zip(names, best_x))
    return quantities, best_obj, "optimal", nodes_explored
