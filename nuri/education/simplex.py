"""A from-scratch Simplex solver (tableau method), built for understanding, not
production use. Real problems should go through nuri/lp.py (HiGHS via SciPy) --
this module exists to build intuition and to be checked against that solver,
not to replace it.

Handles: maximize c^T x subject to A x <= b, x >= 0, with b >= 0 (so the
origin is a feasible starting point -- no Phase 1 / artificial variables
needed yet). That covers every LP case in this project so far, since resource
limits are always non-negative.

Uses Bland's rule (always pick the lowest-index eligible column/row) to
guarantee termination -- without it, a degenerate tableau can cycle forever
between the same sequence of bases.
"""

import numpy as np


def solve_simplex(c, A, b, max_iterations=1000):
    """c: (n,) profit coefficients. A: (m,n) constraint matrix. b: (m,) limits, b>=0.

    Returns (x, objective_value, status), status in
    {"optimal", "unbounded", "max_iterations_exceeded"}.
    """
    c = np.asarray(c, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    m, n = A.shape

    if np.any(b < 0):
        raise ValueError("solve_simplex requires b >= 0 (no Phase 1 implemented yet)")

    # Tableau layout: [ A | I | b ]  with an extra bottom row for -c (the
    # objective, written as z - c^T x = 0 so we pivot away negative entries).
    T = np.zeros((m + 1, n + m + 1))
    T[:m, :n] = A
    T[:m, n:n + m] = np.eye(m)
    T[:m, -1] = b
    T[-1, :n] = -c

    basis = list(range(n, n + m))  # slack variables start basic

    for _ in range(max_iterations):
        obj_row = T[-1, :-1]
        candidates = np.where(obj_row < -1e-9)[0]
        if len(candidates) == 0:
            x = np.zeros(n + m)
            for row, var in enumerate(basis):
                x[var] = T[row, -1]
            return x[:n], T[-1, -1], "optimal"

        pivot_col = candidates[0]  # Bland's rule: smallest eligible column index

        col = T[:m, pivot_col]
        rhs = T[:m, -1]
        ratios = np.where(col > 1e-9, np.divide(rhs, col, out=np.full(m, np.inf), where=col > 1e-9), np.inf)

        if np.all(np.isinf(ratios)):
            return None, None, "unbounded"

        min_ratio = ratios.min()
        tied_rows = np.where(np.abs(ratios - min_ratio) < 1e-9)[0]
        pivot_row = min(tied_rows, key=lambda r: basis[r])  # Bland's rule again, for the leaving variable

        pivot_val = T[pivot_row, pivot_col]
        T[pivot_row, :] /= pivot_val
        for r in range(m + 1):
            if r != pivot_row:
                T[r, :] -= T[r, pivot_col] * T[pivot_row, :]
        basis[pivot_row] = pivot_col

    return None, None, "max_iterations_exceeded"


def solve_simplex_general(c, constraints, max_iterations=1000, big_m=1e7):
    """Big-M method: handles '<=', '>=', and '=' constraints (still x >= 0).

    constraints: list of (row, relation, rhs) with relation in {"<=", ">=", "="}.
    rhs may be negative -- it's normalized (row/rhs negated, relation flipped)
    so every artificial-variable row starts with a non-negative RHS.

    This is what unlocks branch-and-bound's ">=" branches (x_i >= ceil(v)),
    which the plain <=-only solve_simplex() above can't express.
    """
    c = np.asarray(c, dtype=float)
    n = len(c)
    m = len(constraints)

    norm = []
    for row, relation, rhs in constraints:
        row = list(row)
        if rhs < 0:
            row = [-v for v in row]
            rhs = -rhs
            relation = {"<=": ">=", ">=": "<=", "=": "="}[relation]
        norm.append((row, relation, rhs))

    n_slack_surplus = sum(1 for _, rel, _ in norm if rel in ("<=", ">="))
    n_artificial = sum(1 for _, rel, _ in norm if rel in (">=", "="))
    total_cols = n + n_slack_surplus + n_artificial

    T = np.zeros((m + 1, total_cols + 1))
    basis = [None] * m

    slack_idx = n
    artificial_idx = n + n_slack_surplus
    artificial_rows = []

    for i, (row, relation, rhs) in enumerate(norm):
        T[i, :n] = row
        T[i, -1] = rhs
        if relation == "<=":
            T[i, slack_idx] = 1
            basis[i] = slack_idx
            slack_idx += 1
        elif relation == ">=":
            T[i, slack_idx] = -1
            slack_idx += 1
            T[i, artificial_idx] = 1
            basis[i] = artificial_idx
            artificial_rows.append(i)
            artificial_idx += 1
        else:  # "="
            T[i, artificial_idx] = 1
            basis[i] = artificial_idx
            artificial_rows.append(i)
            artificial_idx += 1

    # Objective: maximize c^Tx - M*sum(artificials) -> bottom row entry -c_j
    # for structural vars, 0 for slack/surplus, +M for artificials -- then
    # canonicalize by zeroing out the (currently basic) artificial columns.
    T[-1, :n] = -c
    T[-1, n + n_slack_surplus:total_cols] = big_m
    for i in artificial_rows:
        T[-1, :] -= big_m * T[i, :]

    for _ in range(max_iterations):
        obj_row = T[-1, :-1]
        candidates = np.where(obj_row < -1e-7)[0]
        if len(candidates) == 0:
            break
        pivot_col = candidates[0]

        col = T[:m, pivot_col]
        rhs = T[:m, -1]
        ratios = np.where(col > 1e-9, np.divide(rhs, col, out=np.full(m, np.inf), where=col > 1e-9), np.inf)

        if np.all(np.isinf(ratios)):
            return None, None, "unbounded"

        min_ratio = ratios.min()
        tied_rows = np.where(np.abs(ratios - min_ratio) < 1e-9)[0]
        pivot_row = min(tied_rows, key=lambda r: basis[r])

        pivot_val = T[pivot_row, pivot_col]
        T[pivot_row, :] /= pivot_val
        for r in range(m + 1):
            if r != pivot_row:
                T[r, :] -= T[r, pivot_col] * T[pivot_row, :]
        basis[pivot_row] = pivot_col
    else:
        return None, None, "max_iterations_exceeded"

    x = np.zeros(total_cols)
    for row, var in enumerate(basis):
        x[var] = T[row, -1]

    # Any artificial variable still positive in the basis means the original
    # (non-relaxed) feasible region is empty.
    if np.any(x[n + n_slack_surplus:total_cols] > 1e-6):
        return None, None, "infeasible"

    return x[:n], float(np.dot(c, x[:n])), "optimal"


def solve_simplex_for_problem(problem):
    """Adapt a nuri.models.ProductMixProblem into standard form and solve it.

    max_demand bounds are folded in as extra <= rows (x_i <= cap), since this
    tableau only natively handles A x <= b, not separate variable bounds.
    """
    names = problem.product_names()
    n = len(names)

    c = problem.objective_coefficients()
    A_rows = list(problem.resource_matrix())
    b = list(problem.resource_limits())

    for i, cap in enumerate(problem.demand_caps()):
        if cap is not None:
            row = [0] * n
            row[i] = 1
            A_rows.append(row)
            b.append(cap)

    x, objective_value, status = solve_simplex(c, A_rows, b)
    if status != "optimal":
        return None, None, status

    quantities = dict(zip(names, x))
    return quantities, objective_value, status
