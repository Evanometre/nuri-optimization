"""Multi-supplier, multi-category procurement allocation.

Different shape from nuri/models.py's ProductMixProblem: here the decision is
which supplier(s) to buy each purchasing category from, not how much of each
product to make. Needs a binary "is this supplier used at all" variable per
(supplier, category) pair to enforce minimum order quantities (MOQ) --
you can't buy a little bit from a supplier that requires a minimum order.

Effective cost per usable unit folds in freight, rejection/wastage (you have
to buy more to net the same usable amount), and the financing value of
longer payment terms (a supplier who lets you pay in 60 days instead of 30
is effectively cheaper, since you keep your cash longer) -- so "unit price"
alone is deliberately not what gets minimized.
"""

from dataclasses import dataclass, field

from ortools.linear_solver import pywraplp


@dataclass
class SupplierOffer:
    supplier: str
    category: str
    price: float
    freight: float
    moq: float
    capacity: float
    rejection_rate: float  # fraction wasted/rejected, 0-1
    payment_term_days: float


@dataclass
class ProcurementProblem:
    demand: dict  # category -> annual usable-unit demand
    offers: list  # list of SupplierOffer
    budget: float = None
    concentration_cap: float = None  # max fraction of a category's demand from one supplier
    cost_of_capital_rate: float = 0.20
    single_source_only: bool = False  # forces exactly one supplier per category -- used to isolate "supplier selection" value from "diversification" cost


def effective_unit_cost(offer, cost_of_capital_rate):
    gross = (offer.price + offer.freight) / (1 - offer.rejection_rate)
    financing_benefit = offer.price * cost_of_capital_rate * (offer.payment_term_days / 365)
    return gross - financing_benefit


@dataclass
class ProcurementResult:
    success: bool
    allocation: dict = field(default_factory=dict)  # (supplier, category) -> quantity
    total_cost: float = 0
    cost_by_category: dict = field(default_factory=dict)
    suppliers_used_by_category: dict = field(default_factory=dict)
    concentration_binding: dict = field(default_factory=dict)  # (supplier, category) -> True if at the cap
    budget_binding: bool = False


def solve_procurement(problem):
    solver = pywraplp.Solver.CreateSolver("SCIP")

    offers_by_category = {}
    for offer in problem.offers:
        offers_by_category.setdefault(offer.category, []).append(offer)

    x, y = {}, {}
    for offer in problem.offers:
        key = (offer.supplier, offer.category)
        x[key] = solver.NumVar(0, offer.capacity, f"x_{offer.supplier}_{offer.category}")
        y[key] = solver.BoolVar(f"y_{offer.supplier}_{offer.category}")
        solver.Add(x[key] <= offer.capacity * y[key])
        solver.Add(x[key] >= offer.moq * y[key])

    for category, demand in problem.demand.items():
        relevant = offers_by_category.get(category, [])
        solver.Add(sum(x[(o.supplier, o.category)] for o in relevant) >= demand)

        if problem.concentration_cap is not None:
            for o in relevant:
                solver.Add(x[(o.supplier, o.category)] <= problem.concentration_cap * demand)

        if problem.single_source_only:
            solver.Add(sum(y[(o.supplier, o.category)] for o in relevant) <= 1)

    unit_costs = {
        (o.supplier, o.category): effective_unit_cost(o, problem.cost_of_capital_rate)
        for o in problem.offers
    }

    if problem.budget is not None:
        solver.Add(sum(unit_costs[k] * x[k] for k in x) <= problem.budget)

    solver.Minimize(sum(unit_costs[k] * x[k] for k in x))

    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        return ProcurementResult(success=False)

    allocation = {k: v.solution_value() for k, v in x.items() if v.solution_value() > 1e-6}
    total_cost = sum(unit_costs[k] * q for k, q in allocation.items())

    cost_by_category, suppliers_used, concentration_binding = {}, {}, {}
    for (supplier, category), qty in allocation.items():
        cost_by_category[category] = cost_by_category.get(category, 0) + unit_costs[(supplier, category)] * qty
        suppliers_used.setdefault(category, []).append((supplier, qty))
        if problem.concentration_cap is not None:
            cap = problem.concentration_cap * problem.demand[category]
            concentration_binding[(supplier, category)] = qty >= cap - 1e-6

    total_spend = sum(unit_costs[k] * q for k, q in allocation.items())
    budget_binding = problem.budget is not None and total_spend >= problem.budget - 1e-3

    return ProcurementResult(
        success=True,
        allocation=allocation,
        total_cost=total_cost,
        cost_by_category=cost_by_category,
        suppliers_used_by_category=suppliers_used,
        concentration_binding=concentration_binding,
        budget_binding=budget_binding,
    )


def incumbent_cost(problem, incumbent_by_category):
    """Cost of the naive status-quo plan: single-source everything from a
    designated 'incumbent' supplier per category, ignoring optimization
    entirely -- the baseline the savings estimate is measured against."""
    total = 0.0
    breakdown = {}
    for category, demand in problem.demand.items():
        supplier = incumbent_by_category[category]
        offer = next(o for o in problem.offers if o.supplier == supplier and o.category == category)
        cost = effective_unit_cost(offer, problem.cost_of_capital_rate) * demand
        total += cost
        breakdown[category] = cost
    return total, breakdown


def cost_composition(offer, cost_of_capital_rate):
    """Break one supplier's effective unit cost into its components, for
    explaining WHY a chosen supplier is cheaper -- not an independently
    re-optimizable lever, just a description of the winning offer."""
    base_price = offer.price
    freight_effect = offer.freight / (1 - offer.rejection_rate)
    rejection_effect = offer.price * (1 / (1 - offer.rejection_rate) - 1)
    financing_benefit = offer.price * cost_of_capital_rate * (offer.payment_term_days / 365)
    return {
        "price": base_price,
        "freight_effect": freight_effect,
        "rejection_effect": rejection_effect,
        "financing_benefit": -financing_benefit,
    }
