from dataclasses import dataclass, field


@dataclass
class OptimizationResult:
    success: bool
    quantities: dict
    objective_value: float
    utilization: dict = field(default_factory=dict)
    binding_constraints: list = field(default_factory=list)
    shadow_prices: dict = field(default_factory=dict)
    reduced_costs: dict = field(default_factory=dict)
    decision_label: str = "Recommended production"
    unit_label: str = "produce"
    shadow_prices_are_approximate: bool = False

    def explain(self, currency="N"):
        if not self.success:
            return "No feasible solution found."

        lines = [f"{self.decision_label}:"]
        for name, qty in self.quantities.items():
            lines.append(f"  {name}: {qty:.2f}")
        lines.append(f"Expected contribution: {currency}{self.objective_value:,.2f}")

        if self.utilization:
            lines.append("\nResource utilization:")
            for name, pct in self.utilization.items():
                lines.append(f"  {name}: {pct:.1%}")

        if self.binding_constraints:
            lines.append(f"\nBinding constraint(s): {', '.join(self.binding_constraints)}")
            lines.append("These are the resources actually limiting the solution.")

        if self.shadow_prices:
            header = "\nMarginal value of each resource (shadow price)"
            header += " -- approximate, from the LP relaxation:" if self.shadow_prices_are_approximate else ":"
            lines.append(header)
            for name, price in self.shadow_prices.items():
                lines.append(f"  {name}: {currency}{price:,.2f} per extra unit")
            if self.shadow_prices_are_approximate:
                lines.append(
                    "  (Whole-unit requirements make the true marginal value jump "
                    "discontinuously; treat these as directional, not exact.)"
                )

        skipped, tied = classify_zero_quantity_products(self.quantities, self.reduced_costs)

        if skipped:
            lines.append("\nWhy other products weren't recommended:")
            for name, cost in skipped.items():
                lines.append(
                    f"  {name}: to {self.unit_label} one would cost {currency}{-cost:,.2f} "
                    f"more in resources (at their marginal value) than it earns in profit."
                )

        if tied:
            lines.append("\nExactly tied with the marginal value of resources:")
            for name in tied:
                lines.append(
                    f"  {name}: ties exactly with the marginal value of the resources "
                    "it would use; including it wouldn't change profit."
                )

        return "\n".join(lines)


def classify_zero_quantity_products(quantities, reduced_costs, tolerance=1e-6):
    """Split zero-quantity products into genuinely skipped (negative reduced cost)
    vs. exactly tied with the marginal value of resources (reduced cost ~0).
    """
    skipped = {}
    tied = []

    for name, cost in reduced_costs.items():
        if quantities.get(name, 0) > tolerance:
            continue
        if cost < -tolerance:
            skipped[name] = cost
        elif abs(cost) <= tolerance:
            tied.append(name)

    return skipped, tied


def compute_utilization(problem, quantities, tolerance=1e-4):
    """Given a solved ProductMixProblem, return {resource: pct_used} and binding constraint names."""
    utilization = {}
    binding = []

    for resource, limit in zip(problem.resource_names(), problem.resource_limits()):
        used = sum(
            problem.products[p].get(resource, 0) * quantities[p]
            for p in problem.product_names()
        )
        pct = used / limit if limit else 0
        utilization[resource] = pct
        if limit - used <= tolerance:
            binding.append(resource)

    return utilization, binding


def compute_shadow_prices_and_reduced_costs(problem, shadow_price_list):
    """shadow_price_list: marginal value of each resource, in problem.resource_names() order.

    Returns (shadow_prices dict, reduced_costs dict). Reduced cost for a product is
    profit minus the opportunity cost of the resources it would consume, priced at
    their shadow value. Negative reduced cost = not worth producing.
    """
    shadow_prices = dict(zip(problem.resource_names(), shadow_price_list))

    reduced_costs = {}
    for name, attrs in problem.products.items():
        opportunity_cost = sum(
            attrs.get(r, 0) * shadow_prices[r] for r in problem.resource_names()
        )
        reduced_costs[name] = attrs["profit"] - opportunity_cost

    return shadow_prices, reduced_costs
