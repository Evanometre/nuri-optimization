from dataclasses import dataclass


@dataclass
class ProductMixProblem:
    """A 'choose quantities of things, limited by shared resources' problem.

    products: {item_name: {"profit": float, "max_demand": float, <resource>: usage_per_unit, ...}}
    resources: {resource_name: available_amount}

    decision_label/unit_label only affect display wording (e.g. "Recommended
    production" vs "Recommended purchase") -- the underlying math is identical
    regardless of what business story the numbers represent.
    """

    products: dict
    resources: dict
    decision_label: str = "Recommended production"
    unit_label: str = "produce"

    def product_names(self):
        return list(self.products.keys())

    def resource_names(self):
        return list(self.resources.keys())

    def objective_coefficients(self):
        return [self.products[p]["profit"] for p in self.product_names()]

    def resource_matrix(self):
        """Rows = resources, columns = products. Missing usage defaults to 0."""
        return [
            [self.products[p].get(r, 0) for p in self.product_names()]
            for r in self.resource_names()
        ]

    def resource_limits(self):
        return [self.resources[r] for r in self.resource_names()]

    def demand_caps(self):
        return [self.products[p].get("max_demand") for p in self.product_names()]
