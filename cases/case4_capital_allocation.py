from nuri.models import ProductMixProblem

# Distributor with N2,000,000 cash. Decide how many units of each product to
# buy to maximize expected gross profit, subject to a cash budget and
# per-product demand ceilings. Units must be whole (you can't buy 37.42
# cartons), so this is naturally an ILP.

PRODUCTS = {
    "product_a": {"profit": 2000, "cash": 8000, "max_demand": 100},
    "product_b": {"profit": 4000, "cash": 12000, "max_demand": 80},
    "product_c": {"profit": 2000, "cash": 5000, "max_demand": 150},
    "product_d": {"profit": 5000, "cash": 20000, "max_demand": 60},
}

RESOURCES = {
    "cash": 2_000_000,
}


def capital_allocation_problem():
    return ProductMixProblem(
        products=PRODUCTS,
        resources=RESOURCES,
        decision_label="Recommended purchase quantities",
        unit_label="buy",
    )
