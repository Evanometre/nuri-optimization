from nuri.models import ProductMixProblem

PRODUCTS = {
    "standard_50kg": {
        "profit": 132,
        "max_demand": 41_248,
        "polypropylene": 0.095,
        "loom": 0.0018,
        "printing": 0,
    },
    "printed_50kg": {
        "profit": 263,
        "max_demand": 45_868,
        "polypropylene": 0.100,
        "loom": 0.0020,
        "printing": 0.0008,
    },
    "fertilizer_bag": {
        "profit": 219,
        "max_demand": 21_934,
        "polypropylene": 0.090,
        "loom": 0.0017,
        "printing": 0.0006,
    },
    "feed_bag": {
        "profit": 232,
        "max_demand": 23_241,
        "polypropylene": 0.092,
        "loom": 0.0016,
        "printing": 0.0010,
    },
}

RESOURCES = {
    "polypropylene": 9_000,  # kg
    "loom": 148.6974,  # hours
    "printing": 64.2298,  # hours
}


def pp_sack_factory_problem():
    return ProductMixProblem(
        products=PRODUCTS,
        resources=RESOURCES,
        decision_label="Recommended production",
        unit_label="produce",
    )
