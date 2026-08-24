from nuri.models import ProductMixProblem

PRODUCTS = {
    "tables": {
        "profit": 35000,
        "wood": 12,
        "machine": 3,
        "labour": 4,
        "max_demand": 40,
    },
    "chairs": {
        "profit": 15000,
        "wood": 4,
        "machine": 1,
        "labour": 2,
        "max_demand": 100,
    },
    "shelves": {
        "profit": 20000,
        "wood": 6,
        "machine": 2,
        "labour": 3,
        "max_demand": 60,
    },
}

RESOURCES = {
    "wood": 400,
    "machine": 100,
    "labour": 160,
}


def furniture_problem():
    return ProductMixProblem(products=PRODUCTS, resources=RESOURCES)
