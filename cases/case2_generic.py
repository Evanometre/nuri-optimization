from nuri.models import ProductMixProblem

# max z = 240x + 360y
# s.t. 3x + 2y <= 12
#      x + 4y  <= 14
#      x, y >= 0

PRODUCTS = {
    "x": {"profit": 240, "c1": 3, "c2": 1},
    "y": {"profit": 360, "c1": 2, "c2": 4},
}

RESOURCES = {
    "c1": 12,
    "c2": 14,
}


def case2_problem():
    return ProductMixProblem(products=PRODUCTS, resources=RESOURCES)
