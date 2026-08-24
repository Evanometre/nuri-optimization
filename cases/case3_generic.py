from nuri.models import ProductMixProblem

# max z = 4x + 3y
# s.t. 2x + 3y <= 17
#      2x + y  <= 10
#      x, y >= 0 and integral

PRODUCTS = {
    "x": {"profit": 4, "c1": 2, "c2": 2},
    "y": {"profit": 3, "c1": 3, "c2": 1},
}

RESOURCES = {
    "c1": 17,
    "c2": 10,
}


def case3_problem():
    return ProductMixProblem(products=PRODUCTS, resources=RESOURCES)
