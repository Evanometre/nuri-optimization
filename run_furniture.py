from cases.furniture import furniture_problem
from nuri.lp import solve_lp
from nuri.ilp import solve_ilp

problem = furniture_problem()

print("=== LP (fractional units allowed) ===")
print(solve_lp(problem).explain())

print("\n=== ILP (whole units only) ===")
print(solve_ilp(problem).explain())
