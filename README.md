# Nuri — Optimization Engine

Python tools for turning common business decisions into optimization and
statistics problems, plus a Streamlit app and a set of worked case studies.
Costs in the cases are in naira (₦).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies: numpy, scipy, pandas, ortools, pytest, streamlit.

## Running

| What | Command |
| --- | --- |
| Web app | `streamlit run app.py` |
| Tests | `python -m pytest` |
| A case study | `python run_<case>.py`, e.g. `python run_case8_primesack_production.py` |

## Streamlit app

`app.py` offers five problem types from a dropdown:

1. Product mix (LP/ILP)
2. Workforce scheduling
3. Factorial DOE (statistics)
4. Fractional factorial DOE (statistics)
5. Production planning (forecast + newsvendor)

Procurement and pricing are not in the app yet. They run through their
`run_case*.py` scripts.

## Library layout (`nuri/`)

| Module | Purpose |
| --- | --- |
| `models.py` | `ProductMixProblem`: products with profit, demand cap and per-resource usage, plus resource limits |
| `lp.py` | Continuous LP via SciPy/HiGHS, with shadow prices and reduced costs |
| `ilp.py` | Whole-unit integer solve via OR-Tools/SCIP; shadow prices come from the LP relaxation and are approximate |
| `results.py` | `OptimizationResult`, utilization, binding constraints, plain-language `explain()` |
| `scheduling.py` | Shift scheduling (OR-Tools): minimize total hours or wage cost |
| `procurement.py` | Multi-supplier allocation with MOQs, cost of capital and a concentration cap |
| `pricing.py` | Demand fitting (power-law/linear), single-price optimization, tiered-menu enumeration |
| `statistics/timeseries_forecast.py` | Trend plus seasonal regression, PRESS/LOOCV error, prediction intervals |
| `statistics/newsvendor.py` | Critical ratio, expected costs, production recommendation with capacity and overtime |
| `statistics/factorial_design.py` | 2^k factorial analysis, best combination, confirmation test |
| `statistics/fractional_factorial.py` | 2^(k-1) fractional factorial, alias map, holdout validation, segment effects |
| `inventory/` | Working-capital engine for the Meridian case: data-room loader, reconciliation, lost-sales test, ABC/XYZ segmentation, order-up-to policy with Monte Carlo, scenarios, rebalancing, suppliers, freight, network |
| `education/` | `simplex.py` and `branch_and_bound.py`: step-by-step solvers for teaching; not used for the main solves |

## Case studies (`cases/`)

Each case holds the data and problem definition. A matching `run_*.py` in the
repo root solves it and prints the analysis.

| Case | Topic | Method |
| --- | --- | --- |
| `furniture` | Furniture product mix | LP vs ILP |
| `case2_generic`, `case3_generic` | Textbook product-mix problems | LP/ILP |
| `case4_capital_allocation` | Distributor with a ₦2M cash budget | LP/ILP |
| `case5_restaurant_scheduling` | Restaurant staff scheduling | Scheduling |
| `case6_packaging_defects` | Packaging defect reduction | 2^4 factorial DOE (synthetic data) |
| `case7_choprun_promotions` | ChopRun promo experiment | 2^(4-1) fractional factorial (synthetic data) |
| `case8_primesack_production` | PrimeSack September production (real client data) | Forecast + newsvendor |
| `case9_pp_sack_factory` | PP sack factory capacity allocation | LP/ILP |
| `case10_procurement_optimization` | ~₦600M/yr spend, 6 categories, 12 suppliers | Procurement allocation |
| `case11_learnbridge_pricing` | EdTech single price vs three tiers | Pricing |
| `case12_meridian` | Distributor working capital and service (synthetic data room, not committed) | Inventory policy, simulation, rebalancing, suppliers |

Case 6 and 7 data is synthetic, and the module docstrings say so. Case 8
documents a missing client input (base production cost per sack) in its own
docstring.

## Meridian case

The data room workbook is not in the repo. Run the whole analysis with:

```bash
python run_case12_meridian.py path/to/Meridian_Data_Room.xlsx outputs/meridian
```

It prints each step, writes CSVs, `summary.json` and a model workbook to the output folder (gitignored), and is deterministic. Assumptions that are not in the data are named in `cases/case12_meridian.py`.

## Tests

`tests/` covers the LP/ILP and scheduling cases, procurement, pricing, the
newsvendor, forecasting, both DOE modules and the educational solvers.

## Deployment

The app is deployed on Streamlit Community Cloud. Secrets belong in
`.streamlit/secrets.toml`, which is gitignored; never commit them.
