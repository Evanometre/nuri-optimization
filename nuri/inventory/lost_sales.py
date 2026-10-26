"""Test management's $1.38M stockout lost-sales estimate against transactions.

Demand_History has a stockout flag per month/warehouse/SKU line. Sales in a
flagged month are censored, so lost demand is estimated two ways:

* shortfall: for each flagged line, that pair's average *non*-stockout month
  minus what was actually sold (never below zero). A central estimate.
* ceiling: each flagged line lost a whole average month of demand. This is
  deliberately generous: it assumes the flagged line sold nothing it could
  have sold.

Anything above the ceiling would need flagged lines to have lost more than a
full month of demand each; the observed units show they did not.
"""

import pandas as pd

from nuri.inventory.data import HISTORY_MONTHS
from nuri.inventory.demand import build_demand_matrices

ANNUALIZE = 12 / HISTORY_MONTHS


def estimate_lost_sales(dr, management_estimate=1_380_000):
    units, stockout = build_demand_matrices(dr)
    open_mean = units.where(stockout == 0).mean(axis=1)

    long = (
        units.stack().rename("sold").to_frame().join(stockout.stack().rename("flag")).reset_index()
    )
    long = long[long["flag"] == 1].merge(
        open_mean.rename("open_mean").reset_index(), on=["Warehouse", "SKU"], how="left"
    )
    long["open_mean"] = long["open_mean"].fillna(long["sold"])
    long = long.merge(
        dr.sku[["SKU", "ABC Class", "Category", "Sell Price ($)", "Unit Cost ($)"]], on="SKU"
    )
    long["short_units"] = (long["open_mean"] - long["sold"]).clip(lower=0)
    long["short_revenue"] = long["short_units"] * long["Sell Price ($)"]
    long["short_margin"] = long["short_units"] * (long["Sell Price ($)"] - long["Unit Cost ($)"])
    long["ceiling_revenue"] = long["open_mean"] * long["Sell Price ($)"]

    annual = lambda col: float(long[col].sum() * ANNUALIZE)
    result = {
        "management_estimate": management_estimate,
        "shortfall_revenue": annual("short_revenue"),
        "shortfall_margin": annual("short_margin"),
        "ceiling_revenue": annual("ceiling_revenue"),
        "flagged_lines_per_year": float(len(long) * ANNUALIZE),
        "by_class": (long.groupby("ABC Class")[["short_revenue", "ceiling_revenue"]].sum() * ANNUALIZE),
        "by_category": (long.groupby("Category")[["short_revenue", "ceiling_revenue"]].sum() * ANNUALIZE),
    }
    # No flagged lines -> nothing to compare against; report infinity, not a crash.
    result["management_multiple_of_ceiling"] = (
        management_estimate / result["ceiling_revenue"] if result["ceiling_revenue"] > 0 else float("inf")
    )
    result["management_multiple_of_shortfall"] = (
        management_estimate / result["shortfall_revenue"] if result["shortfall_revenue"] > 0 else float("inf")
    )
    result["stockout_rate_by_class"] = (
        dr.demand.merge(dr.sku[["SKU", "ABC Class"]], on="SKU").groupby("ABC Class")["Stockout Flag"].mean()
    )
    return result
