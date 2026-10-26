"""Independent tie-outs of the data room against the financial baseline.

The RFP says not to take summary figures at face value, so none of these
read the workbook's own Reconciliation_Log: every check recomputes from the
raw tables and compares to Financial_Baseline.
"""

import numpy as np
import pandas as pd

from nuri.inventory.data import HISTORY_MONTHS

ANNUALIZE = 12 / HISTORY_MONTHS


def inventory_value_by_row(dr):
    """Inventory_Snapshot with unit cost and at-cost value attached."""
    inv = dr.inventory.merge(dr.sku[["SKU", "Unit Cost ($)"]], on="SKU", how="left")
    inv["Value"] = inv["On-hand Units"] * inv["Unit Cost ($)"]
    inv["Aged Value"] = inv["Value"] * inv["Aged >180d %"]
    return inv


def run_reconciliation(dr, rel_tol=0.001):
    """Return a DataFrame of tie-outs: check, computed, reported, rel_diff, ok."""
    base = dr.baseline()
    sku = dr.sku
    inv = inventory_value_by_row(dr)
    demand_units = sku["Annual Demand (units)"]

    revenue = (demand_units * sku["Sell Price ($)"]).sum()
    cogs = (demand_units * sku["Unit Cost ($)"]).sum()
    inv_value = inv["Value"].sum()
    freight = dr.freight["Cost ($)"].sum() * ANNUALIZE
    expedite = dr.freight.loc[dr.freight["Expedite Flag"] == 1, "Cost ($)"].sum() * ANNUALIZE
    orders = dr.warehouse["Annual Outbound Orders"]
    otif = (orders * dr.warehouse["Current OTIF"]).sum() / orders.sum()

    hist = dr.demand.groupby("SKU")["Observed Units Sold"].sum()
    expected_hist = sku.set_index("SKU")["Annual Demand (units)"] * 1.5
    per_sku_err = (hist.reindex(expected_hist.index).fillna(0) - expected_hist).abs()

    rows = [
        ("Revenue = sum(annual demand x price)", revenue, base["Annual Revenue"]),
        ("COGS = sum(annual demand x cost)", cogs, base["COGS"]),
        ("Gross margin", 1 - cogs / revenue, base["Gross Margin"]),
        ("Supplier spend = COGS", dr.supplier["Annual Spend ($)"].sum(), base["COGS"]),
        ("Customer revenue = revenue", dr.customer["Annual Revenue ($)"].sum(), base["Annual Revenue"]),
        ("Inventory value (at cost)", inv_value, base["Inventory on hand"]),
        ("Inventory days = inventory / COGS x 365", inv_value / cogs * 365, base["Inventory days"]),
        ("Freight, annualised from 18 months", freight, base["Annual freight spend"]),
        ("Expedite freight, annualised", expedite, base["Expedite / emergency freight"]),
        (
            "Order fill rate = 1 - stockout lines / lines",
            1 - dr.demand["Stockout Flag"].mean(),
            base["Order fill rate"],
        ),
        ("Network OTIF (order-weighted)", otif, base["OTIF (on-time in-full)"]),
        (
            "Write-offs / aged value (implied rate; log says 15.4%)",
            base["Inventory write-offs / obsolescence"] / inv["Aged Value"].sum(),
            0.154,
        ),
    ]
    out = pd.DataFrame(rows, columns=["check", "computed", "reported"])
    out["rel_diff"] = (out["computed"] - out["reported"]) / out["reported"]
    out["ok"] = out["rel_diff"].abs() <= rel_tol
    out.attrs["history_units_max_sku_error"] = float(per_sku_err.max())
    out.attrs["history_units_total_error"] = float(
        hist.sum() - expected_hist.sum()
    )
    return out


def data_quality_findings(dr):
    """Gaps and inconsistencies that matter for decisions, as {finding: detail dict}."""
    sku = dr.sku
    inv_keys = set(zip(dr.inventory["Warehouse"], dr.inventory["SKU"]))
    dem_keys = set(zip(dr.demand["Warehouse"], dr.demand["SKU"]))

    inv = inventory_value_by_row(dr)
    stock_no_sales = inv[[k in (inv_keys - dem_keys) for k in zip(inv["Warehouse"], inv["SKU"])]]

    # Months with no demand row for a SKU-warehouse: treated as zero demand.
    present = dr.demand.groupby(["Warehouse", "SKU"])["Month"].nunique()
    missing_cells = int((18 - present).sum())

    supplier_lt = dr.supplier.set_index("Supplier ID")["Avg Lead Time Days"]
    sku_lt = sku.groupby("Supplier ID")["Supplier Lead Time (days)"].mean()
    lt_corr = float(pd.concat([supplier_lt, sku_lt], axis=1).dropna().corr().iloc[0, 1])

    monthly = dr.demand.groupby(["SKU", "Month"])["Observed Units Sold"].sum().unstack(fill_value=0)
    observed_cv = monthly.std(axis=1) / monthly.mean(axis=1)
    master_cv = sku.set_index("SKU")["Demand CV"]

    return {
        "stock_with_no_sales_18m": {
            "sku_warehouse_pairs": len(inv_keys - dem_keys),
            "value": float(stock_no_sales["Value"].sum()),
        },
        "sales_with_no_inventory_record": {"sku_warehouse_pairs": len(dem_keys - inv_keys)},
        "missing_demand_months_filled_with_zero": {"cells": missing_cells},
        "supplier_vs_sku_lead_time_correlation": {"corr": lt_corr},
        "master_cv_vs_observed_cv": {
            "median_master_cv": float(master_cv.median()),
            "median_observed_sku_month_cv": float(observed_cv.median()),
            "rank_corr": float(pd.concat([observed_cv, master_cv], axis=1).corr(method="spearman").iloc[0, 1]),
        },
        "no_customer_dimension": {
            "note": "Demand_History and Freight_Transactions carry no customer ID, so "
            "strategic-account service cannot be measured directly."
        },
        "operating_profit_missing": {
            "note": "RFP cites $2.85M operating profit; Financial_Baseline tab omits it."
        },
        "inventory_is_single_snapshot": {
            "note": "One point-in-time snapshot: inventory trend and stockout-vs-stock "
            "causality cannot be tested over time."
        },
    }
