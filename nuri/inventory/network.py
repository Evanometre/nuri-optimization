"""Warehouse network economics: load per square foot, pick-pack cost, space effects of the policy."""

import pandas as pd


def warehouse_profile(dr, pairs):
    wh = dr.warehouse.set_index("Warehouse").copy()
    wh["order_share"] = wh["Annual Outbound Orders"] / wh["Annual Outbound Orders"].sum()
    wh["capacity_share"] = wh["Capacity (sqft)"] / wh["Capacity (sqft)"].sum()
    wh["orders_per_sqft"] = wh["Annual Outbound Orders"] / wh["Capacity (sqft)"]
    wh["pick_pack_cost_year"] = wh["Annual Outbound Orders"] * wh["Avg Pick-Pack Cost ($/order)"]
    by = pairs.groupby("Warehouse")
    wh["inventory_value"] = by["value"].sum()
    wh["inventory_units"] = by["On-hand Units"].sum()
    wh["demand_units_month"] = by["mean_month"].sum()
    wh["inventory_share"] = wh["inventory_value"] / wh["inventory_value"].sum()
    wh["demand_share"] = wh["demand_units_month"] / wh["demand_units_month"].sum()
    wh["occupied_sqft"] = wh["Capacity (sqft)"] * wh["Space Utilization %"] / 100
    return wh


def utilization_after_policy(wh, target_units_by_warehouse):
    """Utilization if occupied space scales with on-hand units (cube proxy: no SKU dimensions in the data)."""
    out = wh[["Capacity (sqft)", "Space Utilization %", "inventory_units"]].copy()
    out["target_units"] = target_units_by_warehouse
    out["utilization_after_pct"] = out["Space Utilization %"] * out["target_units"] / out["inventory_units"]
    return out
