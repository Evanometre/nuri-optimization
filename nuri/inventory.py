"""Inventory and service diagnostics: reconciliation tie-outs, baseline
metrics, ABC/XYZ segmentation, and lost-sales estimation from stockout flags.

Everything here takes plain pandas DataFrames with the column names used in
nuri/inventory.py's docstrings below, so the same functions run on any
distributor's extracts, not just the Meridian data room.

Expected frames:
  sku:       SKU, cost, price, demand (annual units), cv, abc, category, supplier
  inventory: warehouse, SKU, on_hand, aged_pct
  history:   month, warehouse, SKU, units, stockout (0/1)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Check:
    name: str
    computed: float
    expected: float
    tolerance: float  # absolute

    @property
    def ok(self):
        return abs(self.computed - self.expected) <= self.tolerance


def reconcile_all(sku, inventory, history, supplier_spend, customer_revenue, freight, baseline,
                  aged_writeoff_rate):
    """Re-derive every headline figure from the raw tables and compare it with
    the stated baseline. Never takes a summary figure at face value."""
    months = history["month"].nunique()
    revenue = (sku.demand * sku.price).sum()
    cogs = (sku.demand * sku.cost).sum()
    inv = inventory.merge(sku[["SKU", "cost"]], on="SKU")
    inv_value = (inv.on_hand * inv.cost).sum()
    aged_value = (inv.on_hand * inv.cost * inv.aged_pct).sum()
    annualize = 12 / months

    per_sku_gap = (
        history.groupby("SKU").units.sum().reindex(sku.SKU).values - sku.demand.values * months / 12
    )

    return [
        Check("Revenue = sum(demand x price)", revenue, baseline["revenue"], baseline["revenue"] * 1e-5),
        Check("COGS = sum(demand x cost)", cogs, baseline["cogs"], baseline["cogs"] * 1e-5),
        Check("Supplier spend = COGS", supplier_spend.sum(), baseline["cogs"], 1),
        Check("Customer revenue = revenue", customer_revenue.sum(), baseline["revenue"], 1),
        Check("Inventory value = sum(on-hand x cost)", inv_value, baseline["inventory"], 150),
        Check("Demand history units tie per SKU (max abs gap, rounding of x1.5)", float(np.abs(per_sku_gap).max()), 0, 0.5),
        Check("Freight total x 12/18", freight["cost"].sum() * annualize, baseline["freight"], 1),
        Check("Expedite freight x 12/18",
              freight.loc[freight.expedite == 1, "cost"].sum() * annualize, baseline["expedite"], 1),
        Check("Fill rate = 1 - stockout lines / lines", 1 - history.stockout.mean(), baseline["fill_rate"], 0.0005),
        Check("Write-offs = aged value x derived rate", aged_value * aged_writeoff_rate, baseline["writeoffs"],
              baseline["writeoffs"] * 0.005),
    ]


def segment_skus(sku, x_max=0.35, y_max=0.60):
    """ABC (given) x XYZ (by demand CV) segmentation."""
    out = sku.copy()
    out["xyz"] = np.where(out.cv <= x_max, "X", np.where(out.cv <= y_max, "Y", "Z"))
    out["segment"] = out.abc + out.xyz
    return out


def baseline_by_warehouse(sku, inventory, history, warehouse_otif=None):
    """Inventory value, cover days (inventory / annual COGS flowing through
    that warehouse), aged value and line-level stockout rate by warehouse."""
    inv = inventory.merge(sku[["SKU", "cost", "price"]], on="SKU")
    inv["value"] = inv.on_hand * inv.cost
    inv["aged_value"] = inv.value * inv.aged_pct

    h = history.merge(sku[["SKU", "cost"]], on="SKU")
    months = history["month"].nunique()
    cogs = (h.units * h.cost).groupby(h.warehouse).sum() * 12 / months

    out = pd.DataFrame({
        "inventory_value": inv.groupby("warehouse").value.sum(),
        "aged_value": inv.groupby("warehouse").aged_value.sum(),
        "annual_cogs": cogs,
        "stockout_rate": history.groupby("warehouse").stockout.mean(),
    })
    out["cover_days"] = out.inventory_value / out.annual_cogs * 365
    out["aged_share"] = out.aged_value / out.inventory_value
    return out


def baseline_by_class(sku, inventory, history):
    """Same view cut by ABC class -- where service is actually failing and
    where the aged stock actually sits."""
    inv = inventory.merge(sku[["SKU", "cost", "abc"]], on="SKU")
    inv["value"] = inv.on_hand * inv.cost
    inv["aged_value"] = inv.value * inv.aged_pct
    h = history.merge(sku[["SKU", "abc"]], on="SKU")
    out = pd.DataFrame({
        "inventory_value": inv.groupby("abc").value.sum(),
        "aged_value": inv.groupby("abc").aged_value.sum(),
        "stockout_rate": h.groupby("abc").stockout.mean(),
        "stockout_lines": h.groupby("abc").stockout.sum(),
    })
    out["aged_share"] = out.aged_value / out.inventory_value
    return out


def lost_sales_estimate(sku, history, margin_by_sku=True):
    """Annualized lost sales implied by stockout-flagged lines.

    For each (warehouse, SKU) series, the 'expected' demand in a stockout
    month is the mean of that series' non-stockout months. Three views:

      shortfall: max(expected - observed, 0) -- sales actually missed, if
                 observed sales in the flagged month are partial
      ceiling:   expected demand in full, as if every flagged line sold
                 nothing -- an upper bound no real outcome can exceed
      observed:  revenue that DID ship on flagged lines (for context)

    A stockout line does not mean zero sales: the flag marks that the line
    ran out at some point, and units were still sold.
    """
    months = history["month"].nunique()
    a = 12 / months
    h = history.merge(sku[["SKU", "price", "cost"]], on="SKU")
    base = (h[h.stockout == 0].groupby(["warehouse", "SKU"]).units.mean().rename("expected"))
    h = h.join(base, on=["warehouse", "SKU"])
    h["expected"] = h.expected.fillna(h.groupby(["warehouse", "SKU"]).units.transform("mean"))
    so = h[h.stockout == 1].copy()
    so["short_units"] = (so.expected - so.units).clip(lower=0)
    so["margin"] = so.price - so.cost

    return {
        "stockout_lines": int(len(so)),
        "shortfall_sales": float((so.short_units * so.price).sum() * a),
        "shortfall_gross_profit": float((so.short_units * so.margin).sum() * a),
        "ceiling_sales": float((so.expected * so.price).sum() * a),
        "ceiling_gross_profit": float((so.expected * so.margin).sum() * a),
        "observed_sales_on_flagged_lines": float((so.units * so.price).sum() * a),
    }
