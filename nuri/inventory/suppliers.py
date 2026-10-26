"""Supplier concentration, performance and working-capital exposure."""

import numpy as np
import pandas as pd

from nuri.inventory.policy import DAYS_PER_MONTH, order_up_to


def concentration(dr):
    spend = dr.supplier.set_index("Supplier ID")["Annual Spend ($)"].sort_values(ascending=False)
    share = spend / spend.sum()
    return {
        "top3_share": float(share.head(3).sum()),
        "top5_share": float(share.head(5).sum()),
        "top10_share": float(share.head(10).sum()),
        "hhi": float((share**2).sum() * 10_000),
        "suppliers_for_80pct": int((share.cumsum() < 0.80).sum() + 1),
        "suppliers_under_1pct": int((share < 0.01).sum()),
        "n_suppliers": int(len(share)),
    }


def supplier_table(dr, pairs, csl_by_class, review_days=7):
    """One row per supplier with spend, performance, SKU footprint and variability-driven stock."""
    base = order_up_to(pairs, csl_by_class, review_days)
    no_var = order_up_to(pairs, csl_by_class, review_days, lt_sd_factor=0.0)
    p = pairs[["Supplier ID", "value", "On-hand Units", "mean_month", "Unit Cost ($)", "Sell Price ($)"]].copy()
    p["ss_value"] = base["safety_stock"] * base["Unit Cost ($)"]
    p["lt_driven_ss_value"] = (base["safety_stock"] - no_var["safety_stock"]).clip(lower=0) * base["Unit Cost ($)"]
    p["a_value"] = np.where(pairs["ABC Class"] == "A", p["value"], 0.0)
    g = p.groupby("Supplier ID").agg(
        inventory_value=("value", "sum"),
        safety_stock_value=("ss_value", "sum"),
        lt_driven_ss_value=("lt_driven_ss_value", "sum"),
        a_class_inventory=("a_value", "sum"),
    )
    skus = dr.sku.groupby("Supplier ID").agg(n_skus=("SKU", "count"), a_skus=("ABC Class", lambda s: (s == "A").sum()))
    out = dr.supplier.set_index("Supplier ID").join(skus).join(g)
    out["spend_share"] = out["Annual Spend ($)"] / out["Annual Spend ($)"].sum()
    # Risk score: share of spend x unreliability x lead-time length (relative to the mean).
    out["risk_score"] = (
        out["spend_share"] * (1 - out["OTIF"]) * (out["Avg Lead Time Days"] / out["Avg Lead Time Days"].mean())
    )
    out["risk_rank"] = out["risk_score"].rank(ascending=False).astype(int)
    return out.sort_values("risk_score", ascending=False)


def otif_improvement_release(pairs, csl_by_class, supplier_ids, new_otif, review_days=7):
    """Extra inventory released if the given suppliers raise OTIF to `new_otif`."""
    improved = pairs.copy()
    mask = improved["Supplier ID"].isin(supplier_ids) & (improved["OTIF"] < new_otif)
    improved.loc[mask, "OTIF"] = new_otif
    before = order_up_to(pairs, csl_by_class, review_days)["target_value"].sum()
    after = order_up_to(improved, csl_by_class, review_days)["target_value"].sum()
    return float(before - after)


def disruption_exposure(pairs, supplier_id, outage_days=30, on_hand_units=None):
    """Revenue at risk if a supplier delivers nothing for `outage_days`.

    Demand over the outage beyond what is on hand is lost; stock on hand is
    assumed fully available (no allocation to other warehouses). Pass
    `on_hand_units` (aligned to `pairs`) to test a post-policy inventory
    position instead of today's.
    """
    mask = (pairs["Supplier ID"] == supplier_id).to_numpy()
    p = pairs[mask]
    on_hand = p["On-hand Units"] if on_hand_units is None else pd.Series(np.asarray(on_hand_units)[mask], index=p.index)
    demand = p["mean_month"] / DAYS_PER_MONTH * outage_days
    lost_units = (demand - on_hand).clip(lower=0)
    return {
        "lost_revenue": float((lost_units * p["Sell Price ($)"]).sum()),
        "skus_stocking_out": int((lost_units > 0).groupby(p["SKU"]).any().sum()),
        "days_of_cover": float(
            (on_hand * p["Unit Cost ($)"]).sum()
            / max((p["mean_month"] / DAYS_PER_MONTH * p["Unit Cost ($)"]).sum(), 1e-9)
        ),
    }
