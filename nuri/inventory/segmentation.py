"""ABC / XYZ segmentation and inventory-health flags per SKU."""

import numpy as np
import pandas as pd

from nuri.inventory.demand import build_demand_matrices

XYZ_X_MAX = 0.25  # monthly demand CV <= 0.25: stable
XYZ_Y_MAX = 0.50  # <= 0.50: variable; above: erratic


def recompute_abc(sku, a_cut=0.80, b_cut=0.95):
    """ABC by cumulative annual usage value (80% / 95%), to verify the master's labels."""
    ordered = sku.sort_values("Annual Usage Value ($)", ascending=False)
    cum_before = ordered["Annual Usage Value ($)"].cumsum() - ordered["Annual Usage Value ($)"]
    share_before = cum_before / ordered["Annual Usage Value ($)"].sum()
    label = np.where(share_before < a_cut, "A", np.where(share_before < b_cut, "B", "C"))
    return pd.Series(label, index=ordered["SKU"]).reindex(sku["SKU"])


def sku_segments(dr):
    """SKU-level table: ABC (master), recomputed ABC, XYZ from observed monthly CV."""
    units, _ = build_demand_matrices(dr)
    monthly = units.groupby(level="SKU").sum()
    cv = monthly.std(axis=1, ddof=1) / monthly.mean(axis=1)
    seg = dr.sku.set_index("SKU")[["Category", "ABC Class", "Annual Usage Value ($)"]].copy()
    seg["abc_recomputed"] = recompute_abc(dr.sku).to_numpy()
    seg["observed_cv"] = cv.reindex(seg.index)
    seg["XYZ"] = np.where(
        seg["observed_cv"] <= XYZ_X_MAX, "X", np.where(seg["observed_cv"] <= XYZ_Y_MAX, "Y", "Z")
    )
    seg["segment"] = seg["ABC Class"] + seg["XYZ"]
    return seg


def segment_matrix(seg, pairs):
    """3x3 ABC/XYZ table of SKU counts and inventory value."""
    value = pairs.groupby("SKU")["value"].sum()
    seg = seg.join(value)
    counts = seg.pivot_table(index="ABC Class", columns="XYZ", values="Category", aggfunc="count", fill_value=0)
    dollars = seg.pivot_table(index="ABC Class", columns="XYZ", values="value", aggfunc="sum", fill_value=0.0)
    return counts, dollars


def health_flags(pairs, target, excess_multiple=1.0):
    """Per-pair flags: dead (stock, no sales), excess (above order-up-to), aged, stockout-prone."""
    frame = pairs.merge(
        target[["Warehouse", "SKU", "order_up_to", "target_units"]], on=["Warehouse", "SKU"]
    )
    frame["dead"] = (frame["On-hand Units"] > 0) & (frame["mean_month"] == 0)
    frame["excess_units"] = np.maximum(frame["On-hand Units"] - excess_multiple * frame["order_up_to"], 0)
    frame["excess_value"] = frame["excess_units"] * frame["Unit Cost ($)"]
    frame["aged"] = frame["Aged >180d %"] >= 0.25
    frame["stockout_prone"] = frame["stockout_months"] / 18 >= 0.20
    return frame
