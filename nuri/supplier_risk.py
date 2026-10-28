"""Supplier concentration, performance and disruption exposure.

Works from what a data room normally holds -- spend, OTIF, lead time, and which
SKUs each supplier feeds -- not from competing price quotes, which are the
input a cost-minimising allocation (nuri/procurement.py) needs and which are
not available here. So this module answers a different question: what is each
supplier's reliability *costing* in working capital, and what happens to
service if it slips?
"""

import numpy as np
import pandas as pd

from nuri.inventory_policy import apply_policy


def concentration(supplier_spend):
    share = supplier_spend / supplier_spend.sum()
    ranked = share.sort_values(ascending=False)
    return {
        "top3_share": float(ranked.head(3).sum()),
        "top5_share": float(ranked.head(5).sum()),
        "hhi": float((share ** 2).sum() * 10_000),
        "n_suppliers": int(len(share)),
    }


def otif_working_capital_cost(series, review_days, hold_rate, target_fill, floors):
    """Safety-stock dollars attributable to each supplier's lateness: policy
    run with the supplier's real OTIF vs run as if lead times were reliable."""
    real, _ = apply_policy(series, review_days, hold_rate, target_fill, floors)
    perfect, _ = apply_policy(series.assign(sd_L=0.0), review_days, hold_rate, target_fill, floors)
    real["ss_val"] = real.safety_stock * real.cost
    perfect["ss_val"] = perfect.safety_stock * perfect.cost
    return (real.groupby("supplier").ss_val.sum() - perfect.groupby("supplier").ss_val.sum()).rename("otif_driven_safety_stock")


def otif_improvement_release(series, supplier_otif, review_days, hold_rate, target_fill, floors, otif_target=0.95):
    """Extra inventory released if every supplier below otif_target reached it,
    versus the policy at today's OTIF. Same service level in both runs."""
    better = series.copy()
    new_otif = better.otif.clip(lower=otif_target)
    better["sd_L"] = better.L * (1 - new_otif)
    a, _ = apply_policy(series, review_days, hold_rate, target_fill, floors)
    b, _ = apply_policy(better, review_days, hold_rate, target_fill, floors)
    return float((a.target_units * a.cost).sum() - (b.target_units * b.cost).sum())


def disruption_exposure(policy, supplier, slip_days):
    """Annualised-rate sales at risk in one disruption of `slip_days` for one
    supplier, under (a) today's on-hand and (b) the lean target inventory.

    A series is covered while stock lasts; once it runs out the remaining slip
    days are lost sales at that series' daily demand and price. Stock is taken
    at its average level -- half a cycle plus safety stock for the target, the
    snapshot for today -- so this is expected exposure, not a worst case.
    """
    p = policy[policy.supplier == supplier]
    out = {}
    for label, units in (("current", p.on_hand), ("target", p.target_units)):
        cover_days = np.where(p.d > 0, units / p.d.where(p.d > 0, 1), np.inf)
        short_days = np.clip(slip_days - cover_days, 0, None)
        out[label] = float((short_days * p.d * p.price).sum())
    out["a_class_sales_at_risk_target"] = float(
        (np.clip(slip_days - np.where(p.d > 0, p.target_units / p.d.where(p.d > 0, 1), np.inf), 0, None)
         * p.d * p.price)[p.abc == "A"].sum())
    return out


def supplier_table(sku, suppliers, policy):
    """One row per supplier: size, performance, and what it holds/costs."""
    inv = policy.assign(cur=policy.on_hand * policy.cost, tgt=policy.target_units * policy.cost,
                        exc=policy.excess_units * policy.cost,
                        a_skus=0).groupby("supplier").agg(
        inventory=("cur", "sum"), target=("tgt", "sum"), excess=("exc", "sum"))
    n = sku.groupby("supplier").agg(n_skus=("SKU", "count"),
                                    n_a=("abc", lambda x: (x == "A").sum()))
    t = suppliers.set_index("supplier")[["otif", "lead_time", "spend"]].join(inv).join(n)
    t["share"] = t.spend / t.spend.sum()
    return t.sort_values("spend", ascending=False)


def lead_time_conflict(sku, suppliers):
    """SKU-master lead time vs supplier-master lead time, aggregated per supplier."""
    m = sku.groupby("supplier").lead_time.mean()
    s = suppliers.set_index("supplier")["lead_time"]
    both = pd.concat([m.rename("sku_master"), s.rename("supplier_master")], axis=1).dropna()
    return {
        "correlation": float(both.corr().iloc[0, 1]),
        "mean_abs_gap_days": float((both.sku_master - both.supplier_master).abs().mean()),
    }


def dual_source_candidates(table, otif_max=0.85, min_share=0.02):
    """Suppliers where concentration and weak delivery coincide."""
    return table[(table.otif < otif_max) & (table.share >= min_share)].sort_values("spend", ascending=False)
