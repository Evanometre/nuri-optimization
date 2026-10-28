"""Target inventory policy by SKU x warehouse, and the working capital it implies.

Periodic-review (every R days) base-stock logic. For each SKU/warehouse series:

    protection interval   P  = L + R           (supplier lead time + review period)
    sigma over interval      = sqrt(P*sd_d^2 + d^2*sd_L^2)
    safety stock             = z * sigma
    target average inventory = safety stock + d*R/2   (cycle stock averages half a cycle)

z is chosen to minimise holding cost subject to a network service constraint,
not set by rule of thumb. Line fill is approximated by the cycle service level
Phi(z), because the data room defines fill rate as 1 - stockout lines / lines.

Minimise  sum_i c_i*h*sigma_i*z_i   s.t.  mean_i Phi(z_i) >= target,  z_i >= floor(class)

KKT gives  phi(z_i) = c_i*h*sigma_i / lambda  -> closed form for z_i, with lambda
found by bisection. Cheap, low-variability lines end up with high service and
expensive, volatile ones with low service -- the economics, not the class label,
decide where availability is spent. Class floors (A, B) override that.
"""

import numpy as np
import pandas as pd
from scipy import stats

Z_MAX = 3.5


def series_stats(sku, inventory, history, supplier_otif, sigma_source="history",
                 lead_time_cv_from_otif=True, supplier_lead_time=None):
    """One row per (warehouse, SKU): daily demand, sigma of daily demand, lead
    time, lead-time sd, current on-hand and aged units.

    supplier_lead_time: optional {supplier: days}; overrides the SKU-master lead time
                  (the two sources disagree in the Meridian data room).
    sigma_source: 'history' = observed monthly sd per series;
                  'master'  = SKU-master demand CV applied to the series mean
                  (more conservative in this data room: 0.50 vs ~0.29 observed).
    """
    g = history.groupby(["warehouse", "SKU"]).units.agg(mean_m="mean", sd_m="std").reset_index()
    # outer join: stock sitting where there is no demand history at all is pure excess
    g = g.merge(inventory[["warehouse", "SKU"]], on=["warehouse", "SKU"], how="outer")
    g[["mean_m", "sd_m"]] = g[["mean_m", "sd_m"]].fillna(0)
    g = g.merge(sku[["SKU", "cost", "price", "abc", "cv", "lead_time", "supplier"]], on="SKU")
    g = g.merge(supplier_otif.rename("otif"), left_on="supplier", right_index=True, how="left")

    g["d"] = g.mean_m * 12 / 365
    sd_m = g.sd_m.fillna(0) if sigma_source == "history" else g.cv * g.mean_m
    g["sd_d"] = sd_m / np.sqrt(365 / 12)
    g["L"] = g.lead_time
    if supplier_lead_time is not None:  # alternative source: supplier-master average lead time
        g["L"] = g.supplier.map(supplier_lead_time)
    # late-delivery share scales lead-time spread: an explicit, stated assumption
    g["sd_L"] = g.L * (1 - g.otif.fillna(1)) if lead_time_cv_from_otif else 0.0

    g = g.merge(inventory, on=["warehouse", "SKU"], how="left")
    g["on_hand"] = g.on_hand.fillna(0)
    g["aged_pct"] = g.aged_pct.fillna(0)
    return g


def sigma_over_interval(s, review_days):
    P = s.L + review_days
    return np.sqrt(P * s.sd_d ** 2 + (s.d * s.sd_L) ** 2)


def solve_service_levels(s, review_days, hold_rate, target_fill, floors):
    """Return z per row. floors: {'A': 0.995, 'B': 0.97, ...} class service floors."""
    sigma = sigma_over_interval(s, review_days).values
    cost = s.cost.values
    z_floor = np.array([stats.norm.ppf(floors.get(c, 0.5)) for c in s.abc])
    marginal = cost * hold_rate * sigma  # holding cost per unit z, per year
    n = len(s)

    def z_of(lam):
        with np.errstate(divide="ignore", invalid="ignore"):
            arg = marginal * np.sqrt(2 * np.pi) / lam
            z = np.where(arg < 1, np.sqrt(np.maximum(-2 * np.log(np.where(arg > 0, arg, 1)), 0)), 0.0)
        z = np.where(marginal == 0, Z_MAX, z)
        return np.clip(np.maximum(z, z_floor), 0, Z_MAX)

    def fill(lam):
        return stats.norm.cdf(z_of(lam)).mean()

    if fill(1e-12) >= target_fill:        # class floors alone already meet the target
        return z_of(1e-12), 0.0
    lo, hi = 1e-12, marginal[marginal > 0].max() * 1e6
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        if fill(mid) >= target_fill:
            hi = mid
        else:
            lo = mid
    return z_of(hi), hi


def apply_policy(s, review_days, hold_rate, target_fill, floors):
    z, lam = solve_service_levels(s, review_days, hold_rate, target_fill, floors)
    out = s.copy()
    out["z"] = z
    out["service_level"] = stats.norm.cdf(z)
    out["safety_stock"] = z * sigma_over_interval(out, review_days)
    out["cycle_stock"] = out.d * review_days / 2
    out["target_units"] = out.safety_stock + out.cycle_stock
    out["excess_units"] = (out.on_hand - out.target_units).clip(lower=0)
    out["need_units"] = (out.target_units - out.on_hand).clip(lower=0)
    out["aged_units"] = out.on_hand * out.aged_pct
    out["aged_excess_units"] = np.minimum(out.excess_units, out.aged_units)
    return out, lam


def summarize(policy):
    c = policy.cost
    return {
        "current_inventory": float((policy.on_hand * c).sum()),
        "target_inventory": float((policy.target_units * c).sum()),
        "excess_value": float((policy.excess_units * c).sum()),
        "need_value": float((policy.need_units * c).sum()),
        "aged_excess_value": float((policy.aged_excess_units * c).sum()),
        "net_release": float(((policy.on_hand - policy.target_units) * c).sum()),
        "network_fill": float(policy.service_level.mean()),
    }


def release_schedule(policy, horizon_months=12):
    """Cumulative inventory released ($) by month if replenishment of every
    excess series stops until it falls to target. Each series burns off at its
    own demand rate; series with no demand never release without liquidation
    (reported as 'stuck')."""
    c = policy.cost.values
    ex = policy.excess_units.values
    d = policy.d.values * 365 / 12  # units / month
    out = []
    for m in range(1, horizon_months + 1):
        out.append(float((np.minimum(ex, d * m) * c).sum()))
    stuck = float((ex[d <= 0] * c[d <= 0]).sum())
    return out, stuck
