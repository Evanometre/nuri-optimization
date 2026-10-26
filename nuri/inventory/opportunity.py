"""Working-capital opportunity: waterfall, release timeline, scenarios, annual benefits."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nuri.inventory.policy import DAYS_PER_MONTH, order_up_to


@dataclass
class Scenario:
    name: str
    csl_by_class: dict
    review_days: int = 7
    lt_sd_factor: float = 1.0
    moq_buffer: float = 0.0  # inflate order-up-to levels for lot-size / case-pack constraints
    write_off_reduction: float = 0.5  # share of today's write-offs avoided
    expedite_premium_captured: float = 0.5  # share of expedite premium avoided


def targets_for(pairs, scenario):
    t = order_up_to(pairs, scenario.csl_by_class, scenario.review_days, scenario.lt_sd_factor)
    if scenario.moq_buffer:
        t["order_up_to"] *= 1 + scenario.moq_buffer
        t["safety_stock"] += (t["order_up_to"] - t["order_up_to"] / (1 + scenario.moq_buffer))
        t["target_units"] = t["safety_stock"] + t["cycle_stock"]
        t["target_value"] = t["target_units"] * t["Unit Cost ($)"]
        t["order_up_to_value"] = t["order_up_to"] * t["Unit Cost ($)"]
    return t


def waterfall(pairs, targets):
    """Split (current - target) inventory into four mutually exclusive buckets (sums exactly)."""
    on_hand = pairs["value"].to_numpy()
    target = targets["target_value"].to_numpy()
    upto = targets["order_up_to_value"].to_numpy()
    dead = (pairs["mean_month"].to_numpy() == 0) & (on_hand > 0)
    above_s = ~dead & (on_hand > upto)
    band = ~dead & ~above_s & (on_hand >= target)
    below = ~dead & ~above_s & (on_hand < target)
    out = pd.Series(
        {
            "current_inventory": on_hand.sum(),
            "dead_stock_no_sales": -on_hand[dead].sum(),
            "excess_above_order_up_to": -(on_hand[above_s] - target[above_s]).sum(),
            "within_policy_band_trim": -(on_hand[band] - target[band]).sum(),
            "build_to_target_deficits": (target[below] - on_hand[below]).sum(),
        }
    )
    out["target_inventory"] = out.sum()
    return out


def release_timeline(pairs, targets, months=(0, 1, 3, 6, 9, 12, 18, 24)):
    """Inventory after m months if buying stops on pairs above target and demand burns the excess.

    Pairs below target are assumed filled straight to target (open purchase
    orders are not in the data room). Pairs with no demand never burn down.
    """
    on_hand = pairs["On-hand Units"].to_numpy()
    cost = pairs["Unit Cost ($)"].to_numpy()
    target_units = targets["target_units"].to_numpy()
    mean = pairs["mean_month"].to_numpy()
    rows = []
    for m in months:
        units = np.maximum(target_units, on_hand - mean * m)
        rows.append((m, float((units * cost).sum())))
    out = pd.DataFrame(rows, columns=["month", "inventory"]).set_index("month")
    out["released_vs_today"] = pairs["value"].sum() - out["inventory"]
    return out


def excess_by_cover(pairs, targets):
    """Excess above order-up-to by months of cover it represents: how fast it clears."""
    units = np.maximum(pairs["On-hand Units"].to_numpy() - targets["order_up_to"].to_numpy(), 0)
    value = units * pairs["Unit Cost ($)"].to_numpy()
    months = np.where(pairs["mean_month"] > 0, units / np.where(pairs["mean_month"] > 0, pairs["mean_month"], 1), np.inf)
    bucket = pd.cut(months, [-1, 3, 6, 12, 24, np.inf], labels=["<=3m", "3-6m", "6-12m", "12-24m", ">24m / no demand"])
    return pd.Series(value).groupby(bucket, observed=False).sum()


def aged_excess_value(pairs, targets):
    """Value of aged (>180d) stock that also sits above the order-up-to level."""
    upto_value = targets["order_up_to_value"].to_numpy()
    excess = np.maximum(pairs["value"].to_numpy() - upto_value, 0)
    return float(np.minimum(pairs["aged_value"].to_numpy(), excess).sum())


def annual_benefits(scenario, release, baseline, expedite, lost_sales, holding_rate=0.15):
    """Recurring P&L benefits (per year) with every assumption named."""
    benefits = {
        "carrying_cost_saved": release * holding_rate,
        "write_offs_avoided": baseline["Inventory write-offs / obsolescence"] * scenario.write_off_reduction,
        "expedite_premium_avoided": expedite["annual_premium"] * scenario.expedite_premium_captured,
        "lost_sales_margin_recovered": lost_sales["shortfall_margin"] * 0.5,
    }
    benefits["total"] = sum(benefits.values())
    return benefits


def compare_scenarios(pairs, scenarios, baseline, expedite, lost_sales, holding_rate=0.15):
    rows = []
    current = pairs["value"].sum()
    for sc in scenarios:
        t = targets_for(pairs, sc)
        target = t["target_value"].sum()
        release = current - target
        ben = annual_benefits(sc, release, baseline, expedite, lost_sales, holding_rate)
        rows.append(
            {
                "scenario": sc.name,
                "target_inventory": target,
                "release": release,
                "release_pct": release / current,
                "inventory_days": target / baseline["COGS"] * 365,
                **{f"benefit_{k}": v for k, v in ben.items()},
            }
        )
    return pd.DataFrame(rows).set_index("scenario")


def service_frontier(pairs, simulate_fn, summarize_fn, grid, review_days=7):
    """Inventory vs simulated service for class-level service-level settings."""
    rows = []
    for label, csl in grid.items():
        t = order_up_to(pairs, csl, review_days)
        sim = simulate_fn(pairs, t["order_up_to"].to_numpy())
        by_class, total = summarize_fn(pairs, sim)
        rows.append(
            {
                "policy": label,
                "A_line_fill": by_class.loc["A", "line_fill"],
                "B_line_fill": by_class.loc["B", "line_fill"],
                "C_line_fill": by_class.loc["C", "line_fill"],
                "network_line_fill": total["line_fill"],
                "avg_inventory": total["avg_inventory_value"],
            }
        )
    return pd.DataFrame(rows).set_index("policy")


def sensitivity(pairs, base, variations, baseline):
    """One-at-a-time changes to the base scenario; returns release for each."""
    current = pairs["value"].sum()
    rows = []
    for label, kwargs in variations.items():
        sc = Scenario(**{**base.__dict__, **kwargs, "name": label})
        t = targets_for(pairs, sc)
        rows.append({"variation": label, "release": current - t["target_value"].sum()})
    return pd.DataFrame(rows).set_index("variation")
