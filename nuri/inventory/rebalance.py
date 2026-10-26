"""Cross-warehouse positioning: which SKUs to transfer, pool or leave alone.

Transfers: for each SKU, warehouses holding more than their order-up-to level
(donors) can fill the shortfall of warehouses below it (receivers). A move is
kept only if the carrying cost it saves on the donor's excess, over the months
the donor would take to sell that excess, exceeds one average lane shipment.
That treats every SKU-lane move as its own shipment, which overstates freight
cost -- consolidation onto shared pallets would make more moves worthwhile.

Pooling: stocking a SKU at one DC instead of four needs less safety stock
(demand variability pools), at the price of longer delivery to the other
regions. The table reports the safety-stock saving; the service trade-off is a
management call and is not priced here.
"""

import numpy as np
import pandas as pd

from nuri.inventory.policy import DAYS_PER_MONTH, _quantile


def lane_costs(freight):
    """Average cost per inter-warehouse shipment by (origin, destination), excluding same-site rows."""
    transfers = freight[freight["Mode"] == "Inter-warehouse Transfer"]
    transfers = transfers[transfers["Origin Warehouse"] != transfers["Destination"]]
    return transfers.groupby(["Origin Warehouse", "Destination"])["Cost ($)"].mean()


def same_site_transfers(freight, annualize=12 / 18):
    """Transfer records whose origin equals destination (not a real transfer)."""
    t = freight[freight["Mode"] == "Inter-warehouse Transfer"]
    same = t[t["Origin Warehouse"] == t["Destination"]]
    return {
        "records": int(len(same)),
        "share_of_transfers": float(len(same) / len(t)),
        "annual_cost": float(same["Cost ($)"].sum() * annualize),
    }


def plan_transfers(pairs, targets, lane_cost, holding_rate=0.15, max_burn_months=24):
    """Greedy per-SKU matching of donor excess to receiver need; returns moves DataFrame."""
    frame = pairs[["Warehouse", "SKU", "On-hand Units", "mean_month", "Unit Cost ($)"]].copy()
    frame["order_up_to"] = targets["order_up_to"].to_numpy()
    frame["supply"] = np.maximum(frame["On-hand Units"] - frame["order_up_to"], 0)
    frame["need"] = np.maximum(frame["order_up_to"] - frame["On-hand Units"], 0)
    frame["burn_months"] = np.where(
        frame["mean_month"] > 0,
        np.minimum(frame["supply"] / frame["mean_month"].where(frame["mean_month"] > 0, 1), max_burn_months),
        max_burn_months,
    )

    moves = []
    for sku, g in frame[(frame["supply"] > 0) | (frame["need"] > 0)].groupby("SKU"):
        donors = g[g["supply"] > 0].sort_values("supply", ascending=False)
        receivers = g[g["need"] > 0].sort_values("need", ascending=False)
        if donors.empty or receivers.empty:
            continue
        supply = donors.set_index("Warehouse")["supply"].to_dict()
        need = receivers.set_index("Warehouse")["need"].to_dict()
        burn = donors.set_index("Warehouse")["burn_months"].to_dict()
        cost_unit = float(g["Unit Cost ($)"].iloc[0])
        for recv in receivers["Warehouse"]:
            for don in donors["Warehouse"]:
                qty = min(supply[don], need[recv])
                if qty <= 0 or don == recv:
                    continue
                lane = lane_cost.get((don, recv), lane_cost.mean())
                value = qty * cost_unit
                benefit = value * holding_rate * burn[don] / 12
                if benefit >= lane:
                    moves.append(
                        {
                            "SKU": sku, "from": don, "to": recv, "units": qty, "value": value,
                            "freight_cost": float(lane), "carry_benefit": benefit,
                        }
                    )
                    supply[don] -= qty
                    need[recv] -= qty
    out = pd.DataFrame(moves, columns=["SKU", "from", "to", "units", "value", "freight_cost", "carry_benefit"])
    return out, frame


def warehouse_balance(frame):
    """Value of need vs donor excess by warehouse (positive net = holds more than it needs)."""
    f = frame.assign(
        need_value=frame["need"] * frame["Unit Cost ($)"], supply_value=frame["supply"] * frame["Unit Cost ($)"]
    )
    g = f.groupby("Warehouse")[["need_value", "supply_value"]].sum()
    g["net_excess"] = g["supply_value"] - g["need_value"]
    return g


def pooling_table(dr, pairs, csl_by_class, review_days=7, lt_sd_factor=1.0):
    """Safety stock if each SKU were held at a single DC vs today's four, per SKU."""
    from nuri.inventory.demand import build_demand_matrices, uncensored_units

    units, stockout = build_demand_matrices(dr)
    adj = uncensored_units(units, stockout)
    monthly = adj.groupby(level="SKU").sum()
    sku = dr.sku.set_index("SKU").join(dr.supplier.set_index("Supplier ID")[["OTIF"]], on="Supplier ID")
    sku = sku.reindex(monthly.index)

    mean_m = monthly.mean(axis=1)
    sd_m = monthly.std(axis=1, ddof=1)
    lead = sku["Supplier Lead Time (days)"]
    horizon = lead + review_days
    mu_day = mean_m / DAYS_PER_MONTH
    lt_sd = lead * (1 - sku["OTIF"]) * lt_sd_factor
    mean_p = mu_day * horizon
    var_p = np.maximum((horizon / DAYS_PER_MONTH) * sd_m**2 + mu_day**2 * lt_sd**2, mean_p)
    csl = sku["ABC Class"].map(csl_by_class).to_numpy(float)
    pooled_ss = np.maximum(_quantile(csl, mean_p.to_numpy(), var_p.to_numpy()) - mean_p.to_numpy(), 0)

    local = pairs.copy()
    from nuri.inventory.policy import order_up_to

    t = order_up_to(local, csl_by_class, review_days, lt_sd_factor)
    local_ss = t.groupby("SKU")["safety_stock"].sum().reindex(monthly.index)
    out = pd.DataFrame(
        {
            "abc": sku["ABC Class"],
            "mean_month_total": mean_m,
            "local_ss_units": local_ss,
            "pooled_ss_units": pooled_ss,
            "unit_cost": sku["Unit Cost ($)"],
        }
    )
    out["saving_units"] = (out["local_ss_units"] - out["pooled_ss_units"]).clip(lower=0)
    out["saving_value"] = out["saving_units"] * out["unit_cost"]
    out["saving_pct"] = out["saving_units"] / out["local_ss_units"].replace(0, np.nan)
    out["per_pair_units_month"] = out["mean_month_total"] / 4
    return out


def positioning_class(pool):
    """Centralize / duplicate / rebalance label per SKU.

    duplicate: A items always stay at every DC (strategic-account protection).
    centralize: B/C items with low per-DC demand where pooling cuts safety stock by >= 30%.
    rebalance: everything else is managed by the transfer rules.
    """
    label = np.where(
        pool["abc"] == "A",
        "duplicate",
        np.where((pool["saving_pct"] >= 0.30) & (pool["per_pair_units_month"] < 10), "centralize", "rebalance"),
    )
    return pd.Series(label, index=pool.index, name="positioning")
