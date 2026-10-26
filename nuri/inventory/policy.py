"""SKU x warehouse inventory policy: order-up-to levels, safety stock, targets.

Policy modelled: periodic review every `review_days`, order up to S where S is
the `csl` quantile of demand over the protection interval (lead time + review
period). Protection-interval demand is negative binomial (Poisson when demand
is under-dispersed), with variance from observed monthly demand plus supplier
lead-time variability. Unmet demand is lost (no backorders).

Lead-time variability: the supplier master gives OTIF but no lead-time spread,
so a late delivery is modelled as a Bernoulli event with probability
(1 - OTIF) and a size chosen so that the lead-time standard deviation is
L x (1 - OTIF). This is an assumption and is exposed as `lt_sd_factor`.
"""

import numpy as np
import pandas as pd
from scipy import stats

from nuri.inventory.demand import build_demand_matrices, demand_stats, uncensored_units

DAYS_PER_MONTH = 30.4


def build_pair_table(dr, censor_adjust=True, lt_source="sku"):
    """One row per SKU-warehouse pair seen in demand history or the inventory snapshot."""
    units, stockout = build_demand_matrices(dr)
    basis = uncensored_units(units, stockout) if censor_adjust else units
    stats_df = demand_stats(basis)
    stats_df["stockout_months"] = stockout.sum(axis=1)

    keys = units.index.union(pd.MultiIndex.from_frame(dr.inventory[["Warehouse", "SKU"]]))
    pairs = pd.DataFrame(index=keys).join(stats_df).reset_index()
    pairs[["mean_month", "sd_month", "months_with_sales", "stockout_months"]] = pairs[
        ["mean_month", "sd_month", "months_with_sales", "stockout_months"]
    ].fillna(0.0)
    pairs = (
        pairs.merge(dr.inventory, on=["Warehouse", "SKU"], how="left")
        .merge(dr.sku, on="SKU", how="left")
        .merge(
            dr.supplier[["Supplier ID", "OTIF", "Avg Lead Time Days"]], on="Supplier ID", how="left"
        )
    )
    pairs["On-hand Units"] = pairs["On-hand Units"].fillna(0.0)
    pairs["Aged >180d %"] = pairs["Aged >180d %"].fillna(0.0)
    pairs["lead_time"] = (
        pairs["Supplier Lead Time (days)"] if lt_source == "sku" else pairs["Avg Lead Time Days"]
    )
    pairs["value"] = pairs["On-hand Units"] * pairs["Unit Cost ($)"]
    pairs["aged_value"] = pairs["value"] * pairs["Aged >180d %"]
    return pairs


def protection_distribution(pairs, review_days=7, lt_sd_factor=1.0):
    """Mean and variance of demand over lead time + review period, per pair."""
    mu_day = pairs["mean_month"] / DAYS_PER_MONTH
    horizon = pairs["lead_time"] + review_days
    lt_sd = pairs["lead_time"] * (1 - pairs["OTIF"]) * lt_sd_factor
    mean = mu_day * horizon
    var = (horizon / DAYS_PER_MONTH) * pairs["sd_month"] ** 2 + mu_day**2 * lt_sd**2
    return mean, np.maximum(var, mean)  # never under-dispersed


def _quantile(csl, mean, var):
    mean = np.asarray(mean, float)
    var = np.asarray(var, float)
    csl = np.broadcast_to(np.asarray(csl, float), mean.shape)
    safe_mean = np.where(mean > 0, mean, 1.0)
    over = var > safe_mean * (1 + 1e-9)
    p = np.where(over, safe_mean / np.where(over, var, 1.0), 0.5)
    r = np.where(over, safe_mean * p / (1 - p), 1.0)
    nb = stats.nbinom.ppf(csl, r, p)
    po = stats.poisson.ppf(csl, safe_mean)
    return np.where(mean > 0, np.where(over, nb, po), 0.0)


def order_up_to(pairs, csl_by_class, review_days=7, lt_sd_factor=1.0):
    """Policy parameters per pair for class-level cycle service levels."""
    mean, var = protection_distribution(pairs, review_days, lt_sd_factor)
    csl = pairs["ABC Class"].map(csl_by_class).to_numpy(float)
    S = _quantile(csl, mean, var)
    out = pairs[["Warehouse", "SKU", "ABC Class", "Unit Cost ($)"]].copy()
    out["order_up_to"] = S
    out["safety_stock"] = np.maximum(S - mean, 0.0)
    out["cycle_stock"] = pairs["mean_month"] / DAYS_PER_MONTH * review_days / 2
    out["target_units"] = out["safety_stock"] + out["cycle_stock"]
    out["target_value"] = out["target_units"] * out["Unit Cost ($)"]
    out["order_up_to_value"] = out["order_up_to"] * out["Unit Cost ($)"]
    return out


def simulate(pairs, order_up_to_units, review_days=7, weeks=104, warmup=30, lt_sd_factor=1.0, seed=1):
    """Weekly Monte Carlo of the policy; returns per-pair service and average stock.

    line_flag_rate is the share of 18-month-style 'months' (52/12 weeks) with at
    least one week of unmet demand -- the closest analogue to the data room's
    monthly stockout flag. unit_fill is units served / units demanded.
    """
    rng = np.random.default_rng(seed)
    n = len(pairs)
    S = np.asarray(order_up_to_units, float)
    mean_w = pairs["mean_month"].to_numpy() * review_days / DAYS_PER_MONTH
    active = mean_w > 0
    mean_w = np.where(active, mean_w, 0.5)
    var_w = np.maximum(pairs["sd_month"].to_numpy() ** 2 * review_days / DAYS_PER_MONTH, mean_w * 1.0001)
    var_w = np.where(active, var_w, 1.0)
    p = mean_w / var_w
    r = mean_w * p / (1 - p)

    lead = pairs["lead_time"].to_numpy(float)
    q = np.clip((1 - pairs["OTIF"].to_numpy(float)) * lt_sd_factor, 1e-6, 0.99)
    late_size = lead * np.sqrt(q / (1 - q))  # Bernoulli(q) lateness with sd = lead x q
    total = weeks + warmup
    max_lt = int(np.ceil((lead + late_size).max() / review_days)) + 2
    arrivals = np.zeros((n, total + max_lt + 2))
    on_hand = S.copy()
    on_order = np.zeros(n)
    short = np.zeros((n, total))
    demand = np.zeros((n, total))
    stock_sum = np.zeros(n)
    rows = np.arange(n)

    for t in range(total):
        on_hand += arrivals[:, t]
        on_order -= arrivals[:, t]
        d = np.where(active, rng.negative_binomial(np.maximum(r, 1e-9), np.clip(p, 1e-9, 1 - 1e-9)), 0)
        sold = np.minimum(on_hand, d)
        on_hand -= sold
        short[:, t] = d - sold
        demand[:, t] = d
        if t >= warmup:
            stock_sum += on_hand
        order = np.maximum(S - (on_hand + on_order), 0)
        late = rng.random(n) < q
        weeks_to_arrive = np.maximum(np.ceil((lead - q * late_size + late_size * late) / review_days), 1).astype(int)
        np.add.at(arrivals, (rows, np.minimum(t + weeks_to_arrive, arrivals.shape[1] - 1)), order)
        on_order += order

    window = slice(warmup, total)
    month_of_week = np.floor(np.arange(weeks) / (52 / 12)).astype(int)
    flagged = short[:, window] > 0
    month_flags = np.stack(
        [flagged[:, month_of_week == m].any(axis=1) for m in range(month_of_week.max() + 1)], axis=1
    )
    served = demand[:, window].sum(axis=1)
    out = pd.DataFrame(
        {
            "line_flag_rate": month_flags.mean(axis=1),
            "unit_fill": 1 - short[:, window].sum(axis=1) / np.maximum(served, 1e-9),
            "avg_on_hand": stock_sum / weeks,
            "demand_units": served,
        },
        index=pairs.index,
    )
    out.loc[~active, ["line_flag_rate"]] = 0.0
    return out


def summarize_simulation(pairs, sim):
    """Class-level line fill (1 - flag rate), unit fill and average inventory value."""
    frame = pd.concat([pairs[["ABC Class", "Unit Cost ($)"]], sim], axis=1)
    frame["avg_value"] = frame["avg_on_hand"] * frame["Unit Cost ($)"]
    g = frame.groupby("ABC Class")
    by_class = pd.DataFrame(
        {
            "line_fill": 1 - g["line_flag_rate"].mean(),
            "unit_fill": g.apply(lambda x: np.average(x["unit_fill"], weights=x["demand_units"] + 1e-9)),
            "avg_inventory_value": g["avg_value"].sum(),
        }
    )
    total = {
        "line_fill": 1 - frame["line_flag_rate"].mean(),
        "avg_inventory_value": float(frame["avg_value"].sum()),
    }
    return by_class, total
