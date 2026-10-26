"""Inter-warehouse rebalancing as a transportation LP.

Per SKU, excess units at one warehouse can cover a shortfall at another
instead of (a) buying new stock while the excess sits, or (b) leaving the
shortfall open. A unit moved from i to j:

  gain  = c * h * min(excess_i / d_i / 365, 1)   holding cost avoided: the time the
          excess would otherwise take to burn off at its own site (capped at 1 year)
  cost  = c * transfer_pct * lane_index_ij       freight to move it
  cash  = c                                      purchase avoided at j (one-time)

Maximise sum (gain - cost) * x  s.t.  sum_j x_ij <= excess_i,  sum_i x_ij <= need_j.
Only moves with positive net gain are made. SKU weights are not in the data
room, so transfer cost is a % of unit cost scaled by each lane's observed $/lb
-- the single weakest input here, flagged in the case file.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


def lane_cost_index(freight):
    """$/lb on each inter-warehouse lane relative to the all-lane average.
    Same-site 'transfers' (origin == destination) are excluded."""
    t = freight[(freight["mode"] == "Inter-warehouse Transfer") & (freight.origin != freight.destination)]
    lane = t.groupby(["origin", "destination"]).apply(lambda x: x.cost.sum() / x.weight_lb.sum(), include_groups=False)
    return (lane / lane.mean()).to_dict()


def solve_rebalancing(policy, lane_index, hold_rate, transfer_pct):
    p = policy[(policy.excess_units > 1e-9) | (policy.need_units > 1e-9)]
    skus = p.SKU.unique()
    rows = p.set_index(["SKU", "warehouse"])
    whs = sorted(policy.warehouse.unique())

    var, supply_idx, need_idx, net, meta = [], {}, {}, [], []
    for sku in skus:
        for i in whs:
            if (sku, i) not in rows.index or rows.loc[(sku, i), "excess_units"] <= 1e-9:
                continue
            ri = rows.loc[(sku, i)]
            burn_years = min(ri.excess_units / ri.d / 365, 1.0) if ri.d > 0 else 1.0
            for j in whs:
                if i == j or (sku, j) not in rows.index or rows.loc[(sku, j), "need_units"] <= 1e-9:
                    continue
                gain = ri.cost * hold_rate * burn_years
                cost = ri.cost * transfer_pct * lane_index.get((i, j), 1.0)
                if gain - cost <= 0:
                    continue
                var.append((sku, i, j))
                net.append(gain - cost)
                meta.append((gain, cost, ri.cost))
    if not var:
        return pd.DataFrame(columns=["SKU", "from", "to", "units", "gain", "freight", "purchase_avoided"])

    ex = {k: rows.loc[k, "excess_units"] for k in rows.index}
    nd = {k: rows.loc[k, "need_units"] for k in rows.index}
    s_keys = sorted({(s, i) for s, i, _ in var})
    n_keys = sorted({(s, j) for s, _, j in var})
    s_pos = {k: r for r, k in enumerate(s_keys)}
    n_pos = {k: r + len(s_keys) for r, k in enumerate(n_keys)}
    r_idx, c_idx = [], []
    for c, (s, i, j) in enumerate(var):
        r_idx += [s_pos[(s, i)], n_pos[(s, j)]]
        c_idx += [c, c]
    A = coo_matrix((np.ones(len(r_idx)), (r_idx, c_idx)), shape=(len(s_keys) + len(n_keys), len(var))).tocsr()
    b = [ex[k] for k in s_keys] + [nd[k] for k in n_keys]
    res = linprog(-np.array(net), A_ub=A, b_ub=b, bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(res.message)

    out = pd.DataFrame(var, columns=["SKU", "from", "to"])
    out["units"] = res.x
    out["gain"] = res.x * np.array([m[0] for m in meta])
    out["freight"] = res.x * np.array([m[1] for m in meta])
    out["purchase_avoided"] = res.x * np.array([m[2] for m in meta])
    return out[out.units > 1e-6].reset_index(drop=True)
