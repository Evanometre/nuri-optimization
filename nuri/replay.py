"""Weekly inventory replay: run a reorder policy against a demand history.

The test a buyer actually cares about -- "would your rule have caused
stockouts last year?" -- cannot be answered by an analytic model. This
replays a policy week by week against the same demand and the same
lead-time draws, so two policies are compared on identical conditions.

Conventions (kept simple on purpose, and stated to the client):
  * one review per week, at the start of the week, after receipts arrive
  * an order placed in week t with lead time L arrives at the start of week t+L (L >= 1)
  * demand that cannot be filled from stock on hand is lost, not backordered
"""

import numpy as np


def simulate(demand, onhand0, pipeline0, lead_weeks, policy, prior_sales=None, prior_soldout=None):
    """demand: [N,T] true weekly demand. onhand0: [N] stock at start of week 0.
    pipeline0: [N,P] units arriving at the start of weeks 0..P-1 from earlier orders.
    lead_weeks: [N,T] integer lead time (>=1) an order placed in week t will take.
    policy(t, onhand, position, hist, soldout) -> [N] order quantity; hist = sales observed so far,
    soldout = [N,t] True where the SKU ended that week with nothing on hand (sales understate demand).
    """
    N, T = demand.shape
    max_l = int(lead_weeks.max())
    arrivals = np.zeros((N, T + max_l + 1))
    if pipeline0 is not None and pipeline0.size:
        arrivals[:, : pipeline0.shape[1]] += pipeline0

    onhand = onhand0.astype(float).copy()
    sales = np.zeros((N, T))
    end_onhand = np.zeros((N, T))
    orders = np.zeros((N, T))
    prior = prior_sales if prior_sales is not None else np.zeros((N, 0))
    prior_so = prior_soldout if prior_soldout is not None else np.zeros((N, 0), dtype=bool)
    rows = np.arange(N)

    for t in range(T):
        onhand += arrivals[:, t]
        position = onhand + arrivals[:, t + 1:].sum(axis=1)
        hist = np.hstack([prior, sales[:, :t]])
        soldout = np.hstack([prior_so, end_onhand[:, :t] <= 0])
        q = np.maximum(policy(t, onhand, position, hist, soldout), 0.0)
        orders[:, t] = q
        arrivals[rows, t + lead_weeks[:, t]] += q
        s = np.minimum(demand[:, t], onhand)
        onhand -= s
        sales[:, t] = s
        end_onhand[:, t] = onhand

    return {"sales": sales, "end_onhand": end_onhand, "orders": orders, "arrivals": arrivals}


class BuyerRule:
    """The 'current behaviour' baseline: trailing-13-week average demand, reorder
    when stock position falls to `rop_weeks` of it, order up to `upto_weeks` of
    it, in whole pack sizes. One flat rule for every SKU and every supplier --
    it ignores lead time, which is exactly how a busy buyer works by habit."""

    def __init__(self, rop_weeks, upto_weeks, pack):
        self.rop, self.upto, self.pack = rop_weeks, upto_weeks, pack

    def __call__(self, t, onhand, position, hist, soldout=None):
        m = hist[:, -13:].mean(axis=1) if hist.shape[1] else np.zeros_like(onhand)
        need = self.upto * m - position
        q = np.ceil(np.maximum(need, 0) / self.pack) * self.pack
        return np.where((position <= self.rop * m) & (m > 0), q, 0.0)


def metrics(result, demand, cost, price, mask=None):
    """Fill, lost revenue and inventory $ over a replay window. `mask` restricts to a SKU subset."""
    sales, oh, orders = result["sales"], result["end_onhand"], result["orders"]
    if mask is not None:
        sales, oh, orders, demand, cost, price = sales[mask], oh[mask], orders[mask], demand[mask], cost[mask], price[mask]
    short = demand - sales
    return {
        "unit_fill": float(sales.sum() / demand.sum()),
        "revenue_fill": float((sales * price[:, None]).sum() / (demand * price[:, None]).sum()),
        "lost_revenue": float((short * price[:, None]).sum()),
        "stockout_sku_weeks": int((short > 1e-9).sum()),
        "avg_inventory": float((oh * cost[:, None]).sum(axis=0).mean()),
        "end_inventory": float((oh[:, -1] * cost).sum()),
        "purchases": float((orders * cost[:, None]).sum()),
    }
