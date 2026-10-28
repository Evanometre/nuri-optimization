"""Service vs inventory curves: the new rules at several fill targets against the
company's own buying rule at several settings, on the same held-out 52 weeks."""
import sys

import numpy as np

from cases.case13_distributor import generate, START_DATE, WEEKS, TRAIN_WEEKS
from nuri.data_cleaning import clean_sales, weekly_matrix, flag_project_orders
from nuri.replay import simulate, BuyerRule, metrics
from nuri.reorder_policy import fit_model, ReorderPolicy

TARGETS = (0.90, 0.93, 0.95, 0.97, 0.98)
BASELINES = ((8, 14), (10, 17), (12, 20), (14, 24), (16, 28))   # (12, 20) is what the company does


def prepare(seed):
    ds = generate(seed)
    tr = ds.truth
    codes = tr["codes"]
    df, _ = clean_sales(ds.sales_lines, ds.sku_master, ds.uom_table, START_DATE)
    sales = weekly_matrix(df, codes, WEEKS)
    soldout = ds.inventory_weekly.pivot(index="sku", columns="week", values="on_hand_end").reindex(codes).values <= 0
    projects = flag_project_orders(sales)
    po = ds.po_history[ds.po_history.receipt_week < TRAIN_WEEKS].copy()
    po["lead"] = po.receipt_week - po.order_week
    sup_codes = sorted(ds.sku_master.supplier.unique())
    sup_idx = ds.sku_master.supplier.map({s: i for i, s in enumerate(sup_codes)}).values
    pools = {i: po[po.supplier == s].lead.values for i, s in enumerate(sup_codes)}
    lead, orders = tr["lead_weeks"][:, :TRAIN_WEEKS], tr["orders"][:, :TRAIN_WEEKS]
    pipe = np.zeros((len(codes), int(lead.max()) + 1))
    for w in range(TRAIN_WEEKS):
        a = w + lead[:, w] - TRAIN_WEEKS
        m = a >= 0
        np.add.at(pipe, (np.flatnonzero(m), a[m]), orders[m, w])
    return ds, sales, soldout, projects, sup_idx, pools, pipe


def frontier(seed):
    ds, sales, soldout, projects, sup_idx, pools, pipe = prepare(seed)
    tr = ds.truth
    cost, price, pack = tr["cost"], tr["price"], tr["pack"]
    D, L = tr["demand"][:, TRAIN_WEEKS:], tr["lead_weeks"][:, TRAIN_WEEKS:]
    on0 = tr["end_onhand"][:, TRAIN_WEEKS - 1]
    prior, prior_so = sales[:, :TRAIN_WEEKS], soldout[:, :TRAIN_WEEKS]
    seen = prior.sum(axis=1) > 0
    rows = {"new": [], "current": []}
    for tf in TARGETS:
        model, Q = fit_model(sales, soldout, projects, ds.sku_master.category.values, cost, price, pack,
                             sup_idx, pools, TRAIN_WEEKS, tf)
        r = simulate(D, on0, pipe, L, ReorderPolicy(model, Q, projects), prior, prior_so)
        rows["new"].append((tf, metrics(r, D, cost, price, seen)))
    for rop, up in BASELINES:
        r = simulate(D, on0, pipe, L, BuyerRule(rop, up, pack), prior, prior_so)
        rows["current"].append(((rop, up), metrics(r, D, cost, price, seen)))
    return rows


def inventory_at_fill(points, fill):
    """Linear interpolation of avg inventory at a given revenue fill along a curve."""
    pts = sorted((m["revenue_fill"], m["avg_inventory"]) for _, m in points)
    xs, ys = zip(*pts)
    return float(np.interp(fill, xs, ys)) if xs[0] <= fill <= xs[-1] else float("nan")


if __name__ == "__main__":
    for seed in [int(a) for a in sys.argv[1:]] or [13]:
        rows = frontier(seed)
        print(f"\n=== seed {seed}: SKUs with sales history, held-out 52 weeks ===")
        print(f"  {'policy':22}{'rev fill':>10}{'unit fill':>11}{'avg inv $K':>12}{'lost rev $K':>13}{'purchases $K':>14}{'stockout SKU-wks':>18}")
        for name, key in (("new rules, target ", "new"), ("current-style rule ", "current")):
            for lab, m in rows[key]:
                lab_s = f"{lab:.0%}" if key == "new" else f"{lab[0]}/{lab[1]} wks"
                print(f"  {name + lab_s:22}{m['revenue_fill']:>10.1%}{m['unit_fill']:>11.1%}{m['avg_inventory']/1e3:>12,.0f}"
                      f"{m['lost_revenue']/1e3:>13,.0f}{m['purchases']/1e3:>14,.0f}{m['stockout_sku_weeks']:>18,}")
        new97 = dict(rows["new"])[0.97]
        need = inventory_at_fill(rows["current"], new97["revenue_fill"])
        cur = dict(rows["current"])[(12, 20)]
        print(f"  -> new rules @97%: {new97['revenue_fill']:.1%} revenue fill on ${new97['avg_inventory']/1e3:,.0f}K. "
              f"The current-style rule needs ~${need/1e3:,.0f}K to match that service "
              f"({1 - new97['avg_inventory']/need:.0%} more than the new rules), and today's 12/20 setting "
              f"holds ${cur['avg_inventory']/1e3:,.0f}K for {cur['revenue_fill']:.1%}.")
