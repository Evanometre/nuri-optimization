import sys
import time

import numpy as np
import pandas as pd

from cases.case13_distributor import generate, START_DATE, WEEKS, TRAIN_WEEKS, BUYER_ROP_WEEKS, BUYER_UPTO_WEEKS
from nuri.data_cleaning import clean_sales, weekly_matrix, flag_project_orders, clean_inventory
from nuri.replay import simulate, BuyerRule, metrics
from nuri.reorder_policy import fit_model, ReorderPolicy

TARGET_FILL = 0.97


def run(seed, verbose=True):
    t0 = time.time()
    ds = generate(seed)
    tr = ds.truth
    codes, cost, price, pack = tr["codes"], tr["cost"], tr["price"], tr["pack"]

    # ---- 1. clean the raw export (no access to truth) ----
    df, rep = clean_sales(ds.sales_lines, ds.sku_master, ds.uom_table, START_DATE)
    sales = weekly_matrix(df, codes, WEEKS)
    soldout = ds.inventory_weekly.pivot(index="sku", columns="week", values="on_hand_end").reindex(codes).values <= 0
    projects = flag_project_orders(sales)
    inv, inv_rep = clean_inventory(ds.inventory_now, ds.sku_master)
    if verbose:
        print("=== 1. Data quality report (what was wrong with the export) ===")
        for k, v in {**rep, **inv_rep}.items():
            print(f"  {k}: {v:,}")
        print(f"  project-order weeks flagged: {int(projects.sum()):,}")
        print(f"  weekly sales recovered exactly vs hidden truth: {np.array_equal(sales, tr['sales'])}")

    # ---- 2. fit on weeks [0, 78) only ----
    po = ds.po_history[ds.po_history.receipt_week < TRAIN_WEEKS].copy()
    po["lead"] = po.receipt_week - po.order_week
    sup_codes = sorted(ds.sku_master.supplier.unique())
    sup_idx = ds.sku_master.supplier.map({s: i for i, s in enumerate(sup_codes)}).values
    pools = {i: po[po.supplier == s].lead.values if (po.supplier == s).any() else np.array([4]) for i, s in enumerate(sup_codes)}
    cats = ds.sku_master.category.values
    model, Q = fit_model(sales, soldout, projects, cats, cost, price, pack, sup_idx, pools,
                         TRAIN_WEEKS, TARGET_FILL)

    # ---- 3. replay the 26 held-out weeks under both policies, same demand, same lead-time draws ----
    D = tr["demand"][:, TRAIN_WEEKS:]
    L = tr["lead_weeks"][:, TRAIN_WEEKS:]
    on0 = tr["end_onhand"][:, TRAIN_WEEKS - 1]
    # orders already placed before week 78 are still in transit: rebuild from the order log
    orders, lead = tr["orders"][:, :TRAIN_WEEKS], tr["lead_weeks"][:, :TRAIN_WEEKS]
    pipe = np.zeros((len(codes), int(lead.max()) + 1))
    for w in range(TRAIN_WEEKS):
        arr = w + lead[:, w] - TRAIN_WEEKS
        m = arr >= 0
        np.add.at(pipe, (np.flatnonzero(m), arr[m]), orders[m, w])
    prior, prior_so = sales[:, :TRAIN_WEEKS], soldout[:, :TRAIN_WEEKS]

    base = simulate(D, on0, pipe, L, BuyerRule(BUYER_ROP_WEEKS, BUYER_UPTO_WEEKS, pack), prior, prior_so)
    ours = simulate(D, on0, pipe, L, ReorderPolicy(model, Q, projects), prior, prior_so)
    seen = sales[:, :TRAIN_WEEKS].sum(axis=1) > 0     # SKUs with any sales in the training window
    mb, mo = metrics(base, D, cost, price), metrics(ours, D, cost, price)
    sb, so_ = metrics(base, D, cost, price, seen), metrics(ours, D, cost, price, seen)
    tail_rev = float((D[~seen] * price[~seen, None]).sum() / (D * price[:, None]).sum())
    actual_matches = np.allclose(base["sales"], tr["sales"][:, TRAIN_WEEKS:])

    if verbose:
        print(f"\n=== 2. Replay of the last 26 weeks (fit on the first 78 only; {time.time() - t0:.0f}s) ===")
        print(f"  baseline replay reproduces what actually happened: {actual_matches}")
        print(f"  SKUs with sales history: {seen.sum():,} of {len(seen):,}; the {(~seen).sum():,} that never sold in the first 78 weeks "
              f"carry {tail_rev:.1%} of held-out revenue and cannot be forecast from history.")
        for title, a_, b_ in (("ALL SKUs", mb, mo), ("SKUs WITH SALES HISTORY (what a forecast can act on)", sb, so_)):
            print(f"\n  {title}\n  {'':24}{'current buying':>16}{'new rules':>14}")
            for k, lab, f in [("unit_fill", "unit fill", "{:.1%}"), ("revenue_fill", "revenue fill", "{:.1%}"),
                              ("lost_revenue", "lost revenue ($)", "{:,.0f}"), ("stockout_sku_weeks", "stockout SKU-weeks", "{:,}"),
                              ("avg_inventory", "avg inventory ($)", "{:,.0f}"), ("end_inventory", "ending inventory ($)", "{:,.0f}")]:
                print(f"  {lab:24}{f.format(a_[k]):>16}{f.format(b_[k]):>14}")
    return sb, so_, actual_matches


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or [13]
    out = [run(s, verbose=(len(seeds) == 1)) for s in seeds]
    if len(seeds) > 1:
        print("seed  fill(cur->new)  avg inv $M(cur->new)  lost rev $K(cur->new)")
        for s, (b, o, _) in zip(seeds, out):
            print(f"{s:>4}  {b['unit_fill']:.1%}->{o['unit_fill']:.1%}   {b['avg_inventory']/1e6:.2f}->{o['avg_inventory']/1e6:.2f}   {b['lost_revenue']/1e3:,.0f}->{o['lost_revenue']/1e3:,.0f}")
