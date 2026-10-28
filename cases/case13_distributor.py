"""Case 13: reorder logic and inventory forecasting for a mid-size distributor.

FICTIONAL company, SYNTHETIC data -- built to demonstrate the method, not to
claim a result. "Northline Industrial Supply": ~$12M revenue, ~40 staff, one
warehouse, ~2,500 SKUs across fasteners, plumbing, electrical, safety, tools,
adhesives and hardware, bought from ~35 suppliers.

Unlike case 12, the raw exports are deliberately dirty, the way a real ERP
export is: SKU codes typed several ways, units of measure mixed (each vs box),
duplicate lines, returns, one-off project orders, missing and optimistic
supplier lead times, negative on-hand, SKUs that are not in the master. The
generator keeps the TRUE demand and TRUE stock hidden in `truth`, so every
cleaning and forecasting step can be scored against it -- something a real
engagement can never do, and the reason this is a fair test of the tools.

How the history is made: true weekly demand is generated, then the company's
actual buying habit (BuyerRule: one flat rule for every SKU, ignoring supplier
lead time) is *run* against it for 130 weeks. So sales are censored by the
company's own stockouts, exactly as in real life.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nuri.replay import simulate, BuyerRule

N_SKU = 2500
WEEKS = 130
TRAIN_WEEKS = 78           # the replay window is the last 52 weeks, held out from every fit
N_SUPPLIERS = 35
TARGET_ANNUAL_REVENUE = 12_000_000
START_DATE = pd.Timestamp("2024-01-01")

# The company's current buying habit (tuned so the baseline looks like a real firm:
# ~3.5 turns, low-90s fill). Flat for every SKU.
BUYER_ROP_WEEKS = 12
BUYER_UPTO_WEEKS = 20

CATEGORIES = {   # prefix, share of SKUs, seasonal amplitude, phase (radians), median unit cost
    "Fasteners":  ("FST", 0.24, 0.15, 0.5, 4.0),
    "Plumbing":   ("PLB", 0.17, 0.25, 2.0, 18.0),
    "Electrical": ("ELC", 0.17, 0.12, 1.0, 14.0),
    "Safety/PPE": ("SAF", 0.12, 0.10, 3.5, 12.0),
    "Tools":      ("TLS", 0.14, 0.20, 0.0, 45.0),
    "Adhesives":  ("ADH", 0.08, 0.30, 5.0, 9.0),
    "Hardware":   ("HDW", 0.08, 0.18, 4.0, 11.0),
}


@dataclass
class Dataset:
    sales_lines: pd.DataFrame
    sku_master: pd.DataFrame
    uom_table: pd.DataFrame
    inventory_now: pd.DataFrame
    inventory_weekly: pd.DataFrame
    po_history: pd.DataFrame
    truth: dict = field(default_factory=dict)


def _week_start(w):
    return START_DATE + pd.Timedelta(days=7 * int(w))


def generate(seed=13):
    rng = np.random.default_rng(seed)
    N = N_SKU

    # ---- SKU master ----
    cats = rng.choice(list(CATEGORIES), size=N, p=[v[1] for v in CATEGORIES.values()])
    prefix = np.array([CATEGORIES[c][0] for c in cats])
    cost = np.array([CATEGORIES[c][4] for c in cats]) * rng.lognormal(0, 1.0, N)
    cost = np.clip(cost, 0.25, 900).round(2)
    margin = rng.uniform(0.18, 0.45, N)
    price = (cost / (1 - margin)).round(2)
    pack = np.where(cost < 3, rng.choice([25, 50, 100], N), np.where(cost < 15, rng.choice([5, 10, 25], N),
                    np.where(cost < 60, rng.choice([1, 2, 5], N), 1))).astype(float)
    supplier = rng.integers(0, N_SUPPLIERS, N)
    codes = np.array([f"{p}-{i:05d}" for i, p in enumerate(prefix)])

    # ---- true demand: intermittent / lumpy, seasonal, mild growth, some dying SKUs ----
    pareto = rng.lognormal(0, 1.5, N)
    order_rate = pareto                           # relative orders per week
    lot = rng.choice([1, 2, 5, 10, 25], N, p=[0.45, 0.2, 0.15, 0.12, 0.08]).astype(float)
    weeks = np.arange(WEEKS)
    amp = np.array([CATEGORIES[c][2] for c in cats]); ph = np.array([CATEGORIES[c][3] for c in cats])
    season = 1 + amp[:, None] * np.sin(2 * np.pi * weeks[None, :] / 52 + ph[:, None])
    trend = (1 + 0.08 * weeks / 52)[None, :]
    alive = np.ones((N, WEEKS))
    dying = rng.random(N) < 0.10
    death_wk = rng.integers(20, 70, N)
    alive[dying[:, None] & (weeks[None, :] >= death_wk[:, None])] = 0.02

    def build_demand(scale, r):
        lam = scale * order_rate[:, None] * season * trend * alive
        qty = np.round(r.poisson(lam) * lot[:, None] * r.uniform(0.7, 1.4, (N, WEEKS)))
        # one-off project orders: rare, a few weeks' worth in one go, not repeatable
        project = r.random((N, WEEKS)) < 0.003
        mean_i = qty.mean(1, keepdims=True)
        size = np.round(r.uniform(4, 10, (N, WEEKS)) * mean_i + lot[:, None])
        return qty + project * size, project

    # calibrate to the target revenue on a throwaway stream, so the main stream stays deterministic
    scale = 0.01
    for _ in range(3):
        d_cal, _ = build_demand(scale, np.random.default_rng(seed + 999))
        scale *= TARGET_ANNUAL_REVENUE / (d_cal[:, 52:104] * price[:, None]).sum()
    demand, project = build_demand(scale, rng)

    # ---- supplier lead times (true), and what the ERP says ----
    sup_lt = rng.uniform(1.5, 8.5, N_SUPPLIERS)
    sup_cv = rng.uniform(0.15, 0.45, N_SUPPLIERS)
    sku_lt_mean = sup_lt[supplier]
    shape = 1 / sup_cv[supplier] ** 2
    lead_weeks = np.maximum(1, np.round(rng.gamma(shape[:, None], (sku_lt_mean / shape)[:, None], (N, WEEKS)))).astype(int)
    quoted = np.round(sku_lt_mean * 0.75).astype(float)   # suppliers quote optimistically
    quoted[rng.random(N) < 0.15] = np.nan                  # and 15% are blank

    # ---- actual operations: the company's own buying habit, run for 130 weeks ----
    mean_wk = demand[:, :13].mean(1)
    onhand0 = np.ceil(np.maximum(mean_wk * 9, 0) / pack) * pack
    buyer = BuyerRule(BUYER_ROP_WEEKS, BUYER_UPTO_WEEKS, pack)
    actual = simulate(demand, onhand0, np.zeros((N, 0)), lead_weeks, buyer)

    # ---- raw exports ----
    sku_master = pd.DataFrame({
        "sku": codes, "category": cats, "supplier": [f"SUP-{s:03d}" for s in supplier],
        "unit_cost": cost, "list_price": price, "pack_size": pack.astype(int), "quoted_lead_weeks": quoted,
    })
    box_skus = np.flatnonzero((cost < 15) & (rng.random(N) < 0.35))
    factor = rng.choice([10, 12, 25, 50, 100], len(box_skus))
    uom_full = pd.DataFrame({"sku": codes[box_skus], "uom": "BX", "factor": factor})
    uom_table = uom_full.sample(frac=0.85, random_state=seed).reset_index(drop=True)  # 15% undocumented

    lines = _build_sales_lines(rng, actual["sales"], codes, price, box_skus, factor)

    on_now = actual["end_onhand"][:, -1] + actual["arrivals"][:, WEEKS:WEEKS + 1].sum(1) * 0
    inv_now = pd.DataFrame({"sku": codes, "on_hand": on_now})
    inv_now.loc[rng.random(N) < 0.008, "on_hand"] *= -1                     # negative on-hand glitches
    orphans = pd.DataFrame({"sku": [f"ZZ-{i:05d}" for i in range(25)], "on_hand": rng.integers(1, 50, 25)})
    inv_now = pd.concat([inv_now, orphans], ignore_index=True)

    wk = np.repeat(np.arange(WEEKS), N)
    inventory_weekly = pd.DataFrame({"week": wk, "sku": np.tile(codes, WEEKS),
                                     "on_hand_end": actual["end_onhand"].T.ravel()})

    po = _build_po_history(actual["orders"], lead_weeks, codes, supplier)

    truth = {"demand": demand, "sales": actual["sales"], "end_onhand": actual["end_onhand"],
             "orders": actual["orders"], "arrivals": actual["arrivals"], "lead_weeks": lead_weeks,
             "codes": codes, "cost": cost, "price": price, "pack": pack, "supplier": supplier,
             "category": cats, "true_lead_mean": sku_lt_mean, "project_mask": project, "dying": dying,
             "box_factor": dict(zip(codes[box_skus], factor)), "onhand0": onhand0}
    return Dataset(lines, sku_master, uom_table, inv_now, inventory_weekly, po, truth)


def _build_sales_lines(rng, sales, codes, price, box_skus, factor):
    N, T = sales.shape
    si, wi = np.nonzero(sales > 0)
    qty = sales[si, wi].astype(int)
    n_lines = np.minimum(rng.integers(1, 4, len(si)), qty)           # split a week's sales into 1-3 lines
    rep = np.repeat(np.arange(len(si)), n_lines)
    base = np.floor(qty / n_lines).astype(int)
    line_qty = base[rep]
    first = np.concatenate([[True], rep[1:] != rep[:-1]])
    line_qty = line_qty + np.where(first, qty - base * n_lines, 0)[...] if False else line_qty
    # put the remainder on the first line of each group
    rem = qty - base * n_lines
    line_qty[first] += rem[rep[first]]
    sku_i, week_i = si[rep], wi[rep]
    day = rng.integers(0, 7, len(rep))
    df = pd.DataFrame({
        "line_id": np.arange(len(rep)),
        "date": [START_DATE + pd.Timedelta(days=int(7 * w + d)) for w, d in zip(week_i, day)],
        "sku_raw": codes[sku_i], "qty": line_qty, "uom": "EA",
        "unit_price": (price[sku_i] * rng.uniform(0.95, 1.05, len(rep))).round(2),
    })
    df["_sku_i"] = sku_i

    # unit-of-measure: box SKUs sell in boxes when the quantity is a whole number of boxes
    f = pd.Series(factor, index=codes[box_skus])
    is_box = df.sku_raw.isin(f.index)
    fac = df.sku_raw.map(f)
    to_box = is_box & (df.qty % fac == 0) & (df.qty > 0) & (rng.random(len(df)) < 0.9)
    df.loc[to_box, "unit_price"] = (df.loc[to_box, "unit_price"] * fac[to_box]).round(2)
    df.loc[to_box, "qty"] = (df.loc[to_box, "qty"] // fac[to_box]).astype(int)
    df.loc[to_box, "uom"] = "BX"

    # SKU code typed several ways
    messy = rng.random(len(df)) < 0.07
    def variant(s, k):
        return [s.lower(), s.replace("-", ""), s + " ", s.replace("-", " ")][k]
    ks = rng.integers(0, 4, len(df))
    df.loc[messy, "sku_raw"] = [variant(s, k) for s, k in zip(df.loc[messy, "sku_raw"], ks[messy])]

    # returns (negative lines) and exact duplicates
    ret = df.sample(frac=0.015, random_state=1).copy()
    ret["line_id"] = np.arange(len(df), len(df) + len(ret)); ret["qty"] = -np.maximum(ret.qty // 2, 1)
    dup = df.sample(frac=0.01, random_state=2).copy()                  # same line_id: a genuine duplicate row
    out = pd.concat([df, ret, dup], ignore_index=True).sample(frac=1, random_state=3).reset_index(drop=True)
    return out


def _build_po_history(orders, lead_weeks, codes, supplier):
    si, wi = np.nonzero(orders > 0)
    return pd.DataFrame({
        "sku": codes[si], "supplier": [f"SUP-{s:03d}" for s in supplier[si]],
        "order_week": wi, "receipt_week": wi + lead_weeks[si, wi], "qty": orders[si, wi],
    })
