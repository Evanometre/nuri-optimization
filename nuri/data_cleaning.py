"""Turn a raw ERP sales/inventory export into clean weekly demand, and report
exactly what was wrong with it.

Every fix is counted, because the count is itself a deliverable: it is the
evidence that the numbers going into a forecast are trustworthy, and the first
thing a client's operations lead wants to see.

Fixes, in order: SKU code normalisation -> exact-duplicate removal -> return
separation -> unit-of-measure conversion (documented factor, else inferred from
the price ratio) -> orphan-SKU removal -> project-order flagging.
"""

import re

import numpy as np
import pandas as pd

ALLOWED_BOX_FACTORS = np.array([5, 10, 12, 25, 50, 100])  # common pack sizes; inferred ones are listed for client confirmation


def normalise_code(s):
    """'fst00012', 'FST 00012 ', 'FST-00012' -> 'FST-00012'."""
    m = re.match(r"^([A-Z]+)[^A-Z0-9]*([0-9]+)$", str(s).upper().strip().replace(" ", ""))
    return f"{m.group(1)}-{m.group(2)}" if m else str(s).upper().strip()


def clean_sales(lines, sku_master, uom_table, start_date):
    report = {"raw_lines": len(lines)}
    df = lines.copy()
    df["sku"] = df.sku_raw.map(normalise_code)
    report["sku_codes_normalised"] = int((df.sku != df.sku_raw).sum())

    before = len(df)
    df = df.drop_duplicates(subset="line_id")
    report["duplicate_lines_removed"] = before - len(df)

    returns = df[df.qty < 0]
    report["return_lines_separated"] = len(returns)
    report["returned_units"] = int(-returns.qty.sum())
    df = df[df.qty > 0].copy()

    known = set(sku_master.sku)
    orphan = ~df.sku.isin(known)
    report["orphan_sku_lines_dropped"] = int(orphan.sum())
    df = df[~orphan]

    # unit of measure: documented factor first, otherwise infer from price vs list price
    factor = uom_table.set_index("sku").factor
    list_price = sku_master.set_index("sku").list_price
    df["factor"] = 1.0
    box = df.uom != "EA"
    df.loc[box, "factor"] = df.loc[box, "sku"].map(factor)
    undoc = box & df.factor.isna()
    ratio = df.loc[undoc, "unit_price"] / df.loc[undoc, "sku"].map(list_price)
    inferred = ALLOWED_BOX_FACTORS[np.abs(ratio.values[:, None] - ALLOWED_BOX_FACTORS[None, :]).argmin(axis=1)]
    df.loc[undoc, "factor"] = inferred
    report["non_each_lines"] = int(box.sum())
    report["uom_inferred_from_price"] = int(undoc.sum())
    df["units"] = df.qty * df.factor

    df["week"] = ((pd.to_datetime(df.date) - start_date).dt.days // 7).astype(int)
    return df, report


def weekly_matrix(df, codes, weeks):
    idx = pd.Series(np.arange(len(codes)), index=codes)
    m = np.zeros((len(codes), weeks))
    g = df.groupby(["sku", "week"]).units.sum()
    rows = idx.reindex(g.index.get_level_values(0)).values
    cols = g.index.get_level_values(1).values
    ok = (cols >= 0) & (cols < weeks) & ~np.isnan(rows)
    m[rows[ok].astype(int), cols[ok]] = g.values[ok]
    return m


def flag_project_orders(matrix, k=6.0, min_units=10):
    """A week is a one-off project order when it exceeds the SKU's median
    positive week by k MADs *and* by min_units. Flagged weeks are excluded from
    the demand level, not deleted -- they are real sales, just not repeatable."""
    flags = np.zeros_like(matrix, dtype=bool)
    for i in range(matrix.shape[0]):
        pos = matrix[i][matrix[i] > 0]
        if len(pos) < 6:
            continue
        med = np.median(pos)
        mad = np.median(np.abs(pos - med)) or 1.0
        flags[i] = (matrix[i] > med + k * 1.4826 * mad) & (matrix[i] > med + min_units)
    return flags


def clean_inventory(inv_now, sku_master):
    known = set(sku_master.sku)
    inv = inv_now.copy()
    report = {
        "negative_on_hand_rows": int((inv.on_hand < 0).sum()),
        "orphan_inventory_rows": int((~inv.sku.isin(known)).sum()),
    }
    inv = inv[inv.sku.isin(known)].copy()
    inv["on_hand"] = inv.on_hand.clip(lower=0)   # a negative count is a posting error; stock cannot be < 0
    return inv, report
