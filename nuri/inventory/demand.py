"""Per SKU-warehouse monthly demand matrices built from Demand_History."""

import numpy as np
import pandas as pd

from nuri.inventory.data import HISTORY_MONTHS


def build_demand_matrices(dr):
    """Return (units, stockout) DataFrames indexed by (Warehouse, SKU), columns = months 1..18.

    Months with no row are zero demand and not a stockout (6,066 such cells in
    the data room; the units still reconcile to the SKU master, so these are
    genuinely quiet months and not dropped rows).
    """
    months = list(range(1, HISTORY_MONTHS + 1))
    units = (
        dr.demand.pivot_table(
            index=["Warehouse", "SKU"], columns="Month", values="Observed Units Sold", aggfunc="sum"
        )
        .reindex(columns=months)
        .fillna(0.0)
    )
    stockout = (
        dr.demand.pivot_table(
            index=["Warehouse", "SKU"], columns="Month", values="Stockout Flag", aggfunc="max"
        )
        .reindex(columns=months)
        .fillna(0)
        .astype(int)
    )
    return units, stockout


def uncensored_units(units, stockout):
    """Raise sales in stockout months to that pair's non-stockout mean.

    Sales in a stockout month are censored (the shelf was empty), so the
    observed number understates demand. Pairs that stocked out in every
    observed month keep their observed values.
    """
    open_mean = (units.where(stockout == 0)).mean(axis=1)
    open_mean = open_mean.fillna(units.mean(axis=1))
    adjusted = units.copy()
    mask = stockout == 1
    floor = pd.DataFrame(
        np.repeat(open_mean.values[:, None], units.shape[1], axis=1), index=units.index, columns=units.columns
    )
    adjusted[mask] = np.maximum(units[mask], floor[mask])
    return adjusted


def demand_stats(units):
    """Monthly mean / sd per SKU-warehouse (sample sd, zero-filled months)."""
    return pd.DataFrame(
        {
            "mean_month": units.mean(axis=1),
            "sd_month": units.std(axis=1, ddof=1),
            "months_with_sales": (units > 0).sum(axis=1),
        }
    )
