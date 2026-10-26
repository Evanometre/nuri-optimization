"""Freight diagnostics: expedite premium, drivers and same-site 'transfers'."""

import pandas as pd

ANNUALIZE = 12 / 18
STANDARD_MODES = ["Ground - LTL", "Ground - Parcel"]


def expedite_analysis(freight):
    """How much of expedite spend is avoidable premium vs baseline shipping cost.

    Cost does not vary with weight in this data (correlation ~0 in every
    mode), so the premium is taken as mean expedited cost minus mean standard
    customer-shipment cost per shipment. That premium is the upper bound on
    what avoiding an expedite saves; the rest of the bill would be paid anyway
    to ship the goods.
    """
    expedite = freight[freight["Expedite Flag"] == 1]
    standard = freight[freight["Mode"].isin(STANDARD_MODES)]
    premium_each = expedite["Cost ($)"].mean() - standard["Cost ($)"].mean()
    n_year = len(expedite) * ANNUALIZE
    by_origin = expedite.groupby("Origin Warehouse")["Cost ($)"].sum() * ANNUALIZE
    share_by_origin = (
        expedite.groupby("Origin Warehouse").size() / freight.groupby("Origin Warehouse").size()
    )
    return {
        "annual_expedite_spend": float(expedite["Cost ($)"].sum() * ANNUALIZE),
        "expedites_per_year": float(n_year),
        "mean_expedite_cost": float(expedite["Cost ($)"].mean()),
        "mean_standard_cost": float(standard["Cost ($)"].mean()),
        "premium_per_expedite": float(premium_each),
        "annual_premium": float(premium_each * n_year),
        "premium_share_of_spend": float(premium_each * n_year / (expedite["Cost ($)"].sum() * ANNUALIZE)),
        "expedite_share_of_shipments": float(len(expedite) / len(freight)),
        "spend_by_origin": by_origin,
        "expedite_rate_by_origin": share_by_origin,
        "monthly_expedites": expedite.groupby("Month").size(),
        "all_expedites_to_customers": bool((expedite["Destination"] == "Customer").all()),
    }
