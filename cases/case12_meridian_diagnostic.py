"""Case 12 (Toledah Foreign Enterprise Case #002): Meridian Fastener & Supply
Co. -- working-capital and distribution diagnostic, steps 1-3 of 8:

  1. reconcile every headline figure from the raw tables
  2. baseline inventory / service / aged stock by warehouse and ABC class
  3. test management's $1.38M lost-sales estimate against transaction data

NOTE ON DATA: synthetic engagement; Meridian is fictional. The workbook in
cases/data/ is the client data room as supplied.
"""

from pathlib import Path

import pandas as pd

DATA_ROOM = Path(__file__).parent / "data" / "meridian_data_room.xlsx"

BASELINE = {
    "revenue": 52_400_000,
    "cogs": 35_841_624,
    "inventory": 11_199_900,
    "freight": 3_050_000,
    "expedite": 465_000,
    "writeoffs": 310_000,
    "fill_rate": 0.918,
    "otif": 0.951,
    "lost_sales_claim": 1_380_000,
}
IMPLIED_WRITEOFF_RATE = 0.154  # of aged (>180d) inventory value, per the Reconciliation_Log


def load_data_room(path=DATA_ROOM):
    x = pd.read_excel(path, sheet_name=None)
    sku = x["SKU_Master"].set_axis(
        ["SKU", "category", "demand", "cost", "price", "gross_margin", "supplier",
         "lead_time", "cv", "usage_value", "abc"], axis=1)
    inventory = x["Inventory_Snapshot"].set_axis(["warehouse", "SKU", "on_hand", "aged_pct"], axis=1)
    history = x["Demand_History"].set_axis(["month", "warehouse", "SKU", "units", "stockout"], axis=1)
    suppliers = x["Supplier_Master"].set_axis(["supplier", "name", "otif", "lead_time", "spend"], axis=1)
    customers = x["Customer_Master"].set_axis(
        ["customer", "name", "revenue", "gross_margin", "cadence", "strategic"], axis=1)
    freight = x["Freight_Transactions"].set_axis(
        ["month", "origin", "destination", "mode", "weight_lb", "cost", "expedite"], axis=1)
    return {"sku": sku, "inventory": inventory, "history": history, "suppliers": suppliers,
            "customers": customers, "freight": freight}


# ---- Steps 4-5: policy and rebalancing assumptions (all stated, all sensitivity-tested) ----
REVIEW_DAYS = 30          # replenishment review period; no ordering-cadence data in the data room
HOLD_RATE = 0.20          # annual carrying cost (capital + storage + shrink); same default as nuri/procurement.py
# Service floors by class. The strategic-account floor ("no stockouts on top 12% of accounts")
# cannot be mapped to SKUs -- the customer master has no order lines -- so A-class SKUs stand in
# for it. A-class already runs 0 stockout lines in 18 months.
SERVICE_FLOORS = {"A": 0.995, "B": 0.97}
NETWORK_FILL_FLOOR = 0.90  # contractual floor; current fill is 91.8%
TRANSFER_PCT = 0.03        # transfer freight as % of unit cost: SKU weights are NOT in the data room
SENSITIVITY = {
    "sigma_source": ["history", "master"],
    "review_days": [14, 30, 45],
    "target_fill": [NETWORK_FILL_FLOOR, BASELINE["fill_rate"]],
}
