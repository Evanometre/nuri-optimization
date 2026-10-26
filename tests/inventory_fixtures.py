"""A tiny, fully deterministic data room for the inventory modules' tests.

Two warehouses (W1, W2), four SKUs, two suppliers. Demand is constant per
month unless a test overrides it, so expected values can be worked by hand.
"""

import pandas as pd

from nuri.inventory.data import DataRoom

WAREHOUSES = ["W1", "W2"]


def make_data_room(monthly_units=None, stockouts=None, on_hand=None):
    skus = pd.DataFrame(
        {
            "SKU": ["S1", "S2", "S3", "S4"],
            "Category": ["Fasteners", "Fasteners", "PPE", "PPE"],
            "Annual Demand (units)": [1200, 600, 120, 24],
            "Unit Cost ($)": [10.0, 5.0, 2.0, 1.0],
            "Sell Price ($)": [15.0, 8.0, 3.0, 2.0],
            "Gross Margin": [0.33, 0.37, 0.33, 0.5],
            "Supplier ID": ["SUP-1", "SUP-1", "SUP-2", "SUP-2"],
            "Supplier Lead Time (days)": [30, 30, 30, 30],
            "Demand CV": [0.3, 0.3, 0.3, 0.3],
            "Annual Usage Value ($)": [12000.0, 3000.0, 240.0, 24.0],
            "ABC Class": ["A", "B", "C", "C"],
        }
    )
    # Per-warehouse monthly demand = half the annual rate (12 months).
    rate = {"S1": 50, "S2": 25, "S3": 5, "S4": 1}
    monthly_units = monthly_units or {}
    stockouts = stockouts or {}
    rows = []
    for wh in WAREHOUSES:
        for sku, base in rate.items():
            for month in range(1, 19):
                units = monthly_units.get((wh, sku, month), base)
                flag = stockouts.get((wh, sku, month), 0)
                rows.append((month, wh, sku, units, flag))
    demand = pd.DataFrame(rows, columns=["Month", "Warehouse", "SKU", "Observed Units Sold", "Stockout Flag"])

    default_on_hand = {"S1": 200, "S2": 100, "S3": 30, "S4": 10}
    on_hand = on_hand or {}
    inv_rows = [
        (wh, sku, on_hand.get((wh, sku), default_on_hand[sku]), 0.1)
        for wh in WAREHOUSES
        for sku in rate
    ]
    inventory = pd.DataFrame(inv_rows, columns=["Warehouse", "SKU", "On-hand Units", "Aged >180d %"])

    supplier = pd.DataFrame(
        {
            "Supplier ID": ["SUP-1", "SUP-2"],
            "Supplier Name": ["One", "Two"],
            "OTIF": [0.9, 0.95],
            "Avg Lead Time Days": [30.0, 30.0],
            "Annual Spend ($)": [15000.0, 264.0],
        }
    )
    customer = pd.DataFrame(
        {
            "Customer ID": ["C1", "C2"],
            "Customer Name": ["c1", "c2"],
            "Annual Revenue ($)": [30000.0, 1000.0],
            "Gross Margin": [0.3, 0.3],
            "Order Cadence": ["Weekly", "Monthly"],
            "Strategic Account": ["Yes", "No"],
        }
    )
    warehouse = pd.DataFrame(
        {
            "Warehouse": WAREHOUSES,
            "Annual Outbound Orders": [600, 400],
            "Avg Pick-Pack Cost ($/order)": [4.0, 5.0],
            "Current OTIF": [0.95, 0.9],
            "Space Utilization %": [70, 80],
            "Capacity (sqft)": [1000, 500],
        }
    )
    freight = pd.DataFrame(
        {
            "Month": [1, 2, 3, 4, 5, 6],
            "Origin Warehouse": ["W1", "W1", "W2", "W1", "W2", "W1"],
            "Destination": ["Customer", "Customer", "Customer", "W2", "W1", "W1"],
            "Mode": [
                "Ground - LTL", "Expedited Air", "Ground - Parcel",
                "Inter-warehouse Transfer", "Inter-warehouse Transfer", "Inter-warehouse Transfer",
            ],
            "Weight (lb)": [10, 10, 10, 10, 10, 10],
            "Cost ($)": [100.0, 250.0, 100.0, 120.0, 140.0, 50.0],
            "Expedite Flag": [0, 1, 0, 0, 0, 0],
        }
    )
    financial = pd.DataFrame(columns=["Metric", "Current Value", "Source/Note"])
    return DataRoom(
        financial=financial, sku=skus, inventory=inventory, demand=demand,
        supplier=supplier, customer=customer, warehouse=warehouse, freight=freight,
    )
