"""Loader for the Meridian data room workbook.

Everything downstream works on a DataRoom of plain DataFrames, so the same
analysis runs on the real workbook or on a small synthetic fixture in tests.
"""

from dataclasses import dataclass

import pandas as pd

SHEETS = {
    "financial": "Financial_Baseline",
    "sku": "SKU_Master",
    "inventory": "Inventory_Snapshot",
    "demand": "Demand_History",
    "supplier": "Supplier_Master",
    "customer": "Customer_Master",
    "warehouse": "Warehouse_Master",
    "freight": "Freight_Transactions",
}

# The history covers 18 months; annual figures are scaled by 12/18.
HISTORY_MONTHS = 18


@dataclass
class DataRoom:
    financial: pd.DataFrame
    sku: pd.DataFrame
    inventory: pd.DataFrame
    demand: pd.DataFrame
    supplier: pd.DataFrame
    customer: pd.DataFrame
    warehouse: pd.DataFrame
    freight: pd.DataFrame

    def baseline(self):
        """Financial_Baseline as {metric: float}; '$52,400,000' / '31.6%' / '114.1 days' parsed."""
        out = {}
        for metric, value in zip(self.financial["Metric"], self.financial["Current Value"]):
            out[metric] = _parse_number(value)
        return out


def _parse_number(value):
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace("$", "").replace(",", "").replace("days", "").strip()
    if text.endswith("%"):
        return float(text[:-1]) / 100
    return float(text)


def load_data_room(path):
    """Read the workbook at `path` into a DataRoom."""
    sheets = pd.read_excel(path, sheet_name=list(SHEETS.values()))
    return DataRoom(**{key: sheets[name] for key, name in SHEETS.items()})
