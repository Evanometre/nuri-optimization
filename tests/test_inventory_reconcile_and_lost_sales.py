import pandas as pd
import pytest

from nuri.inventory.data import DataRoom, _parse_number
from nuri.inventory.demand import build_demand_matrices, uncensored_units
from nuri.inventory.lost_sales import estimate_lost_sales
from nuri.inventory.reconcile import data_quality_findings, inventory_value_by_row, run_reconciliation
from tests.inventory_fixtures import make_data_room


def test_parse_number_handles_dollars_percent_and_days():
    assert _parse_number("$52,400,000") == 52_400_000
    assert _parse_number("31.6%") == pytest.approx(0.316)
    assert _parse_number("114.1 days") == pytest.approx(114.1)
    assert _parse_number(7) == 7.0


def _with_baseline(dr, **overrides):
    inv = inventory_value_by_row(dr)["Value"].sum()
    revenue = (dr.sku["Annual Demand (units)"] * dr.sku["Sell Price ($)"]).sum()
    cogs = (dr.sku["Annual Demand (units)"] * dr.sku["Unit Cost ($)"]).sum()
    values = {
        "Annual Revenue": revenue, "Gross Margin": 1 - cogs / revenue, "COGS": cogs,
        "Inventory on hand": inv, "Inventory days": inv / cogs * 365,
        "Annual freight spend": dr.freight["Cost ($)"].sum() * 12 / 18,
        "Expedite / emergency freight": dr.freight.loc[dr.freight["Expedite Flag"] == 1, "Cost ($)"].sum() * 12 / 18,
        "Annual stockout-related lost sales (estimate)": 1.0,
        "Inventory write-offs / obsolescence": 0.154 * inventory_value_by_row(dr)["Aged Value"].sum(),
        "Order fill rate": 1 - dr.demand["Stockout Flag"].mean(),
        "OTIF (on-time in-full)": (dr.warehouse["Annual Outbound Orders"] * dr.warehouse["Current OTIF"]).sum()
        / dr.warehouse["Annual Outbound Orders"].sum(),
    }
    values.update(overrides)
    dr.financial = pd.DataFrame({"Metric": list(values), "Current Value": list(values.values()), "Source/Note": ""})
    dr.supplier["Annual Spend ($)"] = [cogs * 0.9, cogs * 0.1]
    dr.customer["Annual Revenue ($)"] = [revenue * 0.9, revenue * 0.1]
    return dr


def test_reconciliation_passes_when_baseline_is_consistent():
    dr = _with_baseline(make_data_room())
    result = run_reconciliation(dr)
    assert result["ok"].all()


def test_reconciliation_flags_a_baseline_that_does_not_tie():
    dr = make_data_room()
    right = _with_baseline(dr).baseline()["Inventory on hand"]
    dr = _with_baseline(make_data_room(), **{"Inventory on hand": right * 1.10})
    result = run_reconciliation(dr).set_index("check")
    assert not result.loc["Inventory value (at cost)", "ok"]
    assert result.loc["Inventory value (at cost)", "rel_diff"] == pytest.approx(-1 / 11, rel=1e-6)


def test_data_quality_finds_stock_with_no_sales_and_sales_with_no_stock():
    dr = make_data_room()
    dr.inventory = pd.concat(
        [dr.inventory, pd.DataFrame([{"Warehouse": "W3", "SKU": "S1", "On-hand Units": 5, "Aged >180d %": 0.0}])]
    )
    dr.inventory = dr.inventory[~((dr.inventory["Warehouse"] == "W2") & (dr.inventory["SKU"] == "S4"))]
    findings = data_quality_findings(dr)
    assert findings["stock_with_no_sales_18m"]["sku_warehouse_pairs"] == 1
    assert findings["stock_with_no_sales_18m"]["value"] == pytest.approx(50.0)  # 5 units x $10
    assert findings["sales_with_no_inventory_record"]["sku_warehouse_pairs"] == 1


def test_missing_months_are_zero_demand_not_dropped():
    dr = make_data_room()
    dr.demand = dr.demand[~((dr.demand["Warehouse"] == "W1") & (dr.demand["SKU"] == "S1") & (dr.demand["Month"] <= 3))]
    units, _ = build_demand_matrices(dr)
    assert units.loc[("W1", "S1"), 1] == 0
    assert units.loc[("W1", "S1")].sum() == 50 * 15


def test_uncensored_units_raises_stockout_months_to_open_mean():
    dr = make_data_room(monthly_units={("W1", "S1", 5): 10}, stockouts={("W1", "S1", 5): 1})
    units, stockout = build_demand_matrices(dr)
    adjusted = uncensored_units(units, stockout)
    assert adjusted.loc[("W1", "S1"), 5] == 50  # raised to the non-stockout mean
    assert adjusted.loc[("W1", "S1"), 4] == 50
    assert adjusted.loc[("W1", "S2"), 5] == 25  # untouched pairs unchanged


def test_lost_sales_central_estimate_and_ceiling():
    # One flagged month that sold 10 of a normal 50 units at $15 price, $10 cost.
    dr = make_data_room(monthly_units={("W1", "S1", 5): 10}, stockouts={("W1", "S1", 5): 1})
    result = estimate_lost_sales(dr, management_estimate=1_000)
    annualize = 12 / 18
    assert result["shortfall_revenue"] == pytest.approx(40 * 15 * annualize)
    assert result["shortfall_margin"] == pytest.approx(40 * 5 * annualize)
    assert result["ceiling_revenue"] == pytest.approx(50 * 15 * annualize)
    assert result["shortfall_revenue"] <= result["ceiling_revenue"]
    assert result["management_multiple_of_ceiling"] == pytest.approx(1_000 / (50 * 15 * annualize))


def test_lost_sales_is_zero_without_stockouts():
    result = estimate_lost_sales(make_data_room())
    assert result["shortfall_revenue"] == 0
    assert result["ceiling_revenue"] == 0
