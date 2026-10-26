import pandas as pd
import pytest

from cases.case12_meridian_diagnostic import load_data_room, BASELINE, IMPLIED_WRITEOFF_RATE
from nuri.inventory import (
    reconcile_all, segment_skus, baseline_by_warehouse, baseline_by_class, lost_sales_estimate,
)


@pytest.fixture(scope="module")
def d():
    return load_data_room()


def test_every_data_room_tie_out_reproduces_from_raw_tables(d):
    checks = reconcile_all(d["sku"], d["inventory"], d["history"], d["suppliers"].spend,
                           d["customers"].revenue, d["freight"], BASELINE, IMPLIED_WRITEOFF_RATE)
    failed = [c.name for c in checks if not c.ok]
    assert failed == []


def test_reconciliation_flags_a_tampered_baseline(d):
    bad = dict(BASELINE, inventory=BASELINE["inventory"] * 1.05)
    checks = reconcile_all(d["sku"], d["inventory"], d["history"], d["suppliers"].spend,
                           d["customers"].revenue, d["freight"], bad, IMPLIED_WRITEOFF_RATE)
    assert [c.name for c in checks if not c.ok] == ["Inventory value = sum(on-hand x cost)"]


def test_warehouse_baseline_sums_to_company_inventory_and_cover(d):
    wh = baseline_by_warehouse(d["sku"], d["inventory"], d["history"])
    assert wh.inventory_value.sum() == pytest.approx(BASELINE["inventory"], abs=150)
    company_days = wh.inventory_value.sum() / wh.annual_cogs.sum() * 365
    assert company_days == pytest.approx(114.1, abs=0.1)


def test_all_stockouts_sit_in_c_and_b_class_none_in_a(d):
    cls = baseline_by_class(d["sku"], d["inventory"], d["history"])
    assert cls.loc["A", "stockout_lines"] == 0
    assert cls.loc["C", "stockout_rate"] > 0.10


def test_management_lost_sales_claim_exceeds_even_the_theoretical_ceiling(d):
    est = lost_sales_estimate(d["sku"], d["history"])
    assert est["shortfall_sales"] < est["ceiling_sales"] < BASELINE["lost_sales_claim"] / 2


def test_lost_sales_ceiling_is_total_loss_of_expected_demand_on_a_toy_series():
    sku = pd.DataFrame({"SKU": ["S"], "price": [10.0], "cost": [6.0]})
    hist = pd.DataFrame({"month": [1, 2, 3, 4], "warehouse": ["W"] * 4, "SKU": ["S"] * 4,
                         "units": [10, 10, 4, 10], "stockout": [0, 0, 1, 0]})
    est = lost_sales_estimate(sku, hist)
    # 4 months -> annualize x3; expected 10, observed 4, shortfall 6 units at $10
    assert est["shortfall_sales"] == pytest.approx(6 * 10 * 3)
    assert est["ceiling_sales"] == pytest.approx(10 * 10 * 3)
    assert est["shortfall_gross_profit"] == pytest.approx(6 * 4 * 3)


def test_segmentation_covers_every_sku_with_nine_cells(d):
    seg = segment_skus(d["sku"])
    assert len(seg) == 1400 and seg.segment.nunique() == 9
