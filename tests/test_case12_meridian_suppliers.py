import pandas as pd
import pytest

from cases.case12_meridian_diagnostic import (
    load_data_room, BASELINE, REVIEW_DAYS, HOLD_RATE, SERVICE_FLOORS, OTIF_TARGET,
)
from nuri.inventory_policy import series_stats, apply_policy
from nuri.supplier_risk import (
    concentration, disruption_exposure, otif_improvement_release, otif_working_capital_cost,
    lead_time_conflict, supplier_table, dual_source_candidates,
)


def test_concentration_on_a_known_split():
    c = concentration(pd.Series({"a": 50.0, "b": 30.0, "c": 20.0}))
    assert c["top3_share"] == pytest.approx(1.0)
    assert c["hhi"] == pytest.approx(0.25 * 1e4 + 0.09 * 1e4 + 0.04 * 1e4)


def test_disruption_exposure_counts_only_days_beyond_cover():
    pol = pd.DataFrame({"supplier": ["S", "S"], "abc": ["A", "C"], "d": [10.0, 2.0], "price": [5.0, 50.0],
                        "on_hand": [200.0, 4.0], "target_units": [100.0, 2.0]})
    e = disruption_exposure(pol, "S", 20)
    # today: A has 20d cover -> 0 short; C has 2d cover -> 18d short x 2/day x $50 = 1800
    assert e["current"] == pytest.approx(1800)
    # target: A has 10d cover -> 10d x 10 x $5 = 500; C has 1d cover -> 19 x 2 x 50 = 1900
    assert e["target"] == pytest.approx(2400)
    assert e["a_class_sales_at_risk_target"] == pytest.approx(500)


@pytest.fixture(scope="module")
def d():
    return load_data_room()


@pytest.fixture(scope="module")
def series(d):
    return series_stats(d["sku"], d["inventory"], d["history"], d["suppliers"].set_index("supplier").otif)


def test_top3_suppliers_are_over_sixty_percent_of_spend_as_the_rfp_states(d):
    c = concentration(d["suppliers"].set_index("supplier").spend)
    assert c["top3_share"] > 0.60 and c["n_suppliers"] == 46


def test_master_and_supplier_lead_times_are_effectively_unrelated(d):
    assert abs(lead_time_conflict(d["sku"], d["suppliers"])["correlation"]) < 0.2


def test_lateness_adds_safety_stock_and_fixing_it_releases_inventory(series):
    args = (REVIEW_DAYS, HOLD_RATE, BASELINE["fill_rate"], SERVICE_FLOORS)
    assert otif_working_capital_cost(series, *args).sum() > 0
    assert otif_improvement_release(series, None, *args, otif_target=OTIF_TARGET) > 0


def test_supplier_table_inventory_sums_to_company_total(d, series):
    pol, _ = apply_policy(series, REVIEW_DAYS, HOLD_RATE, BASELINE["fill_rate"], SERVICE_FLOORS)
    t = supplier_table(d["sku"], d["suppliers"], pol)
    assert t.inventory.sum() == pytest.approx(BASELINE["inventory"], abs=150)
    assert "SUP-044" in dual_source_candidates(t).index
