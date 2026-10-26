import numpy as np
import pandas as pd
import pytest

from nuri.inventory.opportunity import (
    Scenario, compare_scenarios, excess_by_cover, release_timeline, sensitivity, targets_for, waterfall,
)
from nuri.inventory.policy import (
    _quantile, build_pair_table, order_up_to, protection_distribution, simulate, summarize_simulation,
)
from nuri.inventory.segmentation import recompute_abc, sku_segments
from tests.inventory_fixtures import make_data_room

CSL = {"A": 0.999, "B": 0.99, "C": 0.95}


def _pairs(**kw):
    return build_pair_table(make_data_room(**kw))


def test_pair_table_covers_every_sku_warehouse_and_values_stock_at_cost():
    pairs = _pairs()
    assert len(pairs) == 8
    row = pairs[(pairs["Warehouse"] == "W1") & (pairs["SKU"] == "S1")].iloc[0]
    assert row["value"] == 200 * 10.0
    assert row["mean_month"] == 50
    assert row["lead_time"] == 30


def test_quantile_is_monotonic_in_service_level_and_zero_without_demand():
    mean = np.array([20.0, 20.0, 0.0])
    var = np.array([50.0, 50.0, 1.0])
    low = _quantile(0.90, mean, var)
    high = _quantile(0.999, mean, var)
    assert (high[:2] > low[:2]).all()
    assert high[2] == 0


def test_constant_demand_and_perfect_supplier_needs_no_safety_stock():
    dr = make_data_room()
    dr.supplier["OTIF"] = 1.0
    pairs = build_pair_table(dr)
    mean, var = protection_distribution(pairs)
    active = (pairs["mean_month"] > 0).to_numpy()
    assert np.allclose(var[active], mean[active])  # zero spread -> variance floored at Poisson (= mean)
    t = order_up_to(pairs, CSL)
    s1 = t[(t["SKU"] == "S1")].iloc[0]
    # 50 units/month, 37-day protection interval -> mean 60.9 units; sd 0 -> Poisson tail only.
    assert s1["order_up_to"] >= 61
    assert s1["order_up_to"] < 61 + 4 * np.sqrt(61)


def test_lower_otif_raises_safety_stock():
    good, bad = make_data_room(), make_data_room()
    good.supplier["OTIF"] = 0.99
    bad.supplier["OTIF"] = 0.70
    # Give the demand some variability so lead-time spread matters.
    for dr in (good, bad):
        dr.demand.loc[dr.demand["Month"] % 2 == 0, "Observed Units Sold"] *= 2
    ss_good = order_up_to(build_pair_table(good), CSL)["safety_stock"].sum()
    ss_bad = order_up_to(build_pair_table(bad), CSL)["safety_stock"].sum()
    assert ss_bad > ss_good


def test_higher_class_service_level_means_more_inventory():
    pairs = _pairs()
    low = order_up_to(pairs, {"A": 0.9, "B": 0.9, "C": 0.9})["target_value"].sum()
    high = order_up_to(pairs, {"A": 0.999, "B": 0.999, "C": 0.999})["target_value"].sum()
    assert high > low


def test_simulation_meets_service_target_and_reports_average_stock():
    dr = make_data_room()
    rng = np.random.default_rng(0)
    dr.demand["Observed Units Sold"] = rng.poisson(40, len(dr.demand)).clip(min=1)
    pairs = build_pair_table(dr)
    t = order_up_to(pairs, {"A": 0.99, "B": 0.99, "C": 0.99})
    sim = simulate(pairs, t["order_up_to"].to_numpy(), seed=3)
    by_class, total = summarize_simulation(pairs, sim)
    assert by_class["unit_fill"].min() > 0.97
    assert total["avg_inventory_value"] > 0
    assert (sim["avg_on_hand"] <= t["order_up_to"] + 1e-9).all()


def test_simulation_zero_demand_pairs_never_flag_stockouts():
    dr = make_data_room()
    dr.demand = dr.demand[dr.demand["SKU"] != "S4"]
    pairs = build_pair_table(dr)
    t = order_up_to(pairs, CSL)
    sim = simulate(pairs, t["order_up_to"].to_numpy())
    s4 = pairs["SKU"] == "S4"
    assert (sim.loc[s4, "line_flag_rate"] == 0).all()


def test_waterfall_buckets_sum_exactly_to_target():
    pairs = _pairs(on_hand={("W1", "S1"): 1000, ("W2", "S2"): 1})
    sc = Scenario("t", CSL)
    t = targets_for(pairs, sc)
    wf = waterfall(pairs, t)
    parts = wf.drop(["current_inventory", "target_inventory"]).sum()
    assert wf["current_inventory"] + parts == pytest.approx(wf["target_inventory"])
    assert wf["target_inventory"] == pytest.approx(t["target_value"].sum())
    assert wf["excess_above_order_up_to"] < 0  # the 1,000-unit pile is excess
    assert wf["build_to_target_deficits"] > 0  # the 1-unit pair needs building


def test_release_timeline_converges_to_target_and_never_rises_after_month_zero():
    pairs = _pairs(on_hand={("W1", "S1"): 1000})
    t = targets_for(pairs, Scenario("t", CSL))
    tl = release_timeline(pairs, t, months=(0, 3, 6, 12, 60))
    assert tl["inventory"].is_monotonic_decreasing
    assert tl["inventory"].iloc[-1] == pytest.approx(t["target_value"].sum(), rel=1e-6)


def test_excess_by_cover_buckets_by_months_of_demand():
    pairs = _pairs(on_hand={("W1", "S1"): 1000})
    t = targets_for(pairs, Scenario("t", CSL))
    ebc = excess_by_cover(pairs, t)
    assert ebc.sum() > 0
    assert ebc.loc["6-12m"] + ebc.loc["3-6m"] + ebc.loc["<=3m"] + ebc.loc["12-24m"] + ebc.loc[">24m / no demand"] == pytest.approx(ebc.sum())


def test_scenarios_order_release_by_conservatism():
    pairs = _pairs(on_hand={("W1", "S1"): 1000, ("W2", "S1"): 1000})
    baseline = {"COGS": 1.0, "Inventory write-offs / obsolescence": 1000.0}
    expedite = {"annual_premium": 1000.0}
    lost = {"shortfall_margin": 100.0}
    scenarios = [
        Scenario("Conservative", {"A": 0.9995, "B": 0.99, "C": 0.95}, review_days=14, lt_sd_factor=1.5, moq_buffer=0.1),
        Scenario("Base", CSL),
    ]
    table = compare_scenarios(pairs, scenarios, baseline, expedite, lost)
    assert table.loc["Conservative", "release"] < table.loc["Base", "release"]
    assert table.loc["Base", "benefit_carrying_cost_saved"] == pytest.approx(table.loc["Base", "release"] * 0.15)


def test_sensitivity_reports_release_per_variation():
    pairs = _pairs(on_hand={("W1", "S1"): 1000})
    out = sensitivity(pairs, Scenario("Base", CSL), {"A 99%": {"csl_by_class": {**CSL, "A": 0.99}}}, {})
    assert list(out.index) == ["A 99%"]
    assert out["release"].iloc[0] > 0


def test_recompute_abc_matches_cumulative_cutoffs():
    sku = make_data_room().sku
    labels = recompute_abc(sku)
    assert labels["S1"] == "A"
    # S2 starts at 78.6% of cumulative usage, below the 80% line, so it is still A.
    assert labels["S2"] == "A"
    assert labels["S3"] == "C" and labels["S4"] == "C"


def test_xyz_thresholds():
    dr = make_data_room()
    seg = sku_segments(dr)
    assert (seg["XYZ"] == "X").all()  # constant demand has zero variation
    dr.demand.loc[(dr.demand["SKU"] == "S1") & (dr.demand["Month"] % 2 == 0), "Observed Units Sold"] = 150
    assert sku_segments(dr).loc["S1", "XYZ"] in {"Y", "Z"}
