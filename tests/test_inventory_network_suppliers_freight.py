import pandas as pd
import pytest

from nuri.inventory.freight import expedite_analysis
from nuri.inventory.network import utilization_after_policy, warehouse_profile
from nuri.inventory.policy import build_pair_table, order_up_to
from nuri.inventory.rebalance import (
    lane_costs, plan_transfers, pooling_table, positioning_class, same_site_transfers, warehouse_balance,
)
from nuri.inventory.suppliers import concentration, disruption_exposure, otif_improvement_release, supplier_table
from tests.inventory_fixtures import make_data_room

CSL = {"A": 0.99, "B": 0.99, "C": 0.95}


def test_lane_costs_exclude_same_site_rows_and_flag_them():
    dr = make_data_room()
    lanes = lane_costs(dr.freight)
    assert lanes[("W1", "W2")] == 120.0 and lanes[("W2", "W1")] == 140.0
    assert ("W1", "W1") not in lanes.index
    same = same_site_transfers(dr.freight)
    assert same["records"] == 1
    assert same["share_of_transfers"] == pytest.approx(1 / 3)
    assert same["annual_cost"] == pytest.approx(50.0 * 12 / 18)


def test_transfer_moves_stock_from_donor_to_receiver():
    pairs = build_pair_table(make_data_room(on_hand={("W1", "S1"): 3000, ("W2", "S1"): 0}))
    t = order_up_to(pairs, CSL)
    moves, frame = plan_transfers(pairs, t, lane_costs(make_data_room().freight), holding_rate=0.15)
    s1 = moves[moves["SKU"] == "S1"]
    assert len(s1) == 1
    assert s1.iloc[0]["from"] == "W1" and s1.iloc[0]["to"] == "W2"
    need = float(t.loc[(pairs["Warehouse"] == "W2") & (pairs["SKU"] == "S1"), "order_up_to"].iloc[0])
    assert s1.iloc[0]["units"] == pytest.approx(need)
    assert s1.iloc[0]["carry_benefit"] >= s1.iloc[0]["freight_cost"]
    bal = warehouse_balance(frame)
    assert bal.loc["W1", "net_excess"] > 0 > bal.loc["W2", "net_excess"]


def test_transfer_skipped_when_freight_exceeds_carrying_benefit():
    pairs = build_pair_table(make_data_room(on_hand={("W1", "S4"): 20, ("W2", "S4"): 0}))
    t = order_up_to(pairs, CSL)
    moves, _ = plan_transfers(pairs, t, lane_costs(make_data_room().freight))
    assert moves[moves["SKU"] == "S4"].empty  # a few dollars of stock never pays for a $120 shipment


def test_pooling_saves_safety_stock_and_a_items_stay_duplicated():
    dr = make_data_room()
    dr.demand.loc[dr.demand["Month"] % 3 == 0, "Observed Units Sold"] *= 3
    pairs = build_pair_table(dr)
    pool = pooling_table(dr, pairs, CSL)
    assert (pool["saving_units"] >= 0).all()
    labels = positioning_class(pool)
    assert labels["S1"] == "duplicate"


def test_concentration_metrics():
    dr = make_data_room()
    c = concentration(dr)
    assert c["n_suppliers"] == 2
    assert c["top3_share"] == pytest.approx(1.0)
    assert c["hhi"] == pytest.approx(((15000 / 15264) ** 2 + (264 / 15264) ** 2) * 10_000)


def test_supplier_table_ranks_big_unreliable_supplier_first_and_otif_improvement_releases_stock():
    dr = make_data_room()
    dr.demand.loc[dr.demand["Month"] % 2 == 0, "Observed Units Sold"] *= 2
    pairs = build_pair_table(dr)
    table = supplier_table(dr, pairs, CSL)
    assert table.index[0] == "SUP-1"
    assert table.loc["SUP-1", "risk_rank"] == 1
    assert otif_improvement_release(pairs, CSL, ["SUP-1"], 0.99) > 0
    assert otif_improvement_release(pairs, CSL, ["SUP-1"], 0.5) == 0  # no-one is made worse


def test_disruption_exposure_uses_override_stock_and_scales_with_outage():
    pairs = build_pair_table(make_data_room(on_hand={("W1", "S1"): 0, ("W2", "S1"): 0}))
    short = disruption_exposure(pairs, "SUP-1", outage_days=30)
    long = disruption_exposure(pairs, "SUP-1", outage_days=90)
    assert long["lost_revenue"] > short["lost_revenue"] > 0
    stocked = disruption_exposure(pairs, "SUP-1", 30, on_hand_units=pairs["mean_month"] * 100)
    assert stocked["lost_revenue"] == 0


def test_expedite_premium_is_cost_above_standard_shipping():
    fr = make_data_room().freight
    e = expedite_analysis(fr)
    assert e["mean_expedite_cost"] == 250.0
    assert e["mean_standard_cost"] == 100.0  # mean of LTL 100 and parcel 100
    assert e["premium_per_expedite"] == 150.0
    assert e["annual_premium"] == pytest.approx(150.0 * 1 * 12 / 18)
    assert e["all_expedites_to_customers"]


def test_warehouse_profile_and_space_projection():
    dr = make_data_room()
    pairs = build_pair_table(dr)
    wh = warehouse_profile(dr, pairs)
    assert wh.loc["W1", "orders_per_sqft"] == pytest.approx(0.6)
    assert wh.loc["W2", "orders_per_sqft"] == pytest.approx(0.8)
    assert wh["order_share"].sum() == pytest.approx(1.0)
    target_units = pd.Series({"W1": wh.loc["W1", "inventory_units"] / 2, "W2": wh.loc["W2", "inventory_units"] / 2})
    after = utilization_after_policy(wh, target_units)
    assert after.loc["W1", "utilization_after_pct"] == pytest.approx(35.0)
