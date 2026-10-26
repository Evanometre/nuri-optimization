import numpy as np
import pandas as pd
import pytest
from scipy import stats

from cases.case12_meridian_diagnostic import (
    load_data_room, BASELINE, REVIEW_DAYS, HOLD_RATE, SERVICE_FLOORS,
)
from nuri.inventory_policy import (
    series_stats, apply_policy, summarize, solve_service_levels, release_schedule, sigma_over_interval,
)
from nuri.rebalancing import solve_rebalancing, lane_cost_index


def toy(n=6):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "warehouse": ["W1"] * n, "SKU": [f"S{i}" for i in range(n)],
        "abc": ["C"] * n, "cost": rng.uniform(5, 200, n), "d": rng.uniform(1, 20, n),
        "sd_d": rng.uniform(0.5, 6, n), "L": 30.0, "sd_L": 0.0,
        "on_hand": rng.uniform(0, 600, n), "aged_pct": 0.1,
    })


def test_service_solver_hits_the_network_fill_target_exactly():
    s = toy()
    z, _ = solve_service_levels(s, 30, 0.2, 0.92, {})
    assert stats.norm.cdf(z).mean() == pytest.approx(0.92, abs=1e-6)


def test_service_solver_beats_a_uniform_z_at_the_same_network_fill():
    # The whole point of optimising z: same average service, less safety-stock cost.
    s = toy()
    sigma = sigma_over_interval(s, 30).values
    z_opt, _ = solve_service_levels(s, 30, 0.2, 0.92, {})
    z_uni = np.full(len(s), stats.norm.ppf(0.92))
    assert (s.cost.values * sigma * z_opt).sum() < (s.cost.values * sigma * z_uni).sum()


def test_class_floor_is_respected_even_when_it_costs_inventory():
    s = toy()
    s.loc[0, "abc"] = "A"
    z, _ = solve_service_levels(s, 30, 0.2, 0.5, {"A": 0.995})
    assert stats.norm.cdf(z[0]) >= 0.995 - 1e-9


def test_rebalancing_moves_excess_to_shortfall_and_respects_both_limits():
    pol = pd.DataFrame({
        "SKU": ["S", "S"], "warehouse": ["W1", "W2"], "cost": [100.0, 100.0], "d": [1.0, 1.0],
        "excess_units": [40.0, 0.0], "need_units": [0.0, 25.0],
    })
    t = solve_rebalancing(pol, {("W1", "W2"): 1.0}, 0.2, 0.02)
    assert t.units.sum() == pytest.approx(25)      # capped by the shortfall, not the excess
    assert t.purchase_avoided.sum() == pytest.approx(2500)


def test_rebalancing_does_not_move_stock_when_freight_exceeds_the_holding_gain():
    pol = pd.DataFrame({
        "SKU": ["S", "S"], "warehouse": ["W1", "W2"], "cost": [100.0, 100.0], "d": [100.0, 1.0],
        "excess_units": [10.0, 0.0], "need_units": [0.0, 10.0],   # burns off in <1 day: nothing to gain
    })
    assert solve_rebalancing(pol, {("W1", "W2"): 1.0}, 0.2, 0.05).empty


@pytest.fixture(scope="module")
def d():
    return load_data_room()


@pytest.fixture(scope="module")
def base(d):
    s = series_stats(d["sku"], d["inventory"], d["history"], d["suppliers"].set_index("supplier").otif)
    return apply_policy(s, REVIEW_DAYS, HOLD_RATE, BASELINE["fill_rate"], SERVICE_FLOORS)[0]


def test_policy_frame_carries_every_dollar_of_current_inventory(base):
    # outer join: stock with no demand history must not be dropped
    assert summarize(base)["current_inventory"] == pytest.approx(BASELINE["inventory"], abs=150)


def test_policy_meets_network_fill_and_a_class_floor(base):
    assert summarize(base)["network_fill"] == pytest.approx(BASELINE["fill_rate"], abs=1e-4)
    assert base[base.abc == "A"].service_level.min() >= 0.995 - 1e-9


def test_conservative_variability_releases_less_than_observed(d):
    otif = d["suppliers"].set_index("supplier").otif
    def rel(src):
        s = series_stats(d["sku"], d["inventory"], d["history"], otif, src)
        return summarize(apply_policy(s, 30, HOLD_RATE, 0.90, SERVICE_FLOORS)[0])["net_release"]
    assert 0 < rel("master") < rel("history")


def test_release_schedule_is_monotone_and_bounded_by_excess(base):
    sched, stuck = release_schedule(base)
    assert all(b >= a for a, b in zip(sched, sched[1:]))
    assert sched[-1] + stuck <= summarize(base)["excess_value"] + 1
