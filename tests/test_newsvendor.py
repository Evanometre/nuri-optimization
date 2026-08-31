import pytest
from scipy import stats

from nuri.statistics.newsvendor import (
    critical_ratio,
    optimal_available_stock,
    expected_costs,
    ProductionRecommendation,
)


def test_critical_ratio_matches_definition():
    assert critical_ratio(underage_cost=95, overage_cost=7.5) == pytest.approx(95 / 102.5)


def test_optimal_available_stock_matches_closed_form_normal_quantile():
    mu, sigma = 36_705, 480
    available_stock, z, cr = optimal_available_stock(mu, sigma, underage_cost=95, overage_cost=7.5)
    expected_z = stats.norm.ppf(95 / 102.5)
    assert z == pytest.approx(expected_z)
    assert available_stock == pytest.approx(mu + expected_z * sigma)


def test_high_underage_cost_relative_to_overage_pushes_quantity_above_mean():
    # Cu >> Co (95 vs 7.5) should push the optimal quantity comfortably above
    # the mean -- this is the whole point of the newsvendor model over a
    # naive "produce to the point forecast" rule.
    available_stock, z, cr = optimal_available_stock(36_705, 480, underage_cost=95, overage_cost=7.5)
    assert z > 1.0
    assert available_stock > 36_705 + 400


def test_expected_costs_are_zero_shortage_and_positive_excess_far_above_mean():
    r = expected_costs(available_stock=50_000, mu=36_705, sigma=480, underage_cost=95, overage_cost=7.5)
    assert r["expected_shortage_units"] == pytest.approx(0, abs=1)
    assert r["expected_excess_units"] > 13_000


def test_recommendation_beats_naive_at_expected_demand_and_extremes():
    rec = ProductionRecommendation(
        mu=36_705, sigma=480, underage_cost=95, overage_cost=7.5,
        starting_inventory=6_500, normal_capacity=48_000, overtime_capacity=55_000,
        overtime_extra_cost=18, max_ending_inventory=15_000,
    )
    optimal = rec.recommend()
    at_mean = rec.evaluate(production=round(36_705 - 6_500))
    conservative = rec.evaluate(production=26_000)
    aggressive = rec.evaluate(production=55_000)

    optimal_cost = optimal["expected_total_cost_including_overtime"]
    assert optimal_cost < at_mean["expected_total_cost_including_overtime"]
    assert optimal_cost < conservative["expected_total_cost_including_overtime"]
    assert optimal_cost < aggressive["expected_total_cost_including_overtime"]


def test_recommendation_stays_within_normal_capacity_given_tight_forecast():
    # Given how tight the forecast uncertainty is relative to the capacity
    # headroom, the newsvendor optimum shouldn't require overtime.
    rec = ProductionRecommendation(
        mu=36_705, sigma=480, underage_cost=95, overage_cost=7.5,
        starting_inventory=6_500, normal_capacity=48_000, overtime_capacity=55_000,
        overtime_extra_cost=18, max_ending_inventory=15_000,
    )
    result = rec.recommend()
    assert not result["capacity_binding"]
    assert result["production"] < rec.normal_capacity
    assert result["overtime_cost"] == 0


def test_ending_inventory_cap_is_not_at_risk_at_the_recommended_quantity():
    rec = ProductionRecommendation(
        mu=36_705, sigma=480, underage_cost=95, overage_cost=7.5,
        starting_inventory=6_500, normal_capacity=48_000, overtime_capacity=55_000,
        overtime_extra_cost=18, max_ending_inventory=15_000,
    )
    result = rec.recommend()
    assert result["prob_ending_inventory_exceeds_cap"] < 0.01


def test_explain_calculation_exposes_the_percentile_chain():
    rec = ProductionRecommendation(
        mu=36_705, sigma=373, underage_cost=95, overage_cost=7.5,
        starting_inventory=6_500, normal_capacity=48_000, overtime_capacity=55_000,
        overtime_extra_cost=18, max_ending_inventory=15_000,
    )
    text = rec.explain_calculation()
    assert "92.68%" in text
    assert "37,247" in text  # the percentile value X
    assert "30,747" in text  # X - starting inventory = production


def test_sticking_with_a_static_plan_is_costly_if_demand_surprises_upward():
    # If true demand turns out 10% higher than forecast, a static production
    # plan sized for the original forecast should look expensive in hindsight
    # -- this is the basis for recommending a mid-month check-in, not a
    # "set and forget" plan.
    base_mu = 36_705
    rec_base = ProductionRecommendation(
        mu=base_mu, sigma=480, underage_cost=95, overage_cost=7.5,
        starting_inventory=6_500, normal_capacity=48_000, overtime_capacity=55_000,
        overtime_extra_cost=18, max_ending_inventory=15_000,
    )
    base_production = rec_base.recommend()["production"]

    rec_high_demand = ProductionRecommendation(
        mu=base_mu * 1.10, sigma=480, underage_cost=95, overage_cost=7.5,
        starting_inventory=6_500, normal_capacity=48_000, overtime_capacity=55_000,
        overtime_extra_cost=18, max_ending_inventory=15_000,
    )
    stuck_with_original_plan = rec_high_demand.evaluate(base_production)
    reoptimized = rec_high_demand.recommend()

    assert (
        stuck_with_original_plan["expected_total_cost_including_overtime"]
        > reoptimized["expected_total_cost_including_overtime"] * 10
    )
