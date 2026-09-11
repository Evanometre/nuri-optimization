import pytest

from cases.case11_learnbridge_pricing import (
    segments, VARIABLE_COST, CURRENT_PRICE, CURRENT_SUBSCRIBERS,
    BASIC_GRID, PLUS_GRID, PRO_GRID, FEATURE_FACTORS, TIER_ORDER,
    MIN_MARGIN, MIN_TIER_SPACING, MAX_BASIC_PRICE, MAX_EXISTING_CUSTOMER_INCREASE,
    MIN_PROFESSIONAL_PRO_REACH, OBSERVATIONAL_PRICES, OBSERVATIONAL_NEW_SUBS,
)
from nuri.pricing import (
    single_price_outcome, enumerate_tier_menus, fit_power_law, SegmentDemand,
)


def test_segment_model_reproduces_current_known_state_exactly():
    # This is the anchor point the whole model is calibrated to -- if this
    # doesn't match the case's given numbers exactly, nothing downstream can
    # be trusted.
    outcome = single_price_outcome(CURRENT_PRICE, segments(), VARIABLE_COST)
    assert outcome["subscribers"] == pytest.approx(CURRENT_SUBSCRIBERS)
    assert outcome["contribution"] == pytest.approx(8_712_000, rel=1e-6)


def test_inelastic_professional_segment_has_no_finite_contribution_maximum():
    # Elasticity -0.5 (|e|<1) under a constant-elasticity curve means
    # contribution rises without bound as price increases -- a real
    # mathematical property of this functional form, not a bug. Pinned so
    # nobody "fixes" it by capping the curve without understanding why.
    professionals = SegmentDemand("professionals", 1499, 2800, -0.5)
    low = single_price_outcome(1899, [professionals], VARIABLE_COST)
    high = single_price_outcome(5999, [professionals], VARIABLE_COST)
    assert high["contribution"] > low["contribution"]


def test_elastic_student_segment_has_an_interior_contribution_maximum():
    # Closed form: P* = c * e / (1 + e) for e < -1. For e=-1.2, c=410:
    # P* = 410 * 6 = 2460.
    students = SegmentDemand("students", 1499, 5200, -1.2)
    at_peak = single_price_outcome(2460, [students], VARIABLE_COST)
    just_below = single_price_outcome(2200, [students], VARIABLE_COST)
    just_above = single_price_outcome(2700, [students], VARIABLE_COST)
    assert at_peak["contribution"] > just_below["contribution"]
    assert at_peak["contribution"] > just_above["contribution"]


def test_single_price_contribution_is_monotonic_across_the_credible_range():
    # Because professionals dominate the blend at higher prices, blended
    # contribution keeps rising across the whole tested range -- meaning the
    # 10%-transition-cap constraint, not the demand curve, is what actually
    # limits the single-price recommendation.
    prices = [1199, 1399, 1499, 1599, 1649, 1799, 1999]
    contributions = [single_price_outcome(p, segments(), VARIABLE_COST)["contribution"] for p in prices]
    assert contributions == sorted(contributions)


def test_observational_table_elasticity_is_more_elastic_than_segment_blend():
    # Acquisition elasticity (new signups) is expected to run more elastic
    # than the existing-base segment elasticities, since new customers have
    # no switching cost. Documents the expected gap rather than silently
    # blending two different demand concepts together.
    _, acquisition_elasticity = fit_power_law(OBSERVATIONAL_PRICES, OBSERVATIONAL_NEW_SUBS)
    blended_segment_elasticity = 0.65 * -1.2 + 0.35 * -0.5
    assert acquisition_elasticity < blended_segment_elasticity


def test_tier_menu_enumeration_matches_known_result():
    max_plus = CURRENT_PRICE * (1 + MAX_EXISTING_CUSTOMER_INCREASE)
    results = enumerate_tier_menus(
        segments(), BASIC_GRID, PLUS_GRID, PRO_GRID, FEATURE_FACTORS, VARIABLE_COST, TIER_ORDER,
        MIN_MARGIN, MIN_TIER_SPACING, MAX_BASIC_PRICE, max_plus, MIN_PROFESSIONAL_PRO_REACH,
    )
    feasible = [r for r in results if r["feasible"]]
    assert len(feasible) == 6

    best = max(feasible, key=lambda r: r["outcome"]["contribution"])
    assert best["prices"] == {"Basic": 1199, "Plus": 1599, "Pro": 2599}
    assert best["outcome"]["contribution"] == pytest.approx(13_364_487, rel=1e-4)


def test_all_basic_999_combinations_fail_the_margin_constraint():
    # (999 - 410) / 999 = 59.0% < 65% floor -- Basic at 999 can never be
    # feasible regardless of Plus/Pro pricing.
    max_plus = CURRENT_PRICE * (1 + MAX_EXISTING_CUSTOMER_INCREASE)
    results = enumerate_tier_menus(
        segments(), [999], PLUS_GRID, PRO_GRID, FEATURE_FACTORS, VARIABLE_COST, TIER_ORDER,
        MIN_MARGIN, MIN_TIER_SPACING, MAX_BASIC_PRICE, max_plus, MIN_PROFESSIONAL_PRO_REACH,
    )
    assert all("margin" in r["failed_constraints"] for r in results)


def test_conservative_tier_menu_still_beats_single_price_and_baseline():
    # Even the least aggressive feasible tier menu (Pro at the bottom of its
    # grid, closest to the tested elasticity range) should still clearly
    # beat both the current baseline and the constrained single-price option
    # -- the case's central point ("redesign the architecture, don't just
    # raise the price") should hold even under the most conservative
    # feasible choice, not only the model's single top pick.
    max_plus = CURRENT_PRICE * (1 + MAX_EXISTING_CUSTOMER_INCREASE)
    results = enumerate_tier_menus(
        segments(), BASIC_GRID, PLUS_GRID, [2199], FEATURE_FACTORS, VARIABLE_COST, TIER_ORDER,
        MIN_MARGIN, MIN_TIER_SPACING, MAX_BASIC_PRICE, max_plus, MIN_PROFESSIONAL_PRO_REACH,
    )
    feasible = [r for r in results if r["feasible"]]
    assert feasible
    conservative_best = max(feasible, key=lambda r: r["outcome"]["contribution"])

    baseline_contribution = 8_712_000
    single_price_at_cap = single_price_outcome(max_plus, segments(), VARIABLE_COST)["contribution"]

    assert conservative_best["outcome"]["contribution"] > baseline_contribution
    assert conservative_best["outcome"]["contribution"] > single_price_at_cap


def test_nested_tier_demand_never_negative_or_double_counted():
    # Sanity check on the nested-WTP allocation: no tier gets negative
    # subscribers, and total across tiers matches the sum (no double count).
    outcome = enumerate_tier_menus(
        segments(), BASIC_GRID, PLUS_GRID, PRO_GRID, FEATURE_FACTORS, VARIABLE_COST, TIER_ORDER,
        MIN_MARGIN, MIN_TIER_SPACING, MAX_BASIC_PRICE, 1_000_000, 0.0,
    )[0]["outcome"]
    assert all(q >= -1e-6 for q in outcome["subs_by_tier"].values())
    assert sum(outcome["subs_by_tier"].values()) == pytest.approx(outcome["total_subscribers"])
