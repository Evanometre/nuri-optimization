import pytest

from cases.case7_choprun_promotions import choprun_experiment
from nuri.statistics.fractional_factorial import (
    analyze_fractional,
    best_combination,
    fit_reduced_model,
    holdout_validation,
    segment_effect_difference,
)


@pytest.fixture(scope="module")
def experiment():
    return choprun_experiment()


@pytest.fixture(scope="module")
def analysis(experiment):
    return analyze_fractional(experiment)


def test_recovers_all_four_planted_main_effects(analysis):
    # All four main effects have a real planted effect and should survive
    # Bonferroni correction across the 7 estimable contrasts.
    for factor in ("A", "B", "C", "D"):
        assert analysis["effects"][factor]["significant_bonferroni"]

    # C (spread) was planted as a pure cost with no reorder benefit -- its
    # net effect on contribution should come back negative.
    assert analysis["effects"]["C"]["effect"] < 0
    # A, B, D were planted with a positive net effect on contribution.
    assert analysis["effects"]["A"]["effect"] > 0
    assert analysis["effects"]["B"]["effect"] > 0
    assert analysis["effects"]["D"]["effect"] > 0


def test_no_spurious_two_way_interactions(analysis):
    # No two-way interaction was planted in the main-effects model -- none
    # of the three aliased pairs should survive Bonferroni correction.
    for eff in ("AB", "AC", "BC"):
        assert not analysis["effects"][eff]["significant_bonferroni"]


def test_aliasing_is_reported_honestly(analysis):
    assert analysis["effects"]["AB"]["aliased_with"] == "CD"
    assert analysis["effects"]["AC"]["aliased_with"] == "BD"
    assert analysis["effects"]["BC"]["aliased_with"] == "AD"


def test_best_combination_matches_the_true_optimum(analysis):
    # True optimum: cheap discount, 7-day timing, selected restaurants,
    # personalized message.
    best_run, best_val = best_combination(analysis)
    assert best_run == {"A": -1, "B": 1, "C": -1, "D": 1}
    assert best_val > 200


def test_reduced_model_prediction_matches_observed_at_best_combo(analysis):
    best_run, best_val = best_combination(analysis)
    _, _, predict = fit_reduced_model(analysis)
    predicted = predict(best_run)
    assert predicted == pytest.approx(best_val, abs=15)


def test_holdout_validates_the_model(analysis, experiment):
    result = holdout_validation(analysis, experiment.holdout_customers)
    assert result["supports_recommendation"]
    assert result["predicted"] == pytest.approx(result["observed_mean"], abs=40)


def test_message_effect_differs_sharply_by_segment(experiment):
    # This is the planted finding: personalization matters far more for the
    # high_value segment than for regular customers.
    result = segment_effect_difference(experiment, "D", "segment")
    assert result["significantly_different"]
    assert result["by_segment"]["high_value"]["effect"] > result["by_segment"]["regular"]["effect"]


def test_other_factors_do_not_differ_by_segment(experiment):
    # Only D was planted with a segment interaction -- A, B, C should not
    # show a significant difference.
    for factor in ("A", "B", "C"):
        result = segment_effect_difference(experiment, factor, "segment")
        assert not result["significantly_different"], f"{factor} unexpectedly differs by segment"
