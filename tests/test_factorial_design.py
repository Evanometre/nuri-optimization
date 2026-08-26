import pytest

from cases.case6_packaging_defects import (
    packaging_defects_experiment,
    CONFIRMATION_RUNS,
    CURRENT_DEFECT_RATE,
    MONTHLY_VOLUME,
    COST_PER_DEFECT,
)
from nuri.statistics.factorial_design import (
    analyze_2k_factorial,
    best_combination,
    fit_reduced_model,
    confirmation_test,
)


def test_logit_scale_recovers_the_true_planted_structure():
    # The synthetic data was generated with real effects only for B (large),
    # C (moderate), and the A x B interaction (large) -- D and every other
    # interaction were generated with zero true effect. On the logit scale
    # (correct for this wide a range of rates), the Bonferroni-corrected
    # results should recover exactly that, with nothing else surviving.
    analysis = analyze_2k_factorial(packaging_defects_experiment(), scale="logit")

    assert analysis["effects"]["B"]["significant_bonferroni"]
    assert analysis["effects"]["C"]["significant_bonferroni"]
    assert analysis["effects"]["AB"]["significant_bonferroni"]
    assert not analysis["effects"]["D"]["significant_bonferroni"]

    # None of the other 11 effects (which have zero true effect) should
    # survive the Bonferroni correction either -- that's the whole point of
    # applying it, given 15 simultaneous tests.
    for eff, r in analysis["effects"].items():
        if eff not in ("B", "C", "AB"):
            assert not r["significant_bonferroni"], f"{eff} unexpectedly survived correction"


def test_raw_rate_scale_is_visibly_worse_for_this_data():
    # This is the actual reason to prefer the logit scale here: naive
    # raw-rate ANOVA manufactures several spurious "significant" interactions
    # purely from the nonlinearity near 0%/100%, which the logit scale avoids.
    rate_analysis = analyze_2k_factorial(packaging_defects_experiment(), scale="rate")
    logit_analysis = analyze_2k_factorial(packaging_defects_experiment(), scale="logit")

    rate_significant = sum(1 for r in rate_analysis["effects"].values() if r["significant"])
    logit_significant_bonferroni = sum(
        1 for r in logit_analysis["effects"].values() if r["significant_bonferroni"]
    )
    assert rate_significant > logit_significant_bonferroni


def test_best_combination_matches_the_true_optimum():
    # True optimum (from the hidden generating process) is A=high, B=low,
    # C=high -- D doesn't matter. The empirically best-observed combination
    # should land on exactly that (up to D, which has no true effect).
    analysis = analyze_2k_factorial(packaging_defects_experiment(), scale="logit")
    best_run, best_rate = best_combination(analysis)

    assert best_run["A"] == 1
    assert best_run["B"] == -1
    assert best_run["C"] == 1
    assert best_rate < 0.02  # true rate at this combo is ~0.7%


def test_confirmation_run_supports_the_recommendation():
    analysis = analyze_2k_factorial(packaging_defects_experiment(), scale="logit")
    best_run, _ = best_combination(analysis)
    _, predictions = fit_reduced_model(analysis)
    best_key = tuple(best_run[f] for f in analysis["factor_names"])
    predicted_rate = predictions[best_key]

    result = confirmation_test(predicted_rate, CONFIRMATION_RUNS)
    assert result["supports_recommendation"]
    assert result["observed_rate"] < 0.02


def test_projected_savings_are_substantial_and_directionally_correct():
    analysis = analyze_2k_factorial(packaging_defects_experiment(), scale="logit")
    best_run, _ = best_combination(analysis)
    _, predictions = fit_reduced_model(analysis)
    best_key = tuple(best_run[f] for f in analysis["factor_names"])
    predicted_rate = predictions[best_key]

    assert predicted_rate < CURRENT_DEFECT_RATE

    monthly_savings = (CURRENT_DEFECT_RATE - predicted_rate) * MONTHLY_VOLUME * COST_PER_DEFECT
    assert monthly_savings == pytest.approx(744_371, rel=0.05)
