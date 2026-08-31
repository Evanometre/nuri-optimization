import pytest

from cases.case8_primesack_production import (
    historical_series,
    SEPTEMBER_MONTH_INDEX,
    SEPTEMBER_CALENDAR_MONTH,
)
from nuri.statistics.timeseries_forecast import (
    fit_trend_seasonal,
    forecast,
    press_rmse,
    seasonal_effects,
    trend_per_period,
)


@pytest.fixture(scope="module")
def fit():
    month_indices, seasons, values = historical_series()
    return fit_trend_seasonal(month_indices, seasons, values, baseline_season=1)


def test_trend_is_positive_and_reasonable(fit):
    # Demand grew from ~21,800 to ~33,400 over 30 months -- trend should be
    # positive and roughly in the 100-300 sacks/month range.
    trend = trend_per_period(fit)
    assert 100 < trend < 300


def test_september_is_the_strongest_seasonal_month(fit):
    # September and October are visibly the demand peaks in the raw data
    # every year -- the fitted seasonal effects should reflect that.
    effects = seasonal_effects(fit)
    assert effects[9] == max(effects.values()) or effects[10] == max(effects.values())
    assert effects[9] > effects[2]  # September comfortably beats the weakest month (Feb)


def test_press_rmse_exceeds_in_sample_residual_std(fit):
    # With only ~2.5 observations per calendar month, in-sample residuals
    # understate true forecast error -- PRESS (LOOCV) RMSE must be larger.
    assert press_rmse(fit) > fit.residual_std


def test_september_forecast_is_a_new_series_high_but_consistent_with_trend(fit):
    sigma = press_rmse(fit)
    fc = forecast(fit, SEPTEMBER_MONTH_INDEX, SEPTEMBER_CALENDAR_MONTH, residual_std=sigma)
    # Aug 2026 was 33,400 (the series' prior high); Sep has historically
    # jumped 11-14% further above August in both prior years.
    assert 35_000 < fc["point"] < 39_000
    assert fc["lower"] < fc["point"] < fc["upper"]


def test_prediction_interval_widens_with_press_adjustment(fit):
    naive = forecast(fit, SEPTEMBER_MONTH_INDEX, SEPTEMBER_CALENDAR_MONTH)
    adjusted = forecast(
        fit, SEPTEMBER_MONTH_INDEX, SEPTEMBER_CALENDAR_MONTH, residual_std=press_rmse(fit)
    )
    assert adjusted["se"] > naive["se"]
