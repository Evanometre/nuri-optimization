"""Trend + seasonal regression forecasting, with proper OLS prediction
intervals -- not just a point forecast.

For monthly business data with a handful of years of history, a full ARIMA/
SARIMA fit is often overkill and hard to explain to a client; a linear trend
plus month-of-year dummy variables is transparent (every coefficient has a
plain-English meaning: "September runs X units above the baseline month"),
easy to validate against the raw numbers, and gives closed-form prediction
intervals via standard OLS theory.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class TrendSeasonalFit:
    coefficients: np.ndarray
    dummy_seasons: list
    baseline_season: int
    residual_std: float
    df_resid: int
    XtX_inv: np.ndarray
    n: int
    fitted: np.ndarray = field(default=None)
    residuals: np.ndarray = field(default=None)
    X: np.ndarray = field(default=None)


def _design_row(month_index, season, dummy_seasons):
    row = np.zeros(2 + len(dummy_seasons))
    row[0] = 1
    row[1] = month_index
    for j, s in enumerate(dummy_seasons):
        row[2 + j] = 1 if season == s else 0
    return row


def fit_trend_seasonal(month_indices, seasons, values, period=12, baseline_season=1):
    """month_indices: sequential time index (1, 2, 3, ...). seasons: month-of-year
    (1=Jan..12=Dec, or any 1..period labeling) for each observation. values: the
    observed series. baseline_season is the reference season folded into the
    intercept (its coefficient is implicitly 0)."""
    n = len(values)
    all_seasons = sorted(set(seasons))
    dummy_seasons = [s for s in all_seasons if s != baseline_season]

    X = np.array([_design_row(month_indices[i], seasons[i], dummy_seasons) for i in range(n)])
    y = np.array(values, dtype=float)

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    k = X.shape[1]
    df_resid = n - k
    residual_std = np.sqrt(np.sum(resid ** 2) / df_resid)
    XtX_inv = np.linalg.inv(X.T @ X)

    return TrendSeasonalFit(
        coefficients=beta,
        dummy_seasons=dummy_seasons,
        baseline_season=baseline_season,
        residual_std=residual_std,
        df_resid=df_resid,
        XtX_inv=XtX_inv,
        n=n,
        fitted=fitted,
        residuals=resid,
        X=X,
    )


def press_residuals(fit):
    """Leave-one-out cross-validation residuals, computed exactly (no
    refitting needed) via the hat-matrix leverage: e_loocv_i = e_i / (1 - h_ii).
    With few observations per seasonal parameter, in-sample residuals can
    understate true forecast error -- PRESS residuals correct for that by
    measuring how wrong the model would have been predicting each point
    WITHOUT having seen it, which is what actually matters for forecasting
    a genuinely new month.
    """
    H_diag = np.einsum("ij,jk,ik->i", fit.X, fit.XtX_inv, fit.X)  # leverage h_ii
    return fit.residuals / (1 - H_diag)


def press_rmse(fit):
    return float(np.sqrt(np.mean(press_residuals(fit) ** 2)))


def forecast(fit, month_index, season, confidence=0.95, residual_std=None):
    """Point forecast plus a genuine prediction interval (accounts for both
    parameter uncertainty and residual noise -- wider than a confidence
    interval on the mean would be, which is the correct interval for
    planning around a single future month's actual demand).

    residual_std: override the in-sample residual std (e.g. with the PRESS/
    LOOCV RMSE from press_rmse(fit)) when there are few observations per
    seasonal parameter and in-sample residuals likely understate real
    forecast error.
    """
    x0 = _design_row(month_index, season, fit.dummy_seasons)
    point = float(x0 @ fit.coefficients)
    leverage = float(x0 @ fit.XtX_inv @ x0)
    sigma = residual_std if residual_std is not None else fit.residual_std
    pred_var = sigma ** 2 * (1 + leverage)
    pred_se = np.sqrt(pred_var)
    t_crit = stats.t.ppf(1 - (1 - confidence) / 2, df=fit.df_resid)

    return {
        "point": point,
        "se": pred_se,
        "lower": point - t_crit * pred_se,
        "upper": point + t_crit * pred_se,
        "confidence": confidence,
        "df": fit.df_resid,
    }


def seasonal_effects(fit):
    """Each season's estimated effect relative to the baseline season, in
    the same units as the original series (e.g. 'September runs +X sacks
    above the baseline month, on average')."""
    effects = {fit.baseline_season: 0.0}
    for j, s in enumerate(fit.dummy_seasons):
        effects[s] = float(fit.coefficients[2 + j])
    return effects


def trend_per_period(fit):
    return float(fit.coefficients[1])
