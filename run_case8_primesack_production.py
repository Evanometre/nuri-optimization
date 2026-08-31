from scipy import stats

from cases.case8_primesack_production import (
    historical_series,
    SEPTEMBER_MONTH_INDEX,
    SEPTEMBER_CALENDAR_MONTH,
    STARTING_INVENTORY,
    NORMAL_CAPACITY,
    OVERTIME_CAPACITY,
    OVERTIME_EXTRA_COST_PER_SACK,
    STOCKOUT_COST_PER_SACK,
    HOLDING_COST_PER_SACK_PER_MONTH,
    MAX_ENDING_INVENTORY,
)
from nuri.statistics.timeseries_forecast import fit_trend_seasonal, forecast, press_rmse, seasonal_effects, trend_per_period
from nuri.statistics.newsvendor import ProductionRecommendation

month_indices, seasons, values = historical_series()
fit = fit_trend_seasonal(month_indices, seasons, values, baseline_season=1)
sigma = press_rmse(fit)
fc = forecast(fit, SEPTEMBER_MONTH_INDEX, SEPTEMBER_CALENDAR_MONTH, residual_std=sigma)
mu = fc["point"]

print("=== Forecast ===")
print(f"Trend: +{trend_per_period(fit):.1f} sacks/month")
print(f"September point forecast: {mu:,.0f}")
print(f"95% prediction interval: {fc['lower']:,.0f} - {fc['upper']:,.0f}")
print(f"(In-sample residual std: {fit.residual_std:.0f}; PRESS/LOOCV RMSE used instead: {sigma:.0f})")

rec = ProductionRecommendation(
    mu=mu, sigma=sigma, underage_cost=STOCKOUT_COST_PER_SACK, overage_cost=HOLDING_COST_PER_SACK_PER_MONTH,
    starting_inventory=STARTING_INVENTORY, normal_capacity=NORMAL_CAPACITY,
    overtime_capacity=OVERTIME_CAPACITY, overtime_extra_cost=OVERTIME_EXTRA_COST_PER_SACK,
    max_ending_inventory=MAX_ENDING_INVENTORY,
)
optimal = rec.recommend()

print()
print("=== Calculation chain ===")
print(rec.explain_calculation())

print()
print("=== Recommendation ===")
print(f"Critical ratio (Cu/(Cu+Co)): {optimal['critical_ratio']:.1%}")
print(f"Recommended production: {optimal['production']:,.0f} sacks")
print(f"Available stock (start inv + production): {optimal['available_stock']:,.0f}")
print(f"Capacity binding (overtime needed): {optimal['capacity_binding']}")
print(f"Expected total cost: N{optimal['expected_total_cost_including_overtime']:,.0f}")
print(f"P(stockout): {optimal['prob_stockout']:.1%}")
print(f"P(ending inventory > {MAX_ENDING_INVENTORY:,}): {optimal['prob_ending_inventory_exceeds_cap']:.4f}")

print()
print("=== Comparison against named alternatives ===")
scenarios = {
    "A. Conservative (26,000)": 26_000,
    "B. At expected demand": round(mu - STARTING_INVENTORY),
    "Recommended": round(optimal["production"]),
    "C. Aggressive (55,000)": 55_000,
}
for name, production in scenarios.items():
    r = rec.evaluate(production)
    print(f"  {name:32s} production={production:>7,.0f}  E[cost]=N{r['expected_total_cost_including_overtime']:>10,.0f}  P(stockout)={r['prob_stockout']:.1%}")

print()
print("=== Robustness ===")
mu_high = mu * 1.10
rec_high = ProductionRecommendation(
    mu=mu_high, sigma=sigma, underage_cost=STOCKOUT_COST_PER_SACK, overage_cost=HOLDING_COST_PER_SACK_PER_MONTH,
    starting_inventory=STARTING_INVENTORY, normal_capacity=NORMAL_CAPACITY,
    overtime_capacity=OVERTIME_CAPACITY, overtime_extra_cost=OVERTIME_EXTRA_COST_PER_SACK,
    max_ending_inventory=MAX_ENDING_INVENTORY,
)
stuck = rec_high.evaluate(optimal["production"])
reoptimized = rec_high.recommend()
print(f"If demand is +10% and we stick with {optimal['production']:,.0f}: E[cost]=N{stuck['expected_total_cost_including_overtime']:,.0f}, P(stockout)={stuck['prob_stockout']:.0%}")
print(f"If demand is +10% and we re-optimize: production={reoptimized['production']:,.0f}, E[cost]=N{reoptimized['expected_total_cost_including_overtime']:,.0f}")

print()
print("Stockout-cost sensitivity (production barely moves across a wide range):")
for cu in (50, 95, 150, 300):
    r = ProductionRecommendation(
        mu=mu, sigma=sigma, underage_cost=cu, overage_cost=HOLDING_COST_PER_SACK_PER_MONTH,
        starting_inventory=STARTING_INVENTORY, normal_capacity=NORMAL_CAPACITY,
        overtime_capacity=OVERTIME_CAPACITY, overtime_extra_cost=OVERTIME_EXTRA_COST_PER_SACK,
        max_ending_inventory=MAX_ENDING_INVENTORY,
    ).recommend()
    print(f"  Cu=N{cu}: production={r['production']:,.0f}")
