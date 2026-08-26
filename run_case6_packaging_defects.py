from cases.case6_packaging_defects import (
    packaging_defects_experiment,
    CONFIRMATION_RUNS,
    CURRENT_DEFECT_RATE,
    MONTHLY_VOLUME,
    COST_PER_DEFECT,
    FACTOR_LABELS,
)
from nuri.statistics.factorial_design import (
    analyze_2k_factorial,
    explain,
    best_combination,
    fit_reduced_model,
    confirmation_test,
    describe_combo,
)

experiment = packaging_defects_experiment()
analysis = analyze_2k_factorial(experiment, scale="logit")
print(explain(analysis))

best_run, best_observed_rate = best_combination(analysis)
significant, predictions = fit_reduced_model(analysis)
best_key = tuple(best_run[f] for f in analysis["factor_names"])
predicted_rate = predictions[best_key]

print()
print("=== Recommended settings, in plain terms ===")
for factor, label in describe_combo(best_run, FACTOR_LABELS).items():
    print(f"  {factor}: {label}")

print()
print("=== Business impact ===")
print(f"Current defect rate: {CURRENT_DEFECT_RATE:.2%}")
print(f"Predicted defect rate at recommended settings: {predicted_rate:.2%}")

monthly_defects_current = CURRENT_DEFECT_RATE * MONTHLY_VOLUME
monthly_defects_new = predicted_rate * MONTHLY_VOLUME
monthly_savings = (monthly_defects_current - monthly_defects_new) * COST_PER_DEFECT
print(f"Current monthly defective bags: {monthly_defects_current:,.0f}")
print(f"Projected monthly defective bags: {monthly_defects_new:,.0f}")
print(f"Projected monthly savings: N{monthly_savings:,.0f}")

print()
print("=== Confirmation run ===")
conf = confirmation_test(predicted_rate, CONFIRMATION_RUNS)
print(f"Predicted rate: {conf['predicted_rate']:.2%}")
print(f"Observed rate in confirmation run: {conf['observed_rate']:.2%}")
ci = conf["confidence_interval"]
print(f"95% CI on observed rate: ({ci.low:.2%}, {ci.high:.2%})")
print(f"p-value (H0: true rate = predicted rate): {conf['p_value']:.4f}")
print(f"Supports the recommendation: {conf['supports_recommendation']}")
