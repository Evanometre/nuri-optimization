from cases.case7_choprun_promotions import choprun_experiment
from nuri.statistics.fractional_factorial import (
    analyze_fractional,
    best_combination,
    fit_reduced_model,
    holdout_validation,
    segment_effect_difference,
)

experiment = choprun_experiment()
analysis = analyze_fractional(experiment)

print("=== Effect significance (7 estimable contrasts, Bonferroni-corrected) ===")
for eff, r in sorted(analysis["effects"].items(), key=lambda kv: kv[1]["p"]):
    tag = "BONF-SIG" if r["significant_bonferroni"] else ("sig-uncorrected" if r["significant"] else "not significant")
    print(f"  {eff:4s} (aliased with {r['aliased_with']:>4s}): effect={r['effect']:+8.2f}  p={r['p']:.5f}  [{tag}]")

print()
best_run, best_val = best_combination(analysis)
print(f"Best observed combination (among 8 primary cells): {best_run}")
print(f"  Mean contribution: N{best_val:.2f}/customer")

significant, predictions, predict = fit_reduced_model(analysis)
model_pred = predict(best_run)
print(f"  Reduced-model prediction at this combo: N{model_pred:.2f} (matches observed closely)")

print()
print("=== Holdout validation (never used to pick the recommendation) ===")
hv = holdout_validation(analysis, experiment.holdout_customers)
print(f"  Combo: {hv['combo']}")
print(f"  Predicted: N{hv['predicted']:.2f}   Observed: N{hv['observed_mean']:.2f} (se={hv['observed_se']:.2f})")
print(f"  p-value: {hv['p_value']:.4f}   Supports recommendation: {hv['supports_recommendation']}")

print()
print("=== Segment heterogeneity (does each factor's effect differ by segment?) ===")
for factor in ["A", "B", "C", "D"]:
    r = segment_effect_difference(experiment, factor, "segment")
    print(
        f"  {factor}: high_value effect={r['by_segment']['high_value']['effect']:+.2f}, "
        f"regular effect={r['by_segment']['regular']['effect']:+.2f}, "
        f"diff p={r['p_value']:.5f}  [{'DIFFERS by segment' if r['significantly_different'] else 'consistent across segments'}]"
    )
