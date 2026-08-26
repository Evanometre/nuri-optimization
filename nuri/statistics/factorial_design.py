"""Analysis engine for a replicated 2^k full-factorial experiment.

Typical use: a quality/production team runs every combination of k two-level
factors (e.g. machine speed, temperature, pressure, time), replicated a few
times each, and wants to know which factors (and which interactions between
them) actually move a measured outcome -- here, a defect rate.

Because the design is replicated, we get a genuine, model-free estimate of
experimental error ("pure error") straight from the spread between repeat
runs at the same settings. That lets us run a real F-test on every main
effect and every interaction, instead of guessing which higher-order terms
are "probably just noise" (the usual workaround when there's no replication).
"""

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class FactorialExperiment:
    factor_names: list  # e.g. ["A", "B", "C", "D"]
    factor_labels: dict  # {"A": ("55 RPM", "70 RPM"), ...} (low, high)
    runs: list  # [{"A": -1, "B": 1, ..., "produced": 250, "rejected": 22}, ...]


def _effect_names(factor_names):
    """All main effects and interactions, e.g. for [A,B]: A, B, AB."""
    names = []
    for size in range(1, len(factor_names) + 1):
        for combo in itertools.combinations(factor_names, size):
            names.append("".join(combo))
    return names


def _coded_column(effect_name, run):
    value = 1
    for factor in effect_name:
        value *= run[factor]
    return value


def _logit(p, produced):
    # Continuity correction (add 0.5/1) so 0% and 100% observed rates don't
    # blow up to +-infinity -- standard practice for logit of small counts.
    adj = (p * produced + 0.5) / (produced + 1)
    return np.log(adj / (1 - adj))


def _inverse_logit(z):
    return 1 / (1 + np.exp(-z))


def analyze_2k_factorial(experiment, alpha=0.05, scale="logit"):
    """scale="logit" (default, recommended when rates span a wide range, e.g.
    <1% to >30% here) analyzes on the log-odds scale, which is the correct
    way to avoid spurious higher-order interactions that raw-proportion
    ANOVA manufactures purely from the nonlinearity near 0%/100%. scale="rate"
    runs the naive linear analysis for comparison.
    """
    factor_names = experiment.factor_names
    runs = experiment.runs
    n = len(runs)

    raw_rates = np.array([r["rejected"] / r["produced"] for r in runs])
    if scale == "logit":
        response = np.array([_logit(r["rejected"] / r["produced"], r["produced"]) for r in runs])
    elif scale == "rate":
        response = raw_rates
    else:
        raise ValueError(f"Unknown scale: {scale!r}")

    # Group run *indices* (not the run dicts) by treatment combination, so
    # pure error can be computed directly from `response` without depending
    # on runs already being sorted by combination.
    combo_indices = {}
    for i, run in enumerate(runs):
        key = tuple(run[f] for f in factor_names)
        combo_indices.setdefault(key, []).append(i)
    n_combos = len(combo_indices)

    effects = _effect_names(factor_names)
    X = np.ones((n, len(effects) + 1))
    for j, eff in enumerate(effects):
        X[:, j + 1] = [_coded_column(eff, r) for r in runs]

    beta, *_ = np.linalg.lstsq(X, response, rcond=None)
    intercept = beta[0]
    # Orthogonal +-1 coding: coefficient = effect / 2, so effect = 2 * coefficient.
    effect_estimates = {eff: 2 * beta[j + 1] for j, eff in enumerate(effects)}

    # Pure error: the spread between replicate runs at the *same* treatment
    # combination, on the analysis scale. This is a model-free estimate of
    # experimental noise.
    sse_pure = 0.0
    for indices in combo_indices.values():
        vals = response[indices]
        sse_pure += np.sum((vals - vals.mean()) ** 2)
    df_pure = n - n_combos
    mse_pure = sse_pure / df_pure

    # With 15 simultaneous tests (for k=4) at alpha=0.05, ~0.75 false
    # positives are expected by chance alone -- so also report a
    # Bonferroni-corrected view (alpha / number of effects) alongside the
    # uncorrected one, rather than taking every p<0.05 hit at face value.
    n_effects = len(effects)
    alpha_bonferroni = alpha / n_effects

    effect_results = {}
    for eff in effects:
        # Standard 2^k formula for an orthogonally-coded contrast:
        # SS = N * effect^2 / 4, with 1 degree of freedom.
        ss = n * (effect_estimates[eff] ** 2) / 4
        f_stat = ss / mse_pure
        p_value = stats.f.sf(f_stat, 1, df_pure)
        effect_results[eff] = {
            "effect": effect_estimates[eff],
            "ss": ss,
            "f": f_stat,
            "p": p_value,
            "significant": p_value < alpha,
            "significant_bonferroni": p_value < alpha_bonferroni,
        }

    # Observed mean defect rate per combination, always on the raw probability
    # scale (model-free) regardless of which scale the ANOVA ran on.
    combo_means = {key: raw_rates[indices].mean() for key, indices in combo_indices.items()}

    return {
        "scale": scale,
        "intercept": intercept,
        "effects": effect_results,
        "mse_pure": mse_pure,
        "df_pure": df_pure,
        "combo_means": combo_means,
        "factor_names": factor_names,
    }


def best_combination(analysis):
    """The treatment combination with the lowest observed mean defect rate,
    read directly from the data -- no model assumptions involved."""
    combo_means = analysis["combo_means"]
    best_key = min(combo_means, key=combo_means.get)
    return dict(zip(analysis["factor_names"], best_key)), combo_means[best_key]


def fit_reduced_model(analysis, use_bonferroni=True):
    """Refit using only the statistically significant effects -- a cross-check
    that the 'best combination' found empirically is explained by effects
    that are actually real, not by noise in one lucky replicate group.
    Predictions are always returned on the probability (defect rate) scale,
    even if the analysis itself ran on the logit scale. Defaults to the
    Bonferroni-corrected significance set, since with 15 simultaneous tests
    the uncorrected p<0.05 set risks including a false positive or two.
    """
    key_name = "significant_bonferroni" if use_bonferroni else "significant"
    significant = [eff for eff, r in analysis["effects"].items() if r[key_name]]
    predictions = {}
    for key in analysis["combo_means"]:
        run = dict(zip(analysis["factor_names"], key))
        pred = analysis["intercept"]
        for eff in significant:
            pred += (analysis["effects"][eff]["effect"] / 2) * _coded_column(eff, run)
        predictions[key] = _inverse_logit(pred) if analysis["scale"] == "logit" else pred
    return significant, predictions


def describe_combo(combo, factor_labels):
    """{'A': 1, 'B': -1} -> {'A': '70 RPM', 'B': '150C'}, using (low, high) labels."""
    return {
        factor: factor_labels[factor][0 if level == -1 else 1]
        for factor, level in combo.items()
    }


def confirmation_test(predicted_rate, confirmation_runs, alpha=0.05):
    """Does a fresh confirmation run (not used to pick the recommendation)
    support the model's predicted defect rate? Exact binomial test on the
    pooled confirmation counts against the predicted rate as the null.
    """
    total_produced = sum(r["produced"] for r in confirmation_runs)
    total_rejected = sum(r["rejected"] for r in confirmation_runs)
    observed_rate = total_rejected / total_produced

    result = stats.binomtest(total_rejected, total_produced, predicted_rate)

    return {
        "observed_rate": observed_rate,
        "predicted_rate": predicted_rate,
        "p_value": result.pvalue,
        "confidence_interval": result.proportion_ci(confidence_level=1 - alpha),
        "supports_recommendation": result.pvalue >= alpha,
    }


def explain(analysis):
    scale = analysis["scale"]
    lines = [f"Effect significance (F-test against pure error, {scale} scale):"]
    for eff, r in sorted(analysis["effects"].items(), key=lambda kv: kv[1]["p"]):
        if r["significant_bonferroni"]:
            flag = "**significant (survives Bonferroni correction)**"
        elif r["significant"]:
            flag = "significant at p<0.05, but NOT after Bonferroni correction -- treat as borderline"
        else:
            flag = "not significant"
        lines.append(
            f"  {eff}: effect={r['effect']:+.4f}, F={r['f']:.2f}, p={r['p']:.4f} ({flag})"
        )

    best_run, best_rate = best_combination(analysis)
    lines.append(f"\nBest observed combination: {best_run}, mean defect rate={best_rate:.2%}")

    significant, predictions = fit_reduced_model(analysis)
    lines.append(f"\nSignificant effects used in the reduced model: {significant}")
    best_key = tuple(best_run[f] for f in analysis["factor_names"])
    lines.append(f"Reduced-model prediction at that combination: {predictions[best_key]:.2%}")

    return "\n".join(lines)
