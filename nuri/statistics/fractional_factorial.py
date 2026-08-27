"""Analysis engine for a 2^(k-1) half-fraction factorial experiment with a
continuous, per-unit outcome (e.g. incremental contribution per customer),
observed on many units per treatment cell.

Full 2^4 factorial (nuri/statistics/factorial_design.py) needs all 16
combinations run. When that's too expensive/slow to deploy (16 marketing
campaigns, 16 machine configurations...), a half-fraction runs only 8,
chosen so every main effect is still estimable cleanly. The cost: two-way
interactions become "aliased" -- e.g. AB and CD are mathematically identical
in an 8-run half-fraction built with generator D=ABC, so the data can't tell
you which one (or what mixture of both) you're actually looking at. A
correct report says so; it doesn't pretend to have resolved it.

Because each cell here has thousands of customers rather than a handful of
replicate runs, "pure error" comes from the within-cell spread across
customers -- the same idea as nuri/statistics/factorial_design.py, just with
far more degrees of freedom.
"""

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class FractionalFactorialExperiment:
    factor_names: list  # e.g. ["A", "B", "C", "D"], D is the generated factor
    factor_labels: dict  # {"A": ("low desc", "high desc"), ...}
    generator: dict  # {"generated_factor": "D", "from": ["A", "B", "C"]}
    customers: list  # [{"A": -1, ..., "contribution": float, "segment": str}, ...]
    holdout_customers: list = field(default_factory=list)  # same shape, one combo


def half_fraction_combos(independent_factors, generated_factor, sign=1):
    """All 2^(k-1) combos of the independent factors, with the generated
    factor set to sign * product(independent factors)."""
    combos = []
    for values in itertools.product([-1, 1], repeat=len(independent_factors)):
        combo = dict(zip(independent_factors, values))
        combo[generated_factor] = sign * int(np.prod(values))
        combos.append(combo)
    return combos


def alias_map(independent_factors, generated_factor):
    """Which other contrast each two-way interaction here is aliased with.
    E.g. for independent=[A,B,C], generated=D (D=ABC): AB is aliased with CD,
    AC with BD, AD with BC, and each main effect is aliased with the
    three-way interaction of the other three (rarely a real concern)."""
    aliases = {}
    for factor in independent_factors + [generated_factor]:
        others = "".join(f for f in independent_factors + [generated_factor] if f != factor)
        aliases[factor] = "".join(sorted(others))  # 3-way alias, e.g. A ~ BCD
    for a, b in itertools.combinations(independent_factors, 2):
        remaining = [f for f in independent_factors + [generated_factor] if f not in (a, b)]
        aliases[a + b] = "".join(remaining)  # e.g. AB ~ CD
    return aliases


def _coded_column(effect_name, row):
    value = 1
    for factor in effect_name:
        value *= row[factor]
    return value


def analyze_fractional(experiment, alpha=0.05):
    factor_names = experiment.factor_names
    generated_factor = experiment.generator["generated_factor"]
    independent_factors = experiment.generator["from"]
    customers = experiment.customers
    n = len(customers)

    contributions = np.array([c["contribution"] for c in customers])

    effects = list(independent_factors) + [generated_factor]
    for a, b in itertools.combinations(independent_factors, 2):
        effects.append(a + b)

    X = np.ones((n, len(effects) + 1))
    for j, eff in enumerate(effects):
        X[:, j + 1] = [_coded_column(eff, c) for c in customers]

    beta, *_ = np.linalg.lstsq(X, contributions, rcond=None)
    intercept = beta[0]
    effect_estimates = {eff: 2 * beta[j + 1] for j, eff in enumerate(effects)}

    combo_indices = {}
    for i, c in enumerate(customers):
        key = tuple(c[f] for f in factor_names)
        combo_indices.setdefault(key, []).append(i)
    n_combos = len(combo_indices)

    sse_pure = 0.0
    for indices in combo_indices.values():
        vals = contributions[np.array(indices)]
        sse_pure += np.sum((vals - vals.mean()) ** 2)
    df_pure = n - n_combos
    mse_pure = sse_pure / df_pure

    n_effects = len(effects)
    alpha_bonferroni = alpha / n_effects
    aliases = alias_map(independent_factors, generated_factor)

    effect_results = {}
    for eff in effects:
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
            "aliased_with": aliases.get(eff),
        }

    combo_means = {key: contributions[np.array(idx)].mean() for key, idx in combo_indices.items()}
    combo_ns = {key: len(idx) for key, idx in combo_indices.items()}

    return {
        "intercept": intercept,
        "effects": effect_results,
        "mse_pure": mse_pure,
        "df_pure": df_pure,
        "combo_means": combo_means,
        "combo_ns": combo_ns,
        "factor_names": factor_names,
        "independent_factors": independent_factors,
        "generated_factor": generated_factor,
    }


def best_combination(analysis):
    combo_means = analysis["combo_means"]
    best_key = max(combo_means, key=combo_means.get)  # maximizing contribution, not minimizing
    return dict(zip(analysis["factor_names"], best_key)), combo_means[best_key]


def fit_reduced_model(analysis, use_bonferroni=True):
    key_name = "significant_bonferroni" if use_bonferroni else "significant"
    significant = [eff for eff, r in analysis["effects"].items() if r[key_name]]
    factor_names = analysis["factor_names"]

    def predict(combo):
        pred = analysis["intercept"]
        for eff in significant:
            pred += (analysis["effects"][eff]["effect"] / 2) * _coded_column(eff, combo)
        return pred

    predictions = {key: predict(dict(zip(factor_names, key))) for key in analysis["combo_means"]}
    return significant, predictions, predict


def holdout_validation(analysis, holdout_customers, alpha=0.05):
    """Predict the held-out combination's mean contribution from the reduced
    model, then check it against the actual (never-used-for-fitting) data."""
    if not holdout_customers:
        return None

    factor_names = analysis["factor_names"]
    combo = {f: holdout_customers[0][f] for f in factor_names}
    _, _, predict = fit_reduced_model(analysis)
    predicted = predict(combo)

    observed = np.array([c["contribution"] for c in holdout_customers])
    t_stat, p_value = stats.ttest_1samp(observed, predicted)

    return {
        "combo": combo,
        "predicted": predicted,
        "observed_mean": observed.mean(),
        "observed_se": observed.std(ddof=1) / np.sqrt(len(observed)),
        "p_value": p_value,
        "supports_recommendation": p_value >= alpha,
    }


def segment_effect_difference(experiment, factor, segment_key, alpha=0.05):
    """Does `factor`'s effect on contribution differ between segment values?
    Splits the primary-fraction customers by segment, estimates the factor's
    contrast effect separately in each, and tests the difference.
    """
    customers = experiment.customers
    segments = sorted({c[segment_key] for c in customers})
    if len(segments) != 2:
        raise ValueError("segment_effect_difference currently supports exactly 2 segment values")

    results = {}
    for seg in segments:
        subset = [c for c in customers if c[segment_key] == seg]
        contributions = np.array([c["contribution"] for c in subset])
        coded = np.array([_coded_column(factor, c) for c in subset])
        n = len(subset)

        # Effect = 2 * (mean at +1 minus overall mean scaled) -- equivalently
        # 2x the OLS coefficient on the coded column, same shortcut as above.
        X = np.column_stack([np.ones(n), coded])
        beta, *_ = np.linalg.lstsq(X, contributions, rcond=None)
        effect = 2 * beta[1]

        combo_indices = {}
        for i, c in enumerate(subset):
            key = tuple(c[f] for f in experiment.factor_names)
            combo_indices.setdefault(key, []).append(i)
        sse = sum(
            np.sum((contributions[np.array(idx)] - contributions[np.array(idx)].mean()) ** 2)
            for idx in combo_indices.values()
        )
        df = n - len(combo_indices)
        mse = sse / df
        se_effect = 2 * np.sqrt(mse / n)  # effect = 2*coef, Var(coef) approx mse/n for a +-1 column

        results[seg] = {"effect": effect, "se": se_effect, "n": n}

    (seg1, r1), (seg2, r2) = results.items()
    diff = r1["effect"] - r2["effect"]
    se_diff = np.sqrt(r1["se"] ** 2 + r2["se"] ** 2)
    z = diff / se_diff
    p_value = 2 * stats.norm.sf(abs(z))

    return {
        "factor": factor,
        "by_segment": results,
        "difference": diff,
        "z": z,
        "p_value": p_value,
        "significantly_different": p_value < alpha,
    }
