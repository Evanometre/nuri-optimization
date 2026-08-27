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
            "se": 2 * np.sqrt(mse_pure / n),
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


def analyze_fractional_from_summary(factor_names, generator, cell_summaries, alpha=0.05):
    """Same analysis, from per-cell SUMMARY statistics (n, mean, std) instead
    of raw per-customer rows -- what a marketer would actually have from an
    analytics dashboard, and what a mobile UI can realistically accept as
    input (entering 48,000 rows by hand isn't practical).

    cell_summaries: list of {"combo": {"A": -1, ...}, "n": int, "mean": float,
    "std": float} -- one entry per treatment cell (std = sample std of the
    per-customer outcome within that cell).

    Uses weighted least squares (weight = n_i, i.e. inverse-variance
    weighting given a common per-customer variance) since unequal cell sizes
    break the simple equal-weight contrast shortcut used in analyze_fractional.
    """
    generated_factor = generator["generated_factor"]
    independent_factors = generator["from"]

    effects = list(independent_factors) + [generated_factor]
    for a, b in itertools.combinations(independent_factors, 2):
        effects.append(a + b)

    n_cells = len(cell_summaries)
    X = np.ones((n_cells, len(effects) + 1))
    for j, eff in enumerate(effects):
        X[:, j + 1] = [_coded_column(eff, cs["combo"]) for cs in cell_summaries]

    y = np.array([cs["mean"] for cs in cell_summaries])
    w = np.array([cs["n"] for cs in cell_summaries], dtype=float)
    W = np.diag(w)

    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y
    beta = np.linalg.solve(XtWX, XtWy)
    intercept = beta[0]
    effect_estimates = {eff: 2 * beta[j + 1] for j, eff in enumerate(effects)}

    total_n = sum(cs["n"] for cs in cell_summaries)
    sse_pure = sum((cs["n"] - 1) * cs["std"] ** 2 for cs in cell_summaries)
    df_pure = total_n - n_cells
    mse_pure = sse_pure / df_pure

    XtWX_inv = np.linalg.inv(XtWX)

    n_effects = len(effects)
    alpha_bonferroni = alpha / n_effects
    aliases = alias_map(independent_factors, generated_factor)

    effect_results = {}
    for j, eff in enumerate(effects):
        var_beta_j = mse_pure * XtWX_inv[j + 1, j + 1]
        var_effect_j = 4 * var_beta_j
        f_stat = (effect_estimates[eff] ** 2) / var_effect_j
        p_value = stats.f.sf(f_stat, 1, df_pure)
        effect_results[eff] = {
            "effect": effect_estimates[eff],
            "se": np.sqrt(var_effect_j),
            "f": f_stat,
            "p": p_value,
            "significant": p_value < alpha,
            "significant_bonferroni": p_value < alpha_bonferroni,
            "aliased_with": aliases.get(eff),
        }

    combo_means = {tuple(cs["combo"][f] for f in factor_names): cs["mean"] for cs in cell_summaries}
    combo_ns = {tuple(cs["combo"][f] for f in factor_names): cs["n"] for cs in cell_summaries}

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


def holdout_validation_from_summary(analysis, holdout_summary, alpha=0.05):
    """Same check as holdout_validation, from a single cell's (n, mean, std)
    summary instead of raw customer rows -- what the mobile UI has to work
    with. Uses a one-sample t-test on the summary stats directly."""
    combo = holdout_summary["combo"]
    _, _, predict = fit_reduced_model(analysis)
    predicted = predict(combo)

    n = holdout_summary["n"]
    mean = holdout_summary["mean"]
    se = holdout_summary["std"] / np.sqrt(n)
    t_stat = (mean - predicted) / se
    p_value = 2 * stats.t.sf(abs(t_stat), df=n - 1)

    return {
        "combo": combo,
        "predicted": predicted,
        "observed_mean": mean,
        "observed_se": se,
        "p_value": p_value,
        "supports_recommendation": p_value >= alpha,
    }


def segment_effect_difference_from_summary(
    factor, factor_names, generator, segment_cell_summaries, alpha=0.05
):
    """Does `factor`'s effect differ between two segments, given each
    segment's own set of 8 per-cell (n, mean, std) summaries -- what the
    mobile UI has to work with instead of raw customer rows.

    segment_cell_summaries: {segment_name: [cell_summary, ...]}, exactly 2 keys.
    """
    if len(segment_cell_summaries) != 2:
        raise ValueError("segment_effect_difference_from_summary needs exactly 2 segments")

    per_segment = {}
    for seg, summaries in segment_cell_summaries.items():
        analysis = analyze_fractional_from_summary(factor_names, generator, summaries, alpha=alpha)
        per_segment[seg] = {
            "effect": analysis["effects"][factor]["effect"],
            "se": analysis["effects"][factor]["se"],
        }

    (seg1, r1), (seg2, r2) = per_segment.items()
    diff = r1["effect"] - r2["effect"]
    se_diff = np.sqrt(r1["se"] ** 2 + r2["se"] ** 2)
    z = diff / se_diff
    p_value = 2 * stats.norm.sf(abs(z))

    return {
        "factor": factor,
        "by_segment": per_segment,
        "difference": diff,
        "z": z,
        "p_value": p_value,
        "significantly_different": p_value < alpha,
    }


def segment_effect_difference(experiment, factor, segment_key, alpha=0.05):
    """Does `factor`'s effect on contribution differ between segment values?
    Splits the primary-fraction customers by segment and refits the FULL
    7-effect model separately in each (not just a 1-factor regression) --
    same methodology as segment_effect_difference_from_summary, so both
    entry points agree exactly rather than giving two different answers to
    the same question.
    """
    customers = experiment.customers
    segments = sorted({c[segment_key] for c in customers})
    if len(segments) != 2:
        raise ValueError("segment_effect_difference currently supports exactly 2 segment values")

    results = {}
    for seg in segments:
        subset_experiment = FractionalFactorialExperiment(
            factor_names=experiment.factor_names,
            factor_labels=experiment.factor_labels,
            generator=experiment.generator,
            customers=[c for c in customers if c[segment_key] == seg],
        )
        seg_analysis = analyze_fractional(subset_experiment, alpha=alpha)
        results[seg] = {
            "effect": seg_analysis["effects"][factor]["effect"],
            "se": seg_analysis["effects"][factor]["se"],
            "n": len(subset_experiment.customers),
        }

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
