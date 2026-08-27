"""Case 4B: ChopRun -- a 2^(4-1) fractional factorial promo experiment.

NOTE ON DATA: no real experiment data exists yet for this case. Customer-level
records below are SYNTHETIC, generated once (fixed seed -> fully
reproducible, same numbers every run) from a hidden model with a
deliberately planted structure:
  - discount size (A) has a real, large effect on reorder probability, but
    it costs money every time it's redeemed -- the case narrative's "increases
    repeat orders substantially but costs too much to be profitable for
    everyone" is baked in on purpose.
  - timing (B): 7 days genuinely outperforms 3 days.
  - spread (C): platform-wide vs selected-restaurants has ~no effect on
    reorder probability, but platform-wide carries a flat operational
    overhead cost per eligible customer -- so it's a pure cost with no
    matching benefit, exactly the "platform-wide isn't necessary" narrative.
  - message (D): personalization has a modest average effect, but a MUCH
    larger effect specifically for the "high_value" customer segment --
    testing whether the engine's segment-heterogeneity check finds this.

The analysis engine (nuri/statistics/fractional_factorial.py) never sees
these parameters -- it has to recover the structure from customer-level
contribution figures alone, same as it would from real ChopRun data.

Factor coding: -1 = option A (first listed), +1 = option B (second listed).
  A: discount      -1 = N500 off,  +1 = N1000 off
  B: timing         -1 = 3 days,    +1 = 7 days
  C: spread         -1 = selected restaurants, +1 = platform-wide
  D: message        -1 = generic ("N1000 off your next order"), +1 = personalized

Design: half-fraction with generator D = A*B*C (8 primary combinations,
5,625 customers each = 45,000). One combination from the complementary half
(A=-1, B=1, C=-1, D=-1) is held out for validation, 3,000 customers -- total
48,000, matching the case narrative.
"""

import numpy as np

from nuri.statistics.fractional_factorial import FractionalFactorialExperiment, half_fraction_combos

FACTOR_NAMES = ["A", "B", "C", "D"]
INDEPENDENT_FACTORS = ["A", "B", "C"]
GENERATED_FACTOR = "D"
GENERATOR = {"generated_factor": GENERATED_FACTOR, "from": INDEPENDENT_FACTORS}

FACTOR_LABELS = {
    "A": ("N500 off", "N1000 off"),
    "B": ("3 days", "7 days"),
    "C": ("selected restaurants", "platform-wide"),
    "D": ("generic message", "personalized message"),
}

CUSTOMERS_PER_PRIMARY_COMBO = 5_625  # 8 x 5625 = 45,000
HOLDOUT_COMBO = {"A": -1, "B": 1, "C": -1, "D": -1}
CUSTOMERS_IN_HOLDOUT = 3_000  # 45,000 + 3,000 = 48,000

AVG_ORDER_GROSS_PROFIT = 1_800  # naira, per reorder
PLATFORM_WIDE_OVERHEAD = 50  # naira, flat per eligible customer if C=platform-wide
HIGH_VALUE_SEGMENT_SHARE = 0.30


def _true_reorder_probability(A, B, C, D, segment):
    intercept = np.log(0.20 / 0.80)
    beta_A = 0.35
    beta_B = 0.15
    beta_C = 0.05
    beta_D_base = 0.10
    beta_D_high_value_extra = 0.35  # personalization matters much more here
    beta_D = beta_D_base + (beta_D_high_value_extra if segment == "high_value" else 0)
    z = intercept + beta_A * A + beta_B * B + beta_C * C + beta_D * D
    return 1 / (1 + np.exp(-z))


def _discount_amount(A):
    return 500 if A == -1 else 1000


def _simulate_customers(combo, n, rng):
    A, B, C, D = combo["A"], combo["B"], combo["C"], combo["D"]
    segments = rng.choice(
        ["high_value", "regular"], size=n, p=[HIGH_VALUE_SEGMENT_SHARE, 1 - HIGH_VALUE_SEGMENT_SHARE]
    )
    customers = []
    for segment in segments:
        p = _true_reorder_probability(A, B, C, D, segment)
        reordered = rng.random() < p
        if reordered:
            gross_profit = max(0.0, rng.normal(AVG_ORDER_GROSS_PROFIT, 300))
            promo_cost = _discount_amount(A)
        else:
            gross_profit = 0.0
            promo_cost = 0.0
        overhead = PLATFORM_WIDE_OVERHEAD if C == 1 else 0
        contribution = gross_profit - promo_cost - overhead
        customers.append(
            {
                "A": A, "B": B, "C": C, "D": D,
                "segment": segment,
                "reordered": bool(reordered),
                "contribution": float(contribution),
            }
        )
    return customers


def choprun_experiment(seed=7):
    rng = np.random.default_rng(seed)

    primary_combos = half_fraction_combos(INDEPENDENT_FACTORS, GENERATED_FACTOR, sign=1)
    customers = []
    for combo in primary_combos:
        customers.extend(_simulate_customers(combo, CUSTOMERS_PER_PRIMARY_COMBO, rng))

    holdout_customers = _simulate_customers(HOLDOUT_COMBO, CUSTOMERS_IN_HOLDOUT, rng)

    return FractionalFactorialExperiment(
        factor_names=FACTOR_NAMES,
        factor_labels=FACTOR_LABELS,
        generator=GENERATOR,
        customers=customers,
        holdout_customers=holdout_customers,
    )
