"""Case 4A: The Packaging Manufacturer -- a 2^4 factorial DOE case.

NOTE ON DATA: no real inspection data was provided for this case. The raw
runs below are SYNTHETIC, generated once from a hidden logistic model (see
scripts/generate_case6_data.py) with a deliberately planted structure: a
large main effect for temperature (B), a much smaller one for speed (A) on
its own, a strong A x B interaction (speed only hurts at high temperature --
matching the case narrative's own example), a moderate main effect for
pressure (C), and no real effect for sealing time (D). The analysis engine
in nuri/statistics/factorial_design.py never sees those parameters -- it has
to recover the structure from the raw counts alone, the same way it would
have to for real inspection records.

Factor coding: -1 = low level, +1 = high level.
  A: machine speed      -1 = 55 RPM,  +1 = 70 RPM
  B: sealing temperature -1 = 150C,   +1 = 170C
  C: sealing pressure    -1 = 5 bar,  +1 = 7 bar
  D: sealing time        -1 = 0.4s,   +1 = 0.7s

Each run produced 250 bags; "rejected" is the count that failed inspection.
"""

from nuri.statistics.factorial_design import FactorialExperiment

FACTOR_NAMES = ["A", "B", "C", "D"]

FACTOR_LABELS = {
    "A": ("55 RPM", "70 RPM"),
    "B": ("150C", "170C"),
    "C": ("5 bar", "7 bar"),
    "D": ("0.4s", "0.7s"),
}

# Current company-wide defect rate, per the case narrative -- a baseline
# figure from routine production, not necessarily one exact corner of the
# experimental grid (the DOE deliberately tests beyond normal operating
# settings to map out the response surface).
CURRENT_DEFECT_RATE = 0.08
MONTHLY_VOLUME = 120_000
COST_PER_DEFECT = 85  # naira

RAW_RUNS = [
    {"A": -1, "B": -1, "C": -1, "D": -1, "produced": 250, "rejected": 17},
    {"A": -1, "B": -1, "C": -1, "D": -1, "produced": 250, "rejected": 14},
    {"A": -1, "B": -1, "C": -1, "D": -1, "produced": 250, "rejected": 18},
    {"A": -1, "B": -1, "C": -1, "D": 1, "produced": 250, "rejected": 18},
    {"A": -1, "B": -1, "C": -1, "D": 1, "produced": 250, "rejected": 11},
    {"A": -1, "B": -1, "C": -1, "D": 1, "produced": 250, "rejected": 24},
    {"A": -1, "B": -1, "C": 1, "D": -1, "produced": 250, "rejected": 9},
    {"A": -1, "B": -1, "C": 1, "D": -1, "produced": 250, "rejected": 9},
    {"A": -1, "B": -1, "C": 1, "D": -1, "produced": 250, "rejected": 4},
    {"A": -1, "B": -1, "C": 1, "D": 1, "produced": 250, "rejected": 8},
    {"A": -1, "B": -1, "C": 1, "D": 1, "produced": 250, "rejected": 7},
    {"A": -1, "B": -1, "C": 1, "D": 1, "produced": 250, "rejected": 12},
    {"A": -1, "B": 1, "C": -1, "D": -1, "produced": 250, "rejected": 16},
    {"A": -1, "B": 1, "C": -1, "D": -1, "produced": 250, "rejected": 18},
    {"A": -1, "B": 1, "C": -1, "D": -1, "produced": 250, "rejected": 14},
    {"A": -1, "B": 1, "C": -1, "D": 1, "produced": 250, "rejected": 13},
    {"A": -1, "B": 1, "C": -1, "D": 1, "produced": 250, "rejected": 16},
    {"A": -1, "B": 1, "C": -1, "D": 1, "produced": 250, "rejected": 10},
    {"A": -1, "B": 1, "C": 1, "D": -1, "produced": 250, "rejected": 10},
    {"A": -1, "B": 1, "C": 1, "D": -1, "produced": 250, "rejected": 8},
    {"A": -1, "B": 1, "C": 1, "D": -1, "produced": 250, "rejected": 9},
    {"A": -1, "B": 1, "C": 1, "D": 1, "produced": 250, "rejected": 7},
    {"A": -1, "B": 1, "C": 1, "D": 1, "produced": 250, "rejected": 14},
    {"A": -1, "B": 1, "C": 1, "D": 1, "produced": 250, "rejected": 12},
    {"A": 1, "B": -1, "C": -1, "D": -1, "produced": 250, "rejected": 5},
    {"A": 1, "B": -1, "C": -1, "D": -1, "produced": 250, "rejected": 2},
    {"A": 1, "B": -1, "C": -1, "D": -1, "produced": 250, "rejected": 3},
    {"A": 1, "B": -1, "C": -1, "D": 1, "produced": 250, "rejected": 1},
    {"A": 1, "B": -1, "C": -1, "D": 1, "produced": 250, "rejected": 2},
    {"A": 1, "B": -1, "C": -1, "D": 1, "produced": 250, "rejected": 4},
    {"A": 1, "B": -1, "C": 1, "D": -1, "produced": 250, "rejected": 2},
    {"A": 1, "B": -1, "C": 1, "D": -1, "produced": 250, "rejected": 4},
    {"A": 1, "B": -1, "C": 1, "D": -1, "produced": 250, "rejected": 1},
    {"A": 1, "B": -1, "C": 1, "D": 1, "produced": 250, "rejected": 1},
    {"A": 1, "B": -1, "C": 1, "D": 1, "produced": 250, "rejected": 2},
    {"A": 1, "B": -1, "C": 1, "D": 1, "produced": 250, "rejected": 1},
    {"A": 1, "B": 1, "C": -1, "D": -1, "produced": 250, "rejected": 80},
    {"A": 1, "B": 1, "C": -1, "D": -1, "produced": 250, "rejected": 80},
    {"A": 1, "B": 1, "C": -1, "D": -1, "produced": 250, "rejected": 83},
    {"A": 1, "B": 1, "C": -1, "D": 1, "produced": 250, "rejected": 95},
    {"A": 1, "B": 1, "C": -1, "D": 1, "produced": 250, "rejected": 87},
    {"A": 1, "B": 1, "C": -1, "D": 1, "produced": 250, "rejected": 94},
    {"A": 1, "B": 1, "C": 1, "D": -1, "produced": 250, "rejected": 51},
    {"A": 1, "B": 1, "C": 1, "D": -1, "produced": 250, "rejected": 45},
    {"A": 1, "B": 1, "C": 1, "D": -1, "produced": 250, "rejected": 50},
    {"A": 1, "B": 1, "C": 1, "D": 1, "produced": 250, "rejected": 37},
    {"A": 1, "B": 1, "C": 1, "D": 1, "produced": 250, "rejected": 51},
    {"A": 1, "B": 1, "C": 1, "D": 1, "produced": 250, "rejected": 53},
]

# A later confirmation run at the settings the analysis recommends
# (A=1, B=-1, C=1, D=1), generated separately from the same hidden process
# (not reused from RAW_RUNS) -- a genuine out-of-sample check, not a re-report
# of data already used to pick the recommendation.
CONFIRMATION_RUNS = [
    {"A": 1, "B": -1, "C": 1, "D": 1, "produced": 250, "rejected": 2},
    {"A": 1, "B": -1, "C": 1, "D": 1, "produced": 250, "rejected": 2},
    {"A": 1, "B": -1, "C": 1, "D": 1, "produced": 250, "rejected": 2},
    {"A": 1, "B": -1, "C": 1, "D": 1, "produced": 250, "rejected": 5},
    {"A": 1, "B": -1, "C": 1, "D": 1, "produced": 250, "rejected": 2},
    {"A": 1, "B": -1, "C": 1, "D": 1, "produced": 250, "rejected": 2},
]


def packaging_defects_experiment():
    return FactorialExperiment(
        factor_names=FACTOR_NAMES, factor_labels=FACTOR_LABELS, runs=RAW_RUNS
    )
