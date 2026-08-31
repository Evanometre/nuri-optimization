"""PrimeSack Industries -- September 2026 production decision for the 50kg
plain PP woven rice sack. Real client data (from the email), not synthetic.

NOTE ON A MISSING INPUT: the client gave the overtime cost PREMIUM (+N18/sack)
but never the base production cost per sack, which is needed to check the
N15M working-capital constraint properly. That constraint is treated as
advisory below (flagged, not silently assumed) -- see run_case8 output.
"""

# Month index 1 = Mar 2024. Season = calendar month (1=Jan..12=Dec).
HISTORICAL_DEMAND = [
    # (month_index, calendar_month, demand)
    (1, 3, 21_800),   # Mar 2024
    (2, 4, 23_400),
    (3, 5, 24_100),
    (4, 6, 23_600),
    (5, 7, 25_200),
    (6, 8, 28_700),
    (7, 9, 31_900),
    (8, 10, 34_100),
    (9, 11, 32_600),
    (10, 12, 29_800),
    (11, 1, 22_400),  # Jan 2025
    (12, 2, 21_900),
    (13, 3, 24_000),
    (14, 4, 25_100),
    (15, 5, 26_300),
    (16, 6, 25_700),
    (17, 7, 27_100),
    (18, 8, 30_600),
    (19, 9, 34_800),
    (20, 10, 36_200),
    (21, 11, 34_100),
    (22, 12, 31_500),
    (23, 1, 24_700),  # Jan 2026
    (24, 2, 24_100),
    (25, 3, 26_500),
    (26, 4, 27_400),
    (27, 5, 28_900),
    (28, 6, 28_100),
    (29, 7, 29_700),
    (30, 8, 33_400),  # Aug 2026 -- most recent
]

SEPTEMBER_MONTH_INDEX = 31
SEPTEMBER_CALENDAR_MONTH = 9

STARTING_INVENTORY = 6_500  # as of Aug 31, 2026
NORMAL_CAPACITY = 48_000  # sacks/month
OVERTIME_CAPACITY = 55_000  # sacks/month, max with overtime
OVERTIME_EXTRA_COST_PER_SACK = 18  # naira, premium over normal cost

STOCKOUT_COST_PER_SACK = 95  # naira, lost contribution + relationship risk
HOLDING_COST_PER_SACK_PER_MONTH = 7.50  # naira

MAX_ENDING_INVENTORY = 15_000  # soft warehouse-capacity cap
WORKING_CAPITAL_AVAILABLE = 15_000_000  # naira -- shared with other products

# Not provided by the client -- needed to translate the N15M working-capital
# figure into a production-quantity cap. Flagged explicitly in the report
# rather than silently guessed.
BASE_PRODUCTION_COST_PER_SACK = None


def historical_series():
    month_indices = [row[0] for row in HISTORICAL_DEMAND]
    seasons = [row[1] for row in HISTORICAL_DEMAND]
    values = [row[2] for row in HISTORICAL_DEMAND]
    return month_indices, seasons, values
