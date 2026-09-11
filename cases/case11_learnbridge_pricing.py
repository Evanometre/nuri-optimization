"""LearnBridge EdTech pricing case: single-price vs. three-tier restructure.

NOTE ON THE SEGMENT ELASTICITIES: given as "demand response to a 10% price
increase" -- i.e. only locally calibrated around the current N1,499 price.
Treated here as constant-elasticity (power-law) demand for tractability, but
extrapolating far from N1,499 (especially the Pro candidate range,
N2,199-2,599, +47% to +73% above anchor) is a real extrapolation risk, not
a precise prediction -- flagged explicitly in run_case11's output rather
than presented as equally confident everywhere.

NOTE ON FEATURE VALUE FACTORS: the case gives qualitative statements
("professionals care more about mentor access...") but no numbers for how
much more. FEATURE_FACTORS below is a disclosed modeling assumption (not
derived from data) translating that into a quality-adjusted-price discount
per segment per tier. Swap in real conjoint/survey data if available.
"""

from nuri.pricing import SegmentDemand

CURRENT_PRICE = 1499
CURRENT_SUBSCRIBERS = 8000
VARIABLE_COST = 410
CURRENT_CHURN = 0.058

STUDENT_SHARE = 0.65
PROFESSIONAL_SHARE = 0.35

STUDENT_ELASTICITY = -1.2  # -12% demand for +10% price
PROFESSIONAL_ELASTICITY = -0.5  # -5% demand for +10% price

OBSERVATIONAL_PRICES = [1199, 1299, 1399, 1499, 1599, 1699, 1799, 1899]
OBSERVATIONAL_NEW_SUBS = [2350, 2180, 2040, 1920, 1770, 1610, 1430, 1250]
OBSERVATIONAL_CHURN = [0.047, 0.049, 0.052, 0.058, 0.061, 0.067, 0.074, 0.083]

BASIC_GRID = [999, 1099, 1199]
PLUS_GRID = [1499, 1599, 1699]
PRO_GRID = [2199, 2399, 2599]
TIER_ORDER = ["Basic", "Plus", "Pro"]

MIN_MARGIN = 0.65
MIN_TIER_SPACING = 1.25  # Plus >= 1.25x Basic, Pro >= 1.25x Plus
MAX_EXISTING_CUSTOMER_INCREASE = 0.10
MIN_PROFESSIONAL_PRO_REACH = 0.20
MAX_BASIC_PRICE = 1199

# Disclosed assumption -- see module docstring.
FEATURE_FACTORS = {
    ("students", "Basic"): 0.78,
    ("students", "Plus"): 0.95,
    ("students", "Pro"): 1.00,
    ("professionals", "Basic"): 0.55,
    ("professionals", "Plus"): 0.75,
    ("professionals", "Pro"): 1.00,
}


def segments():
    return [
        SegmentDemand("students", CURRENT_PRICE, CURRENT_SUBSCRIBERS * STUDENT_SHARE, STUDENT_ELASTICITY),
        SegmentDemand("professionals", CURRENT_PRICE, CURRENT_SUBSCRIBERS * PROFESSIONAL_SHARE, PROFESSIONAL_ELASTICITY),
    ]
