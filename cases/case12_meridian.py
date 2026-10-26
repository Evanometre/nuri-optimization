"""Meridian Fastener & Supply Co. -- working capital & distribution case (synthetic).

Toledah Foreign Enterprise Case 002. The data room is a synthetic workbook; it
is not committed to this repo. Point NURI_MERIDIAN_XLSX (or the first CLI
argument of run_case12_meridian.py) at the workbook.

Every assumption that is not in the data room is named here so it can be
challenged. Nothing below was supplied by the client.
"""

from nuri.inventory.opportunity import Scenario

# Service levels are cycle service levels (probability of no stockout in a
# replenishment cycle) by ABC class. A and B carry the strategic-account
# protection: the data room has no customer dimension on demand, so strategic
# buying cannot be mapped to SKUs; A+B SKUs are 460 items covering 95% of COGS.
BASE_CSL = {"A": 0.999, "B": 0.995, "C": 0.95}
REVIEW_DAYS = 7  # weekly replenishment review
HOLDING_RATE = 0.15  # annual carrying cost, share of inventory value (capital ~9% + handling/insurance ~6%)
AGED_LIQUIDATION_RECOVERY = 0.70  # share of cost recovered when aged excess is sold down
MANAGEMENT_LOST_SALES = 1_380_000

SCENARIOS = [
    Scenario(
        "Conservative", {**BASE_CSL, "A": 0.9995}, review_days=14, lt_sd_factor=1.5, moq_buffer=0.05,
        write_off_reduction=0.25, expedite_premium_captured=0.25,
    ),
    Scenario("Base", BASE_CSL, review_days=REVIEW_DAYS, lt_sd_factor=1.0, moq_buffer=0.05),
    Scenario(
        "Upside", BASE_CSL, review_days=REVIEW_DAYS, lt_sd_factor=0.75, moq_buffer=0.0,
        write_off_reduction=0.75, expedite_premium_captured=0.75,
    ),
]

# Service-level grid for the working-capital vs service frontier (D2).
FRONTIER = {
    "A 99.5 / B 98 / C 90": {"A": 0.995, "B": 0.98, "C": 0.90},
    "A 99.9 / B 99.5 / C 75": {"A": 0.999, "B": 0.995, "C": 0.75},
    "A 99.9 / B 99.5 / C 90": {"A": 0.999, "B": 0.995, "C": 0.90},
    "A 99.9 / B 99.5 / C 95 (base)": BASE_CSL,
    "A 99.9 / B 99.5 / C 99": {"A": 0.999, "B": 0.995, "C": 0.99},
    "A 99.95 / B 99.5 / C 95": {"A": 0.9995, "B": 0.995, "C": 0.95},
    "All 99": {"A": 0.99, "B": 0.99, "C": 0.99},
}

SENSITIVITIES = {
    "A service 99.5%": {"csl_by_class": {**BASE_CSL, "A": 0.995}},
    "A service 99.95%": {"csl_by_class": {**BASE_CSL, "A": 0.9995}},
    "C service 75%": {"csl_by_class": {**BASE_CSL, "C": 0.75}},
    "C service 99%": {"csl_by_class": {**BASE_CSL, "C": 0.99}},
    "Review every 14 days": {"review_days": 14},
    "Review every 3 days": {"review_days": 3},
    "Supplier lead-time spread x2": {"lt_sd_factor": 2.0},
    "No supplier lead-time spread": {"lt_sd_factor": 0.0},
    "Lot-size buffer 0%": {"moq_buffer": 0.0},
    "Lot-size buffer 20%": {"moq_buffer": 0.20},
}
