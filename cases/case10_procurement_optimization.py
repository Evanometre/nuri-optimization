"""Case: a manufacturer spending ~N600M/year on procurement across 6
categories and 12 suppliers (2 per category, deliberately small so the
result stays checkable -- see the conversation this was built in for why
40 suppliers x 12 categories was rejected in favor of a tighter case).

NOTE ON DATA: illustrative, not a real client's numbers. Built to
demonstrate the methodology; the same model runs unchanged on a real
procurement file (same shape: one row per supplier-category offer).

Each category has one "incumbent" supplier (the naive status-quo choice --
what a decentralized, unoptimized purchasing process actually lands on) and
one "alternative" with different tradeoffs (cheaper unit price, different
freight, MOQ, rejection rate, or payment terms). In two categories, the
alternative's capacity alone can't cover full demand -- multi-sourcing is
sometimes a genuine necessity, not just a risk-diversification nicety.
"""

from nuri.procurement import ProcurementProblem, SupplierOffer

DEMAND = {
    "packaging_materials": 500_000,
    "raw_materials": 300_000,
    "maintenance_parts": 20_000,
    "fuel": 200_000,
    "logistics_services": 10_000,
    "office_supplies": 50_000,
}

OFFERS = [
    # category, price, freight, moq, capacity, rejection_rate, payment_term_days
    # packaging_materials and maintenance_parts/fuel/office_supplies: the
    # alternative CAN cover full demand alone, so "best single source" is a
    # genuine preference comparison, not forced by capacity.
    SupplierOffer("S1_incumbent", "packaging_materials", 400, 20, 50_000, 500_000, 0.02, 30),
    SupplierOffer("S2_alt", "packaging_materials", 370, 35, 100_000, 550_000, 0.03, 45),

    # raw_materials: the cheaper alternative's capacity (200k) is BELOW full
    # demand (300k) -- multi-sourcing here is a real necessity, not just a
    # risk-diversification nicety.
    SupplierOffer("S3_incumbent", "raw_materials", 600, 15, 30_000, 300_000, 0.01, 30),
    SupplierOffer("S4_alt", "raw_materials", 560, 40, 150_000, 200_000, 0.04, 60),

    SupplierOffer("S5_incumbent", "maintenance_parts", 3_000, 100, 2_000, 20_000, 0.01, 30),
    SupplierOffer("S6_alt", "maintenance_parts", 2_850, 150, 5_000, 22_000, 0.02, 30),

    SupplierOffer("S7_incumbent", "fuel", 350, 5, 20_000, 200_000, 0.00, 15),
    SupplierOffer("S8_alt", "fuel", 335, 8, 50_000, 220_000, 0.00, 30),

    # logistics_services: same genuine-capacity-shortfall story as raw_materials.
    SupplierOffer("S9_incumbent", "logistics_services", 6_000, 0, 500, 10_000, 0.00, 30),
    SupplierOffer("S10_alt", "logistics_services", 5_700, 0, 3_000, 7_000, 0.00, 45),

    SupplierOffer("S11_incumbent", "office_supplies", 600, 10, 5_000, 50_000, 0.01, 30),
    SupplierOffer("S12_alt", "office_supplies", 560, 25, 10_000, 55_000, 0.02, 30),
]

INCUMBENT_BY_CATEGORY = {
    "packaging_materials": "S1_incumbent",
    "raw_materials": "S3_incumbent",
    "maintenance_parts": "S5_incumbent",
    "fuel": "S7_incumbent",
    "logistics_services": "S9_incumbent",
    "office_supplies": "S11_incumbent",
}

BUDGET = 650_000_000  # generous -- not expected to bind
# With only 2 suppliers per category, a cap below 50% is infeasible (2 x cap
# must reach >=100% of demand). 60% is the tightest meaningfully-diversifying
# cap that still guarantees feasibility here.
CONCENTRATION_CAP = 0.60
COST_OF_CAPITAL_RATE = 0.20  # annual, for valuing payment-term financing benefit


def base_problem(concentration_cap=CONCENTRATION_CAP, single_source_only=False):
    return ProcurementProblem(
        demand=DEMAND,
        offers=OFFERS,
        budget=BUDGET,
        concentration_cap=concentration_cap,
        cost_of_capital_rate=COST_OF_CAPITAL_RATE,
        single_source_only=single_source_only,
    )
