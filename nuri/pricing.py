"""Subscription pricing and tiering: segment-level demand curves, single-price
optimization, and tier-menu design under business constraints.

Two different demand signals get used for two different questions, on
purpose -- conflating them would be a mistake:

- The observational price-test table (new signups/month at various prices)
  measures ACQUISITION elasticity -- how fast new customers show up. New
  customers comparison-shop with no switching cost, so this is typically
  more elastic than the response of an existing installed base.
- The given segment elasticities (student/professional response to a 10%
  price change) describe how the EXISTING subscriber base's demand responds
  to price. That's what matters for repricing the current 8,000 subscribers,
  so it's the primary driver of the demand model here. The acquisition table
  is used as a secondary cross-check, not silently blended in.
"""

from dataclasses import dataclass


@dataclass
class SegmentDemand:
    name: str
    base_price: float
    base_quantity: float
    elasticity: float  # e.g. -1.2 means a 1% price rise -> 1.2% fewer subscribers

    def quantity(self, price):
        return self.base_quantity * (price / self.base_price) ** self.elasticity


def fit_power_law(prices, quantities):
    """Log-log OLS fit -> (scale, elasticity) such that quantity = scale * price**elasticity."""
    import numpy as np
    log_p = np.log(prices)
    log_q = np.log(quantities)
    b, log_a = np.polyfit(log_p, log_q, 1)
    return float(np.exp(log_a)), float(b)


def fit_linear(x, y):
    import numpy as np
    m, c = np.polyfit(x, y, 1)
    return float(m), float(c)


def single_price_outcome(price, segments, variable_cost):
    total_qty = sum(s.quantity(price) for s in segments)
    revenue = price * total_qty
    cost = variable_cost * total_qty
    contribution = revenue - cost
    margin = contribution / revenue if revenue else 0
    return {
        "price": price,
        "subscribers": total_qty,
        "revenue": revenue,
        "variable_cost": cost,
        "contribution": contribution,
        "contribution_margin": margin,
        "by_segment": {s.name: s.quantity(price) for s in segments},
    }


def enumerate_tier_menus(
    segments, basic_grid, plus_grid, pro_grid, feature_factors, variable_cost, tier_order,
    min_margin, min_spacing, max_basic_price, max_plus_price_for_existing_customers,
    min_professional_segment_reach_on_pro,
):
    """All Basic x Plus x Pro combinations from the given grids, each tagged
    with feasibility against every named constraint (not just a pass/fail --
    which specific constraint(s) failed, for an honest infeasibility report)."""
    import itertools

    results = []
    professional_base = next(s for s in segments if s.name == "professionals").base_quantity

    for basic, plus, pro in itertools.product(basic_grid, plus_grid, pro_grid):
        prices = {"Basic": basic, "Plus": plus, "Pro": pro}
        outcome = tier_scenario_outcome(segments, prices, feature_factors, variable_cost, tier_order)

        margins_ok = all((p - variable_cost) / p >= min_margin for p in prices.values())
        spacing_ok = (plus >= min_spacing * basic) and (pro >= min_spacing * plus)
        basic_ok = basic <= max_basic_price
        existing_customer_ok = plus <= max_plus_price_for_existing_customers
        pro_reach = outcome["by_segment"]["professionals"]["Pro"] / professional_base
        reach_ok = pro_reach >= min_professional_segment_reach_on_pro

        failed = [
            name for name, ok in [
                ("margin", margins_ok), ("spacing", spacing_ok), ("basic_price_cap", basic_ok),
                ("existing_customer_cap", existing_customer_ok), ("professional_pro_reach", reach_ok),
            ] if not ok
        ]

        results.append({
            "prices": prices,
            "feasible": len(failed) == 0,
            "failed_constraints": failed,
            "professional_pro_reach": pro_reach,
            "outcome": outcome,
        })

    return results


def optimize_single_price(segments, variable_cost, price_grid, min_margin=None, max_price=None):
    """Grid search (the demand curve is smooth but not concave in general
    once margin/price caps are added, so grid search over a fine grid is more
    transparent and robust than trusting a solver's local optimum here)."""
    best = None
    for price in price_grid:
        outcome = single_price_outcome(price, segments, variable_cost)
        if min_margin is not None and outcome["contribution_margin"] < min_margin:
            continue
        if max_price is not None and price > max_price:
            continue
        if best is None or outcome["contribution"] > best["contribution"]:
            best = outcome
    return best


def tier_scenario_outcome(segments, tier_prices, feature_factors, variable_cost, tier_order):
    """Nested willingness-to-pay demand model: for each segment, each tier's
    quality-adjusted price (price / feature_factor) defines a WTP threshold.
    'Potential subscribers for tier T' = how many in the segment would pay
    at least T's effective price, per the segment's own elasticity curve.
    Higher tiers demand more (higher effective price), so their potential
    pool is a SUBSET of a lower tier's pool -- subtracting consecutive
    thresholds partitions the segment into "who actually lands on which
    tier" without an individual choosing more than one tier and without the
    all-or-nothing degeneracy of a pure best-single-tier rule.

    tier_order: tiers from lowest to highest commitment, e.g. ["Basic","Plus","Pro"].
    """
    subs_by_tier = {t: 0.0 for t in tier_prices}
    by_segment = {}

    for segment in segments:
        effective_prices = {t: tier_prices[t] / feature_factors[(segment.name, t)] for t in tier_order}
        # Potential pool per tier (monotonically non-increasing if effective
        # prices are non-decreasing along tier_order, which sensible feature
        # factors should ensure -- guarded with cummin just in case).
        potential = {t: segment.quantity(effective_prices[t]) for t in tier_order}
        running_min = float("inf")
        for t in tier_order:
            running_min = min(running_min, potential[t])
            potential[t] = running_min

        segment_by_tier = {}
        for i, t in enumerate(tier_order):
            higher_pool = potential[tier_order[i + 1]] if i + 1 < len(tier_order) else 0.0
            qty = max(0.0, potential[t] - higher_pool)
            subs_by_tier[t] += qty
            segment_by_tier[t] = qty
        by_segment[segment.name] = segment_by_tier

    revenue = sum(tier_prices[t] * q for t, q in subs_by_tier.items())
    total_subs = sum(subs_by_tier.values())
    cost = variable_cost * total_subs
    contribution = revenue - cost
    margin_by_tier = {
        t: (tier_prices[t] - variable_cost) / tier_prices[t] for t in tier_prices
    }

    return {
        "tier_prices": dict(tier_prices),
        "subs_by_tier": subs_by_tier,
        "by_segment": by_segment,
        "total_subscribers": total_subs,
        "revenue": revenue,
        "variable_cost": cost,
        "contribution": contribution,
        "margin_by_tier": margin_by_tier,
    }
