from cases.case11_learnbridge_pricing import (
    segments, VARIABLE_COST, CURRENT_PRICE, CURRENT_SUBSCRIBERS, CURRENT_CHURN,
    BASIC_GRID, PLUS_GRID, PRO_GRID, FEATURE_FACTORS, TIER_ORDER,
    MIN_MARGIN, MIN_TIER_SPACING, MAX_BASIC_PRICE, MAX_EXISTING_CUSTOMER_INCREASE,
    MIN_PROFESSIONAL_PRO_REACH, OBSERVATIONAL_PRICES, OBSERVATIONAL_NEW_SUBS, OBSERVATIONAL_CHURN,
)
from nuri.pricing import single_price_outcome, enumerate_tier_menus, fit_power_law, fit_linear

print("=== Cross-check: observational pricing-test table vs segment elasticities ===")
_, acq_elasticity = fit_power_law(OBSERVATIONAL_PRICES, OBSERVATIONAL_NEW_SUBS)
churn_slope, churn_intercept = fit_linear(OBSERVATIONAL_PRICES, OBSERVATIONAL_CHURN)
blended_segment_elasticity = 0.65 * -1.2 + 0.35 * -0.5
print(f"Acquisition elasticity (new-signup table): {acq_elasticity:.3f}")
print(f"Blended segment elasticity (student/professional weighted): {blended_segment_elasticity:.3f}")
print("These differ, as expected: acquisition (new signups) is typically more elastic than")
print("an existing base's demand response, since new customers have no switching cost.")
print(f"Churn model: churn = {churn_intercept:.5f} + {churn_slope:.8f} * price")

print()
print("=== A/B. Single price: sanity check against given current state ===")
current = single_price_outcome(CURRENT_PRICE, segments(), VARIABLE_COST)
print(f"Subscribers: {current['subscribers']:.0f} (given: {CURRENT_SUBSCRIBERS})")
print(f"Contribution: N{current['contribution']:,.0f} (given: N8,712,000)")

print()
print("=== Single-price contribution across the credible range ===")
print("(monotonically increasing -- professionals are inelastic, so the demand curve")
print(" alone has no interior max; the 10% existing-customer cap is the real limit)")
max_single_price = CURRENT_PRICE * (1 + MAX_EXISTING_CUSTOMER_INCREASE)
for p in [1199, 1399, CURRENT_PRICE, 1599, round(max_single_price)]:
    o = single_price_outcome(p, segments(), VARIABLE_COST)
    print(f"  N{p}: subs={o['subscribers']:.0f}  contribution=N{o['contribution']:,.0f}  margin={o['contribution_margin']:.1%}")

single_price_recommendation = single_price_outcome(max_single_price, segments(), VARIABLE_COST)
print(f"\nSingle-price recommendation (capped by the 10% transition rule): N{max_single_price:.0f}")
print(f"  Contribution: N{single_price_recommendation['contribution']:,.0f} "
      f"(+{(single_price_recommendation['contribution']/current['contribution']-1):.1%} vs current)")

print()
print("=== C/D. Tiered pricing: enumerate all 27 candidate menus ===")
max_plus_for_existing = CURRENT_PRICE * (1 + MAX_EXISTING_CUSTOMER_INCREASE)
results = enumerate_tier_menus(
    segments(), BASIC_GRID, PLUS_GRID, PRO_GRID, FEATURE_FACTORS, VARIABLE_COST, TIER_ORDER,
    MIN_MARGIN, MIN_TIER_SPACING, MAX_BASIC_PRICE, max_plus_for_existing, MIN_PROFESSIONAL_PRO_REACH,
)
feasible = [r for r in results if r["feasible"]]
print(f"{len(feasible)} of {len(results)} candidate menus feasible")
feasible.sort(key=lambda r: -r["outcome"]["contribution"])
print("\nAll feasible menus, ranked by contribution:")
for r in feasible:
    o = r["outcome"]
    tiers = {k: round(v) for k, v in o["subs_by_tier"].items()}
    print(f"  {r['prices']}: contribution=N{o['contribution']:,.0f}  by_tier={tiers}  professional Pro reach={r['professional_pro_reach']:.1%}")

top_pick = feasible[0]
conservative = [r for r in feasible if r["prices"]["Pro"] == min(PRO_GRID)]
conservative_pick = max(conservative, key=lambda r: r["outcome"]["contribution"])

print()
print("=== EXTRAPOLATION-RISK WARNING ===")
print("Segment elasticities were only calibrated for a 10% price move (anchor N1,499).")
print(f"Top-ranked menu prices Pro at N{top_pick['prices']['Pro']} "
      f"(+{top_pick['prices']['Pro']/CURRENT_PRICE-1:.0%} vs anchor) -- a large extrapolation.")
print(f"Recommended PILOT menu instead prices Pro at the grid's lowest option, "
      f"N{conservative_pick['prices']['Pro']} (+{conservative_pick['prices']['Pro']/CURRENT_PRICE-1:.0%} vs anchor),")
print("closer to the tested range, with materially less model risk.")

print()
print("=== Recommendation table ===")
rec = conservative_pick["outcome"]
print(f"{'':25s}{'Current':>15s}{'Recommended (pilot)':>22s}")
print(f"{'Subscribers':25s}{CURRENT_SUBSCRIBERS:>15,.0f}{rec['total_subscribers']:>22,.0f}")
print(f"{'Monthly revenue':25s}N{current['revenue']:>14,.0f}N{rec['revenue']:>21,.0f}")
print(f"{'Variable cost':25s}N{current['variable_cost']:>14,.0f}N{rec['variable_cost']:>21,.0f}")
print(f"{'Contribution':25s}N{current['contribution']:>14,.0f}N{rec['contribution']:>21,.0f}")
print(f"{'Contribution margin':25s}{current['contribution_margin']:>15.1%}{rec['contribution']/rec['revenue']:>22.1%}")
print(f"{'Churn (current model)':25s}{CURRENT_CHURN:>15.1%}{'n/a (see note)':>22s}")

print()
print(f"Net contribution gain vs current: N{rec['contribution']-current['contribution']:,.0f}/month "
      f"({rec['contribution']/current['contribution']-1:.1%})")
print(f"Net contribution gain vs single-price recommendation: N{rec['contribution']-single_price_recommendation['contribution']:,.0f}/month")
