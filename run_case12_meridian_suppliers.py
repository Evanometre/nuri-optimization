import pandas as pd

from cases.case12_meridian_diagnostic import (
    load_data_room, BASELINE, REVIEW_DAYS, HOLD_RATE, SERVICE_FLOORS, TRANSFER_PCT,
    OTIF_TARGET, DISRUPTION_SLIP_DAYS, STANDARD_FREIGHT_BENCHMARK_MODES, IMPLIED_WRITEOFF_RATE,
)
from nuri.inventory import lost_sales_estimate
from nuri.inventory_policy import series_stats, apply_policy, summarize
from nuri.rebalancing import lane_cost_index, solve_rebalancing
from nuri.supplier_risk import (
    concentration, otif_working_capital_cost, otif_improvement_release, disruption_exposure,
    supplier_table, lead_time_conflict, dual_source_candidates,
)

pd.options.display.float_format = "{:,.2f}".format
d = load_data_room()
otif = d["suppliers"].set_index("supplier").otif
fill = BASELINE["fill_rate"]
series = series_stats(d["sku"], d["inventory"], d["history"], otif, "history")
base, _ = apply_policy(series, REVIEW_DAYS, HOLD_RATE, fill, SERVICE_FLOORS)

print("=== 6a. Concentration ===")
print({k: round(v, 3) for k, v in concentration(d["suppliers"].set_index("supplier").spend).items()})
tbl = supplier_table(d["sku"], d["suppliers"], base)
print(tbl.head(6).round(3).to_string())

print("\n=== 6b. Data conflict: lead time ===")
print(lead_time_conflict(d["sku"], d["suppliers"]))
lt = d["suppliers"].set_index("supplier")["lead_time"]
alt = series_stats(d["sku"], d["inventory"], d["history"], otif, "history", supplier_lead_time=lt)
alt_pol, _ = apply_policy(alt, REVIEW_DAYS, HOLD_RATE, fill, SERVICE_FLOORS)
print(f"  Net release using SKU-master lead times:      ${summarize(base)['net_release']:,.0f}")
print(f"  Net release using supplier-master lead times: ${summarize(alt_pol)['net_release']:,.0f}")

print("\n=== 6c. Working capital cost of supplier lateness ===")
cost = otif_working_capital_cost(series, REVIEW_DAYS, HOLD_RATE, fill, SERVICE_FLOORS)
print(f"  Safety stock attributable to late deliveries: ${cost.sum():,.0f} total")
print(cost.sort_values(ascending=False).head(5).round(0).to_string())
gain = otif_improvement_release(series, otif, REVIEW_DAYS, HOLD_RATE, fill, SERVICE_FLOORS, OTIF_TARGET)
print(f"  Extra release if every supplier reached {OTIF_TARGET:.0%} OTIF: ${gain:,.0f}")

print("\n=== 6d. Disruption exposure: top 3 suppliers, sales at risk in one slip ===")
print(f"  {'supplier':9}{'slip':>5}{'today':>12}{'lean target':>14}{'A-class (lean)':>16}")
for s in tbl.index[:3]:
    for days in DISRUPTION_SLIP_DAYS:
        e = disruption_exposure(base, s, days)
        print(f"  {s:9}{days:>4}d{e['current']:>12,.0f}{e['target']:>14,.0f}{e['a_class_sales_at_risk_target']:>16,.0f}")

print("\n=== 6e. Dual-source / escalate list (OTIF < 85%, >= 2% of spend) ===")
print(dual_source_candidates(tbl)[["otif", "lead_time", "spend", "share", "n_skus", "n_a"]].round(3).to_string())

print("\n=== 7. Lever comparison: positioning vs purchasing vs policy ===")
fr = d["freight"]
exp = fr[(fr.expedite == 1)]
std_rate = fr[fr["mode"].isin(STANDARD_FREIGHT_BENCHMARK_MODES)]
std_cpl = std_rate.cost.sum() / std_rate.weight_lb.sum()
premium = (exp.cost - exp.weight_lb * std_cpl).clip(lower=0).sum() * 12 / 18
reb = solve_rebalancing(base, lane_cost_index(fr), HOLD_RATE, TRANSFER_PCT)
aged_wo = summarize(base)["aged_excess_value"] * IMPLIED_WRITEOFF_RATE
ls = lost_sales_estimate(d["sku"], d["history"])
levers = pd.DataFrame([
    ("Replenishment / safety-stock policy", summarize(base)["net_release"], summarize(base)["net_release"] * HOLD_RATE, "range $1.7M-$6.7M; needs lead-time & variability data resolved"),
    ("Supplier OTIF to 95%", gain, gain * HOLD_RATE, "incremental to policy; needs supplier agreements"),
    ("Inter-warehouse rebalancing", reb.purchase_avoided.sum(), (reb.gain - reb.freight).sum(), "transfer cost is an assumption (no SKU weights)"),
    ("Expedite freight premium (ceiling)", 0, premium, "ceiling if ALL expedites avoided; no SKU link in freight"),
    ("Stop buying aged excess (write-offs)", 0, aged_wo, "bounded by implied 15.4% write-off rate"),
    ("Recovered lost sales, gross profit (ceiling)", 0, ls["ceiling_gross_profit"], "management claimed $1.38M of sales"),
], columns=["lever", "one_time_cash_$", "annual_$", "caveat"])
print(levers.to_string(index=False))
