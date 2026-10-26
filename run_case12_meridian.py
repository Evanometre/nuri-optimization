"""End-to-end Meridian analysis. Usage: python run_case12_meridian.py [data_room.xlsx] [output_dir]"""

import json
import os
import sys

import numpy as np
import pandas as pd

from cases.case12_meridian import (
    AGED_LIQUIDATION_RECOVERY, BASE_CSL, FRONTIER, HOLDING_RATE, MANAGEMENT_LOST_SALES,
    REVIEW_DAYS, SCENARIOS, SENSITIVITIES,
)
from nuri.inventory.data import load_data_room
from nuri.inventory.freight import expedite_analysis
from nuri.inventory.lost_sales import estimate_lost_sales
from nuri.inventory.network import utilization_after_policy, warehouse_profile
from nuri.inventory.opportunity import (
    aged_excess_value, compare_scenarios, excess_by_cover, release_timeline, sensitivity,
    service_frontier, targets_for, waterfall,
)
from nuri.inventory.policy import build_pair_table, simulate, summarize_simulation
from nuri.inventory.rebalance import (
    lane_costs, plan_transfers, pooling_table, positioning_class, same_site_transfers, warehouse_balance,
)
from nuri.inventory.reconcile import data_quality_findings, run_reconciliation
from nuri.inventory.segmentation import health_flags, sku_segments, segment_matrix
from nuri.inventory.suppliers import (
    concentration, disruption_exposure, otif_improvement_release, supplier_table,
)

path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("NURI_MERIDIAN_XLSX")
out_dir = sys.argv[2] if len(sys.argv) > 2 else "outputs/meridian"
if not path:
    sys.exit("Give the data room workbook path as argv[1] or NURI_MERIDIAN_XLSX")
os.makedirs(out_dir, exist_ok=True)


def save(df, name, **kw):
    df.to_csv(os.path.join(out_dir, name), **kw)


def money(x):
    return f"${x:,.0f}"


dr = load_data_room(path)
base = dr.baseline()
summary = {}

print("=== 1. Reconciliation ===")
rec = run_reconciliation(dr)
print(rec.to_string(index=False))
dq = data_quality_findings(dr)
for k, v in dq.items():
    print(f"  data gap -- {k}: {v}")
save(rec, "reconciliation.csv", index=False)
summary["reconciliation_all_ok"] = bool(rec["ok"].all())
summary["data_quality"] = dq

print("\n=== 2. Management's $1.38M lost-sales estimate ===")
ls = estimate_lost_sales(dr, MANAGEMENT_LOST_SALES)
print(f"Central estimate (shortfall vs normal months): {money(ls['shortfall_revenue'])} sales, {money(ls['shortfall_margin'])} margin")
print(f"Ceiling (every flagged line loses a full average month): {money(ls['ceiling_revenue'])}")
print(f"Management's figure is {ls['management_multiple_of_ceiling']:.1f}x the ceiling, {ls['management_multiple_of_shortfall']:.0f}x the central estimate")
print("Stockout rate by class:", ls["stockout_rate_by_class"].round(4).to_dict())
summary["lost_sales"] = {k: float(v) for k, v in ls.items() if isinstance(v, (int, float, np.floating))}
save(ls["by_class"], "lost_sales_by_class.csv")

print("\n=== 3. Segmentation and inventory health ===")
pairs = build_pair_table(dr)
base_sc = SCENARIOS[1]
targets = targets_for(pairs, base_sc)
seg = sku_segments(dr)
counts, dollars = segment_matrix(seg, pairs)
print("SKU counts by ABC x XYZ:\n", counts, "\nInventory $M:\n", (dollars / 1e6).round(2))
health = health_flags(pairs, targets)
cover = pairs["On-hand Units"] / pairs["mean_month"].where(pairs["mean_month"] > 0)
by_class_cover = pd.DataFrame(
    {"inventory": pairs.groupby("ABC Class")["value"].sum(), "median_cover_months": cover.groupby(pairs["ABC Class"]).median()}
)
print(by_class_cover.round(2))
prone = health[health["stockout_prone"] & (health["mean_month"] > 0)]
prone_cover = (prone["On-hand Units"] / prone["mean_month"]).median()
print(f"Stockout-prone pairs (>=20% of months flagged): {len(prone)}; they hold a median {prone_cover:.1f} months of cover while stocking out")
summary["stockout_prone"] = {"pairs": int(len(prone)), "median_cover_months": float(prone_cover)}
save(seg, "sku_segments.csv")
save(counts, "segment_counts.csv")
save(dollars, "segment_inventory.csv")
save(by_class_cover, "class_cover.csv")

print("\n=== 4. Working-capital opportunity ===")
scen_table = compare_scenarios(pairs, SCENARIOS, base, expedite_analysis(dr.freight), ls, HOLDING_RATE)
print(scen_table.T.round(0).to_string())
save(scen_table, "scenarios.csv")
wf = waterfall(pairs, targets)
print("Waterfall (base):\n", wf.round(0).to_string())
tl = release_timeline(pairs, targets)
print("Release timeline (stop-buy, demand burns excess):\n", tl.round(0).to_string())
ebc = excess_by_cover(pairs, targets)
print("Excess above order-up-to by months of cover:\n", ebc.round(0).to_string())
aged_x = aged_excess_value(pairs, targets)
print(f"Aged >180d stock above order-up-to: {money(aged_x)}; one-time liquidation loss at {1-AGED_LIQUIDATION_RECOVERY:.0%}: {money(aged_x*(1-AGED_LIQUIDATION_RECOVERY))}")
sens = sensitivity(pairs, base_sc, SENSITIVITIES, base)
print("Sensitivity of base release:\n", sens.round(0).to_string())
save(wf.to_frame("value"), "waterfall.csv")
save(tl, "release_timeline.csv")
save(ebc.to_frame("value"), "excess_by_cover.csv")
save(sens, "sensitivity.csv")
summary["base_release"] = float(scen_table.loc["Base", "release"])
summary["aged_excess"] = aged_x

print("\n=== 5. Service vs inventory frontier (Monte Carlo) ===")
frontier = service_frontier(pairs, simulate, summarize_simulation, FRONTIER, REVIEW_DAYS)
print(frontier.round(4).to_string())
save(frontier, "service_frontier.csv")
sim = simulate(pairs, targets["order_up_to"].to_numpy())
sim_class, sim_total = summarize_simulation(pairs, sim)
print("Base policy, simulated by class:\n", sim_class.round(4).to_string(), "\n", sim_total)
summary["base_sim"] = {"by_class": sim_class.round(5).to_dict(), "total": {k: float(v) for k, v in sim_total.items()}}

print("\n=== 6. Positioning across the four DCs ===")
lanes = lane_costs(dr.freight)
moves, frame = plan_transfers(pairs, targets, lanes, HOLDING_RATE)
bal = warehouse_balance(frame)
print(bal.round(0).to_string())
print(f"Worthwhile transfers: {len(moves)} moves, {money(moves['value'].sum())} of stock, freight {money(moves['freight_cost'].sum())}, carrying benefit {money(moves['carry_benefit'].sum())}")
pool = pooling_table(dr, pairs, BASE_CSL, REVIEW_DAYS)
pool["positioning"] = positioning_class(pool)
pool_sum = pool.groupby("positioning").agg(skus=("abc", "count"), safety_stock_saving=("saving_value", "sum"))
print("Pooling (single-DC stocking) safety-stock saving:\n", pool_sum.round(0).to_string())
ss = same_site_transfers(dr.freight)
print("Same-site 'transfers':", ss)
save(moves, "transfer_moves.csv", index=False)
save(bal, "warehouse_balance.csv")
save(pool, "pooling.csv")
summary["transfers"] = {"moves": int(len(moves)), "value": float(moves["value"].sum()), "freight": float(moves["freight_cost"].sum())}
summary["same_site_transfers"] = ss

print("\n=== 7. Warehouse network ===")
wh = warehouse_profile(dr, pairs)
print(wh[["order_share", "capacity_share", "orders_per_sqft", "Avg Pick-Pack Cost ($/order)", "Space Utilization %", "inventory_share", "demand_share"]].round(3).to_string())
tgt_units = targets.assign(wh=pairs["Warehouse"].to_numpy()).groupby("wh")["target_units"].sum()
tgt_value = targets.assign(wh=pairs["Warehouse"].to_numpy()).groupby("wh")["target_value"].sum()
space = wh[["Space Utilization %"]].copy()
space["value_proxy_after_pct"] = space["Space Utilization %"] * tgt_value / wh["inventory_value"]
space["units_proxy_after_pct"] = utilization_after_policy(wh, tgt_units)["utilization_after_pct"]
print("Utilization after policy (bounds: value vs unit proxy):\n", space.round(1).to_string())
save(wh, "warehouses.csv")
save(space, "space_after_policy.csv")

print("\n=== 8. Suppliers ===")
conc = concentration(dr)
print(conc)
st = supplier_table(dr, pairs, BASE_CSL, REVIEW_DAYS)
cols = ["Annual Spend ($)", "spend_share", "OTIF", "Avg Lead Time Days", "n_skus", "a_skus", "inventory_value", "safety_stock_value", "risk_score"]
print(st[cols].head(8).round(3).to_string())
top3 = list(st.sort_values("Annual Spend ($)", ascending=False).index[:3])
rel95 = otif_improvement_release(pairs, BASE_CSL, top3, 0.95, REVIEW_DAYS)
print(f"Top-3 suppliers {top3} to 95% OTIF releases a further {money(rel95)} of safety stock")
expo = {}
for s in top3:
    expo[s] = {}
    for days in (30, 60, 90):
        today = disruption_exposure(pairs, s, days)
        after = disruption_exposure(pairs, s, days, on_hand_units=targets["target_units"].to_numpy())
        expo[s][days] = {"today": today, "after_policy": after}
        print(f"  {s}: {days}-day outage revenue at risk today {money(today['lost_revenue'])} ({today['days_of_cover']:.0f} d cover) -> after policy {money(after['lost_revenue'])} ({after['days_of_cover']:.0f} d cover)")
save(st, "suppliers.csv")
summary["suppliers"] = {"concentration": conc, "top3": top3, "otif95_release": rel95, "exposure": expo}

print("\n=== 9. Freight ===")
ex = expedite_analysis(dr.freight)
for k in ("annual_expedite_spend", "expedites_per_year", "mean_expedite_cost", "mean_standard_cost", "premium_per_expedite", "annual_premium", "premium_share_of_spend", "expedite_share_of_shipments", "all_expedites_to_customers"):
    print(f"  {k}: {ex[k]:,.3f}" if isinstance(ex[k], float) else f"  {k}: {ex[k]}")
print("  expedite rate by origin:", ex["expedite_rate_by_origin"].round(4).to_dict())
summary["expedite"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in ex.items() if isinstance(v, (int, float, np.floating, bool))}

print("\n=== 10. Per SKU-warehouse policy (D2) ===")
act = np.select(
    [pairs["mean_month"] == 0, pairs["On-hand Units"] > targets["order_up_to"] * 1.0,
     pairs["On-hand Units"] < targets["target_units"]],
    ["no sales: transfer or liquidate", "above order-up-to: stop buying, sell down", "below target: order up to S"],
    "hold: within policy band",
)
policy = pd.DataFrame(
    {
        "Warehouse": pairs["Warehouse"], "SKU": pairs["SKU"], "ABC": pairs["ABC Class"], "Category": pairs["Category"],
        "supplier": pairs["Supplier ID"], "lead_time_days": pairs["lead_time"], "unit_cost": pairs["Unit Cost ($)"],
        "mean_units_month": pairs["mean_month"].round(2), "sd_units_month": pairs["sd_month"].round(2),
        "on_hand": pairs["On-hand Units"], "aged_pct": pairs["Aged >180d %"],
        "order_up_to": targets["order_up_to"], "safety_stock": targets["safety_stock"].round(1),
        "target_avg_units": targets["target_units"].round(1), "action": act,
    }
)
save(policy, "sku_warehouse_policy.csv", index=False)
print(policy["action"].value_counts().to_string())

# One workbook with every table, plus the assumptions that are not in the data room.
assumptions = pd.DataFrame(
    [
        ("Service levels (cycle service level)", f"A {BASE_CSL['A']:.1%}, B {BASE_CSL['B']:.1%}, C {BASE_CSL['C']:.1%}", "Our choice; A+B cover 94.9% of COGS"),
        ("Replenishment review", f"every {REVIEW_DAYS} days", "Assumed; current cadence unknown"),
        ("Supplier lead-time spread", "sd = lead time x (1 - OTIF)", "Assumed; no PO receipt dates in the data room"),
        ("Lot-size / case-pack buffer", "+5% on order-up-to level (base)", "Assumed; no MOQs in the data room"),
        ("Carrying cost", f"{HOLDING_RATE:.0%} of inventory value per year", "Assumed: capital ~9% + handling/insurance ~6%"),
        ("Aged excess liquidation recovery", f"{AGED_LIQUIDATION_RECOVERY:.0%} of cost", "Assumed"),
        ("Demand", "18-month mean/sd per SKU-warehouse; stockout months raised to non-stockout mean", "No trend in the 18 months (-0.05%/month, p=0.85)"),
        ("Open purchase orders", "not in data room", "Release timeline ignores stock already on order"),
        ("Customer dimension", "not in data room", "Strategic-account service cannot be measured directly"),
        ("OTIF", "not modelled", "Driven by carriers/operations; monitored, not optimised here"),
        ("Data", "Synthetic case (Toledah Foreign Enterprise Case 002)", "Meridian is fictional"),
    ],
    columns=["assumption", "value", "note"],
)
try:
    with pd.ExcelWriter(os.path.join(out_dir, "Meridian_Phase1_Model_Outputs.xlsx"), engine="openpyxl") as xw:
        assumptions.to_excel(xw, sheet_name="README_Assumptions", index=False)
        sheets = {
            "Reconciliation": rec, "Scenarios": scen_table.reset_index(), "Waterfall": wf.to_frame("value").reset_index(),
            "Release_Timeline": tl.reset_index(), "Excess_by_Cover": ebc.to_frame("value").reset_index(),
            "Sensitivity": sens.reset_index(), "Service_Frontier": frontier.reset_index(),
            "Segments_ABC_XYZ": dollars.reset_index(), "Suppliers": st.reset_index(), "Warehouses": wh.reset_index(),
            "Warehouse_Balance": bal.reset_index(), "Transfer_Moves": moves, "Pooling": pool.reset_index(),
            "SKU_Warehouse_Policy": policy,
        }
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name[:31], index=False)
            ws = xw.sheets[name[:31]]
            ws.freeze_panes = "A2"
            for i, col in enumerate(df.columns, start=1):
                width = min(max(len(str(col)), *(len(str(v)) for v in df[col].head(50))) + 2, 48) if len(df) else len(str(col)) + 2
                ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = width
except ImportError:
    print("openpyxl not installed: skipped workbook")

with open(os.path.join(out_dir, "summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2, default=float)
print(f"\nOutputs written to {out_dir}/")
