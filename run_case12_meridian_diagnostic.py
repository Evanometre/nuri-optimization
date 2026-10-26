import pandas as pd

from cases.case12_meridian_diagnostic import load_data_room, BASELINE, IMPLIED_WRITEOFF_RATE
from nuri.inventory import (
    reconcile_all, segment_skus, baseline_by_warehouse, baseline_by_class, lost_sales_estimate,
)

pd.options.display.float_format = "{:,.3f}".format
d = load_data_room()

print("=== 1. Reconciliation: re-derived from raw tables ===")
for c in reconcile_all(d["sku"], d["inventory"], d["history"], d["suppliers"].spend,
                       d["customers"].revenue, d["freight"], BASELINE, IMPLIED_WRITEOFF_RATE):
    print(f"  [{'OK ' if c.ok else 'FAIL'}] {c.name}: computed {c.computed:,.4f} vs stated {c.expected:,.4f}")

print("\n=== 2a. Baseline by warehouse ===")
print(baseline_by_warehouse(d["sku"], d["inventory"], d["history"]).round(3).to_string())
print("\n=== 2b. Baseline by ABC class ===")
print(baseline_by_class(d["sku"], d["inventory"], d["history"]).round(3).to_string())
print("\n=== 2c. ABC x XYZ segment counts ===")
print(segment_skus(d["sku"]).segment.value_counts().sort_index().to_string())

print("\n=== 3. Management's lost-sales claim vs transaction data ===")
est = lost_sales_estimate(d["sku"], d["history"])
claim = BASELINE["lost_sales_claim"]
for k, v in est.items():
    print(f"  {k}: {v:,.0f}" + (f"  ({v / claim:.0%} of claim)" if "sales" in k or "profit" in k else ""))
print(f"  Management claim: {claim:,.0f}")
