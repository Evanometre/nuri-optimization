import itertools

import pandas as pd

from cases.case12_meridian_diagnostic import (
    load_data_room, BASELINE, REVIEW_DAYS, HOLD_RATE, SERVICE_FLOORS, NETWORK_FILL_FLOOR,
    TRANSFER_PCT, SENSITIVITY,
)
from nuri.inventory_policy import series_stats, apply_policy, summarize, release_schedule
from nuri.rebalancing import lane_cost_index, solve_rebalancing

pd.options.display.float_format = "{:,.2f}".format
d = load_data_room()
otif = d["suppliers"].set_index("supplier").otif


def run(sigma_source, review_days, target_fill, floors=SERVICE_FLOORS):
    s = series_stats(d["sku"], d["inventory"], d["history"], otif, sigma_source)
    return apply_policy(s, review_days, HOLD_RATE, target_fill, floors)[0]


print("=== 4a. Sensitivity: net releasable inventory ($M) ===")
rows = []
for src, R, tf in itertools.product(*SENSITIVITY.values()):
    r = summarize(run(src, R, tf))
    rows.append({"variability": src, "review_days": R, "fill_target": tf,
                 "target_inv": r["target_inventory"] / 1e6, "net_release": r["net_release"] / 1e6,
                 "carrying_saving_$M/yr": r["net_release"] * HOLD_RATE / 1e6})
grid = pd.DataFrame(rows)
print(grid.to_string(index=False))
print(f"\nDefensible range: ${grid.net_release.min():.1f}M - ${grid.net_release.max():.1f}M "
      f"of ${BASELINE['inventory'] / 1e6:.1f}M. Lower end = master CV, 45-day review (most conservative).")

print("\n=== 4b. Base case: observed variability, 30-day review, fill held at current 91.8% ===")
base = run("history", REVIEW_DAYS, BASELINE["fill_rate"])
r = summarize(base)
for k, v in r.items():
    print(f"  {k}: {v:,.3f}" if k == "network_fill" else f"  {k}: ${v:,.0f}")
print("\n  By ABC class (service level is the model's chosen line fill):")
by = base.assign(cur=base.on_hand * base.cost, tgt=base.target_units * base.cost,
                 exc=base.excess_units * base.cost).groupby("abc").agg(
    current=("cur", "sum"), target=("tgt", "sum"), excess=("exc", "sum"), service=("service_level", "mean"))
print(by.to_string())
sched, stuck = release_schedule(base)
print("\n  Release if purchasing stops on excess series (cumulative $M, months 1,3,6,12):",
      [round(sched[m - 1] / 1e6, 2) for m in (1, 3, 6, 12)], f" never self-liquidating: ${stuck:,.0f}")

print("\n=== 4c. What the A-class floor costs ===")
for fl in (0.99, 0.995, 0.999):
    p = run("history", REVIEW_DAYS, BASELINE["fill_rate"], {"A": fl, "B": 0.97})
    print(f"  A floor {fl}: target inventory ${summarize(p)['target_inventory']:,.0f}")

print("\n=== 5. Rebalancing between warehouses (transportation LP) ===")
print("Excess vs shortfall by warehouse (at cost):")
wh = base.assign(exc=base.excess_units * base.cost, need=base.need_units * base.cost).groupby("warehouse")[["exc", "need"]].sum()
print(wh.to_string())
lanes = lane_cost_index(d["freight"])
print("\nTransfer cost assumption is the weak input (no SKU weights in data room):")
for tp in (0.02, TRANSFER_PCT, 0.05, 0.10):
    t = solve_rebalancing(base, lanes, HOLD_RATE, tp)
    print(f"  transfer cost {tp:.0%} of unit cost: {len(t)} moves, purchases avoided ${t.purchase_avoided.sum():,.0f}, "
          f"freight ${t.freight.sum():,.0f}, net annual gain ${(t.gain - t.freight).sum():,.0f}")
