# Case 12 -- Meridian Fastener & Supply Co.: Phase 1 findings (working draft)

Synthetic engagement (Toledah Foreign Enterprise Case #002). All figures come from
`run_case12_meridian_diagnostic.py`, `run_case12_meridian_policy.py` and
`run_case12_meridian_suppliers.py`; nothing here is hand-edited. Treat as a draft
until the three open data questions at the bottom are answered by the client.

## 1. What is actually going on

The problem is **overstock everywhere, not stock in the wrong place**, and
management's stockout estimate is not supported by its own transaction data.

| Question (RFP s.10) | Finding |
|---|---|
| Is the $1.38M lost-sales estimate supported? | **No.** Sales that could have been lost on stockout-flagged lines: $69K (shortfall method) to $392K (theoretical ceiling, i.e. every flagged line sold nothing) -- 5% to 28% of the claim. Gross profit at the ceiling: $145K. |
| Where do stockouts occur? | Entirely in C-class: A = 0 stockout lines in 18 months, B = 28, C = 7,421 (12.7% of C lines). Rates are flat across warehouses (8.0-8.4%). |
| How much inventory can be released without breaching the service floors? | **Base case $5.4M** (of $11.2M) holding fill at today's 91.8% and A-class at 99.5%. **Defensible range $1.7M-$6.7M**; the spread is driven by one unresolved data conflict (below). |
| Is inventory in the wrong warehouse? | Barely. Excess sits at all four sites ($0.9M-$2.5M each); total shortfalls are only $0.7M. Rebalancing avoids ~$0.6M of purchases once and is worth ~$14K/yr net at a 3% transfer-cost assumption. |
| Which suppliers are the risk? | SUP-044 (27% of spend, 7 SKUs, 81% OTIF) and SUP-045 (21%, 309 SKUs, 82% OTIF). Top 3 = 60.7% of spend. |
| What does supplier lateness cost in working capital? | $333K of safety stock; a 95% OTIF standard would release a further ~$297K. Reliability is a risk, not the main lever. |
| How much emergency freight could be avoided? | Expedite is 17-18% of customer freight at every warehouse, every month -- a policy/process pattern, not a positioning one. Premium over standard ground rates is **at most ~$254K/yr**; no SKU link in freight data, so an achievable figure cannot be derived yet. |
| Does lean inventory expose us to a supplier slip? | **Only for short slips.** A 14-30 day slip at the top 3 suppliers puts <$25K of sales at risk either way. At **60 days it does matter**: SUP-045 exposure rises from $105K (today's stock) to **$324K under the lean target, $166K of it A-class**; SUP-020 is ~$159K ($88K A-class). So the lean policy needs a supplier-slip buffer on SUP-045's A-class SKUs, or a dual-source plan, before it is rolled out. |

Annual economic value of the levers, largest first (carrying cost at 20%):
replenishment policy ~$1.08M; aged-stock/write-off avoidance <= $0.29M; expedite
premium <= $0.25M; recovered gross profit <= $0.15M; supplier OTIF ~$0.06M;
rebalancing ~$0.01M. These are not additive without double-counting and are
ceilings, not commitments -- per RFP s.14 no savings are promised before the
three questions below are closed.

## 2. Data-quality findings to show the client
1. Revenue/COGS tie out to within $330 / $305 (0.001%), not "exactly" as the log states.
2. SKU-master lead time vs supplier-master lead time: correlation -0.06, mean gap 12 days.
3. Observed monthly demand CV ~0.29 vs SKU-master CV ~0.50.
4. 4 of 16 "inter-warehouse transfer" lanes have origin = destination (~$149K over 18 months).
5. Supplier OTIF does not correlate with observed stockouts (r = 0.03).
6. 381 inventory rows have no demand history at all (pure excess).

## 3. Dashboard (D5) -- minimum weekly set
| KPI | Threshold / trigger | Owner |
|---|---|---|
| A-class stockout lines | Any > 0 -> same-day escalation | VP Sales |
| Network fill and OTIF | Fill < 91%, OTIF < 95% | COO |
| Inventory vs target ($, by warehouse x ABC) | Excess > 10% of target | CFO |
| Aged > 180d inventory ($) | Rising week on week | Controller |
| Expedite share of freight | > 15% | COO |
| Supplier OTIF, top 10 by spend | < 85% two weeks running | Procurement |
| Open purchase orders on excess SKUs | Any > 0 | Procurement |

## 4. 30 / 60 / 90 days (no new hires, per RFP s.9)
- **Days 1-30 (quick wins):** freeze purchasing on SKUs above target (C-class and aged first); reconcile the lead-time and CV conflicts; one weekly cross-source number set.
- **Days 31-60:** roll out target stock levels for A and B first (hold a 60-day buffer on SUP-045 A-class SKUs until its OTIF improves), then C; escalate SUP-044/045 on OTIF and lead-time commitments; start the weekly operating cadence.
- **Days 61-90:** review actual sell-down against the schedule (model: ~$1.8M month 1, ~$4.1M month 3); decide on dual-sourcing for SUP-044's two A-class SKUs; set Phase 2 scope.

## 5. Phase 2 scope (D7 outline; RFP range $18-24K, 8-10 weeks)
Replenishment-rule rollout and change management; weekly cadence support;
supplier negotiation support for the top 3; a lightweight decision-support layer
(this repo's policy + rebalancing models behind a weekly refresh). **Priced
separately** per RFP s.13: any system integration or ERP work.

## 6. Open questions that decide the headline number
1. **Which demand variability is right** -- history (0.29) or master (0.50)? This alone moves the release between ~$5.4M and ~$3.1M.
2. **Which lead time is right** -- SKU master or supplier master? (Release moves only ~$0.14M, but the data-quality point stands.)
3. **Per-SKU shipping weight** -- needed to price transfers properly; and **order-line data** to tie strategic accounts to SKUs and expedites to stockouts.

## 7. Model assumptions that matter
30-day review period; 20% carrying cost; A floor 99.5%, B floor 97%; C service set
by cost-minimisation against the fill target; lead-time spread = lead time x
(1 - supplier OTIF); transfer cost 3% of unit cost x lane index; no minimum-stock
or order-quantity constraints (so C-class targets are optimistic). Sensitivity grid
is in `run_case12_meridian_policy.py`.
