# Case 13 -- Northline Industrial Supply: reorder logic & inventory forecasting (draft)

FICTIONAL distributor, SYNTHETIC data (~$12M revenue, 2,500 SKUs, one warehouse, 35 suppliers,
130 weeks of history). Built to show the method on messy exports. It is not evidence about any
real company. Run: `python run_case13_reorder.py 13` (cleaning + replay) and
`python run_case13_frontier.py 21 22 23` (service-vs-inventory curves).

## What the engagement does
1. **Clean the raw export** and report every fix (below). Scored against hidden truth: weekly sales
   are recovered exactly.
2. **Fit on weeks 1-78 only** (a test proves the fit never sees later weeks).
3. **Replay the last 52 weeks** under (a) the company's current buying habit and (b) the new rules,
   on identical demand and identical lead-time draws. The replay of (a) reproduces what actually
   happened, so the comparison is anchored to history.

## Data-quality report (seed 13)
92,049 raw lines: 6,398 SKU codes typed inconsistently, 898 duplicate lines, 1,347 return lines,
211 lines in boxes not eaches (16 with no documented conversion, inferred from price),
11 negative on-hand rows, 25 inventory rows for SKUs not in the master, 301 project-order weeks flagged.
Supplier-quoted lead times are blank for 15% of SKUs and optimistic for the rest; the model uses
actual receipt history instead.

## Result: held-out 52 weeks, SKUs with sales history, three seeds not used during development
| seed | current rule (12/20 wks): revenue fill / avg inventory | new rules @97%: revenue fill / avg inventory | inventory the current-style rule needs to match the new service |
|---|---|---|---|
| 21 | 98.2% / $2,488K | 98.6% / $2,065K | ~$2,796K (+26%) |
| 22 | 97.3% / $2,571K | 98.4% / $2,264K | ~$2,903K (+22%) |
| 23 | 97.2% / $2,335K | 98.8% / $2,031K | ~$2,756K (+26%) |

Purchases over the year were also ~$0.7-0.9M lower under the new rules.

## What this does NOT show -- say this on camera
- **The baseline is a rule I designed** (one flat rule, ignores supplier lead time). The gain depends
  on how weak a client's real habit is; a client with good min/max levels would see less.
- **The demand and lead-time generators use the same family of assumptions the model leans on**
  (compound-Poisson demand, gamma lead times). Real demand is messier. This is a method demo, not a
  field result.
- **More, smaller stockouts.** The new rules have roughly twice the stockout SKU-weeks (e.g. 1,418 vs 691
  for seed 21): they accept short gaps on slow, low-value SKUs to avoid large ones.
- **Forecast accuracy is not the story.** Weekly per-SKU forecasts are no more accurate than the buyer's
  trailing average (WAPE ~53% for both); the gain comes from lead-time-aware, cost-aware stock levels,
  not from a better point forecast.
- **A cold tail cannot be forecast.** 679 of 2,500 SKUs never sold in the training window and carry
  ~5% of later revenue; no method recovers that from history.
- **Targets 90-95% give identical results**: the top-revenue protection floor (98%) binds. Only
  targets >= 95% change behaviour.
- Developed on seed 13; the table uses other seeds, but all seeds come from one generator.

## Still to build before this is a sellable demo
- One-page "cash trap" scan output (excess vs target, dead stock, this week's purchase list).
- A real public dataset run to show the engine on data I did not generate.
