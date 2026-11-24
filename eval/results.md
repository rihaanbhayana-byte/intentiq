# IntentIQ offline evaluation

Run 2026-11-24 17:17 · 30 queries · top 10 · labels: criteria from golden.yaml

## Overall

| System | NDCG@10 | MRR | P@5 | Zero-result rate | Avg latency (ms) |
|---|---|---|---|---|---|
| raw | 0.583 | 0.766 | 0.586 | 0% | 119 |
| A | 0.821 | 0.914 | 0.876 | 0% | 124 |
| B | 0.835 | 0.922 | 0.883 | 0% | 124 |

## Headphones

| System | NDCG@10 | MRR | P@5 | Zero-result rate | Avg latency (ms) |
|---|---|---|---|---|---|
| raw | 0.726 | 0.911 | 0.714 | 0% | 35 |
| A | 0.892 | 0.952 | 0.943 | 0% | 40 |
| B | 0.878 | 0.929 | 0.929 | 0% | 39 |

## Laptops

| System | NDCG@10 | MRR | P@5 | Zero-result rate | Avg latency (ms) |
|---|---|---|---|---|---|
| raw | 0.450 | 0.631 | 0.467 | 0% | 198 |
| A | 0.755 | 0.878 | 0.813 | 0% | 203 |
| B | 0.794 | 0.917 | 0.840 | 0% | 203 |

## Head to head (NDCG@10 per query)

| Comparison | Wins | Ties | Losses |
|---|---|---|---|
| A vs raw | 21 | 3 | 5 |
| B vs raw | 21 | 4 | 4 |
| A vs B | 5 | 12 | 12 |

## Brand concentration (bias audit)

Exposure = share of all result slots across the golden queries. Catalog = share of products in the catalog.

| System | Avg unique brands in top 10 | Avg top-brand share | Top 3 brands' exposure |
|---|---|---|---|
| raw | 7.1 | 31% | lenovo 13%, hp 10%, asus 7% |
| A | 7.2 | 24% | lenovo 12%, hp 9%, asus 8% |
| B | 7.3 | 24% | lenovo 12%, hp 9%, asus 8% |

Catalog share of the same brands: lenovo 8%, hp 11%, asus 5%

## Popularity bias

| System | Median ratings count of shown products | Share with fewer than 20 ratings |
|---|---|---|
| raw | 154 | 14% |
| A | 274 | 10% |
| B | 190 | 13% |

## Queries with no relevant product in the catalog

These are excluded from NDCG. Each one is a catalog gap worth mentioning: H04

## Worst queries for ranker A

| Query | NDCG@10 | Raw NDCG@10 |
|---|---|---|
| L13: powerful laptop for programming under $1500 | 0.208 | 0.066 |
| H10: Sony WH-1000XM4 | 0.307 | 0.850 |
| L07: computer for a student that lasts all day | 0.587 | 0.162 |
| L12: 2-in-1 touchscreen laptop for taking notes | 0.594 | 0.664 |
| L08: macbook air | 0.631 | 1.000 |

## Caveats

- Criteria labels reward the same specs ranker A scores, so they can flatter it. Hand labels from `python src/label.py` remove that bias for the queries you label.
- Missing specs count as not meeting a criterion.
- 30 queries is a small set: treat differences under about 0.03 NDCG as noise.
