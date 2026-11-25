# IntentIQ offline evaluation

Run 2026-11-24 20:11 · 30 queries · top 10 · labels: criteria from golden.yaml

## Overall

| System | NDCG@10 | MRR | P@5 | Zero-result rate | Avg latency (ms) |
|---|---|---|---|---|---|
| raw | 0.583 | 0.766 | 0.586 | 0% | 12 |
| A | 0.888 | 0.983 | 0.938 | 0% | 19 |
| B | 0.909 | 1.000 | 0.931 | 0% | 18 |

## Headphones

| System | NDCG@10 | MRR | P@5 | Zero-result rate | Avg latency (ms) |
|---|---|---|---|---|---|
| raw | 0.726 | 0.911 | 0.714 | 0% | 9 |
| A | 0.933 | 1.000 | 0.957 | 0% | 16 |
| B | 0.936 | 1.000 | 0.943 | 0% | 15 |

## Laptops

| System | NDCG@10 | MRR | P@5 | Zero-result rate | Avg latency (ms) |
|---|---|---|---|---|---|
| raw | 0.450 | 0.631 | 0.467 | 0% | 15 |
| A | 0.847 | 0.967 | 0.920 | 0% | 22 |
| B | 0.884 | 1.000 | 0.920 | 0% | 21 |

## Head to head (NDCG@10 per query)

| Comparison | Wins | Ties | Losses |
|---|---|---|---|
| A vs raw | 23 | 4 | 2 |
| B vs raw | 22 | 4 | 3 |
| A vs B | 6 | 12 | 11 |

## Brand concentration (bias audit)

Exposure = share of all result slots across the golden queries. Catalog = share of products in the catalog.

| System | Avg unique brands in top 10 | Avg top-brand share | Top 3 brands' exposure |
|---|---|---|---|
| raw | 7.1 | 31% | lenovo 13%, hp 10%, asus 7% |
| A | 6.9 | 28% | lenovo 13%, hp 9%, asus 7% |
| B | 6.9 | 28% | lenovo 13%, hp 9%, asus 7% |

Catalog share of the same brands: lenovo 8%, hp 11%, asus 5%

## Popularity bias

| System | Median ratings count of shown products | Share with fewer than 20 ratings |
|---|---|---|
| raw | 154 | 14% |
| A | 241 | 11% |
| B | 180 | 13% |

## Queries with no relevant product in the catalog

These are excluded from NDCG. Each one is a catalog gap worth mentioning: H04

## Worst queries for ranker A

| Query | NDCG@10 | Raw NDCG@10 |
|---|---|---|
| L07: computer for a student that lasts all day | 0.587 | 0.162 |
| L08: macbook air | 0.631 | 1.000 |
| L04: ultralight laptop under 3.5 lbs, 14 inch, 512GB SSD under $900 | 0.646 | 0.220 |
| L09: laptop for video editing with 32GB RAM | 0.783 | 0.069 |
| L03: cheap laptop for college under $400 | 0.784 | 0.274 |

## Caveats

- Criteria labels reward the same specs ranker A scores, so they can flatter it. Hand labels from `python src/label.py` remove that bias for the queries you label.
- Missing specs count as not meeting a criterion.
- 30 queries is a small set: treat differences under about 0.03 NDCG as noise.
