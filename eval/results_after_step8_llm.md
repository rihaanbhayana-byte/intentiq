# IntentIQ offline evaluation

Run 2026-11-24 20:11 · 30 queries · top 10 · labels: eval/labels_llm.csv for 30 queries, criteria for the rest

## Overall

| System | NDCG@10 | MRR | P@5 | Zero-result rate | Avg latency (ms) |
|---|---|---|---|---|---|
| raw | 0.624 | 0.851 | 0.613 | 0% | 12 |
| A | 0.862 | 0.928 | 0.827 | 0% | 18 |
| B | 0.889 | 0.961 | 0.820 | 0% | 18 |

## Headphones

| System | NDCG@10 | MRR | P@5 | Zero-result rate | Avg latency (ms) |
|---|---|---|---|---|---|
| raw | 0.697 | 0.839 | 0.600 | 0% | 9 |
| A | 0.874 | 0.933 | 0.813 | 0% | 15 |
| B | 0.897 | 0.967 | 0.787 | 0% | 14 |

## Laptops

| System | NDCG@10 | MRR | P@5 | Zero-result rate | Avg latency (ms) |
|---|---|---|---|---|---|
| raw | 0.552 | 0.864 | 0.627 | 0% | 16 |
| A | 0.849 | 0.922 | 0.840 | 0% | 22 |
| B | 0.881 | 0.956 | 0.853 | 0% | 21 |

## Head to head (NDCG@10 per query)

| Comparison | Wins | Ties | Losses |
|---|---|---|---|
| A vs raw | 23 | 3 | 4 |
| B vs raw | 23 | 3 | 4 |
| A vs B | 8 | 10 | 12 |

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

## Worst queries for ranker A

| Query | NDCG@10 | Raw NDCG@10 |
|---|---|---|
| L15: budget gaming laptop with RTX under $800 | 0.530 | 0.556 |
| H05: headset with microphone for zoom calls under $80 | 0.574 | 0.860 |
| H09: wireless over-ear headphones with long battery | 0.627 | 0.281 |
| L08: macbook air | 0.631 | 1.000 |
| L13: powerful laptop for programming under $1500 | 0.660 | 0.376 |

## Caveats

- Criteria labels reward the same specs ranker A scores, so they can flatter it. Hand labels from `python src/label.py` remove that bias for the queries you label.
- Missing specs count as not meeting a criterion.
- 30 queries is a small set: treat differences under about 0.03 NDCG as noise.
