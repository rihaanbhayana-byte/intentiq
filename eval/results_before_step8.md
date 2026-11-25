# IntentIQ offline evaluation

Run 2026-11-24 20:10 · 30 queries · top 10 · labels: eval/labels_llm.csv for 30 queries, criteria for the rest

## Overall

| System | NDCG@10 | MRR | P@5 | Zero-result rate | Avg latency (ms) |
|---|---|---|---|---|---|
| raw | 0.624 | 0.851 | 0.613 | 0% | 21 |
| A | 0.811 | 0.894 | 0.800 | 0% | 27 |
| B | 0.847 | 0.944 | 0.800 | 0% | 26 |

## Headphones

| System | NDCG@10 | MRR | P@5 | Zero-result rate | Avg latency (ms) |
|---|---|---|---|---|---|
| raw | 0.697 | 0.839 | 0.600 | 0% | 10 |
| A | 0.826 | 0.933 | 0.800 | 0% | 16 |
| B | 0.848 | 0.967 | 0.773 | 0% | 15 |

## Laptops

| System | NDCG@10 | MRR | P@5 | Zero-result rate | Avg latency (ms) |
|---|---|---|---|---|---|
| raw | 0.552 | 0.864 | 0.627 | 0% | 33 |
| A | 0.797 | 0.856 | 0.800 | 0% | 38 |
| B | 0.846 | 0.922 | 0.827 | 0% | 38 |

## Head to head (NDCG@10 per query)

| Comparison | Wins | Ties | Losses |
|---|---|---|---|
| A vs raw | 21 | 2 | 7 |
| B vs raw | 21 | 3 | 6 |
| A vs B | 6 | 9 | 15 |

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

## Worst queries for ranker A

| Query | NDCG@10 | Raw NDCG@10 |
|---|---|---|
| H10: Sony WH-1000XM4 | 0.275 | 0.945 |
| L13: powerful laptop for programming under $1500 | 0.535 | 0.376 |
| L15: budget gaming laptop with RTX under $800 | 0.535 | 0.556 |
| L12: 2-in-1 touchscreen laptop for taking notes | 0.540 | 0.640 |
| H05: headset with microphone for zoom calls under $80 | 0.574 | 0.860 |

## Caveats

- Criteria labels reward the same specs ranker A scores, so they can flatter it. Hand labels from `python src/label.py` remove that bias for the queries you label.
- Missing specs count as not meeting a criterion.
- 30 queries is a small set: treat differences under about 0.03 NDCG as noise.
