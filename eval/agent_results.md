# IntentIQ agent evaluation

Run 2026-11-24 20:14 · 10 conversations

| Metric | rules | llm |
|---|---|---|
| turns passed | 13/16 | 15/16 |
| conversations fully passed | 7/10 | 9/10 |
| asked when it should | 2/2 | 2/2 |
| questions that were needed | 2/5 | 2/3 |
| answers blocked by the safety check | 0 | 0 |
| turns answered by the rule planner | 16 | 6 |
| LLM calls | 0 | 18 |
| median latency (ms) | 17 | 1910 |
| p95 latency (ms) | 84 | 30297 |

## Failed turns

- **rules** A04-flights turn 1 ("something to block noise on long flights"): category None != headphones; constraint anc=None (want True); fewer than 2 of top 3 meet {'anc': True}
- **rules** A06-implicit-category turn 1 ("computer for a student that lasts all day"): category None != laptops
- **rules** A09-only-one-question turn 2 ("not sure, something under $100"): clarify should be False; max_price None != 100
- **llm** A06-implicit-category turn 1 ("computer for a student that lasts all day"): category None != laptops

The safety check counts LLM answers that mentioned a product, ref or price not returned by the tools; those were replaced with a template answer before the user saw them.
