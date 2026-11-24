"""Step 6: offline evaluation of raw retrieval vs. ranker A vs. ranker B.

Run from the project folder:
    python src/evaluate.py                         # labels from the criteria in eval/golden.yaml
    python src/evaluate.py --labels eval/labels.csv   # your hand labels replace them where present

Writes eval/results.md (the tables for your case study) and eval/results_per_query.csv.
"""
import argparse
import csv
import math
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)  # data paths in search.py are relative to the project root
sys.path.insert(0, str(ROOT / "src"))

from criteria import grade  # noqa: E402
from parse import parse_query  # noqa: E402
from rank import rank  # noqa: E402
from search import Searcher  # noqa: E402

K = 10
SYSTEMS = ["raw", "A", "B"]


def dcg(grades):
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def load_labels(path):
    labels = {}
    if path and Path(path).exists():
        with open(path) as f:
            for row in csv.DictReader(f):
                labels.setdefault(row["query_id"], {})[row["product_id"]] = int(row["grade"])
    return labels


def run_systems(s, case, prior):
    q, cat = case["query"], case["category"]
    intent = parse_query(q, default_category=cat)
    t0 = time.perf_counter()
    cand = s.search(q, cat, intent["max_price"], k=100, mode="hybrid")
    t_ret = (time.perf_counter() - t0) * 1000
    out, ms = {}, {}
    if cand.empty:
        return {name: cand for name in SYSTEMS}, {name: t_ret for name in SYSTEMS}
    out["raw"], ms["raw"] = cand.head(K), t_ret
    for name in ("A", "B"):
        t1 = time.perf_counter()
        out[name] = rank(cand, intent, cat, prior, name, k=K, brand_cap=3)
        ms[name] = t_ret + (time.perf_counter() - t1) * 1000
    return out, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="eval/golden.yaml")
    ap.add_argument("--labels", default="eval/labels.csv")
    ap.add_argument("--out", default="eval/results.md")
    a = ap.parse_args()

    cases = yaml.safe_load(open(a.golden))
    human = load_labels(a.labels)
    s = Searcher()
    cat_df = {c: s.df[s.df["category"] == c] for c in s.df["category"].unique()}
    prior = {c: float(d["avg_rating"].mean()) for c, d in cat_df.items()}

    per_query, exposure = [], {name: Counter() for name in SYSTEMS}
    pop = {name: [] for name in SYSTEMS}
    unanswerable = []
    for case in cases:
        cat = case["category"]
        hl = human.get(case["id"])
        if hl:  # hand labels: unlabeled products count as 0
            ideal = sorted(hl.values(), reverse=True)[:K]
            gfun = lambda r, hl=hl: hl.get(str(r["product_id"]), 0)  # noqa: E731
        else:
            ideal = sorted((grade(r, case) for _, r in cat_df[cat].iterrows()), reverse=True)[:K]
            gfun = lambda r, case=case: grade(r, case)  # noqa: E731
        idcg = dcg(ideal)
        if idcg == 0:
            unanswerable.append(case["id"])
        results, ms = run_systems(s, case, prior[cat])
        for name in SYSTEMS:
            df = results[name]
            g = [gfun(r) for _, r in df.iterrows()]
            first = next((i for i, x in enumerate(g) if x >= 1), None)
            brands = [str(b).lower() for b in df["brand"]] if len(df) else []
            exposure[name].update(brands)
            pop[name] += [float(x) for x in df["num_ratings"]] if len(df) else []
            per_query.append({
                "query_id": case["id"], "category": cat, "system": name, "labels": "hand" if hl else "criteria",
                "ndcg10": round(dcg(g) / idcg, 4) if idcg else None,
                "mrr": round(1 / (first + 1), 4) if first is not None else 0.0,
                "p5": round(sum(x >= 1 for x in g[:5]) / 5, 4),
                "n_results": len(df), "unique_brands": len(set(brands)),
                "top_brand_share": round(Counter(brands).most_common(1)[0][1] / len(brands), 3) if brands else None,
                "latency_ms": round(ms[name], 1),
                "query": case["query"],
            })

    pq = pd.DataFrame(per_query)
    Path("eval").mkdir(exist_ok=True)
    pq.to_csv("eval/results_per_query.csv", index=False)
    ans = pq[pq["ndcg10"].notna()]

    lines = [f"# IntentIQ offline evaluation", "",
             f"Run {datetime.now():%Y-%m-%d %H:%M} · {len(cases)} queries · top {K} · labels: "
             f"{'hand labels for ' + str(len(human)) + ' queries, criteria for the rest' if human else 'criteria from golden.yaml'}", ""]

    def table(df, title):
        lines.extend([f"## {title}", "", "| System | NDCG@10 | MRR | P@5 | Zero-result rate | Avg latency (ms) |", "|---|---|---|---|---|---|"])
        for name in SYSTEMS:
            d = df[df["system"] == name]
            z = (pq[(pq["system"] == name) & pq["query_id"].isin(d["query_id"])]["n_results"] == 0).mean()
            lines.append(f"| {name} | {d['ndcg10'].mean():.3f} | {d['mrr'].mean():.3f} | {d['p5'].mean():.3f} | {z:.0%} | "
                         f"{d['latency_ms'].mean():.0f} |")
        lines.append("")

    table(ans, "Overall")
    for cat in sorted(ans["category"].unique()):
        table(ans[ans["category"] == cat], cat.capitalize())

    lines.extend(["## Head to head (NDCG@10 per query)", "", "| Comparison | Wins | Ties | Losses |", "|---|---|---|---|"])
    piv = ans.pivot(index="query_id", columns="system", values="ndcg10")
    for x, y in [("A", "raw"), ("B", "raw"), ("A", "B")]:
        d = piv[x] - piv[y]
        lines.append(f"| {x} vs {y} | {(d > 0.001).sum()} | {(d.abs() <= 0.001).sum()} | {(d < -0.001).sum()} |")
    lines.append("")

    lines.extend(["## Brand concentration (bias audit)", "",
                  "Exposure = share of all result slots across the golden queries. Catalog = share of products in the catalog.", "",
                  "| System | Avg unique brands in top 10 | Avg top-brand share | Top 3 brands' exposure |", "|---|---|---|---|"])
    for name in SYSTEMS:
        d = pq[pq["system"] == name]
        tot = sum(exposure[name].values()) or 1
        top3 = ", ".join(f"{b} {n / tot:.0%}" for b, n in exposure[name].most_common(3))
        lines.append(f"| {name} | {d['unique_brands'].mean():.1f} | {d['top_brand_share'].mean():.0%} | {top3} |")
    cat_counts = Counter(s.df["brand"].astype(str).str.lower())
    total = sum(cat_counts.values())
    lines.extend(["", "Catalog share of the same brands: " + ", ".join(
        f"{b} {cat_counts[b] / total:.0%}" for b, _ in exposure["A"].most_common(3)), ""])

    lines.extend(["## Popularity bias", "", "| System | Median ratings count of shown products | Share with fewer than 20 ratings |",
                  "|---|---|---|"])
    for name in SYSTEMS:
        v = np.array(pop[name]) if pop[name] else np.array([0.0])
        lines.append(f"| {name} | {np.median(v):,.0f} | {(v < 20).mean():.0%} |")
    lines.append("")

    if unanswerable:
        lines.extend([f"## Queries with no relevant product in the catalog", "",
                      "These are excluded from NDCG. Each one is a catalog gap worth mentioning: " + ", ".join(unanswerable), ""])

    lines.extend(["## Worst queries for ranker A", "", "| Query | NDCG@10 | Raw NDCG@10 |", "|---|---|---|"])
    for qid, row in piv.sort_values("A").head(5).iterrows():
        q = pq[(pq["query_id"] == qid)]["query"].iloc[0]
        lines.append(f"| {qid}: {q} | {row['A']:.3f} | {row['raw']:.3f} |")
    lines.extend(["", "## Caveats", "",
                  "- Criteria labels reward the same specs ranker A scores, so they can flatter it. Hand labels from "
                  "`python src/label.py` remove that bias for the queries you label.",
                  "- Missing specs count as not meeting a criterion.",
                  "- 30 queries is a small set: treat differences under about 0.03 NDCG as noise."])

    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {a.out} and eval/results_per_query.csv")


if __name__ == "__main__":
    main()
