"""Step 6b: let the free LLM grade search results (LLM-as-judge), then spot-check it yourself.

    python src/judge.py              # grades the top 10 from raw, A and B for all 30 golden queries
    python src/judge.py --check 20   # you grade 20 random items it judged; prints how often you agree
    python src/evaluate.py --labels eval/labels_llm.csv

Resumable: queries already in eval/labels_llm.csv are skipped, so just run it again if it stops.
Grades: 2 = great match, 1 = acceptable with a clear weakness, 0 = wrong for the query.
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

import llm  # noqa: E402
from evaluate import run_systems  # noqa: E402
from search import Searcher  # noqa: E402

OUT = Path("eval/labels_llm.csv")
PROMPT = """You are grading shopping search results for relevance to one query.
Query: "{query}"  (category: {category})

For each product, give a grade:
2 = a great match: you would recommend it to this shopper as is
1 = acceptable but with a clear weakness (e.g. weak processor, a bit heavy, missing a nice-to-have)
0 = wrong: misses a stated requirement (budget, specs, type, brand/model asked for) or is the wrong kind of product
Judge only against the query, using the specs given. Prices are in USD.

Products:
{products}

Reply with JSON only, no other text:
{{"grades": [{{"ref": 1, "grade": 2, "reason": "under 8 words"}}, ...]}}"""


def describe(r):
    if r["category"] == "laptops":
        spec = f"{r['ram_gb']}GB RAM, {r['storage_gb']}GB storage, {r['screen_in']}in, {r['weight_lb']}lb, CPU {r['processor']}"
    else:
        spec = f"{r['form_factor']}, {r['connection']}, noise cancelling: {r['has_anc']}"
    return f"${float(r['price']):.0f} | {str(r['title'])[:140]} | {spec} | rating {r['avg_rating']} ({int(r['num_ratings'])})"


def parse(text, n):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError("no JSON in reply")
    out = {}
    for g in json.loads(m.group(0)).get("grades", []):
        ref, grade = int(g.get("ref", 0)), int(g.get("grade", -1))
        if 1 <= ref <= n and grade in (0, 1, 2):
            out[ref] = (grade, str(g.get("reason", ""))[:80])
    if len(out) < n * 0.8:
        raise ValueError(f"only {len(out)} of {n} grades parsed")
    return out


def judge(args):
    if not llm.available():
        sys.exit("No key in .env. Finish Step 6 first.")
    cases = yaml.safe_load(open("eval/golden.yaml"))
    done = {r["query_id"] for r in csv.DictReader(open(OUT))} if OUT.exists() else set()
    new = not OUT.exists()
    f = open(OUT, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["query_id", "product_id", "grade", "reason"])
    s = Searcher()
    prior = s.df.groupby("category")["avg_rating"].mean().to_dict()
    for case in cases:
        if case["id"] in done:
            continue
        results, _ = run_systems(s, case, prior[case["category"]])
        pool = {}
        for df in results.values():
            for _, r in df.head(10).iterrows():
                pool.setdefault(str(r["product_id"]), r)
        items = list(pool.items())
        lines = "\n".join(f"{i}. {describe(r)}" for i, (_, r) in enumerate(items, 1))
        msg = [{"role": "user", "content": PROMPT.format(query=case["query"], category=case["category"], products=lines)}]
        for attempt in range(4):
            try:
                reply, who = llm.chat(msg, temperature=0, timeout=90)
                grades = parse(reply.get("content"), len(items))
                break
            except (llm.LLMUnavailable, ValueError, json.JSONDecodeError) as e:
                wait = 30 * (attempt + 1)
                print(f"  {case['id']}: {str(e)[:120]} - waiting {wait}s and retrying")
                time.sleep(wait)
        else:
            print(f"  {case['id']}: skipped after 4 tries. Run the script again later to fill it in.")
            continue
        for ref, (pid, _) in enumerate(items, 1):
            if ref in grades:
                w.writerow([case["id"], pid, grades[ref][0], grades[ref][1]])
        f.flush()
        counts = [sum(g == k for g, _ in grades.values()) for k in (2, 1, 0)]
        print(f"{case['id']} ({who}): {len(grades)} graded  ->  2s: {counts[0]}  1s: {counts[1]}  0s: {counts[2]}")
        time.sleep(args.pause)  # stay inside the free tokens-per-minute limit
    f.close()
    print(f"\nSaved {OUT}. Next: python src/judge.py --check 20")


def check(args):
    rows = list(csv.DictReader(open(OUT)))
    queries = {c["id"]: c["query"] for c in yaml.safe_load(open("eval/golden.yaml"))}
    s = Searcher()
    df = s.df.set_index(s.df["product_id"].astype(str))
    random.seed(1)
    sample = random.sample(rows, min(args.check, len(rows)))
    agree, close, n = 0, 0, 0
    print("You'll grade the same products the AI graded, without seeing its grade.\n2 = great, 1 = acceptable, 0 = wrong\n")
    for i, row in enumerate(sample, 1):
        r = df.loc[row["product_id"]]
        r = r.iloc[0] if hasattr(r, "iloc") and r.ndim == 2 else r
        print(f"[{i}/{len(sample)}] Query: {queries[row['query_id']]}")
        print(f"  {describe(r)}")
        while (g := input("  your grade 2/1/0: ").strip()) not in ("0", "1", "2"):
            pass
        n += 1
        agree += int(g) == int(row["grade"])
        close += abs(int(g) - int(row["grade"])) <= 1
        if int(g) != int(row["grade"]):
            print(f"  AI said {row['grade']} ({row['reason']})")
        print()
    text = (f"# LLM judge spot check\n\n{n} random judged items re-graded by hand.\n\n"
            f"- Exact agreement: {agree}/{n} ({agree / n:.0%})\n- Within one grade: {close}/{n} ({close / n:.0%})\n")
    Path("eval/judge_agreement.md").write_text(text)
    print(text + "\nSaved eval/judge_agreement.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=int, default=0, help="spot-check this many AI grades by hand")
    ap.add_argument("--pause", type=float, default=10, help="seconds between queries")
    a = ap.parse_args()
    check(a) if a.check else judge(a)
