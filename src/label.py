"""Step 6 (optional but recommended): hand-label results so the evaluation isn't graded by its own rules.

For each golden query it pools the top 5 from raw, ranker A and ranker B (about 8-12 products),
shuffles them so you can't tell which system found what, and asks you for a grade:
    2 = great match   1 = acceptable   0 = wrong   s = skip   q = save and quit
Progress is saved to eval/labels.csv after every answer, so you can stop and resume any time.

Run:  python src/label.py            (all queries)
      python src/label.py L01 H02    (only these)
"""
import csv
import os
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from evaluate import run_systems  # noqa: E402
from search import Searcher  # noqa: E402

OUT = Path("eval/labels.csv")


def specs(r):
    if r["category"] == "laptops":
        return f"{r['ram_gb']}GB RAM · {r['storage_gb']}GB · {r['screen_in']}in · {r['weight_lb']}lb · {r['processor']}"
    return f"{r['form_factor']} · {r['connection']} · ANC: {r['has_anc']}"


def main():
    only = set(sys.argv[1:])
    cases = [c for c in yaml.safe_load(open("eval/golden.yaml")) if not only or c["id"] in only]
    done = set()
    if OUT.exists():
        done = {(r["query_id"], r["product_id"]) for r in csv.DictReader(open(OUT))}
    new_file = not OUT.exists()
    f = open(OUT, "a", newline="")
    w = csv.writer(f)
    if new_file:
        w.writerow(["query_id", "product_id", "grade"])

    s = Searcher()
    prior = s.df.groupby("category")["avg_rating"].mean().to_dict()
    for case in cases:
        results, _ = run_systems(s, case, prior[case["category"]])
        pool = {}
        for df in results.values():
            for _, r in df.head(5).iterrows():
                pool[str(r["product_id"])] = r
        todo = [(pid, r) for pid, r in pool.items() if (case["id"], pid) not in done]
        if not todo:
            continue
        random.shuffle(todo)
        print(f"\n=== {case['id']}: {case['query']}  ({len(todo)} to label)")
        for pid, r in todo:
            print(f"\n  ${r['price']:.0f} · {r['brand']} · ★{r['avg_rating']} ({int(r['num_ratings']):,})")
            print(f"  {str(r['title'])[:110]}")
            print(f"  {specs(r)}")
            while True:
                g = input("  grade 2/1/0, s=skip, q=quit: ").strip().lower()
                if g in ("0", "1", "2", "s", "q"):
                    break
            if g == "q":
                f.close()
                print(f"Saved to {OUT}. Run the same command to continue.")
                return
            if g != "s":
                w.writerow([case["id"], pid, g])
                f.flush()
    f.close()
    print(f"\nAll done. Now run: python src/evaluate.py --labels {OUT}")


if __name__ == "__main__":
    main()
