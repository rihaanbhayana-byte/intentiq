"""Step 4: rank retrieved candidates using the parsed intent, with explanations."""
import argparse
import math
import re

import numpy as np
import pandas as pd

from parse import parse_query
from search import Searcher

WEIGHTS = {
    "A": {"relevance": 0.30, "spec_fit": 0.25, "priority_fit": 0.15, "quality": 0.20, "popularity": 0.10},
    "B": {"relevance": 0.30, "spec_fit": 0.25, "priority_fit": 0.15, "quality": 0.30, "popularity": 0.00},
}
GPU = re.compile(r"nvidia|geforce|rtx|gtx|radeon rx|radeon pro|quadro|arc a\d", re.I)
BATTERY = [
    re.compile(r"(\d{1,2})\s*(?:\+\s*)?(?:hours?|hrs?|h)\b\s*(?:of\s*)?(?:battery|playtime|play time|playback|talk time)", re.I),
    re.compile(r"battery(?:\s+life)?\s*(?:of|up to|:)?\s*(?:up to\s*)?(\d{1,2})\s*(?:hours?|hrs?|h)\b", re.I),
]


def num(x):
    return None if x is None or pd.isna(x) else float(x)


def truthy(x):
    return False if x is None or pd.isna(x) else bool(x)


def clip01(x):
    return float(min(1.0, max(0.0, x)))


def short(have, need):
    """1.0 if the requirement is met, otherwise a steep penalty (8GB vs 16GB -> 0.25)."""
    return 1.0 if have >= need else (have / need) ** 2


def battery_hours(text):
    for pat in BATTERY:
        m = pat.search(text or "")
        if m and 3 <= float(m.group(1)) <= 70:
            return float(m.group(1))
    return None


def spec_fit(r, cons, cat):
    """How well the product meets the hard-ish requirements. Returns (score or None, notes)."""
    parts, notes = [], []
    if cat == "laptops":
        ram, sto = num(r["ram_gb"]), num(r["storage_gb"])
        wt, scr = num(r["weight_lb"]), num(r["screen_in"])
        if "min_ram_gb" in cons:
            need = cons["min_ram_gb"]
            if ram is None:
                parts.append(0.3)
            else:
                parts.append(short(ram, need))
                notes.append(f"{ram:g}GB RAM " + (f"meets your {need:g}GB need" if ram >= need else f"(you wanted {need:g}GB)"))
        if "min_storage_gb" in cons:
            need = cons["min_storage_gb"]
            if sto is None:
                parts.append(0.3)
            else:
                parts.append(short(sto, need))
                notes.append(f"{sto:g}GB storage " + ("is enough" if sto >= need else f"(you wanted {need:g}GB)"))
        if cons.get("needs_gpu"):
            has = bool(GPU.search(f"{r['gpu']} {r['title']}"))
            parts.append(1.0 if has else 0.0)
            notes.append("dedicated GPU" if has else "no dedicated GPU")
        if "max_weight_lb" in cons:
            mx = cons["max_weight_lb"]
            if wt is None:
                parts.append(0.4)
            else:
                parts.append(1.0 if wt <= mx else clip01(mx / wt) ** 2)
                notes.append(f"{wt:g} lb " + ("is under your limit" if wt <= mx else f"(limit {mx:g})"))
        if "screen_in" in cons:
            want = cons["screen_in"]
            if scr is None:
                parts.append(0.3)
            else:
                ok = abs(scr - want) <= 0.7
                parts.append(1.0 if ok else 0.0)
                notes.append(f"{scr:g}-inch screen" + ("" if ok else f" (you wanted {want:g})"))
    else:
        if cons.get("anc"):
            ok = truthy(r["has_anc"])
            parts.append(1.0 if ok else 0.0)
            notes.append("noise cancelling" if ok else "no noise cancelling listed")
        if cons.get("wireless"):
            ok = truthy(r["is_wireless"])
            parts.append(1.0 if ok else 0.0)
            notes.append("wireless" if ok else "not wireless")
        if cons.get("wired"):
            ok = r["connection"] in ("wired", "both")
            parts.append(1.0 if ok else 0.0)
            notes.append("wired" if ok else "not wired")
        if "form_factor" in cons:
            ok = r["form_factor"] == cons["form_factor"]
            parts.append(1.0 if ok else 0.0)
            notes.append(cons["form_factor"] if ok else f"not {cons['form_factor']}")
    return (float(np.mean(parts)) if parts else None), notes


def priority_fit(r, priorities, cat):
    """Softer preferences (portability, performance, battery...). Returns (score or None, notes)."""
    parts, notes = [], []
    for p in priorities:
        if p == "portability" and cat == "laptops":
            wt = num(r["weight_lb"])
            if wt is not None:
                parts.append(clip01((5.0 - wt) / 2.5))  # 2.5 lb -> 1.0, 5 lb -> 0.0
                if wt <= 3.5:
                    notes.append(f"light at {wt:g} lb")
        elif p == "performance" and cat == "laptops":
            ram, sto = num(r["ram_gb"]), num(r["storage_gb"])
            gpu = bool(GPU.search(f"{r['gpu']} {r['title']}"))
            parts.append(float(np.mean([clip01((ram or 0) / 16), clip01((sto or 0) / 512), 1.0 if gpu else 0.4])))
        elif p == "battery":
            h = battery_hours(r["search_text"])
            if h:
                parts.append(clip01(h / (12 if cat == "laptops" else 30)))
                notes.append(f"lists {h:g} hrs battery")
            else:
                parts.append(0.3)  # unknown: no battery field in the catalog yet
        elif p == "noise_cancelling":
            parts.append(1.0 if truthy(r["has_anc"]) else 0.0)
    return (float(np.mean(parts)) if parts else None), notes


def rank(cand, intent, cat, prior_mean, ranker="A", k=8, brand_cap=3):
    w = WEIGHTS[ranker]
    cons, pri = intent["constraints"], intent["priorities"]
    rel_max = cand["score"].max() or 1.0
    log_max = math.log1p(cand["num_ratings"].max()) or 1.0
    finals, whys = [], []
    for _, r in cand.iterrows():
        sf, n1 = spec_fit(r, cons, cat)
        pf, n2 = priority_fit(r, pri, cat)
        v, m = float(r["num_ratings"]), 50.0
        bayes = (v / (v + m)) * float(r["avg_rating"]) + (m / (v + m)) * prior_mean
        comps = {
            "relevance": r["score"] / rel_max,
            "spec_fit": sf,
            "priority_fit": pf,
            "quality": clip01((bayes - 3.0) / 2.0),  # 3 stars -> 0, 5 stars -> 1
            "popularity": math.log1p(v) / log_max,
        }
        used = {c: x for c, x in comps.items() if x is not None and w[c] > 0}
        finals.append(sum(w[c] * x for c, x in used.items()) / sum(w[c] for c in used))
        notes = (n1 + n2)[:4] + [f"rated {r['avg_rating']:.1f} by {int(v):,} buyers"]
        whys.append(" | ".join(notes))
    out = cand.assign(final=finals, why=whys).sort_values("final", ascending=False)

    chosen, counts, skipped = [], {}, []
    for i, r in out.iterrows():
        b = str(r["brand"]).lower()
        if brand_cap and counts.get(b, 0) >= brand_cap:
            skipped.append(i)
            continue
        chosen.append(i)
        counts[b] = counts.get(b, 0) + 1
        if len(chosen) == k:
            break
    chosen += skipped[: k - len(chosen)]
    return out.loc[chosen]


def show(df, cat, with_why=True):
    for n, (_, r) in enumerate(df.iterrows(), 1):
        if cat == "laptops":
            spec = f"{r['ram_gb']:g}GB/{r['storage_gb']:g}GB/{r['screen_in']:g}in/{r['weight_lb']:.1f}lb"
        else:
            spec = f"{r['form_factor']}/{r['connection']}" + ("/ANC" if truthy(r["has_anc"]) else "")
        score = f"{r['final']:.2f}" if "final" in df.columns else "  - "
        print(f"{n:>2}. ${r['price']:>5.0f}  {str(r['brand'])[:11]:<11} {spec:<28} {score}  {r['title'][:55]}")
        if with_why and "why" in df.columns:
            print(f"      why: {r['why']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--category", choices=["laptops", "headphones"], default=None)
    ap.add_argument("--ranker", choices=["A", "B"], default="A")
    ap.add_argument("--brand-cap", type=int, default=3)
    ap.add_argument("--raw", action="store_true", help="show retrieval order only, no re-ranking")
    ap.add_argument("-k", type=int, default=8)
    a = ap.parse_args()

    intent = parse_query(a.query, default_category=a.category)
    cat = a.category or intent["category"]
    if cat is None:
        print("Couldn't tell whether you want laptops or headphones. Add --category laptops|headphones.")
        print("(In the real product the app would ask this as a follow-up question.)")
        raise SystemExit
    print("\nIntent:", {k: v for k, v in intent.items() if v not in (None, [], {})})

    s = Searcher()
    cand = s.search(a.query, cat, intent["max_price"], k=100, mode="hybrid")
    if cand.empty:
        print("No products matched. Try a higher budget.")
        raise SystemExit
    prior = float(s.df.loc[s.df["category"] == cat, "avg_rating"].mean())

    if a.raw:
        print(f"\n[RAW retrieval order, no re-ranking] {a.query!r}\n")
        show(cand.head(a.k), cat, with_why=False)
    else:
        print(f"\n[RANKED, ranker {a.ranker}, brand cap {a.brand_cap}] {a.query!r}\n")
        show(rank(cand, intent, cat, prior, a.ranker, a.k, a.brand_cap), cat)