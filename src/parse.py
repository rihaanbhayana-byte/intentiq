"""Step 3: turn a natural-language query into structured intent (rule-based, no API needed)."""
import json
import re
import sys
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
CONFIGS = {p.stem: yaml.safe_load(p.read_text()) for p in sorted(CONFIG_DIR.glob("*.yaml"))}


def has(text, phrase):
    """True if the phrase appears as whole words ('mba' must not match inside another word)."""
    pat = r"(?<![a-z0-9])" + re.escape(phrase.lower()) + r"(?![a-z0-9])"
    return re.search(pat, text) is not None


def any_has(text, phrases):
    return any(has(text, p) for p in phrases)


# ---------- price ----------
NUM = r"(\d[\d,]*(?:\.\d+)?)(?!\d|\.\d)\s*(k(?![a-z]))?"
NOT_A_UNIT = r"(?!\s*(?:lbs?|pounds?|gb|tb|inch(?:es)?|in\b|hours?|hrs?|oz|kg|mm|hz|%))"
PRICE_PATTERNS = [
    r"(?:under|below|less than|cheaper than|up to|max(?:imum)?|no more than|within|at most)\s*(?:of\s*)?\$?\s*" + NUM + NOT_A_UNIT,
    r"\$\s*" + NUM + r"\s*(?:budget|max|or less|or under|limit)",
    r"budget\s*(?:of|is|:)?\s*\$?\s*" + NUM + NOT_A_UNIT,
]


def parse_price(text):
    for pat in PRICE_PATTERNS:
        m = re.search(pat, text)
        if m:
            value = float(m.group(1).replace(",", ""))
            return value * 1000 if m.group(2) else value
    return None


# ---------- category ----------
def detect_category(text):
    hits = {cat: sum(has(text, k) for k in cfg["keywords"]) for cat, cfg in CONFIGS.items()}
    best = max(hits, key=hits.get)
    if hits[best] == 0 or list(hits.values()).count(hits[best]) > 1:
        return None  # nothing matched, or a tie
    return best


# ---------- explicit specs the user typed ----------
def explicit_laptop(text, c):
    if m := re.search(r"(\d+)\s*gb\s*(?:of\s*)?(?:ram|memory)", text):
        c["min_ram_gb"] = float(m.group(1))
    if m := re.search(r"(\d+(?:\.\d+)?)\s*(gb|tb)\s*(?:ssd|storage|hard drive|hdd|nvme)", text):
        c["min_storage_gb"] = float(m.group(1)) * (1000 if m.group(2) == "tb" else 1)
    if m := re.search(r"(?:under|below|less than|max|at most|no more than)\s*(\d+(?:\.\d+)?)\s*(?:lbs?|pounds?)", text):
        c["max_weight_lb"] = float(m.group(1))
    if m := re.search(r"(\d{2}(?:\.\d)?)\s*(?:-?\s*inch(?:es)?|in\b|\")", text):
        c["screen_in"] = float(m.group(1))


def explicit_headphones(text, c, priorities):
    if "noise_cancelling" in priorities:
        c["anc"] = True
    if any_has(text, ["wireless", "bluetooth"]):
        c["wireless"] = True
        c.pop("wired", None)
    elif any_has(text, ["wired"]):
        c["wired"] = True
        c.pop("wireless", None)
    for form, words in [
        ("over-ear", ["over-ear", "over ear"]),
        ("on-ear", ["on-ear", "on ear"]),
        ("in-ear", ["in-ear", "in ear", "earbuds", "earbud"]),
    ]:
        if any_has(text, words):
            c["form_factor"] = form


# ---------- main parser ----------
def merge(constraints, key, value):
    if key.startswith("min_"):
        constraints[key] = max(constraints.get(key, 0), value)  # strictest minimum wins
    else:
        constraints.setdefault(key, value)


def add(lst, items):
    for i in items:
        if i not in lst:
            lst.append(i)


def parse_query(query, default_category=None):
    text = query.lower()
    cat = detect_category(text) or default_category
    out = {"category": cat, "max_price": parse_price(text), "use_cases": [], "priorities": [], "constraints": {}}
    if cat is None:
        return out
    cfg = CONFIGS[cat]
    c = out["constraints"]

    for name, uc in cfg["use_cases"].items():
        if any_has(text, uc["triggers"]):
            out["use_cases"].append(name)
            add(out["priorities"], uc.get("priorities", []))
            for k, v in uc.get("requires", {}).items():
                merge(c, k, v)
    for name, words in cfg["priorities"].items():
        if any_has(text, words):
            add(out["priorities"], [name])

    if cat == "laptops":
        explicit_laptop(text, c)
    else:
        explicit_headphones(text, c, out["priorities"])
    return out


if __name__ == "__main__":
    tests = sys.argv[1:] or [
        "I need a laptop for my MBA, occasional video editing, good battery life, under $1,000",
        "noise cancelling headphones for flights and commuting",
        "wireless earbuds for the gym under $50",
        "gaming laptop with 16GB RAM under 1k",
        "ultralight laptop under 3.5 lbs, 14 inch, 512GB SSD under $900",
        "studio headphones wired over-ear under $200",
        "something good for my dad",
    ]
    for q in tests:
        print("\n>", q)
        for k, v in parse_query(q).items():
            print(f"   {k}: {v}")