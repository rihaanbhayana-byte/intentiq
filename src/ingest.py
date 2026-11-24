"""Step 1b: extract laptops + headphones from the Electronics shard and save a clean file."""
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

# Skips the download if the file is already on disk
path = hf_hub_download(
    repo_id="McAuley-Lab/Amazon-Reviews-2023",
    repo_type="dataset",
    revision="refs/pr/10",
    filename="raw_meta_Electronics/full-00000-of-00010.parquet",
    local_dir=RAW,
)
df = pd.read_parquet(path)
print("loaded:", df.shape)


def to_list(x):
    if x is None or isinstance(x, float):
        return []
    return [str(i) for i in x]


df["cats"] = df["categories"].apply(to_list)

# ---- Exploration: what category names actually exist? ----
def top_cats(keyword, n=12):
    c = Counter(cat for cats in df["cats"] for cat in cats if keyword in cat.lower())
    return c.most_common(n)


print("\ncategory names containing 'laptop':", top_cats("laptop"))
print("category names containing 'headphone':", top_cats("headphone"))

# ---- Filtering ----
LAPTOP_CATS = {"Laptops"}
HEADPHONE_CATS = {"Headphones", "Earbud Headphones", "On-Ear Headphones", "Over-Ear Headphones"}


def label(cats):
    s = set(cats)
    if s & LAPTOP_CATS:
        return "laptops"
    if s & HEADPHONE_CATS:
        return "headphones"
    return None


df["price_num"] = pd.to_numeric(df["price"], errors="coerce")
base = df[df["title"].notna() & df["price_num"].notna() & (df["rating_number"] >= 5)].copy()
base["category"] = base["cats"].apply(label)
base = base.dropna(subset=["category"])

# Price floors remove most accessories that slip through
keep = ((base["category"] == "laptops") & (base["price_num"] >= 150)) | (
    (base["category"] == "headphones") & (base["price_num"] >= 10)
)
base = base[keep].copy()


def parse_details(x):
    try:
        return json.loads(x) if isinstance(x, str) else {}
    except Exception:
        return {}


def first_image(imgs):
    try:
        large = imgs.get("large")
        return str(large[0]) if large is not None and len(large) > 0 else None
    except Exception:
        return None


base["details_dict"] = base["details"].apply(parse_details)
base["brand"] = [
    (d.get("Brand") or s or "Unknown") for d, s in zip(base["details_dict"], base["store"])
]

clean = pd.DataFrame(
    {
        "product_id": base["parent_asin"],
        "category": base["category"],
        "title": base["title"],
        "brand": base["brand"],
        "price": base["price_num"],
        "avg_rating": base["average_rating"],
        "num_ratings": base["rating_number"],
        "features": base["features"].apply(to_list),
        "description": base["description"].apply(lambda x: " ".join(to_list(x))),
        "details_json": base["details_dict"].apply(json.dumps),
        "image_url": base["images"].apply(first_image),
    }
).reset_index(drop=True)

clean.to_parquet("data/electronics_clean.parquet", index=False)
print("\nSAVED data/electronics_clean.parquet:", clean.shape)

# ---- Sanity checks ----
for cat, g in clean.groupby("category"):
    print(f"\n===== {cat.upper()}: {len(g)} products =====")
    print(g["price"].describe().round(0).to_string())
    print("\n5 cheapest titles:")
    for t in g.nsmallest(5, "price")["title"]:
        print("  -", t[:90])
    print("\n5 random titles:")
    for t in g.sample(5, random_state=1)["title"]:
        print("  -", t[:90])
    keys = Counter(k for d in g["details_json"].apply(json.loads) for k in d)
    print("\nmost common spec fields:", keys.most_common(15))
    print("top brands:", g["brand"].value_counts().head(8).to_dict())
    