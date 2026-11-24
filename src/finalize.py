"""Step 1d: final cleanup. Reads data/products.parquet, writes data/catalog.parquet."""
import re

import numpy as np
import pandas as pd

df = pd.read_parquet("data/products.parquet")
print("start:", df.shape)

# 1. Default catalog excludes renewed/refurbished items and duplicate ids
df = df[~df["is_renewed"]].drop_duplicates("product_id").copy()
print("after dropping renewed + duplicates:", df.shape)

# 2. Drop accessories that slipped into headphones (ear pads, tips, stands...)
ACCESSORY = re.compile(
    r"ear\s?pads?|earpads?|ear\s?cushions?|replacement\s+(?:ear|cushion|pad|cable|tips?|cord)|"
    r"ear\s?tips?|eartips?|headphone\s+(?:stand|hanger|holder)|headset\s+stand|splitter|"
    r"(?:hard|carrying|protective|silicone|travel)\s+case\s+(?:for|compatible)|case\s+cover|skin\s+cover",
    re.I,
)
is_hp = df["category"] == "headphones"
bad_form = df["form_factor"].fillna("").str.lower().isin(["case", "bi-fold"])
bad_title = df["title"].str.contains(ACCESSORY)
drop = is_hp & (bad_form | bad_title)
print(f"\ndropping {drop.sum()} accessory-like headphone rows, e.g.:")
for t in df[drop]["title"].head(8):
    print("  -", t[:90])
df = df[~drop].copy()

# 3. Normalize headphone labels
FORM_RULES = [
    ("open-ear", r"open[\s-]?ear|bone conduction|air conduction"),
    ("over-ear", r"\bover[\s-]?ear|closed[\s-]?back|around[\s-]?ear"),
    ("on-ear", r"\bon[\s-]?ear|supra[\s-]?aural"),
    ("in-ear", r"\bin[\s-]?ear|earbuds?|earphones?|canal"),
]


def norm_form(ff, title):
    for text in (str(ff or "").lower(), title.lower()):
        for name, pat in FORM_RULES:
            if re.search(pat, text):
                return name
    return None


def norm_conn(conn, title):
    s = str(conn or "").lower()
    wireless = bool(re.search(r"wireless|bluetooth", s))
    wired = bool(re.search(r"wired|usb|3\.5|aux", s))
    if not (wireless or wired):
        t = title.lower()
        wireless = bool(re.search(r"wireless|bluetooth", t))
        wired = bool(re.search(r"\bwired\b", t))
    if wireless and wired:
        return "both"
    return "wireless" if wireless else "wired" if wired else None


hp_mask = df["category"] == "headphones"
df.loc[hp_mask, "form_factor"] = [norm_form(f, t) for f, t in zip(df.loc[hp_mask, "form_factor"], df.loc[hp_mask, "title"])]
df["connection"] = None
df.loc[hp_mask, "connection"] = [norm_conn(c, t) for c, t in zip(df.loc[hp_mask, "connectivity"], df.loc[hp_mask, "title"])]
df["is_wireless"] = df["connection"].isin(["wireless", "both"])

# 4. Laptop sanity: plausible weight, and must have RAM + screen size
lap_mask = df["category"] == "laptops"
bad_w = lap_mask & ((df["weight_lb"] < 0.8) | (df["weight_lb"] > 9))
df.loc[bad_w, "weight_lb"] = np.nan
missing = lap_mask & (df["ram_gb"].isna() | df["screen_in"].isna())
print(f"\ndropping {missing.sum()} laptops missing RAM or screen size")
df = df[~missing].copy()

# 5. Tidy brand names ("Visit the Sony Store" -> "Sony", "LENOVO" -> "Lenovo")
def tidy_brand(b):
    b = str(b or "Unknown").strip()
    b = re.sub(r"^Visit the\s+|\s+Store$|^Brand:\s*", "", b)
    return b.title() if b.isupper() and len(b) > 4 else b


df["brand"] = df["brand"].apply(tidy_brand)

# 6. Text that will be embedded in Step 2
def s(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)


def g(x, unit):
    return "" if pd.isna(x) else f"{x:g}{unit}"


def build_text(r):
    feats = " ".join(list(r["features"])[:4])[:400]
    if r["category"] == "laptops":
        spec = ", ".join(
            p for p in [g(r["ram_gb"], "GB RAM"), g(r["storage_gb"], "GB storage"),
                        g(r["screen_in"], " inch screen"), s(r["processor"]), s(r["gpu"]), s(r["os"])] if p
        )
    else:
        anc = ", active noise cancelling" if bool(r["has_anc"]) else ""
        spec = f"{s(r['form_factor'])} {s(r['connection'])} headphones{anc}".strip()
    return f"{r['title']}. {spec}. {feats}"


df["search_text"] = df.apply(build_text, axis=1)
df = df.reset_index(drop=True)
df.to_parquet("data/catalog.parquet", index=False)

# ---------- sanity checks ----------
print("\nSAVED data/catalog.parquet:", df.shape, df["category"].value_counts().to_dict())
hp = df[df.category == "headphones"]
lap = df[df.category == "laptops"]
print("\nHEADPHONES")
print("  form_factor:", hp["form_factor"].value_counts(dropna=False).to_dict())
print("  connection:", hp["connection"].value_counts(dropna=False).to_dict())
print("  ANC share:", round(hp["has_anc"].astype(float).mean(), 2))
print("  top brands:", hp["brand"].value_counts().head(8).to_dict())
print("LAPTOPS")
print("  <= $1000:", (lap["price"] <= 1000).sum(), "of", len(lap))
print("  weight filled:", f"{lap['weight_lb'].notna().mean():.0%}")
print("  top brands:", lap["brand"].value_counts().head(8).to_dict())
print("\nExample embedding text:")
for cat in ["laptops", "headphones"]:
    print(" -", df[df.category == cat].sample(1, random_state=0)["search_text"].iloc[0][:300])
    