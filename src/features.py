"""Step 1c: turn messy spec text into clean columns.
Reads data/electronics_clean.parquet, writes data/products.parquet."""
import json
import re

import numpy as np
import pandas as pd

df = pd.read_parquet("data/electronics_clean.parquet")
specs = df["details_json"].apply(json.loads)

RENEWED = re.compile(r"renewed|refurbished|restored|pre-owned", re.I)


def get(d, *keys):
    """First non-empty value among the given spec keys."""
    for k in keys:
        v = d.get(k)
        if v not in (None, "", "Unknown"):
            return str(v)
    return None


def to_gb(text):
    """'16 GB' / '1 TB' / '512GB SSD' -> GB as a number."""
    if not text:
        return np.nan
    m = re.search(r"(\d+(?:\.\d+)?)\s*(tb|gb|mb)", text, re.I)
    if not m:
        return np.nan
    n, unit = float(m.group(1)), m.group(2).lower()
    return n * 1000 if unit == "tb" else n / 1000 if unit == "mb" else n


def to_inches(text):
    if not text:
        return np.nan
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else np.nan


def to_pounds(text):
    if not text:
        return np.nan
    m = re.search(r"(\d+(?:\.\d+)?)\s*(pounds?|lbs?|ounces?|oz|kilograms?|kg|grams?|g)\b", text, re.I)
    if not m:
        return np.nan
    n, u = float(m.group(1)), m.group(2).lower()
    if u.startswith(("ounce", "oz")):
        return n / 16
    if u.startswith(("kilo", "kg")):
        return n * 2.2046
    if u.startswith(("gram", "g")):
        return n / 453.6
    return n


TITLE_RAM = re.compile(r"(\d+)\s*GB\s*(?:DDR\d\w*\s*)?(?:RAM|Memory|DDR\d)", re.I)
TITLE_STORAGE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:TB|GB)\s*(?:NVMe\s*)?(?:PCIe\s*)?(?:SSD|HDD|eMMC|Solid State|Flash|NVMe)", re.I
)
TITLE_SCREEN = re.compile(r"(\d{2}(?:\.\d)?)\s*(?:\"|”|-?inch|in\b)", re.I)


def laptop_row(title, d):
    ram = to_gb(get(d, "RAM", "Ram Memory Installed Size"))
    if np.isnan(ram):
        m = TITLE_RAM.search(title)
        ram = float(m.group(1)) if m else np.nan
    sto = to_gb(get(d, "Hard Disk Size", "Hard Drive"))
    if np.isnan(sto):
        m = TITLE_STORAGE.search(title)
        sto = to_gb(m.group(0)) if m else np.nan
    scr = to_inches(get(d, "Standing screen display size"))
    if np.isnan(scr):
        m = TITLE_SCREEN.search(title)
        scr = float(m.group(1)) if m else np.nan
    return {
        "ram_gb": ram if 2 <= ram <= 128 else np.nan,
        "storage_gb": sto if 16 <= sto <= 8000 else np.nan,
        "screen_in": scr if 10 <= scr <= 19 else np.nan,
        "weight_lb": to_pounds(get(d, "Item Weight")),
        "processor": get(d, "Processor", "CPU Model", "Processor Brand"),
        "gpu": get(d, "Graphics Coprocessor", "Card Description"),
        "os": get(d, "Operating System"),
    }


def headphone_row(title, feats, d):
    conn = get(d, "Connectivity Technology")
    text = (title + " " + " ".join(feats)).lower()
    return {
        "form_factor": get(d, "Form Factor"),
        "connectivity": conn,
        "is_wireless": bool(re.search(r"wireless|bluetooth", (conn or "").lower() + " " + text)),
        "has_anc": bool(re.search(r"noise[\s-]*(?:cancel|reduc)|\banc\b", text)),
        "weight_lb": to_pounds(get(d, "Item Weight")),
        "color": get(d, "Color"),
    }


rows = []
for (_, r), d in zip(df.iterrows(), specs):
    if r["category"] == "laptops":
        rows.append(laptop_row(r["title"], d))
    else:
        rows.append(headphone_row(r["title"], list(r["features"]), d))

out = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
out["is_renewed"] = out["title"].str.contains(RENEWED)
for c in ["is_wireless", "has_anc"]:
    out[c] = out[c].astype("boolean")

out.to_parquet("data/products.parquet", index=False)
print("SAVED data/products.parquet:", out.shape)
print("renewed:", out.groupby("category")["is_renewed"].sum().to_dict())

# ---------- sanity checks ----------
lap = out[(out.category == "laptops") & ~out.is_renewed]
print(f"\n=== LAPTOPS (not renewed): {len(lap)} ===")
for c in ["ram_gb", "storage_gb", "screen_in", "weight_lb", "processor", "gpu", "os"]:
    print(f"  {c}: {lap[c].notna().mean():.0%} filled")
print("  price <= $1000:", (lap.price <= 1000).sum())
print(lap[["ram_gb", "storage_gb", "screen_in", "weight_lb"]].describe().round(1).to_string())
print("  ram_gb counts:", lap.ram_gb.value_counts().head(6).to_dict())
print("  screen_in counts:", lap.screen_in.value_counts().head(6).to_dict())
print("  sample rows:")
for _, r in lap.sample(5, random_state=2).iterrows():
    print(f"   - ${r.price:.0f} | {r.ram_gb}GB RAM | {r.storage_gb}GB | {r.screen_in}in | {r.weight_lb}lb | {r.title[:55]}")

hp = out[(out.category == "headphones") & ~out.is_renewed]
print(f"\n=== HEADPHONES (not renewed): {len(hp)} ===")
print("  form_factor:", hp.form_factor.value_counts().head(8).to_dict())
print("  connectivity:", hp.connectivity.value_counts().head(8).to_dict())
print("  wireless share:", round(hp.is_wireless.astype(float).mean(), 2), "| ANC share:", round(hp.has_anc.astype(float).mean(), 2))
print("  filled: weight", f"{hp.weight_lb.notna().mean():.0%}", "| color", f"{hp.color.notna().mean():.0%}")
