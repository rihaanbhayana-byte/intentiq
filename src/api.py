"""Step 5: FastAPI backend. Run from anywhere:  uvicorn api:app --app-dir src --reload"""
import hashlib
import os
import sqlite3
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)  # the data paths in search.py are relative to the project root

from parse import parse_query  # noqa: E402
from rank import rank  # noqa: E402
from search import Searcher  # noqa: E402

DB = ROOT / "data" / "events.db"
app = FastAPI(title="IntentIQ")
searcher = Searcher()
CAT_MEAN = searcher.df.groupby("category")["avg_rating"].mean().to_dict()

SPEC_FIELDS = {
    "laptops": [("ram_gb", "RAM (GB)"), ("storage_gb", "Storage (GB)"), ("screen_in", "Screen (in)"),
                ("weight_lb", "Weight (lb)"), ("processor", "Processor"), ("os", "OS")],
    "headphones": [("form_factor", "Type"), ("connection", "Connection"), ("has_anc", "Noise cancelling"),
                   ("weight_lb", "Weight (lb)"), ("color", "Color")],
}


# ---------- helpers ----------
def py(v):
    """Convert numpy/pandas values to plain JSON-safe Python (NaN -> None)."""
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.floating):
        v = float(v)
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def card(r):
    cat = r["category"]
    return {
        "product_id": r["product_id"], "category": cat, "title": r["title"], "brand": py(r["brand"]),
        "price": py(r["price"]), "avg_rating": py(r["avg_rating"]), "num_ratings": py(r["num_ratings"]),
        "image_url": py(r["image_url"]),
        "specs": {label: py(r[col]) for col, label in SPEC_FIELDS[cat]},
        "why": py(r.get("why")), "score": py(r.get("final")),
    }


def assign_ranker(session_id):
    """Deterministic 50/50 split: the same visitor always sees the same ranker."""
    return "A" if int(hashlib.md5(session_id.encode()).hexdigest(), 16) % 2 == 0 else "B"


def db():
    con = sqlite3.connect(DB)
    con.execute(
        """CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, session_id TEXT, ranker TEXT, event TEXT,
        query TEXT, category TEXT, product_id TEXT, position INTEGER)"""
    )
    return con


def log_events(rows):
    con = db()
    con.executemany(
        "INSERT INTO events (ts, session_id, ranker, event, query, category, product_id, position) "
        "VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    con.commit()
    con.close()


@lru_cache(maxsize=512)
def run_search(query, category, ranker):
    """Cached: the same query + category + ranker returns instantly the second time."""
    intent = parse_query(query, default_category=category)
    cat = category or intent["category"]
    if cat is None:
        return {"needs_category": True, "category": None, "intent": intent, "results": []}
    cand = searcher.search(query, cat, intent["max_price"], k=100, mode="hybrid")
    if cand.empty:
        return {"needs_category": False, "category": cat, "intent": intent, "results": []}
    out = rank(cand, intent, cat, CAT_MEAN[cat], ranker, k=8, brand_cap=3)
    return {"needs_category": False, "category": cat, "intent": intent,
            "results": [card(r) for _, r in out.iterrows()]}


# ---------- endpoints ----------
class SearchReq(BaseModel):
    query: str
    session_id: str
    category: str | None = None


class FeedbackReq(BaseModel):
    session_id: str
    event: str  # click | up | down | compare
    product_id: str
    query: str = ""
    category: str = ""
    position: int | None = None


@app.get("/")
def home():
    return FileResponse(ROOT / "app" / "index.html")


@app.post("/search")
def search(req: SearchReq):
    q = req.query.strip()
    if not 1 <= len(q) <= 300:
        raise HTTPException(400, "Query must be 1-300 characters.")
    if req.category not in (None, "laptops", "headphones"):
        raise HTTPException(400, "Unknown category.")
    ranker = assign_ranker(req.session_id)

    t0 = time.perf_counter()
    hits_before = run_search.cache_info().hits
    res = run_search(q, req.category, ranker)
    cached = run_search.cache_info().hits > hits_before
    ms = round((time.perf_counter() - t0) * 1000, 1)

    now, cat, sid = time.time(), res.get("category") or "", req.session_id
    rows = [(now, sid, ranker, "search", q, cat, "", None)]
    if res["needs_category"]:
        rows.append((now, sid, ranker, "needs_category", q, cat, "", None))
    elif not res["results"]:
        rows.append((now, sid, ranker, "zero_result", q, cat, "", None))
    for pos, r in enumerate(res["results"], 1):
        rows.append((now, sid, ranker, "impression", q, cat, r["product_id"], pos))
    log_events(rows)
    return {**res, "ranker": ranker, "latency_ms": ms, "cached": cached}


@app.post("/feedback")
def feedback(req: FeedbackReq):
    if req.event not in ("click", "up", "down", "compare"):
        raise HTTPException(400, "Unknown event.")
    log_events([(time.time(), req.session_id, assign_ranker(req.session_id), req.event,
                 req.query[:300], req.category, req.product_id, req.position)])
    return {"ok": True}


def highlights(cards, cat):
    out = []

    def pick(label, key, best, fmt):
        vals = [(key(c), c) for c in cards]
        if any(v is None for v, _ in vals):
            return
        v, c = best(vals, key=lambda t: t[0])
        out.append(f"{label}: {c['brand']} ({fmt(v)})")

    pick("Lowest price", lambda c: c["price"], min, lambda v: f"${v:,.0f}")
    pick("Highest rated", lambda c: c["avg_rating"], max, lambda v: f"{v:.1f} stars")
    if cat == "laptops":
        pick("Most RAM", lambda c: c["specs"]["RAM (GB)"], max, lambda v: f"{v:g} GB")
        pick("Lightest", lambda c: c["specs"]["Weight (lb)"], min, lambda v: f"{v:g} lb")
    return out


@app.get("/compare")
def compare(ids: str = Query(...)):
    id_list = [i for i in ids.split(",") if i][:3]
    df = searcher.df[searcher.df["product_id"].isin(id_list)]
    if len(df) < 2:
        raise HTTPException(400, "Pick at least 2 products to compare.")
    if df["category"].nunique() > 1:
        raise HTTPException(400, "You can only compare products in the same category.")
    cat = df["category"].iloc[0]
    cards = sorted((card(r) for _, r in df.iterrows()), key=lambda c: id_list.index(c["product_id"]))
    return {"category": cat, "products": cards, "labels": [l for _, l in SPEC_FIELDS[cat]],
            "highlights": highlights(cards, cat)}


@app.get("/stats")
def stats():
    """Aggregate event counts per ranker (used later for the A/B write-up)."""
    con = db()
    rows = con.execute("SELECT ranker, event, COUNT(*) FROM events GROUP BY ranker, event").fetchall()
    con.close()
    out = {}
    for ranker, event, n in rows:
        out.setdefault(ranker, {})[event] = n
    return out
    