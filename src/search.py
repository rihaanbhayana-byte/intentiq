"""Step 2b: hybrid search = vector (FAISS) + keyword (BM25), merged with reciprocal rank fusion."""
import argparse
import re

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN.findall(text.lower())


class Searcher:
    def __init__(self):
        self.df = pd.read_parquet("data/catalog.parquet")
        self.emb = np.load("data/embeddings.npy")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.parts = {}  # one FAISS index + one BM25 index per category
        for cat in self.df["category"].unique():
            idx = np.where(self.df["category"].values == cat)[0]
            index = faiss.IndexFlatIP(self.emb.shape[1])
            index.add(self.emb[idx])
            bm25 = BM25Okapi([tokenize(t) for t in self.df["search_text"].values[idx]])
            self.parts[cat] = (idx, index, bm25)

    def search(self, query, category, max_price=None, k=10, mode="hybrid", pool=100):
        idx, index, bm25 = self.parts[category]
        n = len(idx)
        ok = np.ones(n, dtype=bool)
        if max_price is not None:
            ok = self.df["price"].values[idx] <= max_price  # hard constraint

        # vector ranking over the whole category
        q = self.model.encode([query], normalize_embeddings=True).astype("float32")
        _, order = index.search(q, n)
        vec_order = [i for i in order[0] if ok[i]]

        # keyword ranking (only items that match at least one query word)
        scores = bm25.get_scores(tokenize(query))
        bm_order = [i for i in np.argsort(-scores) if ok[i] and scores[i] > 0]

        # reciprocal rank fusion: each list gives 1/(60+rank) to its items
        fused = {}
        if mode in ("vector", "hybrid"):
            for r, i in enumerate(vec_order[:pool]):
                fused[i] = fused.get(i, 0) + 1 / (60 + r + 1)
        if mode in ("bm25", "hybrid"):
            for r, i in enumerate(bm_order[:pool]):
                fused[i] = fused.get(i, 0) + 1 / (60 + r + 1)

        top = sorted(fused, key=fused.get, reverse=True)[:k]
        out = self.df.iloc[idx[top]].copy()
        out["score"] = [fused[i] for i in top]
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--category", default="laptops", choices=["laptops", "headphones"])
    ap.add_argument("--max-price", type=float, default=None)
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "vector", "bm25"])
    ap.add_argument("-k", type=int, default=8)
    a = ap.parse_args()

    res = Searcher().search(a.query, a.category, a.max_price, a.k, a.mode)
    print(f"\n[{a.mode}] {a.query!r} in {a.category}, max_price={a.max_price}\n")
    for n, (_, r) in enumerate(res.iterrows(), 1):
        if a.category == "laptops":
            spec = f"{r.ram_gb:g}GB/{r.storage_gb:g}GB/{r.screen_in:g}in"
        else:
            spec = f"{r.form_factor}/{r.connection}" + ("/ANC" if r.has_anc else "")
        print(f"{n:>2}. ${r.price:>6.0f}  {r.brand[:12]:<12} {spec:<22} {r.title[:60]}")