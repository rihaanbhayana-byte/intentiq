"""Shared relevance checks used by evaluate.py and evaluate_agent.py.

A golden query lists "must" criteria (a product that fails any of them is not relevant)
and optional "nice" criteria (meeting them makes it highly relevant).
Unknown values (missing spec) count as NOT meeting the criterion, which is the strict choice.
"""
import re

import pandas as pd

from rank import GPU


def _num(x):
    try:
        return None if x is None or pd.isna(x) else float(x)
    except (TypeError, ValueError):
        return None


def _truthy(x):
    try:
        return False if x is None or pd.isna(x) else bool(x)
    except (TypeError, ValueError):
        return bool(x)


def _text(r):
    return f"{r.get('title', '')} {r.get('search_text', '')}"


def _all_regex(text, pats):
    pats = [pats] if isinstance(pats, str) else pats
    return all(re.search(p, text, re.I) for p in pats)


def _le(col):
    return lambda r, v: (x := _num(r.get(col))) is not None and x <= v


def _ge(col):
    return lambda r, v: (x := _num(r.get(col))) is not None and x >= v


CHECKS = {
    "max_price": _le("price"),
    "min_ram_gb": _ge("ram_gb"),
    "min_storage_gb": _ge("storage_gb"),
    "max_weight_lb": _le("weight_lb"),
    "min_screen_in": _ge("screen_in"),
    "max_screen_in": _le("screen_in"),
    "min_rating": _ge("avg_rating"),
    "min_num_ratings": _ge("num_ratings"),
    "needs_gpu": lambda r, v: bool(GPU.search(f"{r.get('gpu', '')} {r.get('title', '')}")) == bool(v),
    "anc": lambda r, v: _truthy(r.get("has_anc")) == bool(v),
    "wireless": lambda r, v: _truthy(r.get("is_wireless")) == bool(v),
    "wired": lambda r, v: (r.get("connection") in ("wired", "both")) == bool(v),
    "form_factor": lambda r, v: r.get("form_factor") == v,
    "title_regex": lambda r, v: _all_regex(str(r.get("title", "")), v),
    "text_regex": lambda r, v: _all_regex(_text(r), v),
    "not_brand": lambda r, v: str(r.get("brand", "")).lower() not in {b.lower() for b in ([v] if isinstance(v, str) else v)},
}


def meets(r, criteria):
    """True if the product row meets every criterion in the dict."""
    for key, val in (criteria or {}).items():
        if key not in CHECKS:
            raise KeyError(f"Unknown criterion '{key}'. Valid: {', '.join(CHECKS)}")
        if not CHECKS[key](r, val):
            return False
    return True


def grade(r, case):
    """0 = not relevant, 1 = meets every 'must', 2 = meets every 'must' and every 'nice'."""
    if not meets(r, case.get("must")):
        return 0
    nice = case.get("nice")
    return 2 if not nice or meets(r, nice) else 1
