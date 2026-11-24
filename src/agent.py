"""Step 7: the IntentIQ shopping agent.

Each turn, an LLM (Groq, then Gemini, both free tiers) decides what to do: ask ONE clarifying
question, search with updated filters, or compare products. The scoring is still done by rank.py,
so results stay explainable and testable. Before an answer is shown, code checks that every
product reference and price in it came from the tools; if not, a safe template answer is used.
With no API key, or when the free limits run out, a rule-based planner takes over, so it never breaks.

Try it in the terminal:
    python src/agent.py            # uses an LLM if a key is in .env
    python src/agent.py --rules    # rules only, no LLM
"""
import argparse
import json
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import llm  # noqa: E402
from parse import (CONFIGS, add, any_has, detect_category, explicit_headphones,  # noqa: E402
                   explicit_laptop, merge, parse_price)
from rank import rank  # noqa: E402

ALL_PRIORITIES = sorted({p for cfg in CONFIGS.values() for p in cfg["priorities"]} |
                        {p for cfg in CONFIGS.values() for uc in cfg["use_cases"].values() for p in uc.get("priorities", [])})
NUM_KEYS = ["min_ram_gb", "min_storage_gb", "max_weight_lb"]
BOOL_KEYS = ["needs_gpu", "anc", "wireless", "wired"]
MAX_STEPS = 4

TOOLS = [
    {"type": "function", "function": {
        "name": "search_products",
        "description": "Search the catalog and rank the results. Call this before recommending anything, and call it "
                       "again whenever the user adds or changes a requirement (cheaper, lighter, no brand X...).",
        "parameters": {"type": "object", "properties": {
            "request": {"type": "string", "description": "Everything the user wants so far, combined into one plain request, "
                                                         "e.g. 'lightweight laptop for MBA and video editing, long battery'."},
            "category": {"type": "string", "enum": ["laptops", "headphones"]},
            "max_price": {"type": "number", "description": "Budget in US dollars, if the user gave one."},
            "min_ram_gb": {"type": "number"},
            "min_storage_gb": {"type": "number"},
            "max_weight_lb": {"type": "number"},
            "needs_gpu": {"type": "boolean", "description": "Dedicated graphics card required (gaming, 3D)."},
            "anc": {"type": "boolean", "description": "Active noise cancelling required."},
            "wireless": {"type": "boolean"},
            "wired": {"type": "boolean"},
            "form_factor": {"type": "string", "enum": ["over-ear", "on-ear", "in-ear"]},
            "exclude_brands": {"type": "array", "items": {"type": "string"}},
            "priorities": {"type": "array", "items": {"type": "string", "enum": ALL_PRIORITIES}},
        }, "required": ["request", "category"]}}},
    {"type": "function", "function": {
        "name": "compare_products",
        "description": "Side-by-side specs for 2 or 3 products from the latest results, by their ref numbers.",
        "parameters": {"type": "object", "properties": {"refs": {"type": "array", "items": {"type": "integer"}}},
                       "required": ["refs"]}}},
    {"type": "function", "function": {
        "name": "ask_user",
        "description": "Ask ONE short clarifying question when you cannot search well yet, for example when you "
                       "can't tell whether they want laptops or headphones. This ends your turn.",
        "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}}},
]

SYSTEM = """You are IntentIQ, a shopping assistant for laptops and headphones.
Rules:
- The catalog has only laptops and headphones. Prices are from a 2023 Amazon dataset, not live prices.
- If you can't tell the category or the request is too vague to search, call ask_user with ONE short question.
  Don't ask more than one question in a row: if they've already answered one, search with your best reading.
- Before recommending anything, call search_products. When the user changes or adds anything ("cheaper",
  "lighter", "no HP", "actually for gaming"), call search_products again with the full combined request
  and updated filters. Keep earlier requirements unless the user drops them.
- Translate needs into filters: video editing -> min_ram_gb 16; gaming laptop -> needs_gpu; flights -> anc;
  "hates wires" -> wireless; "cheaper" -> a max_price below the prices shown.
- Only mention products from the latest search results, by their ref number in square brackets, e.g. [1].
  Never invent products, specs or prices. Only write prices exactly as listed in the results.
- Answer in under 120 words: the best pick and why, one or two alternatives with their trade-off, and any
  requirement that nothing met. Use the 'why' notes honestly, including misses like "(you wanted 16GB)"."""

CHEAPER = re.compile(r"\b(cheaper|less expensive|lower price|more affordable|too expensive|too pricey|cheapest)\b", re.I)
LIGHTER = re.compile(r"\b(lighter|less heavy|more portable|too heavy)\b", re.I)
NEG_BRAND = re.compile(r"\b(?:no|not|without|except|exclude|avoid|skip|hate)\s+(?:any\s+)?([a-z][a-z0-9&\-]*)", re.I)
WIRELESS_HINT = ["cordless", "wire-free", "wire free", "no wires", "true wireless", "hate wires", "hates wires", "without wires"]


def intent_for(text, cat):
    """Same logic as parse_query, but with the category already decided."""
    t = text.lower()
    cfg = CONFIGS[cat]
    out = {"category": cat, "max_price": parse_price(t), "use_cases": [], "priorities": [], "constraints": {}}
    c = out["constraints"]
    for name, uc in cfg["use_cases"].items():
        if any_has(t, uc["triggers"]):
            out["use_cases"].append(name)
            add(out["priorities"], uc.get("priorities", []))
            for k, v in uc.get("requires", {}).items():
                merge(c, k, v)
    for name, words in cfg["priorities"].items():
        if any_has(t, words):
            add(out["priorities"], [name])
    if cat == "laptops":
        explicit_laptop(t, c)
    else:
        if any_has(t, WIRELESS_HINT):
            c["wireless"] = True
        explicit_headphones(t, c, out["priorities"])
    return out


def _f(x, fmt="{:g}"):
    try:
        return "?" if x is None or pd.isna(x) else fmt.format(float(x))
    except (TypeError, ValueError):
        return str(x)


def spec_str(r, cat):
    if cat == "laptops":
        return (f"{_f(r['ram_gb'])}GB RAM, {_f(r['storage_gb'])}GB storage, {_f(r['screen_in'])}in, "
                f"{_f(r['weight_lb'], '{:.1f}')}lb, {r.get('processor') or '?'}")
    anc = "noise cancelling" if bool(r.get("has_anc")) and not pd.isna(r.get("has_anc")) else "no ANC listed"
    return f"{r.get('form_factor') or '?'}, {r.get('connection') or '?'}, {anc}"


class Agent:
    def __init__(self, searcher=None, cat_mean=None, max_sessions=2000):
        if searcher is None:
            from search import Searcher
            searcher = Searcher()
        self.s = searcher
        self.cat_mean = cat_mean or searcher.df.groupby("category")["avg_rating"].mean().to_dict()
        self.brands = {c: set(searcher.df.loc[searcher.df["category"] == c, "brand"].dropna().astype(str).str.lower())
                       for c in CONFIGS}
        self.sessions = OrderedDict()
        self.max_sessions = max_sessions

    # ---------- session state ----------
    def session(self, sid, reset=False):
        if reset or sid not in self.sessions:
            self.sessions[sid] = {"request": "", "category": None, "overrides": {}, "intent": None,
                                  "results": None, "history": [], "pending": "", "known_prices": set(), "asked": 0}
        self.sessions.move_to_end(sid)
        while len(self.sessions) > self.max_sessions:
            self.sessions.popitem(last=False)
        return self.sessions[sid]

    # ---------- the one place that searches ----------
    def _search(self, sess, request, cat, ov, ranker):
        intent = intent_for(request, cat)
        if ov.get("max_price"):
            intent["max_price"] = float(ov["max_price"])
        c = intent["constraints"]
        for k in NUM_KEYS:
            if ov.get(k):
                c[k] = float(ov[k])
        for k in BOOL_KEYS:
            if ov.get(k) is True:
                c[k] = True
        if c.get("wireless") and ov.get("wireless"):
            c.pop("wired", None)
        if c.get("wired") and ov.get("wired"):
            c.pop("wireless", None)
        if ov.get("form_factor") in ("over-ear", "on-ear", "in-ear"):
            c["form_factor"] = ov["form_factor"]
        add(intent["priorities"], [p for p in ov.get("priorities", []) if p in ALL_PRIORITIES])
        excl = sorted({str(b).lower().strip() for b in ov.get("exclude_brands", []) if str(b).strip()})
        intent["exclude_brands"] = excl

        cand = self.s.search(request, cat, intent["max_price"], k=100, mode="hybrid")
        if excl and not cand.empty:
            cand = cand[~cand["brand"].astype(str).str.lower().isin(excl)]
        out = rank(cand, intent, cat, self.cat_mean[cat], ranker, k=8, brand_cap=3) if not cand.empty else cand
        sess.update(request=request, category=cat, overrides=ov, intent=intent, results=out, pending="")
        sess["known_prices"].update(round(float(p), 2) for p in out["price"]) if len(out) else None

        understood = {k: v for k, v in intent.items() if v not in (None, [], {})}
        if out.empty:
            return {"understood": understood, "results": [],
                    "note": "No products matched. Suggest raising the budget or dropping a requirement."}
        return {"understood": understood, "results": [
            {"ref": i, "id": str(r["product_id"]), "title": str(r["title"])[:90], "brand": str(r["brand"]),
             "price": round(float(r["price"]), 2), "rating": _f(r["avg_rating"], "{:.1f}"),
             "num_ratings": int(r["num_ratings"]), "specs": spec_str(r, cat), "why": r.get("why", "")}
            for i, (_, r) in enumerate(out.iterrows(), 1)]}

    def _compare(self, sess, refs):
        df = sess["results"]
        if df is None or df.empty:
            return {"error": "No results yet. Search first."}
        rows = []
        for ref in refs[:3]:
            if isinstance(ref, int) and 1 <= ref <= len(df):
                r = df.iloc[ref - 1]
                rows.append({"ref": ref, "title": str(r["title"])[:90], "price": round(float(r["price"]), 2),
                             "rating": _f(r["avg_rating"], "{:.1f}"), "num_ratings": int(r["num_ratings"]),
                             "specs": spec_str(r, sess["category"])})
        return {"products": rows} if len(rows) >= 2 else {"error": "Give 2 or 3 valid ref numbers from the latest results."}

    # ---------- LLM planner ----------
    def _respond_llm(self, sess, msg, ranker):
        msgs = [{"role": "system", "content": SYSTEM}]
        if sess["category"]:
            msgs.append({"role": "system", "content": "Current search state: " + json.dumps(
                {"category": sess["category"], "request_so_far": sess["request"], "filters": sess["overrides"]})})
        if sess["asked"]:
            msgs.append({"role": "system", "content": "You already asked a clarifying question. Search now with your best reading."})
        msgs += sess["history"][-10:] + [{"role": "user", "content": msg}]
        res = {"clarify": False, "searched": False, "llm_calls": 0, "provider": None, "reply": None, "tools": []}
        for _ in range(MAX_STEPS):
            m, res["provider"] = llm.chat(msgs, TOOLS)
            res["llm_calls"] += 1
            calls = m.get("tool_calls") or []
            if not calls:
                res["reply"] = (m.get("content") or "").strip() or None
                return res
            msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["function"]["name"], "arguments": c["function"].get("arguments") or "{}"}}
                for c in calls]})
            for c in calls:
                name = c["function"]["name"]
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                res["tools"].append(name)
                if name == "ask_user":
                    res.update(clarify=True, reply=str(args.get("question") or "Could you tell me a bit more?")[:300])
                    return res
                if name == "search_products":
                    cat = args.get("category")
                    if cat not in CONFIGS:
                        out = {"error": "category must be laptops or headphones; ask the user"}
                    else:
                        ov = {k: v for k, v in args.items() if k not in ("request", "category") and v is not None}
                        out = self._search(sess, str(args.get("request") or msg)[:400], cat, ov, ranker)
                        res["searched"] = True
                elif name == "compare_products":
                    out = self._compare(sess, args.get("refs") or [])
                else:
                    out = {"error": f"unknown tool {name}"}
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(out, default=str)[:8000]})
        return res  # ran out of steps: reply stays None and the template is used

    # ---------- rule-based planner (no LLM) ----------
    def _apply_refinements(self, text, cat, ov, sess):
        df = sess["results"]
        if CHEAPER.search(text) and df is not None and len(df):
            cur = ov.get("max_price") or (sess["intent"] or {}).get("max_price") or float("inf")
            ov["max_price"] = round(min(cur, float(df["price"].max())) * 0.8)  # 20% below the priciest result shown
        if (p := parse_price(text)) is not None:
            ov["max_price"] = p
        if cat == "laptops" and LIGHTER.search(text) and df is not None and len(df):
            w = pd.to_numeric(df["weight_lb"], errors="coerce").dropna()
            if len(w):
                ov["max_weight_lb"] = round(float(w.median()) * 0.9, 1)
        for m in NEG_BRAND.finditer(text):
            b = m.group(1).lower()
            if b in self.brands.get(cat, set()):
                ov.setdefault("exclude_brands", [])
                if b not in ov["exclude_brands"]:
                    ov["exclude_brands"].append(b)

    def _respond_rules(self, sess, msg, ranker):
        t = msg.lower()
        res = {"clarify": False, "searched": False, "llm_calls": 0, "provider": None, "reply": None, "tools": []}
        new_cat = detect_category(t)
        have = sess["category"] is not None and sess["results"] is not None
        refinement = have and new_cat in (None, sess["category"]) and not (new_cat and len(t.split()) >= 4)
        if refinement:
            cat = sess["category"]
            ov = {**sess["overrides"], "exclude_brands": list(sess["overrides"].get("exclude_brands", []))}
            self._apply_refinements(t, cat, ov, sess)
            extra = LIGHTER.sub("", CHEAPER.sub("", NEG_BRAND.sub("", msg))).strip(" ,.!")
            request = f"{sess['request']} {extra}".strip()
        else:
            request = f"{sess['pending']} {msg}".strip()
            cat = detect_category(request.lower())
            if cat is None:
                sess["pending"] = request
                sess["asked"] += 1
                res.update(clarify=True, reply="Are you shopping for laptops or headphones, and what will it mostly be used for?")
                return res
            ov = {}
            self._apply_refinements(t, cat, ov, sess)
        self._search(sess, request, cat, ov, ranker)
        res["searched"] = True
        res["tools"].append("search_products")
        return res

    # ---------- safety check + fallback text ----------
    def verify(self, text, sess):
        df = sess["results"]
        n = 0 if df is None else len(df)
        ids = set(df["product_id"].astype(str)) if n else set()
        problems = []
        for m in re.finditer(r"\[(\d{1,2})\]", text):
            if not 1 <= int(m.group(1)) <= n:
                problems.append(f"ref [{m.group(1)}] is not in the results")
        for m in re.finditer(r"\b(B0[A-Z0-9]{8})\b", text):
            if m.group(1) not in ids:
                problems.append(f"product id {m.group(1)} is not in the results")
        budget = (sess["intent"] or {}).get("max_price")
        for m in re.finditer(r"\$\s?(\d[\d,]*(?:\.\d{1,2})?)", text):
            v = float(m.group(1).replace(",", ""))
            if not any(abs(v - p) <= 1.0 for p in sess["known_prices"]) and not (budget and abs(v - budget) <= 1):
                problems.append(f"price ${v:g} does not match any listed product")
        return problems

    def template(self, sess):
        df, intent = sess["results"], sess["intent"] or {}
        if df is None:
            return "Tell me what you're shopping for (laptops or headphones) and what you'll use it for."
        if df.empty:
            return "Nothing matched all of that. Try a higher budget or drop one requirement."
        budget = intent.get("max_price")
        lines = [f"Here are {len(df)} {sess['category']} ranked for you" + (f" under ${budget:,.0f}" if budget else "") + "."]
        for n, (_, r) in enumerate(df.head(3).iterrows(), 1):
            why = " · ".join(str(r.get("why", "")).split(" | ")[:2])
            lines.append(f"[{n}] {r['brand']} at ${float(r['price']):,.0f}: {why}")
        lines.append("Tell me what to change, e.g. 'cheaper', 'lighter' or 'no HP'.")
        return "\n".join(lines)

    # ---------- public entry point ----------
    def respond(self, sid, message, ranker="A", reset=False, mode="auto"):
        """mode: auto (LLM if a key exists, else rules), llm, or rules."""
        t0 = time.perf_counter()
        sess = self.session(sid, reset)
        msg = message.strip()[:500]
        out, fallback = None, None
        if mode == "llm" or (mode == "auto" and llm.available()):
            try:
                out = self._respond_llm(sess, msg, ranker)
                out["mode"] = "llm"
            except llm.LLMUnavailable as e:
                fallback = str(e)[:300]
            except Exception as e:  # never let an LLM hiccup break the page
                fallback = f"agent error: {type(e).__name__}: {e}"[:300]
        if out is None:
            out = self._respond_rules(sess, msg, ranker)
            out["mode"] = "rules"
        out["guard_problems"] = []
        if out["clarify"]:
            sess["asked"] += 1 if out["mode"] == "llm" else 0
        else:
            sess["asked"] = 0
            if out["reply"]:
                out["guard_problems"] = self.verify(out["reply"], sess)
            if not out["reply"] or out["guard_problems"]:
                out["reply"] = self.template(sess)
        sess["history"] += [{"role": "user", "content": msg}, {"role": "assistant", "content": out["reply"]}]
        return {
            "reply": out["reply"], "clarify": out["clarify"], "mode": out["mode"], "provider": out["provider"],
            "fallback_reason": fallback, "guard": bool(out["guard_problems"]), "guard_problems": out["guard_problems"],
            "category": sess["category"], "intent": sess["intent"], "tools": out["tools"], "llm_calls": out["llm_calls"],
            "results": sess["results"] if out["searched"] else None,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", action="store_true", help="rule-based planner only, no LLM")
    a = ap.parse_args()
    agent = Agent()
    mode = "rules" if a.rules else "auto"
    print(f"IntentIQ agent ({'rules only' if a.rules else 'LLM: ' + (', '.join(p['name'] for p in llm.available()) or 'no key, using rules')})")
    print("Type a request. 'new' starts over, 'quit' exits.\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in ("quit", "exit"):
            break
        if not q:
            continue
        reset = q == "new"
        if reset:
            agent.session("cli", reset=True)
            print("(started over)\n")
            continue
        r = agent.respond("cli", q, mode=mode)
        print(f"\nagent [{r['mode']}{'/' + r['provider'] if r['provider'] else ''}, {r['latency_ms']:.0f} ms]> {r['reply']}")
        if r["fallback_reason"]:
            print(f"  (LLM unavailable, used rules: {r['fallback_reason'][:120]})")
        if r["guard"]:
            print(f"  (safety check replaced the LLM answer: {'; '.join(r['guard_problems'])})")
        if r["results"] is not None and len(r["results"]):
            for n, (_, x) in enumerate(r["results"].iterrows(), 1):
                print(f"   [{n}] ${float(x['price']):>5.0f}  {str(x['brand'])[:10]:<10} {str(x['title'])[:60]}")
        print()
