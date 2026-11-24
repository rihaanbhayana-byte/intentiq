"""Step 7: test the agent on multi-turn conversations, rules vs. LLM.

Run from the project folder:
    python src/evaluate_agent.py --mode rules          # free, instant, no key needed
    python src/evaluate_agent.py --mode llm            # uses your Groq/Gemini key (about 25-40 calls)
    python src/evaluate_agent.py --mode rules llm      # both, side by side

Writes eval/agent_results.md.
"""
import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

import llm  # noqa: E402
from agent import Agent  # noqa: E402
from criteria import meets  # noqa: E402


def check(expect, r, sess):
    """Returns a list of failed checks for one turn."""
    fails = []
    intent = sess["intent"] or {}
    cons = intent.get("constraints", {})
    df = r["results"] if r["results"] is not None else sess["results"]
    for key, val in expect.items():
        if key == "clarify" and r["clarify"] != val:
            fails.append(f"clarify should be {val}")
        elif key == "category" and sess["category"] != val:
            fails.append(f"category {sess['category']} != {val}")
        elif key == "max_price" and not (intent.get("max_price") and abs(intent["max_price"] - val) <= 1):
            fails.append(f"max_price {intent.get('max_price')} != {val}")
        elif key == "max_price_below" and not (intent.get("max_price") and intent["max_price"] < val):
            fails.append(f"max_price {intent.get('max_price')} not below {val}")
        elif key == "constraints":
            for k, v in val.items():
                if k.startswith("min_") and not (cons.get(k) is not None and cons[k] >= v):
                    fails.append(f"constraint {k}={cons.get(k)} (want >= {v})")
                elif not k.startswith("min_") and cons.get(k) != v:
                    fails.append(f"constraint {k}={cons.get(k)} (want {v})")
        elif key == "no_brand":
            if df is None or (df["brand"].astype(str).str.lower() == val.lower()).any():
                fails.append(f"brand {val} still shown" if df is not None else "no results")
        elif key == "min_results" and (df is None or len(df) < val):
            fails.append(f"fewer than {val} results")
        elif key == "results_meet":
            if df is None or not all(meets(x, val) for _, x in df.iterrows()):
                fails.append(f"not every result meets {val}")
        elif key == "top3_meet":
            if df is None or sum(meets(x, val) for _, x in df.head(3).iterrows()) < 2:
                fails.append(f"fewer than 2 of top 3 meet {val}")
    return fails


def run(agent, cases, mode, pause):
    rows = []
    for case in cases:
        sid = f"eval-{mode}-{case['id']}"
        agent.session(sid, reset=True)
        for i, turn in enumerate(case["turns"], 1):
            r = agent.respond(sid, turn["say"], mode=mode)
            fails = check(turn.get("expect", {}), r, agent.sessions[sid])
            rows.append({"case": case["id"], "turn": i, "say": turn["say"], "pass": not fails, "fails": fails,
                         "mode_used": r["mode"], "clarify": r["clarify"], "want_clarify": turn.get("expect", {}).get("clarify"),
                         "guard": r["guard"], "llm_calls": r["llm_calls"], "ms": r["latency_ms"], "tools": r["tools"]})
            mark = "PASS" if not fails else "FAIL " + "; ".join(fails)
            print(f"[{mode}] {case['id']} t{i} ({r['mode']}, {r['latency_ms']:.0f} ms) {mark}")
            if mode == "llm" and pause:
                time.sleep(pause)  # stay well inside the free per-minute limits
    return rows


def summary(rows):
    n = len(rows)
    convs = {}
    for r in rows:
        convs.setdefault(r["case"], True)
        convs[r["case"]] &= r["pass"]
    should = [r for r in rows if r["want_clarify"] is True]
    asked = [r for r in rows if r["clarify"]]
    good_ask = [r for r in asked if r["want_clarify"] is not False and r["pass"]]
    ms = sorted(r["ms"] for r in rows)
    return {
        "turns passed": f"{sum(r['pass'] for r in rows)}/{n}",
        "conversations fully passed": f"{sum(convs.values())}/{len(convs)}",
        "asked when it should": f"{sum(r['clarify'] for r in should)}/{len(should)}",
        "questions that were needed": f"{len(good_ask)}/{len(asked)}" if asked else "0/0",
        "answers blocked by the safety check": sum(r["guard"] for r in rows),
        "turns answered by the rule planner": sum(r["mode_used"] == "rules" for r in rows),
        "LLM calls": sum(r["llm_calls"] for r in rows),
        "median latency (ms)": f"{ms[len(ms) // 2]:.0f}",
        "p95 latency (ms)": f"{ms[min(len(ms) - 1, int(len(ms) * 0.95))]:.0f}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", nargs="+", default=["rules"], choices=["rules", "llm"])
    ap.add_argument("--cases", default="eval/agent_cases.yaml")
    ap.add_argument("--pause", type=float, default=2.0, help="seconds between LLM turns")
    a = ap.parse_args()
    if "llm" in a.mode and not llm.available():
        sys.exit("No GROQ_API_KEY or GEMINI_API_KEY in .env. Add one, or run with --mode rules.")

    cases = yaml.safe_load(open(a.cases))
    agent = Agent()
    results = {m: run(agent, cases, m, a.pause) for m in a.mode}
    sums = {m: summary(rows) for m, rows in results.items()}

    lines = ["# IntentIQ agent evaluation", "", f"Run {datetime.now():%Y-%m-%d %H:%M} · {len(cases)} conversations", "",
             "| Metric | " + " | ".join(a.mode) + " |", "|---|" + "---|" * len(a.mode)]
    for metric in next(iter(sums.values())):
        lines.append(f"| {metric} | " + " | ".join(str(sums[m][metric]) for m in a.mode) + " |")
    lines += ["", "## Failed turns", ""]
    for m, rows in results.items():
        for r in rows:
            if not r["pass"]:
                lines.append(f"- **{m}** {r['case']} turn {r['turn']} (\"{r['say']}\"): {'; '.join(r['fails'])}")
    lines += ["", "The safety check counts LLM answers that mentioned a product, ref or price not returned by the tools; "
              "those were replaced with a template answer before the user saw them."]
    Path("eval").mkdir(exist_ok=True)
    Path("eval/agent_results.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines) + "\n\nSaved eval/agent_results.md")


if __name__ == "__main__":
    main()
