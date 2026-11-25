"""Free LLM client for the agent: Groq first, Gemini second, and an error if neither works
(the agent then falls back to its rule-based planner, so the app keeps working at $0).

Keys go in a .env file in the project folder (never commit it):
    GROQ_API_KEY=gsk_...
    GEMINI_API_KEY=AIza...
Optional model overrides:
    GROQ_MODEL=openai/gpt-oss-120b
    GEMINI_MODEL=gemini-flash-latest
Uses only the Python standard library, so there is nothing new to install.
"""
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_env():
    p = ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()

PROVIDERS = [
    {"name": "groq", "key": "GROQ_API_KEY", "model": os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
     "url": "https://api.groq.com/openai/v1/chat/completions"},
    {"name": "gemini", "key": "GEMINI_API_KEY", "model": os.environ.get("GEMINI_MODEL", "gemini-flash-latest"),
     "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"},
]
_cooldown = {}  # provider name -> time until which we skip it (after a rate-limit error)
STATS = {"calls": 0, "errors": 0, "rate_limited": 0}


class LLMUnavailable(Exception):
    pass


def available():
    return [p for p in PROVIDERS if os.environ.get(p["key"])]


def chat(messages, tools=None, temperature=0.2, timeout=8):
    """Send an OpenAI-style chat request. Returns (assistant_message_dict, provider_name)."""
    errors = []
    for p in available():
        if _cooldown.get(p["name"], 0) > time.time():
            errors.append(f"{p['name']}: rate-limited, cooling down")
            continue
        body = {"model": p["model"], "messages": messages, "temperature": temperature}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        req = urllib.request.Request(
            p["url"], data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ[p['key']]}",
                     "User-Agent": "IntentIQ/1.0"})
        STATS["calls"] += 1
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
            return data["choices"][0]["message"], p["name"]
        except urllib.error.HTTPError as e:
            STATS["errors"] += 1
            if e.code == 429:
                STATS["rate_limited"] += 1
                _cooldown[p["name"]] = time.time() + 60
            detail = e.read()[:300].decode("utf-8", "replace")
            errors.append(f"{p['name']}: HTTP {e.code} {detail}")
        except Exception as e:  # network error, timeout, bad JSON
            STATS["errors"] += 1
            errors.append(f"{p['name']}: {type(e).__name__}: {e}")
    raise LLMUnavailable("; ".join(errors) or "no API key in .env")


if __name__ == "__main__":
    print("Keys found:", [p["name"] for p in available()] or "none")
    msg, who = chat([{"role": "user", "content": "Reply with exactly: IntentIQ connected"}])
    print(f"{who}: {msg.get('content')}")
