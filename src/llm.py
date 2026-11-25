"""Minimal OpenRouter chat client with retries, concurrency, and cost tracking."""
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
for line in ((ROOT / ".env").read_text().splitlines() if (ROOT / ".env").exists() else []):
    if "=" in line:
        k, v = line.strip().split("=", 1)
        os.environ.setdefault(k, v)

URL = "https://openrouter.ai/api/v1/chat/completions"
PRICES = {  # $ per 1M tokens (input, output), from the OpenRouter model list
    "google/gemini-2.5-flash-lite": (0.10, 0.40),
    "openai/gpt-4.1-nano": (0.10, 0.40),
    "openai/gpt-4o-mini": (0.15, 0.60),
}
_lock = threading.Lock()
SPENT = {"usd": 0.0, "calls": 0}
LOG = ROOT / "logs" / "llm_usage.jsonl"


def chat(model, messages, temperature=0.7, max_tokens=1500, tools=None, json_mode=False, retries=5):
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if tools:
        body["tools"] = tools
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}", "X-Title": "research-eval"}
    for a in range(retries):
        try:
            r = requests.post(URL, headers=headers, json=body, timeout=120)
            if r.status_code == 200:
                d = r.json()
                if "choices" not in d:
                    raise RuntimeError(str(d)[:300])
                u = d.get("usage", {})
                pi, po = PRICES.get(model, (1.0, 1.0))
                cost = u.get("prompt_tokens", 0) * pi / 1e6 + u.get("completion_tokens", 0) * po / 1e6
                with _lock:
                    SPENT["usd"] += cost; SPENT["calls"] += 1
                    with open(LOG, "a") as f:
                        f.write(json.dumps({"model": model, **u, "cost": cost}) + "\n")
                return d["choices"][0]["message"]
            if r.status_code in (402, 401, 403):
                raise SystemExit(f"OpenRouter error {r.status_code}: {r.text[:300]}")
            time.sleep(2 ** a)
        except (requests.RequestException, RuntimeError):
            time.sleep(2 ** a)
    return None


def parse_json(text):
    if not text:
        return None
    m = re.search(r"\{.*\}|\[.*\]", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0), strict=False)
    except json.JSONDecodeError:
        return None


def pmap(fn, items, workers=8):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


def total_spent():
    if not LOG.exists():
        return 0.0
    return sum(json.loads(l)["cost"] for l in LOG.read_text().splitlines())
