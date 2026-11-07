"""Phase 2 proof of principle: one OpenRouter call, logging model, provider, prompt, response, tokens and cost.

Usage:  python scripts/test_openrouter.py [model_id]       (default qwen/qwen-2.5-7b-instruct)
The API key is read from the OPENROUTER_API_KEY environment variable (set in ~/.bashrc), never hard-coded.
Writes logs/test_openrouter_<timestamp>.json.
"""
import datetime, json, os, sys, time
from pathlib import Path

import requests

MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen/qwen-2.5-7b-instruct"
PROMPT = "In one sentence, what is HarmBench?"
ROOT = Path(__file__).resolve().parent.parent

key = os.environ.get("OPENROUTER_API_KEY")
if not key:
    sys.exit("OPENROUTER_API_KEY is not set (try: source ~/.bashrc)")

t0 = time.time()
r = requests.post(
    "https://openrouter.ai/api/v1/chat/completions",
    headers={"Authorization": f"Bearer {key}", "X-Title": "harmbench-local-vs-api"},
    json={"model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
          "temperature": 0, "max_tokens": 128, "usage": {"include": True}},
    timeout=120,
)
elapsed = time.time() - t0
r.raise_for_status()
d = r.json()
usage = d.get("usage", {})
rec = {
    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    "requested_model": MODEL,
    "served_model": d.get("model"),
    "provider": d.get("provider"),
    "prompt": PROMPT,
    "response": d["choices"][0]["message"]["content"],
    "prompt_tokens": usage.get("prompt_tokens"),
    "completion_tokens": usage.get("completion_tokens"),
    "cost_usd": usage.get("cost"),
    "latency_s": round(elapsed, 2),
}
out = ROOT / "logs" / f"test_openrouter_{datetime.datetime.now():%Y%m%d-%H%M%S}.json"
out.write_text(json.dumps(rec, indent=2))
for k, v in rec.items():
    print(f"{k:18s} {v}")
print(f"\nsaved {out}")
