"""Generate completions through OpenRouter for one condition, with the provider pinned.

Usage: python exp2/gen_api.py RUN_DIR COND_NAME [--concurrency 16]

* provider pinned: {order: [<endpoint>], allow_fallbacks: false}; the provider that actually served each
  response, its id, token usage, finish reason and latency are stored per completion (records.jsonl).
* permanent errors (4xx except 408/409/429) abort immediately; transient errors are retried with backoff.
* the run aborts if more than 5% of completions end as $ERROR$.
* checkpointing: every finished record is appended to <cond>/partial.jsonl; a restart reuses the successful
  ones and only requests the rest. Retry errors are logged as they happen.
"""
import argparse, asyncio, hashlib, json, os, random, sys, time
from pathlib import Path

import openai
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditions import CONDITIONS, MODELS, SAMPLED, GREEDY, messages  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("cond"); ap.add_argument("--concurrency", type=int, default=16)
a = ap.parse_args()
run = Path(a.run)
c = next(x for x in CONDITIONS if x["name"] == a.cond)
assert c["path"] == "api" and c["provider"]
model_id = MODELS[c["model"]]["api"]
dec = SAMPLED if c["decoding"] == "sampled" else GREEDY
out = run / c["name"]; out.mkdir(parents=True, exist_ok=True)
if (out / "completions.json").exists():
    sys.exit(print("skip (exists)", c["name"]))
test_cases = json.load(open(run / "test_cases.json"))
client = AsyncOpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1", timeout=180,
                     max_retries=0, default_headers={"X-Title": "HarmBench exp2"})
ERR = "$ERROR$"


class Permanent(Exception):
    pass


async def one(sem, bid, tc, k):
    seed = int(hashlib.sha256(f"{bid}".encode()).hexdigest()[:8], 16) + k
    rec = dict(behavior_id=bid, sample=k, condition=c["name"], seed=seed)
    async with sem:
        for attempt in range(10):
            t0 = time.time()
            try:
                r = await client.chat.completions.create(
                    model=model_id, messages=messages(c["system"], tc), max_tokens=dec["max_tokens"],
                    temperature=dec["temperature"], top_p=dec["top_p"], seed=seed,
                    extra_body={"provider": {"order": [c["provider"]], "allow_fallbacks": False}})
                text = r.choices[0].message.content
                rec.update(provider=getattr(r, "provider", None), response_id=r.id, latency=round(time.time() - t0, 2),
                           prompt_tokens=r.usage.prompt_tokens if r.usage else None,
                           completion_tokens=r.usage.completion_tokens if r.usage else None,
                           finish_reason=r.choices[0].finish_reason, generation=text if text is not None else ERR,
                           attempts=attempt + 1)
                return rec
            except openai.APIStatusError as e:
                if 400 <= e.status_code < 500 and e.status_code not in (408, 409, 429):
                    raise Permanent(f"{e.status_code}: {e}") from e
                rec["last_error"] = f"{e.status_code}: {str(e)[:200]}"
                print(f"retry {bid}/{k} attempt {attempt + 1}: {rec['last_error'][:160]}", flush=True)
            except (openai.APIConnectionError, openai.APITimeoutError) as e:
                rec["last_error"] = f"{type(e).__name__}: {str(e)[:200]}"
                print(f"retry {bid}/{k} attempt {attempt + 1}: {rec['last_error'][:160]}", flush=True)
            except Exception as e:   # malformed responses (e.g. choices=None from the provider)
                rec["last_error"] = f"{type(e).__name__}: {str(e)[:200]}"
                print(f"retry {bid}/{k} attempt {attempt + 1}: {rec['last_error'][:160]}", flush=True)
            await asyncio.sleep(min(90, 2 ** attempt * 2) * (0.5 + random.random()))   # jittered backoff (429s)
    rec.update(generation=ERR, attempts=10)
    return rec


PARTIAL = out / "partial.jsonl"


def load_partial():
    """Successful records from an interrupted earlier attempt (failed ones are retried)."""
    done = {}
    if PARTIAL.exists():
        for line in open(PARTIAL):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:   # last line cut off by a kill
                continue
            if r.get("generation") != ERR:
                done[(r["behavior_id"], r["sample"])] = r
    return done


async def main():
    sem = asyncio.Semaphore(a.concurrency)
    done = load_partial()
    todo = [(bid, tc, k) for bid, tcs in test_cases.items() for tc in tcs for k in range(dec["n"]) if (bid, k) not in done]
    total = len(todo) + len(done)
    if done:
        print(f"{c['name']}: resuming, {len(done)} records from {PARTIAL.name}, {len(todo)} to request", flush=True)
    jobs = [one(sem, bid, tc, k) for bid, tc, k in todo]
    t0, recs = time.time(), list(done.values())
    with open(PARTIAL, "a") as pf:
        for i, fut in enumerate(asyncio.as_completed(jobs), 1):
            r = await fut
            recs.append(r); pf.write(json.dumps(r) + "\n"); pf.flush()
            if i % 200 == 0 or i == len(jobs):
                errs = sum(x["generation"] == ERR for x in recs)
                print(f"{c['name']}: {len(recs)}/{total} done, {errs} errors, {time.time() - t0:.0f}s", flush=True)
    return recs, time.time() - t0

try:
    recs, secs = asyncio.run(main())
except Permanent as e:
    sys.exit(f"ABORT {c['name']}: permanent API error {e}")

recs.sort(key=lambda r: (r["behavior_id"], r["sample"]))
errs = sum(r["generation"] == ERR for r in recs)
providers = {}
for r in recs:
    providers[r.get("provider")] = providers.get(r.get("provider"), 0) + 1
with open(out / "records.jsonl", "w") as f:
    for r in recs: f.write(json.dumps(r) + "\n")
comp = {}
for r in recs:
    if r["generation"] == ERR:   # failed requests are left out (not scored as refusals); rates use the samples that exist
        continue
    comp.setdefault(r["behavior_id"], []).append(dict(test_case=test_cases[r["behavior_id"]][0], generation=r["generation"]))
json.dump(dict(condition=c, model_id=model_id, generation_seconds=round(secs, 1), n_completions=len(recs), errors=errs,
               providers_used=providers,
               prompt_tokens=sum(r.get("prompt_tokens") or 0 for r in recs),
               completion_tokens=sum(r.get("completion_tokens") or 0 for r in recs)),
          open(out / "gen_meta.json", "w"), indent=2)
print(f"{c['name']}: {len(recs)} completions, {errs} errors, providers {providers}, {secs:.0f}s")
if errs / len(recs) > 0.05:
    sys.exit(f"ABORT {c['name']}: error rate {errs / len(recs):.1%} > 5% (completions.json not written)")
json.dump(comp, open(out / "completions.json", "w"))
