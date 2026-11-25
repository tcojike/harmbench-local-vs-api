"""Probe: why does Llama-3.1-8B on DeepInfra fp8 reproduce far less copyrighted text? (exp2 open question)

Usage: python exp2/probe_copyright_sampling.py EXP2_RUN_DIR OUT_RUN_DIR [--concurrency 6]
Greedy (T = 0), template system prompt, the 80 copyright behaviors, max_tokens 512, provider pinned, no fallbacks.
Arms (OpenRouter passes these sampling parameters through to the provider):
  deepinfra-default   no extra parameters (reproduces the exp2 setting)
  deepinfra-neutral   repetition_penalty 1.0, frequency_penalty 0, presence_penalty 0, min_p 0 (all explicitly off)
  deepinfra-rep1.1    repetition_penalty 1.1 (positive control: does a mild penalty alone cut verbatim recall?)
  coreweave-default   no extra parameters (same-day reference)
Writes OUT_RUN_DIR/<arm>/{records.jsonl,completions.json,gen_meta.json}; checkpointed in partial.jsonl. Score with classify.py.
"""
import argparse, asyncio, csv, json, os, random, shutil, sys, time
from pathlib import Path

import openai
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditions import MODELS, messages  # noqa: E402

ARMS = {
    "deepinfra-default": ("deepinfra/fp8", {}),
    "deepinfra-neutral": ("deepinfra/fp8", dict(repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0, min_p=0.0)),
    "deepinfra-rep1.1": ("deepinfra/fp8", dict(repetition_penalty=1.1)),
    "coreweave-default": ("coreweave/bf16", {}),
}
ERR = "$ERROR$"
ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--concurrency", type=int, default=6)
a = ap.parse_args()
src, out_run = Path(a.src), Path(a.out); out_run.mkdir(parents=True, exist_ok=True)
BEH = Path.home() / "HarmBench/data/behavior_datasets/harmbench_behaviors_text_test.csv"
copyright_ids = {b["BehaviorID"] for b in csv.DictReader(open(BEH, encoding="utf-8")) if b["FunctionalCategory"] == "copyright"}
tcs = {b: v for b, v in json.load(open(src / "test_cases.json")).items() if b in copyright_ids}
assert len(tcs) == 80, len(tcs)
json.dump(tcs, open(out_run / "test_cases.json", "w"), indent=2)
model_id = MODELS["llama31_8b"]["api"]
client = AsyncOpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1", timeout=180,
                     max_retries=0, default_headers={"X-Title": "HarmBench exp2 copyright probe"})


async def one(sem, arm, prov, extra, bid, tc):
    rec = dict(behavior_id=bid, sample=0, condition=arm, provider_pinned=prov, extra=extra)
    async with sem:
        for attempt in range(10):
            t0 = time.time()
            try:
                r = await client.chat.completions.create(
                    model=model_id, messages=messages("template", tc), max_tokens=512, temperature=0.0, top_p=1.0, seed=0,
                    extra_body={"provider": {"order": [prov], "allow_fallbacks": False}, **extra})
                text = r.choices[0].message.content
                rec.update(provider=getattr(r, "provider", None), response_id=r.id, latency=round(time.time() - t0, 2),
                           prompt_tokens=r.usage.prompt_tokens, completion_tokens=r.usage.completion_tokens,
                           finish_reason=r.choices[0].finish_reason, generation=text if text is not None else ERR,
                           attempts=attempt + 1)
                return rec
            except openai.APIStatusError as e:
                if 400 <= e.status_code < 500 and e.status_code not in (408, 409, 429):
                    raise RuntimeError(f"{arm}: permanent {e.status_code}: {e}") from e
                rec["last_error"] = f"{e.status_code}: {str(e)[:200]}"
            except Exception as e:
                rec["last_error"] = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"retry {arm} {bid} attempt {attempt + 1}: {rec['last_error'][:140]}", flush=True)
            await asyncio.sleep(min(90, 2 ** attempt * 2) * (0.5 + random.random()))
    rec.update(generation=ERR, attempts=10)
    return rec


async def run_arm(arm, prov, extra):
    out = out_run / arm; out.mkdir(exist_ok=True)
    if (out / "completions.json").exists():
        return print("skip (exists)", arm)
    part = out / "partial.jsonl"; done = {}
    if part.exists():
        for line in open(part):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r["generation"] != ERR:
                done[r["behavior_id"]] = r
    sem = asyncio.Semaphore(a.concurrency); t0 = time.time()
    recs = list(done.values())
    with open(part, "a") as pf:
        for fut in asyncio.as_completed([one(sem, arm, prov, extra, b, v[0]) for b, v in tcs.items() if b not in done]):
            r = await fut; recs.append(r); pf.write(json.dumps(r) + "\n"); pf.flush()
    recs.sort(key=lambda r: r["behavior_id"])
    errs = sum(r["generation"] == ERR for r in recs)
    provs = sorted({str(r.get("provider")) for r in recs})
    with open(out / "records.jsonl", "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    json.dump({r["behavior_id"]: [dict(test_case=tcs[r["behavior_id"]][0], generation=r["generation"])] for r in recs if r["generation"] != ERR},
              open(out / "completions.json", "w"))
    json.dump(dict(arm=arm, provider=prov, extra=extra, model_id=model_id, n=len(recs), errors=errs, providers_used=provs,
                   seconds=round(time.time() - t0, 1), date=time.strftime("%Y-%m-%d %H:%M %Z")), open(out / "gen_meta.json", "w"), indent=2)
    print(f"{arm}: {len(recs)} completions, {errs} errors, providers {provs}, {time.time() - t0:.0f}s", flush=True)


async def main():
    for arm, (prov, extra) in ARMS.items():   # sequential: the three DeepInfra arms share one rate limit
        await run_arm(arm, prov, extra)

asyncio.run(main())
