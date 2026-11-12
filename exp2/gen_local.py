"""Generate completions locally with vLLM for one or more conditions of the same model+precision.

Usage: python exp2/gen_local.py RUN_DIR COND_NAME [COND_NAME ...]   (all conditions must share model & precision)

Writes, per condition, RUN_DIR/<cond>/records.jsonl (one line per completion, with prompt token
count, finish reason, sample index, seed) and RUN_DIR/<cond>/completions.json (HarmBench format).
"""
import json, sys, time, hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditions import CONDITIONS, MODELS, SAMPLED, GREEDY, messages, render  # noqa: E402

run, names = Path(sys.argv[1]), sys.argv[2:]
conds = [c for c in CONDITIONS if c["name"] in names]
assert len(conds) == len(names), f"unknown condition in {names}"
assert all(c["path"] == "local" for c in conds)
assert len({(c["model"], c["precision"]) for c in conds}) == 1, "one model+precision per process"
m, prec = MODELS[conds[0]["model"]], conds[0]["precision"]
test_cases = json.load(open(run / "test_cases.json"))

from vllm import LLM, SamplingParams  # noqa: E402

def load(batch):
    kw = dict(model=m["hf"], revision=m["revision"], dtype="bfloat16", seed=0, max_num_seqs=batch,
              gpu_memory_utilization=0.85, max_model_len=4096)
    if prec == "fp8":
        kw["quantization"] = "fp8"   # dynamic per-tensor FP8 weights (L40S supports FP8 natively)
    return LLM(**kw)

llm, cur_batch = None, None
for c in conds:
    out = run / c["name"]; out.mkdir(parents=True, exist_ok=True)
    if (out / "completions.json").exists():
        print("skip (exists)", c["name"]); continue
    if llm is None or c["batch"] != cur_batch:          # batch size is part of the condition (noise-floor runs)
        del llm
        llm, cur_batch = load(c["batch"]), c["batch"]
    tok = llm.get_tokenizer()
    dec = SAMPLED if c["decoding"] == "sampled" else GREEDY
    items = [(bid, tc) for bid, tcs in test_cases.items() for tc in tcs]
    prompts = [render(c["model"], tok, messages(c["system"], tc)) for _, tc in items]
    # One seed per (behavior, sample) so reruns are reproducible; temperature 0 ignores it.
    params = [SamplingParams(n=dec["n"], temperature=dec["temperature"], top_p=dec["top_p"], max_tokens=dec["max_tokens"],
                             seed=int(hashlib.sha256(f"{bid}".encode()).hexdigest()[:8], 16)) for bid, _ in items]
    t0 = time.time()
    # Pass token ids, not text: vLLM would otherwise prepend a second <|begin_of_text|> to Llama prompts.
    ids = [tok(p, add_special_tokens=False).input_ids for p in prompts]
    outs = llm.generate([{"prompt_token_ids": x} for x in ids], params)
    secs = time.time() - t0
    recs, comp = [], {}
    for (bid, tc), p, o in zip(items, prompts, outs):
        for k, s in enumerate(o.outputs):
            recs.append(dict(behavior_id=bid, sample=k, condition=c["name"], prompt_tokens=len(o.prompt_token_ids),
                             prompt_sha=hashlib.sha256(p.encode()).hexdigest()[:16], completion_tokens=len(s.token_ids),
                             finish_reason=s.finish_reason, provider="local-vllm", generation=s.text))
            comp.setdefault(bid, []).append(dict(test_case=tc, generation=s.text))
    with open(out / "records.jsonl", "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    json.dump(comp, open(out / "completions.json", "w"))
    json.dump(dict(condition=c, generation_seconds=round(secs, 1), n_completions=len(recs), errors=0,
                   example_prompt=prompts[0]), open(out / "gen_meta.json", "w"), indent=2)
    print(f"{c['name']}: {len(recs)} completions in {secs:.0f}s")
