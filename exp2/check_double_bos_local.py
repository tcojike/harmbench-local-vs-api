"""Does a doubled BOS token change Llama-3.1-8B's HarmBench behavior? (follow-up to exp2/check_bos.py)

CoreWeave and Parasail feed <|begin_of_text|><|begin_of_text|>... while local vLLM and DeepInfra feed one BOS.
This regenerates two local exp2 conditions with exactly one change, an extra BOS id in front of the prompt:
  llama31_8b__local-bf16__sys-template__greedy-bs256  (greedy)
  llama31_8b__local-bf16__sys-template__sampled       (5 samples, T 0.7, same per-behavior seeds)
Same model revision, dtype, batch size, seeds, decoding and test cases as the exp2 run.

Usage: CUDA_VISIBLE_DEVICES=k python exp2/check_double_bos_local.py EXP2_RUN_DIR OUT_RUN_DIR
Writes OUT_RUN_DIR/<cond>__double-bos/{records.jsonl,completions.json,gen_meta.json}; score with exp2/classify.py.
"""
import hashlib, json, shutil, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditions import CONDITIONS, MODELS, SAMPLED, GREEDY, messages, render  # noqa: E402

src, out_run = Path(sys.argv[1]), Path(sys.argv[2]); out_run.mkdir(parents=True, exist_ok=True)
shutil.copy(src / "test_cases.json", out_run / "test_cases.json")
test_cases = json.load(open(out_run / "test_cases.json"))
BASE = ["llama31_8b__local-bf16__sys-template__greedy-bs256", "llama31_8b__local-bf16__sys-template__sampled"]
conds = [next(c for c in CONDITIONS if c["name"] == n) for n in BASE]
assert {c["batch"] for c in conds} == {256}
m = MODELS["llama31_8b"]

from vllm import LLM, SamplingParams  # noqa: E402
llm = LLM(model=m["hf"], revision=m["revision"], dtype="bfloat16", seed=0, max_num_seqs=256,
          gpu_memory_utilization=0.85, max_model_len=4096)
tok = llm.get_tokenizer()
for c in conds:
    out = out_run / (c["name"] + "__double-bos"); out.mkdir(exist_ok=True)
    dec = SAMPLED if c["decoding"] == "sampled" else GREEDY
    items = [(bid, tc) for bid, tcs in test_cases.items() for tc in tcs]
    prompts = [render(c["model"], tok, messages(c["system"], tc)) for _, tc in items]
    params = [SamplingParams(n=dec["n"], temperature=dec["temperature"], top_p=dec["top_p"], max_tokens=dec["max_tokens"],
                             seed=int(hashlib.sha256(f"{bid}".encode()).hexdigest()[:8], 16)) for bid, _ in items]
    ids = [tok(p, add_special_tokens=False).input_ids for p in prompts]
    assert all(x[0] == tok.bos_token_id and x[1] != tok.bos_token_id for x in ids)
    ids = [[tok.bos_token_id] + x for x in ids]          # the only change vs exp2
    t0 = time.time()
    outs = llm.generate([{"prompt_token_ids": x} for x in ids], params)
    secs = time.time() - t0
    recs, comp = [], {}
    for (bid, tc), o in zip(items, outs):
        for k, s in enumerate(o.outputs):
            recs.append(dict(behavior_id=bid, sample=k, condition=out.name, prompt_tokens=len(o.prompt_token_ids),
                             completion_tokens=len(s.token_ids), finish_reason=s.finish_reason, provider="local-vllm",
                             generation=s.text))
            comp.setdefault(bid, []).append(dict(test_case=tc, generation=s.text))
    with open(out / "records.jsonl", "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    json.dump(comp, open(out / "completions.json", "w"))
    json.dump(dict(base_condition=c, change="extra leading BOS token id", generation_seconds=round(secs, 1),
                   n_completions=len(recs), errors=0), open(out / "gen_meta.json", "w"), indent=2)
    print(f"{out.name}: {len(recs)} completions in {secs:.0f}s, prompt tokens e.g. {len(ids[0])}", flush=True)
