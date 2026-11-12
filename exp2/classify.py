"""Score every condition in a run with the official HarmBench classifier, loading it once.

Usage: CUDA_VISIBLE_DEVICES=k python exp2/classify.py RUN_DIR BEHAVIORS_CSV [COND ...]   (default: all unscored)

Same logic as HarmBench's evaluate_completions.py (truncate to 512 classifier tokens, hash check for
copyright behaviors, HarmBench-Llama-2-13b-cls otherwise). Writes RUN_DIR/<cond>/results.json and
adds the label to labels.jsonl (behavior_id, sample, label) so the analysis never needs the text.
"""
import csv, json, os, sys
from pathlib import Path

HB = Path(os.environ.get("HARMBENCH_DIR", Path.home() / "HarmBench"))
sys.path.insert(0, str(HB))
os.chdir(HB)   # eval_utils loads copyright hashes from relative paths
from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from eval_utils import compute_results_hashing, compute_results_classifier  # noqa: E402

CLS = "cais/HarmBench-Llama-2-13b-cls"
run, beh_csv, only = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
behaviors = {b["BehaviorID"]: b for b in csv.DictReader(open(beh_csv, encoding="utf-8"))}
todo = [d for d in sorted(run.iterdir()) if (d / "completions.json").exists() and not (d / "results.json").exists()
        and (not only or d.name in only)]
print("to score:", [d.name for d in todo])
if not todo:
    sys.exit(0)

tok = AutoTokenizer.from_pretrained(CLS); tok.truncation_side = "right"
cls = LLM(model=CLS, tensor_parallel_size=1, gpu_memory_utilization=0.85)
cls.llm_engine.tokenizer.tokenizer.truncation_side = "left"
params = SamplingParams(temperature=0.0, max_tokens=1)

for d in todo:
    comp = json.load(open(d / "completions.json"))
    for lst in comp.values():
        for x in lst:
            x["generation"] = tok.decode(tok.encode(x["generation"], max_length=512, truncation=True), skip_special_tokens=True)
    res = {}
    for bid, data in comp.items():
        b = behaviors[bid]
        res[bid] = (compute_results_hashing(b, data) if "hash_check" in b["Tags"].split(", ")
                    else compute_results_classifier(b, data, cls, params))
    json.dump(res, open(d / "results.json", "w"))
    with open(d / "labels.jsonl", "w") as f:
        for bid, lst in res.items():
            for k, x in enumerate(lst):
                f.write(json.dumps(dict(behavior_id=bid, sample=k, label=int(x["label"]))) + "\n")
    n = sum(len(v) for v in res.values()); h = sum(x["label"] == 1 for v in res.values() for x in v)
    bad = sum(x["label"] == -1 for v in res.values() for x in v)
    print(f"{d.name}: {n} scored, ASR {h / n:.3f}, unparseable labels {bad}", flush=True)
