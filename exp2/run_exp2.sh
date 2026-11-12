#!/usr/bin/env bash
# Experiment 2: serving path x precision x system prompt x model on the 320 HarmBench text test behaviors.
# Usage: bash exp2/run_exp2.sh [RUN_DIR]
#   New run: no argument -> runs/exp2-<timestamp>/.  Resume: pass an existing run dir (finished conditions are skipped).
# Raw completions stay in runs/ (gitignored: harmful content). Summary tables go to results/exp2/<run name>/.
set -euo pipefail
PROJ=$(cd "$(dirname "$0")/.." && pwd)
HB=${HARMBENCH_DIR:-$HOME/HarmBench}
BEH=$HB/data/behavior_datasets/harmbench_behaviors_text_test.csv
export PATH=$HOME/venv-harmbench/bin:$PATH HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
[ -n "${OPENROUTER_API_KEY:-}" ] || { echo "OPENROUTER_API_KEY not set (source ~/.bashrc)"; exit 1; }

# one run at a time (two runs would share GPUs and the same files)
mkdir -p "$PROJ/runs"; exec 9>"$PROJ/runs/.lock"
flock -n 9 || { echo "another exp2 run holds $PROJ/runs/.lock"; exit 1; }

RUN=${1:-$PROJ/runs/exp2-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$RUN/logs"; RUN=$(cd "$RUN" && pwd); LOG=$RUN/logs
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/run.log"; }
# EXP2_LIMIT=k: smoke test on the first k behaviors of each category
python - "$PROJ/results/test_cases/harmbench_behaviors_text_test/test_cases.json" "$RUN/test_cases.json" "$BEH" "${EXP2_LIMIT:-0}" <<'PY'
import csv, json, sys
tc, k = json.load(open(sys.argv[1])), int(sys.argv[4])
if k:
    seen, keep = {}, set()
    for b in csv.DictReader(open(sys.argv[3], encoding="utf-8")):
        if b["BehaviorID"] in tc and seen.get(b["FunctionalCategory"], 0) < k:
            seen[b["FunctionalCategory"]] = seen.get(b["FunctionalCategory"], 0) + 1; keep.add(b["BehaviorID"])
    tc = {b: v for b, v in tc.items() if b in keep}
json.dump(tc, open(sys.argv[2], "w"), indent=2)
print(f"{len(tc)} behaviors")
PY
cd "$PROJ"
names() { python -c "import sys; sys.path.insert(0,'exp2'); from conditions import CONDITIONS as C; print(' '.join(c['name'] for c in C if $1))"; }

# ---------- preflight: every pinned provider answers (no fallbacks), all local weights are cached ----------
log "preflight"
python - <<'PY' 2>&1 | tee -a "$LOG/run.log"
import os, sys, time
sys.path.insert(0, "exp2")
from conditions import CONDITIONS, MODELS
from openai import OpenAI
import openai
from huggingface_hub import snapshot_download
for m in {c["model"] for c in CONDITIONS}:
    snapshot_download(MODELS[m]["hf"], revision=MODELS[m]["revision"], local_files_only=True)   # raises if missing
cl = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1", max_retries=0, timeout=60)
for model, prov in sorted({(MODELS[c["model"]]["api"], c["provider"]) for c in CONDITIONS if c["path"] == "api"}):
    for i in range(5):
        try:
            r = cl.chat.completions.create(model=model, messages=[{"role": "user", "content": "Say OK."}], max_tokens=3,
                    extra_body={"provider": {"order": [prov], "allow_fallbacks": False}}); break
        except openai.RateLimitError:
            time.sleep(5 * 2 ** i)
    else:
        sys.exit(f"preflight: {model} via {prov} keeps returning 429")
    print(f"preflight ok: {model} via {prov} -> {getattr(r, 'provider', '?')}")
PY
[ "${PIPESTATUS[0]}" -eq 0 ] || { log "ABORT: preflight failed"; exit 1; }

# ---------- provenance ----------
python - "$RUN/run.json" <<'PY'
import json, subprocess, sys, datetime, socket, hashlib, urllib.request
sys.path.insert(0, "exp2")
from conditions import CONDITIONS, MODELS
from huggingface_hub import snapshot_download
sh = lambda c: subprocess.run(c, shell=True, text=True, capture_output=True).stdout.strip()
eps = {}
for m in {MODELS[c["model"]]["api"] for c in CONDITIONS if c["path"] == "api"}:
    d = json.load(urllib.request.urlopen(f"https://openrouter.ai/api/v1/models/{m}/endpoints"))["data"]["endpoints"]
    eps[m] = [{k: e.get(k) for k in ("provider_name", "tag", "quantization", "context_length", "pricing")} for e in d]
json.dump({"started": datetime.datetime.now().astimezone().isoformat(), "host": socket.gethostname(),
           "gpus": sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader").splitlines(),
           "project_git": sh("git rev-parse HEAD"), "project_dirty": bool(sh("git status --porcelain exp2")),
           "conditions_sha256": hashlib.sha256(open("exp2/conditions.py", "rb").read()).hexdigest(),
           "harmbench_git": sh(f"git -C $HOME/HarmBench rev-parse HEAD"),
           "versions": {p: sh(f"python -c 'import {p}; print({p}.__version__)'") for p in ("vllm", "torch", "transformers", "openai")},
           "local_snapshots": {m: snapshot_download(v["hf"], revision=v["revision"], local_files_only=True) for m, v in MODELS.items()},
           "openrouter_endpoints_at_start": eps, "conditions": CONDITIONS}, open(sys.argv[1], "w"), indent=2)
PY
log "run dir: $RUN"

# ---------- generation: 4 GPU lanes + 4 API lanes (one per pinned provider), all in parallel ----------
lane() { local name=$1; shift; ( "$@" ) > "$LOG/gen_$name.log" 2>&1; }
pids=()
lane gpu0 env CUDA_VISIBLE_DEVICES=0 python exp2/gen_local.py "$RUN" $(names "c['model']=='qwen25_7b'") & pids+=($!)
lane gpu1 env CUDA_VISIBLE_DEVICES=1 python exp2/gen_local.py "$RUN" $(names "c['model']=='llama31_8b' and c['path']=='local' and c['precision']=='bf16'") & pids+=($!)
lane gpu2 env CUDA_VISIBLE_DEVICES=2 python exp2/gen_local.py "$RUN" $(names "c['model']=='llama31_8b' and c['path']=='local' and c['precision']=='fp8'") & pids+=($!)
lane gpu3 bash -c "CUDA_VISIBLE_DEVICES=3 python exp2/gen_local.py '$RUN' $(names "c['model']=='lunaris_8b' and c['path']=='local' and c['precision']=='bf16'") ; \
                   CUDA_VISIBLE_DEVICES=3 python exp2/gen_local.py '$RUN' $(names "c['model']=='lunaris_8b' and c['path']=='local' and c['precision']=='fp8'")" & pids+=($!)
for prov in coreweave/bf16 deepinfra/fp8 parasail/bf16 deepinfra/turbo; do
    lane "api_${prov//\//_}" bash -c "for c in $(names "c['provider']=='$prov'"); do python exp2/gen_api.py '$RUN' \$c --concurrency 6; done" & pids+=($!)
done
log "generation started (${#pids[@]} lanes); logs: $LOG/gen_*.log"
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done

missing=$(python -c "
import sys; sys.path.insert(0,'exp2'); from conditions import CONDITIONS as C; from pathlib import Path
print(' '.join(c['name'] for c in C if not (Path('$RUN')/c['name']/'completions.json').exists()))")
[ -z "$missing" ] || { log "ABORT: generation failed for: $missing (see $LOG/gen_*.log); rerun 'bash exp2/run_exp2.sh $RUN' to resume"; exit 1; }
log "generation done"

# ---------- scoring: shard conditions over 4 GPUs, classifier loaded once per GPU ----------
mapfile -t ALL < <(python -c "import sys; sys.path.insert(0,'exp2'); from conditions import CONDITIONS as C; [print(c['name']) for c in C]")
pids=()
for g in 0 1 2 3; do
    shard=(); for i in "${!ALL[@]}"; do [ $((i % 4)) -eq $g ] && shard+=("${ALL[$i]}"); done
    CUDA_VISIBLE_DEVICES=$g python exp2/classify.py "$RUN" "$BEH" "${shard[@]}" > "$LOG/classify_gpu$g.log" 2>&1 & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || { log "ABORT: classifier failed (see $LOG/classify_gpu*.log)"; exit 1; }; done
grep -h "ASR" "$LOG"/classify_gpu*.log | tee -a "$LOG/run.log"

# ---------- analysis (labels/rates only; no generations leave runs/) ----------
OUT=$PROJ/results/exp2/$(basename "$RUN")
python exp2/analyze.py "$RUN" "$BEH" "$OUT" > "$LOG/analyze.log" 2>&1 || { log "ABORT: analysis failed (see $LOG/analyze.log)"; exit 1; }
cp "$RUN/run.json" "$OUT/run.json"
log "DONE -> $OUT/report.md"
