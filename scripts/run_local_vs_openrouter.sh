#!/usr/bin/env bash
# HarmBench DirectRequest: the same model run locally (vLLM on m3) and via OpenRouter.
# Usage: bash ~/harmbench_runs/run_local_vs_openrouter.sh [behaviors_csv] [run_dir]
# Every run gets its own directory (default ~/harmbench_runs/runs/<timestamp>) holding test cases,
# completions, results, logs and run.json, so a new run can never overwrite or mix with an older one.
# The OpenRouter key comes from $OPENROUTER_API_KEY (~/.bashrc); it is never written anywhere.
set -euo pipefail

BEHAVIORS=${1:-./data/behavior_datasets/harmbench_behaviors_text_val.csv}
RUN=${2:-$HOME/harmbench_runs/runs/$(date +%Y%m%d-%H%M%S)}
LOCAL=qwen2_5_7b_instruct
API=qwen2_5_7b_instruct_openrouter
CLS=cais/HarmBench-Llama-2-13b-cls
MAX_ERR_FRAC=0.05                       # abort if more than 5% of completions are $ERROR$

# Only one run at a time: two jobs writing the same files / sharing GPU0 corrupts both.
exec 9>"$HOME/harmbench_runs/.run.lock"
flock -n 9 || { echo "another run_local_vs_openrouter.sh is already running (lock $HOME/harmbench_runs/.run.lock)"; exit 1; }

[ -e "$RUN" ] && { echo "run dir $RUN already exists; refusing to overwrite"; exit 1; }
RUN=$(mkdir -p "$RUN" && cd "$RUN" && pwd)
LOGS=$RUN/logs; D=$RUN/DirectRequest/default; mkdir -p "$LOGS"

cd ~/HarmBench
export PATH=$HOME/venv-harmbench/bin:$PATH HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1
[ -n "${OPENROUTER_API_KEY:-}" ] || { echo "OPENROUTER_API_KEY not set (source ~/.bashrc)"; exit 1; }

# need FILE [max_error_fraction]: file must exist, be non-empty and (for completions) mostly free of $ERROR$.
need() {
    [ -s "$1" ] || { echo "MISSING OUTPUT: $1 (see $LOGS)"; exit 1; }
    [ $# -lt 2 ] && return 0
    python - "$1" "$2" <<'PY' || exit 1
import json, sys
d = json.load(open(sys.argv[1])); gens = [x["generation"] for v in d.values() for x in v]
err = sum(g == "$ERROR$" for g in gens); frac = err / max(len(gens), 1)
print(f"{sys.argv[1]}: {len(gens)} completions, {err} errors ({frac:.1%})")
sys.exit(f"ABORT: error rate {frac:.1%} > {float(sys.argv[2]):.0%}" if frac > float(sys.argv[2]) else 0)
PY
}

# Model id + provider settings exactly as generate_completions.py will use them.
API_CFG=$(python - "$API" <<'PY'
import json, sys, yaml
c = yaml.safe_load(open("configs/model_configs/models.yaml"))[sys.argv[1]]["model"]
print(json.dumps({"id": c["model_name_or_path"].removeprefix("openrouter/"),
                  "provider": c.get("provider"), "system_prompt": c.get("system_prompt")}))
PY
)
API_ID=$(python -c 'import json,sys;print(json.loads(sys.argv[1])["id"])' "$API_CFG")

echo "[0/3] preflight: one OpenRouter request to $API_ID"
python - "$API_CFG" <<'PY' || { echo "ABORT: OpenRouter preflight failed"; exit 1; }
import json, os, sys
from openai import OpenAI
c = json.loads(sys.argv[1])
cl = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1", timeout=120)
r = cl.chat.completions.create(model=c["id"], messages=[{"role": "user", "content": "Say OK."}], max_tokens=5,
                               temperature=0, extra_body={"provider": c["provider"]} if c["provider"] else None)
print("preflight ok, provider:", getattr(r, "provider", "unknown"))
PY

# Provenance for this run (no key, no completions).
python - "$RUN/run.json" "$BEHAVIORS" "$API_CFG" <<'PY'
import json, subprocess, sys, datetime, socket, os, glob
def sh(c):
    try: return subprocess.check_output(c, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None
api = json.loads(sys.argv[3])
snap = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/*"))
json.dump({"started": datetime.datetime.now().astimezone().isoformat(), "host": socket.gethostname(),
           "behaviors": sys.argv[2], "max_new_tokens": 512, "temperature": 0,
           "harmbench_git": sh("git rev-parse HEAD"), "harmbench_dirty": bool(sh("git status --porcelain")),
           "vllm": sh("python -c 'import vllm;print(vllm.__version__)'"),
           "local_model_revision": [os.path.basename(p) for p in snap], "local_precision": "bfloat16",
           "openrouter_model": api["id"], "openrouter_provider_prefs": api["provider"],
           "api_system_prompt": api["system_prompt"],
           "openrouter_providers_used": "see logs/step2_api.log ('OpenRouter providers used')"},
          open(sys.argv[1], "w"), indent=2)
PY

echo "[1/3] test cases -> $RUN"
python -u generate_test_cases.py --method_name=DirectRequest --experiment_name=default \
  --behaviors_path=$BEHAVIORS --save_dir=$D/test_cases > $LOGS/step1.log 2>&1
need $D/test_cases/test_cases.json

echo "[2/3] completions: local (vLLM, GPU0) and OpenRouter in parallel"
CUDA_VISIBLE_DEVICES=0 python -u generate_completions.py --model_name=$LOCAL --behaviors_path=$BEHAVIORS \
  --test_cases_path=$D/test_cases/test_cases.json --save_path=$D/completions/$LOCAL.json \
  --max_new_tokens=512 --generate_with_vllm > $LOGS/step2_local.log 2>&1 & p1=$!
CUDA_VISIBLE_DEVICES="" python -u generate_completions.py --model_name=$API --behaviors_path=$BEHAVIORS \
  --test_cases_path=$D/test_cases/test_cases.json --save_path=$D/completions/$API.json \
  --max_new_tokens=512 > $LOGS/step2_api.log 2>&1 & p2=$!
rc=0; wait $p1 || rc=$?; wait $p2 || rc=$?
[ $rc -eq 0 ] || { echo "ABORT: a generation job failed (exit $rc); see $LOGS/step2_*.log"; exit 1; }
need $D/completions/$LOCAL.json $MAX_ERR_FRAC; need $D/completions/$API.json $MAX_ERR_FRAC

echo "[3/3] classifier: local on GPU0, API on GPU1"
CUDA_VISIBLE_DEVICES=0 python -u evaluate_completions.py --cls_path=$CLS --behaviors_path=$BEHAVIORS \
  --completions_path=$D/completions/$LOCAL.json --save_path=$D/results/$LOCAL.json > $LOGS/step3_local.log 2>&1 & p1=$!
CUDA_VISIBLE_DEVICES=1 python -u evaluate_completions.py --cls_path=$CLS --behaviors_path=$BEHAVIORS \
  --completions_path=$D/completions/$API.json --save_path=$D/results/$API.json > $LOGS/step3_api.log 2>&1 & p2=$!
rc=0; wait $p1 || rc=$?; wait $p2 || rc=$?
[ $rc -eq 0 ] || { echo "ABORT: a classifier job failed (exit $rc); see $LOGS/step3_*.log"; exit 1; }
need $D/results/$LOCAL.json; need $D/results/$API.json

echo "$LOCAL (local vLLM):  $(grep 'Average ASR' $LOGS/step3_local.log)"
echo "$API (OpenRouter): $(grep 'Average ASR' $LOGS/step3_api.log)"
echo "DONE -> $RUN"
