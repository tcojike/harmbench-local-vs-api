#!/usr/bin/env bash
# DeepInfra copyright probe: generate 4 greedy arms on the 80 copyright behaviors, score them (hash check).
set -uo pipefail
source ~/.bashrc >/dev/null 2>&1
PROJ=$HOME/harmbench-local-vs-api; cd $PROJ
OUT=${1:-$PROJ/runs/probe-copyright-$(date +%Y%m%d-%H%M%S)}; mkdir -p "$OUT/logs"
BEH=$HOME/HarmBench/data/behavior_datasets/harmbench_behaviors_text_test.csv
export PATH=$HOME/venv-harmbench/bin:$PATH HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
[ -n "${OPENROUTER_API_KEY:-}" ] || { echo "ABORT: OPENROUTER_API_KEY not set"; exit 1; }
echo "[$(date +%T)] out: $OUT"
python exp2/probe_copyright_sampling.py runs/exp2-20261111-220502 "$OUT" > "$OUT/logs/gen.log" 2>&1 || { echo "ABORT gen"; tail -3 "$OUT/logs/gen.log"; exit 1; }
grep "completions," "$OUT/logs/gen.log"
CUDA_VISIBLE_DEVICES=${GPU:-1} python exp2/classify.py "$OUT" "$BEH" > "$OUT/logs/classify.log" 2>&1 || { echo "ABORT classify"; tail -3 "$OUT/logs/classify.log"; exit 1; }
grep ASR "$OUT/logs/classify.log"
echo "[$(date +%T)] PROBE DONE"
