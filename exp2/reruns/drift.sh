#!/usr/bin/env bash
# Time drift: regenerate exp2 API conditions unchanged on a later day, score them.
# Usage: bash exp2/reruns/drift.sh OUT_RUN_DIR COND [WAIT_PID]   (WAIT_PID: start after that process exits, e.g. a job on the same provider)
set -uo pipefail
source ~/.bashrc >/dev/null 2>&1
PROJ=$HOME/harmbench-local-vs-api; cd $PROJ
OUT=$1; COND=$2; WAIT=${3:-}
SRC=$PROJ/runs/exp2-20261111-220502
BEH=$HOME/HarmBench/data/behavior_datasets/harmbench_behaviors_text_test.csv
export PATH=$HOME/venv-harmbench/bin:$PATH HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p "$OUT/logs"; [ -f "$OUT/test_cases.json" ] || cp "$SRC/test_cases.json" "$OUT/"
if [ -n "$WAIT" ]; then while kill -0 "$WAIT" 2>/dev/null; do sleep 15; done; fi
[ -n "${OPENROUTER_API_KEY:-}" ] || { echo "ABORT: OPENROUTER_API_KEY not set"; exit 1; }
echo "[$(date +%T)] generate $COND"
python exp2/gen_api.py "$OUT" "$COND" --concurrency 6 > "$OUT/logs/gen_$COND.log" 2>&1 || { echo "ABORT gen $COND"; tail -3 "$OUT/logs/gen_$COND.log"; exit 1; }
tail -1 "$OUT/logs/gen_$COND.log"
# one scorer at a time on the shared GPU
exec 8>"$OUT/.classify.lock"; flock 8
CUDA_VISIBLE_DEVICES=${GPU:-3} python exp2/classify.py "$OUT" "$BEH" "$COND" > "$OUT/logs/classify_$COND.log" 2>&1 || { echo "ABORT classify $COND"; tail -3 "$OUT/logs/classify_$COND.log"; exit 1; }
grep ASR "$OUT/logs/classify_$COND.log"; echo "[$(date +%T)] DRIFT DONE $COND"
