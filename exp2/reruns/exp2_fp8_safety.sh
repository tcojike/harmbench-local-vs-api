#!/usr/bin/env bash
# Second missing exp2 condition (never started on 2026-11-11): Llama-3.1-8B via DeepInfra fp8 with the safety system prompt.
# Waits for exp2_fp8_rerun.sh (same provider) so the two do not compete for DeepInfra's rate limit.
set -uo pipefail
source ~/.bashrc >/dev/null 2>&1
PROJ=$HOME/harmbench-local-vs-api; cd $PROJ
RUN=$PROJ/runs/exp2-20261111-220502; LOG=$RUN/logs; COND=llama31_8b__api-fp8__sys-safety__sampled
BEH=$HOME/HarmBench/data/behavior_datasets/harmbench_behaviors_text_test.csv
export PATH=$HOME/venv-harmbench/bin:$PATH HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/run.log"; }
WAIT_PID=${1:?pid of exp2_fp8_rerun.sh}
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 15; done
[ -n "${OPENROUTER_API_KEY:-}" ] || { log "ABORT: OPENROUTER_API_KEY not set"; exit 1; }
log "run $COND (never started on 2026-11-11)"
python exp2/gen_api.py "$RUN" $COND --concurrency 6 >> "$LOG/gen_api_deepinfra_fp8_safety.log" 2>&1 || { log "ABORT: generation failed (see gen_api_deepinfra_fp8_safety.log)"; exit 1; }
[ -f "$RUN/$COND/completions.json" ] || { log "ABORT: no completions.json"; exit 1; }
tail -1 "$LOG/gen_api_deepinfra_fp8_safety.log" | tee -a "$LOG/run.log"
CUDA_VISIBLE_DEVICES=0 python exp2/classify.py "$RUN" "$BEH" $COND > "$LOG/classify_fp8_safety.log" 2>&1 || { log "ABORT: classifier failed"; exit 1; }
grep ASR "$LOG/classify_fp8_safety.log" | tee -a "$LOG/run.log"
log "FP8 SAFETY GEN+SCORE DONE"
