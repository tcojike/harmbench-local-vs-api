#!/usr/bin/env bash
# Rerun the one missing exp2 condition (Llama-3.1-8B via DeepInfra fp8), score it, redo the analysis.
set -uo pipefail
source ~/.bashrc >/dev/null 2>&1
PROJ=$HOME/harmbench-local-vs-api; cd $PROJ
RUN=$PROJ/runs/exp2-20261111-220502; LOG=$RUN/logs; COND=llama31_8b__api-fp8__sys-template__sampled
BEH=$HOME/HarmBench/data/behavior_datasets/harmbench_behaviors_text_test.csv
export PATH=$HOME/venv-harmbench/bin:$PATH HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/run.log"; }
[ -n "${OPENROUTER_API_KEY:-}" ] || { log "ABORT: OPENROUTER_API_KEY not set"; exit 1; }
log "rerun $COND (lane died 2026-11-11 23:18 at 400/1600, nothing saved)"
python exp2/gen_api.py "$RUN" $COND --concurrency 6 >> "$LOG/gen_api_deepinfra_fp8_rerun.log" 2>&1 || { log "ABORT: generation failed (see gen_api_deepinfra_fp8_rerun.log)"; exit 1; }
[ -f "$RUN/$COND/completions.json" ] || { log "ABORT: no completions.json"; exit 1; }
tail -1 "$LOG/gen_api_deepinfra_fp8_rerun.log" | tee -a "$LOG/run.log"
CUDA_VISIBLE_DEVICES=0 python exp2/classify.py "$RUN" "$BEH" $COND > "$LOG/classify_fp8_rerun.log" 2>&1 || { log "ABORT: classifier failed"; exit 1; }
grep ASR "$LOG/classify_fp8_rerun.log" | tee -a "$LOG/run.log"
log "GEN+SCORE DONE (analysis is rerun separately)"
