#!/usr/bin/env bash
set -uo pipefail
PROJ=$HOME/harmbench-local-vs-api; cd $PROJ
RUN=$PROJ/runs/exp2-20261111-220502; LOG=$RUN/logs
BEH=$HOME/HarmBench/data/behavior_datasets/harmbench_behaviors_text_test.csv
export PATH=$HOME/venv-harmbench/bin:$PATH HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/run.log"; }
log "scoring 17 finished conditions (DeepInfra fp8 lane died at 400/1600; condition excluded)"
mapfile -t ALL < <(for d in $RUN/*/; do [ -f $d/completions.json ] && basename $d; done)
pids=()
for g in 0 1 2 3; do
  shard=(); for i in "${!ALL[@]}"; do [ $((i % 4)) -eq $g ] && shard+=("${ALL[$i]}"); done
  CUDA_VISIBLE_DEVICES=$g python exp2/classify.py "$RUN" "$BEH" "${shard[@]}" > "$LOG/classify_gpu$g.log" 2>&1 & pids+=($!)
done
for p in "${pids[@]}"; do wait $p || { log "ABORT: classifier failed"; exit 1; }; done
grep -h "ASR" $LOG/classify_gpu*.log | tee -a $LOG/run.log
OUT=$PROJ/results/exp2/$(basename $RUN)
python exp2/analyze.py "$RUN" "$BEH" "$OUT" > "$LOG/analyze.log" 2>&1 || { log "ABORT: analysis failed"; exit 1; }
cp $RUN/run.json $OUT/run.json
log "DONE -> $OUT/report.md"
