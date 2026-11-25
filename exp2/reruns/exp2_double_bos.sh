#!/usr/bin/env bash
set -uo pipefail
PROJ=$HOME/harmbench-local-vs-api; cd $PROJ
SRC=$PROJ/runs/exp2-20261111-220502; OUT=$PROJ/runs/exp2-bos-check-$(date +%Y%m%d-%H%M%S)
BEH=$HOME/HarmBench/data/behavior_datasets/harmbench_behaviors_text_test.csv
export PATH=$HOME/venv-harmbench/bin:$PATH HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
echo "[$(date +%T)] out: $OUT"
CUDA_VISIBLE_DEVICES=1 python exp2/check_double_bos_local.py "$SRC" "$OUT" > /tmp/double_bos_gen.log 2>&1 || { echo "ABORT gen (see /tmp/double_bos_gen.log)"; tail -5 /tmp/double_bos_gen.log; exit 1; }
grep "completions in" /tmp/double_bos_gen.log; mkdir -p "$OUT/logs"; mv /tmp/double_bos_gen.log "$OUT/logs/gen.log"
CUDA_VISIBLE_DEVICES=1 python exp2/classify.py "$OUT" "$BEH" > "$OUT/logs/classify.log" 2>&1 || { echo "ABORT classify"; tail -5 "$OUT/logs/classify.log"; exit 1; }
grep ASR "$OUT/logs/classify.log"
python exp2/compare_double_bos.py "$SRC" "$OUT" > "$OUT/logs/compare.log" 2>&1 || { echo "ABORT compare"; tail -5 "$OUT/logs/compare.log"; exit 1; }
echo "[$(date +%T)] DOUBLE-BOS DONE -> $OUT/comparison.md"
