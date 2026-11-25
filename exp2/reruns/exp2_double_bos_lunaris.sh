#!/usr/bin/env bash
set -uo pipefail
cd $HOME/harmbench-local-vs-api
export PATH=$HOME/venv-harmbench/bin:$PATH HF_HOME=$HOME/.cache/huggingface HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
SRC=$PWD/runs/exp2-20261111-220502; OUT=$PWD/runs/exp2-bos-check-lunaris-$(date +%Y%m%d-%H%M%S); mkdir -p $OUT/logs
BEH=$HOME/HarmBench/data/behavior_datasets/harmbench_behaviors_text_test.csv
echo "[$(date +%T)] out: $OUT"
CUDA_VISIBLE_DEVICES=2 python exp2/check_double_bos_local.py $SRC $OUT lunaris_8b > $OUT/logs/gen.log 2>&1 || { echo ABORT gen; tail -3 $OUT/logs/gen.log; exit 1; }
grep "completions in" $OUT/logs/gen.log
CUDA_VISIBLE_DEVICES=2 python exp2/classify.py $OUT $BEH > $OUT/logs/classify.log 2>&1 || { echo ABORT classify; tail -3 $OUT/logs/classify.log; exit 1; }
grep ASR $OUT/logs/classify.log
python exp2/compare_double_bos.py $SRC $OUT lunaris_8b > $OUT/logs/compare.log 2>&1 || { echo ABORT compare; tail -3 $OUT/logs/compare.log; exit 1; }
echo "[$(date +%T)] LUNARIS DOUBLE-BOS DONE -> $OUT/comparison_lunaris_8b.md"
