#!/usr/bin/env bash
# Local run: HarmBench DirectRequest on Qwen2.5-7B-Instruct served by vLLM on this machine.
# Usage: bash scripts/run_harmbench_local.sh [behaviors_csv] [gpu_id]
#   default behaviors = HarmBench text validation set (80); smoke test: ./data/behavior_datasets/2_behaviors.csv
source "$(dirname "$0")/_common.sh"
GPU=${2:-0}
MODEL=qwen2_5_7b_instruct
OUT=$PROJ/results/local/$SET; mkdir -p "$OUT"

make_test_cases
log "step 2: local completions (vLLM, GPU $GPU, greedy, max_new_tokens=$MAX_NEW_TOKENS)"
t0=$(date +%s)
CUDA_VISIBLE_DEVICES=$GPU python -u generate_completions.py --model_name=$MODEL \
    --models_config_file="$PROJ/configs/local_model.yaml" --behaviors_path="$BEHAVIORS" \
    --test_cases_path="$TEST_CASES" --save_path="$OUT/completions.json" \
    --max_new_tokens=$MAX_NEW_TOKENS --generate_with_vllm > "$PROJ/logs/step2_local_$SET.log" 2>&1
t1=$(date +%s)
need "$OUT/completions.json"
evaluate "$OUT/completions.json" "$OUT/results.json" "$PROJ/logs/step3_local_$SET.log" "$GPU"
t2=$(date +%s)

cat > "$OUT/run_meta.json" <<EOF
{"deployment": "local", "model": "$MODEL", "hf_id": "Qwen/Qwen2.5-7B-Instruct", "behaviors": "$SET",
 "host": "$(hostname)", "gpu": "$(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader)",
 "precision": "bfloat16", "engine": "vllm $(python -c 'import vllm;print(vllm.__version__)' 2>/dev/null)",
 "temperature": 0, "max_new_tokens": $MAX_NEW_TOKENS,
 "generation_seconds": $((t1-t0)), "scoring_seconds": $((t2-t1)), "cost_usd": 0,
 "finished": "$(date -Iseconds)"}
EOF
log "DONE -> $OUT"
