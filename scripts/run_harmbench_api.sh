#!/usr/bin/env bash
# API run: HarmBench DirectRequest on qwen/qwen-2.5-7b-instruct via OpenRouter (same test cases as the local run).
# Usage: bash scripts/run_harmbench_api.sh [behaviors_csv] [gpu_id_for_classifier]
# Key: $OPENROUTER_API_KEY from ~/.bashrc. Cost = OpenRouter key usage before/after the run
# (exact as long as nothing else uses the key at the same time).
source "$(dirname "$0")/_common.sh"
GPU=${2:-1}
MODEL=qwen2_5_7b_instruct_openrouter
OUT=$PROJ/results/api/$SET; mkdir -p "$OUT"
[ -n "${OPENROUTER_API_KEY:-}" ] || { echo "OPENROUTER_API_KEY not set (source ~/.bashrc)"; exit 1; }
grep -q 'class OpenRouter' api_models.py || { echo "apply configs/harmbench_openrouter.patch to $HB first"; exit 1; }

key_usage() { curl -s https://openrouter.ai/api/v1/key -H "Authorization: Bearer $OPENROUTER_API_KEY" \
              | python -c 'import json,sys; print(json.load(sys.stdin)["data"]["usage"])'; }

make_test_cases
u0=$(key_usage)
log "step 2: API completions via OpenRouter (temperature 0, max_tokens=$MAX_NEW_TOKENS); key usage before: \$$u0"
t0=$(date +%s)
CUDA_VISIBLE_DEVICES="" python -u generate_completions.py --model_name=$MODEL \
    --models_config_file="$PROJ/configs/api_model.yaml" --behaviors_path="$BEHAVIORS" \
    --test_cases_path="$TEST_CASES" --save_path="$OUT/completions.json" \
    --max_new_tokens=$MAX_NEW_TOKENS > "$PROJ/logs/step2_api_$SET.log" 2>&1
t1=$(date +%s)
need "$OUT/completions.json"
sleep 20   # OpenRouter usage counters update with a short delay
u1=$(key_usage)
ERRS=$( (grep -o '\$ERROR\$' "$OUT/completions.json" || true) | wc -l)   # grep exits 1 on zero matches
log "key usage after: \$$u1   API errors: $ERRS"
evaluate "$OUT/completions.json" "$OUT/results.json" "$PROJ/logs/step3_api_$SET.log" "$GPU"
t2=$(date +%s)

cat > "$OUT/run_meta.json" <<EOF
{"deployment": "openrouter", "model": "$MODEL", "openrouter_id": "qwen/qwen-2.5-7b-instruct", "behaviors": "$SET",
 "provider": "Phala (only endpoint listed)", "precision": "unknown (provider-reported quantization: unknown)",
 "temperature": 0, "top_p": 1.0, "max_new_tokens": $MAX_NEW_TOKENS, "api_errors": $ERRS,
 "generation_seconds": $((t1-t0)), "scoring_seconds": $((t2-t1)),
 "cost_usd_key_usage_delta": $(python -c "print(round($u1-$u0, 6))"), "cost_note": "key-usage delta lags; see compare_scores.py token-based cost", "finished": "$(date -Iseconds)"}
EOF
log "DONE -> $OUT"
