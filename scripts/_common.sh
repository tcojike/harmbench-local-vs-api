# Shared setup for run_harmbench_local.sh / run_harmbench_api.sh (sourced, not run directly).
set -euo pipefail

PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
HB=${HARMBENCH_DIR:-$HOME/HarmBench}
VENV=${HARMBENCH_VENV:-$HOME/venv-harmbench}

# Behavior set: arg 1, default = HarmBench text validation set (80 behaviors).
# Smoke test: ./data/behavior_datasets/2_behaviors.csv
BEHAVIORS=${1:-./data/behavior_datasets/harmbench_behaviors_text_val.csv}
SET=$(basename "$BEHAVIORS" .csv)
MAX_NEW_TOKENS=512                              # HarmBench standard
CLS=cais/HarmBench-Llama-2-13b-cls              # official HarmBench classifier
TEST_CASES=$PROJ/results/test_cases/$SET/test_cases.json

export PATH=$VENV/bin:$PATH
# Both the target model and the classifier are cached in ~/.cache/huggingface (the HF_HOME in ~/.bashrc
# points elsewhere); offline mode guarantees nothing is downloaded.
export HF_HOME=${HARMBENCH_HF_HOME:-$HOME/.cache/huggingface} HF_HUB_OFFLINE=1

cd "$HB"
log() { echo "[$(date +%H:%M:%S)] $*"; }
need() { [ -s "$1" ] || { echo "FAILED: missing output $1 (see $PROJ/logs)"; exit 1; }; }

# Step 1 (shared by both runs): DirectRequest test cases = the behavior strings themselves.
make_test_cases() {
    if [ -s "$TEST_CASES" ]; then log "test cases exist: $TEST_CASES"; return; fi
    log "step 1: generating DirectRequest test cases for $SET"
    python -u generate_test_cases.py --method_name=DirectRequest --experiment_name=default \
        --behaviors_path="$BEHAVIORS" --save_dir="$(dirname "$TEST_CASES")" \
        > "$PROJ/logs/step1_$SET.log" 2>&1
    need "$TEST_CASES"
}

# Step 3: score completions with the HarmBench classifier (needs one GPU, ~27 GB).
evaluate() {  # $1 = completions.json, $2 = results.json, $3 = log, $4 = GPU id
    log "step 3: classifier scoring on GPU $4"
    CUDA_VISIBLE_DEVICES=$4 python -u evaluate_completions.py --cls_path=$CLS --behaviors_path="$BEHAVIORS" \
        --completions_path="$1" --save_path="$2" > "$3" 2>&1
    need "$2"
    grep 'Average ASR' "$3"
}
