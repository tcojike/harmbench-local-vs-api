#!/usr/bin/env bash
# Preflight check before loading a model on this machine: GPUs, memory, disk, model caches, Ollama.
# Read-only: it installs, starts and downloads nothing.
# Usage: bash scripts/preflight.sh            (output is also saved to logs/preflight_<host>_<timestamp>.txt)
set -uo pipefail   # no -e: a failed check is reported, not fatal

PROJ=$(cd "$(dirname "$0")/.." && pwd)
OUT=$PROJ/logs/preflight_$(hostname)_$(date +%Y%m%d-%H%M%S).txt
mkdir -p "$PROJ/logs"
exec > >(tee "$OUT") 2>&1

sec() { printf '\n===== %s =====\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

sec "HOST"
echo "date:     $(date -Iseconds)"
echo "host:     $(hostname)"
echo "os:       $(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-unknown}") | kernel $(uname -r) | arch $(uname -m)"
echo "cpu:      $(nproc) cores | $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs)"
echo "user:     $(id -un)"
[ "$(uname -s)" = Darwin ] && echo "NOTE: macOS / Apple Silicon detected: no CUDA; use MPS / llama.cpp / MLX"

sec "MEMORY"
free -h 2>/dev/null || vm_stat 2>/dev/null

sec "GPUs (nvidia-smi)"
if have nvidia-smi; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv
    echo
    echo "driver / CUDA: $(nvidia-smi | grep -o 'Driver Version: [0-9.]*\|CUDA Version: [0-9.]*' | xargs)"
    echo
    echo "-- processes currently using GPU memory:"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null \
        | while IFS=, read -r uuid pid name mem; do
            idx=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | grep "$(echo "$uuid" | xargs)" | cut -d, -f1)
            owner=$(ps -o user= -p "$(echo "$pid" | xargs)" 2>/dev/null)
            echo "   GPU $idx | pid$pid | ${owner:-?} |$name |$mem"
          done
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && echo "   (none)"
    echo
    echo "-- GPUs with >= 40 GB free (enough for a 7-8B model in bf16 + KV cache, or a 13B judge):"
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | awk -F, '$2+0 >= 40000 {printf "   GPU%s (%.1f GB free)\n", $1, $2/1024; n++} END {if (!n) print "   none"}'
else
    echo "nvidia-smi not found: no NVIDIA GPU / driver visible"
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

sec "DISK"
df -h / "$HOME" /data-sda /tmp 2>/dev/null | awk 'NR==1 || !seen[$0]++'
echo
for v in HF_HOME OLLAMA_MODELS; do echo "$v=${!v:-<unset>}"; done

sec "MODEL CACHES"
declare -A seen_dir=()
for d in "$HOME/.cache/huggingface/hub" "${HF_HOME:-/nonexistent}/hub" "$HOME/.ollama/models" \
         "/usr/share/ollama/.ollama/models" "${OLLAMA_MODELS:-/nonexistent}"; do
    [ -d "$d" ] || continue
    real=$(realpath "$d"); [ -n "${seen_dir[$real]:-}" ] && continue; seen_dir[$real]=1   # skip the same dir twice
    echo "-- $d  ($(du -sh "$d" 2>/dev/null | cut -f1))"
    ls "$d" 2>/dev/null | grep '^models--' | sed 's/^models--/   /; s/--/\//'
done

sec "OLLAMA"
if have ollama; then
    echo "binary:   $(command -v ollama) | $(ollama --version 2>&1 | head -1)"
else
    echo "binary:   NOT INSTALLED (ollama not on PATH)"
fi
if have systemctl; then
    echo "service:  $(systemctl is-active ollama 2>/dev/null || true) (systemd unit 'ollama'; enabled: $(systemctl is-enabled ollama 2>/dev/null || echo n/a))"
fi
echo "process:  $(pgrep -a ollama 2>/dev/null | head -3 | tr '\n' ';' || true)"
if have curl && curl -s -m 3 http://localhost:11434/api/version >/dev/null 2>&1; then
    echo "api:      UP at localhost:11434 ($(curl -s -m 3 http://localhost:11434/api/version))"
    echo "-- installed models:"
    curl -s -m 5 http://localhost:11434/api/tags | python3 -c '
import json,sys
m=json.load(sys.stdin).get("models",[])
print("\n".join(f"   {x[\"name\"]:40s} {x[\"size\"]/1e9:6.1f} GB  {x.get(\"details\",{}).get(\"quantization_level\",\"\")}" for x in m) or "   (none)")'
    echo "-- loaded right now:"
    curl -s -m 5 http://localhost:11434/api/ps | python3 -c '
import json,sys
m=json.load(sys.stdin).get("models",[])
print("\n".join(f"   {x[\"name\"]} ({x.get(\"size_vram\",0)/1e9:.1f} GB VRAM)" for x in m) or "   (none)")'
else
    echo "api:      NOT reachable at localhost:11434"
fi

sec "PYTHON ENVIRONMENTS"
for v in "$HOME/venv-harmbench" "$HOME/venv-safety"; do
    [ -x "$v/bin/python" ] || continue
    echo "-- $v: $("$v/bin/python" -c 'import torch; print("torch", torch.__version__, "| cuda ok:", torch.cuda.is_available(), "| gpus:", torch.cuda.device_count())' 2>&1 | tail -1)"
done

sec "SUMMARY"
echo "saved to: $OUT"
