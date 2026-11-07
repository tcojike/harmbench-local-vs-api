# Environment setup on `m3`

## Important: what "M3" is in this project

The project brief assumed an **Apple Silicon M3 Mac**. The machine actually used is the lab server whose
hostname is **`m3`**:

| | |
|---|---|
| Hostname | `m3` |
| OS | Ubuntu 22.04.5 LTS, Linux 5.15, x86_64 |
| GPUs | 4 × NVIDIA L40S (46 GB each), driver 595.71, CUDA 13.2 driver / CUDA 12.4 PyTorch build |
| Python | 3.10 |

So **no Apple Silicon workarounds (MPS/CPU fallback) were needed or tested**. HarmBench's CUDA-based
stack (vLLM, the 13B classifier) runs natively. If the comparison has to be repeated on an actual Apple
M3 laptop, see "If this must run on an Apple Silicon Mac" at the end.

## 1. HarmBench

```bash
git clone https://github.com/centerforaisafety/HarmBench.git ~/HarmBench   # commit used: 8e1604d (2024-08-05)
cd ~/HarmBench
git apply ~/harmbench-local-vs-api/configs/harmbench_openrouter.patch       # adds the OpenRouter target class
```

The patch adds an `OpenRouter` class to `api_models.py` (OpenAI-compatible client pointed at
`https://openrouter.ai/api/v1`). Any `models.yaml` entry with `model_name_or_path: openrouter/<id>` uses it. The
key is read from `OPENROUTER_API_KEY` at runtime and never stored in a config file.

## 2. Python environment (separate from `venv-safety`)

```bash
/usr/bin/python3.10 -m venv ~/venv-harmbench
~/venv-harmbench/bin/pip install -r ~/harmbench-local-vs-api/environment/requirements.txt
```

A separate venv is required: in the existing `~/venv-safety`, `import vllm` fails
(`OSError: libcudart.so.13`) because its torchaudio 2.11 build expects CUDA 13 while torch is 2.5.1+cu124.

### Version pins that matter (and why)

| Package | Version | Why |
|---|---|---|
| torch / vllm | 2.5.1 / 0.6.6.post1 | matching pair; vLLM API used by HarmBench still works |
| transformers | 4.47.1 | HarmBench + fschat predate transformers 5 |
| anthropic | 0.39.0 | `api_models.py` imports `HUMAN_PROMPT`, removed in newer SDKs |
| mistralai | 0.4.2 (<1.0) | `api_models.py` imports `MistralClient`, removed in 1.0 |
| google-generativeai | 0.8.6 | imported unconditionally by `api_models.py` |
| sentencepiece | 0.2.0 | 0.2.3 ships protobuf-7 generated code -> `VersionError` with protobuf 5.29 |
| datasketch | 1.6.5 | 2.0 refuses the shipped copyright MinHash `.pkl` files ("different permutation schemes") |
| spaCy model | en_core_web_sm 3.7.1 (wheel URL) | `spacy download` builds a broken URL with spacy 3.7.2 |

## 3. Models (no downloads needed)

Both are already in `~/.cache/huggingface`. Note `~/.bashrc` sets `HF_HOME=/data-sda/llms/huggingface`, which
does **not** contain them, so the run scripts set `HF_HOME=~/.cache/huggingface HF_HUB_OFFLINE=1`.

| Role | Model | Size |
|---|---|---|
| Target (local) | `Qwen/Qwen2.5-7B-Instruct` | 15 GB, bf16 |
| Judge | `cais/HarmBench-Llama-2-13b-cls` (official HarmBench classifier) | 25 GB |

Llama-2-7b-chat (the manual's Quick Start model) was not used: it is gated and there is no Hugging Face token on m3.

## 4. OpenRouter

```bash
grep OPENROUTER_API_KEY ~/.bashrc     # export OPENROUTER_API_KEY=... (professor's setup)
~/venv-harmbench/bin/python ~/harmbench-local-vs-api/scripts/test_openrouter.py
```

## 5. Known traps

* `scripts/run_pipeline.py --mode local` hides failures: it runs steps 2/3 with `capture_output=True` and never
  checks the return code, so it can exit 0 with no completions. This project calls `generate_completions.py` /
  `evaluate_completions.py` directly and checks that every output file exists.
* `scripts/*.sh` in HarmBench start with `source /opt/rh/devtoolset-10/enable` (RHEL only); harmless on Ubuntu.

## If this must run on an Apple Silicon Mac

Not tested. Expected changes: vLLM has no Metal backend, so step 2 would use HarmBench's HF-transformers path
(omit `--generate_with_vllm`; set the device to `mps`) or a separate runner such as llama.cpp/MLX, which would mean
a different (often quantized) model build. The 13B classifier needs about 26 GB in fp16, so it would need a Mac
with at least 32 GB unified memory, or it can be run on this server instead. Expect generation to take much longer.
