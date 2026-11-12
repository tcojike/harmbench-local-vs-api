# harmbench-local-vs-api

**Does the same model score differently on HarmBench when run locally versus through OpenRouter?**

Model: Qwen2.5-7B-Instruct. Local = vLLM on the lab server `m3` (Ubuntu, NVIDIA L40S, *not* an Apple Silicon Mac,
see `environment/setup_m3.md`). API = `qwen/qwen-2.5-7b-instruct` on OpenRouter (served by Phala).
Same HarmBench test cases, same decoding settings, same official classifier.

* Results: `report/report.md` (write-up), `results/comparison/<set>/comparison.md` (tables)
* Plan and status: `PROJECT.md`

## Layout

```
environment/  setup_m3.md (setup + version pins + traps), requirements.txt (pip freeze)
configs/      local_model.yaml, api_model.yaml, harmbench_openrouter.patch (apply to ~/HarmBench)
scripts/      test_openrouter.py       one logged OpenRouter call (provider, tokens, cost)
              run_harmbench_local.sh   steps 1-3 on the local model
              run_harmbench_api.sh     steps 1-3 via OpenRouter (records key usage)
              compare_scores.py        tables, McNemar test, cost, reproducibility
results/      test_cases/<set>/   local/<set>/   api/<set>/   comparison/<set>/
logs/         every step's full log
report/       report.md
```

## Reproduce

```bash
source ~/.bashrc                                   # OPENROUTER_API_KEY
cd ~/harmbench-local-vs-api
python3 scripts/test_openrouter.py                 # or ~/venv-harmbench/bin/python
# smoke test (2 behaviors, ~3 min, < $0.001)
bash scripts/run_harmbench_local.sh ./data/behavior_datasets/2_behaviors.csv
bash scripts/run_harmbench_api.sh   ./data/behavior_datasets/2_behaviors.csv
# full validation set (80 behaviors, ~10 min, about one cent)
bash scripts/run_harmbench_local.sh
bash scripts/run_harmbench_api.sh
HF_HOME=~/.cache/huggingface ~/venv-harmbench/bin/python scripts/compare_scores.py
```

Run the local script before the API script (it creates the shared test cases); GPU ids are the optional 2nd argument.

## Handling the outputs

`results/*/*/completions.json` and `results.json` contain the model's raw responses to harmful requests, some of
them harmful. Keep this directory access-controlled and do not publish raw completions; `.gitignore` excludes
them so only scores and per-behavior labels get committed.

## Experiment 2 (Nov 11): serving path × precision × system prompt × model

Follow-up requested after the Nov 7 review. Qwen2.5-7B is no longer on OpenRouter, so the
serving-path comparison moved to models that are served both ways:

* **Models:** Qwen2.5-7B-Instruct (local only), Llama-3.1-8B-Instruct, L3-8B-Lunaris (a less safety-tuned Llama-3 fine-tune)
* **Factors:** local vLLM vs OpenRouter with the provider **pinned** (no fallbacks); **bf16 vs fp8** weights on both sides;
  system prompt = model template / explicit empty / Llama-2 safety prompt
* **Design:** all 320 HarmBench text **test** behaviors, **5 samples per behavior at T = 0.7**, plus greedy runs
  (local at batch 256 vs 8 = noise floor, and the API at temperature 0)
* **Stats:** paired bootstrap CI over behaviors, sign-flip permutation p-values with Holm correction, TOST equivalence at ±5 points,
  everything broken down by category (standard / contextual / copyright)
* **Logging:** one record per completion (provider that served it, response id, prompt/completion tokens, finish reason,
  latency, seed), run.json with git hashes, package versions, model revisions and the OpenRouter endpoint list at start
* **Same prompt on both sides:** local Llama prompts are rendered the way the providers render them (prompt lengths checked against
  provider-reported usage, `exp2/check_prompts.py`, `results/exp2_prompt_check.json`). DeepInfra matches exactly. CoreWeave and Parasail
  report one more token for every prompt: they prepend a BOS token to a template that already starts with one, so their models see
  two BOS tokens (`exp2/check_bos.py`, `results/exp2_bos_check.json`; inferred from counts, providers return no token ids)
* **Robustness:** flock, a new directory for each run, provider preflight, fail-fast on permanent 4xx errors, abort when more than 5% of completions are errors

```bash
source ~/.bashrc; cd ~/harmbench-local-vs-api
EXP2_LIMIT=2 bash exp2/run_exp2.sh runs/smoke     # 6-behavior smoke test, ~5 min
bash exp2/run_exp2.sh                              # full run -> results/exp2/<run>/report.md
```
Raw generations stay in `runs/` (gitignored).
