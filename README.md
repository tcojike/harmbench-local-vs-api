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
