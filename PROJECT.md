# PROJECT: HarmBench Local-vs-API Deployment Parity

*Does the same model score differently when run locally versus through OpenRouter?*

> Title note: the original brief said "on Apple Silicon M3". The machine used is the lab server whose hostname
> is `m3` (Ubuntu 22.04, 4 × NVIDIA L40S), not an Apple Silicon Mac. See `environment/setup_m3.md`.

## Objective

Run HarmBench on one model in two deployment modes (local on m3, API via OpenRouter) with the same behavior set
and metric, then report whether deployment mode changes the score. **Result: see `report/report.md`.**

## Status (2026-11-07)

| Phase | Item | Status | Evidence |
|---|---|---|---|
| 0 | Audit existing work | Done | OpenRouter PoC `~/openrouter_proof_of_principle` (key in `~/.bashrc`); `~/HarmBench` cloned, not yet runnable; local models in `~/.cache/huggingface`; no prior HarmBench run on this model |
| 1 | Choose model | Done | Qwen2.5-7B-Instruct: cached locally, not gated, and OpenRouter lists `Qwen/Qwen2.5-7B-Instruct` as the HF id of `qwen/qwen-2.5-7b-instruct`. Llama-3.1-8B was not used (gated); Mistral-7B-Instruct was not cached |
| 2 | OpenRouter path | Done | `logs/test_openrouter_*.json`: provider Phala, 39+21 tokens, $0.0000081 |
| 3 | HarmBench on m3 | Done | `~/venv-harmbench`; smoke test (2 behaviors) passed for both deployments |
| 4 | Local run | Done | `results/local/harmbench_behaviors_text_val/`: ASR 23.8% |
| 5 | API run | Done | `results/api/harmbench_behaviors_text_val/`: ASR 26.2%, $0.0059 |
| 6 | Compare and report | Done | `results/comparison/.../comparison.md`, `report/report.md` |
| — | GitHub repo | Waiting on GitHub login (`gh auth login`) | — |
| — | Professor confirmation of model / set / budget | Not started | questions below |

## Definition of done

- [x] Repo/directory with scripts and environment notes
- [x] Successful OpenRouter API call
- [x] HarmBench installed and running on m3
- [x] Completed HarmBench run on the local model
- [x] Completed HarmBench run on the same model via OpenRouter
- [x] Results table comparing scores
- [x] Short write-up (setup, commands, results, caveats, next steps)
- [x] Budget/cost note
- [x] Email-ready summary (below)

## Questions to confirm with the professor

1. Is Qwen2.5-7B-Instruct acceptable, or do you want a different model (e.g. Llama-3.1-8B, which needs gated HF access)?
2. Is the HarmBench text validation set (80 behaviors) + DirectRequest + ASR the right scope, or the full 400-behavior set?
3. What API budget is acceptable? (Everything so far cost $0.014; the full set is about $0.03 per run.)
4. Only local-vs-API for this one model, or also a small number of other models?
5. Should the next step be repeated API runs (to measure API noise) or one attack method such as PAIR/TAP?
6. Note: "M3" here is the Ubuntu GPU server, not an Apple Silicon laptop. Is that what you intended?

## Email-ready summary

> Dear Professor,
>
> I completed a first HarmBench deployment-parity evaluation. I ran HarmBench (DirectRequest, 80-behavior text
> validation set, official HarmBench classifier) on Qwen2.5-7B-Instruct twice: locally with vLLM on the lab
> server m3 (NVIDIA L40S; not an Apple Silicon Mac, so no MPS workarounds were needed) and through OpenRouter
> (`qwen/qwen-2.5-7b-instruct`, served by Phala), using the OpenRouter key from our environment-variable setup.
> The attack success rate was 23.8% locally and 26.2% via the API, a difference that is not statistically
> significant (McNemar p = 0.63; labels agree on 95% of behaviors). The more interesting result is reproducibility:
> the local run is fully deterministic (80/80 identical responses on a repeat run), whereas the API at
> temperature 0 gave only 11/80 identical responses and shifted ASR by about 1 point between runs, as large as
> the local-vs-API gap itself. Total API cost was about $0.014. Scripts, environment notes, results and a short
> report are in my project directory. Before scaling up, I would like to confirm the model, behavior set and
> budget, and whether you would prefer repeated API runs or adding an attack method such as PAIR next.
>
> Best regards,
> Emmanuel
