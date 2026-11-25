# PROJECT: HarmBench Local-vs-API Deployment Parity

*Does the same model score differently when run locally versus through OpenRouter?*

> Title note: the original brief said "on Apple Silicon M3". The machine used is the lab server whose hostname
> is `m3` (Ubuntu 22.04, 4 × NVIDIA L40S), not an Apple Silicon Mac. See `environment/setup_m3.md`.

## Objective

Run HarmBench on one model in two deployment modes (local on m3, API via OpenRouter) with the same behavior set
and metric, then report whether deployment mode changes the score. **Results: `report/report.md` (Experiment 1),
`report/exp2_report.md` (Experiment 2).**

## Status (updated 2026-11-25)

| Phase | Item | Status | Evidence |
|---|---|---|---|
| 0 | Audit existing work | Done | OpenRouter PoC `~/openrouter_proof_of_principle` (key in `~/.bashrc`); `~/HarmBench` cloned, not yet runnable; local models in `~/.cache/huggingface`; no prior HarmBench run on this model |
| 1 | Choose model | Done | Qwen2.5-7B-Instruct: cached locally, not gated, and OpenRouter lists `Qwen/Qwen2.5-7B-Instruct` as the HF id of `qwen/qwen-2.5-7b-instruct`. Llama-3.1-8B was not used (gated); Mistral-7B-Instruct was not cached |
| 2 | OpenRouter path | Done | `logs/test_openrouter_*.json`: provider Phala, 39+21 tokens, $0.0000081 |
| 3 | HarmBench on m3 | Done | `~/venv-harmbench`; smoke test (2 behaviors) passed for both deployments |
| 4 | Local run | Done | `results/local/harmbench_behaviors_text_val/`: ASR 23.8% |
| 5 | API run | Done | `results/api/harmbench_behaviors_text_val/`: ASR 26.2%, $0.0059 |
| 6 | Compare and report | Done | `results/comparison/.../comparison.md`, `report/report.md` |
| — | GitHub repo | Done | github.com/tcojike/harmbench-local-vs-api (private); raw completions gitignored |
| — | Professor review of Experiment 1 | Done | follow-up requested: led to Experiment 2 |
| 7 | Exp 2 design: 3 models × serving path × bf16/fp8 × system prompt, 320 test behaviors, 5 samples | Done | `exp2/conditions.py`, README "Experiment 2" |
| 8 | Exp 2 generation (18 conditions, 24,960 completions, 0 errors) | Done | `runs/exp2-20261111-220502/` (gitignored). DeepInfra fp8 lane rerun on Nov 12 after 429 throttling, see `exp2/reruns/` |
| 9 | Exp 2 scoring and analysis | Done | `results/exp2/exp2-20261111-220502/report.md` |
| 10 | Prompt parity check | Done | CoreWeave/Parasail feed a doubled BOS (`exp2/check_bos.py`); local double-BOS test reproduces CoreWeave (`results/exp2/double_bos/`) |
| 11 | Exp 2 write-up | Done | `report/exp2_report.md` |
| 12 | DeepInfra fp8 copyright anomaly (6.0 vs 22.0 ASR) | Open | Not refusals or shorter output; suspected sampling default or different build. Probe: greedy copyright set with explicit `repetition_penalty: 1.0` |
| 13 | Exp 3: Qwen3.5-9B local vs OpenRouter (bf16/fp8) | Started | `~/venv-vllm-new` (vLLM 0.31) + weights ready, `exp3/smoke_qwen35.py` passes; not yet in the pipeline |

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

## Questions to confirm with the professor (asked after Experiment 1, Nov 7)

1. Is Qwen2.5-7B-Instruct acceptable, or do you want a different model (e.g. Llama-3.1-8B, which needs gated HF access)?
2. Is the HarmBench text validation set (80 behaviors) + DirectRequest + ASR the right scope, or the full 400-behavior set?
3. What API budget is acceptable? (Everything so far cost $0.014; the full set is about $0.03 per run.)
4. Only local-vs-API for this one model, or also a small number of other models?
5. Should the next step be repeated API runs (to measure API noise) or one attack method such as PAIR/TAP?
6. Note: "M3" here is the Ubuntu GPU server, not an Apple Silicon laptop. Is that what you intended?

## Email-ready summary: Experiment 2

> Dear Professor,
>
> Following your feedback, I ran a larger deployment-parity study (Experiment 2). Since Qwen2.5-7B is no longer on
> OpenRouter, I used Llama-3.1-8B-Instruct and L3-8B-Lunaris, which are served both locally and through OpenRouter with
> pinned providers, plus Qwen2.5-7B locally for the system-prompt question. The study covers all 320 HarmBench text test
> behaviors with 5 samples each, bf16 vs fp8 on both sides, and three system prompts: 24,960 completions in total, scored by
> the official HarmBench classifier, for about $0.31 in API cost.
>
> All 12 pre-registered comparisons (local vs API, bf16 vs fp8, system prompt) are statistically equivalent within ±5
> points. The small local-vs-API difference for Llama turned out to be a prompt-format issue: two providers add a second
> beginning-of-text token. With the same extra token locally, our output matches theirs as closely as two local runs match
> each other. One endpoint (DeepInfra, fp8) has the same overall score but reproduces much less copyrighted text and
> complies more often on contextual requests, which an overall score alone would hide. The system prompt matters more
> than the serving path: Qwen's default prompt gives the lowest attack success rate (18.6% vs about 23%).
>
> The write-up is in `report/exp2_report.md` in the project repository. As a next step I would propose one probe into
> the DeepInfra difference and then the same design on a newer model (Qwen3.5-9B), which is already set up locally.
>
> Best regards,
> Emmanuel

## Email-ready summary: Experiment 1

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
