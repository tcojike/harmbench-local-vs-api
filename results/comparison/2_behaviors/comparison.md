# HarmBench local vs OpenRouter: `2_behaviors` (2 behaviors)

## Run metrics

| Field | Local | API |
|---|---|---|
| Model | Qwen/Qwen2.5-7B-Instruct | qwen/qwen-2.5-7b-instruct |
| Deployment | m3 (Ubuntu, NVIDIA L40S), vllm 0.6.6.post1 | OpenRouter -> Phala (only endpoint listed) |
| Behavior set | 2_behaviors | 2_behaviors |
| Method / metric | DirectRequest / ASR (HarmBench-Llama-2-13b-cls) | same |
| **Score (ASR)** | **50.0%** (1/2) | **50.0%** (1/2) |
| Generation time | 44 s | 8 s |
| Scoring time | 34 s | 54 s |
| Cost | $0 (own GPU) | $0.0001 (352 in + 463 out tokens) |
| Errors | 0 | 0 |
| Precision | bfloat16 | unknown (provider-reported quantization: unknown) |
| Sampling | greedy (temperature 0), max_new_tokens 512 | temperature 0, top_p 1, max_tokens 512 |

## ASR by category

| Category | n | Local | API | Diff (pts) |
|---|---|---|---|---|
| standard | 2 | 50.0% | 50.0% | +0.0 |
| ALL | 2 | 50.0% | 50.0% | +0.0 |

## Paired comparison

- Label agreement: 100.0%
- Harmful only locally: 0; harmful only via API: 0
- Exact McNemar test: p = 1.000 (not significant at 0.05)
- Byte-identical responses: 0/2; same first 40 characters: 100.0%

## Run-to-run reproducibility (vs an earlier identical run)

| Deployment | ASR earlier | ASR now | Label agreement | Identical responses |
|---|---|---|---|---|
| local | 50.0% | 50.0% | 100.0% | 2/2 |
| api | 50.0% | 50.0% | 100.0% | 0/2 |
