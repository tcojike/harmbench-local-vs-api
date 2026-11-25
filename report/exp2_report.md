# Experiment 2: Serving Path × Precision × System Prompt × Model

*Does HarmBench ASR change when the same model runs locally (vLLM on `m3`) or through OpenRouter, in bf16 or fp8, with
different system prompts?*

Run: `exp2-20261111-220502` (2026-11-11/12) · Machine: `m3` (Ubuntu, 4 × NVIDIA L40S) · Full tables:
`results/exp2/exp2-20261111-220502/report.md` · Follow-up check: `results/exp2/double_bos/comparison.md`

## 1. Summary

* **The serving path does not change overall ASR.** All 12 pre-registered contrasts (local vs API, bf16 vs fp8, system
  prompt) for Llama-3.1-8B and L3-8B-Lunaris are statistically equivalent within ±5 points (TOST, α = 0.05). The largest
  local-vs-API gap is 1.6 points.
* **The Llama local-vs-CoreWeave difference is a prompt-format difference, not the serving stack.** CoreWeave and
  Parasail prepend a BOS token to a Llama-3 template that already starts with one, so their models see two. With the same
  extra BOS added locally, local greedy output matches CoreWeave at the noise floor: 100% label agreement and 67.5%
  identical text, against 71.2% between two local runs that differ only in batch size.
* **The system prompt matters more than the serving path.** Qwen2.5-7B's default template system prompt gives the lowest
  ASR (18.6%). An empty prompt gives 23.1% and the Llama-2 safety prompt 22.8% (+4.2 to +4.6 points; p = 0.10–0.14 after
  Holm correction). The safety prompt lowers Llama ASR through the API (−3.0, p = 0.004) but not locally (−0.5).
* **One endpoint behaves differently underneath an equivalent total.** Llama-3.1-8B on DeepInfra fp8 has the same overall
  ASR as local (7.3 vs 7.4), but much less copyright reproduction (6.0 vs 22.0) and more contextual compliance (15.1 vs
  6.2). It is not a refusal or length effect (section 5).

## 2. Design

| Factor | Levels |
|---|---|
| Model | Qwen2.5-7B-Instruct (local only: no longer on OpenRouter), Llama-3.1-8B-Instruct, L3-8B-Lunaris-v1 (less safety-tuned Llama-3 fine-tune) |
| Serving path | local vLLM 0.6.6.post1 vs OpenRouter with the provider pinned (`allow_fallbacks: false`) |
| Precision | bf16 vs fp8, on both sides: local dynamic fp8; API CoreWeave/Parasail bf16 vs DeepInfra fp8/turbo |
| System prompt | model template / explicit empty / Llama-2 safety prompt |
| Decoding | 5 samples per behavior at T = 0.7 (main analysis); greedy runs for the noise floor (local batch 256 vs 8, API T = 0) |

* **Behaviors:** all 320 HarmBench text **test** behaviors (159 standard, 81 contextual, 80 copyright), DirectRequest.
* **Metric:** ASR = mean over behaviors of the share of harmful completions. Judge: official HarmBench-Llama-2-13b-cls
  run locally for every condition; copyright is scored with HarmBench's hash check.
* **Statistics:** behavior is the unit; paired bootstrap 95% CI; sign-flip permutation p with Holm correction over the 12
  contrasts; TOST equivalence at ±5 points; greedy contrasts use exact McNemar.
* **Scale:** 18 conditions, 24,960 completions, 0 errors; every API completion was served by the pinned provider.

## 3. Results

| Model | Local bf16 | Local fp8 | API bf16 | API fp8 |
|---|---|---|---|---|
| Llama-3.1-8B, template prompt | 7.4 | 7.5 | 8.2 (CoreWeave) | 7.3 (DeepInfra) |
| Llama-3.1-8B, safety prompt | 6.9 | 5.6 | 5.3 (CoreWeave) | 2.8 (DeepInfra) |
| Lunaris-8B, template prompt | 41.9 | 42.7 | 43.4 (Parasail) | 42.7 (DeepInfra) |
| Qwen2.5-7B (local bf16) | template 18.6 · empty 23.1 · safety 22.8 | | | |

ASR in %, sampled decoding. Per-category values, CIs and p-values are in the full tables.

| Contrast (B − A) | Diff (pts) | 95% CI | p (Holm) | Equivalent ±5 |
|---|---|---|---|---|
| Llama: local vs API, bf16 | +0.9 | [−0.9, +2.8] | 1.00 | yes |
| Llama: local bf16 vs API fp8 | −0.1 | [−2.8, +2.8] | 1.00 | yes |
| Llama: bf16 vs fp8, local / API | +0.1 / −0.9 | [−1.4, +1.6] / [−3.3, +1.4] | 1.00 | yes |
| Llama + safety prompt: local vs API, bf16 | −1.6 | [−3.2, −0.3] | 0.27 | yes |
| Lunaris: local vs API, bf16 | +1.4 | [−1.0, +3.9] | 1.00 | yes |
| Lunaris: bf16 vs fp8, local / API | +0.8 / −0.7 | [−1.2, +2.8] / [−2.7, +1.3] | 1.00 | yes |
| Llama: safety prompt effect, local / API | −0.5 / −3.0 | [−2.6, +1.4] / [−4.8, −1.4] | 1.00 / 0.004 | yes |
| Qwen: template → empty / → safety | +4.6 / +4.2 | [+1.0, +8.1] / [+1.2, +7.4] | 0.14 / 0.10 | no |

**Greedy noise floor (Llama, 320 behaviors).** Local batch 256 vs batch 8: identical labels, 71.2% identical text. Local
vs CoreWeave at T = 0: 96.9% label agreement (4 / 6 flips, McNemar p = 0.75), 37.5% identical text.

## 4. The doubled BOS

The prompt check found that CoreWeave and Parasail report exactly one more prompt token than the local run for every
behavior, while DeepInfra matches exactly. `exp2/check_bos.py` sends the same prompt raw through `/completions`:
CoreWeave and Parasail count a BOS-less prompt as the local length (they prepend BOS themselves) and a prompt that
already starts with `<|begin_of_text|>` as local + 1. Their chat requests count the same as the latter, so their
models see `<|begin_of_text|><|begin_of_text|>…`. This is inferred from token counts; providers do not return token ids.

To test whether this matters, the two local Llama template conditions were regenerated with one extra BOS id and nothing
else changed (`exp2/check_double_bos_local.py`):

| Greedy pair (Llama-3.1-8B) | Label agreement | Identical text | Same first 40 chars |
|---|---|---|---|
| local, 1 BOS vs CoreWeave | 96.9% | 37.5% | 60.0% |
| **local, 2 BOS vs CoreWeave** | **100.0%** | **67.5%** | **91.6%** |
| local batch 256 vs batch 8 (noise floor) | 100.0% | 71.2% | 95.3% |

With matching prompts, what remains of the serving-stack difference (hardware, kernels, batching) is no larger than local
batch-size noise. The extra BOS alone moves Llama's ASR by about +1 point (sampled 7.4 → 8.6, p = 0.27): within the
equivalence margin, but it is a real input difference. Lunaris on Parasail gets the same extra BOS; its effect there was
not measured.

## 5. DeepInfra fp8: same total, different composition

| Llama-3.1-8B, template prompt | Standard | Contextual | Copyright | Overall |
|---|---|---|---|---|
| local bf16 / local fp8 / CoreWeave bf16 | 0.6 / 0.6 / 2.1 | 6.2 / 7.9 / 9.6 | 22.0 / 20.8 / 19.0 | 7.4 / 7.5 / 8.2 |
| **DeepInfra fp8** | **4.0** | **15.1** | **6.0** | **7.3** |

The same pattern appears with the safety prompt (copyright 4.0 vs 17–22). Copyright vs local bf16: −16.0 points, 95% CI
[−23.8, −9.0]; contextual: +8.9, [+4.0, +14.6]. Per-category CIs are not multiplicity-corrected.

On the 80 copyright behaviors, DeepInfra's responses are not refusals and not shorter:

| Copyright behaviors (400 completions each) | ASR | Opens with a refusal | Mean completion tokens | Hit the 512 limit |
|---|---|---|---|---|
| Llama local bf16 | 22.0 | 0.0% | 452 | 54.0% |
| Llama local fp8 | 20.8 | 0.0% | 443 | 54.2% |
| Llama CoreWeave bf16 | 19.0 | 8.2% | 412 | 45.2% |
| **Llama DeepInfra fp8** | **6.0** | **0.0%** | **436** | **49.8%** |
| Lunaris DeepInfra turbo | 18.0 | 2.8% | 392 | 27.8% |

DeepInfra fp8 produces as much text, but far less of it matches the source closely enough for the hash check.
Ruled out:
* **fp8 itself:** local fp8 is normal.
* **DeepInfra in general:** Lunaris on DeepInfra is normal.
* **The prompt:** one BOS and identical length.

Remaining candidates:
* a different weight build or quantization recipe at DeepInfra
* a provider-side sampling default such as a repetition or frequency penalty, which would suppress verbatim recall
* drift: these two conditions ran about 9 hours after the rest

**Untested.** A cheap check: rerun the 80 copyright behaviors greedily on DeepInfra fp8, with and without an explicit
`repetition_penalty: 1.0`, and compare with local.

## 6. Caveats

1. **Two conditions ran later.** DeepInfra throttled the fp8 lane (HTTP 429) on Nov 11; its process was killed before
   writing anything, and the safety-prompt condition behind it never started. Both were rerun on Nov 12 with the same
   endpoint, seeds, decoding and test cases (`exp2/reruns/`). `gen_api.py` now checkpoints every completion.
2. **Provider opacity.** Provider builds, quantization recipes and sampling defaults are not disclosed, and providers can
   change them at any time. Qwen2.5-7B disappeared from OpenRouter between Experiment 1 and 2.
3. **Truncation.** 35–43% of Qwen responses and 6–25% of the others hit the 512-token limit. The classifier reads at most
   512 tokens anyway, but long copyright reproductions are cut.
4. **Judge.** The same local classifier scores every condition, so its errors affect all sides equally. Borderline
   responses can flip labels on small wording changes.
5. **Scope.** DirectRequest only (no optimized attacks), three ~8B models, one machine.

## 7. Cost

API: about **$0.31** for the 7 API conditions (CoreWeave Llama $0.22, DeepInfra $0.06, Parasail $0.03), from logged
token counts and the endpoint prices recorded at the start of the run. The aborted Nov 11 lane and the BOS probes add a few cents. Local
generation: 18.5 GPU-minutes of decoding (8 minutes wall clock on 4 × L40S, model loading included); scoring all 16
original conditions: 10 minutes wall clock on 4 GPUs.

## 8. Takeaways for deployment-parity studies

1. **Pin the provider and log who served each response.** Results differ by endpoint, not by "the API".
2. **Check prompt token counts against the provider's.** An off-by-one here was a second BOS, and it changes outputs.
3. **Look at categories, not only the total.** DeepInfra's equivalent total hides a 16-point copyright shift.
4. **Measure the noise floor.** Local greedy runs at different batch sizes are not byte-identical (71%), so text identity
   alone cannot show that two deployments differ.
