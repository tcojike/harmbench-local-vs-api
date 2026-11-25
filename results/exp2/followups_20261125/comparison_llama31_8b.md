# Double BOS vs single BOS (local vLLM, Llama-3.1-8B-Instruct bf16, template system prompt)

Only change: one extra `<|begin_of_text|>` id in front of the prompt, as CoreWeave does.

## Greedy (one completion per behavior)

| n | ASR 1 BOS | ASR 2 BOS | harmful only 1 BOS / only 2 BOS | McNemar p | Identical text |
|---|---|---|---|---|---|
| 320 | 8.1 | 8.8 | 4 / 6 | 0.754 | 38.1% |

## Sampled (5 completions per behavior, T 0.7, same seeds)

| Category | n | ASR 1 BOS | ASR 2 BOS | Diff (pts) | 95% CI | 90% CI | p (permutation) | Equivalent ±5? |
|---|---|---|---|---|---|---|---|---|
| ALL | 320 | 7.4 | 8.6 | +1.2 | [-0.8, +3.2] | [-0.4, +2.9] | 0.269 | yes |
| standard | 159 | 0.6 | 2.3 | +1.6 | [+0.0, +3.6] | [+0.4, +3.3] | 0.253 | yes |
| contextual | 81 | 6.2 | 9.9 | +3.7 | [+0.0, +8.6] | [+1.2, +7.4] | 0.254 | no |
| copyright | 80 | 22.0 | 19.7 | -2.2 | [-8.0, +3.2] | [-7.0, +2.3] | 0.501 | no |

Identical text with the same seed: 32.2% of sampled completions.

## Local vs API (CoreWeave bf16, which feeds 2 BOS), sampled

### Local with 1 BOS (the exp2 contrast)

| Category | n | ASR local | ASR API | Diff (pts) | 95% CI | 90% CI | p (permutation) | Equivalent ±5? |
|---|---|---|---|---|---|---|---|---|
| ALL | 320 | 7.4 | 8.2 | +0.9 | [-0.9, +2.8] | [-0.7, +2.5] | 0.397 | yes |
| standard | 159 | 0.6 | 2.1 | +1.5 | [-0.3, +3.5] | [+0.0, +3.3] | 0.147 | yes |
| contextual | 81 | 6.2 | 9.6 | +3.5 | [-1.0, +8.4] | [-0.2, +7.7] | 0.187 | no |
| copyright | 80 | 22.0 | 19.0 | -3.0 | [-7.5, +1.0] | [-6.8, +0.5] | 0.221 | no |

### Local with 2 BOS (prompts matched)

| Category | n | ASR local | ASR API | Diff (pts) | 95% CI | 90% CI | p (permutation) | Equivalent ±5? |
|---|---|---|---|---|---|---|---|---|
| ALL | 320 | 8.6 | 8.2 | -0.3 | [-1.7, +1.1] | [-1.5, +0.8] | 0.727 | yes |
| standard | 159 | 2.3 | 2.1 | -0.1 | [-1.4, +1.1] | [-1.1, +0.9] | 1.000 | yes |
| contextual | 81 | 9.9 | 9.6 | -0.2 | [-3.7, +3.0] | [-3.0, +2.5] | 1.000 | yes |
| copyright | 80 | 19.7 | 19.0 | -0.8 | [-4.3, +3.0] | [-3.8, +2.2] | 0.796 | yes |

Sampled text is not compared across local and API: the provider's sampler and seeds differ, so identical text is not expected.

Per-category rows are descriptive (small n, no multiplicity correction).
