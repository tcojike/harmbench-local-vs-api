# Double BOS vs single BOS (local vLLM, L3-8B-Lunaris-v1 bf16, template system prompt)

Only change: one extra `<|begin_of_text|>` id in front of the prompt, as Parasail does.

*No local greedy condition for lunaris_8b in exp2; sampled comparison only.*

## Sampled (5 completions per behavior, T 0.7, same seeds)

| Category | n | ASR 1 BOS | ASR 2 BOS | Diff (pts) | 95% CI | 90% CI | p (permutation) | Equivalent ±5? |
|---|---|---|---|---|---|---|---|---|
| ALL | 320 | 41.9 | 43.3 | +1.4 | [-1.2, +4.1] | [-0.8, +3.6] | 0.333 | yes |
| standard | 159 | 29.7 | 30.9 | +1.3 | [-2.3, +4.8] | [-1.6, +4.2] | 0.579 | yes |
| contextual | 81 | 86.4 | 89.6 | +3.2 | [-1.7, +8.6] | [-1.0, +7.7] | 0.282 | no |
| copyright | 80 | 21.2 | 21.0 | -0.3 | [-6.2, +5.8] | [-5.3, +4.8] | 1.000 | no |

Identical text with the same seed: 5.2% of sampled completions.

## Local vs API (Parasail bf16, which feeds 2 BOS), sampled

### Local with 1 BOS (the exp2 contrast)

| Category | n | ASR local | ASR API | Diff (pts) | 95% CI | 90% CI | p (permutation) | Equivalent ±5? |
|---|---|---|---|---|---|---|---|---|
| ALL | 320 | 41.9 | 43.4 | +1.4 | [-1.1, +4.0] | [-0.6, +3.5] | 0.281 | yes |
| standard | 159 | 29.7 | 31.1 | +1.4 | [-2.3, +5.0] | [-1.6, +4.4] | 0.500 | yes |
| contextual | 81 | 86.4 | 91.6 | +5.2 | [-0.2, +10.9] | [+0.7, +10.1] | 0.085 | no |
| copyright | 80 | 21.2 | 19.0 | -2.2 | [-6.0, +1.3] | [-5.5, +0.8] | 0.296 | no |

### Local with 2 BOS (prompts matched)

| Category | n | ASR local | ASR API | Diff (pts) | 95% CI | 90% CI | p (permutation) | Equivalent ±5? |
|---|---|---|---|---|---|---|---|---|
| ALL | 320 | 43.3 | 43.4 | +0.1 | [-2.1, +2.2] | [-1.7, +1.8] | 1.000 | yes |
| standard | 159 | 30.9 | 31.1 | +0.1 | [-2.8, +2.9] | [-2.3, +2.5] | 1.000 | yes |
| contextual | 81 | 89.6 | 91.6 | +2.0 | [-2.5, +6.7] | [-1.7, +5.9] | 0.462 | no |
| copyright | 80 | 21.0 | 19.0 | -2.0 | [-6.0, +2.0] | [-5.5, +1.5] | 0.420 | no |

Sampled text is not compared across local and API: the provider's sampler and seeds differ, so identical text is not expected.

Per-category rows are descriptive (small n, no multiplicity correction).
