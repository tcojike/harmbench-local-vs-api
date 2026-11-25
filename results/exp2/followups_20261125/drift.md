# Time drift: exp2 API conditions regenerated on a later day

Same pinned endpoint, model id, prompts, seeds, decoding and test cases; only the date differs.

## `llama31_8b__api-bf16__sys-template__sampled`

Providers then {'CoreWeave': 1600}, now {'CoreWeave': 1600}; per-behavior prompt tokens identical: yes; mean completion tokens 134 -> 134; errors now 0.

| Category | n | ASR then | ASR now | Diff (pts) | 95% CI | p (permutation) | Equivalent ±5? | Behaviors with same rate |
|---|---|---|---|---|---|---|---|---|
| ALL | 320 | 8.2 | 8.2 | +0.0 | [+0.0, +0.0] | 1.000 | yes | 100.0% |
| standard | 159 | 2.1 | 2.1 | +0.0 | [+0.0, +0.0] | 1.000 | yes | 100.0% |
| contextual | 81 | 9.6 | 9.6 | +0.0 | [+0.0, +0.0] | 1.000 | yes | 100.0% |
| copyright | 80 | 19.0 | 19.0 | +0.0 | [+0.0, +0.0] | 1.000 | yes | 100.0% |

## `llama31_8b__api-fp8__sys-template__sampled`

Providers then {'DeepInfra': 1600}, now {'DeepInfra': 1600}; per-behavior prompt tokens identical: yes; mean completion tokens 153 -> 153; errors now 0.

| Category | n | ASR then | ASR now | Diff (pts) | 95% CI | p (permutation) | Equivalent ±5? | Behaviors with same rate |
|---|---|---|---|---|---|---|---|---|
| ALL | 320 | 7.3 | 7.1 | -0.3 | [-1.1, +0.6] | 0.662 | yes | 91.2% |
| standard | 159 | 4.0 | 3.9 | -0.1 | [-0.6, +0.3] | 1.000 | yes | 98.1% |
| contextual | 81 | 15.1 | 15.1 | -0.0 | [-3.0, +3.0] | 1.000 | yes | 74.1% |
| copyright | 80 | 6.0 | 5.3 | -0.8 | [-2.2, +0.2] | 0.504 | yes | 95.0% |

Per-category rows are descriptive (small n, no multiplicity correction). "Same rate" = the behavior's harmful share over its 5 samples is unchanged; at T 0.7 this is not expected to be 100% even without drift.
