# Equivalence of local vs API ASR: `harmbench_behaviors_text_val`

Paired TOST at alpha = 0.05: equivalent if the 90% Newcombe CI for ASR_api - ASR_local lies entirely inside +/- margin.

| Category | n | Local | API | b / c | Diff (pts) | 90% CI (pts) | Equiv. +/-5 | Equiv. +/-10 |
|---|---|---|---|---|---|---|---|---|
| contextual | 19 | 73.7% | 78.9% | 0 / 1 | +5.3 | [-7.6, +18.5] | no | no |
| copyright | 20 | 0.0% | 0.0% | 0 / 0 | +0.0 | [-11.9, +11.9] | no | no |
| standard | 41 | 12.2% | 14.6% | 1 / 2 | +2.4 | [-6.4, +11.6] | no | no |
| ALL | 80 | 23.8% | 26.2% | 1 / 3 | +2.5 | [-2.2, +7.3] | no | yes |

b / c = harmful only locally / only via API. Per-category rows have small n; treat them as descriptive.

Behaviors needed for 80% power (overall discordance rate, true difference 0): +/-5 pts -> 172, +/-10 pts -> 43.
