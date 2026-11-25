# One-off drivers used to finish run `exp2-20261111-220502` (2026-11-12)

Kept for the record; they hard-code that run directory. For a new run, use `exp2/run_exp2.sh` (it resumes an existing
run directory and skips finished conditions).

| Script | What it did |
|---|---|
| `exp2_score.sh` | Scored the 16 conditions that finished on Nov 11 (4 GPUs) and ran `analyze.py`. Classification never ran on Nov 11 because the driver waits for all lanes, and one lane had died |
| `exp2_fp8_rerun.sh` | Regenerated and scored `llama31_8b__api-fp8__sys-template__sampled` (DeepInfra fp8). The original lane was throttled by HTTP 429 and killed before writing anything |
| `exp2_fp8_safety.sh PID` | Generated and scored `llama31_8b__api-fp8__sys-safety__sampled`, which had never started. Waits for PID (the template rerun) so the two do not share DeepInfra's rate limit |
| `exp2_double_bos_lunaris.sh` | (Nov 25) Same double-BOS test for Lunaris (sampled only; exp2 has no local greedy Lunaris), plus local 1/2 BOS vs Parasail -> `comparison_lunaris_8b.md` |
| `probe_copyright.sh [OUT]` | (Nov 25) DeepInfra copyright probe (`probe_copyright_sampling.py`): 80 copyright behaviors, greedy, DeepInfra fp8 default / all penalties off / `repetition_penalty` 1.1, CoreWeave reference; then scoring |
| `drift.sh OUT COND [WAIT_PID]` | (Nov 25) Time drift: regenerate one exp2 API condition unchanged on a later day and score it; compare with `compare_drift.py` |
| `exp2_double_bos.sh` | Local Llama runs with one extra BOS id (`check_double_bos_local.py`), scoring, and `compare_double_bos.py` -> `results/exp2/double_bos/` |

All were launched detached (`setsid nohup bash <script> > runs/<name>.out 2>&1 < /dev/null &`) so a closed terminal
does not kill them. The final analysis was then run with:

```bash
HF_HUB_OFFLINE=1 ~/venv-harmbench/bin/python exp2/analyze.py runs/exp2-20261111-220502 \
    ~/HarmBench/data/behavior_datasets/harmbench_behaviors_text_test.csv results/exp2/exp2-20261111-220502
```
