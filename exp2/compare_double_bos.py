"""Compare the double-BOS local runs with their single-BOS exp2 counterparts (labels only, no generations printed).

Usage: python exp2/compare_double_bos.py EXP2_RUN_DIR BOS_RUN_DIR   -> BOS_RUN_DIR/comparison.md + .json
Sampled: paired bootstrap CI + sign-flip permutation p over behaviors (as in analyze.py). Greedy: exact McNemar.
"""
import csv, json, sys
from math import comb
from pathlib import Path
import numpy as np, pandas as pd

src, bos = Path(sys.argv[1]), Path(sys.argv[2])
BEH = Path.home() / "HarmBench/data/behavior_datasets/harmbench_behaviors_text_test.csv"
cat = {b["BehaviorID"]: b["FunctionalCategory"] for b in csv.DictReader(open(BEH, encoding="utf-8"))}
rng = np.random.default_rng(0); B = 10_000
lab = lambda d: pd.read_json(d / "labels.jsonl", lines=True).query("label >= 0")
rate = lambda d: lab(d).groupby("behavior_id").label.mean()
gen = lambda d: pd.read_json(d / "records.jsonl", lines=True).set_index(["behavior_id", "sample"]).generation
md, res = ["# Double BOS vs single BOS (local vLLM, Llama-3.1-8B-Instruct bf16, template system prompt)\n",
           "Only change: one extra `<|begin_of_text|>` id in front of the prompt, as CoreWeave and Parasail do.\n"], {}

g1, g2 = src / "llama31_8b__local-bf16__sys-template__greedy-bs256", bos / "llama31_8b__local-bf16__sys-template__greedy-bs256__double-bos"
x, y = rate(g1).align(rate(g2), join="inner"); b, c = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
p = min(1.0, 2 * sum(comb(b + c, k) for k in range(min(b, c) + 1)) / 2 ** (b + c)) if b + c else 1.0
t1, t2 = gen(g1).align(gen(g2), join="inner")
res["greedy"] = dict(n=len(x), asr_single=float(x.mean()), asr_double=float(y.mean()), single_only=b, double_only=c, mcnemar_p=p,
                     identical_text=float((t1.str.strip() == t2.str.strip()).mean()))
r = res["greedy"]
md += ["## Greedy (one completion per behavior)\n", "| n | ASR 1 BOS | ASR 2 BOS | harmful only 1 BOS / only 2 BOS | McNemar p | Identical text |",
       "|---|---|---|---|---|---|",
       f"| {r['n']} | {100*r['asr_single']:.1f} | {100*r['asr_double']:.1f} | {b} / {c} | {p:.3f} | {100*r['identical_text']:.1f}% |\n"]

s1, s2 = src / "llama31_8b__local-bf16__sys-template__sampled", bos / "llama31_8b__local-bf16__sys-template__sampled__double-bos"
R = pd.DataFrame({"a": rate(s1), "b": rate(s2)}).dropna(); R["cat"] = R.index.map(cat)
def paired(df):
    d = (df.b - df.a).to_numpy(); n = len(d)
    boots = d[rng.integers(0, n, size=(B, n))].mean(1); perm = np.abs((rng.choice([-1, 1], size=(B, n)) * d).mean(1))
    return dict(n=n, asr_single=float(df.a.mean()), asr_double=float(df.b.mean()), diff=float(d.mean()),
                ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                ci90=[float(np.percentile(boots, 5)), float(np.percentile(boots, 95))],
                p=float((np.sum(perm >= abs(d.mean()) - 1e-12) + 1) / (B + 1)))
res["sampled"] = {"ALL": paired(R), **{k: paired(R[R.cat == k]) for k in ("standard", "contextual", "copyright")}}
t1, t2 = gen(s1).align(gen(s2), join="inner")
res["sampled_identical_text"] = float((t1.str.strip() == t2.str.strip()).mean())
md += ["## Sampled (5 completions per behavior, T 0.7, same seeds)\n",
       "| Category | n | ASR 1 BOS | ASR 2 BOS | Diff (pts) | 95% CI | 90% CI | p (permutation) | Equivalent ±5? |", "|---|---|---|---|---|---|---|---|---|"]
for k, r in res["sampled"].items():
    md.append(f"| {k} | {r['n']} | {100*r['asr_single']:.1f} | {100*r['asr_double']:.1f} | {100*r['diff']:+.1f} | "
              f"[{100*r['ci95'][0]:+.1f}, {100*r['ci95'][1]:+.1f}] | [{100*r['ci90'][0]:+.1f}, {100*r['ci90'][1]:+.1f}] | {r['p']:.3f} | "
              f"{'yes' if -0.05 < r['ci90'][0] and r['ci90'][1] < 0.05 else 'no'} |")
md.append(f"\nIdentical text with the same seed: {100*res['sampled_identical_text']:.1f}% of sampled completions. "
          "Per-category rows are descriptive (small n, no multiplicity correction).")
json.dump(res, open(bos / "comparison.json", "w"), indent=2)
(bos / "comparison.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
