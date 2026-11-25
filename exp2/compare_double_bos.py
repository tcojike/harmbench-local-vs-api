"""Compare the double-BOS local runs with their single-BOS exp2 counterparts (labels only, no generations printed).

Usage: python exp2/compare_double_bos.py EXP2_RUN_DIR BOS_RUN_DIR [MODEL]   -> BOS_RUN_DIR/comparison[_MODEL].md + .json
MODEL defaults to llama31_8b (output name stays comparison.md). Sections:
  * greedy (only if the model has a local greedy condition): exact McNemar, identical text
  * sampled (5 per behavior, T 0.7, same seeds): paired bootstrap CI + sign-flip permutation p over behaviors, as in analyze.py
  * sampled, local 1 BOS and local 2 BOS vs the model's bf16 API condition (a 2-BOS provider): same statistics
"""
import csv, json, sys
from math import comb
from pathlib import Path
import numpy as np, pandas as pd

src, bos = Path(sys.argv[1]), Path(sys.argv[2])
MODEL = sys.argv[3] if len(sys.argv) > 3 else "llama31_8b"
NAMES = {"llama31_8b": "Llama-3.1-8B-Instruct", "lunaris_8b": "L3-8B-Lunaris-v1"}
API_BF16 = {"llama31_8b": "CoreWeave", "lunaris_8b": "Parasail"}
SUFFIX = "" if MODEL == "llama31_8b" else f"_{MODEL}"
BEH = Path.home() / "HarmBench/data/behavior_datasets/harmbench_behaviors_text_test.csv"
cat = {b["BehaviorID"]: b["FunctionalCategory"] for b in csv.DictReader(open(BEH, encoding="utf-8"))}
rng = np.random.default_rng(0); B = 10_000
lab = lambda d: pd.read_json(d / "labels.jsonl", lines=True).query("label >= 0")
rate = lambda d: lab(d).groupby("behavior_id").label.mean()
gen = lambda d: pd.read_json(d / "records.jsonl", lines=True).set_index(["behavior_id", "sample"]).generation.str.strip()
P = lambda x: f"{100 * x:.1f}"; S = lambda x: f"{100 * x:+.1f}"


def paired(a, b):
    df = pd.DataFrame({"a": a, "b": b}).dropna(); d = (df.b - df.a).to_numpy(); n = len(d)
    boots = d[rng.integers(0, n, size=(B, n))].mean(1); perm = np.abs((rng.choice([-1, 1], size=(B, n)) * d).mean(1))
    ci90 = [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))]
    return dict(n=n, asr_a=float(df.a.mean()), asr_b=float(df.b.mean()), diff=float(d.mean()),
                ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], ci90=ci90,
                p=float((np.sum(perm >= abs(d.mean()) - 1e-12) + 1) / (B + 1)), equivalent=bool(-0.05 < ci90[0] and ci90[1] < 0.05))


def by_category(a, b):
    out = {"ALL": paired(a, b)}
    for k in ("standard", "contextual", "copyright"):
        idx = [i for i in a.index if cat.get(i) == k]
        out[k] = paired(a.loc[idx], b.reindex(idx))
    return out


def table(rows, col_a, col_b):
    md = [f"| Category | n | {col_a} | {col_b} | Diff (pts) | 95% CI | 90% CI | p (permutation) | Equivalent ±5? |",
          "|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {k} | {r['n']} | {P(r['asr_a'])} | {P(r['asr_b'])} | {S(r['diff'])} | [{S(r['ci95'][0])}, {S(r['ci95'][1])}] | "
           f"[{S(r['ci90'][0])}, {S(r['ci90'][1])}] | {r['p']:.3f} | {'yes' if r['equivalent'] else 'no'} |" for k, r in rows.items()]
    return md


md, res = [f"# Double BOS vs single BOS (local vLLM, {NAMES[MODEL]} bf16, template system prompt)\n",
           f"Only change: one extra `<|begin_of_text|>` id in front of the prompt, as {API_BF16[MODEL]} does.\n"], {}

# ---- greedy ----
g1 = src / f"{MODEL}__local-bf16__sys-template__greedy-bs256"
g2 = bos / f"{MODEL}__local-bf16__sys-template__greedy-bs256__double-bos"
if g1.exists() and g2.exists():
    x, y = rate(g1).align(rate(g2), join="inner")
    b, c = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    p = min(1.0, 2 * sum(comb(b + c, k) for k in range(min(b, c) + 1)) / 2 ** (b + c)) if b + c else 1.0
    t1, t2 = gen(g1).align(gen(g2), join="inner")
    res["greedy"] = r = dict(n=len(x), asr_single=float(x.mean()), asr_double=float(y.mean()), single_only=b, double_only=c,
                             mcnemar_p=p, identical_text=float((t1 == t2).mean()))
    md += ["## Greedy (one completion per behavior)\n",
           "| n | ASR 1 BOS | ASR 2 BOS | harmful only 1 BOS / only 2 BOS | McNemar p | Identical text |", "|---|---|---|---|---|---|",
           f"| {r['n']} | {P(r['asr_single'])} | {P(r['asr_double'])} | {b} / {c} | {p:.3f} | {P(r['identical_text'])}% |\n"]
else:
    md += [f"*No local greedy condition for {MODEL} in exp2; sampled comparison only.*\n"]

# ---- sampled: 1 BOS vs 2 BOS ----
s1 = src / f"{MODEL}__local-bf16__sys-template__sampled"
s2 = bos / f"{MODEL}__local-bf16__sys-template__sampled__double-bos"
res["sampled"] = by_category(rate(s1), rate(s2))
t1, t2 = gen(s1).align(gen(s2), join="inner")
res["sampled_identical_text"] = float((t1 == t2).mean())
md += ["## Sampled (5 completions per behavior, T 0.7, same seeds)\n"] + table(res["sampled"], "ASR 1 BOS", "ASR 2 BOS")
md.append(f"\nIdentical text with the same seed: {P(res['sampled_identical_text'])}% of sampled completions.\n")

# ---- sampled: each local arm vs the bf16 API (2-BOS provider) ----
api = src / f"{MODEL}__api-bf16__sys-template__sampled"
if api.exists():
    ra = rate(api)
    res["vs_api"] = {"local_1bos": by_category(rate(s1), ra), "local_2bos": by_category(rate(s2), ra)}
    md += [f"## Local vs API ({API_BF16[MODEL]} bf16, which feeds 2 BOS), sampled\n", "### Local with 1 BOS (the exp2 contrast)\n"]
    md += table(res["vs_api"]["local_1bos"], "ASR local", "ASR API")
    md += ["\n### Local with 2 BOS (prompts matched)\n"] + table(res["vs_api"]["local_2bos"], "ASR local", "ASR API")
    md.append("\nSampled text is not compared across local and API: the provider's sampler and seeds differ, so identical text is not expected.")
md.append("\nPer-category rows are descriptive (small n, no multiplicity correction).")
json.dump(res, open(bos / f"comparison{SUFFIX}.json", "w"), indent=2)
(bos / f"comparison{SUFFIX}.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
