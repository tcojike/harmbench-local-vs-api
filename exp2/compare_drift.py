"""Time drift: compare API conditions regenerated on a later day with the same exp2 conditions (labels only).

Usage: python exp2/compare_drift.py EXP2_RUN_DIR DRIFT_RUN_DIR   -> DRIFT_RUN_DIR/drift.md + drift.json
For every condition scored in DRIFT_RUN_DIR: ASR then vs now per category, paired bootstrap 95% CI and sign-flip
permutation p over behaviors (as in analyze.py), TOST at +/-5 pts, plus the share of behaviors whose 5-sample harmful
count is unchanged. Also reports prompt tokens and providers, to confirm the input and endpoint did not change.
"""
import csv, json, sys
from pathlib import Path
import numpy as np, pandas as pd

src, drift = Path(sys.argv[1]), Path(sys.argv[2])
BEH = Path.home() / "HarmBench/data/behavior_datasets/harmbench_behaviors_text_test.csv"
cat = {b["BehaviorID"]: b["FunctionalCategory"] for b in csv.DictReader(open(BEH, encoding="utf-8"))}
rng = np.random.default_rng(0); B = 10_000
P = lambda x: f"{100 * x:.1f}"; S = lambda x: f"{100 * x:+.1f}"
rate = lambda d: pd.read_json(d / "labels.jsonl", lines=True).query("label >= 0").groupby("behavior_id").label.mean()
recs = lambda d: pd.read_json(d / "records.jsonl", lines=True)


def paired(a, b):
    df = pd.DataFrame({"a": a, "b": b}).dropna(); d = (df.b - df.a).to_numpy(); n = len(d)
    boots = d[rng.integers(0, n, size=(B, n))].mean(1); perm = np.abs((rng.choice([-1, 1], size=(B, n)) * d).mean(1))
    ci90 = [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))]
    return dict(n=n, asr_then=float(df.a.mean()), asr_now=float(df.b.mean()), diff=float(d.mean()),
                ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], ci90=ci90,
                p=float((np.sum(perm >= abs(d.mean()) - 1e-12) + 1) / (B + 1)), equivalent=bool(-0.05 < ci90[0] and ci90[1] < 0.05),
                same_rate=float((df.a == df.b).mean()))


res, md = {}, ["# Time drift: exp2 API conditions regenerated on a later day\n",
               "Same pinned endpoint, model id, prompts, seeds, decoding and test cases; only the date differs.\n"]
for d in sorted(p for p in drift.iterdir() if (p / "labels.jsonl").exists()):
    then, now = src / d.name, d
    a, b = rate(then), rate(now)
    rows = {"ALL": paired(a, b), **{k: paired(a[a.index.map(cat) == k], b) for k in ("standard", "contextual", "copyright")}}
    rt, rn = recs(then), recs(now)
    mt, mn = json.load(open(then / "gen_meta.json")), json.load(open(now / "gen_meta.json"))
    info = dict(providers_then=rt.provider.value_counts().to_dict(), providers_now=rn.provider.value_counts().to_dict(),
                prompt_tokens_equal=bool((rt.groupby("behavior_id").prompt_tokens.first() == rn.groupby("behavior_id").prompt_tokens.first()).all()),
                mean_completion_tokens_then=float(rt.completion_tokens.mean()), mean_completion_tokens_now=float(rn.completion_tokens.mean()),
                errors_now=int(mn.get("errors", 0)))
    res[d.name] = dict(contrast=rows, **info)
    md += [f"## `{d.name}`\n",
           f"Providers then {info['providers_then']}, now {info['providers_now']}; per-behavior prompt tokens identical: "
           f"{'yes' if info['prompt_tokens_equal'] else '**no**'}; mean completion tokens {info['mean_completion_tokens_then']:.0f} -> "
           f"{info['mean_completion_tokens_now']:.0f}; errors now {info['errors_now']}.\n",
           "| Category | n | ASR then | ASR now | Diff (pts) | 95% CI | p (permutation) | Equivalent ±5? | Behaviors with same rate |",
           "|---|---|---|---|---|---|---|---|---|"]
    md += [f"| {k} | {r['n']} | {P(r['asr_then'])} | {P(r['asr_now'])} | {S(r['diff'])} | [{S(r['ci95'][0])}, {S(r['ci95'][1])}] | "
           f"{r['p']:.3f} | {'yes' if r['equivalent'] else 'no'} | {P(r['same_rate'])}% |" for k, r in rows.items()]
    md.append("")
md.append("Per-category rows are descriptive (small n, no multiplicity correction). \"Same rate\" = the behavior's harmful share "
          "over its 5 samples is unchanged; at T 0.7 this is not expected to be 100% even without drift.")
json.dump(res, open(drift / "drift.json", "w"), indent=2, default=str)
(drift / "drift.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
