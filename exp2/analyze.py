"""Analyze an exp2 run: ASR per condition and the pre-registered paired contrasts.

Usage: python exp2/analyze.py RUN_DIR BEHAVIORS_CSV OUT_DIR

Unit of analysis = behavior (320). For each behavior and condition, rate = share of its sampled
completions judged harmful. A contrast B - A is the mean over behaviors of (rate_B - rate_A):
  * 95% CI: percentile bootstrap over behaviors (10,000 resamples)
  * p-value: paired sign-flip permutation test (10,000 permutations), Holm-corrected over the family
  * equivalence (TOST, alpha 0.05): the 90% bootstrap CI lies inside +/- EQUIV_MARGIN
Greedy contrasts (one completion per behavior) use the exact McNemar test instead.
Writes OUT_DIR/{report.md, summary.json, per_behavior_rates.csv}; never copies generations.
"""
import csv, json, sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditions import CONDITIONS, CONTRASTS, GREEDY_CONTRASTS, EQUIV_MARGIN  # noqa: E402

run, beh_csv, OUT = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
cat = {b["BehaviorID"]: b["FunctionalCategory"] for b in csv.DictReader(open(beh_csv, encoding="utf-8"))}
rng = np.random.default_rng(0)
B = 10_000

# ---------- load ----------
labels, records, meta = {}, {}, {}
for c in CONDITIONS:
    d = run / c["name"]
    if not (d / "labels.jsonl").exists():
        continue
    L = pd.read_json(d / "labels.jsonl", lines=True)
    labels[c["name"]] = L[L.label >= 0]
    records[c["name"]] = pd.read_json(d / "records.jsonl", lines=True)
    meta[c["name"]] = json.load(open(d / "gen_meta.json"))
cond = {c["name"]: c for c in CONDITIONS if c["name"] in labels}
rates = pd.DataFrame({n: L.groupby("behavior_id").label.mean() for n, L in labels.items()})
rates["category"] = rates.index.map(cat)
rates.rename_axis("BehaviorID").to_csv(OUT / "per_behavior_rates.csv", float_format="%.3f")

# ---------- per-condition table ----------
rows = []
for n, c in cond.items():
    R, M = records[n], meta[n]
    ok = R[R.generation != "$ERROR$"]
    row = dict(condition=n, model=c["model"], path=c["path"], precision=c["precision"], system=c["system"],
               decoding=c["decoding"], n_completions=len(labels[n]), errors=int(M.get("errors", 0)),
               asr=float(rates[n].mean()),
               **{f"asr_{k}": float(rates.loc[rates.category == k, n].mean()) for k in ("standard", "contextual", "copyright")},
               truncated=float((ok.finish_reason == "length").mean()), mean_completion_tokens=float(ok.completion_tokens.mean()),
               mean_prompt_tokens=float(ok.prompt_tokens.mean()),
               providers=dict(ok.provider.value_counts()) if "provider" in ok else {})
    rows.append(row)
cond_df = pd.DataFrame(rows)

# ---------- prompt check: does the API see the same prompt length as the local run? ----------
prompt_check = []
for n, c in cond.items():
    if c["path"] != "api":
        continue
    loc = next((m for m, x in cond.items() if x["path"] == "local" and x["model"] == c["model"] and x["system"] == c["system"]
                and x["precision"] == "bf16"), None)
    if loc is None:
        continue
    a = records[n].groupby("behavior_id").prompt_tokens.first()
    b = records[loc].groupby("behavior_id").prompt_tokens.first()
    diff = (a - b).dropna()
    prompt_check.append(dict(api=n, local=loc, median_diff=float(diff.median()), min_diff=int(diff.min()),
                             max_diff=int(diff.max()), share_equal=float((diff == 0).mean())))


# ---------- contrasts ----------
def paired(a, b):
    d = (b - a).dropna().to_numpy(); n = len(d)
    idx = rng.integers(0, n, size=(B, n)); boots = d[idx].mean(1)
    signs = rng.choice([-1, 1], size=(B, n)); perm = np.abs((signs * d).mean(1))
    return dict(n=n, diff=float(d.mean()), ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                ci90=[float(np.percentile(boots, 5)), float(np.percentile(boots, 95))],
                p=float((1 + (perm >= abs(d.mean()) - 1e-12).sum()) / (B + 1)))


res = []
for q, A, Bn in CONTRASTS:
    if A not in rates or Bn not in rates:
        continue
    r = dict(question=q, A=A, B=Bn, asr_A=float(rates[A].mean()), asr_B=float(rates[Bn].mean()), **paired(rates[A], rates[Bn]))
    r["by_category"] = {k: paired(rates.loc[rates.category == k, A], rates.loc[rates.category == k, Bn]) for k in ("standard", "contextual", "copyright")}
    r["equivalent"] = bool(-EQUIV_MARGIN < r["ci90"][0] and r["ci90"][1] < EQUIV_MARGIN)
    res.append(r)
# Holm correction over the pre-registered family
order = np.argsort([r["p"] for r in res]); m = len(res); running = 0.0
for rank, i in enumerate(order):
    running = max(running, min(1.0, (m - rank) * res[i]["p"])); res[i]["p_holm"] = running


def mcnemar(x, y):
    b, c = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum()); n = b + c
    return b, c, (min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0)


greedy = []
for q, A, Bn in GREEDY_CONTRASTS:
    if A not in rates or Bn not in rates:
        continue
    x, y = rates[A].dropna(), rates[Bn].dropna(); x, y = x.align(y, join="inner")
    b, c, p = mcnemar(x, y)
    ga = records[A].set_index("behavior_id").generation; gb = records[Bn].set_index("behavior_id").generation
    ga, gb = ga.align(gb, join="inner")
    greedy.append(dict(question=q, A=A, B=Bn, n=len(x), asr_A=float(x.mean()), asr_B=float(y.mean()),
                       label_agreement=float((x == y).mean()), A_only=b, B_only=c, mcnemar_p=p,
                       identical=float((ga.str.strip() == gb.str.strip()).mean()),
                       same_first_40=float((ga.str.strip().str[:40] == gb.str.strip().str[:40]).mean())))

json.dump(dict(conditions=rows, contrasts=res, greedy=greedy, prompt_check=prompt_check, equiv_margin=EQUIV_MARGIN),
          open(OUT / "summary.json", "w"), indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

missing = [c["name"] for c in CONDITIONS if c["name"] not in labels]
skipped = [q for q, A, Bn in list(CONTRASTS) + list(GREEDY_CONTRASTS) if A not in rates or Bn not in rates]
notes = (run / "notes.md").read_text().strip() if (run / "notes.md").exists() else ""
bos_file = Path(__file__).resolve().parent.parent / "results/exp2_bos_check.json"
bos = {(b["model"], b["provider"]): b for b in json.load(open(bos_file))["results"]} if bos_file.exists() else {}

# ---------- report ----------
P = lambda x: f"{100 * x:.1f}"
S = lambda x: f"{100 * x:+.1f}"
md = ["# Experiment 2 results\n", f"Run: `{run.name}`. 320 HarmBench text test behaviors, DirectRequest. "
      "Sampled conditions: 5 completions per behavior at temperature 0.7; ASR = mean over behaviors of the "
      "share of harmful completions (HarmBench-Llama-2-13b-cls; hash check for copyright).\n",
      *([f"**Incomplete run:** {len(missing)} of {len(CONDITIONS)} planned conditions have no scored completions "
         f"({', '.join(f'`{m}`' for m in missing)}); {len(skipped)} contrast(s) could not be computed: "
         + "; ".join(skipped) + ".\n"] if missing else []),
      *(["## Run notes\n", notes + "\n"] if notes else []),
      "## ASR by condition\n",
      "| Condition | Path | Precision | System | Decoding | n | Errors | ASR | Standard | Contextual | Copyright | Truncated | Providers |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    dec_label = ("sampled" if r["decoding"] == "sampled" else
                 f"greedy bs{cond[r['condition']]['batch']}" if r["path"] == "local" else "greedy (T = 0)")
    md.append(f"| {r['model']} | {r['path']} | {r['precision']} | {r['system']} | {dec_label} "
              f"| {r['n_completions']} | {r['errors']} | **{P(r['asr'])}** | {P(r['asr_standard'])} | {P(r['asr_contextual'])} | {P(r['asr_copyright'])} "
              f"| {P(r['truncated'])}% | {', '.join(f'{k} {v}' for k, v in r['providers'].items())} |")
md += ["\n## Pre-registered contrasts (B − A, percentage points)\n",
       f"Equivalent = 90% CI inside ±{100 * EQUIV_MARGIN:g} pts (TOST, α = 0.05). p = paired sign-flip permutation test, Holm-corrected over {len(res)} contrasts.\n",
       "| Question | ASR A | ASR B | Diff | 95% CI | p (Holm) | Equivalent ±5? | Standard | Contextual | Copyright |",
       "|---|---|---|---|---|---|---|---|---|---|"]
for r in res:
    bc = r["by_category"]
    md.append(f"| {r['question']} | {P(r['asr_A'])} | {P(r['asr_B'])} | **{S(r['diff'])}** | [{S(r['ci95'][0])}, {S(r['ci95'][1])}] "
              f"| {r['p_holm']:.4f} | {'yes' if r['equivalent'] else 'no'} | "
              + " | ".join(f"{S(bc[k]['diff'])} [{S(bc[k]['ci95'][0])}, {S(bc[k]['ci95'][1])}]" for k in ("standard", "contextual", "copyright")) + " |")
if greedy:
    md += ["\n## Greedy decoding (one completion per behavior)\n",
           "| Question | ASR A | ASR B | Label agreement | A-only / B-only | McNemar p | Identical text | Same first 40 chars |",
           "|---|---|---|---|---|---|---|---|"]
    md += [f"| {g['question']} | {P(g['asr_A'])} | {P(g['asr_B'])} | {P(g['label_agreement'])}% | {g['A_only']} / {g['B_only']} | {g['mcnemar_p']:.3f} | {P(g['identical'])}% | {P(g['same_first_40'])}% |" for g in greedy]
if prompt_check:
    def bos_col(api):
        b = bos.get((cond[api]["model"], cond[api]["provider"]))
        return "not checked" if b is None else ("**2 BOS** (provider adds one)" if b["double_bos_in_chat"] else "1 BOS (same as local)")
    md += ["\n## Prompt check (API prompt tokens − local prompt tokens, per behavior)\n",
           "| API condition | Provider | Median diff | Range | Share identical length | BOS tokens in chat prompt |", "|---|---|---|---|---|---|"]
    md += [f"| {p['api']} | {cond[p['api']]['provider']} | {p['median_diff']:+.0f} | [{p['min_diff']:+d}, {p['max_diff']:+d}] | {P(p['share_equal'])}% | {bos_col(p['api'])} |"
           for p in prompt_check]
    if bos:
        md += ["", "The constant +1 is a second BOS token, not a different prompt: `exp2/check_bos.py` "
               "(`results/exp2_bos_check.json`) sends the same prompt raw through `/completions`. Providers marked "
               "\"2 BOS\" count a BOS-less raw prompt as local length (they prepend BOS themselves) and our rendered "
               "prompt, which already starts with `<|begin_of_text|>`, as local + 1; their chat count equals the latter. "
               "So their models see `<|begin_of_text|><|begin_of_text|>…` while local vLLM and DeepInfra see one. "
               "Inferred from token counts: providers do not return token ids, so beyond the extra BOS the check shows equal lengths, not identical tokens."]
(OUT / "report.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
