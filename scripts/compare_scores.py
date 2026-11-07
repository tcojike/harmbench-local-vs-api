"""Compare HarmBench results for the same model run locally vs via OpenRouter.

Usage:
  python scripts/compare_scores.py [behavior_set] [--prior-local results.json --prior-api results.json]

  behavior_set  default harmbench_behaviors_text_val (reads results/{local,api}/<set>/)
  --prior-*     optional results.json files from an earlier run, for a run-to-run reproducibility check

Writes results/comparison/<set>/{comparison.md, per_behavior.csv, summary.json}.
Generations are never printed or copied into the comparison (they can contain harmful content).
"""
import argparse, json, os
from math import comb
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
HB = Path(os.environ.get("HARMBENCH_DIR", Path.home() / "HarmBench"))
# OpenRouter list price for qwen/qwen-2.5-7b-instruct (USD per token), checked 2026-11-07.
PRICE_IN, PRICE_OUT = 0.10 / 1e6, 0.20 / 1e6

ap = argparse.ArgumentParser()
ap.add_argument("set", nargs="?", default="harmbench_behaviors_text_val")
ap.add_argument("--prior-local"); ap.add_argument("--prior-api")
a = ap.parse_args()

behaviors = pd.read_csv(HB / "data/behavior_datasets" / f"{a.set}.csv").set_index("BehaviorID")
OUT = PROJ / "results/comparison" / a.set; OUT.mkdir(parents=True, exist_ok=True)


def load(path):
    rows = {}
    for bid, lst in json.load(open(path)).items():
        g = lst[0]["generation"]
        rows[bid] = dict(label=int(lst[0]["label"]), gen=g, words=len(g.split()), err=g == "$ERROR$")
    return pd.DataFrame.from_dict(rows, orient="index")


def mcnemar(x, y):
    b, c = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    n = b + c
    p = min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0
    return b, c, p


L = load(PROJ / "results/local" / a.set / "results.json")
A = load(PROJ / "results/api" / a.set / "results.json")
d = L.join(A, lsuffix="_local", rsuffix="_api", how="inner")
d["category"] = behaviors["FunctionalCategory"].reindex(d.index).values if "FunctionalCategory" in behaviors else "standard"
meta = {k: json.load(open(PROJ / "results" / k / a.set / "run_meta.json")) for k in ("local", "api")}

# ---- token-based API cost (local tokenizer matched OpenRouter's prompt-token count exactly) ----
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tc = json.load(open(PROJ / "results/test_cases" / a.set / "test_cases.json"))
p_tok = sum(len(tok.apply_chat_template([{"role": "user", "content": t}], add_generation_prompt=True))
            for v in tc.values() for t in v)
c_tok = sum(len(tok(g)["input_ids"]) + 1 for g in d.gen_api)  # +1 for the end-of-turn token
cost = p_tok * PRICE_IN + c_tok * PRICE_OUT

# ---- tables ----
cat_rows = []
for cat, g in list(d.groupby("category")) + [("ALL", d)]:
    cat_rows.append(dict(category=cat, n=len(g), asr_local=g.label_local.mean(), asr_api=g.label_api.mean(),
                         diff_pts=100 * (g.label_api.mean() - g.label_local.mean())))
cats = pd.DataFrame(cat_rows)
b, c, p = mcnemar(d.label_local, d.label_api)
identical = int((d.gen_local.str.strip() == d.gen_api.str.strip()).sum())
same_start = float((d.gen_local.str.strip().str[:40] == d.gen_api.str.strip().str[:40]).mean())
flips = d[d.label_local != d.label_api]

repro = {}
for k, path, cur in (("local", a.prior_local, L), ("api", a.prior_api, A)):
    if path and Path(path).exists():
        P = load(path).reindex(cur.index)
        repro[k] = dict(label_agree=float((P.label == cur.label).mean()),
                        identical=int((P.gen.str.strip() == cur.gen.str.strip()).sum()),
                        asr_prior=float(P.label.mean()), asr_now=float(cur.label.mean()), n=len(cur))

summary = dict(set=a.set, n=len(d), asr_local=float(d.label_local.mean()), asr_api=float(d.label_api.mean()),
               agreement=float((d.label_local == d.label_api).mean()), local_only_harmful=b, api_only_harmful=c,
               mcnemar_p=p, identical_responses=identical, same_first_40_chars=same_start,
               api_prompt_tokens=p_tok, api_completion_tokens=c_tok, api_cost_usd=cost,
               api_errors=int(d.err_api.sum()), local_errors=int(d.err_local.sum()), reproducibility=repro,
               meta=meta, by_category=cat_rows)
json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
d.drop(columns=["gen_local", "gen_api"]).rename_axis("BehaviorID").to_csv(OUT / "per_behavior.csv")

# ---- markdown ----
pct = lambda x: f"{100 * x:.1f}%"
ml, ma = meta["local"], meta["api"]
md = [f"# HarmBench local vs OpenRouter: `{a.set}` ({len(d)} behaviors)\n",
      "## Run metrics\n",
      "| Field | Local | API |", "|---|---|---|",
      f"| Model | Qwen/Qwen2.5-7B-Instruct | qwen/qwen-2.5-7b-instruct |",
      f"| Deployment | {ml['host']} (Ubuntu, {ml['gpu']}), {ml['engine']} | OpenRouter -> {ma['provider']} |",
      f"| Behavior set | {a.set} | {a.set} |",
      "| Method / metric | DirectRequest / ASR (HarmBench-Llama-2-13b-cls) | same |",
      f"| **Score (ASR)** | **{pct(summary['asr_local'])}** ({int(d.label_local.sum())}/{len(d)}) | **{pct(summary['asr_api'])}** ({int(d.label_api.sum())}/{len(d)}) |",
      f"| Generation time | {ml['generation_seconds']} s | {ma['generation_seconds']} s |",
      f"| Scoring time | {ml['scoring_seconds']} s | {ma['scoring_seconds']} s |",
      f"| Cost | $0 (own GPU) | ${cost:.4f} ({p_tok:,} in + {c_tok:,} out tokens) |",
      f"| Errors | {summary['local_errors']} | {summary['api_errors']} |",
      f"| Precision | {ml['precision']} | {ma['precision']} |",
      f"| Sampling | greedy (temperature 0), max_new_tokens {ml['max_new_tokens']} | temperature 0, top_p 1, max_tokens {ma['max_new_tokens']} |\n",
      "## ASR by category\n", "| Category | n | Local | API | Diff (pts) |", "|---|---|---|---|---|"]
md += [f"| {r['category']} | {r['n']} | {pct(r['asr_local'])} | {pct(r['asr_api'])} | {r['diff_pts']:+.1f} |" for r in cat_rows]
md += ["\n## Paired comparison\n",
       f"- Label agreement: {pct(summary['agreement'])}",
       f"- Harmful only locally: {b}; harmful only via API: {c}",
       f"- Exact McNemar test: p = {p:.3f} ({'significant' if p < 0.05 else 'not significant'} at 0.05)",
       f"- Byte-identical responses: {identical}/{len(d)}; same first 40 characters: {pct(same_start)}"]
if len(flips):
    md += ["\n### Behaviors whose label differs\n", "| BehaviorID | Category | Local | API | Words local/API |", "|---|---|---|---|---|"]
    md += [f"| {bid} | {r['category']} | {r['label_local']} | {r['label_api']} | {r['words_local']}/{r['words_api']} |"
           for bid, r in flips.iterrows()]
if repro:
    md += ["\n## Run-to-run reproducibility (vs an earlier identical run)\n",
           "| Deployment | ASR earlier | ASR now | Label agreement | Identical responses |", "|---|---|---|---|---|"]
    md += [f"| {k} | {pct(v['asr_prior'])} | {pct(v['asr_now'])} | {pct(v['label_agree'])} | {v['identical']}/{v['n']} |"
           for k, v in repro.items()]
(OUT / "comparison.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
print(f"\nwrote {OUT}/comparison.md, per_behavior.csv, summary.json")
