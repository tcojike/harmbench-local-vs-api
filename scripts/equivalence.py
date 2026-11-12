"""Paired equivalence test (TOST) for local vs API attack success rate (ASR).

Usage: python scripts/equivalence.py [behavior_set] [--margins 0.05 0.10] [--alpha 0.05]

Each behavior is run on BOTH deployments, so the two ASRs are paired proportions. A two-sample
proportion TOST (e.g. statsmodels.stats.proportion.test_proportions_2indep) would ignore the pairing
and be badly conservative. Here:

  * difference  d = ASR_api - ASR_local = (c - b) / n,  b/c = harmful only locally / only via API
  * TOST at level alpha  <=>  the (1 - 2*alpha) CI for d lies inside (-margin, +margin)
    (Schuirmann 1987; Walker & Nowacki 2011, J Gen Intern Med 26:192-196)
  * CI: Newcombe (1998) hybrid score interval for paired proportions, method 10
    (Stat Med 17:2635-2650). Unlike the Wald interval it behaves with few discordant pairs.

Also prints the n needed for 80% power at each margin, given the observed discordance rate.
Reads results/comparison/<set>/per_behavior.csv (labels only, no generations).
Writes results/comparison/<set>/equivalence.{md,json}.
"""
import argparse, json
from math import sqrt, ceil
from pathlib import Path

import pandas as pd
from scipy.stats import norm

PROJ = Path(__file__).resolve().parent.parent


def wilson(x, n, z):
    p = x / n
    centre, half = p + z * z / (2 * n), z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - half) / (1 + z * z / n), (centre + half) / (1 + z * z / n)


def newcombe_paired(a, b, c, d, conf):
    """CI for p_api - p_local. 2x2: a = both harmful, b = local only, c = api only, d = neither."""
    n = a + b + c + d
    z = norm.ppf(1 - (1 - conf) / 2)
    p1, p2 = (a + c) / n, (a + b) / n                     # p1 = API ASR, p2 = local ASR
    l1, u1 = wilson(a + c, n, z); l2, u2 = wilson(a + b, n, z)
    num = a * d - b * c
    num = max(num - n / 2, 0) if num > 0 else num          # continuity correction from Newcombe
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = num / sqrt(den) if den else 0.0
    theta = p1 - p2
    lo = theta - sqrt(max((p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2, 0))
    hi = theta + sqrt(max((u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2, 0))
    return theta, lo, hi


def n_needed(disc_rate, margin, alpha, power=0.8):
    """Paired TOST sample size assuming the true difference is 0 (normal approximation)."""
    if disc_rate == 0:
        return None
    z = norm.ppf(1 - alpha) + norm.ppf(1 - (1 - power) / 2)
    return ceil(z * z * disc_rate / margin ** 2)


ap = argparse.ArgumentParser()
ap.add_argument("set", nargs="?", default="harmbench_behaviors_text_val")
ap.add_argument("--margins", type=float, nargs="+", default=[0.05, 0.10])
ap.add_argument("--alpha", type=float, default=0.05)
args = ap.parse_args()

OUT = PROJ / "results/comparison" / args.set
df = pd.read_csv(OUT / "per_behavior.csv")
conf = 1 - 2 * args.alpha

rows = []
for cat, g in list(df.groupby("category")) + [("ALL", df)]:
    L, A = g.label_local, g.label_api
    a, b = int(((L == 1) & (A == 1)).sum()), int(((L == 1) & (A == 0)).sum())
    c, d = int(((L == 0) & (A == 1)).sum()), int(((L == 0) & (A == 0)).sum())
    theta, lo, hi = newcombe_paired(a, b, c, d, conf)
    disc = (b + c) / len(g)
    rows.append(dict(category=cat, n=len(g), asr_local=L.mean(), asr_api=A.mean(), local_only=b, api_only=c,
                     diff=theta, ci_lo=lo, ci_hi=hi,
                     equivalent={f"{m:g}": bool(-m < lo and hi < m) for m in args.margins},
                     n_needed_80pct_power={f"{m:g}": n_needed(disc, m, args.alpha) for m in args.margins}))

pts = lambda x: f"{100 * x:+.1f}"
md = [f"# Equivalence of local vs API ASR: `{args.set}`\n",
      f"Paired TOST at alpha = {args.alpha:g}: equivalent if the {100 * conf:.0f}% Newcombe CI for "
      "ASR_api - ASR_local lies entirely inside +/- margin.\n",
      "| Category | n | Local | API | b / c | Diff (pts) | " + f"{100 * conf:.0f}% CI (pts) | "
      + " | ".join(f"Equiv. +/-{100 * m:g}" for m in args.margins) + " |",
      "|---|---|---|---|---|---|---|" + "---|" * len(args.margins)]
for r in rows:
    md.append(f"| {r['category']} | {r['n']} | {100 * r['asr_local']:.1f}% | {100 * r['asr_api']:.1f}% | "
              f"{r['local_only']} / {r['api_only']} | {pts(r['diff'])} | [{pts(r['ci_lo'])}, {pts(r['ci_hi'])}] | "
              + " | ".join("yes" if r["equivalent"][f"{m:g}"] else "no" for m in args.margins) + " |")
all_ = rows[-1]
md += ["\nb / c = harmful only locally / only via API. Per-category rows have small n; treat them as descriptive.\n",
       "Behaviors needed for 80% power (overall discordance rate, true difference 0): "
       + ", ".join(f"+/-{100 * m:g} pts -> {all_['n_needed_80pct_power'][f'{m:g}']}" for m in args.margins) + "."]
(OUT / "equivalence.md").write_text("\n".join(md) + "\n")
json.dump(dict(alpha=args.alpha, ci_method="Newcombe 1998 method 10 (paired)", rows=rows),
          open(OUT / "equivalence.json", "w"), indent=2)
print("\n".join(md))
