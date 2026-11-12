"""Check that local prompts have the same token length as the prompts the pinned OpenRouter providers build.

Usage: python exp2/check_prompts.py [n_behaviors]   (default 6; a few cents at most)
For every (model, system prompt) used on both paths, compares len(local token ids) with the
provider-reported usage.prompt_tokens. Writes results/exp2_prompt_check.json.
"""
import json, os, sys, time
from pathlib import Path
import openai
from openai import OpenAI
from transformers import AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditions import CONDITIONS, MODELS, messages, render  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
tcs = json.load(open(PROJ / "results/test_cases/harmbench_behaviors_text_test/test_cases.json"))
ex = [v[0] for v in list(tcs.values())[::max(1, len(tcs) // N)][:N]]
cl = OpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1", max_retries=0)

def usage(model, prov, msgs):
    for i in range(6):
        try:
            return cl.chat.completions.create(model=model, messages=msgs, max_tokens=4, temperature=0,
                   extra_body={"provider": {"order": [prov], "allow_fallbacks": False}}).usage.prompt_tokens
        except openai.RateLimitError:
            time.sleep(3 * 2 ** i)

out = []
for model, system, prov in sorted({(c["model"], c["system"], c["provider"]) for c in CONDITIONS if c["path"] == "api"}):
    tok = AutoTokenizer.from_pretrained(MODELS[model]["hf"], revision=MODELS[model]["revision"])
    diffs = [usage(MODELS[model]["api"], prov, messages(system, tc))
             - len(tok(render(model, tok, messages(system, tc)), add_special_tokens=False).input_ids) for tc in ex]
    out.append(dict(model=model, system=system, provider=prov, api_minus_local=diffs))
    print(f"{model:12s} {system:9s} {prov:16s} API - local prompt tokens: {diffs}", flush=True)
json.dump(out, open(PROJ / "results/exp2_prompt_check.json", "w"), indent=2)
