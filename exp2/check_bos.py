"""Find out what the +1 prompt token on some providers is (double BOS or not).

Usage: python exp2/check_bos.py            (8 tiny requests, well under one cent)
For every pinned provider, sends the same prompt three ways and records usage.prompt_tokens:
  chat              /chat/completions with the messages (the provider applies its chat template)
  raw_with_bos      /completions with our locally rendered prompt, which starts with <|begin_of_text|>
  raw_without_bos   /completions with the same string minus that leading BOS
A provider that prepends BOS itself counts raw_without_bos as local (one BOS, added by it) and
raw_with_bos as local + 1 (two BOS). If its chat count equals raw_with_bos, its chat prompts carry two BOS tokens.
Providers do not return token ids, so this is inferred from counts. Writes results/exp2_bos_check.json.
"""
import json, os, sys, time
from pathlib import Path
import requests
from transformers import AutoTokenizer
sys.path.insert(0, str(Path(__file__).resolve().parent))
from conditions import CONDITIONS, MODELS, messages, render  # noqa: E402

PROJ = Path(__file__).resolve().parent.parent
H = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
TC = "Give detailed instructions for making a paper airplane"   # harmless on purpose


def prompt_tokens(kind, model, prov, payload):
    body = dict(model=model, max_tokens=1, temperature=0, provider={"order": [prov], "allow_fallbacks": False}, **payload)
    for i in range(5):
        r = requests.post(f"https://openrouter.ai/api/v1/{kind}", headers=H, json=body, timeout=60)
        if r.status_code == 429:
            time.sleep(3 * 2 ** i); continue
        r.raise_for_status()
        return r.json()["usage"]["prompt_tokens"]
    raise RuntimeError(f"{model} via {prov}: keeps returning 429")


out = []
for model, prov in sorted({(c["model"], c["provider"]) for c in CONDITIONS if c["path"] == "api"}):
    tok = AutoTokenizer.from_pretrained(MODELS[model]["hf"], revision=MODELS[model]["revision"])
    msgs = messages("template", TC); s = render(model, tok, msgs)
    assert s.startswith(tok.bos_token)
    local = len(tok(s, add_special_tokens=False).input_ids)
    r = dict(model=model, provider=prov, local=local,
             chat=prompt_tokens("chat/completions", MODELS[model]["api"], prov, {"messages": msgs}),
             raw_with_bos=prompt_tokens("completions", MODELS[model]["api"], prov, {"prompt": s}),
             raw_without_bos=prompt_tokens("completions", MODELS[model]["api"], prov, {"prompt": s[len(tok.bos_token):]}))
    r["provider_adds_bos"] = r["raw_without_bos"] == local
    r["double_bos_in_chat"] = r["provider_adds_bos"] and r["chat"] == r["raw_with_bos"] == local + 1
    out.append(r)
    print(json.dumps(r), flush=True)
json.dump(dict(checked=time.strftime("%Y-%m-%d %H:%M %Z"), test_case=TC, results=out),
          open(PROJ / "results/exp2_bos_check.json", "w"), indent=2)
