"""Smoke test: Qwen3.5-9B on vLLM (new venv), thinking off, zh + en, greedy."""
import time
from vllm import LLM, SamplingParams


def main():
    t = time.time()
    llm = LLM("Qwen/Qwen3.5-9B", dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.85)
    print(f"load {time.time() - t:.0f}s")
    msgs = [[{"role": "user", "content": "用一句话介绍你自己。"}],
            [{"role": "user", "content": "Introduce yourself in one sentence."}]]
    outs = llm.chat(msgs, SamplingParams(temperature=0, max_tokens=128),
                    chat_template_kwargs={"enable_thinking": False})
    for o in outs:
        c = o.outputs[0]
        print(f"--- prompt_tokens={len(o.prompt_token_ids)} gen_tokens={len(c.token_ids)} "
              f"finish={c.finish_reason} think_tag={'<think>' in c.text}")
        print(c.text)


if __name__ == "__main__":   # required: vLLM starts its engine with the spawn method
    main()
