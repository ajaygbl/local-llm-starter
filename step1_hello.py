"""
STEP 1 — Prove the model is running on your machine.

    python step1_hello.py

What to notice:
  - Nothing here touches the internet. Pull your network cable; it still works.
  - The first call is slow because the model is being loaded into memory.
    The second is much faster. Ollama keeps it loaded for ~5 minutes.
  - temperature=0 makes output nearly deterministic. Run it twice and compare.
"""

import time

import ollama_client as llm


def main():
    print(f"Host:  {llm.HOST}")
    print(f"Model: {llm.DEFAULT_MODEL}\n")

    llm.require(llm.DEFAULT_MODEL)

    prompt = "In one sentence: what is quantization in the context of LLMs?"
    print(f"> {prompt}\n")

    t0 = time.time()
    reply = llm.chat(
        [{"role": "user", "content": prompt}],
        temperature=0,
    )
    elapsed = time.time() - t0

    print(reply.strip())
    print(f"\n[{elapsed:.1f}s — first call includes model load time]")

    # Second call: same prompt, no load cost.
    t0 = time.time()
    llm.chat([{"role": "user", "content": prompt}], temperature=0)
    print(f"[{time.time() - t0:.1f}s — second call, model already warm]")


if __name__ == "__main__":
    main()
