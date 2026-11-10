"""
STEP 6 — Score a model on the held-out set. Do this BEFORE and AFTER training.

TWO BACKENDS:

  ollama (default) -- score any model Ollama is serving.

      # baseline, before you train anything:
      python step6_evaluate.py --model qwen3:4b

      # try constrained decoding -- often beats a fine-tune, for free:
      python step6_evaluate.py --model qwen3:4b --force-json

  mlx -- score an MLX adapter straight from step5_train_mlx.py, with no
  GGUF conversion. It runs the base model twice, once with the adapter and
  once without, so the comparison is exactly like-for-like:

      python step6_evaluate.py --backend mlx --adapter-path adapters \\
          --mlx-model mlx-community/Qwen3-4B-Instruct-2507-4bit

THIS IS THE STEP PEOPLE SKIP, AND IT IS THE ONE THAT MATTERS. A fine-tune
that does not move your target metric failed, no matter how pretty the
training loss curve looked. Run the baseline first so you have a number to
beat -- sometimes the honest finding is that prompting already got you there
and the fine-tune was unnecessary.

Metrics:
  valid_json      did it emit parseable JSON at all (the format win)
  schema_ok       exactly the right keys, no extras
  exact_match     every field correct
  per-field       which field it gets wrong -- usually more useful than the total
"""

import argparse
import json
import os
import sys

import ollama_client as llm

EVAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "data", "eval.jsonl")
KEYS = ["category", "severity", "component", "needs_human"]


def load(path):
    if not os.path.exists(path):
        sys.exit(f"{path} missing. Run: python step4_make_dataset.py")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ------------------------------------------------------------- backends
# A backend is just: messages -> reply string. Keeping it this small is what
# lets the same scoring code compare an Ollama model against an MLX adapter.

def ollama_backend(model, force_json=False):
    def run(messages):
        return llm.chat(messages, model=model, temperature=0,
                        fmt="json" if force_json else None)
    return run


def mlx_backend(model_id, adapter_path=None, max_tokens=120):
    """Load an MLX model once, optionally with LoRA adapters attached."""
    try:
        from mlx_lm import load as mlx_load, generate as mlx_generate
    except ImportError:
        sys.exit("mlx-lm is not installed.\n  pip install mlx-lm")

    kwargs = {"adapter_path": adapter_path} if adapter_path else {}
    label = f"{model_id}{' +adapter' if adapter_path else ' (base)'}"
    print(f"  loading {label} ...")
    model, tokenizer = mlx_load(model_id, **kwargs)

    def run(messages):
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        # mlx_lm defaults to greedy decoding, which is what we want here:
        # a score you cannot reproduce is not a score.
        return mlx_generate(model, tokenizer, prompt=prompt,
                            max_tokens=max_tokens, verbose=False)

    return run


def extract_json(text):
    """Models wrap JSON in prose and code fences. Be generous when parsing;
    be strict when scoring."""
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def evaluate(label, run, rows, verbose=False):
    n = len(rows)
    valid = schema = exact = 0
    field_ok = {k: 0 for k in KEYS}
    failures = []

    for i, row in enumerate(rows, 1):
        msgs = row["messages"][:2]           # system + user, not the answer
        gold = json.loads(row["messages"][2]["content"])

        try:
            raw = run(msgs)
        except llm.OllamaError as e:
            sys.exit(str(e))

        got = extract_json(raw)
        if got is None:
            failures.append((row["messages"][1]["content"], raw[:120], None))
            print(f"\r  {i}/{n}", end="", flush=True)
            continue
        valid += 1

        if set(got.keys()) == set(KEYS):
            schema += 1

        correct = 0
        for k in KEYS:
            if k in got and got[k] == gold[k]:
                field_ok[k] += 1
                correct += 1
        if correct == len(KEYS) and set(got.keys()) == set(KEYS):
            exact += 1
        elif len(failures) < 5:
            failures.append((row["messages"][1]["content"], got, gold))

        print(f"\r  {i}/{n}", end="", flush=True)

    print("\r" + " " * 20 + "\r", end="")

    result = {
        "model": label,
        "n": n,
        "valid_json": valid / n,
        "schema_ok": schema / n,
        "exact_match": exact / n,
        "fields": {k: v / n for k, v in field_ok.items()},
    }
    if verbose and failures:
        print(f"\n  first failures for {label}:")
        for ticket, got, gold in failures[:3]:
            print(f"    ticket: {ticket[:70]}")
            print(f"    got:    {got}")
            print(f"    gold:   {gold}\n")
    return result


def show(r):
    print(f"\n{r['model']}  (n={r['n']})")
    print(f"  valid JSON     {r['valid_json']:6.1%}")
    print(f"  schema exact   {r['schema_ok']:6.1%}")
    print(f"  all fields     {r['exact_match']:6.1%}   <-- headline number")
    for k, v in r["fields"].items():
        print(f"    {k:14s} {v:6.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ollama", "mlx"], default="ollama")
    ap.add_argument("--model", default=llm.DEFAULT_MODEL,
                    help="ollama backend: model to score")
    ap.add_argument("--baseline", help="ollama backend: model to compare against")
    ap.add_argument("--mlx-model", default="mlx-community/Qwen3-4B-Instruct-2507-4bit",
                    help="mlx backend: base model id")
    ap.add_argument("--adapter-path", default="adapters",
                    help="mlx backend: adapters from step5_train_mlx.py")
    ap.add_argument("--limit", type=int, default=0, help="score fewer rows")
    ap.add_argument("--force-json", action="store_true",
                    help="ollama backend: JSON-constrained decoding")
    ap.add_argument("--verbose", action="store_true", help="show failures")
    args = ap.parse_args()

    rows = load(EVAL)
    if args.limit:
        rows = rows[:args.limit]

    print(f"Scoring {len(rows)} held-out tickets.\n")

    results = []
    if args.backend == "mlx":
        if not os.path.isdir(args.adapter_path):
            sys.exit(f"{args.adapter_path}/ not found. "
                     "Run: python step5_train_mlx.py")
        # Same base model both times -- the adapter is the only variable.
        for label, adapter in (("base (no adapter)", None),
                               ("fine-tuned", args.adapter_path)):
            run = mlx_backend(args.mlx_model, adapter)
            results.append(evaluate(label, run, rows, args.verbose))
    else:
        for m in filter(None, [args.baseline, args.model]):
            label = m + (" +json-mode" if args.force_json else "")
            print(f"{label} ...")
            run = ollama_backend(m, args.force_json)
            results.append(evaluate(label, run, rows, args.verbose))

    for r in results:
        show(r)

    if len(results) == 2:
        base, tuned = results
        delta = tuned["exact_match"] - base["exact_match"]
        print(f"\n{'=' * 50}")
        print(f"exact_match: {base['exact_match']:.1%} -> "
              f"{tuned['exact_match']:.1%}  ({delta:+.1%})")
        if delta <= 0.02:
            print("No real gain. Before training longer, check: is the "
                  "dataset consistent? Did prompting alone already work?")
        print("=" * 50)


if __name__ == "__main__":
    main()
