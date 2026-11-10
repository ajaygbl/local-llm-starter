"""
STEP 4 — Build a fine-tuning dataset.

    python step4_make_dataset.py
    python step4_make_dataset.py --n 1200 --seed 7

Produces (the folder layout mlx_lm.lora expects):
    data/train.jsonl   80%  -- what the model learns from
    data/valid.jsonl   10%  -- validation loss during training
    data/test.jsonl    10%  -- held out, never seen until step 6
    data/eval.jsonl         -- identical to test.jsonl, the name step 6 uses

THE TASK: turn a free-text support ticket into a strict JSON triage record.
This is deliberately a *behaviour* task, not a knowledge task. Fine-tuning is
good at "always respond in exactly this shape"; it is bad at "know these
facts" (that was step 3).

ABOUT SYNTHETIC DATA: this generator uses templates so you have something to
run today. Real fine-tunes use real data. The rule that matters is the same
either way -- 500 clean, consistent examples beat 5,000 noisy ones. Every
label here follows the same rules with no contradictions, which is exactly
what makes a dataset trainable.

Format is "messages" (chat) style, which Unsloth, Axolotl, TRL and
LLaMA-Factory all accept directly.
"""

import argparse
import json
import os
import random

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

SYSTEM = (
    "You are a ticket triage system for the Meridian platform. "
    "Reply with a single JSON object and nothing else, using exactly these "
    'keys: "category", "severity", "component", "needs_human".'
)

# (category, component, severity, [templates])
PATTERNS = [
    ("deploy", "controller", "medium", [
        "Deploy for {stack} has been sitting at 'waiting for controller' for {n} minutes.",
        "My {stack} deploy isn't progressing. Stuck on the controller step.",
        "{stack} deploy hangs. No error, just no movement for {n} min.",
        "Is the controller backed up? {stack} won't finish deploying.",
    ]),
    ("deploy", "manifest", "low", [
        "Getting exit code 2 when I deploy {stack}. Manifest problem?",
        "meridian deploy --stack {stack} fails validation immediately.",
        "What does exit code 2 mean? Happens every time on {stack}.",
        "Deploy rejected before it starts, says something about the manifest for {stack}.",
    ]),
    ("quota", "quota", "medium", [
        "Exit code 3 deploying {stack}. I think we're out of quota in {lane}.",
        "Can someone bump our CPU quota in {lane}? {stack} won't deploy.",
        "{stack} deploy failing with code 3. quota show says we're at the limit.",
        "We need more memory budget in {lane} for {stack}.",
    ]),
    ("auth", "auth", "low", [
        "meridian auth keeps logging me out, way more often than 8 hours.",
        "Getting exit code 4 on every command. Token expired again?",
        "I re-ran auth login but {stack} commands still say authentication failed.",
        "Constant re-auth prompts since this morning on my laptop.",
    ]),
    ("secrets", "secrets", "medium", [
        "Added a secret for {stack} but the container reads it as empty.",
        "{stack} can't see the new API key I set with secrets set.",
        "Secret is set according to the CLI, but the running pod has nothing.",
        "Env var missing inside {stack} even though I added it an hour ago.",
    ]),
    ("rollback", "envelope", "high", [
        "Rolled {stack} back in {lane} and the errors are still happening.",
        "Rollback didn't fix {stack}. Users still seeing 500s.",
        "{stack} rolled back to a known-good Envelope but prod is still broken.",
        "Rollback completed for {stack} but no change in behaviour. Prod impact.",
    ]),
    ("rollback", "envelope", "low", [
        "How do I find the Envelope ID to roll {stack} back to?",
        "What's the command to roll back {stack} one version?",
        "Does rolling back {stack} rebuild anything or is it instant?",
    ]),
    ("deploy", "promotion", "medium", [
        "Can't promote {stack} from staging to prod, says it needs a second approver.",
        "Promotion of {stack} to prod is blocked. Who can approve?",
        "{stack} promote --to prod rejected. Is low_risk set on our manifest?",
    ]),
    ("other", "console", "low", [
        "Meridian Console won't load for me, just a blank page.",
        "Console shows stale data for {stack}, CLI shows the right thing.",
        "Is console.meridian.internal down? Spinner forever.",
    ]),
    ("other", "docs", "low", [
        "Where is the documentation for the --json flag?",
        "Is there a reference for all the exit codes?",
        "New on the team -- what's the difference between a Lane and an Envelope?",
    ]),
    ("deploy", "controller", "high", [
        "{lane} is down. Every Stack including {stack} is failing to deploy. Customer impact.",
        "Total outage in {lane}. Controller unresponsive, {stack} and others affected.",
        "URGENT: {stack} is serving errors in prod and we cannot deploy a fix.",
    ]),
]

STACKS = ["checkout-api", "billing-worker", "search-indexer", "auth-gateway",
          "notify-svc", "ledger-api", "report-batch", "media-resizer",
          "pricing-engine", "session-store"]
LANES = ["dev", "staging", "prod"]

PREFIXES = ["", "Hi team -- ", "Hey, ", "Quick one: ", "Sorry to bother, ",
            "Raising this here: "]
SUFFIXES = ["", " Thanks!", " Any ideas?", " Let me know.", " Happy to pair on it.",
            " Not urgent.", " Please advise."]


def needs_human(category, severity):
    """One consistent rule. Consistency is what makes a label learnable."""
    return severity == "high" or category in ("quota", "secrets")


def make_example(rng):
    category, component, severity, templates = rng.choice(PATTERNS)
    text = rng.choice(templates).format(
        stack=rng.choice(STACKS),
        lane=rng.choice(LANES),
        n=rng.choice([3, 5, 8, 10, 12, 20, 45]),
    )
    text = rng.choice(PREFIXES) + text + rng.choice(SUFFIXES)

    answer = {
        "category": category,
        "severity": severity,
        "component": component,
        "needs_human": needs_human(category, severity),
    }
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text},
            # separators=... keeps the target format byte-identical every time.
            # Inconsistent target formatting is a common, silent cause of a
            # fine-tune that never quite learns the shape.
            {"role": "assistant",
             "content": json.dumps(answer, separators=(", ", ": "))},
        ]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    seen = set()
    rows = []
    attempts = 0
    while len(rows) < args.n and attempts < args.n * 50:
        attempts += 1
        ex = make_example(rng)
        key = ex["messages"][1]["content"]
        if key in seen:          # exact duplicates teach nothing
            continue
        seen.add(key)
        rows.append(ex)

    rng.shuffle(rows)
    n_train = int(len(rows) * 0.8)
    n_valid = int(len(rows) * 0.1)
    train = rows[:n_train]
    valid = rows[n_train:n_train + n_valid]
    test = rows[n_train + n_valid:]

    os.makedirs(OUT_DIR, exist_ok=True)
    # eval.jsonl is a copy of test.jsonl: mlx_lm wants "test", step 6 says
    # "eval". Same rows either way -- the held-out set is the held-out set.
    for name, data in (("train", train), ("valid", valid),
                       ("test", test), ("eval", test)):
        path = os.path.join(OUT_DIR, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")
        print(f"{path}: {len(data)} examples")

    counts = {}
    for row in rows:
        c = json.loads(row["messages"][2]["content"])["category"]
        counts[c] = counts.get(c, 0) + 1
    print("\nCategory balance:")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:10s} {v:4d}  ({v / len(rows):.0%})")

    print("\nOne example:")
    print(json.dumps(rows[0], indent=2)[:600])


if __name__ == "__main__":
    main()
