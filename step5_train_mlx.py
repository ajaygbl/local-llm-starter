"""
STEP 5 (Apple Silicon) — LoRA fine-tune with MLX.

    pip install mlx-lm
    python step5_train_mlx.py

This is the step 5 for your machine. Unsloth (the usual recommendation) is
CUDA-only and will not run on an M-series Mac. MLX is Apple's own array
framework and trains on the GPU through unified memory.

WHAT 8GB MEANS HERE. macOS holds ~3GB, so you have roughly 4.5GB to play
with. That puts the ceiling at a 3-4B model quantized to 4-bit. The defaults
below are chosen to stay under it:

    --batch-size 1        bigger batches are the main cause of OOM
    --num-layers 8        only the top 8 blocks get adapters, not all of them
    --grad-checkpoint     recomputes activations instead of storing them
    4-bit base model      weights stay quantized and frozen

Expect roughly 20-40 minutes for 400 iterations on an M2. Close Chrome first,
and keep Activity Monitor open on the Memory tab -- if the pressure graph
goes yellow you are swapping to SSD and everything crawls.

IF IT STILL OOMs, in this order: --num-layers 4, then --max-seq-length 512,
then drop to Qwen3-1.7B. Do not raise the batch size to "go faster".

AFTER TRAINING, score it without any file conversion:

    python step6_evaluate.py --backend mlx --adapter-path adapters \\
        --mlx-model mlx-community/Qwen3-4B-Instruct-4bit
"""

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
ADAPTERS = os.path.join(HERE, "adapters")

# 4-bit community conversions. The 1.7B is the safe choice on 8GB; the 4B
# works with the settings below but leaves little headroom.
MODELS = {
    "4b": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "1.7b": "mlx-community/Qwen3-1.7B-4bit",
}


def check_env():
    if sys.platform != "darwin":
        sys.exit("MLX only runs on macOS. On Linux/Windows with an NVIDIA "
                 "GPU use step5_train_qlora.py instead.")
    if shutil.which("mlx_lm.lora") is None:
        try:
            import mlx_lm  # noqa: F401
        except ImportError:
            sys.exit("mlx-lm is not installed.\n  pip install mlx-lm")
    for name in ("train.jsonl", "valid.jsonl"):
        if not os.path.exists(os.path.join(DATA, name)):
            sys.exit(f"data/{name} missing. Run: python step4_make_dataset.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=list(MODELS), default="4b",
                    help="4b is the default; drop to 1.7b if you hit OOM")
    ap.add_argument("--model", help="override with any HF/MLX model id")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-layers", type=int, default=8)
    ap.add_argument("--max-seq-length", type=int, default=1024)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the command without running it")
    args = ap.parse_args()

    check_env()
    model = args.model or MODELS[args.size]

    cmd = [
        "mlx_lm.lora",
        "--model", model,
        "--train",
        "--data", DATA,
        "--adapter-path", ADAPTERS,
        "--iters", str(args.iters),
        "--batch-size", str(args.batch_size),
        "--num-layers", str(args.num_layers),
        "--max-seq-length", str(args.max_seq_length),
        "--learning-rate", str(args.learning_rate),
        # loss on the assistant's reply only, not on the ticket text.
        # For chat-format data this is what you want: you are teaching the
        # model to *answer* in a shape, not to write support tickets.
        "--mask-prompt",
        "--grad-checkpoint",
        "--steps-per-report", "20",
        "--steps-per-eval", "100",
    ]

    print("Model:   ", model)
    print("Data:    ", DATA)
    print("Adapters:", ADAPTERS)
    print("\n" + " ".join(cmd) + "\n")

    if args.dry_run:
        return

    print("Watch the two numbers that matter:")
    print("  Train loss should fall then flatten.")
    print("  Val loss should fall too. If val stops improving while train")
    print("  keeps dropping, you are overfitting -- stop and use fewer iters.")
    print("-" * 68)

    rc = subprocess.call(cmd)
    if rc != 0:
        print("\nTraining failed.", file=sys.stderr)
        print("If it was an out-of-memory error, retry with:", file=sys.stderr)
        print("  python step5_train_mlx.py --num-layers 4 "
              "--max-seq-length 512", file=sys.stderr)
        print("and if that still fails:  --size 1.7b", file=sys.stderr)
        sys.exit(rc)

    print(f"""
{'=' * 68}
Done. The adapters are in {ADAPTERS} -- a few MB, because that is all you
trained. The base model was never modified.

Try it on one ticket:

  mlx_lm.generate --model {model} \\
      --adapter-path {ADAPTERS} \\
      --prompt "Deploy for checkout-api stuck on the controller for 20 min."

Then score it against the untuned baseline on the held-out set:

  python step6_evaluate.py --backend mlx --adapter-path {ADAPTERS} \\
      --mlx-model {model}
{'=' * 68}""")


if __name__ == "__main__":
    main()
