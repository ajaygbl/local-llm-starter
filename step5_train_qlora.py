"""
STEP 5 (NVIDIA GPUs) — QLoRA fine-tune with Unsloth.

>>> ON AN APPLE SILICON MAC, USE step5_train_mlx.py INSTEAD. <<<
Unsloth is CUDA-only and will not run on an M-series chip.

This file is kept for when you have access to an NVIDIA GPU -- your own, or
a free Colab T4, which is the cheapest way to tune something bigger than
your laptop can hold.

Options, cheapest first:

  A) Google Colab free tier (T4, 16GB) -- upload data/train.jsonl, paste this
     file into a cell. Costs nothing. Start here.
  B) Your own NVIDIA GPU, 8GB+ VRAM. Install:
         pip install unsloth
  C) Rent an hour of an A100/H100 if you want to tune something larger.

Run:
    python step5_train_qlora.py
    python step5_train_qlora.py --model unsloth/Qwen3-4B-Instruct --epochs 2

WHAT IS ACTUALLY HAPPENING: the base model's weights stay frozen and
quantized to 4 bits. You train small low-rank adapter matrices alongside
them -- a fraction of a percent of the parameters. That is why this fits in
8GB when full fine-tuning the same model would need well over 100GB.

ROUGH VRAM, 4-bit QLoRA:
    3-4B  ~5GB      7-8B  ~8GB      13B  ~14GB      32B  ~24GB + offload

WHAT TO WATCH WHILE IT TRAINS:
    loss should fall and then flatten. Flat from the start = learning rate
    too low, or your data has no pattern to learn. Falling to near zero =
    you are memorising; stop earlier or add data.
"""

import argparse
import json
import os

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "data", "train.jsonl")


def parse_args():
    ap = argparse.ArgumentParser()
    # A small model is the right choice for a first run: it trains in minutes,
    # so you can iterate on the *dataset*, which is what actually matters.
    ap.add_argument("--model", default="unsloth/Qwen3-4B-Instruct-bnb-4bit")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=8)   # effective batch = 16
    return ap.parse_args()


def main():
    args = parse_args()

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise SystemExit(
            "unsloth is not installed.\n"
            "  GPU machine:  pip install unsloth\n"
            "  No GPU:       run this on Google Colab (free T4), or use\n"
            "                mlx-lm on Apple Silicon -- see the docstring.\n"
        )

    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    if not os.path.exists(DATA):
        raise SystemExit(f"{DATA} missing. Run: python step4_make_dataset.py")

    print(f"Loading {args.model} in 4-bit ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq,
        load_in_4bit=True,
    )

    # The adapters. rank 16 with alpha = 2 * rank is a sane default;
    # raise the rank only if the model is clearly failing to fit the task.
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=0,
    )

    dataset = load_dataset("json", data_files=DATA, split="train")

    def to_text(batch):
        return {"text": [
            tokenizer.apply_chat_template(m, tokenize=False)
            for m in batch["messages"]
        ]}

    dataset = dataset.map(to_text, batched=True,
                          remove_columns=dataset.column_names)
    print(f"{len(dataset)} training examples")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=args.max_seq,
            per_device_train_batch_size=args.batch,
            # Raise accumulation, not batch size, when you hit OOM.
            # Accumulation costs no extra memory.
            gradient_accumulation_steps=args.accum,
            warmup_ratio=0.05,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            logging_steps=5,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=0,
            output_dir=OUT_DIR,
            report_to="none",
        ),
    )

    stats = trainer.train()
    print(f"\nFinal loss: {stats.training_loss:.4f}")

    adapter_dir = os.path.join(OUT_DIR, "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Adapter saved to {adapter_dir} (a few MB -- that is the point)")

    # Export to GGUF so Ollama can serve it, which lets step 6 score the
    # tuned model the exact same way it scores the base model.
    gguf_dir = os.path.join(OUT_DIR, "gguf")
    print("\nExporting merged GGUF (this takes a few minutes) ...")
    try:
        model.save_pretrained_gguf(gguf_dir, tokenizer,
                                   quantization_method="q4_k_m")
    except Exception as e:                                  # noqa: BLE001
        print(f"GGUF export failed ({e}).")
        print("Not fatal -- the adapter above is still usable with "
              "transformers/peft.")
        return

    modelfile = os.path.join(OUT_DIR, "Modelfile")
    gguf = next((os.path.join(gguf_dir, f) for f in os.listdir(gguf_dir)
                 if f.endswith(".gguf")), None)
    with open(modelfile, "w") as f:
        f.write(f"FROM {gguf}\n")

    print(f"""
Done. To serve it with Ollama and score it:

    ollama create meridian-triage -f {modelfile}
    python step6_evaluate.py --model meridian-triage --baseline {json.dumps(os.environ.get('LLM_MODEL', 'qwen3:8b'))}
""")


if __name__ == "__main__":
    main()
