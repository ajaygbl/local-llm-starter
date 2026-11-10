# local-llm-starter

Six small scripts that take you from "I have never run a model locally" to
"I fine-tuned one and measured whether it helped."

**Pinned for: MacBook Air M2, 8 GB unified memory, macOS Tahoe 26.6.2.**
Every default in here is chosen for that machine.

---

## What 8 GB actually gives you

On Apple Silicon the GPU shares memory with everything else. macOS holds
around 3 GB, so your real budget for a model plus its KV cache is roughly
**4.5 GB**. That is the number that decides everything below.

| Model | Size on disk | On your Mac |
|---|---|---|
| `qwen3:1.7b` | ~1.4 GB | Fast. Good for steps 1-2. |
| `qwen3:4b` | ~2.6 GB | **The default here.** Best quality that fits comfortably. |
| `qwen3:8b` | ~5.2 GB | Will swap to SSD. Avoid on this machine. |

Keep Activity Monitor open on the **Memory** tab while you work. If the
pressure graph turns yellow you are swapping, and everything will feel
broken when it is really just full.

The 378 GB of free disk you have is plenty — models are the only large
thing here, and you will download maybe 5 GB total.

---

## Setup (5 minutes)

```bash
# 1. Ollama
brew install ollama
brew services start ollama        # or just run: ollama serve

# 2. Models
ollama pull qwen3:4b              # chat
ollama pull nomic-embed-text      # embeddings, ~274 MB

# 3. Python deps
pip install -r requirements.txt

# 4. Sanity check that needs no model at all
python test_chunking.py
```

If 4b feels sluggish, drop down — nothing else needs changing:

```bash
export LLM_MODEL=qwen3:1.7b
```

---

## The six steps

### 1. Hello
```bash
python step1_hello.py
```
Proves the model is running locally. The gap between the first and second
call is model load time, not inference.

### 2. Chat
```bash
python step2_chat.py
```
Streaming chat loop. Type `/tokens` to watch context fill up. The lesson:
"conversation memory" is a Python list you resend every turn. The model
remembers nothing between calls.

### 3. RAG — the knowledge problem
```bash
python step3_rag.py ingest
python step3_rag.py compare "how do I roll back a deploy in Meridian?"
```

`docs/` describes a deployment platform called Meridian that does not exist.
`compare` asks the same question twice — once with the model on its own, once
with the relevant chunks retrieved into the prompt. The first answer is a
confident invention. The second is right.

```bash
python step3_rag.py ask "why is my secret empty inside the container?" --show-chunks
```

`--show-chunks` shows what retrieval actually found. When an answer is wrong,
look there first — it is usually retrieval, not the model.

No vector database here, just a list of vectors and a dot product in ~40
lines. Chroma/FAISS/pgvector add scale and filtering on top of exactly this.

### 4. Build a training set
```bash
python step4_make_dataset.py
```
Writes `data/` with 640 train / 80 valid / 80 held-out test.

The task: turn a free-text support ticket into a strict JSON triage record.
That is a **behaviour** task — "always answer in exactly this shape" — which
is what fine-tuning is good at. Step 3 already handled knowledge.

### 5. Fine-tune — use the MLX script
```bash
pip install mlx-lm
python step5_train_mlx.py --dry-run     # see the command first
python step5_train_mlx.py
```

**`step5_train_qlora.py` will not run on your Mac.** Unsloth is CUDA-only.
That file is kept for when you have an NVIDIA GPU or a free Colab T4.
`step5_train_mlx.py` is your step 5 — MLX is Apple's own framework and
trains on the M2 GPU through unified memory.

The defaults are tuned to stay inside 8 GB: `--batch-size 1`,
`--num-layers 8` (adapters on the top 8 blocks only), `--grad-checkpoint`,
and a 4-bit base model. Expect **20-40 minutes** for 400 iterations. Quit
Chrome first.

If it runs out of memory, in this order:

```bash
python step5_train_mlx.py --num-layers 4
python step5_train_mlx.py --num-layers 4 --max-seq-length 512
python step5_train_mlx.py --size 1.7b
```

Do not raise `--batch-size` to "go faster" — it is the main cause of OOM.

### 6. Measure — do not skip this
```bash
# BEFORE training, get a baseline:
python step6_evaluate.py --model qwen3:4b

# free thing worth trying before you train anything at all:
python step6_evaluate.py --model qwen3:4b --force-json

# AFTER training — no file conversion needed:
python step6_evaluate.py --backend mlx --adapter-path adapters
```

The `mlx` backend runs the same base model twice, once with your adapter and
once without, so the only variable is the adapter. It scores valid-JSON rate,
schema correctness, exact match, and per-field accuracy.

Run the baseline **first**. Sometimes the honest finding is that `--force-json`
alone already scores well and the fine-tune was unnecessary. That is a
successful experiment, not a failed one.

---

## The three things worth taking away

**Retrieval for knowledge, fine-tuning for behaviour.** "Make it know our
docs" is step 3. "Make it always answer in our format" is step 5. Picking
wrong wastes days.

**Dataset quality beats quantity.** 500 clean, consistently-labelled examples
beat 5,000 noisy ones. In `step4_make_dataset.py` the `needs_human()` rule is
a single consistent function and the JSON targets are byte-identical in
format every time. Inconsistent labels are the most common reason a fine-tune
quietly fails to learn.

**A fine-tune that does not move your metric failed,** however nicely the
training loss fell. Hold out an eval set before you train, and compare.

---

## Things to try next

- Point `docs/` at your own markdown and re-run `ingest`. Most useful change
  you can make to this project.
- Compare `qwen3:1.7b` against `qwen3:4b` on the same eval. That delta is
  what model size buys you *for your task*, on your hardware.
- Replace the synthetic tickets in step 4 with 200 real examples of whatever
  you actually want formatted, and retrain.
- Watch train loss vs validation loss during step 5. If train keeps dropping
  while validation flattens, you are overfitting — use fewer iterations.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not reach Ollama` | `ollama serve` in another terminal |
| `404` from any script | `ollama pull <model>` |
| Everything is crawling | Memory pressure — check Activity Monitor, drop to `qwen3:1.7b` |
| OOM in step 5 | `--num-layers 4`, then `--max-seq-length 512`, then `--size 1.7b` |
| `mlx_lm` not found | `pip install mlx-lm` (macOS only) |
| RAG answers are wrong | `--show-chunks` first — it is usually retrieval |

## Layout

```
ollama_client.py        all HTTP calls to Ollama live here
step1_hello.py          first local inference
step2_chat.py           streaming chat + history
step3_rag.py            chunk, embed, retrieve, answer
step4_make_dataset.py   build train/valid/test jsonl
step5_train_mlx.py      LoRA on Apple Silicon      <-- yours
step5_train_qlora.py    QLoRA on NVIDIA/Colab
step6_evaluate.py       score base vs tuned
test_chunking.py        offline sanity checks
docs/                   fictional knowledge base for step 3
```
