"""
STEP 3 — RAG: make the model answer questions about documents it has never seen.

    python step3_rag.py ingest          # chunk + embed docs/ into index.json
    python step3_rag.py ask "how do I roll back a deploy?"
    python step3_rag.py ask "..." --show-chunks    # see what was retrieved
    python step3_rag.py compare "..."              # with RAG vs without

Needs an embedding model as well as a chat model:
    ollama pull nomic-embed-text

What to notice:
  - docs/ describes a product called Meridian that does not exist. The base
    model will confidently invent an answer. With retrieval it gets it right.
    That contrast is the whole lesson: RAG is for knowledge.
  - There is no vector database here. It is a list of vectors and a dot
    product. Pinecone/Chroma/FAISS add scale and filtering, not magic.
  - Retrieval quality dominates answer quality. If the answer is wrong,
    look at --show-chunks before you blame the model.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

import ollama_client as llm

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.json")

CHUNK_CHARS = 700
OVERLAP_CHARS = 120
TOP_K = 3


# ---------------------------------------------------------------- chunking

def chunk_text(text, size=CHUNK_CHARS, overlap=OVERLAP_CHARS):
    """Split text into overlapping windows, preferring paragraph boundaries.

    Overlap matters: without it, a fact that straddles a boundary becomes
    unretrievable. This is pure string handling, so it is easy to test --
    see test_chunking.py.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # back off to the last paragraph or sentence break in this window
            window = text[start:end]
            for sep in ("\n\n", "\n", ". "):
                idx = window.rfind(sep)
                if idx > size * 0.5:
                    end = start + idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def cosine_top_k(query_vec, matrix, k):
    """Return (index, score) for the k closest rows. Plain numpy."""
    q = np.asarray(query_vec, dtype=np.float32)
    m = np.asarray(matrix, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-9)
    m = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
    scores = m @ q
    order = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in order]


# ---------------------------------------------------------------- ingest

def ingest():
    paths = sorted(glob.glob(os.path.join(DOCS_DIR, "*.md")))
    if not paths:
        sys.exit(f"No .md files found in {DOCS_DIR}")

    records = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        pieces = chunk_text(text)
        print(f"  {os.path.basename(path):40s} {len(pieces)} chunks")
        for i, piece in enumerate(pieces):
            records.append({
                "source": os.path.basename(path),
                "chunk_id": i,
                "text": piece,
            })

    print(f"\nEmbedding {len(records)} chunks with {llm.DEFAULT_EMBED_MODEL} ...")
    vectors = llm.embed([r["text"] for r in records])
    for r, v in zip(records, vectors):
        r["embedding"] = v

    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump({"model": llm.DEFAULT_EMBED_MODEL, "records": records}, f)
    print(f"Wrote {INDEX_PATH}  ({len(records)} chunks, "
          f"{len(vectors[0])} dimensions each)")


def load_index():
    if not os.path.exists(INDEX_PATH):
        sys.exit("No index found. Run:  python step3_rag.py ingest")
    with open(INDEX_PATH, encoding="utf-8") as f:
        return json.load(f)["records"]


# ---------------------------------------------------------------- ask

PROMPT = """Answer the question using only the context below.
If the context does not contain the answer, say "I don't know" -- do not guess.

Context:
{context}

Question: {question}"""


def retrieve(question, records, k=TOP_K):
    qvec = llm.embed([question])[0]
    matrix = [r["embedding"] for r in records]
    return [(records[i], score) for i, score in cosine_top_k(qvec, matrix, k)]


def ask(question, show_chunks=False):
    records = load_index()
    hits = retrieve(question, records)

    if show_chunks:
        print("--- retrieved ---")
        for rec, score in hits:
            print(f"[{score:.3f}] {rec['source']} #{rec['chunk_id']}")
            print(rec["text"][:200].replace("\n", " ") + "...\n")
        print("--- answer ---")

    context = "\n\n---\n\n".join(
        f"({rec['source']})\n{rec['text']}" for rec, _ in hits
    )
    reply = llm.chat(
        [{"role": "user", "content": PROMPT.format(context=context,
                                                   question=question)}],
        temperature=0,
    )
    print(reply.strip())
    if not show_chunks:
        srcs = sorted({rec["source"] for rec, _ in hits})
        print(f"\n[sources: {', '.join(srcs)}]")


def compare(question):
    print("=" * 70)
    print("WITHOUT retrieval (the model is on its own):")
    print("=" * 70)
    print(llm.chat([{"role": "user", "content": question}],
                   temperature=0).strip())
    print()
    print("=" * 70)
    print("WITH retrieval:")
    print("=" * 70)
    ask(question)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest")
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--show-chunks", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("question")

    args = p.parse_args()
    if args.cmd == "ingest":
        llm.require(llm.DEFAULT_EMBED_MODEL)
        ingest()
    elif args.cmd == "ask":
        llm.require(llm.DEFAULT_MODEL)
        ask(args.question, args.show_chunks)
    else:
        llm.require(llm.DEFAULT_MODEL)
        compare(args.question)


if __name__ == "__main__":
    main()
