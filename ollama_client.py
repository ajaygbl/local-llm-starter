"""
Tiny wrapper around the Ollama HTTP API.

Ollama runs a local server on port 11434. Everything in this project talks to
it through these three functions, so there is only one place to look when
something breaks.

Docs: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

import json
import os
import sys

import requests

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Default sized for an 8GB Apple Silicon Mac. macOS itself holds ~3GB, so
# the practical budget for a model plus its KV cache is ~4.5GB.
#   qwen3:1.7b  ~1.4GB   fast, good enough for steps 1-2
#   qwen3:4b    ~2.6GB   the sweet spot here  <-- default
#   qwen3:8b    ~5.2GB   will swap on this machine; avoid
DEFAULT_MODEL = os.environ.get("LLM_MODEL", "qwen3:4b")
DEFAULT_EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")

TIMEOUT = 300  # seconds; local generation on CPU can be slow


class OllamaError(RuntimeError):
    pass


def _post(path, payload, stream=False):
    try:
        r = requests.post(f"{HOST}{path}", json=payload, stream=stream, timeout=TIMEOUT)
    except requests.exceptions.ConnectionError:
        raise OllamaError(
            f"Could not reach Ollama at {HOST}.\n"
            "  - Is it installed?  https://ollama.com/download\n"
            "  - Is it running?    try `ollama serve` in another terminal\n"
        )
    if r.status_code == 404:
        raise OllamaError(
            f"Ollama returned 404 for {path}. The model is probably not pulled yet.\n"
            f"  Try: ollama pull {payload.get('model', DEFAULT_MODEL)}"
        )
    r.raise_for_status()
    return r


def chat(messages, model=DEFAULT_MODEL, temperature=0.7, fmt=None):
    """Send a list of {role, content} messages, get the full reply as a string.

    fmt="json" asks Ollama to constrain decoding to valid JSON. This is worth
    knowing about: it is often a cheaper fix than fine-tuning.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    if fmt:
        payload["format"] = fmt
    data = _post("/api/chat", payload).json()
    return data["message"]["content"]


def chat_stream(messages, model=DEFAULT_MODEL, temperature=0.7):
    """Same as chat(), but yields tokens as they are produced."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": temperature},
    }
    with _post("/api/chat", payload, stream=True) as r:
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("done"):
                break
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                yield piece


def embed(texts, model=DEFAULT_EMBED_MODEL):
    """Turn a list of strings into a list of vectors."""
    if isinstance(texts, str):
        texts = [texts]
    data = _post("/api/embed", {"model": model, "input": texts}).json()
    return data["embeddings"]


def list_models():
    try:
        r = requests.get(f"{HOST}/api/tags", timeout=10)
        r.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise OllamaError(f"Could not reach Ollama at {HOST}. Is it running?")
    return [m["name"] for m in r.json().get("models", [])]


def require(model):
    """Exit with a friendly message if a model has not been pulled."""
    try:
        have = list_models()
    except OllamaError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    # Ollama reports "qwen3:8b"; allow the user to write "qwen3" too.
    if model not in have and f"{model}:latest" not in have:
        print(f"Model '{model}' is not downloaded yet.", file=sys.stderr)
        print(f"  Run: ollama pull {model}", file=sys.stderr)
        if have:
            print(f"  You currently have: {', '.join(have)}", file=sys.stderr)
        sys.exit(1)
