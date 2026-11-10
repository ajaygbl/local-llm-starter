"""
The parts of step 3 that do not need a model running. Useful as a sanity
check that your Python environment is fine before you blame Ollama.

    python test_chunking.py
"""

import numpy as np

from step3_rag import chunk_text, cosine_top_k


def test_short_text_is_one_chunk():
    assert chunk_text("hello world") == ["hello world"]


def test_empty():
    assert chunk_text("   ") == []


def test_long_text_is_split_and_covers_everything():
    text = ("Alpha paragraph here.\n\n" * 40).strip()
    chunks = chunk_text(text, size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)
    # nothing silently dropped: every chunk is a real substring, and the
    # first and last of the source survive
    assert all(c in text for c in chunks)
    assert text.startswith(chunks[0][:20])
    assert text.rstrip().endswith(chunks[-1][-20:])


def test_chunks_overlap():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_text(text, size=300, overlap=100)
    # consecutive chunks should share some tail/head text
    a_tail = chunks[0][-50:]
    assert any(tok in chunks[1] for tok in a_tail.split()[:3])


def test_cosine_ranks_identical_vector_first():
    matrix = [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]
    hits = cosine_top_k([1.0, 0.0], matrix, k=2)
    assert hits[0][0] == 0
    assert hits[0][1] > 0.99
    assert hits[1][0] == 2


def test_cosine_is_scale_invariant():
    matrix = [[2.0, 0.0], [0.0, 5.0]]
    hits = cosine_top_k([10.0, 0.0], matrix, k=1)
    assert hits[0][0] == 0
    assert np.isclose(hits[0][1], 1.0, atol=1e-5)


if __name__ == "__main__":
    passed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
            passed += 1
    print(f"\n{passed} passed")
