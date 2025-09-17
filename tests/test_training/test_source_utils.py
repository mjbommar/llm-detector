from __future__ import annotations

from llm_detector.training.sources.utils import chunk_text, content_digest


def test_content_digest_is_stable() -> None:
    digest_a = content_digest("The quick brown fox")
    digest_b = content_digest("The quick brown fox")
    digest_c = content_digest("The quick brown box")

    assert len(digest_a) == 8
    assert digest_a == digest_b
    assert digest_a != digest_c


def test_chunk_text_handles_small_inputs() -> None:
    text = "A short passage that fits in one chunk."
    chunks = list(chunk_text(text, chunk_size=None, chunk_overlap=5))
    assert chunks == [(text.strip(), 0)]

    zero_chunks = list(chunk_text(text, chunk_size=0, chunk_overlap=2))
    assert zero_chunks == [(text.strip(), 0)]


def test_chunk_text_respects_separators() -> None:
    text = "Sentence one. Sentence two! Sentence three?"  # separators should be honoured
    chunks = list(chunk_text(text, chunk_size=20, chunk_overlap=0))

    # Expect at least two chunks that end on natural boundaries
    assert len(chunks) >= 2
    assert all(chunk for chunk, _ in chunks)
    assert chunks[0][0].endswith((".", "!", "?"))


def test_chunk_text_applies_overlap_and_indices() -> None:
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    chunks = list(chunk_text(text, chunk_size=25, chunk_overlap=10))

    indices = [index for _, index in chunks]
    assert indices == list(range(len(indices)))

    # Ensure overlap caused the second chunk to share some characters with the first
    if len(chunks) >= 2:
        first_chunk = chunks[0][0]
        second_chunk = chunks[1][0]
        overlap = set(first_chunk.split()) & set(second_chunk.split())
        assert overlap, "expected overlapping context between chunks"


def test_chunk_text_ignores_empty_input() -> None:
    assert list(chunk_text("", chunk_size=50, chunk_overlap=5)) == []
