"""Utility helpers shared across streaming sources."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

_BREAK_SEPARATORS: tuple[str, ...] = ("\n\n", ". ", ".\n", "! ", "!\n", "? ", "?\n")


def content_digest(text: str) -> str:
    """Return a short stable digest for ``text``."""

    return hashlib.md5(text.encode("utf-8", "ignore")).hexdigest()[:8]


def chunk_text(
    text: str,
    chunk_size: int | None,
    chunk_overlap: int,
    break_separators: tuple[str, ...] = _BREAK_SEPARATORS,
) -> Iterator[tuple[str, int]]:
    """Yield ``(chunk, index)`` pairs for ``text`` using soft boundaries.

    The function keeps chunks roughly ``chunk_size`` characters long while
    trying to respect sentence boundaries and paragraph breaks. Overlap is
    applied to preserve context between adjacent chunks.
    """

    if not text:
        return
    if not chunk_size or chunk_size <= 0 or len(text) <= chunk_size:
        yield text.strip(), 0
        return

    n = len(text)
    start = 0
    index = 0
    safe_overlap = max(0, min(chunk_overlap, chunk_size - 1))

    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            boundary = -1
            for sep in break_separators:
                candidate = text.rfind(sep, start + chunk_size // 2, end)
                if candidate != -1:
                    boundary = candidate + len(sep)
                    break
            if boundary != -1 and boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            yield chunk, index
            index += 1
        if end >= n:
            break
        start = max(start + 1, end - safe_overlap)


__all__ = ["chunk_text", "content_digest"]
