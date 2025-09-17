"""Shared text segmentation utilities for training and inference."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import nupunkt_rs

DEFAULT_MIN_SENTENCE_LENGTH = 5


@dataclass(frozen=True)
class SentenceSegment:
    """A single sentence extracted from a document."""

    text: str
    paragraph_index: int
    sentence_index: int


def segment_sentences(
    text: str,
    *,
    min_length: int = DEFAULT_MIN_SENTENCE_LENGTH,
) -> list[SentenceSegment]:
    """Split ``text`` into sentences using the bundled Punkt model."""

    clean = text.strip()
    if not clean:
        return []

    segments: list[SentenceSegment] = []
    paragraphs = nupunkt_rs.para_tokenize(clean)
    for p_idx, sentences in enumerate(paragraphs):
        if not sentences:
            continue
        for s_idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) < min_length:
                continue
            segments.append(
                SentenceSegment(
                    text=sentence,
                    paragraph_index=p_idx,
                    sentence_index=s_idx,
                )
            )
    return segments


def iter_sentence_texts(
    text: str,
    *,
    min_length: int = DEFAULT_MIN_SENTENCE_LENGTH,
) -> Iterable[str]:
    """Yield sentence strings from ``text`` respecting ``min_length``."""

    for segment in segment_sentences(text, min_length=min_length):
        yield segment.text


__all__ = [
    "SentenceSegment",
    "DEFAULT_MIN_SENTENCE_LENGTH",
    "segment_sentences",
    "iter_sentence_texts",
]
