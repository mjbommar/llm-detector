from __future__ import annotations

from llm_detector.textops import (
    DEFAULT_MIN_SENTENCE_LENGTH,
    iter_sentence_texts,
    segment_sentences,
)


def test_segment_sentences_handles_paragraphs() -> None:
    text = "Paragraph one has two sentences. Another sentence follows.\n\nSecond paragraph here."
    segments = segment_sentences(text, min_length=DEFAULT_MIN_SENTENCE_LENGTH)
    assert len(segments) == 3
    assert segments[0].paragraph_index == 0
    assert segments[2].paragraph_index == 1


def test_segment_sentences_respects_min_length() -> None:
    text = "Tiny. Adequate length sentence."  # first sentence filtered by min_length=10
    segments = segment_sentences(text, min_length=10)
    assert len(segments) == 1
    assert segments[0].text.endswith("sentence.")


def test_iter_sentence_texts_matches_segment_sentences() -> None:
    text = "Alpha sentence. Beta sentence."  # two sentences
    texts = list(iter_sentence_texts(text, min_length=DEFAULT_MIN_SENTENCE_LENGTH))
    segments = segment_sentences(text, min_length=DEFAULT_MIN_SENTENCE_LENGTH)
    assert texts == [segment.text for segment in segments]


def test_segment_sentences_empty_text_returns_empty_list() -> None:
    assert segment_sentences("   ") == []
