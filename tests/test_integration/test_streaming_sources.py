"""Integration tests for streaming corpora."""

from __future__ import annotations

import time
from collections.abc import Callable

import pytest

from llm_detector.training import BaseDataSource, BatchConfig
from llm_detector.training.sources import (
    CosmopediaSource,
    FinePDFsSource,
    GutenbergSource,
    LMSYSSource,
    UltraChatSource,
    WikipediaSource,
)

pytestmark = pytest.mark.integration

pytest.importorskip("datasets")


SourceFactory = Callable[[], BaseDataSource]


def _source_factories() -> list[tuple[str, SourceFactory]]:
    return [
        (
            "wikipedia",
            lambda: WikipediaSource(
                min_article_length=200,
                min_sentence_length=5,
            ),
        ),
        (
            "gutenberg",
            lambda: GutenbergSource(
                min_book_length=1200,
                min_sentence_length=5,
            ),
        ),
        (
            "finepdfs",
            lambda: FinePDFsSource(
                min_doc_length=200,
                min_language_score=0.7,
                min_sentence_length=5,
            ),
        ),
        (
            "cosmopedia",
            lambda: CosmopediaSource(
                min_text_length=150,
                min_sentence_length=5,
            ),
        ),
        (
            "lmsys",
            lambda: LMSYSSource(
                min_response_length=30,
                min_sentence_length=5,
            ),
        ),
        (
            "ultrachat",
            lambda: UltraChatSource(
                min_response_length=30,
                min_sentence_length=5,
            ),
        ),
    ]


@pytest.mark.parametrize(
    "name,factory",
    _source_factories(),
    ids=lambda param: param[0] if isinstance(param, tuple) else str(param),
)
def test_streams_100_samples_quickly(name: str, factory: SourceFactory) -> None:
    source = factory()
    config = BatchConfig(max_samples=100, min_text_length=1, shuffle=False)
    source.configure(config)

    try:
        source.prepare()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"{name} dataset unavailable: {exc}")

    start = time.perf_counter()
    samples = list(source)
    duration = time.perf_counter() - start

    if len(samples) < 100:
        pytest.skip(f"{name} produced only {len(samples)} samples")

    assert duration < 10.0, f"{name} took {duration:.2f}s to stream 100 samples"
