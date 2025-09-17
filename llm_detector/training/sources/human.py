"""Human-written streaming corpora."""

from __future__ import annotations

import logging
from collections.abc import Iterator

from llm_detector.textops import DEFAULT_MIN_SENTENCE_LENGTH, segment_sentences
from llm_detector.types import TextSample

from .base import BatchConfig, StreamingShuffleMixin, require_datasets
from .utils import content_digest

logger = logging.getLogger(__name__)


class WikipediaSource(StreamingShuffleMixin):
    """Stream English Wikipedia articles via Hugging Face."""

    def __init__(
        self,
        dataset_name: str = "wikimedia/wikipedia",
        dataset_config: str = "20231101.en",
        split: str = "train",
        include_title: bool = True,
        min_article_length: int = 200,
        min_sentence_length: int = DEFAULT_MIN_SENTENCE_LENGTH,
        config: BatchConfig | None = None,
    ) -> None:
        cfg = config or BatchConfig(min_text_length=min_sentence_length)
        if cfg.min_text_length < min_sentence_length:
            cfg.min_text_length = min_sentence_length
        super().__init__(cfg)
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.include_title = include_title
        self.min_article_length = min_article_length
        self.min_sentence_length = min_sentence_length
        self._dataset = None

    @property
    def name(self) -> str:
        return "wikipedia"

    def prepare(self) -> None:
        if self._dataset is None:
            ds = require_datasets()
            logger.info(
                "Loading %s (%s) split=%s in streaming mode",
                self.dataset_name,
                self.dataset_config,
                self.split,
            )
            self._dataset = ds.load_dataset(
                self.dataset_name,
                self.dataset_config,
                split=self.split,
                streaming=True,
                trust_remote_code=False,
                verification_mode="no_checks",
            )

    def _generate_samples(self) -> Iterator[TextSample]:
        self.prepare()
        assert self._dataset is not None
        dataset_iter = iter(self._dataset)

        for row in dataset_iter:
            article_text = row.get("text", "") if isinstance(row, dict) else ""
            if not article_text or len(article_text) < self.min_article_length:
                continue
            title = row.get("title", "") if isinstance(row, dict) else ""
            full_text = (
                f"{title}\n\n{article_text}" if (self.include_title and title) else article_text
            )
            article_id = str(row.get("id", "")) if isinstance(row, dict) else ""
            url = row.get("url", "") if isinstance(row, dict) else ""

            for segment in segment_sentences(full_text, min_length=self.min_sentence_length):
                sid_base = (
                    f"wiki_{article_id}" if article_id else f"wiki_{content_digest(full_text)}"
                )
                sid = f"{sid_base}_p{segment.paragraph_index}_s{segment.sentence_index}"
                yield TextSample(
                    text=segment.text,
                    is_llm=False,
                    source=self.name,
                    sample_id=sid,
                    metadata={
                        "article_id": article_id or None,
                        "article_title": title or None,
                        "article_url": url or None,
                        "paragraph_index": segment.paragraph_index,
                        "sentence_index": segment.sentence_index,
                        "content_hash": content_digest(segment.text),
                    },
                )


class GutenbergSource(StreamingShuffleMixin):
    """Stream Project Gutenberg books in English via Hugging Face."""

    def __init__(
        self,
        dataset_name: str = "manu/project_gutenberg",
        split: str = "en",
        min_book_length: int = 2000,
        min_sentence_length: int = DEFAULT_MIN_SENTENCE_LENGTH,
        config: BatchConfig | None = None,
    ) -> None:
        cfg = config or BatchConfig(min_text_length=min_sentence_length)
        if cfg.min_text_length < min_sentence_length:
            cfg.min_text_length = min_sentence_length
        super().__init__(cfg)
        self.dataset_name = dataset_name
        self.split = split
        self.min_book_length = min_book_length
        self.min_sentence_length = min_sentence_length
        self._dataset = None

    @property
    def name(self) -> str:
        return "gutenberg"

    def prepare(self) -> None:
        if self._dataset is None:
            ds = require_datasets()
            logger.info("Loading %s split=%s in streaming mode", self.dataset_name, self.split)
            self._dataset = ds.load_dataset(
                self.dataset_name,
                split=self.split,
                streaming=True,
                trust_remote_code=False,
                verification_mode="no_checks",
            )

    def _generate_samples(self) -> Iterator[TextSample]:
        self.prepare()
        assert self._dataset is not None
        dataset_iter = iter(self._dataset)

        for row in dataset_iter:
            if not isinstance(row, dict):
                continue
            text = row.get("text", "")
            if not text or len(text) < self.min_book_length:
                continue
            book_id = str(row.get("id", ""))
            title = row.get("title", "")
            author = row.get("author", "")

            for segment in segment_sentences(text, min_length=self.min_sentence_length):
                sid_base = (
                    f"gutenberg_{book_id}" if book_id else f"gutenberg_{content_digest(text)}"
                )
                sid = f"{sid_base}_p{segment.paragraph_index}_s{segment.sentence_index}"
                yield TextSample(
                    text=segment.text,
                    is_llm=False,
                    source=self.name,
                    sample_id=sid,
                    metadata={
                        "book_id": book_id or None,
                        "title": title or None,
                        "author": author or None,
                        "paragraph_index": segment.paragraph_index,
                        "sentence_index": segment.sentence_index,
                        "content_hash": content_digest(segment.text),
                    },
                )


class FinePDFsSource(StreamingShuffleMixin):
    """Stream FinePDF documents filtered for English content."""

    def __init__(
        self,
        dataset_name: str = "HuggingFaceFW/finepdfs",
        dataset_config: str = "eng_Latn",
        split: str = "train",
        include_test_split: bool = False,
        min_doc_length: int = 300,
        min_language_score: float = 0.8,
        min_sentence_length: int = DEFAULT_MIN_SENTENCE_LENGTH,
        config: BatchConfig | None = None,
    ) -> None:
        cfg = config or BatchConfig(min_text_length=min_sentence_length)
        if cfg.min_text_length < min_sentence_length:
            cfg.min_text_length = min_sentence_length
        super().__init__(cfg)
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.include_test_split = include_test_split
        self.min_doc_length = min_doc_length
        self.min_language_score = min_language_score
        self.min_sentence_length = min_sentence_length
        self._train = None
        self._test = None

    @property
    def name(self) -> str:
        return "finepdfs"

    def prepare(self) -> None:
        if self._train is None:
            ds = require_datasets()
            logger.info(
                "Loading %s config=%s split=%s in streaming mode",
                self.dataset_name,
                self.dataset_config,
                self.split,
            )
            self._train = ds.load_dataset(
                self.dataset_name,
                self.dataset_config,
                split=self.split,
                streaming=True,
                trust_remote_code=False,
                verification_mode="no_checks",
            )
            if self.include_test_split:
                try:
                    self._test = ds.load_dataset(
                        self.dataset_name,
                        self.dataset_config,
                        split="test",
                        streaming=True,
                        trust_remote_code=False,
                        verification_mode="no_checks",
                    )
                except Exception as exc:  # pragma: no cover - optional split
                    logger.warning("FinePDF test split unavailable: %s", exc)
                    self._test = None

    def _iter_rows(self) -> Iterator[dict]:
        self.prepare()
        assert self._train is not None
        for row in iter(self._train):
            if isinstance(row, dict):
                yield row
        if self._test is not None:
            for row in iter(self._test):
                if isinstance(row, dict):
                    yield row

    def _generate_samples(self) -> Iterator[TextSample]:
        for row in self._iter_rows():
            text = row.get("text", "")
            if not text or len(text) < self.min_doc_length:
                continue
            try:
                score = max(
                    float(row.get("page_average_lid_score", 0.0) or 0.0),
                    float(row.get("full_doc_lid_score", 0.0) or 0.0),
                )
            except (TypeError, ValueError):
                score = 0.0
            if score < self.min_language_score:
                continue
            doc_id = str(row.get("id", ""))

            for segment in segment_sentences(text, min_length=self.min_sentence_length):
                sid_base = f"finepdf_{doc_id}" if doc_id else f"finepdf_{content_digest(text)}"
                sid = f"{sid_base}_p{segment.paragraph_index}_s{segment.sentence_index}"
                yield TextSample(
                    text=segment.text,
                    is_llm=False,
                    source=self.name,
                    sample_id=sid,
                    metadata={
                        "document_id": doc_id or None,
                        "paragraph_index": segment.paragraph_index,
                        "sentence_index": segment.sentence_index,
                        "content_hash": content_digest(segment.text),
                        "language_score": score,
                    },
                )
