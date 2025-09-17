"""LLM-generated streaming corpora."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Iterator

from llm_detector.textops import DEFAULT_MIN_SENTENCE_LENGTH, segment_sentences
from llm_detector.types import TextSample

from .base import BatchConfig, StreamingShuffleMixin, require_datasets
from .utils import content_digest

logger = logging.getLogger(__name__)


class CosmopediaSource(StreamingShuffleMixin):
    """Stream long-form synthetic articles from Cosmopedia."""

    def __init__(
        self,
        dataset_name: str = "HuggingFaceTB/cosmopedia",
        subset: str = "web_samples_v2",
        split: str = "train",
        min_text_length: int = 200,
        min_sentence_length: int = DEFAULT_MIN_SENTENCE_LENGTH,
        config: BatchConfig | None = None,
    ) -> None:
        cfg = config or BatchConfig(min_text_length=min_sentence_length)
        if cfg.min_text_length < min_sentence_length:
            cfg.min_text_length = min_sentence_length
        super().__init__(cfg)
        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split
        self.min_text_length = min_text_length
        self.min_sentence_length = min_sentence_length
        self._dataset = None

    @property
    def name(self) -> str:
        return f"cosmopedia_{self.subset}"

    def prepare(self) -> None:
        if self._dataset is None:
            ds = require_datasets()
            logger.info(
                "Loading %s subset=%s split=%s in streaming mode",
                self.dataset_name,
                self.subset,
                self.split,
            )
            self._dataset = ds.load_dataset(
                self.dataset_name,
                self.subset,
                split=self.split,
                streaming=True,
                trust_remote_code=False,
                verification_mode="no_checks",
            )

    def _generate_samples(self) -> Iterator[TextSample]:
        self.prepare()
        assert self._dataset is not None
        dataset_iter = iter(self._dataset)

        for idx, row in enumerate(dataset_iter):
            if not isinstance(row, dict):
                continue
            text = row.get("text", "")
            if not text or len(text) < self.min_text_length:
                continue
            doc_id = row.get("id", f"cosmo_{idx}")
            subset = row.get("subset", self.subset)

            for segment in segment_sentences(text, min_length=self.min_sentence_length):
                sid_base = f"cosmo_{doc_id}" if doc_id else f"cosmo_{content_digest(text)}"
                sid = f"{sid_base}_p{segment.paragraph_index}_s{segment.sentence_index}"
                digest = content_digest(segment.text)
                yield TextSample(
                    text=segment.text,
                    is_llm=True,
                    source="cosmopedia",
                    sample_id=sid,
                    metadata={
                        "document_id": doc_id,
                        "subset": subset,
                        "paragraph_index": segment.paragraph_index,
                        "sentence_index": segment.sentence_index,
                        "content_hash": digest,
                    },
                )


class LMSYSSource(StreamingShuffleMixin):
    """Stream assistant responses from LMSYS multi-model chats."""

    def __init__(
        self,
        dataset_name: str = "lmsys/lmsys-chat-1m",
        split: str = "train",
        min_response_length: int = 50,
        min_sentence_length: int = DEFAULT_MIN_SENTENCE_LENGTH,
        config: BatchConfig | None = None,
    ) -> None:
        cfg = config or BatchConfig(min_text_length=min_sentence_length)
        if cfg.min_text_length < min_sentence_length:
            cfg.min_text_length = min_sentence_length
        super().__init__(cfg)
        self.dataset_name = dataset_name
        self.split = split
        self.min_response_length = min_response_length
        self.min_sentence_length = min_sentence_length
        self._dataset = None

    @property
    def name(self) -> str:
        return "lmsys"

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

    def _extract_responses(self, row: dict) -> Iterable[str]:
        conv = row.get("conversation") or row.get("messages") or row.get("conversations")
        if isinstance(conv, str):
            try:
                conv = json.loads(conv)
            except Exception:
                conv = []
        if not isinstance(conv, list):
            return []

        assistant_roles = {"assistant", "gpt", "bot", "model", "ai"}
        responses: list[str] = []
        for msg in conv:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or msg.get("from") or "").lower()
            if role not in assistant_roles:
                continue
            content = msg.get("content") or msg.get("value") or ""
            if isinstance(content, list):  # some variants store list of segments
                content = "\n\n".join(
                    seg.get("text", "") if isinstance(seg, dict) else str(seg) for seg in content
                )
            if not isinstance(content, str):
                continue
            content = content.strip()
            if len(content) >= self.min_response_length:
                responses.append(content)
        return responses

    def _generate_samples(self) -> Iterator[TextSample]:
        self.prepare()
        assert self._dataset is not None
        dataset_iter = iter(self._dataset)

        for idx, row in enumerate(dataset_iter):
            if not isinstance(row, dict):
                continue
            responses = list(self._extract_responses(row))
            if not responses:
                continue
            conversation_id = row.get("id", f"conv_{idx}")

            for turn_idx, resp in enumerate(responses):
                for segment in segment_sentences(resp, min_length=self.min_sentence_length):
                    sid = f"lmsys_{conversation_id}_turn{turn_idx}_p{segment.paragraph_index}_s{segment.sentence_index}"
                    digest = content_digest(segment.text)
                    yield TextSample(
                        text=segment.text,
                        is_llm=True,
                        source=self.name,
                        sample_id=sid,
                        metadata={
                            "conversation_id": conversation_id,
                            "turn_index": turn_idx,
                            "paragraph_index": segment.paragraph_index,
                            "sentence_index": segment.sentence_index,
                            "content_hash": digest,
                        },
                    )


class UltraChatSource(StreamingShuffleMixin):
    """Stream assistant messages from the UltraChat dataset."""

    def __init__(
        self,
        dataset_name: str = "HuggingFaceH4/ultrachat_200k",
        split: str = "train_sft",
        min_response_length: int = 50,
        min_sentence_length: int = DEFAULT_MIN_SENTENCE_LENGTH,
        config: BatchConfig | None = None,
    ) -> None:
        cfg = config or BatchConfig(min_text_length=min_sentence_length)
        if cfg.min_text_length < min_sentence_length:
            cfg.min_text_length = min_sentence_length
        super().__init__(cfg)
        self.dataset_name = dataset_name
        self.split = split
        self.min_response_length = min_response_length
        self.min_sentence_length = min_sentence_length
        self._dataset = None

    @property
    def name(self) -> str:
        return "ultrachat"

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

    def _extract_responses(self, row: dict) -> Iterable[str]:
        messages = row.get("messages")
        if isinstance(messages, str):
            try:
                messages = json.loads(messages)
            except Exception:
                messages = []
        if not isinstance(messages, list):
            return []

        responses: list[str] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "")).lower()
            if role != "assistant":
                continue
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "\n\n".join(
                    seg.get("text", "") if isinstance(seg, dict) else str(seg) for seg in content
                )
            if not isinstance(content, str):
                continue
            content = content.strip()
            if len(content) >= self.min_response_length:
                responses.append(content)
        return responses

    def _generate_samples(self) -> Iterator[TextSample]:
        self.prepare()
        assert self._dataset is not None
        dataset_iter = iter(self._dataset)

        for idx, row in enumerate(dataset_iter):
            if not isinstance(row, dict):
                continue
            responses = list(self._extract_responses(row))
            if not responses:
                continue
            conversation_id = row.get("id", f"conv_{idx}")

            for turn_idx, resp in enumerate(responses):
                for segment in segment_sentences(resp, min_length=self.min_sentence_length):
                    sid = f"ultrachat_{conversation_id}_turn{turn_idx}_p{segment.paragraph_index}_s{segment.sentence_index}"
                    digest = content_digest(segment.text)
                    yield TextSample(
                        text=segment.text,
                        is_llm=True,
                        source=self.name,
                        sample_id=sid,
                        metadata={
                            "conversation_id": conversation_id,
                            "turn_index": turn_idx,
                            "paragraph_index": segment.paragraph_index,
                            "sentence_index": segment.sentence_index,
                            "content_hash": digest,
                        },
                    )
