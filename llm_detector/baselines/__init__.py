"""Baseline distribution utilities for detector training and evaluation."""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from llm_detector.training.sources.base import BaseDataSource, BatchConfig, SourceFactory
from llm_detector.training.sources.registry import (
    DEFAULT_REGISTRY,
    SourceCategory,
    SourceRegistry,
)
from llm_detector.types import TextSample

_WORD_PUNCT_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_PUNCT_OR_SPACE = set(".,;:!?-—'\"()[]{}<>/\\")


@dataclass(slots=True)
class BaselineArtifact:
    """A probability distribution aligned to a fixed vocabulary."""

    distribution: list[float]
    vocabulary: list[str]
    metadata: dict[str, Any]
    version: str = "v1"

    def as_mapping(self) -> dict[str, float]:
        """Return the distribution as a mapping keyed by vocabulary tokens."""

        return {
            token: float(prob)
            for token, prob in zip(self.vocabulary, self.distribution, strict=False)
        }


@dataclass(slots=True)
class BaselineSet:
    """Paired baselines for human vs LLM corpora across feature spaces."""

    unicode: tuple[BaselineArtifact, BaselineArtifact]
    regex: tuple[BaselineArtifact, BaselineArtifact]
    punctws: tuple[BaselineArtifact, BaselineArtifact]


def _normalize(counter: Counter[str]) -> tuple[list[float], list[str]]:
    if not counter:
        return [], []
    vocab = list(counter.keys())
    total = sum(counter[token] for token in vocab)
    if total <= 0:
        return [0.0 for _ in vocab], vocab
    return [counter[token] / total for token in vocab], vocab


def _build_unicode(texts: Sequence[str], version: str) -> BaselineArtifact:
    counts: Counter[str] = Counter()
    total_chars = 0
    for text in texts:
        counts.update(text)
        total_chars += len(text)
    distribution, vocab = _normalize(counts)
    metadata = {"total_chars": total_chars}
    return BaselineArtifact(
        distribution=distribution, vocabulary=vocab, metadata=metadata, version=version
    )


def _build_regex(texts: Sequence[str], version: str) -> BaselineArtifact:
    counts: Counter[str] = Counter()
    total_tokens = 0
    for text in texts:
        tokens = _WORD_PUNCT_RE.findall(text)
        counts.update(tokens)
        total_tokens += len(tokens)
    distribution, vocab = _normalize(counts)
    metadata = {"total_tokens": total_tokens, "pattern": _WORD_PUNCT_RE.pattern}
    return BaselineArtifact(
        distribution=distribution, vocabulary=vocab, metadata=metadata, version=version
    )


def _build_punctws(texts: Sequence[str], version: str) -> BaselineArtifact:
    counts: Counter[str] = Counter()
    total = 0
    for text in texts:
        for ch in text:
            if ch in _PUNCT_OR_SPACE or ch.isspace():
                counts.update([ch])
                total += 1
    distribution, vocab = _normalize(counts)
    metadata = {"total_punct_ws": total}
    return BaselineArtifact(
        distribution=distribution, vocabulary=vocab, metadata=metadata, version=version
    )


def build_from_samples(samples: Iterable[TextSample], version: str = "v1") -> BaselineSet:
    """Construct paired baselines from labeled text samples."""

    human_texts: list[str] = []
    llm_texts: list[str] = []
    for sample in samples:
        if sample.is_llm:
            llm_texts.append(sample.text)
        else:
            human_texts.append(sample.text)

    unicode_h = _build_unicode(human_texts, version)
    unicode_l = _build_unicode(llm_texts, version)

    regex_h = _build_regex(human_texts, version)
    regex_l = _build_regex(llm_texts, version)

    punctws_h = _build_punctws(human_texts, version)
    punctws_l = _build_punctws(llm_texts, version)

    return BaselineSet(
        unicode=(unicode_h, unicode_l),
        regex=(regex_h, regex_l),
        punctws=(punctws_h, punctws_l),
    )


def _artifact_to_dict(artifact: BaselineArtifact) -> dict[str, Any]:
    return {
        "distribution": artifact.distribution,
        "vocabulary": artifact.vocabulary,
        "metadata": artifact.metadata,
        "version": artifact.version,
    }


def _artifact_from_dict(payload: dict[str, Any]) -> BaselineArtifact:
    return BaselineArtifact(
        distribution=list(payload.get("distribution", [])),
        vocabulary=list(payload.get("vocabulary", [])),
        metadata=dict(payload.get("metadata", {})),
        version=str(payload.get("version", "v1")),
    )


class BaselineCache:
    """Persist baseline artifacts to disk using gzip-compressed JSON."""

    @staticmethod
    def save(baselines: BaselineSet, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "unicode": [
                _artifact_to_dict(baselines.unicode[0]),
                _artifact_to_dict(baselines.unicode[1]),
            ],
            "regex": [_artifact_to_dict(baselines.regex[0]), _artifact_to_dict(baselines.regex[1])],
            "punctws": [
                _artifact_to_dict(baselines.punctws[0]),
                _artifact_to_dict(baselines.punctws[1]),
            ],
        }

        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)

    @staticmethod
    def load(path: Path) -> BaselineSet:
        path = Path(path)
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)

        unicode_items = [_artifact_from_dict(item) for item in payload.get("unicode", [])]
        regex_items = [_artifact_from_dict(item) for item in payload.get("regex", [])]
        punct_items = [_artifact_from_dict(item) for item in payload.get("punctws", [])]

        def _ensure_pair(
            items: list[BaselineArtifact],
        ) -> tuple[BaselineArtifact, BaselineArtifact]:
            if len(items) >= 2:
                return items[0], items[1]
            if len(items) == 1:
                empty = BaselineArtifact(
                    distribution=[], vocabulary=[], metadata={}, version=items[0].version
                )
                return items[0], empty
            empty = BaselineArtifact(distribution=[], vocabulary=[], metadata={}, version="v1")
            return empty, empty

        return BaselineSet(
            unicode=_ensure_pair(unicode_items),
            regex=_ensure_pair(regex_items),
            punctws=_ensure_pair(punct_items),
        )

    @staticmethod
    def get_or_compute(path: Path, compute_fn: Callable[[], BaselineSet]) -> BaselineSet:
        path = Path(path)
        if path.exists():
            return BaselineCache.load(path)
        baselines = compute_fn()
        BaselineCache.save(baselines, path)
        return baselines


def _with_limit(config: BatchConfig, max_samples: int | None) -> BatchConfig:
    updates: dict[str, Any] = {"shuffle": False}
    if max_samples is not None:
        if max_samples < 0:
            raise ValueError("max_samples must be non-negative")
        updates["max_samples"] = max_samples
    return replace(config, **updates)


def collect_samples_from_factories(
    factories: Sequence[SourceFactory],
    limit_per_source: int | None,
    *,
    expected_is_llm: bool,
) -> list[TextSample]:
    """Instantiate sources from ``factories`` and gather samples.

    Parameters:
        factories: Callables that produce preconfigured :class:`BaseDataSource` instances.
        limit_per_source: Maximum samples to request from each source (``None`` keeps existing limit).
        expected_is_llm: Whether each emitted sample should be labeled as LLM-generated.
    """

    samples: list[TextSample] = []
    for factory in factories:
        source = factory()
        limit = limit_per_source if limit_per_source is not None else source.config.max_samples
        if limit is not None and limit <= 0:
            continue
        limited_config = _with_limit(source.config, limit)
        source.configure(limited_config)

        for sample in source:
            if sample.is_llm != expected_is_llm:
                raise ValueError(
                    f"source '{source.name}' emitted sample with is_llm={sample.is_llm},"
                    f" expected {expected_is_llm}"
                )
            samples.append(sample)
    return samples


def build_from_source_factories(
    human_factories: Sequence[SourceFactory],
    llm_factories: Sequence[SourceFactory],
    *,
    samples_per_source: int | None = None,
    version: str = "v1",
) -> BaselineSet:
    """Construct baselines by streaming from configured data source factories."""

    human_samples = collect_samples_from_factories(
        human_factories,
        samples_per_source,
        expected_is_llm=False,
    )
    llm_samples = collect_samples_from_factories(
        llm_factories,
        samples_per_source,
        expected_is_llm=True,
    )
    combined: list[TextSample] = []
    combined.extend(human_samples)
    combined.extend(llm_samples)
    return build_from_samples(combined, version=version)


def build_from_registry(
    registry: SourceRegistry,
    *,
    samples_per_source: int | None = None,
    version: str = "v1",
    enabled_only: bool = True,
) -> BaselineSet:
    """Construct baselines from a :class:`SourceRegistry` selection."""

    human_factories = registry.factories(
        category=SourceCategory.HUMAN,
        enabled_only=enabled_only,
    )
    llm_factories = registry.factories(
        category=SourceCategory.LLM,
        enabled_only=enabled_only,
    )
    return build_from_source_factories(
        human_factories,
        llm_factories,
        samples_per_source=samples_per_source,
        version=version,
    )


def build_default_registry_baselines(
    *,
    samples_per_source: int | None = None,
    version: str = "v1",
    enabled_only: bool = True,
    registry: SourceRegistry | None = None,
) -> BaselineSet:
    """Convenience wrapper that uses the package default source registry."""

    reg = registry or DEFAULT_REGISTRY
    return build_from_registry(
        reg,
        samples_per_source=samples_per_source,
        version=version,
        enabled_only=enabled_only,
    )


def compute_baselines_to_path(
    output_path: Path,
    *,
    samples_per_source: int | None = None,
    version: str = "v1",
    enabled_only: bool = True,
    overwrite: bool = False,
    registry: SourceRegistry | None = None,
) -> BaselineSet:
    """Build and persist baselines to ``output_path``.

    When ``overwrite`` is ``False`` (default) this will reuse an existing cache
    if the file already exists.
    """

    path = Path(output_path)

    def _compute() -> BaselineSet:
        return build_default_registry_baselines(
            samples_per_source=samples_per_source,
            version=version,
            enabled_only=enabled_only,
            registry=registry,
        )

    if not overwrite:
        return BaselineCache.get_or_compute(path, _compute)

    baselines = _compute()
    BaselineCache.save(baselines, path)
    return baselines


def divergence_baseline_overrides(
    baselines: BaselineSet,
    *,
    cohort: Literal["human", "llm"] = "human",
) -> dict[str, Mapping[str, float]]:
    """Construct baseline distributions keyed by divergence feature name."""

    if cohort not in {"human", "llm"}:
        raise ValueError("cohort must be either 'human' or 'llm'")

    index = 0 if cohort == "human" else 1
    unicode_artifact = baselines.unicode[index]
    regex_artifact = baselines.regex[index]
    punct_artifact = baselines.punctws[index]

    return {
        "div.char_jsd": unicode_artifact.as_mapping(),
        "div.regex_jsd": regex_artifact.as_mapping(),
        "div.punct_jsd": punct_artifact.as_mapping(),
    }


__all__ = [
    "BaselineArtifact",
    "BaselineSet",
    "BaselineCache",
    "build_from_samples",
    "BaseDataSource",
    "SourceFactory",
    "collect_samples_from_factories",
    "build_from_source_factories",
    "build_from_registry",
    "build_default_registry_baselines",
    "compute_baselines_to_path",
    "divergence_baseline_overrides",
]
