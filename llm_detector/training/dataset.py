"""Build feature matrices for model training."""

from __future__ import annotations

import random
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace

try:  # Optional dependency for progress display
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback when tqdm missing
    tqdm = None  # type: ignore[assignment]

from llm_detector.features.vectorizer import FeatureVectorizer
from llm_detector.types import TextSample

from .sources.base import BatchConfig, SourceFactory
from .sources.registry import SourceCategory, SourceRegistry

_WARNED_TQDM_MISSING = False


def _warn_tqdm_missing() -> None:
    global _WARNED_TQDM_MISSING
    if not _WARNED_TQDM_MISSING:
        _WARNED_TQDM_MISSING = True
        print(
            "progress requested but tqdm is not installed; install `tqdm` for progress bars",
            file=sys.stderr,
        )


@dataclass(slots=True)
class FeatureDataset:
    """Matrix-style representation of features and labels."""

    feature_names: list[str]
    matrix: list[list[float]]
    labels: list[int]
    sources: list[str]
    texts: list[str] | None = None

    def __post_init__(self) -> None:
        n_rows = len(self.matrix)
        if not (len(self.labels) == len(self.sources) == n_rows):
            raise ValueError("matrix, labels, and sources must have the same length")
        if self.texts is not None and len(self.texts) != n_rows:
            raise ValueError("texts length must match number of rows")

    def __len__(self) -> int:  # pragma: no cover - simple w/o logic
        return len(self.labels)

    def split(
        self, test_ratio: float, *, seed: int | None = None
    ) -> tuple[FeatureDataset, FeatureDataset]:
        """Split the dataset into train/test partitions."""

        if not 0.0 < test_ratio < 1.0:
            raise ValueError("test_ratio must be between 0 and 1")
        total = len(self)
        if total == 0:
            raise ValueError("cannot split an empty dataset")

        indices = list(range(total))
        rng = random.Random(seed)
        rng.shuffle(indices)

        test_size = max(1, int(total * test_ratio))
        test_indices = set(indices[:test_size])

        def _subset(selected: list[int]) -> FeatureDataset:
            texts = [self.texts[i] for i in selected] if self.texts is not None else None
            return FeatureDataset(
                feature_names=list(self.feature_names),
                matrix=[self.matrix[i] for i in selected],
                labels=[self.labels[i] for i in selected],
                sources=[self.sources[i] for i in selected],
                texts=texts,
            )

        train_indices = [i for i in indices if i not in test_indices]
        test_indices_list = [i for i in indices if i in test_indices]
        if not train_indices:
            raise ValueError("train split would be empty")

        return _subset(train_indices), _subset(test_indices_list)


def _with_limit(config: BatchConfig, max_samples: int | None) -> BatchConfig:
    updates = {"shuffle": False}
    if max_samples is not None:
        if max_samples < 0:
            raise ValueError("max_samples must be non-negative")
        updates["max_samples"] = max_samples
    return replace(config, **updates)


def _iter_factory_samples(
    factory: SourceFactory,
    limit_per_source: int | None,
    *,
    expected_is_llm: bool,
) -> Iterator[TextSample]:
    source = factory()
    limited_config = _with_limit(source.config, limit_per_source)
    source.configure(limited_config)

    for sample in source:
        if sample.is_llm != expected_is_llm:
            raise ValueError(
                f"source '{source.name}' emitted sample with is_llm={sample.is_llm}, expected {expected_is_llm}"
            )
        yield sample


def _collect_from_factories(
    factories: Sequence[SourceFactory],
    limit_per_source: int | None,
    *,
    expected_is_llm: bool,
) -> list[TextSample]:
    samples: list[TextSample] = []
    for factory in factories:
        samples.extend(
            list(_iter_factory_samples(factory, limit_per_source, expected_is_llm=expected_is_llm))
        )
    return samples


def build_feature_dataset(
    vectorizer: FeatureVectorizer,
    samples: Iterable[TextSample],
    *,
    keep_text: bool = False,
    show_progress: bool = False,
) -> FeatureDataset:
    """Transform text samples into a dense feature matrix."""

    feature_names = vectorizer.feature_names
    matrix: list[list[float]] = []
    labels: list[int] = []
    sources: list[str] = []
    texts: list[str] | None = [] if keep_text else None

    iterable: Iterable[TextSample]
    total: int | None = None

    progress_bar = None
    if show_progress and tqdm is not None:
        if hasattr(samples, "__len__"):
            try:
                total = len(samples)  # type: ignore[arg-type]
            except Exception:
                total = None
        progress_bar = tqdm(samples, total=total, desc="Vectorizing", leave=False)
        iterable = progress_bar
    else:
        if show_progress and tqdm is None:
            _warn_tqdm_missing()
        iterable = samples

    for sample in iterable:
        vector = vectorizer.vectorize(sample.text)
        matrix.append(vector.values)
        labels.append(1 if sample.is_llm else 0)
        sources.append(sample.source)
        if texts is not None:
            texts.append(sample.text)

    if progress_bar is not None:
        progress_bar.close()

    return FeatureDataset(
        feature_names=feature_names,
        matrix=matrix,
        labels=labels,
        sources=sources,
        texts=texts,
    )


def build_dataset_from_registry(
    registry: SourceRegistry,
    vectorizer: FeatureVectorizer,
    *,
    samples_per_source: int | None = None,
    enabled_only: bool = True,
    balance: bool = True,
    shuffle: bool = True,
    seed: int | None = None,
    keep_text: bool = False,
    show_progress: bool = False,
) -> FeatureDataset:
    """Stream samples from the registry and build a feature dataset."""

    human_factories = registry.factories(
        category=SourceCategory.HUMAN,
        enabled_only=enabled_only,
    )
    llm_factories = registry.factories(
        category=SourceCategory.LLM,
        enabled_only=enabled_only,
    )

    human_samples = _collect_from_factories(
        human_factories,
        samples_per_source,
        expected_is_llm=False,
    )
    llm_samples = _collect_from_factories(
        llm_factories,
        samples_per_source,
        expected_is_llm=True,
    )

    if balance and human_samples and llm_samples:
        target = min(len(human_samples), len(llm_samples))
        human_samples = human_samples[:target]
        llm_samples = llm_samples[:target]

    combined = human_samples + llm_samples

    if shuffle and combined:
        rng = random.Random(seed)
        rng.shuffle(combined)

    return build_feature_dataset(
        vectorizer,
        combined,
        keep_text=keep_text,
        show_progress=show_progress,
    )


__all__ = [
    "FeatureDataset",
    "build_feature_dataset",
    "build_dataset_from_registry",
]
