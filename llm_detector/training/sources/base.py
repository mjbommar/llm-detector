"""Base primitives for streaming training data sources."""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from llm_detector.types import TextSample

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BatchConfig:
    """Controls sample filtering and ordering for a data source."""

    max_samples: int | None = None
    min_text_length: int = 1
    max_text_length: int | None = None
    shuffle: bool = False
    buffer_size: int = 10_000
    seed: int | None = None
    skip_samples: int = 0


class BaseDataSource:
    """Shared logic for streaming sources that yield :class:`TextSample`."""

    def __init__(self, config: BatchConfig | None = None) -> None:
        self.config = config or BatchConfig()
        self._yielded = 0
        self._skipped = 0

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def estimated_size(self) -> int | None:
        return None

    def configure(self, config: BatchConfig) -> None:
        self.config = config
        self._yielded = 0
        self._skipped = 0

    def prepare(self) -> None:  # pragma: no cover - hook for subclasses
        """Allow subclasses to perform lazy initialisation before iteration."""

    def __iter__(self) -> Iterator[TextSample]:
        self._yielded = 0
        self._skipped = 0
        self.prepare()

        for sample in self._generate_samples():
            if self._should_drop(sample):
                continue

            yield sample
            self._yielded += 1

            if self.config.max_samples is not None and self._yielded >= self.config.max_samples:
                break

    def _should_drop(self, sample: TextSample) -> bool:
        if self._skipped < self.config.skip_samples:
            self._skipped += 1
            return True

        text_len = len(sample.text)
        if text_len < self.config.min_text_length:
            return True
        if self.config.max_text_length is not None and text_len > self.config.max_text_length:
            return True

        return False

    def _generate_samples(self) -> Iterator[TextSample]:  # pragma: no cover - abstract
        raise NotImplementedError


class StreamingShuffleMixin(BaseDataSource):
    """Reservoir-style shuffle for large streaming corpora."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._rng: random.Random | None = None

    def _init_rng(self) -> None:
        if self._rng is None:
            seed = self.config.seed
            self._rng = random.Random(seed)

    def _maybe_shuffle(self, samples: Iterator[TextSample]) -> Iterator[TextSample]:
        if not self.config.shuffle or self.config.buffer_size <= 1:
            return samples

        self._init_rng()
        assert self._rng is not None

        buffer_size = self.config.buffer_size
        buffer: list[TextSample] = []
        iterator = iter(samples)

        try:
            while len(buffer) < buffer_size:
                buffer.append(next(iterator))
        except StopIteration:
            self._rng.shuffle(buffer)
            return iter(buffer)

        def _generator() -> Iterator[TextSample]:
            nonlocal buffer
            rng = self._rng
            assert rng is not None

            while True:
                idx = rng.randrange(len(buffer))
                yield buffer[idx]
                try:
                    buffer[idx] = next(iterator)
                except StopIteration:
                    break

            rng.shuffle(buffer)
            yield from buffer

        return _generator()

    def _generate_samples(self) -> Iterator[TextSample]:  # pragma: no cover - abstract
        raise NotImplementedError

    def __iter__(self) -> Iterator[TextSample]:
        self._yielded = 0
        self._skipped = 0
        self.prepare()

        raw_iter = self._generate_samples()
        stream = self._maybe_shuffle(raw_iter)

        for sample in stream:
            if self._should_drop(sample):
                continue

            yield sample
            self._yielded += 1

            if self.config.max_samples is not None and self._yielded >= self.config.max_samples:
                break


def require_datasets() -> Any:
    """Import and return the optional :mod:`datasets` package."""

    try:
        import datasets  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "datasets package not installed. Install with: uv pip install 'llm-detector[training]'"
        ) from exc

    return datasets


def take(iterable: Iterable[TextSample], n: int) -> list[TextSample]:
    """Collect up to ``n`` samples from ``iterable``."""

    out: list[TextSample] = []
    for sample in iterable:
        out.append(sample)
        if len(out) >= n:
            break
    return out


SourceFactory = Callable[[], BaseDataSource]


__all__ = [
    "BatchConfig",
    "BaseDataSource",
    "StreamingShuffleMixin",
    "require_datasets",
    "take",
    "SourceFactory",
]
