from __future__ import annotations

import builtins
import sys
from collections.abc import Iterator
from types import ModuleType

import pytest

from llm_detector.training.sources import base
from llm_detector.types import TextSample


class DummySource(base.BaseDataSource):
    def __init__(self, texts: list[str], *, config: base.BatchConfig | None = None) -> None:
        super().__init__(config=config)
        self._texts = texts

    @property
    def name(self) -> str:
        return "dummy"

    def _generate_samples(self) -> Iterator[TextSample]:
        for idx, text in enumerate(self._texts):
            yield TextSample(text=text, is_llm=False, source=self.name, sample_id=str(idx))


class DummyStreamingSource(base.StreamingShuffleMixin):
    def __init__(self, texts: list[str], *, config: base.BatchConfig) -> None:
        super().__init__(config=config)
        self._texts = texts

    @property
    def name(self) -> str:
        return "stream"

    def _generate_samples(self) -> Iterator[TextSample]:
        for idx, text in enumerate(self._texts):
            yield TextSample(text=text, is_llm=False, source=self.name, sample_id=str(idx))


def test_base_source_respects_config_filters() -> None:
    cfg = base.BatchConfig(max_samples=3, min_text_length=5, max_text_length=30, skip_samples=1)
    source = DummySource(
        ["tiny", "short one", "this text is fine", "this text will be dropped for length"],
        config=cfg,
    )

    texts = [sample.text for sample in source]
    assert texts == ["short one", "this text is fine"]


def test_take_collects_requested_items() -> None:
    source = DummySource(["first", "second", "third"])
    taken = base.take(iter(source), 2)
    assert [sample.text for sample in taken] == ["first", "second"]


def test_streaming_shuffle_handles_small_buffers() -> None:
    cfg = base.BatchConfig(shuffle=True, buffer_size=5, seed=42)
    source = DummyStreamingSource(["a", "b", "c"], config=cfg)
    texts = [sample.text for sample in source]
    assert sorted(texts) == ["a", "b", "c"]


def test_streaming_shuffle_identity_when_disabled() -> None:
    cfg = base.BatchConfig(shuffle=False)
    source = DummyStreamingSource(["a", "b"], config=cfg)
    texts = [sample.text for sample in source]
    assert texts == ["a", "b"]


def test_require_datasets_returns_cached_module(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = ModuleType("datasets")
    monkeypatch.setitem(sys.modules, "datasets", sentinel)
    loaded = base.require_datasets()
    assert loaded is sentinel


def test_require_datasets_raises_when_import_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # type: ignore[override]
        if name == "datasets":
            raise ImportError("missing module")
        return original_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "datasets", None)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError):
        base.require_datasets()


def test_streaming_shuffle_with_small_buffer_returns_order(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = base.BatchConfig(shuffle=True, buffer_size=1)
    source = DummyStreamingSource(["x", "y", "z"], config=cfg)
    texts = [sample.text for sample in source]
    assert texts == ["x", "y", "z"]
