"""Tests for baseline construction using streaming source factories."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

import pytest

from llm_detector.baselines import (
    build_default_registry_baselines,
    build_from_registry,
    build_from_source_factories,
    compute_baselines_to_path,
)
from llm_detector.training import (
    BaseDataSource,
    BatchConfig,
    SourceCategory,
    SourceDefinition,
    SourceRegistry,
)
from llm_detector.types import TextSample


class StaticSource(BaseDataSource):
    """Deterministic data source for testing orchestration helpers."""

    def __init__(
        self,
        name: str,
        *,
        is_llm: bool,
        payloads: Iterable[str],
    ) -> None:
        super().__init__(BatchConfig(shuffle=True))
        self._name = name
        self._is_llm = is_llm
        self._payloads = list(payloads)
        self.last_config = self.config

    @property
    def name(self) -> str:
        return self._name

    def configure(self, config: BatchConfig) -> None:
        super().configure(config)
        self.last_config = config

    def _generate_samples(self):
        for idx, text in enumerate(self._payloads):
            yield TextSample(
                text=text,
                is_llm=self._is_llm,
                source=self._name,
                sample_id=f"{self._name}_{idx}",
                metadata={"index": idx},
            )


def make_factory(
    name: str, *, is_llm: bool, payloads: Iterable[str]
) -> tuple[StaticSource, Callable[[], StaticSource]]:
    source = StaticSource(name, is_llm=is_llm, payloads=payloads)
    return source, (lambda: source)


def make_counting_factory(
    name: str,
    *,
    is_llm: bool,
    payloads: Iterable[str],
) -> tuple[dict[str, object], Callable[[], StaticSource]]:
    payload_list = list(payloads)
    record: dict[str, object] = {"count": 0, "last_source": None}

    def factory() -> StaticSource:
        record["count"] = int(record["count"]) + 1
        source = StaticSource(name, is_llm=is_llm, payloads=payload_list)
        record["last_source"] = source
        return source

    return record, factory


def test_build_from_source_factories_limits_and_metadata():
    human_payloads = ["A", "BB", "CCC"]
    llm_payloads = ["XX", "YYY", "ZZZZ"]
    human_source, human_factory = make_factory("human", is_llm=False, payloads=human_payloads)
    llm_source, llm_factory = make_factory("llm", is_llm=True, payloads=llm_payloads)

    baselines = build_from_source_factories(
        [human_factory],
        [llm_factory],
        samples_per_source=2,
        version="test",
    )

    assert human_source.last_config.max_samples == 2
    assert llm_source.last_config.max_samples == 2
    assert not human_source.last_config.shuffle
    assert not llm_source.last_config.shuffle

    expected_human_chars = len("A") + len("BB")
    expected_llm_chars = len("XX") + len("YYY")
    assert baselines.unicode[0].metadata["total_chars"] == expected_human_chars
    assert baselines.unicode[1].metadata["total_chars"] == expected_llm_chars
    assert baselines.unicode[0].version == "test"
    assert baselines.unicode[1].version == "test"


def test_build_from_source_factories_validates_labels():
    _, human_factory = make_factory("human", is_llm=False, payloads=["Valid human text"])
    _, llm_factory = make_factory("llm", is_llm=False, payloads=["Mislabeled response"])

    with pytest.raises(ValueError):
        build_from_source_factories(
            [human_factory],
            [llm_factory],
            samples_per_source=1,
        )


def test_build_from_registry_filters_sources():
    registry = SourceRegistry()
    human_record, human_factory = make_counting_factory("human", is_llm=False, payloads=["A", "BB"])
    llm_record, llm_factory = make_counting_factory("llm", is_llm=True, payloads=["XX", "YYY"])
    disabled_record, disabled_factory = make_counting_factory(
        "llm_disabled", is_llm=True, payloads=["unused"]
    )

    registry.register(
        SourceDefinition(
            name="human",
            category=SourceCategory.HUMAN,
            factory=human_factory,
        )
    )
    registry.register(
        SourceDefinition(
            name="llm",
            category=SourceCategory.LLM,
            factory=llm_factory,
        )
    )
    registry.register(
        SourceDefinition(
            name="llm_disabled",
            category=SourceCategory.LLM,
            factory=disabled_factory,
            enabled=False,
        )
    )

    baselines = build_from_registry(registry, samples_per_source=1)

    assert human_record["count"] == 1
    assert llm_record["count"] == 1
    assert disabled_record["count"] == 0

    human_source = human_record["last_source"]
    llm_source = llm_record["last_source"]
    assert isinstance(human_source, StaticSource)
    assert isinstance(llm_source, StaticSource)
    assert human_source.last_config.max_samples == 1
    assert llm_source.last_config.max_samples == 1

    assert baselines.unicode[0].metadata["total_chars"] == len("A")
    assert baselines.unicode[1].metadata["total_chars"] == len("XX")


def test_build_default_registry_baselines_uses_custom_registry():
    registry = SourceRegistry()
    human_record, human_factory = make_counting_factory("human", is_llm=False, payloads=["HH"])
    llm_record, llm_factory = make_counting_factory("llm", is_llm=True, payloads=["LLL"])
    registry.register(
        SourceDefinition(name="human", category=SourceCategory.HUMAN, factory=human_factory)
    )
    registry.register(
        SourceDefinition(name="llm", category=SourceCategory.LLM, factory=llm_factory)
    )

    baselines = build_default_registry_baselines(
        samples_per_source=1, registry=registry, version="custom"
    )
    assert human_record["count"] == 1
    assert llm_record["count"] == 1
    assert baselines.unicode[0].version == "custom"
    assert baselines.unicode[1].version == "custom"


def test_compute_baselines_to_path_respects_cache(tmp_path: Path):
    registry = SourceRegistry()
    human_record, human_factory = make_counting_factory("human", is_llm=False, payloads=["A", "BB"])
    llm_record, llm_factory = make_counting_factory("llm", is_llm=True, payloads=["XX", "YYY"])
    registry.register(
        SourceDefinition(name="human", category=SourceCategory.HUMAN, factory=human_factory)
    )
    registry.register(
        SourceDefinition(name="llm", category=SourceCategory.LLM, factory=llm_factory)
    )

    output = tmp_path / "baselines.json.gz"

    first = compute_baselines_to_path(
        output,
        samples_per_source=1,
        registry=registry,
    )
    assert output.exists()
    assert human_record["count"] == 1
    assert llm_record["count"] == 1

    second = compute_baselines_to_path(
        output,
        samples_per_source=2,
        registry=registry,
    )
    assert second.unicode[0].metadata == first.unicode[0].metadata
    assert human_record["count"] == 1  # cached, factories not invoked

    third = compute_baselines_to_path(
        output,
        samples_per_source=2,
        registry=registry,
        overwrite=True,
    )
    assert human_record["count"] == 2
    assert llm_record["count"] == 2
    assert third.unicode[0].metadata["total_chars"] == len("A") + len("BB")
