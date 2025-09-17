from __future__ import annotations

import pytest

from llm_detector.features import register_default_features
from llm_detector.features.vectorizer import FeatureVectorizer
from llm_detector.training import SourceCategory, SourceDefinition, SourceRegistry
from llm_detector.training.dataset import (
    FeatureDataset,
    build_dataset_from_registry,
    build_feature_dataset,
)
from llm_detector.training.sources.base import BaseDataSource, BatchConfig
from llm_detector.types import TextSample


class _StubSource(BaseDataSource):
    def __init__(self, name: str, texts: list[str], *, is_llm: bool) -> None:
        super().__init__(BatchConfig())
        self._name = name
        self._texts = texts
        self._is_llm = is_llm

    @property
    def name(self) -> str:
        return self._name

    def _generate_samples(self):
        for idx, text in enumerate(self._texts):
            yield TextSample(
                text=text,
                is_llm=self._is_llm,
                source=self._name,
                sample_id=f"{self._name}_{idx}",
            )


def _make_registry() -> SourceRegistry:
    registry = SourceRegistry()
    human_texts = ["Human text one.", "Another person wrote this."]
    llm_texts = ["Synthetic response 1.", "Synthetic response 2.", "Synthetic response 3."]

    registry.register(
        SourceDefinition(
            name="stub_human",
            category=SourceCategory.HUMAN,
            factory=lambda: _StubSource("stub_human", human_texts, is_llm=False),
        )
    )
    registry.register(
        SourceDefinition(
            name="stub_llm",
            category=SourceCategory.LLM,
            factory=lambda: _StubSource("stub_llm", llm_texts, is_llm=True),
        )
    )
    return registry


def _vectorizer(feature_names: list[str]) -> FeatureVectorizer:
    registry = register_default_features()
    return FeatureVectorizer(registry, feature_names=feature_names)


def test_build_feature_dataset_roundtrip():
    vectorizer = _vectorizer(["stat.mean_word_length"])
    samples = [
        TextSample(text="hello world", is_llm=False, source="stub"),
        TextSample(text="synthetic reply", is_llm=True, source="stub"),
    ]

    dataset = build_feature_dataset(vectorizer, samples, keep_text=True)

    assert dataset.feature_names == ["stat.mean_word_length"]
    assert len(dataset.matrix) == 2
    assert dataset.labels == [0, 1]
    assert dataset.texts == [sample.text for sample in samples]


def test_build_dataset_from_registry_balances_and_shuffles():
    registry = _make_registry()
    vectorizer = _vectorizer(["stat.mean_sentence_length"])

    dataset = build_dataset_from_registry(
        registry,
        vectorizer,
        shuffle=True,
        seed=123,
        keep_text=True,
    )

    assert len(dataset.matrix) == 4  # balanced to two per class
    assert sum(dataset.labels) == 2
    assert dataset.texts is not None
    assert sorted(dataset.sources) == ["stub_human", "stub_human", "stub_llm", "stub_llm"]


def test_feature_dataset_split_creates_partitions():
    vectorizer = _vectorizer(["stat.mean_word_length"])
    samples = [
        TextSample(text=f"sample {i}", is_llm=(i % 2 == 1), source="stub") for i in range(10)
    ]
    dataset = build_feature_dataset(vectorizer, samples)

    train, test = dataset.split(0.2, seed=42)

    assert len(train.matrix) + len(test.matrix) == len(dataset.matrix)
    assert len(test.matrix) == 2  # 20% of 10 -> max(1,2)


def test_feature_dataset_split_validation():
    dataset = FeatureDataset(feature_names=[], matrix=[], labels=[], sources=[])

    with pytest.raises(ValueError):
        dataset.split(0.2)

    vectorizer = _vectorizer(["stat.mean_word_length"])
    dataset = build_feature_dataset(vectorizer, [])

    with pytest.raises(ValueError):
        dataset.split(0.0)
