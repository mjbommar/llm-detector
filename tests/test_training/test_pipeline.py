from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sklearn")

from llm_detector.training.pipeline import TrainingArtifacts, train_logistic_from_registry
from llm_detector.training.sources.base import BaseDataSource, BatchConfig
from llm_detector.training.sources.registry import SourceCategory, SourceDefinition, SourceRegistry
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


def _registry() -> SourceRegistry:
    registry = SourceRegistry()
    registry.register(
        SourceDefinition(
            name="stub_human",
            category=SourceCategory.HUMAN,
            factory=lambda: _StubSource(
                "stub_human",
                ["Human text one.", "Another human text."],
                is_llm=False,
            ),
        )
    )
    registry.register(
        SourceDefinition(
            name="stub_llm",
            category=SourceCategory.LLM,
            factory=lambda: _StubSource(
                "stub_llm",
                ["Synthetic answer 1.", "Synthetic answer 2."],
                is_llm=True,
            ),
        )
    )
    return registry


def test_train_logistic_from_registry_creates_artifacts(tmp_path: Path):
    model_path = tmp_path / "model.json.gz"
    baseline_path = tmp_path / "baselines.json.gz"

    artifacts = train_logistic_from_registry(
        model_path=model_path,
        baseline_path=baseline_path,
        registry=_registry(),
        samples_per_source=2,
        test_ratio=None,
        seed=123,
    )

    assert isinstance(artifacts, TrainingArtifacts)
    assert model_path.exists()
    assert baseline_path.exists()
    assert artifacts.train_accuracy >= 0.5
    assert "train_f1" in artifacts.metrics
    assert artifacts.feature_names


def test_cli_invocation(monkeypatch, tmp_path: Path, capsys):
    calls: dict[str, object] = {}

    def _fake_train(**kwargs):
        calls["kwargs"] = kwargs
        model_path = kwargs["model_path"]
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(model_path).write_bytes(b"data")
        return TrainingArtifacts(
            model_path=Path(model_path),
            baselines_path=None,
            train_accuracy=0.9,
            metrics={"train_f1": 0.9},
            feature_names=["feat1", "feat2"],
        )

    monkeypatch.setattr("llm_detector.training.cli.train_logistic_from_registry", _fake_train)

    model_path = tmp_path / "model.json.gz"
    args = [
        "--model-path",
        str(model_path),
        "--samples-per-source",
        "10",
        "--test-ratio",
        "0",
    ]

    from llm_detector.training import cli

    assert cli.main(args) == 0
    captured = capsys.readouterr()
    assert "train accuracy: 0.9000" in captured.out
    assert calls["kwargs"]["samples_per_source"] == 10
    assert calls["kwargs"]["test_ratio"] is None
    assert calls["kwargs"]["show_progress"] is False
