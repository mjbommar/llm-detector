from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sklearn")

from llm_detector.baselines import (
    BaselineArtifact,
    BaselineCache,
    BaselineSet,
    divergence_baseline_overrides,
)
from llm_detector.features import FeatureVectorizer, register_default_features
from llm_detector.models import LogisticRegressionModel
from llm_detector.runtime import DetectionResult, DetectorRuntime
from llm_detector.training.dataset import build_feature_dataset
from llm_detector.types import TextSample


def _baseline_set() -> BaselineSet:
    artifact = BaselineArtifact(
        distribution=[1.0],
        vocabulary=["a"],
        metadata={},
        version="test",
    )
    return BaselineSet(
        unicode=(artifact, artifact),
        regex=(artifact, artifact),
        punctws=(artifact, artifact),
    )


def _train_model(tmp_path: Path, baselines: BaselineSet) -> Path:
    registry = register_default_features()
    overrides = divergence_baseline_overrides(baselines, cohort="human")
    vectorizer = FeatureVectorizer(
        registry,
        feature_names=["stat.mean_word_length"],
        baseline_overrides=overrides,
    )

    samples = [
        TextSample(text="Human authored sentence.", is_llm=False, source="human"),
        TextSample(text="Another human text sample.", is_llm=False, source="human"),
        TextSample(text="Synthetic machine reply sequence.", is_llm=True, source="llm"),
        TextSample(text="AI written response example.", is_llm=True, source="llm"),
    ]

    dataset = build_feature_dataset(vectorizer, samples)
    model = LogisticRegressionModel()
    model.fit(dataset)

    model_path = tmp_path / "model.json.gz"
    model.save(model_path)
    return model_path


def test_detector_runtime_predicts_and_returns_features(tmp_path: Path):
    baselines = _baseline_set()
    baseline_path = tmp_path / "baselines.json.gz"
    BaselineCache.save(baselines, baseline_path)

    model_path = _train_model(tmp_path, baselines)

    runtime = DetectorRuntime(
        model_path=model_path,
        baseline_path=baseline_path,
        return_features=True,
    )

    result = runtime.predict("A short example sentence.")

    assert isinstance(result, DetectionResult)
    assert 0.0 <= result.p_llm <= 1.0
    assert 0.0 <= result.p_human <= 1.0
    assert abs(result.p_llm + result.p_human - 1.0) < 1e-6
    assert result.features is not None
    assert "stat.mean_word_length" in result.features
    assert result.details is not None
    assert "sentences" in result.details
    assert result.details["sentence_count"] == len(result.details["sentences"])
    assert result.details.get("primary_metric") == "logit_weighted_mean"
    metrics = result.details.get("document_metrics")
    assert isinstance(metrics, dict) and "logit_weighted_mean" in metrics
    assert set(result.details.get("diagnostic_metrics", [])) >= {
        "simple_mean",
        "max_score",
        "vote_fraction",
    }

    batch_results = list(runtime.predict_stream(["First", "Second"]))
    assert len(batch_results) == 2
    assert all(isinstance(item, DetectionResult) for item in batch_results)
