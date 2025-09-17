from __future__ import annotations

from pathlib import Path

from llm_detector.api import classify_text
from llm_detector.baselines import (
    BaselineArtifact,
    BaselineCache,
    BaselineSet,
    divergence_baseline_overrides,
)
from llm_detector.features import FeatureVectorizer, register_default_features
from llm_detector.models import LogisticRegressionModel
from llm_detector.runtime import DetectionResult
from llm_detector.training.dataset import build_feature_dataset
from llm_detector.types import TextSample


def _prepare_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    artifact = BaselineArtifact(
        distribution=[1.0],
        vocabulary=["a"],
        metadata={},
        version="test",
    )
    baselines = BaselineSet(
        unicode=(artifact, artifact),
        regex=(artifact, artifact),
        punctws=(artifact, artifact),
    )
    baseline_path = tmp_path / "baselines.json.gz"
    BaselineCache.save(baselines, baseline_path)

    registry = register_default_features()
    overrides = divergence_baseline_overrides(baselines, cohort="human")
    vectorizer = FeatureVectorizer(
        registry,
        feature_names=["stat.mean_word_length"],
        baseline_overrides=overrides,
    )
    samples = [
        TextSample(text="Human sample.", is_llm=False, source="human"),
        TextSample(text="Another person.", is_llm=False, source="human"),
        TextSample(text="Robot reply.", is_llm=True, source="llm"),
        TextSample(text="Synthetic text.", is_llm=True, source="llm"),
    ]
    dataset = build_feature_dataset(vectorizer, samples)
    model = LogisticRegressionModel()
    model.fit(dataset)
    model_path = tmp_path / "model.joblib"
    model.save(model_path)
    return model_path, baseline_path


def test_classify_text_returns_dict(tmp_path):
    model_path, baseline_path = _prepare_artifacts(tmp_path)
    result = classify_text(
        "A tiny snippet.",
        model_path=model_path,
        baseline_path=baseline_path,
        include_diagnostics=True,
    )
    assert set(result) >= {"is_llm", "p_llm", "confidence"}
    diagnostics = result.get("diagnostics")
    assert diagnostics is not None
    assert "simple_mean" in diagnostics


def test_classify_text_return_detection_result(tmp_path):
    model_path, baseline_path = _prepare_artifacts(tmp_path)
    result = classify_text(
        "Another snippet.",
        model_path=model_path,
        baseline_path=baseline_path,
        return_result=True,
    )
    assert isinstance(result, DetectionResult)
