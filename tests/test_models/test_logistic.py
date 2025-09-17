from __future__ import annotations

from pathlib import Path

from llm_detector.models import LogisticRegressionModel
from llm_detector.training.dataset import FeatureDataset


def _simple_dataset() -> FeatureDataset:
    matrix = [[0.0], [0.25], [0.5], [0.75], [1.0]]
    labels = [0, 0, 0, 1, 1]
    sources = ["human", "human", "human", "llm", "llm"]
    return FeatureDataset(
        feature_names=["feature"],
        matrix=matrix,
        labels=labels,
        sources=sources,
    )


def test_logistic_model_trains_and_predicts(tmp_path: Path):
    dataset = _simple_dataset()
    model = LogisticRegressionModel()

    result = model.fit(dataset)

    assert result.train_accuracy >= 0.6
    assert "train_f1" in result.metrics

    probs = model.predict_proba([0.9])
    assert probs[1] > probs[0]

    model_path = tmp_path / "model.joblib"
    model.save(model_path)

    loaded = LogisticRegressionModel.load(model_path)
    assert loaded.feature_names == ["feature"]

    evaluation = loaded.evaluate(dataset)
    assert evaluation["accuracy"] >= 0.6


def test_logistic_model_rejects_incorrect_shapes():
    dataset = _simple_dataset()
    model = LogisticRegressionModel()
    model.fit(dataset)

    try:
        model.predict_proba([0.1, 0.2])
    except ValueError:
        pass
    else:  # pragma: no cover - defensive branch
        raise AssertionError("Expected ValueError for mismatched feature length")

    empty_dataset = FeatureDataset(feature_names=["feature"], matrix=[], labels=[], sources=[])
    try:
        model.evaluate(empty_dataset)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for empty evaluation dataset")
