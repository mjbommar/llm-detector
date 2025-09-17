from __future__ import annotations

import json

import pytest

from llm_detector.baselines import (
    BaselineArtifact,
    BaselineCache,
    BaselineSet,
    divergence_baseline_overrides,
)
from llm_detector.features import FeatureVectorizer, register_default_features
from llm_detector.models import LogisticRegressionModel
from llm_detector.training.dataset import build_feature_dataset
from llm_detector.types import TextSample


@pytest.fixture(scope="module")
def _artifacts(tmp_path_factory: pytest.TempPathFactory):
    tmp_dir = tmp_path_factory.mktemp("runtime_cli")

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
    baseline_path = tmp_dir / "baselines.json.gz"
    BaselineCache.save(baselines, baseline_path)

    registry = register_default_features()
    overrides = divergence_baseline_overrides(baselines, cohort="human")
    vectorizer = FeatureVectorizer(
        registry,
        feature_names=["stat.mean_word_length"],
        baseline_overrides=overrides,
    )

    samples = [
        TextSample(text="Human authored content.", is_llm=False, source="human"),
        TextSample(text="Second human text.", is_llm=False, source="human"),
        TextSample(text="Synthetic reply from AI.", is_llm=True, source="llm"),
        TextSample(text="Another AI generated reply.", is_llm=True, source="llm"),
    ]
    dataset = build_feature_dataset(vectorizer, samples)
    model = LogisticRegressionModel()
    model.fit(dataset)
    model_path = tmp_dir / "model.joblib"
    model.save(model_path)

    return model_path, baseline_path


def test_runtime_cli_human_output(_artifacts, capsys):
    model_path, baseline_path = _artifacts
    args = [
        "--model",
        str(model_path),
        "--baselines",
        str(baseline_path),
        "--text",
        "Short test sentence",
        "--show-diagnostics",
    ]

    from llm_detector import cli

    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert "p_llm=" in out
    assert "Short test sentence" in out
    assert "[logit_weighted_mean]" in out
    assert "diagnostics:" in out


def test_runtime_cli_json_output(_artifacts, capsys):
    model_path, baseline_path = _artifacts
    args = [
        "--model",
        str(model_path),
        "--baselines",
        str(baseline_path),
        "--text",
        "Another test sentence",
        "--json",
        "--return-features",
    ]

    from llm_detector import cli

    assert cli.main(args) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert len(payload) == 1
    record = payload[0]
    assert set(record.keys()) >= {
        "is_llm",
        "p_llm",
        "p_human",
        "confidence",
        "text",
        "features",
        "details",
        "aggregators",
        "primary_metric",
    }
    assert isinstance(record["features"], dict)
    assert isinstance(record["details"], dict)
    assert "sentences" in record["details"]
    assert record["primary_metric"] == "logit_weighted_mean"
    assert isinstance(record.get("aggregators"), dict)


def test_runtime_cli_defaults(monkeypatch, _artifacts, capsys):
    model_path, baseline_path = _artifacts

    class _Context:
        def __enter__(self):
            return model_path, baseline_path

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("llm_detector.cli.assets.default_artifacts", lambda: _Context())

    from llm_detector import cli

    args = ["--text", "Default artifact run"]
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    assert "Default artifact run" in out
