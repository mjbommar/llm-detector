import pytest

from llm_detector.baselines import BaselineArtifact, BaselineSet, divergence_baseline_overrides
from llm_detector.features import divergence, register_default_features
from llm_detector.features.vectorizer import FeatureVectorizer


def _baseline_set() -> BaselineSet:
    unicode_artifact = BaselineArtifact(
        distribution=[1.0],
        vocabulary=["a"],
        metadata={},
        version="test",
    )
    regex_artifact = BaselineArtifact(
        distribution=[1.0],
        vocabulary=["token"],
        metadata={},
        version="test",
    )
    punct_artifact = BaselineArtifact(
        distribution=[1.0],
        vocabulary=["."],
        metadata={},
        version="test",
    )
    return BaselineSet(
        unicode=(unicode_artifact, unicode_artifact),
        regex=(regex_artifact, regex_artifact),
        punctws=(punct_artifact, punct_artifact),
    )


def test_vectorizer_respects_baseline_override():
    registry = register_default_features()
    overrides = divergence_baseline_overrides(_baseline_set(), cohort="human")

    vectorizer = FeatureVectorizer(
        registry,
        feature_names=["div.char_jsd"],
        baseline_overrides=overrides,
    )

    text = "bbbb"
    expected = divergence.char_divergence_score(text, baseline=overrides["div.char_jsd"])
    computed = vectorizer.compute(text)["div.char_jsd"]

    assert computed == pytest.approx(expected)


def test_vectorize_returns_ordered_values():
    registry = register_default_features()
    feature_names = [
        "stat.mean_sentence_length",
        "stat.type_token_ratio",
    ]
    vectorizer = FeatureVectorizer(registry, feature_names=feature_names)

    vector = vectorizer.vectorize("Hello world. This is a test.")

    assert vector.names == sorted(feature_names)
    assert len(vector.values) == len(vector.names)
    assert all(name in vector.as_dict() for name in vector.names)


def test_transform_to_matrix_batches_inputs():
    registry = register_default_features()
    vectorizer = FeatureVectorizer(
        registry,
        feature_names=["stat.mean_word_length"],
    )

    texts = ["hello world", "aa bb"]
    matrix = vectorizer.transform_to_matrix(texts)

    assert len(matrix) == 2
    assert matrix[0] != matrix[1]
