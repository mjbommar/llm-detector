import pytest

from llm_detector.baselines import (
    BaselineArtifact,
    BaselineSet,
    divergence_baseline_overrides,
)


def _artifact(vocabulary, distribution, version="test"):
    return BaselineArtifact(
        distribution=list(distribution),
        vocabulary=list(vocabulary),
        metadata={},
        version=version,
    )


def test_baseline_artifact_as_mapping_roundtrip():
    artifact = _artifact(["a", "b"], [0.25, 0.75])
    mapping = artifact.as_mapping()
    assert mapping == {"a": 0.25, "b": 0.75}


def test_divergence_baseline_overrides_extracts_expected_slice():
    unicode_artifact = _artifact(["a"], [1.0])
    regex_artifact = _artifact(["word"], [1.0])
    punct_artifact = _artifact(["."], [1.0])
    baselines = BaselineSet(
        unicode=(unicode_artifact, unicode_artifact),
        regex=(regex_artifact, regex_artifact),
        punctws=(punct_artifact, punct_artifact),
    )

    overrides = divergence_baseline_overrides(baselines, cohort="llm")

    assert overrides["div.char_jsd"] == {"a": 1.0}
    assert overrides["div.regex_jsd"] == {"word": 1.0}
    assert overrides["div.punct_jsd"] == {".": 1.0}

    with pytest.raises(ValueError):
        divergence_baseline_overrides(baselines, cohort="unknown")
