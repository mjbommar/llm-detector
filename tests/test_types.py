from __future__ import annotations

import pytest

from llm_detector.types import TextSample


def test_text_sample_validation_success() -> None:
    sample = TextSample(text="Valid sample", is_llm=False, source="dummy")
    assert sample.metadata == {}


def test_text_sample_requires_text() -> None:
    with pytest.raises(ValueError):
        TextSample(text="", is_llm=False, source="dummy")


def test_text_sample_requires_source() -> None:
    with pytest.raises(ValueError):
        TextSample(text="text", is_llm=False, source="")


def test_text_sample_metadata_must_be_mapping() -> None:
    with pytest.raises(TypeError):
        TextSample(text="text", is_llm=False, source="dummy", metadata=[])  # type: ignore[arg-type]
