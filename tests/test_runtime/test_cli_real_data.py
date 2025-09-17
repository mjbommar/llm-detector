from __future__ import annotations

from pathlib import Path

import pytest

from llm_detector import cli
from llm_detector.training.sources.human import GutenbergSource


@pytest.mark.integration
@pytest.mark.requires_datasets
def test_cli_with_gutenberg_sample(tmp_path: Path):
    source = GutenbergSource(min_sentence_length=5)
    iterator = iter(source)
    try:
        text = next(iterator).text
    except Exception as exc:  # pragma: no cover - network dependent
        pytest.skip(f"dataset unavailable: {exc}")

    args = ["--text", text, "--json"]
    assert cli.main(args) == 0
