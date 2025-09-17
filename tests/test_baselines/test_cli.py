"""Tests for the baseline CLI interface."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_detector.baselines import BaselineArtifact, BaselineSet
from llm_detector.baselines.cli import main as cli_main


def test_cli_invokes_builder(monkeypatch, tmp_path):
    called: dict[str, object] = {}

    def fake_compute(path: Path, **kwargs) -> BaselineSet:
        called["path"] = path
        called["kwargs"] = kwargs
        empty = BaselineArtifact(distribution=[], vocabulary=[], metadata={}, version="test")
        return BaselineSet(unicode=(empty, empty), regex=(empty, empty), punctws=(empty, empty))

    monkeypatch.setattr("llm_detector.baselines.cli.compute_baselines_to_path", fake_compute)

    output = tmp_path / "out.json.gz"
    exit_code = cli_main(
        [
            "--output",
            str(output),
            "--samples-per-source",
            "25",
            "--version",
            "v9",
            "--include-disabled",
            "--overwrite",
        ]
    )

    assert exit_code == 0
    assert called["path"] == output
    kwargs = called["kwargs"]
    assert kwargs["samples_per_source"] == 25
    assert kwargs["version"] == "v9"
    assert kwargs["enabled_only"] is False
    assert kwargs["overwrite"] is True


def test_cli_rejects_negative_sample_limits():
    with pytest.raises(SystemExit):
        cli_main(["--output", "baseline.json.gz", "--samples-per-source", "-5"])
