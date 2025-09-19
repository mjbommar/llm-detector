from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from llm_detector import api
from llm_detector.api import classify_text, clear_runtime_cache


@pytest.fixture(autouse=True)
def _reset_runtime_cache() -> Iterator[None]:
    clear_runtime_cache()
    yield
    clear_runtime_cache()


def _copy_artifacts(source_model: Path, source_baseline: Path, dest_dir: Path) -> tuple[Path, Path]:
    model_copy = dest_dir / source_model.name
    baseline_copy = dest_dir / source_baseline.name
    shutil.copy(source_model, model_copy)
    shutil.copy(source_baseline, baseline_copy)
    return model_copy, baseline_copy


def test_runtime_cache_reused_for_same_artifacts(tmp_path: Path, packaged_model) -> None:
    model_src, baseline_src = packaged_model
    model_copy, baseline_copy = _copy_artifacts(model_src, baseline_src, tmp_path)

    text = "This is a quick sentence to classify."
    first = classify_text(text, model_path=model_copy, baseline_path=baseline_copy)
    assert "p_llm" in first

    cache_key = (model_copy.resolve(), baseline_copy.resolve())
    assert cache_key in api._RUNTIME_CACHE
    runtime1 = api._RUNTIME_CACHE[cache_key].runtime

    second = classify_text(text, model_path=model_copy, baseline_path=baseline_copy)
    assert second["p_llm"] == pytest.approx(first["p_llm"])
    runtime2 = api._RUNTIME_CACHE[cache_key].runtime
    assert runtime2 is runtime1


def test_runtime_cache_invalidated_when_artifact_changes(tmp_path: Path, packaged_model) -> None:
    model_src, baseline_src = packaged_model
    model_copy, baseline_copy = _copy_artifacts(model_src, baseline_src, tmp_path)

    classify_text("First example sentence.", model_path=model_copy, baseline_path=baseline_copy)
    cache_key = (model_copy.resolve(), baseline_copy.resolve())
    runtime1 = api._RUNTIME_CACHE[cache_key].runtime

    stat = baseline_copy.stat()
    os.utime(baseline_copy, (stat.st_atime, stat.st_mtime + 5))

    classify_text(
        "Second sample sentence that should refresh.",
        model_path=model_copy,
        baseline_path=baseline_copy,
    )
    runtime2 = api._RUNTIME_CACHE[cache_key].runtime

    assert runtime2 is not runtime1
