from __future__ import annotations

from pathlib import Path

import pytest

from llm_detector.assets import default_artifacts


@pytest.fixture(scope="session")
def packaged_model(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    with default_artifacts() as bundle:
        assert bundle is not None, "Bundled assets missing"
        model_path, baseline_path = bundle
        tmp_dir = tmp_path_factory.mktemp("runtime_model")
        model_copy = tmp_dir / "model.joblib"
        baseline_copy = tmp_dir / "baselines.json.gz"
        model_copy.write_bytes(model_path.read_bytes())
        baseline_copy.write_bytes(baseline_path.read_bytes())
        return model_copy, baseline_copy
