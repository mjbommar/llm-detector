"""Bundled runtime artifacts for the detector."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
from pathlib import Path

_PACKAGE = __name__
_MODEL_FILE = "default_model.joblib"
_BASELINE_FILE = "default_baselines.json.gz"


def _get_resource(name: str):
    try:
        ref = resources.files(_PACKAGE).joinpath(name)
    except AttributeError:  # pragma: no cover - very old Python
        return None
    if not ref.is_file():
        return None
    return ref


def has_default_artifacts() -> bool:
    return _get_resource(_MODEL_FILE) is not None and _get_resource(_BASELINE_FILE) is not None


@contextmanager
def default_artifacts() -> Iterator[tuple[Path, Path] | None]:
    """Yield filesystem paths to bundled model/baselines if available."""

    model_ref = _get_resource(_MODEL_FILE)
    baseline_ref = _get_resource(_BASELINE_FILE)
    if model_ref is None or baseline_ref is None:
        yield None
        return

    with (
        resources.as_file(model_ref) as model_path,
        resources.as_file(baseline_ref) as baseline_path,
    ):
        yield Path(model_path), Path(baseline_path)


__all__ = ["has_default_artifacts", "default_artifacts"]
