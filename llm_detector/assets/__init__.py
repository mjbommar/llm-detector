"""Bundled runtime artifacts for the detector."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from importlib import resources
from pathlib import Path
from typing import Any

_PACKAGE = __name__
_MODEL_CANDIDATES = ("default_model.json.gz", "default_model.joblib")
_BASELINE_FILE = "default_baselines.json.gz"


def _get_resource(name: str):
    try:
        ref = resources.files(_PACKAGE).joinpath(name)
    except AttributeError:  # pragma: no cover - very old Python
        return None
    if not ref.is_file():
        return None
    return ref


def _resolve_model_resource() -> tuple[str | None, Any | None]:
    for candidate in _MODEL_CANDIDATES:
        ref = _get_resource(candidate)
        if ref is not None:
            return candidate, ref
    return None, None


def has_default_artifacts() -> bool:
    _, model_ref = _resolve_model_resource()
    return model_ref is not None and _get_resource(_BASELINE_FILE) is not None


@contextmanager
def default_artifacts() -> Iterator[tuple[Path, Path] | None]:
    """Yield filesystem paths to bundled model/baselines if available."""

    _, model_ref = _resolve_model_resource()
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
