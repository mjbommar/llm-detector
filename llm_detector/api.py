"""Highly simplified public API for quick text classification."""

from __future__ import annotations

import atexit
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from . import assets
from .runtime import DetectionResult, DetectorRuntime

_RuntimeKey = tuple[Path | None, Path | None]
_RUNTIME_CACHE: dict[_RuntimeKey, _CacheEntry] = {}
_DEFAULT_CONTEXT = None
_DEFAULT_BUNDLE: tuple[Path, Path] | None = None
_CACHE_LOCK = RLock()


@dataclass(slots=True)
class _CacheEntry:
    runtime: DetectorRuntime
    model_path: Path
    baseline_path: Path
    model_sig: tuple[int, int] | None
    baseline_sig: tuple[int, int] | None


def _normalize_path(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    if not isinstance(path, Path):
        path = Path(path)
    return path.resolve()


def _file_signature(path: Path | None) -> tuple[int, int] | None:
    if path is None:
        return None
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def _entry_is_fresh(entry: _CacheEntry) -> bool:
    try:
        model_sig = _file_signature(entry.model_path)
        baseline_sig = _file_signature(entry.baseline_path)
    except FileNotFoundError:
        return False
    return model_sig == entry.model_sig and baseline_sig == entry.baseline_sig


def _ensure_default_bundle() -> tuple[Path, Path]:
    global _DEFAULT_CONTEXT, _DEFAULT_BUNDLE

    with _CACHE_LOCK:
        if _DEFAULT_BUNDLE is not None:
            return _DEFAULT_BUNDLE

    ctx = assets.default_artifacts()
    bundle = ctx.__enter__()
    if bundle is None:
        ctx.__exit__(None, None, None)
        raise RuntimeError(
            "Bundled artifacts unavailable; supply --model/--baselines or install extras."
        )

    model_path, baseline_path = (path.resolve() for path in bundle)
    resolved: tuple[Path, Path] = (model_path, baseline_path)

    with _CACHE_LOCK:
        _DEFAULT_CONTEXT = ctx
        _DEFAULT_BUNDLE = resolved

    return resolved


def _get_runtime(
    model_path: Path | str | None,
    baseline_path: Path | str | None,
) -> DetectorRuntime:
    normalized_model = _normalize_path(model_path)
    normalized_baseline = _normalize_path(baseline_path)
    if normalized_model is None or normalized_baseline is None:
        normalized_model, normalized_baseline = _ensure_default_bundle()

    key: _RuntimeKey = (normalized_model, normalized_baseline)

    with _CACHE_LOCK:
        entry = _RUNTIME_CACHE.get(key)
        if entry and _entry_is_fresh(entry):
            return entry.runtime
        if entry is not None:
            _RUNTIME_CACHE.pop(key, None)

    runtime = DetectorRuntime(
        model_path=normalized_model,
        baseline_path=normalized_baseline,
    )

    try:
        model_sig = _file_signature(normalized_model)
        baseline_sig = _file_signature(normalized_baseline)
    except FileNotFoundError as exc:
        raise RuntimeError(f"artifact missing during runtime initialisation: {exc}") from exc

    entry = _CacheEntry(
        runtime=runtime,
        model_path=normalized_model,
        baseline_path=normalized_baseline,
        model_sig=model_sig,
        baseline_sig=baseline_sig,
    )

    with _CACHE_LOCK:
        _RUNTIME_CACHE[key] = entry

    return runtime


def _close_default_context() -> None:
    global _DEFAULT_CONTEXT, _DEFAULT_BUNDLE
    with _CACHE_LOCK:
        context = _DEFAULT_CONTEXT
        _DEFAULT_CONTEXT = None
        _DEFAULT_BUNDLE = None
    if context is not None:
        context.__exit__(None, None, None)


atexit.register(_close_default_context)


def clear_runtime_cache(*, close_artifacts: bool = True) -> None:
    """Reset cached runtimes. Optionally close bundled artifact handles."""

    with _CACHE_LOCK:
        _RUNTIME_CACHE.clear()
    if close_artifacts:
        _close_default_context()


def classify_text(
    text: str,
    *,
    model_path: Path | str | None = None,
    baseline_path: Path | str | None = None,
    include_diagnostics: bool = False,
    return_result: bool = False,
) -> dict[str, Any] | DetectionResult:
    """Classify ``text`` using bundled or user-specified artifacts.

    Args:
        text: Sample to classify.
        model_path: Optional path to a trained ``.json.gz`` logistic model (or legacy
            ``.joblib`` file).
        baseline_path: Optional path to the baseline JSON.gz matching the model.
        include_diagnostics: When ``True`` and returning a dictionary, include
            secondary aggregation metrics (simple_mean, max_score, vote_fraction).
        return_result: If ``True`` return the full :class:`DetectionResult` instead
            of a simplified dictionary.

    Returns:
        Either a :class:`DetectionResult` or a compact dictionary with
        ``is_llm``, ``p_llm``, ``confidence``, and optional ``diagnostics``.
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")

    runtime = _get_runtime(model_path, baseline_path)
    result = runtime.predict(text)

    if return_result:
        return result

    payload: dict[str, Any] = {
        "is_llm": result.is_llm,
        "p_llm": result.p_llm,
        "confidence": result.confidence,
    }

    if include_diagnostics and result.details:
        metrics = result.details.get("document_metrics", {})
        diagnostics = {
            name: metrics.get(name)
            for name in result.details.get("diagnostic_metrics", [])
            if name in metrics
        }
        payload["diagnostics"] = diagnostics

    return payload


__all__ = ["classify_text", "clear_runtime_cache"]
