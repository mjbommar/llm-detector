"""Bridge feature registry definitions to ordered numeric vectors."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass

from .registry import FeatureDefinition, FeatureRegistry


def _to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return 0.0


@dataclass(slots=True)
class FeatureVector:
    """Container for an ordered set of feature values."""

    names: list[str]
    values: list[float]

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values, strict=False))


class FeatureVectorizer:
    """Convert texts into numeric feature vectors using a registry."""

    def __init__(
        self,
        registry: FeatureRegistry,
        *,
        feature_names: Sequence[str] | None = None,
        scale_invariant_only: bool = True,
        enabled_only: bool = True,
        baseline_overrides: Mapping[str, Mapping[str, float]] | None = None,
    ) -> None:
        self.registry = registry
        self.scale_invariant_only = scale_invariant_only
        self.enabled_only = enabled_only
        self._baseline_overrides = dict(baseline_overrides or {})

        if feature_names is None:
            definitions = registry.all(
                scale_invariant_only=scale_invariant_only,
                enabled_only=enabled_only,
            )
        else:
            definitions = []
            for name in feature_names:
                definition = registry.get(name)
                if definition is None:
                    raise KeyError(f"feature '{name}' is not registered")
                if enabled_only and not definition.enabled:
                    raise ValueError(f"feature '{name}' is disabled in the registry")
                if scale_invariant_only and not definition.scale_invariant:
                    raise ValueError(f"feature '{name}' is not scale invariant")
                definitions.append(definition)

        self._feature_defs: list[FeatureDefinition] = sorted(
            definitions, key=lambda item: item.name
        )
        self._feature_names: list[str] = [definition.name for definition in self._feature_defs]
        self._computers: dict[str, Callable[[str], float]] = {
            definition.name: self._bind_compute(definition) for definition in self._feature_defs
        }

    def _bind_compute(self, definition: FeatureDefinition) -> Callable[[str], float]:
        compute_fn = definition.compute_fn
        baseline = self._baseline_overrides.get(definition.name)

        if baseline is None:

            def _compute(text: str, fn: Callable[[str], float] = compute_fn) -> float:
                return _to_float(fn(text))

            return _compute

        def _compute_with_baseline(
            text: str,
            fn: Callable[[str], float] = compute_fn,
            base: Mapping[str, float] = baseline,
        ) -> float:
            try:
                return _to_float(fn(text, baseline=base))  # type: ignore[misc]
            except TypeError:
                return _to_float(fn(text))

        return _compute_with_baseline

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def compute(self, text: str) -> dict[str, float]:
        if not text:
            return {name: 0.0 for name in self._feature_names}
        return {name: computer(text) for name, computer in self._computers.items()}

    def vectorize(self, text: str) -> FeatureVector:
        mapping = self.compute(text)
        values = [mapping[name] for name in self._feature_names]
        return FeatureVector(names=self.feature_names, values=values)

    def transform(self, texts: Iterable[str]) -> Iterator[FeatureVector]:
        for text in texts:
            yield self.vectorize(text)

    def transform_to_matrix(self, texts: Iterable[str]) -> list[list[float]]:
        return [vector.values for vector in self.transform(texts)]


__all__ = ["FeatureVector", "FeatureVectorizer"]
