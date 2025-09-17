"""Utilities for aggregating sentence-level predictions."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

Probability = float

_SENTENCE_THRESHOLD = 0.7
_EPSILON = 1e-6


def _safe_probability(value: float) -> float:
    return min(max(value, _EPSILON), 1.0 - _EPSILON)


def _normalise_weights(weights: Sequence[float] | None, count: int) -> Sequence[float]:
    if weights is None:
        return [1.0] * count
    if len(weights) != count:
        raise ValueError("weights length must match scores length")
    return weights


def mean(scores: Sequence[float], weights: Sequence[float] | None = None) -> Probability:
    if not scores:
        return 0.5
    weights = _normalise_weights(weights, len(scores))
    total = float(sum(weights))
    if total <= 0.0:
        return 0.5
    weighted = sum(score * weight for score, weight in zip(scores, weights, strict=False))
    return weighted / total


def length_weighted_mean(
    scores: Sequence[float], lengths: Sequence[int] | None = None
) -> Probability:
    if not scores:
        return 0.5
    if lengths is None:
        lengths = [1.0] * len(scores)
    weights = [float(max(1, int(length))) for length in lengths]
    return mean(scores, weights)


def trimmed_mean(
    scores: Sequence[float],
    weights: Sequence[float] | None = None,
    *,
    trim_ratio: float = 0.1,
) -> Probability:
    if not scores:
        return 0.5
    if not 0.0 <= trim_ratio < 0.5:
        raise ValueError("trim_ratio must satisfy 0.0 <= trim_ratio < 0.5")
    weights = list(_normalise_weights(weights, len(scores)))
    total_weight = float(sum(weights))
    if total_weight <= 0.0:
        return 0.5
    if trim_ratio == 0.0 or len(scores) == 1:
        return mean(scores, weights)

    pairs = sorted(zip(scores, weights, strict=False), key=lambda item: item[0])
    lower_clip = total_weight * trim_ratio
    upper_clip = total_weight * (1.0 - trim_ratio)
    running_weight = 0.0
    accum = 0.0

    for score, weight in pairs:
        next_weight = running_weight + weight
        if next_weight <= lower_clip:
            running_weight = next_weight
            continue
        usable_weight = weight
        if running_weight < lower_clip:
            usable_weight -= lower_clip - running_weight
        if next_weight > upper_clip:
            usable_weight -= next_weight - upper_clip
        if usable_weight > 0:
            accum += score * usable_weight
        running_weight = next_weight
        if running_weight >= upper_clip:
            break

    denom = upper_clip - lower_clip
    if denom <= 0.0:
        return mean(scores, weights)
    return accum / denom


def median(scores: Sequence[float]) -> Probability:
    if not scores:
        return 0.5
    ordered = sorted(scores)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def vote_fraction(
    scores: Sequence[float],
    weights: Sequence[float] | None = None,
    *,
    threshold: float = _SENTENCE_THRESHOLD,
) -> Probability:
    if not scores:
        return 0.5
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be within [0, 1]")
    weights = _normalise_weights(weights, len(scores))
    total = float(sum(weights))
    if total <= 0.0:
        return 0.5
    positive = sum(
        weight for score, weight in zip(scores, weights, strict=False) if score >= threshold
    )
    return positive / total


def max_score(scores: Sequence[float]) -> Probability:
    if not scores:
        return 0.5
    return max(scores)


def logit_weighted_mean(
    scores: Sequence[float], weights: Sequence[float] | None = None
) -> Probability:
    if not scores:
        return 0.5
    weights = _normalise_weights(weights, len(scores))
    total = float(sum(weights))
    if total <= 0.0:
        return 0.5
    logit_sum = 0.0
    for score, weight in zip(scores, weights, strict=False):
        prob = _safe_probability(score)
        logit = math.log(prob / (1.0 - prob))
        logit_sum += logit * weight
    averaged = logit_sum / total
    return 1.0 / (1.0 + math.exp(-averaged))


@dataclass(frozen=True)
class AggregatedDocument:
    """Structured output for aggregation experiments."""

    doc_id: str
    source: str
    is_llm: bool
    scores: Sequence[float]
    lengths: Sequence[int]

    def as_dict(self) -> dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "is_llm": self.is_llm,
            "scores": list(self.scores),
            "lengths": list(self.lengths),
        }


Aggregator = Callable[[Sequence[float], Sequence[int] | None], Probability]


DEFAULT_AGGREGATORS: Mapping[str, Aggregator] = {
    "length_weighted_mean": lambda scores, lengths: length_weighted_mean(scores, lengths),
    "simple_mean": lambda scores, lengths: mean(scores),
    "trimmed_mean": lambda scores, lengths: trimmed_mean(scores, weights=lengths, trim_ratio=0.1),
    "median": lambda scores, lengths: median(scores),
    "vote_fraction": lambda scores, lengths: vote_fraction(
        scores, weights=lengths, threshold=_SENTENCE_THRESHOLD
    ),
    "logit_weighted_mean": lambda scores, lengths: logit_weighted_mean(scores, lengths),
    "max_score": lambda scores, lengths: max_score(scores),
}


def apply_aggregators(
    scores: Sequence[float],
    lengths: Sequence[int],
    aggregators: Mapping[str, Aggregator] | None = None,
) -> dict[str, Probability]:
    if aggregators is None:
        aggregators = DEFAULT_AGGREGATORS
    results: dict[str, Probability] = {}
    for name, func in aggregators.items():
        try:
            results[name] = func(scores, lengths)
        except Exception:
            results[name] = 0.5
    return results


__all__ = [
    "AggregatedDocument",
    "Aggregator",
    "DEFAULT_AGGREGATORS",
    "apply_aggregators",
    "length_weighted_mean",
    "logit_weighted_mean",
    "max_score",
    "mean",
    "median",
    "trimmed_mean",
    "vote_fraction",
]
