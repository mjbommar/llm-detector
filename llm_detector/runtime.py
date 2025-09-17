"""Runtime helpers for loading trained models and scoring new text."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from llm_detector.aggregation import (
    logit_weighted_mean,
    max_score,
    mean,
    vote_fraction,
)
from llm_detector.baselines import BaselineCache, divergence_baseline_overrides
from llm_detector.features import FeatureVectorizer, register_default_features
from llm_detector.models import LogisticRegressionModel
from llm_detector.textops import SentenceSegment, segment_sentences

DocumentMetrics = dict[str, float]

PRIMARY_METRIC = "logit_weighted_mean"
DEFAULT_DIAGNOSTICS = ("simple_mean", "max_score", "vote_fraction")


def _compute_metric(name: str, scores: Sequence[float], lengths: Sequence[int]) -> float:
    if name == "logit_weighted_mean":
        return logit_weighted_mean(scores, lengths)
    if name == "simple_mean":
        return mean(scores)
    if name == "length_weighted_mean":
        return mean(scores, lengths)
    if name == "max_score":
        return max_score(scores)
    if name == "vote_fraction":
        return vote_fraction(scores, weights=lengths)
    raise KeyError(f"Unknown document metric '{name}'")


@dataclass(slots=True)
class DetectionResult:
    """Prediction output returned by :class:`DetectorRuntime`."""

    is_llm: bool
    confidence: float
    p_llm: float
    p_human: float
    features: dict[str, float] | None = None
    details: dict[str, Any] | None = None


class DetectorRuntime:
    """Convenience wrapper that loads artifacts and scores text samples."""

    def __init__(
        self,
        *,
        model_path: Path | str,
        baseline_path: Path | str,
        cohort: Literal["human", "llm"] = "human",
        return_features: bool = False,
        min_sentence_length: int = 1,
        primary_metric: str = PRIMARY_METRIC,
        diagnostic_metrics: Sequence[str] | None = None,
    ) -> None:
        self.model = LogisticRegressionModel.load(model_path)
        if not self.model.feature_names:
            raise RuntimeError("loaded model is missing feature metadata")

        baselines = BaselineCache.load(Path(baseline_path))
        overrides = divergence_baseline_overrides(baselines, cohort=cohort)

        registry = register_default_features()
        self.vectorizer = FeatureVectorizer(
            registry,
            feature_names=self.model.feature_names,
            baseline_overrides=overrides,
            scale_invariant_only=True,
        )
        self.return_features = return_features

        self.min_sentence_length = max(1, min_sentence_length)
        self.primary_metric = primary_metric
        diagnostics = (
            tuple(diagnostic_metrics) if diagnostic_metrics is not None else DEFAULT_DIAGNOSTICS
        )
        self.diagnostic_metrics: tuple[str, ...] = diagnostics

        metric_names = {self.primary_metric, *self.diagnostic_metrics}
        for name in metric_names:
            _ = _compute_metric(name, [0.5], [1])

    def _ordered_features(
        self, text: str, *, include_mapping: bool
    ) -> tuple[list[float], dict[str, float] | None]:
        mapping = self.vectorizer.compute(text)
        values = [mapping[name] for name in self.vectorizer.feature_names]
        if include_mapping:
            return values, mapping
        return values, None

    def predict(self, text: str) -> DetectionResult:
        if not text:
            raise ValueError("text must be a non-empty string")

        sentences: list[SentenceSegment] = segment_sentences(
            text,
            min_length=self.min_sentence_length,
        )
        if not sentences:
            clean = text.strip()
            if not clean:
                raise ValueError("text must contain non-whitespace characters")
            sentences = [SentenceSegment(text=clean, paragraph_index=0, sentence_index=0)]

        sentence_details: list[dict[str, Any]] = []
        aggregate_features: dict[str, float] | None = None
        best_length = -1
        scores: list[float] = []
        lengths: list[int] = []

        for segment in sentences:
            vector, mapping = self._ordered_features(
                segment.text, include_mapping=self.return_features
            )
            p_human, p_llm = self.model.predict_proba(vector)
            length = max(1, len(segment.text))
            entry: dict[str, Any] = {
                "paragraph_index": segment.paragraph_index,
                "sentence_index": segment.sentence_index,
                "length": length,
                "p_llm": p_llm,
                "p_human": p_human,
                "text": segment.text,
            }
            if self.return_features and mapping is not None:
                entry["features"] = mapping
                if length >= best_length:
                    aggregate_features = mapping
                    best_length = length
            sentence_details.append(entry)
            scores.append(p_llm)
            lengths.append(length)

        metrics_to_compute = {self.primary_metric, *self.diagnostic_metrics}
        document_metrics: DocumentMetrics = {}
        for name in metrics_to_compute:
            try:
                value = _compute_metric(name, scores, lengths)
            except KeyError:
                continue
            document_metrics[name] = value

        p_llm = document_metrics.get(self.primary_metric, 0.5)
        p_human = 1.0 - p_llm
        is_llm = p_llm >= 0.5
        confidence = p_llm if is_llm else p_human

        details = {
            "sentence_count": len(sentence_details),
            "sentences": sentence_details,
            "primary_metric": self.primary_metric,
            "diagnostic_metrics": list(self.diagnostic_metrics),
            "document_metrics": document_metrics,
        }

        return DetectionResult(
            is_llm=is_llm,
            confidence=confidence,
            p_llm=p_llm,
            p_human=p_human,
            features=aggregate_features if self.return_features else None,
            details=details,
        )

    def predict_batch(self, texts: Sequence[str]) -> list[DetectionResult]:
        return [self.predict(text) for text in texts]

    def predict_stream(self, texts: Iterable[str]) -> Iterable[DetectionResult]:
        for text in texts:
            yield self.predict(text)


__all__ = ["DetectorRuntime", "DetectionResult"]
