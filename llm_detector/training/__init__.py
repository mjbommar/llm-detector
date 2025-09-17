"""Training utilities for the LLM detector."""

from __future__ import annotations

from .sources.base import BaseDataSource, BatchConfig, SourceFactory, StreamingShuffleMixin
from .sources.human import FinePDFsSource, GutenbergSource, WikipediaSource
from .sources.llm import CosmopediaSource, LMSYSSource, UltraChatSource
from .sources.registry import (
    DEFAULT_REGISTRY,
    SourceCategory,
    SourceDefinition,
    SourceRegistry,
    register_default_sources,
)

__all__ = [
    "BatchConfig",
    "BaseDataSource",
    "StreamingShuffleMixin",
    "SourceFactory",
    "FinePDFsSource",
    "GutenbergSource",
    "WikipediaSource",
    "CosmopediaSource",
    "LMSYSSource",
    "UltraChatSource",
    "SourceRegistry",
    "SourceCategory",
    "SourceDefinition",
    "register_default_sources",
    "DEFAULT_REGISTRY",
    "FeatureDataset",
    "build_feature_dataset",
    "build_dataset_from_registry",
    "TrainingArtifacts",
    "train_logistic_from_registry",
]


def __getattr__(name: str):  # pragma: no cover - thin lazy import helper
    if name in {"FeatureDataset", "build_feature_dataset", "build_dataset_from_registry"}:
        from . import dataset as _dataset

        return getattr(_dataset, name)
    if name in {"TrainingArtifacts", "train_logistic_from_registry"}:
        from . import pipeline as _pipeline

        return getattr(_pipeline, name)
    raise AttributeError(f"module 'llm_detector.training' has no attribute '{name}'")


def __dir__() -> list[str]:  # pragma: no cover - simple reflection aid
    return sorted(__all__)
