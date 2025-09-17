"""Mechanical registry for feature functions."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum

from . import divergence, statistical, tokenizer


class FeatureCategory(Enum):
    """Coarse grouping used when organising features."""

    STATISTICAL = "statistical"
    DIVERGENCE = "divergence"
    TOKENIZER = "tokenizer"


@dataclass(slots=True)
class FeatureDefinition:
    """Metadata describing a feature extraction function."""

    name: str
    category: FeatureCategory
    compute_fn: Callable[[str], float]
    scale_invariant: bool
    description: str = ""
    enabled: bool = True


class FeatureRegistry:
    """In-memory registry keyed by feature name."""

    def __init__(self) -> None:
        self._features: dict[str, FeatureDefinition] = {}
        self._scale_cache: list[str] | None = None

    def register(self, feature: FeatureDefinition, *, overwrite: bool = False) -> None:
        if feature.name in self._features and not overwrite:
            raise ValueError(f"feature '{feature.name}' already registered")
        self._features[feature.name] = feature
        self._scale_cache = None

    def register_many(
        self, features: Iterable[FeatureDefinition], *, overwrite: bool = False
    ) -> None:
        for definition in features:
            self.register(definition, overwrite=overwrite)

    def get(self, name: str) -> FeatureDefinition | None:
        return self._features.get(name)

    def names(self) -> list[str]:
        return sorted(self._features.keys())

    def all(
        self,
        *,
        category: FeatureCategory | None = None,
        scale_invariant_only: bool = False,
        enabled_only: bool = True,
    ) -> list[FeatureDefinition]:
        results: list[FeatureDefinition] = []
        for feature in self._features.values():
            if enabled_only and not feature.enabled:
                continue
            if scale_invariant_only and not feature.scale_invariant:
                continue
            if category is not None and feature.category != category:
                continue
            results.append(feature)
        return results

    def compute(
        self,
        text: str,
        feature_names: Iterable[str] | None = None,
        *,
        scale_invariant_only: bool = False,
    ) -> dict[str, float]:
        if not text:
            return {}
        if feature_names is None:
            selected = self.all(scale_invariant_only=scale_invariant_only, enabled_only=True)
        else:
            selected = [self._features[name] for name in feature_names if name in self._features]
        values: dict[str, float] = {}
        for definition in selected:
            if scale_invariant_only and not definition.scale_invariant:
                continue
            values[definition.name] = definition.compute_fn(text)
        return values

    def scale_invariant_names(self) -> list[str]:
        if self._scale_cache is None:
            self._scale_cache = [
                name
                for name, definition in self._features.items()
                if definition.scale_invariant and definition.enabled
            ]
        return list(self._scale_cache)

    def __len__(self) -> int:
        return len(self._features)


def register_default_features(registry: FeatureRegistry | None = None) -> FeatureRegistry:
    reg = registry or FeatureRegistry()

    definitions = [
        FeatureDefinition(
            name="stat.type_token_ratio",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.type_token_ratio,
            scale_invariant=True,
            description="Approximate type-token ratio (windowed)",
        ),
        FeatureDefinition(
            name="stat.herdan_c",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.herdan_c,
            scale_invariant=True,
            description="Herdan's C lexical diversity",
        ),
        FeatureDefinition(
            name="stat.mattr",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.mattr,
            scale_invariant=True,
            description="Moving-average type-token ratio",
        ),
        FeatureDefinition(
            name="stat.mean_word_length",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.mean_word_length,
            scale_invariant=True,
            description="Average characters per word",
        ),
        FeatureDefinition(
            name="stat.mean_sentence_length",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.mean_sentence_length,
            scale_invariant=True,
            description="Average words per sentence",
        ),
        FeatureDefinition(
            name="stat.word_length_cv",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.word_length_cv,
            scale_invariant=True,
            description="Coefficient of variation for word length",
        ),
        FeatureDefinition(
            name="stat.sentence_length_cv",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.sentence_length_cv,
            scale_invariant=True,
            description="Coefficient of variation for sentence length",
        ),
        FeatureDefinition(
            name="stat.char_entropy_norm",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.char_entropy_normalized,
            scale_invariant=True,
            description="Normalized character entropy",
        ),
        FeatureDefinition(
            name="stat.word_entropy_norm",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.word_entropy_normalized,
            scale_invariant=True,
            description="Normalized word entropy",
        ),
        FeatureDefinition(
            name="stat.punctuation_ratio",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.punctuation_ratio,
            scale_invariant=True,
            description="Punctuation marks per character",
        ),
        FeatureDefinition(
            name="stat.lowercase_char_ratio",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.lowercase_char_ratio,
            scale_invariant=True,
            description="Lowercase characters per character",
        ),
        FeatureDefinition(
            name="stat.uppercase_char_ratio",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.uppercase_char_ratio,
            scale_invariant=True,
            description="Uppercase characters per character",
        ),
        FeatureDefinition(
            name="stat.digit_char_ratio",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.digit_char_ratio,
            scale_invariant=True,
            description="Digit characters per character",
        ),
        FeatureDefinition(
            name="stat.whitespace_char_ratio",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.whitespace_char_ratio,
            scale_invariant=True,
            description="Whitespace characters per character",
        ),
        FeatureDefinition(
            name="stat.other_char_ratio",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.other_char_ratio,
            scale_invariant=True,
            description="Other characters per character",
        ),
        FeatureDefinition(
            name="stat.comma_ratio",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.comma_ratio,
            scale_invariant=True,
            description="Comma frequency per character",
        ),
        FeatureDefinition(
            name="stat.semicolon_ratio",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.semicolon_ratio,
            scale_invariant=True,
            description="Semicolon frequency per character",
        ),
        FeatureDefinition(
            name="stat.question_ratio",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.question_ratio,
            scale_invariant=True,
            description="Question-mark frequency per character",
        ),
        FeatureDefinition(
            name="stat.exclamation_ratio",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.exclamation_ratio,
            scale_invariant=True,
            description="Exclamation-mark frequency per character",
        ),
        FeatureDefinition(
            name="stat.period_ratio",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.period_ratio,
            scale_invariant=True,
            description="Period frequency per character",
        ),
        FeatureDefinition(
            name="stat.colon_ratio",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.colon_ratio,
            scale_invariant=True,
            description="Colon frequency per character",
        ),
        FeatureDefinition(
            name="stat.quote_ratio",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.quote_ratio,
            scale_invariant=True,
            description="Quote character frequency per character",
        ),
        FeatureDefinition(
            name="stat.function_word_ratio",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.function_word_ratio,
            scale_invariant=True,
            description="Function-word tokens per total tokens",
        ),
        FeatureDefinition(
            name="stat.capitalized_word_ratio",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.capitalized_word_ratio,
            scale_invariant=True,
            description="Capitalized tokens excluding sentence starts",
        ),
        FeatureDefinition(
            name="stat.max_char_run_ratio",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.max_char_run_ratio,
            scale_invariant=True,
            description="Longest repeated character run normalized",
        ),
        FeatureDefinition(
            name="stat.repeated_punctuation_ratio",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.repeated_punctuation_ratio,
            scale_invariant=True,
            description="Repeated punctuation fraction",
        ),
        FeatureDefinition(
            name="stat.whitespace_burstiness",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.whitespace_burstiness,
            scale_invariant=True,
            description="Whitespace run burstiness (CV)",
        ),
        FeatureDefinition(
            name="div.char_jsd",
            category=FeatureCategory.DIVERGENCE,
            compute_fn=divergence.char_divergence_score,
            scale_invariant=True,
            description="Character distribution Jensen-Shannon divergence",
        ),
        FeatureDefinition(
            name="div.punct_jsd",
            category=FeatureCategory.DIVERGENCE,
            compute_fn=divergence.punct_divergence_score,
            scale_invariant=True,
            description="Punctuation distribution Jensen-Shannon divergence",
        ),
        FeatureDefinition(
            name="div.regex_jsd",
            category=FeatureCategory.DIVERGENCE,
            compute_fn=divergence.regex_divergence_score,
            scale_invariant=True,
            description="Regex bucket Jensen-Shannon divergence",
        ),
        FeatureDefinition(
            name="stat.total_words",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.total_words,
            scale_invariant=False,
            description="Total number of word tokens",
            enabled=True,
        ),
        FeatureDefinition(
            name="stat.total_sentences",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.total_sentences,
            scale_invariant=False,
            description="Total number of sentences",
            enabled=True,
        ),
        FeatureDefinition(
            name="stat.unique_words",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.unique_words,
            scale_invariant=False,
            description="Unique word count",
            enabled=True,
        ),
        FeatureDefinition(
            name="stat.hapax_legomena_count",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.hapax_legomena_count,
            scale_invariant=False,
            description="Words occurring exactly once",
            enabled=True,
        ),
        FeatureDefinition(
            name="stat.total_characters",
            category=FeatureCategory.STATISTICAL,
            compute_fn=statistical.total_characters,
            scale_invariant=False,
            description="Total number of characters",
            enabled=True,
        ),
    ]

    for definition in definitions:
        try:
            reg.register(definition)
        except ValueError:
            # Ignore duplicates to keep the helper idempotent
            pass

    tokenizer.register_tokenizer_features(reg)

    return reg


DEFAULT_REGISTRY = register_default_features()


__all__ = [
    "FeatureCategory",
    "FeatureDefinition",
    "FeatureRegistry",
    "register_default_features",
    "DEFAULT_REGISTRY",
]
