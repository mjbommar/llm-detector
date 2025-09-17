"""Registry of streaming data sources for baseline collection."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from llm_detector.textops import DEFAULT_MIN_SENTENCE_LENGTH

from .base import SourceFactory
from .human import FinePDFsSource, GutenbergSource, WikipediaSource
from .llm import CosmopediaSource, LMSYSSource, UltraChatSource


class SourceCategory(Enum):
    """Categorise streaming sources by origin."""

    HUMAN = "human"
    LLM = "llm"


@dataclass(slots=True)
class SourceDefinition:
    """Metadata describing a streaming data source factory."""

    name: str
    category: SourceCategory
    factory: SourceFactory
    description: str = ""
    language: str = "en"
    enabled: bool = True


class SourceRegistry:
    """Collection of source definitions keyed by name."""

    def __init__(self) -> None:
        self._sources: dict[str, SourceDefinition] = {}

    def register(self, definition: SourceDefinition, *, overwrite: bool = False) -> None:
        if definition.name in self._sources and not overwrite:
            raise ValueError(f"source '{definition.name}' already registered")
        self._sources[definition.name] = definition

    def register_many(
        self, definitions: Iterable[SourceDefinition], *, overwrite: bool = False
    ) -> None:
        for definition in definitions:
            self.register(definition, overwrite=overwrite)

    def get(self, name: str) -> SourceDefinition | None:
        return self._sources.get(name)

    def names(
        self, *, category: SourceCategory | None = None, enabled_only: bool = True
    ) -> list[str]:
        return [
            definition.name for definition in self.all(category=category, enabled_only=enabled_only)
        ]

    def all(
        self,
        *,
        category: SourceCategory | None = None,
        enabled_only: bool = True,
    ) -> list[SourceDefinition]:
        items: list[SourceDefinition] = []
        for definition in self._sources.values():
            if enabled_only and not definition.enabled:
                continue
            if category is not None and definition.category != category:
                continue
            items.append(definition)
        return sorted(items, key=lambda item: item.name)

    def factories(
        self,
        *,
        category: SourceCategory | None = None,
        enabled_only: bool = True,
    ) -> list[SourceFactory]:
        return [
            definition.factory
            for definition in self.all(category=category, enabled_only=enabled_only)
        ]

    def __len__(self) -> int:
        return len(self._sources)


def register_default_sources(registry: SourceRegistry | None = None) -> SourceRegistry:
    reg = registry or SourceRegistry()

    definitions = [
        SourceDefinition(
            name="finepdfs",
            category=SourceCategory.HUMAN,
            factory=lambda: FinePDFsSource(
                min_doc_length=400,
                min_language_score=0.75,
                min_sentence_length=DEFAULT_MIN_SENTENCE_LENGTH,
            ),
            description="English FinePDF documents",
        ),
        SourceDefinition(
            name="gutenberg",
            category=SourceCategory.HUMAN,
            factory=lambda: GutenbergSource(
                min_book_length=1500,
                min_sentence_length=DEFAULT_MIN_SENTENCE_LENGTH,
            ),
            description="Project Gutenberg English books",
        ),
        SourceDefinition(
            name="wikipedia",
            category=SourceCategory.HUMAN,
            factory=lambda: WikipediaSource(
                min_article_length=200,
                min_sentence_length=DEFAULT_MIN_SENTENCE_LENGTH,
            ),
            description="English Wikipedia articles",
        ),
        SourceDefinition(
            name="cosmopedia_web_samples_v2",
            category=SourceCategory.LLM,
            factory=lambda: CosmopediaSource(
                min_text_length=200,
                min_sentence_length=DEFAULT_MIN_SENTENCE_LENGTH,
            ),
            description="Cosmopedia synthetic long-form articles",
        ),
        SourceDefinition(
            name="lmsys",
            category=SourceCategory.LLM,
            factory=lambda: LMSYSSource(
                min_response_length=50,
                min_sentence_length=DEFAULT_MIN_SENTENCE_LENGTH,
            ),
            description="LMSYS multi-model assistant responses",
        ),
        SourceDefinition(
            name="ultrachat",
            category=SourceCategory.LLM,
            factory=lambda: UltraChatSource(
                min_response_length=50,
                min_sentence_length=DEFAULT_MIN_SENTENCE_LENGTH,
            ),
            description="UltraChat assistant responses",
        ),
    ]

    reg.register_many(definitions)
    return reg


DEFAULT_REGISTRY = register_default_sources()


__all__ = [
    "SourceCategory",
    "SourceDefinition",
    "SourceRegistry",
    "register_default_sources",
    "DEFAULT_REGISTRY",
]
