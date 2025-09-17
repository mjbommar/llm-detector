"""Tests for the streaming source registry."""

from __future__ import annotations

import pytest

from llm_detector.training import (
    DEFAULT_REGISTRY,
    BaseDataSource,
    SourceCategory,
    SourceDefinition,
    SourceRegistry,
    register_default_sources,
)


def _collect_names(registry: SourceRegistry, category: SourceCategory) -> set[str]:
    return set(registry.names(category=category))


def test_default_registry_contains_expected_sources():
    human_names = _collect_names(DEFAULT_REGISTRY, SourceCategory.HUMAN)
    llm_names = _collect_names(DEFAULT_REGISTRY, SourceCategory.LLM)

    assert human_names == {"finepdfs", "gutenberg", "wikipedia"}
    assert llm_names == {"cosmopedia_web_samples_v2", "lmsys", "ultrachat"}

    for name in sorted(human_names | llm_names):
        definition = DEFAULT_REGISTRY.get(name)
        assert definition is not None
        source = definition.factory()
        assert isinstance(source, BaseDataSource)
        assert source.name == name or source.name.startswith(name)


def test_registry_filters_and_duplicates():
    registry = SourceRegistry()
    definition = DEFAULT_REGISTRY.get("wikipedia")
    assert definition is not None

    def dummy_factory():
        return definition.factory()

    registry.register(
        SourceDefinition(
            name="dummy",
            category=SourceCategory.HUMAN,
            factory=dummy_factory,
            enabled=False,
        )
    )

    assert registry.names(enabled_only=True) == []
    assert registry.names(enabled_only=False) == ["dummy"]

    with pytest.raises(ValueError):
        registry.register(
            SourceDefinition(
                name="dummy",
                category=SourceCategory.LLM,
                factory=dummy_factory,
            )
        )


def test_register_default_sources_returns_shared_instance():
    registry = register_default_sources(SourceRegistry())
    assert len(registry.names()) == len(DEFAULT_REGISTRY.names())
