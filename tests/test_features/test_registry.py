from llm_detector.features import (
    FeatureCategory,
    FeatureDefinition,
    FeatureRegistry,
    register_default_features,
)


def test_register_and_retrieve():
    registry = FeatureRegistry()
    feature = FeatureDefinition(
        name="test.mean",
        category=FeatureCategory.STATISTICAL,
        compute_fn=lambda text: float(len(text)),
        scale_invariant=True,
        description="Length of text",
    )
    registry.register(feature)
    assert registry.get("test.mean") is feature
    assert registry.names() == ["test.mean"]
    result = registry.compute("abc")
    assert result == {"test.mean": 3.0}


def test_duplicate_registration_requires_overwrite():
    registry = FeatureRegistry()
    feature = FeatureDefinition(
        name="dup",
        category=FeatureCategory.STATISTICAL,
        compute_fn=lambda text: 1.0,
        scale_invariant=True,
    )
    registry.register(feature)
    try:
        registry.register(feature)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for duplicate registration")


def test_register_default_features_is_idempotent():
    registry = register_default_features()
    first_names = set(registry.names())
    assert first_names  # ensure defaults are populated
    registry = register_default_features(registry)
    assert set(registry.names()) == first_names
    # ensure scale invariant selection returns enabled feature names
    invariant = set(registry.scale_invariant_names())
    enabled_invariant = set()
    for name in first_names:
        definition = registry.get(name)
        if definition and definition.scale_invariant and definition.enabled:
            enabled_invariant.add(name)
    assert invariant == enabled_invariant
