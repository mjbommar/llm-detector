from __future__ import annotations

from dataclasses import dataclass

from llm_detector.features.registry import FeatureRegistry
from llm_detector.features.tokenizer import TokenizerSpec, register_tokenizer_features


@dataclass
class FakeEncoding:
    tokens: list[str]


class FakeTokenizer:
    def __init__(self, mapping: dict[str, list[str]]) -> None:
        self.mapping = mapping

    def encode(self, text: str) -> FakeEncoding:
        return FakeEncoding(self.mapping.get(text, []))


def _build_registry() -> tuple[FeatureRegistry, str]:
    text = "Hello world 123"
    spec_a = TokenizerSpec(
        key="mockA",
        loader=lambda: FakeTokenizer({text: ["Hello", "world", "123"]}),
        description="Mock tokenizer A",
        enabled=True,
    )
    spec_b = TokenizerSpec(
        key="mockB",
        loader=lambda: FakeTokenizer(
            {
                text: ["ĠHello", "##world", "##123", "!"],
            }
        ),
        description="Mock tokenizer B",
        enabled=True,
    )
    registry = FeatureRegistry()
    register_tokenizer_features(registry, specs=[spec_a, spec_b])
    return registry, text


def test_tokenizer_features_compute_expected_metrics():
    registry, text = _build_registry()
    feature_names = [
        "tok.mockA.tokenization_efficiency",
        "tok.mockA.word_match_rate",
        "tok.mockB.single_char_ratio",
        "tok.mockB.start_word_ratio",
        "tok.efficiency_variance",
    ]
    features = registry.compute(text, feature_names=feature_names, scale_invariant_only=True)

    assert "tok.mockA.tokenization_efficiency" in features
    assert features["tok.mockA.word_match_rate"] == 1.0
    assert 0 <= features["tok.mockB.single_char_ratio"] <= 1
    assert features["tok.mockB.start_word_ratio"] < 1  # presence of ## tokens

    # Aggregated variance exists because two tokenizers enabled
    assert "tok.efficiency_variance" in features
    assert features["tok.efficiency_variance"] >= 0


def test_registry_compute_cached_results():
    registry, text = _build_registry()
    first = registry.compute(text, scale_invariant_only=True)
    second = registry.compute(text, scale_invariant_only=True)
    assert first == second
