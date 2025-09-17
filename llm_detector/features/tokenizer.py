"""Tokenizer-based feature extraction with optional dependencies."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .statistical import _tokenize_words as _words_lower  # reuse helper

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .registry import FeatureRegistry

try:
    from importlib import resources
except ImportError:  # pragma: no cover
    import importlib_resources as resources  # type: ignore[import]

try:  # Optional dependency; handled gracefully if absent
    from tokenizers import Tokenizer as HFTokenizer
except Exception:  # pragma: no cover - dependency missing
    HFTokenizer = None  # type: ignore[assignment]


class TokenizerFeatureError(RuntimeError):
    """Raised when tokenizer-based features cannot be computed."""


def _load_tokenizer_from_assets(tokenizer_name: str) -> Any:
    """Load a tokenizer from packaged assets.

    Args:
        tokenizer_name: Name of the tokenizer file (without .json extension)

    Returns:
        Loaded tokenizer instance

    Raises:
        TokenizerFeatureError: If tokenizer cannot be loaded
    """
    if HFTokenizer is None:
        raise TokenizerFeatureError(
            "tokenizers package not installed; install with `pip install tokenizers`"
        )

    try:
        # Try to load from package resources
        if hasattr(resources, "files"):
            # Python 3.9+
            assets_path = resources.files("llm_detector.assets.tokenizers")
            tokenizer_path = assets_path / f"{tokenizer_name}.json"
            with tokenizer_path.open("r") as f:
                tokenizer_json = f.read()
        else:
            # Fallback for older Python
            with resources.open_text(
                "llm_detector.assets.tokenizers", f"{tokenizer_name}.json"
            ) as f:
                tokenizer_json = f.read()

        return HFTokenizer.from_str(tokenizer_json)
    except Exception as exc:
        raise TokenizerFeatureError(
            f"Failed to load tokenizer from assets '{tokenizer_name}': {exc}"
        ) from exc


def _coefficient_of_variation(values: Iterable[int | float]) -> float:
    data = [float(v) for v in values]
    if len(data) < 2:
        return 0.0
    mean = sum(data) / len(data)
    if mean == 0:
        return 0.0
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return math.sqrt(variance) / mean


@dataclass(slots=True)
class TokenizerSpec:
    """Describe a tokenizer used for feature extraction."""

    key: str
    pretrained: str | None = None
    loader: Callable[[], Any] | None = None
    description: str = ""
    enabled: bool = False


class TokenizerFeatureExtractor:
    """Compute metrics derived from a single tokenizer."""

    def __init__(self, spec: TokenizerSpec) -> None:
        self.spec = spec
        self._tokenizer: Any | None = None
        self._last_text: str | None = None
        self._last_metrics: dict[str, float] | None = None

    def _load(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        if self.spec.loader is not None:
            tokenizer = self.spec.loader()
        else:
            if not self.spec.pretrained:
                raise TokenizerFeatureError("Tokenizer spec missing pretrained identifier")

            # Map pretrained names to our asset file names
            tokenizer_map = {
                "gpt2": "gpt2_tokenizer",
                "bert-base-uncased": "bert_base_uncased_tokenizer",
                "roberta-base": "roberta_base_tokenizer",
                "openai/gpt-oss-20b": "openai_gpt_oss_20b_tokenizer",
            }

            if self.spec.pretrained in tokenizer_map:
                # Load from packaged assets
                tokenizer = _load_tokenizer_from_assets(tokenizer_map[self.spec.pretrained])
            else:
                # Fallback to downloading from HuggingFace
                if HFTokenizer is None:
                    raise TokenizerFeatureError(
                        "tokenizers package not installed; install with `pip install tokenizers`"
                    )
                try:
                    tokenizer = HFTokenizer.from_pretrained(self.spec.pretrained)
                except Exception as exc:  # pragma: no cover - runtime env specific
                    raise TokenizerFeatureError(
                        f"Failed to load tokenizer '{self.spec.pretrained}': {exc}"
                    ) from exc
        self._tokenizer = tokenizer
        return tokenizer

    @staticmethod
    def _is_special(token: str) -> bool:
        if not token:
            return True
        if token.startswith("[") and token.endswith("]"):
            return True
        if token.startswith("<") and token.endswith(">"):
            return True
        if token in {"<s>", "</s>", "<pad>", "<unk>", "<mask>"}:
            return True
        return False

    @staticmethod
    def _starts_word(token: str) -> bool:
        if token.startswith("##"):
            return False
        if token.startswith("Ġ") or token.startswith("▁"):
            return True
        return True

    @staticmethod
    def _clean_token(token: str) -> str:
        cleaned = token
        if cleaned.startswith("##"):
            cleaned = cleaned[2:]
        while cleaned and cleaned[0] in {"▁", "Ġ", "Ċ"}:
            cleaned = cleaned[1:]
        return cleaned.strip()

    def _encoding_tokens(self, tokenizer: Any, text: str) -> list[str]:
        encoding = tokenizer.encode(text)
        tokens = getattr(encoding, "tokens", None)
        if tokens is None:
            raise TokenizerFeatureError("Tokenizer encoding did not provide tokens list")
        return list(tokens)

    def _compute_metrics(self, tokens: list[str], text: str) -> dict[str, float]:
        if not tokens:
            return {name: 0.0 for name in TOKEN_METRIC_NAMES}
        usable_tokens = [t for t in tokens if not self._is_special(t)]
        if not usable_tokens:
            return {name: 0.0 for name in TOKEN_METRIC_NAMES}

        cleaned = [self._clean_token(t) for t in usable_tokens]
        cleaned = [t for t in cleaned if t]

        metrics: dict[str, float] = {}

        # Efficiency: characters per token
        metrics["tokenization_efficiency"] = (
            len(text) / len(usable_tokens) if usable_tokens else 0.0
        )

        starts = [t for t in usable_tokens if self._starts_word(t)]
        start_count = len(starts)
        metrics["start_word_ratio"] = start_count / len(usable_tokens)

        subword_count = sum(1 for t in usable_tokens if t.startswith("##"))
        metrics["subword_ratio"] = subword_count / len(usable_tokens)

        if cleaned:
            single_char = sum(1 for t in cleaned if len(t) == 1)
            metrics["single_char_ratio"] = single_char / len(cleaned)
            metrics["avg_token_length"] = sum(len(t) for t in cleaned) / len(cleaned)
            metrics["token_length_cv"] = _coefficient_of_variation(len(t) for t in cleaned)
            metrics["non_alnum_ratio"] = sum(1 for t in cleaned if not t.isalnum()) / len(cleaned)
            metrics["digit_token_ratio"] = sum(1 for t in cleaned if t.isdigit()) / len(cleaned)
            metrics["unique_token_ratio"] = len(set(cleaned)) / len(cleaned)
            metrics["short_token_ratio_le2"] = sum(1 for t in cleaned if len(t) <= 2) / len(cleaned)
        else:
            for name in (
                "single_char_ratio",
                "avg_token_length",
                "token_length_cv",
                "non_alnum_ratio",
                "digit_token_ratio",
                "unique_token_ratio",
                "short_token_ratio_le2",
            ):
                metrics[name] = 0.0

        word_set = set(_words_lower(text, lowercase=True))
        if cleaned and word_set:
            cleaned_lower = [t.lower() for t in cleaned]
            matches = sum(
                1 for token, lower in zip(cleaned, cleaned_lower, strict=False) if lower in word_set
            )
            metrics["word_match_rate"] = matches / len(cleaned)
        else:
            metrics["word_match_rate"] = 0.0

        metrics["whole_token_ratio"] = start_count / len(usable_tokens)
        return metrics

    def compute(self, text: str) -> dict[str, float]:
        if text == self._last_text and self._last_metrics is not None:
            return self._last_metrics

        tokenizer = self._load()
        tokens = self._encoding_tokens(tokenizer, text)
        metrics = self._compute_metrics(tokens, text)
        self._last_text = text
        self._last_metrics = metrics
        return metrics


TOKEN_METRIC_NAMES = [
    "tokenization_efficiency",
    "word_match_rate",
    "whole_token_ratio",
    "single_char_ratio",
    "avg_token_length",
    "token_length_cv",
    "start_word_ratio",
    "subword_ratio",
    "non_alnum_ratio",
    "digit_token_ratio",
    "unique_token_ratio",
    "short_token_ratio_le2",
]

AGGREGATED_METRICS = {
    "efficiency_variance": "Variance of tokenization efficiency across tokenizers",
    "efficiency_std": "Standard deviation of tokenization efficiency across tokenizers",
}


class TokenizerFeatureCalculator:
    """Manage multiple tokenizers and expose computed metrics."""

    def __init__(self, specs: Sequence[TokenizerSpec]) -> None:
        self.specs = list(specs)
        self.extractors = {spec.key: TokenizerFeatureExtractor(spec) for spec in self.specs}
        self._last_text: str | None = None
        self._last_values: dict[str, float] = {}

    def available_keys(self) -> list[str]:
        return [spec.key for spec in self.specs]

    def compute(self, text: str) -> dict[str, float]:
        if not text:
            return {}
        if text == self._last_text:
            return self._last_values

        values: dict[str, float] = {}
        efficiencies: list[float] = []

        for key, extractor in self.extractors.items():
            metrics = extractor.compute(text)
            for metric_name, value in metrics.items():
                namespaced = f"tok.{key}.{metric_name}"
                values[namespaced] = value
            eff = metrics.get("tokenization_efficiency")
            if eff is not None:
                efficiencies.append(eff)

        if len(efficiencies) > 1:
            mean = sum(efficiencies) / len(efficiencies)
            variance = sum((e - mean) ** 2 for e in efficiencies) / len(efficiencies)
            values["tok.efficiency_variance"] = variance
            values["tok.efficiency_std"] = math.sqrt(variance)

        self._last_text = text
        self._last_values = values
        return values


DEFAULT_TOKENIZER_SPECS: Sequence[TokenizerSpec] = (
    TokenizerSpec(
        key="gpt2",
        pretrained="gpt2",
        description="OpenAI GPT-2 byte-pair encoding",
        enabled=True,
    ),
    TokenizerSpec(
        key="bert",
        pretrained="bert-base-uncased",
        description="BERT WordPiece tokenizer",
        enabled=True,
    ),
    TokenizerSpec(
        key="roberta",
        pretrained="roberta-base",
        description="RoBERTa byte-pair tokenizer",
        enabled=True,
    ),
    TokenizerSpec(
        key="gpt_oss_20b",
        pretrained="openai/gpt-oss-20b",
        description="OpenAI GPT-OSS 20B tokenizer",
        enabled=True,
    ),
)


def register_tokenizer_features(
    registry: FeatureRegistry | None = None,
    *,
    specs: Sequence[TokenizerSpec] | None = None,
) -> FeatureRegistry:
    """Register tokenizer-based features against ``registry``.

    Features are disabled by default to avoid dependency issues. Projects that
    provide the required tokenizers can enable them explicitly after
    registration.
    """

    if registry is None:
        from .registry import FeatureRegistry  # local import to avoid cycles

        reg: FeatureRegistry = FeatureRegistry()
    else:
        reg = registry
    tokenizer_specs = specs or DEFAULT_TOKENIZER_SPECS
    calculator = TokenizerFeatureCalculator(tokenizer_specs)

    from .registry import FeatureCategory, FeatureDefinition  # local import to avoid cycles

    metric_docs = {
        "tokenization_efficiency": "Characters per non-special token",
        "word_match_rate": "Fraction of tokens matching corpus words",
        "whole_token_ratio": "Fraction of tokens that begin a word",
        "single_char_ratio": "Fraction of cleaned tokens with length 1",
        "avg_token_length": "Average length of cleaned tokens",
        "token_length_cv": "Coefficient of variation of token lengths",
        "start_word_ratio": "Token fraction that starts a word",
        "subword_ratio": "Token fraction marked as subwords",
        "non_alnum_ratio": "Fraction of cleaned tokens that are non-alphanumeric",
        "digit_token_ratio": "Fraction of cleaned tokens that are digits",
        "unique_token_ratio": "Unique cleaned tokens / total cleaned tokens",
        "short_token_ratio_le2": "Fraction of cleaned tokens with length ≤ 2",
    }

    def make_compute(metric_name: str) -> Callable[[str], float]:
        def _compute(text: str) -> float:
            values = calculator.compute(text)
            return float(values.get(metric_name, 0.0))

        return _compute

    for spec in tokenizer_specs:
        prefix = f"tok.{spec.key}"
        for metric_key, description in metric_docs.items():
            feature_name = f"{prefix}.{metric_key}"
            definition = FeatureDefinition(
                name=feature_name,
                category=FeatureCategory.TOKENIZER,
                compute_fn=make_compute(feature_name),
                scale_invariant=True,
                description=f"{description} ({spec.description or spec.key})",
                enabled=spec.enabled,
            )
            try:
                reg.register(definition)
            except ValueError:
                pass

    for agg_name, description in AGGREGATED_METRICS.items():
        feature_name = f"tok.{agg_name}"
        definition = FeatureDefinition(
            name=feature_name,
            category=FeatureCategory.TOKENIZER,
            compute_fn=make_compute(feature_name),
            scale_invariant=True,
            description=description,
            enabled=True,
        )
        try:
            reg.register(definition)
        except ValueError:
            pass

    return reg


__all__ = [
    "TokenizerSpec",
    "TokenizerFeatureExtractor",
    "TokenizerFeatureCalculator",
    "TokenizerFeatureError",
    "register_tokenizer_features",
    "DEFAULT_TOKENIZER_SPECS",
]
