"""Lightweight divergence features used for heuristics and baselines."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping

_PUNCTUATION = set(".,;:!?-—'\"()[]{}<>/\\")
_REGEX_PATTERNS: Mapping[str, str] = {
    "uppercase": r"[A-Z]",
    "lowercase": r"[a-z]",
    "digit": r"\d",
    "whitespace": r"\s",
    "word": r"\w+",
    "sentence_end": r"[.!?]",
    "comma": r",",
    "quote": r"['\"]",
}


def _normalize(dist: Mapping[str, float]) -> dict[str, float]:
    total = float(sum(v for v in dist.values() if v > 0))
    if total <= 0:
        return {}
    return {k: float(v) / total for k, v in dist.items() if v > 0}


def _jensen_shannon(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    keys = set(p.keys()) | set(q.keys())
    if not keys:
        return 0.0

    p_norm = _normalize({k: p.get(k, 0.0) for k in keys})
    q_norm = _normalize({k: q.get(k, 0.0) for k in keys})
    if not p_norm or not q_norm:
        return 0.0

    m = {k: 0.5 * (p_norm.get(k, 0.0) + q_norm.get(k, 0.0)) for k in keys}

    def _kl(src: Mapping[str, float], ref: Mapping[str, float]) -> float:
        val = 0.0
        for k in keys:
            s = src.get(k, 0.0)
            r = ref.get(k, 0.0)
            if s > 0 and r > 0:
                val += s * math.log2(s / r)
        return val

    js = 0.5 * (_kl(p_norm, m) + _kl(q_norm, m))
    return max(0.0, min(1.0, js))


def _uniform(keys: Iterable[str]) -> dict[str, float]:
    key_list = [k for k in keys if k]
    if not key_list:
        return {}
    weight = 1.0 / len(key_list)
    return {k: weight for k in key_list}


def _char_distribution(text: str) -> dict[str, float]:
    if not text:
        return {}
    counts = Counter(text)
    return _normalize(counts)


def _punct_distribution(text: str) -> dict[str, float]:
    counts = Counter(c for c in text if c in _PUNCTUATION)
    return _normalize(counts)


def _regex_distribution(text: str) -> dict[str, float]:
    counts: dict[str, float] = {}
    for name, pattern in _REGEX_PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            counts[name] = float(len(matches))
    return _normalize(counts)


def char_divergence_score(text: str, baseline: Mapping[str, float] | None = None) -> float:
    """Jensen–Shannon divergence between text char distribution and baseline."""

    distribution = _char_distribution(text)
    if not distribution:
        return 0.0
    if baseline is None:
        baseline = _uniform(distribution.keys())
    return _jensen_shannon(distribution, baseline)


def punct_divergence_score(text: str, baseline: Mapping[str, float] | None = None) -> float:
    """Divergence over punctuation usage patterns."""

    distribution = _punct_distribution(text)
    if not distribution:
        return 0.0
    if baseline is None:
        baseline = _uniform(distribution.keys())
    return _jensen_shannon(distribution, baseline)


def regex_divergence_score(text: str, baseline: Mapping[str, float] | None = None) -> float:
    """Divergence over coarse regex-derived categories."""

    distribution = _regex_distribution(text)
    if not distribution:
        return 0.0
    if baseline is None:
        baseline = _uniform(distribution.keys())
    return _jensen_shannon(distribution, baseline)


__all__ = [
    "char_divergence_score",
    "punct_divergence_score",
    "regex_divergence_score",
]
