"""Statistical and lexical features derived directly from raw text."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from functools import lru_cache

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")
_FUNCTION_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "is",
    "was",
    "are",
    "were",
    "been",
    "be",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "can",
    "shall",
    "must",
    "that",
    "this",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "them",
    "their",
    "what",
    "which",
    "who",
    "when",
    "where",
    "why",
    "how",
}
_PUNCTUATION_CHARS = set(".,;:!?-—'\"\\/()[]{}<>")


def _tokenize_words(text: str, *, lowercase: bool = False) -> list[str]:
    tokens = _WORD_RE.findall(text)
    if lowercase:
        return [t.lower() for t in tokens]
    return tokens


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    return [seg.strip() for seg in _SENTENCE_SPLIT_RE.split(text) if seg.strip()]


def _coefficient_of_variation(values: Iterable[float]) -> float:
    data = [float(v) for v in values]
    if len(data) < 2:
        return 0.0
    mean = sum(data) / len(data)
    if mean == 0:
        return 0.0
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return math.sqrt(variance) / mean


def _entropy_from_counts(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


@lru_cache(maxsize=64)
def _char_class_counts(text: str) -> tuple[int, int, int, int, int, int]:
    lower = upper = digit = space = punct = other = 0
    for ch in text:
        if ch.islower():
            lower += 1
        elif ch.isupper():
            upper += 1
        elif ch.isdigit():
            digit += 1
        elif ch.isspace():
            space += 1
        elif ch in _PUNCTUATION_CHARS:
            punct += 1
        else:
            other += 1
    return lower, upper, digit, space, punct, other


# ---------------------------------------------------------------------------
# Scale-invariant lexical and punctuation features
# ---------------------------------------------------------------------------


def type_token_ratio(text: str, window: int = 100) -> float:
    """Approximate type-token ratio using fixed windows to reduce length bias."""

    tokens = _tokenize_words(text, lowercase=True)
    n = len(tokens)
    if n == 0:
        return 0.0
    w = max(1, min(window, n))
    if n <= w:
        return len(set(tokens)) / float(n)
    ratios: list[float] = []
    for start in range(0, n, w):
        segment = tokens[start : start + w]
        if not segment:
            continue
        ratios.append(len(set(segment)) / float(len(segment)))
    return sum(ratios) / len(ratios) if ratios else 0.0


def herdan_c(text: str) -> float:
    """Herdan's C lexical diversity metric (log V / log N)."""

    tokens = _tokenize_words(text, lowercase=True)
    N = len(tokens)
    if N <= 1:
        return 0.0
    V = len(set(tokens))
    if V <= 1:
        return 0.0
    value = math.log(V, 10) / math.log(N, 10)
    return max(0.0, min(1.0, float(value)))


def mattr(text: str, window: int = 50) -> float:
    """Moving-average type-token ratio using a sliding window."""

    tokens = _tokenize_words(text, lowercase=True)
    n = len(tokens)
    if n == 0:
        return 0.0

    w = max(1, min(window, n))
    if n <= w:
        return len(set(tokens)) / float(n)

    ratios: list[float] = []
    for start in range(0, n - w + 1):
        window_tokens = tokens[start : start + w]
        ratios.append(len(set(window_tokens)) / float(w))
    return sum(ratios) / len(ratios)


def mean_word_length(text: str) -> float:
    """Average number of characters per token."""

    words = _tokenize_words(text)
    if not words:
        return 0.0
    return sum(len(word) for word in words) / len(words)


def mean_sentence_length(text: str) -> float:
    """Average number of word tokens per sentence."""

    sentences = _split_sentences(text)
    if not sentences:
        return 0.0
    total_words = sum(len(_tokenize_words(sentence)) for sentence in sentences)
    return total_words / len(sentences)


def word_length_cv(text: str) -> float:
    """Coefficient of variation for word lengths."""

    words = _tokenize_words(text)
    return _coefficient_of_variation(len(word) for word in words)


def sentence_length_cv(text: str) -> float:
    """Coefficient of variation for sentence lengths (measured in words)."""

    sentences = _split_sentences(text)
    word_counts = [len(_tokenize_words(sentence)) for sentence in sentences]
    return _coefficient_of_variation(word_counts)


def char_entropy_normalized(text: str) -> float:
    """Normalized Shannon entropy over characters."""

    if not text:
        return 0.0
    counts = Counter(text)
    entropy = _entropy_from_counts(counts)
    max_entropy = math.log2(len(counts)) if counts else 0.0
    if max_entropy == 0:
        return 0.0
    value = entropy / max_entropy
    return max(0.0, min(1.0, value))


def word_entropy_normalized(text: str) -> float:
    """Normalized Shannon entropy over word tokens."""

    words = _tokenize_words(text, lowercase=True)
    counts = Counter(words)
    if not counts:
        return 0.0
    entropy = _entropy_from_counts(counts)
    max_entropy = math.log2(len(counts)) if len(counts) > 0 else 0.0
    if max_entropy == 0:
        return 0.0
    value = entropy / max_entropy
    return max(0.0, min(1.0, value))


def punctuation_ratio(text: str) -> float:
    """Ratio of punctuation characters to total characters."""

    if not text:
        return 0.0
    count = sum(1 for ch in text if ch in _PUNCTUATION_CHARS)
    return count / len(text)


def lowercase_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    lower, *_ = _char_class_counts(text)
    return lower / len(text)


def uppercase_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    _, upper, *_ = _char_class_counts(text)
    return upper / len(text)


def digit_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    _, _, digit, *_ = _char_class_counts(text)
    return digit / len(text)


def whitespace_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    _, _, _, space, _, _ = _char_class_counts(text)
    return space / len(text)


def other_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    counts = _char_class_counts(text)
    other = counts[-1]
    return other / len(text)


def _char_ratio(text: str, target: str) -> float:
    return text.count(target) / len(text) if text else 0.0


def comma_ratio(text: str) -> float:
    return _char_ratio(text, ",")


def semicolon_ratio(text: str) -> float:
    return _char_ratio(text, ";")


def question_ratio(text: str) -> float:
    return _char_ratio(text, "?")


def exclamation_ratio(text: str) -> float:
    return _char_ratio(text, "!")


def period_ratio(text: str) -> float:
    return _char_ratio(text, ".")


def colon_ratio(text: str) -> float:
    return _char_ratio(text, ":")


def quote_ratio(text: str) -> float:
    if not text:
        return 0.0
    return (text.count("'") + text.count('"')) / len(text)


def function_word_ratio(text: str) -> float:
    """Fraction of tokens that are common English function words."""

    words = _tokenize_words(text, lowercase=True)
    if not words:
        return 0.0
    function_count = sum(1 for w in words if w in _FUNCTION_WORDS)
    return function_count / len(words)


def capitalized_word_ratio(text: str) -> float:
    """Ratio of capitalized words (excluding sentence starts) to total words."""

    words = _tokenize_words(text)
    if not words:
        return 0.0
    first_words = set()
    for sentence in _split_sentences(text):
        sentence_words = _tokenize_words(sentence)
        if sentence_words:
            first_words.add(sentence_words[0])
    capitalized = sum(1 for word in words if word[0].isupper() and word not in first_words)
    return capitalized / len(words)


def max_char_run_ratio(text: str) -> float:
    """Maximum repeated-character run length normalized by text length."""

    if not text:
        return 0.0
    max_run = 1
    current_run = 1
    last_char = text[0]
    for ch in text[1:]:
        if ch == last_char:
            current_run += 1
        else:
            if current_run > max_run:
                max_run = current_run
            current_run = 1
            last_char = ch
    if current_run > max_run:
        max_run = current_run
    return max_run / len(text)


def repeated_punctuation_ratio(text: str) -> float:
    """Fraction of punctuation characters that appear in repeated runs."""

    if not text:
        return 0.0
    total_punct = 0
    repeated = 0
    last_char = None
    run_length = 0

    for ch in text:
        if ch in _PUNCTUATION_CHARS:
            total_punct += 1
            if ch == last_char:
                run_length += 1
            else:
                run_length = 1
            last_char = ch
            if run_length >= 2:
                repeated += 1
        else:
            last_char = None
            run_length = 0

    if total_punct == 0:
        return 0.0
    return repeated / total_punct


def whitespace_burstiness(text: str) -> float:
    """Coefficient of variation of whitespace run lengths (clamped)."""

    runs: list[int] = []
    current = 0
    for ch in text:
        if ch.isspace():
            current += 1
        else:
            if current > 0:
                runs.append(current)
                current = 0
    if current > 0:
        runs.append(current)
    if len(runs) < 2:
        return 0.0
    cv = _coefficient_of_variation(runs)
    return min(cv, 10.0)


# ---------------------------------------------------------------------------
# Length-dependent metrics (disabled by default in the registry)
# ---------------------------------------------------------------------------


def total_words(text: str) -> float:
    return float(len(_tokenize_words(text)))


def total_sentences(text: str) -> float:
    return float(len(_split_sentences(text)))


def unique_words(text: str) -> float:
    return float(len(set(_tokenize_words(text, lowercase=True))))


def hapax_legomena_count(text: str) -> float:
    words = _tokenize_words(text, lowercase=True)
    counts = Counter(words)
    return float(sum(1 for value in counts.values() if value == 1))


def total_characters(text: str) -> float:
    return float(len(text))


__all__ = [
    "type_token_ratio",
    "herdan_c",
    "mattr",
    "mean_word_length",
    "mean_sentence_length",
    "word_length_cv",
    "sentence_length_cv",
    "char_entropy_normalized",
    "word_entropy_normalized",
    "punctuation_ratio",
    "lowercase_char_ratio",
    "uppercase_char_ratio",
    "digit_char_ratio",
    "whitespace_char_ratio",
    "other_char_ratio",
    "comma_ratio",
    "semicolon_ratio",
    "question_ratio",
    "exclamation_ratio",
    "period_ratio",
    "colon_ratio",
    "quote_ratio",
    "function_word_ratio",
    "capitalized_word_ratio",
    "max_char_run_ratio",
    "repeated_punctuation_ratio",
    "whitespace_burstiness",
    "total_words",
    "total_sentences",
    "unique_words",
    "hapax_legomena_count",
    "total_characters",
]
