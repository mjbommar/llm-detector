import pytest

from llm_detector.aggregation import (
    DEFAULT_AGGREGATORS,
    apply_aggregators,
    length_weighted_mean,
    logit_weighted_mean,
    mean,
    median,
    trimmed_mean,
    vote_fraction,
)


def test_length_weighted_mean_prioritises_long_sentences():
    scores = [0.1, 0.9]
    lengths = [10, 1]
    result = length_weighted_mean(scores, lengths)
    assert result == pytest.approx((0.1 * 10 + 0.9 * 1) / 11)


def test_trimmed_mean_rejects_outliers():
    scores = [0.05, 0.5, 0.55, 0.6, 0.95]
    weights = [1, 1, 1, 1, 1]
    trimmed = trimmed_mean(scores, weights, trim_ratio=0.2)
    assert trimmed == pytest.approx((0.5 + 0.55 + 0.6) / 3)


def test_trimmed_mean_requires_valid_ratio():
    with pytest.raises(ValueError):
        trimmed_mean([0.1, 0.2], [1, 1], trim_ratio=0.5)


def test_median_balances_even_counts():
    assert median([0.1, 0.9]) == pytest.approx(0.5)
    assert median([0.1, 0.4, 0.9]) == pytest.approx(0.4)


def test_vote_fraction_uses_threshold():
    scores = [0.2, 0.4, 0.85, 0.91]
    lengths = [10, 10, 5, 5]
    fraction = vote_fraction(scores, lengths, threshold=0.8)
    assert fraction == pytest.approx((5 + 5) / sum(lengths))


def test_logit_weighted_mean_combines_confident_scores():
    scores = [0.2, 0.8]
    lengths = [1, 1]
    result = logit_weighted_mean(scores, lengths)
    # logit mean of 0.2 and 0.8 is 0.5
    assert result == pytest.approx(0.5)


@pytest.mark.parametrize("scores", [[], [0.3, 0.7]])
def test_apply_aggregators_returns_all(scores):
    lengths = [len(scores) or 1] * len(scores)
    output = apply_aggregators(scores, lengths)
    assert set(output) == set(DEFAULT_AGGREGATORS)
    for value in output.values():
        assert 0.0 <= value <= 1.0


def test_mean_handles_empty_scores():
    assert mean([]) == pytest.approx(0.5)
    assert mean([0.2, 0.8]) == pytest.approx(0.5)
