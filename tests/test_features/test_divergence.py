from llm_detector.features.divergence import (
    char_divergence_score,
    punct_divergence_score,
    regex_divergence_score,
)


def test_divergence_scores_are_in_range():
    text = "Hello, world! Numbers 123. Another sentence?"
    assert 0 <= char_divergence_score(text) <= 1
    assert 0 <= punct_divergence_score(text) <= 1
    assert 0 <= regex_divergence_score(text) <= 1
