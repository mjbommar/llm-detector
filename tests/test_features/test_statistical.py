import math

from llm_detector.features.statistical import (
    capitalized_word_ratio,
    char_entropy_normalized,
    colon_ratio,
    comma_ratio,
    digit_char_ratio,
    exclamation_ratio,
    function_word_ratio,
    hapax_legomena_count,
    herdan_c,
    lowercase_char_ratio,
    mattr,
    max_char_run_ratio,
    mean_sentence_length,
    mean_word_length,
    other_char_ratio,
    period_ratio,
    punctuation_ratio,
    question_ratio,
    quote_ratio,
    repeated_punctuation_ratio,
    semicolon_ratio,
    sentence_length_cv,
    total_sentences,
    total_words,
    type_token_ratio,
    unique_words,
    uppercase_char_ratio,
    whitespace_burstiness,
    whitespace_char_ratio,
    word_entropy_normalized,
    word_length_cv,
)


def test_mattr_scale_invariance():
    base = "The quick brown fox jumps over the lazy dog."
    short = base
    medium = base * 10
    long = base * 100
    v = [mattr(t, window=9) for t in (short, medium, long)]
    mean = sum(v) / len(v)
    cv = (math.sqrt(sum((x - mean) ** 2 for x in v) / len(v)) / mean) if mean > 0 else 0
    assert cv < 0.05


def test_basic_statistical_functions_do_not_error():
    text = "Hello world! This is a test text, with punctuation; and variety."
    assert mean_word_length(text) > 0
    assert mean_sentence_length(text) > 0
    assert 0 <= word_length_cv(text) <= 10
    assert 0 <= sentence_length_cv(text) <= 10
    assert 0 <= char_entropy_normalized(text) <= 1
    assert 0 <= word_entropy_normalized(text) <= 1
    assert 0 <= punctuation_ratio(text) <= 1
    assert 0 <= comma_ratio(text) <= punctuation_ratio(text)
    assert 0 <= semicolon_ratio(text) <= punctuation_ratio(text)
    assert 0 <= question_ratio(text) <= punctuation_ratio(text)
    assert 0 <= exclamation_ratio(text) <= punctuation_ratio(text)
    assert 0 <= period_ratio(text) <= punctuation_ratio(text)
    assert 0 <= colon_ratio(text) <= punctuation_ratio(text)
    assert 0 <= quote_ratio(text) <= 1
    assert 0 <= function_word_ratio(text) <= 1
    assert 0 <= capitalized_word_ratio(text) <= 1
    assert 0 <= lowercase_char_ratio(text) <= 1
    assert 0 <= uppercase_char_ratio(text) <= 1
    assert 0 <= digit_char_ratio(text) <= 1
    assert 0 <= whitespace_char_ratio(text) <= 1
    assert 0 <= max_char_run_ratio(text) <= 1
    assert 0 <= repeated_punctuation_ratio(text) <= 1
    assert 0 <= whitespace_burstiness(text) <= 10


def test_type_token_ratio_and_herdan_bounds():
    text = "Dogs dogs dogs and cats." * 5
    assert 0 <= type_token_ratio(text) <= 1
    assert 0 <= herdan_c(text) <= 1


def test_length_dependent_metrics_match_counts():
    text = "Alpha beta gamma. Second sentence!"
    assert total_words(text) == 5
    assert total_sentences(text) == 2
    assert unique_words(text) == 5
    assert hapax_legomena_count(text) == 5


def test_function_word_ratio_controls():
    text = "The quick brown fox jumps over the lazy dog"
    ratio = function_word_ratio(text)
    assert 0 < ratio < 1


def test_char_class_ratios_sum_to_one():
    text = "Hello World 123!!"
    total = (
        lowercase_char_ratio(text)
        + uppercase_char_ratio(text)
        + digit_char_ratio(text)
        + whitespace_char_ratio(text)
        + punctuation_ratio(text)
        + other_char_ratio(text)
    )
    assert abs(total - 1.0) < 1e-6
