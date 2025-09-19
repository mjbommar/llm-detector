"""Tests for 2024-2025 LLM data sources."""

import pytest

from llm_detector.training.sources.llm_2024 import (
    Claude35SonnetSource,
    Claude3OpusSource,
    GPT4AlpacaSource,
    MixtralInstructSource,
)
from llm_detector.types import TextSample


class TestGPT4AlpacaSource:
    """Test GPT-4 Alpaca dataset source."""

    def test_initialization(self):
        """Test source can be initialized."""
        source = GPT4AlpacaSource()
        assert source.name == "gpt4_alpaca"
        assert source.category == "llm"
        assert source.min_length == 10

    def test_initialization_with_params(self):
        """Test source initialization with custom parameters."""
        source = GPT4AlpacaSource(min_length=50, max_length=1000)
        assert source.min_length == 50
        assert source.max_length == 1000

    @pytest.mark.integration
    def test_sample_generation(self):
        """Test that source generates valid LLM samples."""
        source = GPT4AlpacaSource(min_length=20)
        samples = list(source.__iter__())[:3]  # Get first 3 samples

        assert len(samples) > 0
        for sample in samples:
            assert isinstance(sample, TextSample)
            assert sample.is_llm is True
            assert sample.source == "gpt4_alpaca"
            assert len(sample.text) >= 20
            assert "model" in sample.metadata
            assert sample.metadata["model"] == "gpt-4"

    @pytest.mark.integration
    def test_field_extraction(self):
        """Test that the correct field (output) is extracted."""
        source = GPT4AlpacaSource()
        sample = next(iter(source))

        # The text should be from the 'output' field, not 'instruction' or 'text'
        assert sample.text
        assert sample.is_llm is True
        # Should not contain the instruction template
        assert "Below is an instruction" not in sample.text


class TestClaude3OpusSource:
    """Test Claude 3 Opus dataset source."""

    def test_initialization(self):
        """Test source can be initialized."""
        source = Claude3OpusSource()
        assert source.name == "claude3_opus"
        assert source.category == "llm"

    @pytest.mark.integration
    def test_sample_generation(self):
        """Test that source generates valid Claude 3 samples."""
        source = Claude3OpusSource(min_length=20)
        samples = list(source.__iter__())[:3]

        assert len(samples) > 0
        for sample in samples:
            assert isinstance(sample, TextSample)
            assert sample.is_llm is True
            assert sample.source == "claude3_opus"
            assert len(sample.text) >= 20
            assert sample.metadata["model"] == "claude-3-opus"

    @pytest.mark.integration
    def test_field_extraction(self):
        """Test that the correct field (response) is extracted."""
        source = Claude3OpusSource()
        sample = next(iter(source))

        # Should extract the 'response' field
        assert sample.text
        assert sample.is_llm is True


class TestClaude35SonnetSource:
    """Test Claude 3.5 Sonnet dataset source."""

    def test_initialization(self):
        """Test source can be initialized."""
        source = Claude35SonnetSource()
        assert source.name == "claude35_sonnet"
        assert source.category == "llm"

    @pytest.mark.integration
    def test_sample_generation(self):
        """Test that source generates valid Claude 3.5 samples."""
        source = Claude35SonnetSource(min_length=20)

        try:
            samples = list(source.__iter__())[:3]
            assert len(samples) > 0
            for sample in samples:
                assert isinstance(sample, TextSample)
                assert sample.is_llm is True
                assert sample.source == "claude35_sonnet"
                assert sample.metadata["model"] == "claude-3.5-sonnet"
        except RuntimeError as e:
            # Dataset might not be available
            pytest.skip(f"Dataset not available: {e}")


class TestMixtralInstructSource:
    """Test Mixtral instruction dataset source."""

    def test_initialization(self):
        """Test source can be initialized."""
        source = MixtralInstructSource()
        assert source.name == "mixtral_instruct"
        assert source.category == "llm"

    @pytest.mark.integration
    def test_sample_generation(self):
        """Test that source generates valid Mixtral samples."""
        source = MixtralInstructSource(min_length=20, max_length=500)
        samples = list(source.__iter__())[:3]

        assert len(samples) > 0
        for sample in samples:
            assert isinstance(sample, TextSample)
            assert sample.is_llm is True
            assert sample.source == "mixtral_instruct"
            assert 20 <= len(sample.text) <= 500
            assert sample.metadata["model"] == "mixtral-8x7b"

    @pytest.mark.integration
    def test_field_extraction(self):
        """Test that the correct field (text) is extracted from Cosmopedia."""
        source = MixtralInstructSource()
        sample = next(iter(source))

        # Should extract the 'text' field from Cosmopedia
        assert sample.text
        assert sample.is_llm is True
        # Text is capped at 10000 chars in the source
        assert len(sample.text) <= 10000


class TestLLMSourceCategories:
    """Test that all LLM sources are properly categorized."""

    def test_all_sources_marked_as_llm(self):
        """Ensure all LLM sources generate is_llm=True samples."""
        sources = [
            GPT4AlpacaSource(),
            Claude3OpusSource(),
            Claude35SonnetSource(),
            MixtralInstructSource(),
        ]

        for source in sources:
            assert source.category == "llm"

    @pytest.mark.integration
    def test_all_sources_generate_llm_samples(self):
        """Integration test that all sources produce is_llm=True."""
        sources = [
            GPT4AlpacaSource(min_length=10),
            Claude3OpusSource(min_length=10),
            MixtralInstructSource(min_length=10, max_length=500),
        ]

        for source in sources:
            try:
                sample = next(iter(source))
                assert sample.is_llm is True, f"{source.name} should produce is_llm=True"
                assert sample.source == source.name
            except (RuntimeError, StopIteration):
                # Dataset might not be available
                pytest.skip(f"Dataset not available for {source.name}")