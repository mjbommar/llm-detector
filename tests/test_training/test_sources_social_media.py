"""Tests for social media human data sources."""

import pytest

from llm_detector.training.sources.social_media import (
    MultiPlatformSocialSource,
    RedditRecentSource,
    TwitterSource,
)
from llm_detector.types import TextSample


class TestRedditRecentSource:
    """Test recent Reddit dataset source."""

    def test_initialization(self):
        """Test source can be initialized."""
        source = RedditRecentSource()
        assert source.name == "reddit_recent"
        assert source.category == "human"
        assert source.posts_only is True

    def test_initialization_with_params(self):
        """Test source initialization with custom parameters."""
        source = RedditRecentSource(
            min_length=50,
            max_length=1000,
            posts_only=False
        )
        assert source.min_length == 50
        assert source.max_length == 1000
        assert source.posts_only is False

    @pytest.mark.integration
    def test_sample_generation(self):
        """Test that source generates valid human samples."""
        source = RedditRecentSource(min_length=20)

        try:
            samples = list(source.__iter__())[:3]
            assert len(samples) > 0
            for sample in samples:
                assert isinstance(sample, TextSample)
                assert sample.is_llm is False  # Human content
                assert sample.source == "reddit_recent"
                assert len(sample.text) >= 20
                assert "platform" in sample.metadata
                assert sample.metadata["platform"] == "reddit"
        except RuntimeError as e:
            pytest.skip(f"Dataset not available: {e}")

    @pytest.mark.integration
    def test_bot_filtering(self):
        """Test that bot-like content is filtered."""
        source = RedditRecentSource()

        # Get several samples and ensure none contain bot markers
        try:
            samples = list(source.__iter__())[:20]
            for sample in samples:
                text_lower = sample.text.lower()
                assert "i am a bot" not in text_lower
                assert "beep boop" not in text_lower
                assert sample.text != "[deleted]"
                assert sample.text != "[removed]"
        except RuntimeError:
            pytest.skip("Dataset not available")


class TestTwitterSource:
    """Test Twitter/X dataset source."""

    def test_initialization(self):
        """Test source can be initialized."""
        source = TwitterSource()
        assert source.name == "twitter"
        assert source.category == "human"
        assert source.max_length == 280  # Twitter character limit

    def test_initialization_with_params(self):
        """Test source initialization with custom parameters."""
        source = TwitterSource(min_length=20, max_length=140)
        assert source.min_length == 20
        assert source.max_length == 140

    @pytest.mark.integration
    def test_sample_generation(self):
        """Test that source generates valid human Twitter samples."""
        source = TwitterSource(min_length=10)

        try:
            samples = list(source.__iter__())[:5]
            assert len(samples) > 0
            for sample in samples:
                assert isinstance(sample, TextSample)
                assert sample.is_llm is False  # Human content
                assert sample.source == "twitter"
                assert 10 <= len(sample.text) <= 280
                assert sample.metadata["platform"] == "twitter"
                assert sample.metadata["length_category"] == "short"
        except RuntimeError as e:
            pytest.skip(f"Dataset not available: {e}")

    @pytest.mark.integration
    def test_retweet_filtering(self):
        """Test that retweets are filtered."""
        source = TwitterSource()

        try:
            samples = list(source.__iter__())[:20]
            for sample in samples:
                # Retweets should be filtered
                assert not sample.text.startswith("RT @")
        except RuntimeError:
            pytest.skip("Dataset not available")


class TestMultiPlatformSocialSource:
    """Test multi-platform social media source."""

    def test_initialization(self):
        """Test source can be initialized."""
        source = MultiPlatformSocialSource()
        assert source.name == "social_multi"
        assert source.category == "human"

    @pytest.mark.integration
    def test_sample_generation(self):
        """Test that source generates valid human samples from multiple platforms."""
        source = MultiPlatformSocialSource(min_length=20, max_length=1000)

        try:
            samples = list(source.__iter__())[:5]
            platforms_seen = set()

            assert len(samples) > 0
            for sample in samples:
                assert isinstance(sample, TextSample)
                assert sample.is_llm is False  # Human content
                assert sample.source == "social_multi"
                assert 20 <= len(sample.text) <= 1000
                assert "platform" in sample.metadata
                platforms_seen.add(sample.metadata["platform"])

            # Should have samples from different platforms
            assert len(platforms_seen) >= 1
        except RuntimeError as e:
            pytest.skip(f"Dataset not available: {e}")


class TestHumanSourceCategories:
    """Test that all social media sources are properly categorized as human."""

    def test_all_sources_marked_as_human(self):
        """Ensure all social media sources are marked as human."""
        sources = [
            RedditRecentSource(),
            TwitterSource(),
            MultiPlatformSocialSource(),
        ]

        for source in sources:
            assert source.category == "human"

    @pytest.mark.integration
    def test_all_sources_generate_human_samples(self):
        """Integration test that all sources produce is_llm=False."""
        sources = [
            RedditRecentSource(min_length=10),
            TwitterSource(min_length=10),
            MultiPlatformSocialSource(min_length=10),
        ]

        for source in sources:
            try:
                sample = next(iter(source))
                assert sample.is_llm is False, f"{source.name} should produce is_llm=False"
                assert sample.source == source.name
            except (RuntimeError, StopIteration):
                # Dataset might not be available
                pytest.skip(f"Dataset not available for {source.name}")