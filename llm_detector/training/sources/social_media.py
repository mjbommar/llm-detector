"""Social media and informal human text sources."""

from __future__ import annotations

import hashlib
import logging
import random
from typing import TYPE_CHECKING

from llm_detector.types import TextSample
from .base import BaseDataSource, BatchConfig

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


class RedditRecentSource(BaseDataSource):
    """Recent Reddit posts and comments (2024-2025).

    Contains 388M+ posts and comments from November 2024 to January 2025.
    Focuses on human-generated content from various subreddits.
    """

    def __init__(
        self,
        *,
        batch_config: BatchConfig | None = None,
        min_length: int = 10,
        max_length: int | None = None,
        posts_only: bool = True,
    ) -> None:
        super().__init__(config=batch_config)
        self.min_length = min_length
        self.max_length = max_length
        self.posts_only = posts_only
        self._dataset_name = "bit0/reddit_dataset_12"

    @property
    def name(self) -> str:
        return "reddit_recent"

    @property
    def category(self) -> str:
        return "human"

    def _generate_samples(self, limit: int | None = None) -> Iterator[TextSample]:
        """Stream recent Reddit content."""
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise RuntimeError(f"{self.name}: datasets library required") from e

        try:
            dataset = load_dataset(
                self._dataset_name,
                split="train",
                streaming=True,
            )
        except Exception as e:
            logger.error(f"Failed to load {self._dataset_name}: {e}")
            # Fallback to general Reddit dataset
            alt_dataset = "SocialGrep/the-reddit-dataset-dataset"
            logger.info(f"Trying alternative: {alt_dataset}")
            try:
                dataset = load_dataset(alt_dataset, "posts", split="train", streaming=True)
                self._dataset_name = alt_dataset
            except Exception as e2:
                raise RuntimeError(f"Could not load Reddit dataset") from e2

        count = 0
        for idx, item in enumerate(dataset):
            if limit is not None and count >= limit:
                break

            # Skip if posts_only and this is a comment
            if self.posts_only and item.get("type") == "comment":
                continue

            # Extract text content
            text = item.get("body", item.get("selftext", "")).strip()
            if not text or text == "[deleted]" or text == "[removed]":
                continue

            # Skip bot-like content
            if any(bot_marker in text.lower() for bot_marker in [
                "i am a bot",
                "this is a bot",
                "beep boop",
                "^(i am a bot)",
            ]):
                continue

            if len(text) < self.min_length:
                continue
            if self.max_length and len(text) > self.max_length:
                continue

            cleaned = text.strip()
            if not cleaned or len(cleaned) < self.min_length:
                continue

            sample_id = f"{self.name}_{idx:08d}"
            text_hash = hashlib.md5(cleaned.encode("utf-8")).hexdigest()[:8]

            yield TextSample(
                text=cleaned,
                source=self.name,
                is_llm=False,
                metadata={
                    "dataset": self._dataset_name,
                    "sample_id": sample_id,
                    "text_hash": text_hash,
                    "platform": "reddit",
                    "subreddit": item.get("subreddit", "unknown"),
                    "content_type": "post" if self.posts_only else item.get("type", "unknown"),
                },
            )
            count += 1


class TwitterSource(BaseDataSource):
    """Twitter/X posts for short-form human text.

    Real tweets from various Twitter datasets,
    providing authentic short-form human communication.
    """

    def __init__(
        self,
        *,
        batch_config: BatchConfig | None = None,
        min_length: int = 10,
        max_length: int = 280,  # Twitter limit
    ) -> None:
        super().__init__(config=batch_config)
        self.min_length = min_length
        self.max_length = max_length
        self._dataset_name = "cardiffnlp/tweet_eval"

    @property
    def name(self) -> str:
        return "twitter"

    @property
    def category(self) -> str:
        return "human"

    def _generate_samples(self, limit: int | None = None) -> Iterator[TextSample]:
        """Stream Twitter posts."""
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise RuntimeError(f"{self.name}: datasets library required") from e

        # Try multiple Twitter datasets
        dataset_options = [
            ("cardiffnlp/tweet_eval", "emoji", "train"),
            ("cardiffnlp/tweet_eval", "emotion", "train"),
            ("cardiffnlp/tweet_eval", "sentiment", "train"),
            ("tner/tweetner7", None, "train"),
        ]

        dataset = None
        for dataset_name, config, split in dataset_options:
            try:
                if config:
                    dataset = load_dataset(dataset_name, config, split=split, streaming=True)
                else:
                    dataset = load_dataset(dataset_name, split=split, streaming=True)
                self._dataset_name = f"{dataset_name}/{config}" if config else dataset_name
                logger.info(f"Loaded Twitter dataset: {self._dataset_name}")
                break
            except Exception as e:
                logger.debug(f"Could not load {dataset_name}: {e}")
                continue

        if dataset is None:
            raise RuntimeError("Could not load any Twitter dataset")

        count = 0
        for idx, item in enumerate(dataset):
            if limit is not None and count >= limit:
                break

            # Extract tweet text
            text = item.get("text", item.get("tokens", ""))
            if isinstance(text, list):
                text = " ".join(text)
            text = text.strip()

            if not text:
                continue

            # Skip retweets (often start with RT @)
            if text.startswith("RT @"):
                continue

            # Convert Twitter entities back to normal text
            text = text.replace("{{URL}}", "[URL]")
            text = text.replace("@USER", "@user")

            if len(text) < self.min_length:
                continue
            if len(text) > self.max_length:
                continue

            cleaned = text.strip()
            if not cleaned or len(cleaned) < self.min_length:
                continue

            sample_id = f"{self.name}_{idx:08d}"
            text_hash = hashlib.md5(cleaned.encode("utf-8")).hexdigest()[:8]

            yield TextSample(
                text=cleaned,
                source=self.name,
                is_llm=False,
                metadata={
                    "dataset": self._dataset_name,
                    "sample_id": sample_id,
                    "text_hash": text_hash,
                    "platform": "twitter",
                    "length_category": "short",
                },
            )
            count += 1


class MultiPlatformSocialSource(BaseDataSource):
    """Multi-platform social media dataset.

    Combines content from various social media platforms
    captured in real-time during December 2024.
    """

    def __init__(
        self,
        *,
        batch_config: BatchConfig | None = None,
        min_length: int = 10,
        max_length: int | None = None,
    ) -> None:
        super().__init__(config=batch_config)
        self.min_length = min_length
        self.max_length = max_length
        self._dataset_name = "Exorde/exorde-social-media-december-2024-week1"

    @property
    def name(self) -> str:
        return "social_multi"

    @property
    def category(self) -> str:
        return "human"

    def _generate_samples(self, limit: int | None = None) -> Iterator[TextSample]:
        """Stream multi-platform social media content."""
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise RuntimeError(f"{self.name}: datasets library required") from e

        try:
            dataset = load_dataset(
                self._dataset_name,
                split="train",
                streaming=True,
            )
        except Exception as e:
            logger.error(f"Failed to load {self._dataset_name}: {e}")
            raise RuntimeError(f"Could not load dataset {self._dataset_name}") from e

        count = 0
        platforms_seen = set()

        for idx, item in enumerate(dataset):
            if limit is not None and count >= limit:
                break

            # Extract text content
            text = item.get("original_text", item.get("text", "")).strip()
            if not text:
                continue

            platform = item.get("platform", "unknown")
            platforms_seen.add(platform)

            # Skip if too short or too long
            if len(text) < self.min_length:
                continue
            if self.max_length and len(text) > self.max_length:
                continue

            # Categorize by length
            length_category = "short" if len(text) < 280 else "medium" if len(text) < 2000 else "long"

            cleaned = text.strip()
            if not cleaned or len(cleaned) < self.min_length:
                continue

            sample_id = f"{self.name}_{idx:08d}"
            text_hash = hashlib.md5(cleaned.encode("utf-8")).hexdigest()[:8]

            yield TextSample(
                text=cleaned,
                source=self.name,
                is_llm=False,
                metadata={
                    "dataset": self._dataset_name,
                    "sample_id": sample_id,
                    "text_hash": text_hash,
                    "platform": platform,
                    "length_category": length_category,
                    "date": item.get("date", "2024-12"),
                },
            )
            count += 1

        if platforms_seen:
            logger.info(f"Platforms seen: {sorted(platforms_seen)}")


# Registry helper
def register_social_media_sources(registry):
    """Register all social media sources."""
    from .registry import SourceDefinition

    sources = [
        SourceDefinition(
            name="reddit_recent",
            factory=RedditRecentSource,
            category="human",
            description="Recent Reddit posts (Nov 2024 - Jan 2025)",
            enabled=True,
        ),
        SourceDefinition(
            name="twitter",
            factory=TwitterSource,
            category="human",
            description="Twitter/X posts (short-form)",
            enabled=True,
        ),
        SourceDefinition(
            name="social_multi",
            factory=MultiPlatformSocialSource,
            category="human",
            description="Multi-platform social media (Dec 2024)",
            enabled=False,  # Large dataset, disabled by default
        ),
    ]

    for source_def in sources:
        registry.register(source_def.name, source_def)


__all__ = [
    "RedditRecentSource",
    "TwitterSource",
    "MultiPlatformSocialSource",
    "register_social_media_sources",
]