"""Core data structures shared across the detector package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class TextSample:
    """A unit of text with provenance metadata.

    Attributes:
        text: Raw text payload for downstream feature extraction.
        is_llm: Whether the text originates from an LLM (True) or a human corpus (False).
        source: Short identifier for the upstream corpus or dataset.
        sample_id: Optional stable identifier supplied by the source implementation.
        metadata: Additional structured metadata emitted by the source.
    """

    text: str
    is_llm: bool
    source: str
    sample_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("TextSample.text must be a non-empty string")
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("TextSample.source must be a non-empty string")
        if not isinstance(self.metadata, dict):
            raise TypeError("TextSample.metadata must be a dictionary")


__all__ = ["TextSample"]
