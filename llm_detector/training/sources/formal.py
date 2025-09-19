"""Formal human text sources (legal, academic, technical)."""

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


class LegalDocumentsSource(BaseDataSource):
    """US legal documents including contracts, court opinions, and legislation.

    From the Pile of Law dataset (256GB), containing diverse legal texts
    with formal structure and technical language.
    """

    def __init__(
        self,
        *,
        batch_config: BatchConfig | None = None,
        min_length: int = 100,  # Legal docs are typically longer
        max_length: int | None = 10000,  # Cap for training efficiency
        doc_types: list[str] | None = None,
    ) -> None:
        super().__init__(config=batch_config)
        self.min_length = min_length
        self.max_length = max_length
        self.doc_types = doc_types or ["contracts", "court_opinions", "legislation"]
        self._dataset_name = "pile-of-law/pile-of-law"

    @property
    def name(self) -> str:
        return "legal_docs"

    @property
    def category(self) -> str:
        return "human"

    def _generate_samples(self, limit: int | None = None) -> Iterator[TextSample]:
        """Stream legal documents."""
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise RuntimeError(f"{self.name}: datasets library required") from e

        # The Pile of Law has multiple configurations
        configs_to_try = [
            "atticus_contracts",  # Contracts
            "courtlistener_opinions",  # Court opinions
            "federal_register",  # Federal regulations
            "cfpb_cc",  # Credit card agreements
            "irs_legal_advice",  # IRS memos
        ]

        # Try to load one of the configurations
        dataset = None
        for config in configs_to_try:
            try:
                dataset = load_dataset(
                    self._dataset_name,
                    config,
                    split="train",
                    streaming=True,
                )
                logger.info(f"Loaded legal dataset config: {config}")
                self._current_config = config
                break
            except Exception as e:
                logger.debug(f"Could not load config {config}: {e}")
                continue

        if dataset is None:
            # Fallback to simpler legal dataset
            alt_dataset = "pile-of-law/pile-of-law"
            logger.info(f"Trying alternative with different config: {alt_dataset}")
            try:
                # Try a simple configuration that should work
                dataset = load_dataset(
                    "monology/pile-uncopyrighted",  # Alternative public domain texts
                    split="train",
                    streaming=True,
                )
                self._dataset_name = "monology/pile-uncopyrighted"
                self._current_config = "legal_subset"
                logger.info("Using pile-uncopyrighted as fallback for legal texts")
            except Exception as e2:
                logger.error(f"Could not load legal dataset, skipping: {e2}")
                # Return empty iterator instead of failing
                return iter([])

        count = 0
        for idx, item in enumerate(dataset):
            if limit is not None and count >= limit:
                break

            # Extract text content
            text = item.get("text", item.get("content", "")).strip()
            if not text:
                continue

            # Take a reasonable chunk if very long
            if self.max_length and len(text) > self.max_length:
                # Take from the middle for contracts (important clauses)
                # or beginning for opinions (summary/holding)
                if "contract" in self._current_config.lower():
                    start = max(0, (len(text) - self.max_length) // 2)
                    text = text[start:start + self.max_length]
                else:
                    text = text[:self.max_length]

            if len(text) < self.min_length:
                continue

            cleaned = text.strip()
            if not cleaned or len(cleaned) < self.min_length:
                continue

            # Categorize document type
            doc_type = "unknown"
            if "contract" in self._current_config.lower():
                doc_type = "contract"
            elif "opinion" in self._current_config.lower():
                doc_type = "court_opinion"
            elif "register" in self._current_config.lower():
                doc_type = "regulation"
            elif "irs" in self._current_config.lower():
                doc_type = "tax_guidance"

            sample_id = f"{self.name}_{idx:08d}"
            text_hash = hashlib.md5(cleaned.encode("utf-8")).hexdigest()[:8]

            yield TextSample(
                text=cleaned,
                source=self.name,
                is_llm=False,
                metadata={
                    "dataset": self._dataset_name,
                    "config": self._current_config,
                    "sample_id": sample_id,
                    "text_hash": text_hash,
                    "doc_type": doc_type,
                    "domain": "legal",
                    "formality": "high",
                },
            )
            count += 1


class ScientificPapersSource(BaseDataSource):
    """Scientific papers from ArXiv and PubMed.

    Academic papers with citations, abstracts, and technical content.
    Provides formal, technical human writing.
    """

    def __init__(
        self,
        *,
        batch_config: BatchConfig | None = None,
        min_length: int = 100,
        max_length: int | None = 10000,
        use_abstracts: bool = False,  # Use abstracts or full text
    ) -> None:
        super().__init__(config=batch_config)
        self.min_length = min_length
        self.max_length = max_length
        self.use_abstracts = use_abstracts
        self._dataset_name = "armanc/scientific_papers"

    @property
    def name(self) -> str:
        return "scientific_papers"

    @property
    def category(self) -> str:
        return "human"

    def _generate_samples(self, limit: int | None = None) -> Iterator[TextSample]:
        """Stream scientific papers."""
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise RuntimeError(f"{self.name}: datasets library required") from e

        # Try both ArXiv and PubMed configurations
        for config in ["arxiv", "pubmed"]:
            try:
                dataset = load_dataset(
                    self._dataset_name,
                    config,
                    split="train",
                    streaming=True,
                )
                logger.info(f"Loaded scientific papers from {config}")
                self._current_source = config
                break
            except Exception as e:
                logger.debug(f"Could not load {config}: {e}")
                if config == "pubmed":
                    # Both failed, return empty
                    logger.error(f"Could not load any scientific papers dataset, skipping")
                    return iter([])
                continue

        count = 0
        for idx, item in enumerate(dataset):
            if limit is not None and count >= limit:
                break

            # Choose abstract or full text
            if self.use_abstracts:
                text = item.get("abstract", "").strip()
            else:
                # Try article text, fallback to abstract
                text = item.get("article", item.get("text", "")).strip()
                if not text:
                    text = item.get("abstract", "").strip()

            if not text:
                continue

            # Handle sectioned papers (LaTeX format)
            if "\\section" in text or "\\begin{" in text:
                # Basic LaTeX cleaning
                text = text.replace("\\section{", "\n\n")
                text = text.replace("\\subsection{", "\n")
                text = text.replace("}", "")
                text = text.replace("\\begin{abstract}", "Abstract: ")
                text = text.replace("\\end{abstract}", "\n")
                # Remove other LaTeX commands (simplified)
                import re
                text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
                text = re.sub(r'\\[a-zA-Z]+', '', text)

            # Take a reasonable chunk if very long
            if self.max_length and len(text) > self.max_length:
                # For papers, take introduction + conclusion if possible
                text = text[:self.max_length]

            if len(text) < self.min_length:
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
                    "source_db": self._current_source,
                    "sample_id": sample_id,
                    "text_hash": text_hash,
                    "content_type": "abstract" if self.use_abstracts else "full_text",
                    "domain": "academic",
                    "formality": "high",
                },
            )
            count += 1


class TechnicalDocsSource(BaseDataSource):
    """Technical documentation and manuals.

    Formal technical writing from documentation,
    specifications, and technical guides.
    """

    def __init__(
        self,
        *,
        batch_config: BatchConfig | None = None,
        min_length: int = 50,
        max_length: int | None = 5000,
    ) -> None:
        super().__init__(config=batch_config)
        self.min_length = min_length
        self.max_length = max_length
        # Could use various technical documentation datasets
        self._dataset_name = "legacy-datasets/wikihow"  # How-to guides as proxy

    @property
    def name(self) -> str:
        return "technical_docs"

    @property
    def category(self) -> str:
        return "human"

    def _generate_samples(self, limit: int | None = None) -> Iterator[TextSample]:
        """Stream technical documentation."""
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise RuntimeError(f"{self.name}: datasets library required") from e

        try:
            dataset = load_dataset(
                self._dataset_name,
                "all",
                split="train",
                streaming=True,
            )
        except Exception as e:
            logger.error(f"Failed to load {self._dataset_name}: {e}")
            # Fallback to a different technical dataset
            raise RuntimeError(f"Could not load technical docs dataset") from e

        count = 0
        for idx, item in enumerate(dataset):
            if limit is not None and count >= limit:
                break

            # Extract instructional content
            text = item.get("text", "").strip()
            if not text:
                continue

            # Skip very short instructions
            if len(text) < self.min_length:
                continue

            # Cap length if needed
            if self.max_length and len(text) > self.max_length:
                text = text[:self.max_length]

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
                    "domain": "technical",
                    "formality": "medium-high",
                    "content_type": "instructions",
                },
            )
            count += 1


# Registry helper
def register_formal_sources(registry):
    """Register all formal text sources."""
    from .registry import SourceDefinition

    sources = [
        SourceDefinition(
            name="legal_docs",
            factory=LegalDocumentsSource,
            category="human",
            description="US legal documents (contracts, opinions, regulations)",
            enabled=True,
        ),
        SourceDefinition(
            name="scientific_papers",
            factory=ScientificPapersSource,
            category="human",
            description="Scientific papers from ArXiv and PubMed",
            enabled=True,
        ),
        SourceDefinition(
            name="technical_docs",
            factory=TechnicalDocsSource,
            category="human",
            description="Technical documentation and guides",
            enabled=False,  # Optional, disabled by default
        ),
    ]

    for source_def in sources:
        registry.register(source_def.name, source_def)


__all__ = [
    "LegalDocumentsSource",
    "ScientificPapersSource",
    "TechnicalDocsSource",
    "register_formal_sources",
]