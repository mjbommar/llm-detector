"""Tests for formal human text data sources."""

import pytest

from llm_detector.training.sources.formal import (
    LegalDocumentsSource,
    ScientificPapersSource,
    TechnicalDocsSource,
)
from llm_detector.types import TextSample


class TestLegalDocumentsSource:
    """Test legal documents dataset source."""

    def test_initialization(self):
        """Test source can be initialized."""
        source = LegalDocumentsSource()
        assert source.name == "legal_docs"
        assert source.category == "human"
        assert source.min_length == 100  # Legal docs are longer
        assert source.max_length == 10000

    def test_initialization_with_params(self):
        """Test source initialization with custom parameters."""
        source = LegalDocumentsSource(
            min_length=200,
            max_length=5000,
            doc_types=["contracts"]
        )
        assert source.min_length == 200
        assert source.max_length == 5000
        assert source.doc_types == ["contracts"]

    @pytest.mark.integration
    def test_sample_generation(self):
        """Test that source generates valid human legal samples."""
        source = LegalDocumentsSource(min_length=100, max_length=1000)

        try:
            samples = list(source.__iter__())[:3]

            # May return empty if datasets not available
            if len(samples) > 0:
                for sample in samples:
                    assert isinstance(sample, TextSample)
                    assert sample.is_llm is False  # Human content
                    assert sample.source == "legal_docs"
                    assert 100 <= len(sample.text) <= 10000
                    assert sample.metadata["domain"] == "legal"
                    assert sample.metadata["formality"] == "high"
        except RuntimeError as e:
            pytest.skip(f"Dataset not available: {e}")


class TestScientificPapersSource:
    """Test scientific papers dataset source."""

    def test_initialization(self):
        """Test source can be initialized."""
        source = ScientificPapersSource()
        assert source.name == "scientific_papers"
        assert source.category == "human"
        assert source.use_abstracts is False  # Full text by default

    def test_initialization_with_params(self):
        """Test source initialization with custom parameters."""
        source = ScientificPapersSource(
            min_length=50,
            max_length=2000,
            use_abstracts=True
        )
        assert source.min_length == 50
        assert source.max_length == 2000
        assert source.use_abstracts is True

    @pytest.mark.integration
    def test_sample_generation(self):
        """Test that source generates valid human scientific samples."""
        source = ScientificPapersSource(
            min_length=100,
            max_length=1000,
            use_abstracts=True
        )

        try:
            samples = list(source.__iter__())[:3]

            # May return empty if datasets not available
            if len(samples) > 0:
                for sample in samples:
                    assert isinstance(sample, TextSample)
                    assert sample.is_llm is False  # Human content
                    assert sample.source == "scientific_papers"
                    assert 100 <= len(sample.text) <= 1000
                    assert sample.metadata["domain"] == "academic"
                    assert sample.metadata["formality"] == "high"
                    assert "content_type" in sample.metadata
        except RuntimeError as e:
            pytest.skip(f"Dataset not available: {e}")

    @pytest.mark.integration
    def test_latex_cleaning(self):
        """Test that LaTeX formatting is cleaned from papers."""
        source = ScientificPapersSource()

        try:
            samples = list(source.__iter__())[:10]
            for sample in samples:
                # Basic check that common LaTeX commands are cleaned
                assert "\\section{" not in sample.text
                assert "\\begin{abstract}" not in sample.text
                assert "\\end{abstract}" not in sample.text
        except (RuntimeError, StopIteration):
            pytest.skip("Dataset not available")


class TestTechnicalDocsSource:
    """Test technical documentation dataset source."""

    def test_initialization(self):
        """Test source can be initialized."""
        source = TechnicalDocsSource()
        assert source.name == "technical_docs"
        assert source.category == "human"

    def test_initialization_with_params(self):
        """Test source initialization with custom parameters."""
        source = TechnicalDocsSource(min_length=100, max_length=2000)
        assert source.min_length == 100
        assert source.max_length == 2000

    @pytest.mark.integration
    def test_sample_generation(self):
        """Test that source generates valid human technical samples."""
        source = TechnicalDocsSource(min_length=50, max_length=1000)

        try:
            samples = list(source.__iter__())[:3]

            if len(samples) > 0:
                for sample in samples:
                    assert isinstance(sample, TextSample)
                    assert sample.is_llm is False  # Human content
                    assert sample.source == "technical_docs"
                    assert 50 <= len(sample.text) <= 1000
                    assert sample.metadata["domain"] == "technical"
                    assert sample.metadata["formality"] == "medium-high"
        except RuntimeError as e:
            pytest.skip(f"Dataset not available: {e}")


class TestFormalSourceCategories:
    """Test that all formal sources are properly categorized as human."""

    def test_all_sources_marked_as_human(self):
        """Ensure all formal sources are marked as human."""
        sources = [
            LegalDocumentsSource(),
            ScientificPapersSource(),
            TechnicalDocsSource(),
        ]

        for source in sources:
            assert source.category == "human"

    @pytest.mark.integration
    def test_all_sources_generate_human_samples(self):
        """Integration test that all sources produce is_llm=False."""
        sources = [
            LegalDocumentsSource(min_length=50),
            ScientificPapersSource(min_length=50),
            TechnicalDocsSource(min_length=50),
        ]

        for source in sources:
            try:
                # These sources might return empty iterators if datasets unavailable
                samples = list(source.__iter__())[:1]
                if samples:
                    sample = samples[0]
                    assert sample.is_llm is False, f"{source.name} should produce is_llm=False"
                    assert sample.source == source.name
            except (RuntimeError, StopIteration):
                # Dataset might not be available
                pytest.skip(f"Dataset not available for {source.name}")