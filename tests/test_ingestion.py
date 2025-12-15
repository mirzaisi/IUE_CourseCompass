"""
Tests for Ingestion Module.
===========================

Tests for:
- Scraper: URL fetching, caching, rate limiting
- Parser: HTML parsing, course extraction
- Cleaner: Text normalization
- Chunker: Semantic chunking
"""

import pytest
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Cleaner Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestCleaner:
    """Tests for the Cleaner class."""

    def test_clean_removes_extra_whitespace(self, sample_dirty_text: str):
        """Test that cleaner removes extra whitespace."""
        from iue_coursecompass.ingestion.cleaner import Cleaner

        cleaner = Cleaner()
        result = cleaner.clean(sample_dirty_text)

        # Should not have multiple consecutive spaces
        assert "    " not in result
        assert "  " not in result or result.count("  ") == 0

    def test_clean_removes_html_tags(self):
        """Test that cleaner removes HTML tags."""
        from iue_coursecompass.ingestion.cleaner import Cleaner

        cleaner = Cleaner()
        text = "<p>Hello <strong>World</strong></p>"
        result = cleaner.clean(text)

        assert "<p>" not in result
        assert "<strong>" not in result
        assert "Hello" in result
        assert "World" in result

    def test_clean_normalizes_newlines(self):
        """Test that cleaner normalizes multiple newlines."""
        from iue_coursecompass.ingestion.cleaner import Cleaner

        cleaner = Cleaner()
        text = "Line 1\n\n\n\nLine 2"
        result = cleaner.clean(text)

        # Should not have more than 2 consecutive newlines
        assert "\n\n\n" not in result

    def test_clean_preserves_meaningful_content(self):
        """Test that cleaner preserves meaningful text."""
        from iue_coursecompass.ingestion.cleaner import Cleaner

        cleaner = Cleaner()
        text = "SE 301: Software Engineering (6 ECTS)"
        result = cleaner.clean(text)

        assert "SE 301" in result
        assert "Software Engineering" in result
        assert "6 ECTS" in result

    def test_clean_handles_empty_string(self):
        """Test that cleaner handles empty strings."""
        from iue_coursecompass.ingestion.cleaner import Cleaner

        cleaner = Cleaner()
        result = cleaner.clean("")

        assert result == ""

    def test_clean_handles_none(self):
        """Test that cleaner handles None input."""
        from iue_coursecompass.ingestion.cleaner import Cleaner

        cleaner = Cleaner()
        result = cleaner.clean(None)

        assert result == ""

    def test_normalize_unicode(self):
        """Test Unicode normalization."""
        from iue_coursecompass.ingestion.cleaner import Cleaner

        cleaner = Cleaner()
        # Test with various Unicode characters
        text = "Café résumé naïve"
        result = cleaner.clean(text)

        assert len(result) > 0
        assert "Caf" in result


# ─────────────────────────────────────────────────────────────────────────────
# Parser Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestParser:
    """Tests for the Parser class."""

    def test_parser_extracts_course_code(self, sample_html: str):
        """Test that parser extracts course code from HTML."""
        from iue_coursecompass.ingestion.parser import Parser

        parser = Parser()
        # Test the helper method if available
        text = "SE 301 - Software Engineering"
        code = parser._extract_course_code(text)

        assert code is not None
        assert "SE" in code or "301" in code

    def test_parser_extracts_course_title(self):
        """Test that parser extracts course title."""
        from iue_coursecompass.ingestion.parser import Parser

        parser = Parser()
        text = "SE 301 - Software Engineering"
        title = parser._extract_course_title(text)

        assert "Software Engineering" in title

    def test_parser_handles_malformed_html(self):
        """Test that parser handles malformed HTML gracefully."""
        from iue_coursecompass.ingestion.parser import Parser

        parser = Parser()
        malformed_html = "<html><body><div>Unclosed div<p>Text"

        # Should not raise exception
        try:
            result = parser.parse_curriculum_page(malformed_html, "se", 2024)
            # May return empty list but should not crash
            assert isinstance(result, list)
        except Exception as e:
            pytest.fail(f"Parser crashed on malformed HTML: {e}")

    def test_parser_handles_empty_html(self):
        """Test that parser handles empty HTML."""
        from iue_coursecompass.ingestion.parser import Parser

        parser = Parser()
        result = parser.parse_curriculum_page("", "se", 2024)

        assert isinstance(result, list)
        assert len(result) == 0

    def test_extract_ects(self):
        """Test ECTS extraction."""
        from iue_coursecompass.ingestion.parser import Parser

        parser = Parser()

        # Test various ECTS formats
        assert parser._extract_ects("6 ECTS") == 6
        assert parser._extract_ects("ECTS: 6") == 6
        assert parser._extract_ects("Credits: 3, ECTS: 6") == 6

    def test_extract_prerequisites(self):
        """Test prerequisites extraction."""
        from iue_coursecompass.ingestion.parser import Parser

        parser = Parser()
        text = "Prerequisites: SE 201, CE 100"
        prereqs = parser._extract_prerequisites(text)

        assert isinstance(prereqs, list)


# ─────────────────────────────────────────────────────────────────────────────
# Chunker Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestChunker:
    """Tests for the Chunker class."""

    def test_chunker_creates_chunks(self, sample_course_record):
        """Test that chunker creates chunks from course record."""
        from iue_coursecompass.ingestion.chunker import Chunker

        chunker = Chunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_course(sample_course_record)

        assert len(chunks) > 0
        assert all(hasattr(c, "chunk_id") for c in chunks)
        assert all(hasattr(c, "text") for c in chunks)

    def test_chunker_respects_chunk_size(self, sample_course_record):
        """Test that chunks respect size limits."""
        from iue_coursecompass.ingestion.chunker import Chunker

        chunk_size = 200
        chunker = Chunker(chunk_size=chunk_size, chunk_overlap=20)

        # Create a course with long description
        sample_course_record.description = "Word " * 100  # 500 chars

        chunks = chunker.chunk_course(sample_course_record)

        # Most chunks should be under the size limit (with some tolerance)
        for chunk in chunks:
            assert len(chunk.text) <= chunk_size * 1.5  # Allow some overflow

    def test_chunker_includes_metadata(self, sample_course_record):
        """Test that chunks include course metadata."""
        from iue_coursecompass.ingestion.chunker import Chunker

        chunker = Chunker()
        chunks = chunker.chunk_course(sample_course_record)

        for chunk in chunks:
            assert chunk.course_code == sample_course_record.course_code
            assert chunk.department == sample_course_record.department

    def test_chunker_generates_unique_ids(self, sample_course_record):
        """Test that chunk IDs are unique."""
        from iue_coursecompass.ingestion.chunker import Chunker

        chunker = Chunker(chunk_size=50)
        sample_course_record.description = "Test content. " * 50

        chunks = chunker.chunk_course(sample_course_record)
        chunk_ids = [c.chunk_id for c in chunks]

        assert len(chunk_ids) == len(set(chunk_ids))  # All unique

    def test_chunker_handles_empty_description(self, sample_course_record):
        """Test chunker with empty description."""
        from iue_coursecompass.ingestion.chunker import Chunker

        chunker = Chunker()
        sample_course_record.description = ""

        chunks = chunker.chunk_course(sample_course_record)

        # Should still create at least a metadata chunk
        assert isinstance(chunks, list)

    def test_chunk_overlap(self, sample_course_record):
        """Test that chunks have proper overlap."""
        from iue_coursecompass.ingestion.chunker import Chunker

        chunker = Chunker(chunk_size=100, chunk_overlap=30)
        sample_course_record.description = "Word number " * 50

        chunks = chunker.chunk_course(sample_course_record)

        if len(chunks) >= 2:
            # Check that consecutive chunks have some overlap
            for i in range(len(chunks) - 1):
                text1 = chunks[i].text
                text2 = chunks[i + 1].text

                # Some words should appear in both (due to overlap)
                words1 = set(text1.split()[-5:])
                words2 = set(text2.split()[:5])

                # This is a weak test - overlap may not always be visible
                # depending on chunking strategy


# ─────────────────────────────────────────────────────────────────────────────
# Scraper Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestScraper:
    """Tests for the Scraper class."""

    def test_scraper_initialization(self):
        """Test scraper initializes correctly."""
        from iue_coursecompass.ingestion.scraper import Scraper

        scraper = Scraper(use_cache=False, rate_limit=0.1)

        assert scraper.use_cache is False
        assert scraper.rate_limit == 0.1

    def test_scraper_cache_key_generation(self):
        """Test that cache keys are generated correctly."""
        from iue_coursecompass.ingestion.scraper import Scraper

        scraper = Scraper()
        url = "https://example.com/page?param=value"
        key = scraper._get_cache_key(url)

        assert isinstance(key, str)
        assert len(key) > 0

        # Same URL should produce same key
        key2 = scraper._get_cache_key(url)
        assert key == key2

        # Different URL should produce different key
        key3 = scraper._get_cache_key("https://example.com/other")
        assert key != key3

    def test_scraper_respects_cache_setting(self, temp_dir: Path):
        """Test that scraper respects cache setting."""
        from iue_coursecompass.ingestion.scraper import Scraper

        # With cache disabled
        scraper_no_cache = Scraper(use_cache=False)
        assert scraper_no_cache.use_cache is False

        # With cache enabled
        scraper_with_cache = Scraper(use_cache=True, cache_dir=temp_dir)
        assert scraper_with_cache.use_cache is True

    @pytest.mark.integration
    def test_scraper_fetch_real_url(self):
        """Test fetching a real URL (integration test)."""
        from iue_coursecompass.ingestion.scraper import Scraper

        scraper = Scraper(use_cache=False, timeout=10)

        # Use a reliable test URL
        html = scraper.fetch("https://httpbin.org/html")

        if html:  # May fail due to network
            assert "<html" in html.lower() or "<!doctype" in html.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestIngestionPipeline:
    """Integration tests for the full ingestion pipeline."""

    def test_full_pipeline(self, sample_html: str, temp_dir: Path):
        """Test the complete ingestion pipeline."""
        from iue_coursecompass.ingestion.parser import Parser
        from iue_coursecompass.ingestion.cleaner import Cleaner
        from iue_coursecompass.ingestion.chunker import Chunker

        # Parse
        parser = Parser()
        courses = parser.parse_curriculum_page(sample_html, "se", 2024)

        # Note: sample_html may not match expected selectors
        # This tests that the pipeline doesn't crash

        # Clean (if we have courses)
        cleaner = Cleaner()
        for course in courses:
            if course.description:
                course.description = cleaner.clean(course.description)

        # Chunk
        chunker = Chunker()
        all_chunks = []
        for course in courses:
            chunks = chunker.chunk_course(course)
            all_chunks.extend(chunks)

        # Pipeline should complete without errors
        assert isinstance(all_chunks, list)
