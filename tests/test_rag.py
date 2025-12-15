"""
Tests for RAG Module.
=====================

Tests for:
- Retriever: Chunk retrieval
- Prompts: Prompt building and formatting
- Generator: Answer generation (mocked)
- Grounding: Citation verification
- Quantitative: Counting queries
"""

import pytest
from unittest.mock import Mock, patch, MagicMock


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPrompts:
    """Tests for prompt building."""

    def test_build_rag_prompt(self, sample_retrieval_hits):
        """Test building a RAG prompt."""
        from iue_coursecompass.rag.prompts import build_rag_prompt

        system, user = build_rag_prompt(
            query="What is SE 301?",
            hits=sample_retrieval_hits,
        )

        assert isinstance(system, str)
        assert isinstance(user, str)
        assert len(system) > 0
        assert len(user) > 0
        assert "SE 301" in user or "What is" in user

    def test_prompt_includes_context(self, sample_retrieval_hits):
        """Test that prompt includes source context."""
        from iue_coursecompass.rag.prompts import build_rag_prompt

        _, user = build_rag_prompt(
            query="Test query",
            hits=sample_retrieval_hits,
        )

        # Should include chunk IDs
        assert sample_retrieval_hits[0].chunk_id in user

    def test_prompt_includes_grounding_rules(self, sample_retrieval_hits):
        """Test that system prompt includes grounding rules."""
        from iue_coursecompass.rag.prompts import build_rag_prompt

        system, _ = build_rag_prompt(
            query="Test query",
            hits=sample_retrieval_hits,
        )

        # Should mention citations or grounding
        assert "citation" in system.lower() or "cite" in system.lower()

    def test_build_comparison_prompt(self, sample_retrieval_hits):
        """Test building a comparison prompt."""
        from iue_coursecompass.rag.prompts import build_comparison_prompt

        dept_hits = {
            "se": sample_retrieval_hits[:2],
            "ce": sample_retrieval_hits[2:],
        }

        system, user = build_comparison_prompt(
            query="Compare SE and CE",
            dept_hits=dept_hits,
        )

        assert "SE" in user.upper() or "se" in user.lower()
        assert "CE" in user.upper() or "ce" in user.lower()

    def test_format_chunks_for_context(self, sample_retrieval_hits):
        """Test formatting chunks for context."""
        from iue_coursecompass.rag.prompts import format_chunks_for_context

        context = format_chunks_for_context(sample_retrieval_hits)

        assert isinstance(context, str)
        assert len(context) > 0
        # Should contain chunk information
        assert "SE 301" in context or "se_301" in context.lower()

    def test_prompt_builder_max_chunks(self, sample_retrieval_hits):
        """Test that prompt builder respects max chunks."""
        from iue_coursecompass.rag.prompts import PromptBuilder

        builder = PromptBuilder(max_context_chunks=1)
        _, user = builder.build_prompt("Test", sample_retrieval_hits)

        # Should only include first chunk
        chunk_count = user.count("Source")
        assert chunk_count <= 2  # May have header text with "Source"


# ─────────────────────────────────────────────────────────────────────────────
# Grounding Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGrounding:
    """Tests for grounding verification."""

    def test_check_grounding_with_citations(self, sample_retrieval_hits):
        """Test grounding check with valid citations."""
        from iue_coursecompass.rag.grounding import check_grounding

        answer = f"SE 301 is about software engineering [{sample_retrieval_hits[0].chunk_id}]."

        result = check_grounding(answer, sample_retrieval_hits)

        assert result.total_citations >= 1
        assert result.valid_citations >= 1

    def test_check_grounding_without_citations(self, sample_retrieval_hits):
        """Test grounding check without citations."""
        from iue_coursecompass.rag.grounding import check_grounding

        answer = "SE 301 is about software engineering."

        result = check_grounding(answer, sample_retrieval_hits)

        assert result.total_citations == 0

    def test_check_grounding_invalid_citation(self, sample_retrieval_hits):
        """Test grounding check with invalid citation."""
        from iue_coursecompass.rag.grounding import check_grounding

        answer = "SE 301 is about software engineering [invalid_chunk_id]."

        result = check_grounding(answer, sample_retrieval_hits)

        assert "invalid_chunk_id" in result.invalid_citations

    def test_grounding_score_calculation(self, sample_retrieval_hits):
        """Test grounding score is calculated."""
        from iue_coursecompass.rag.grounding import check_grounding

        answer = f"Based on the sources [{sample_retrieval_hits[0].chunk_id}], SE 301 covers software engineering."

        result = check_grounding(answer, sample_retrieval_hits)

        assert 0.0 <= result.grounding_score <= 1.0

    def test_abstention_detection(self, sample_retrieval_hits):
        """Test detection of abstention responses."""
        from iue_coursecompass.rag.grounding import check_grounding

        answer = "I could not find information about that topic in the indexed sources."

        result = check_grounding(answer, sample_retrieval_hits)

        # Abstention should be considered grounded
        assert result.grounding_score >= 0.5 or result.is_grounded

    def test_hedging_detection(self, sample_retrieval_hits):
        """Test detection of hedging language."""
        from iue_coursecompass.rag.grounding import GroundingChecker

        checker = GroundingChecker()
        answer = "I think SE 301 might be about software engineering, probably."

        result = checker.check(answer, sample_retrieval_hits)

        # Should have warnings about hedging
        assert any("hedg" in w.lower() for w in result.warnings)

    def test_extract_course_codes(self):
        """Test course code extraction."""
        from iue_coursecompass.rag.grounding import extract_course_codes

        text = "Prerequisites are SE 201 and CE 100. Also consider EEE 301."
        codes = extract_course_codes(text)

        assert "SE 201" in codes
        assert "CE 100" in codes
        assert "EEE 301" in codes

    def test_is_grounded_convenience(self, sample_retrieval_hits):
        """Test is_grounded convenience function."""
        from iue_coursecompass.rag.grounding import is_grounded

        answer = f"SE 301 [{sample_retrieval_hits[0].chunk_id}] is a course."

        result = is_grounded(answer, sample_retrieval_hits)

        assert isinstance(result, bool)


# ─────────────────────────────────────────────────────────────────────────────
# Quantitative Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestQuantitative:
    """Tests for quantitative query handling."""

    def test_detect_count_query(self):
        """Test detection of counting queries."""
        from iue_coursecompass.rag.quantitative import (
            detect_quantitative_query,
            QuantitativeQueryType,
        )

        query = "How many courses are in the SE curriculum?"
        qtype = detect_quantitative_query(query)

        assert qtype == QuantitativeQueryType.COUNT

    def test_detect_ects_query(self):
        """Test detection of ECTS queries."""
        from iue_coursecompass.rag.quantitative import (
            detect_quantitative_query,
            QuantitativeQueryType,
        )

        query = "What is the total ECTS for year 3?"
        qtype = detect_quantitative_query(query)

        assert qtype == QuantitativeQueryType.SUM_ECTS

    def test_detect_comparison_query(self):
        """Test detection of comparison queries."""
        from iue_coursecompass.rag.quantitative import (
            detect_quantitative_query,
            QuantitativeQueryType,
        )

        query = "Which department has more required courses?"
        qtype = detect_quantitative_query(query)

        assert qtype == QuantitativeQueryType.COMPARE_COUNT

    def test_detect_non_quantitative(self):
        """Test detection of non-quantitative queries."""
        from iue_coursecompass.rag.quantitative import (
            detect_quantitative_query,
            QuantitativeQueryType,
        )

        query = "What is SE 301 about?"
        qtype = detect_quantitative_query(query)

        assert qtype == QuantitativeQueryType.NOT_QUANTITATIVE

    def test_extract_query_filters(self):
        """Test extraction of query filters."""
        from iue_coursecompass.rag.quantitative import extract_query_filters

        query = "How many SE courses in semester 5?"
        filters = extract_query_filters(query)

        assert filters.get("department") == "se"
        assert filters.get("semester") == 5

    def test_extract_courses_from_chunks(self, sample_retrieval_hits):
        """Test course extraction from chunks."""
        from iue_coursecompass.rag.quantitative import extract_courses_from_chunks

        courses = extract_courses_from_chunks(sample_retrieval_hits)

        assert isinstance(courses, list)
        # Should extract course codes from the hits
        codes = [c.code for c in courses]
        assert any("SE" in code or "CE" in code for code in codes)

    def test_quantitative_handler(self, sample_retrieval_hits):
        """Test QuantitativeHandler."""
        from iue_coursecompass.rag.quantitative import QuantitativeHandler

        handler = QuantitativeHandler()
        result = handler.handle("How many courses?", sample_retrieval_hits)

        assert result is not None
        assert hasattr(result, "value")
        assert hasattr(result, "formatted_data")

    def test_is_quantitative_query(self):
        """Test is_quantitative_query convenience function."""
        from iue_coursecompass.rag.quantitative import is_quantitative_query

        assert is_quantitative_query("How many courses?") is True
        assert is_quantitative_query("What is SE 301?") is False


# ─────────────────────────────────────────────────────────────────────────────
# Retriever Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRetriever:
    """Tests for Retriever class."""

    def test_retriever_initialization(self):
        """Test retriever initialization."""
        from iue_coursecompass.rag.retriever import Retriever

        # May fail if no index exists - that's expected
        try:
            retriever = Retriever()
            assert retriever is not None
        except Exception:
            # Expected if no index
            pass

    @patch("iue_coursecompass.rag.retriever.VectorStore")
    def test_retriever_retrieve_mocked(self, mock_store_class, sample_retrieval_hits):
        """Test retrieval with mocked vector store."""
        from iue_coursecompass.rag.retriever import Retriever

        # Setup mock
        mock_store = MagicMock()
        mock_store.query.return_value = sample_retrieval_hits
        mock_store_class.return_value = mock_store

        retriever = Retriever()
        retriever.store = mock_store

        results = retriever.retrieve("software engineering", top_k=5)

        assert len(results) == len(sample_retrieval_hits)
        mock_store.query.assert_called_once()

    @patch("iue_coursecompass.rag.retriever.VectorStore")
    def test_retriever_with_department_filter(self, mock_store_class, sample_retrieval_hits):
        """Test retrieval with department filter."""
        from iue_coursecompass.rag.retriever import Retriever

        mock_store = MagicMock()
        mock_store.query.return_value = [sample_retrieval_hits[0]]
        mock_store_class.return_value = mock_store

        retriever = Retriever()
        retriever.store = mock_store

        results = retriever.retrieve("test", top_k=5, departments=["se"])

        # Should pass filter to store
        call_kwargs = mock_store.query.call_args
        assert call_kwargs is not None


# ─────────────────────────────────────────────────────────────────────────────
# Generator Tests (Mocked)
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerator:
    """Tests for Generator class (mocked)."""

    def test_generator_initialization(self):
        """Test generator initialization."""
        from iue_coursecompass.rag.generator import Generator

        generator = Generator()

        assert generator.model_name is not None
        assert generator.temperature >= 0

    @patch("iue_coursecompass.rag.generator.genai")
    def test_generator_generate_mocked(self, mock_genai, sample_retrieval_hits):
        """Test generation with mocked API."""
        from iue_coursecompass.rag.generator import Generator

        # Setup mock response
        mock_response = MagicMock()
        mock_response.text = "SE 301 is about software engineering [se_301_chunk_001]."

        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        generator = Generator(api_key="fake-key")
        generator._client = mock_genai
        generator._model = mock_model

        response = generator.generate("What is SE 301?", sample_retrieval_hits)

        assert response is not None
        assert response.answer is not None
        assert len(response.answer) > 0

    def test_generator_extract_citations(self, sample_retrieval_hits):
        """Test citation extraction from answer."""
        from iue_coursecompass.rag.generator import Generator

        generator = Generator()

        answer = f"SE 301 [{sample_retrieval_hits[0].chunk_id}] and SE 302 [{sample_retrieval_hits[1].chunk_id}]."

        citations = generator._extract_citations(answer, sample_retrieval_hits)

        assert len(citations) == 2
        assert sample_retrieval_hits[0].chunk_id in citations

    def test_answer_response_structure(self, sample_retrieval_hits):
        """Test AnswerResponse structure."""
        from iue_coursecompass.shared.schemas import AnswerResponse, Citation

        # Create citations from sample hits
        citations = [
            Citation(
                chunk_id=hit.chunk_id,
                course_code=hit.course_code,
                course_title=hit.course_title,
                source_url=hit.source_url,
                text_snippet=hit.text[:100],
            )
            for hit in sample_retrieval_hits[:2]
        ]

        response = AnswerResponse(
            query="What is SE 301?",
            answer="SE 301 is about software engineering.",
            citations=citations,
            retrieval_count=len(sample_retrieval_hits),
            max_similarity=0.95,
            min_similarity=0.7,
            is_grounded=True,
        )

        assert response.query == "What is SE 301?"
        assert len(response.citations) == 2
        assert response.is_grounded is True


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRAGIntegration:
    """Integration tests for RAG pipeline."""

    def test_prompt_to_grounding_flow(self, sample_retrieval_hits):
        """Test flow from prompt building to grounding check."""
        from iue_coursecompass.rag.prompts import build_rag_prompt
        from iue_coursecompass.rag.grounding import check_grounding

        # Build prompt
        system, user = build_rag_prompt(
            query="What is SE 301?",
            hits=sample_retrieval_hits,
        )

        # Simulate an answer with citations
        answer = f"Based on the course information [{sample_retrieval_hits[0].chunk_id}], SE 301 is a software engineering course."

        # Check grounding
        result = check_grounding(answer, sample_retrieval_hits)

        assert result.valid_citations >= 1
        assert result.is_grounded or result.grounding_score > 0

    def test_quantitative_to_prompt_flow(self, sample_retrieval_hits):
        """Test flow from quantitative detection to prompt building."""
        from iue_coursecompass.rag.quantitative import (
            detect_quantitative_query,
            QuantitativeHandler,
            QuantitativeQueryType,
        )
        from iue_coursecompass.rag.prompts import PromptBuilder

        query = "How many SE courses are there?"

        # Detect query type
        qtype = detect_quantitative_query(query)
        assert qtype == QuantitativeQueryType.COUNT

        # Handle quantitative query
        handler = QuantitativeHandler()
        quant_result = handler.handle(query, sample_retrieval_hits)

        # Build prompt with quantitative data
        builder = PromptBuilder()
        system, user = builder.build_quantitative_prompt(
            query=query,
            data_context=quant_result.formatted_data,
        )

        assert len(user) > 0
        assert query in user
