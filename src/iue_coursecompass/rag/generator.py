"""
Generator Module - LLM generation for RAG answers.
===================================================

Provides grounded generation using Gemini API with:
- Streaming support
- Citation enforcement
- Configurable generation parameters
- Error handling and retries
"""

import os
from typing import Iterator, Optional

from tenacity import retry, stop_after_attempt, wait_exponential

from iue_coursecompass.shared.config import get_settings
from iue_coursecompass.shared.logging import get_logger
from iue_coursecompass.shared.schemas import AnswerResponse, Citation, RetrievalHit
from iue_coursecompass.rag.prompts import PromptBuilder

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Generator Class
# ─────────────────────────────────────────────────────────────────────────────


class Generator:
    """
    LLM generator for RAG answers using Gemini API.

    Supports:
    - Standard and streaming generation
    - Configurable temperature and parameters
    - Automatic prompt building
    - Citation tracking

    Example:
        >>> generator = Generator()
        >>> response = generator.generate(query, hits)
        >>> print(response.answer)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the generator.

        Args:
            model_name: Gemini model name (default from config)
            temperature: Generation temperature (default from config)
            max_tokens: Maximum tokens to generate (default from config)
            api_key: Gemini API key (default from env)
        """
        settings = get_settings()
        gen_config = settings.generation

        self.model_name = model_name or gen_config.model_name
        self.temperature = temperature if temperature is not None else gen_config.temperature
        self.max_tokens = max_tokens or gen_config.max_output_tokens
        self.api_key = api_key or settings.gemini_api_key or os.getenv("GEMINI_API_KEY")

        self._client = None
        self._model = None
        self.prompt_builder = PromptBuilder()

        logger.info(
            f"Generator initialized: model={self.model_name}, "
            f"temp={self.temperature}, max_tokens={self.max_tokens}"
        )

    @property
    def client(self):
        """Lazy-load Gemini client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "Gemini API key not found. Set GEMINI_API_KEY environment variable."
                )
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                self._client = genai
                logger.debug("Gemini client initialized")
            except ImportError:
                raise ImportError(
                    "google-generativeai package not installed. "
                    "Run: pip install google-generativeai"
                )
        return self._client

    @property
    def model(self):
        """Lazy-load Gemini model."""
        if self._model is None:
            self._model = self.client.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                    "top_p": 0.95,
                    "top_k": 40,
                },
            )
            logger.debug(f"Gemini model loaded: {self.model_name}")
        return self._model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _generate_content(
        self,
        system_prompt: str,
        user_prompt: str,
        stream: bool = False,
    ):
        """
        Generate content with retry logic.

        Args:
            system_prompt: System instructions
            user_prompt: User message with context
            stream: Whether to stream response

        Returns:
            Generated content or stream iterator
        """
        # Combine prompts (Gemini uses a single prompt style)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        if stream:
            return self.model.generate_content(full_prompt, stream=True)
        else:
            return self.model.generate_content(full_prompt)

    def generate(
        self,
        query: str,
        hits: list[RetrievalHit],
        low_confidence: bool = False,
    ) -> AnswerResponse:
        """
        Generate a grounded answer for a query.

        Args:
            query: User query
            hits: Retrieved chunks for context
            low_confidence: Whether retrieval confidence is low

        Returns:
            AnswerResponse with generated answer and metadata
        """
        logger.info(f"Generating answer for: {query[:50]}...")

        # Build prompt
        if low_confidence:
            system_prompt, user_prompt = self.prompt_builder.build_trap_aware_prompt(
                query, hits, low_confidence=True
            )
        else:
            system_prompt, user_prompt = self.prompt_builder.build_prompt(query, hits)

        # Generate
        try:
            response = self._generate_content(system_prompt, user_prompt)
            answer_text = response.text

            # Extract cited chunk IDs from answer
            cited_ids = self._extract_citations(answer_text, hits)
            
            # Build citation objects from cited hits
            cited_hits = [h for h in hits if h.chunk_id in cited_ids]
            citations = self._build_citations(cited_hits)

            logger.info(f"Generated answer with {len(citations)} citations")
            
            # Calculate similarity stats
            scores = [h.score for h in hits] if hits else [0.0]

            return AnswerResponse(
                query=query,
                answer=answer_text,
                citations=citations,
                retrieval_count=len(hits),
                max_similarity=max(scores),
                min_similarity=min(scores),
                is_grounded=len(citations) > 0,
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return AnswerResponse(
                query=query,
                answer=f"I encountered an error generating the answer: {str(e)}",
                citations=[],
                is_grounded=False,
            )

    def generate_stream(
        self,
        query: str,
        hits: list[RetrievalHit],
    ) -> Iterator[str]:
        """
        Generate answer with streaming.

        Args:
            query: User query
            hits: Retrieved chunks

        Yields:
            Answer text chunks
        """
        logger.info(f"Streaming answer for: {query[:50]}...")

        system_prompt, user_prompt = self.prompt_builder.build_prompt(query, hits)

        try:
            response = self._generate_content(system_prompt, user_prompt, stream=True)

            for chunk in response:
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"Error generating answer: {str(e)}"

    def generate_comparison(
        self,
        query: str,
        dept_hits: dict[str, list[RetrievalHit]],
    ) -> AnswerResponse:
        """
        Generate a comparison across departments.

        Args:
            query: Comparison query
            dept_hits: Dictionary of department -> hits

        Returns:
            AnswerResponse with comparison answer
        """
        logger.info(f"Generating comparison for: {query[:50]}...")

        system_prompt, user_prompt = self.prompt_builder.build_comparison_prompt(
            query, dept_hits
        )

        try:
            response = self._generate_content(system_prompt, user_prompt)
            answer_text = response.text

            # Flatten all hits for source tracking
            all_hits = []
            for hits in dept_hits.values():
                all_hits.extend(hits)

            cited_ids = self._extract_citations(answer_text, all_hits)
            cited_hits = [h for h in all_hits if h.chunk_id in cited_ids]
            citations = self._build_citations(cited_hits)

            logger.info(f"Generated comparison with {len(citations)} citations")
            
            # Calculate similarity stats
            scores = [h.score for h in all_hits] if all_hits else [0.0]

            return AnswerResponse(
                query=query,
                answer=answer_text,
                citations=citations,
                retrieval_count=len(all_hits),
                max_similarity=max(scores),
                min_similarity=min(scores),
                is_grounded=len(citations) > 0,
            )

        except Exception as e:
            logger.error(f"Comparison generation failed: {e}")
            return AnswerResponse(
                query=query,
                answer=f"I encountered an error generating the comparison: {str(e)}",
                citations=[],
                is_grounded=False,
            )

    def generate_quantitative(
        self,
        query: str,
        data_context: str,
        source_hits: list[RetrievalHit],
    ) -> AnswerResponse:
        """
        Generate answer for quantitative queries.

        Args:
            query: Quantitative query (counting, ECTS, etc.)
            data_context: Formatted data for counting
            source_hits: Original source hits

        Returns:
            AnswerResponse with quantitative answer
        """
        logger.info(f"Generating quantitative answer for: {query[:50]}...")

        system_prompt, user_prompt = self.prompt_builder.build_quantitative_prompt(
            query, data_context
        )

        try:
            response = self._generate_content(system_prompt, user_prompt)
            answer_text = response.text

            cited_ids = self._extract_citations(answer_text, source_hits)
            cited_hits = [h for h in source_hits if h.chunk_id in cited_ids]
            citations = self._build_citations(cited_hits)
            
            # Calculate similarity stats
            scores = [h.score for h in source_hits] if source_hits else [0.0]

            return AnswerResponse(
                query=query,
                answer=answer_text,
                citations=citations,
                retrieval_count=len(source_hits),
                max_similarity=max(scores),
                min_similarity=min(scores),
                is_grounded=True,  # Quantitative answers are grounded in data
                is_quantitative=True,
            )

        except Exception as e:
            logger.error(f"Quantitative generation failed: {e}")
            return AnswerResponse(
                query=query,
                answer=f"I encountered an error generating the answer: {str(e)}",
                citations=[],
                is_grounded=False,
            )

    def _build_citations(self, hits: list[RetrievalHit]) -> list[Citation]:
        """
        Build Citation objects from RetrievalHit objects.

        Args:
            hits: List of RetrievalHit objects to convert

        Returns:
            List of Citation objects
        """
        citations = []
        for hit in hits:
            citation = Citation(
                chunk_id=hit.chunk_id,
                course_code=hit.course_code,
                course_title=hit.course_title,
                source_url=hit.source_url,
                text_snippet=hit.text[:200] if hit.text else "",  # First 200 chars
            )
            citations.append(citation)
        return citations

    def _extract_citations(
        self,
        answer: str,
        hits: list[RetrievalHit],
    ) -> set[str]:
        """
        Extract cited chunk IDs from generated answer.

        Looks for patterns like [chunk_id] in the answer.

        Args:
            answer: Generated answer text
            hits: Available chunks that could be cited

        Returns:
            Set of cited chunk IDs
        """
        import re

        cited = set()
        hit_ids = {h.chunk_id for h in hits}

        # Find all bracket citations [xxx]
        citations = re.findall(r"\[([^\]]+)\]", answer)

        for citation in citations:
            # Clean up the citation
            citation = citation.strip()

            # Check if it's a valid chunk ID
            if citation in hit_ids:
                cited.add(citation)

        return cited


# ─────────────────────────────────────────────────────────────────────────────
# Global Instance
# ─────────────────────────────────────────────────────────────────────────────


_generator: Optional[Generator] = None


def get_generator() -> Generator:
    """Get or create global generator instance."""
    global _generator
    if _generator is None:
        _generator = Generator()
    return _generator


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────


def generate_answer(
    query: str,
    hits: list[RetrievalHit],
    low_confidence: bool = False,
) -> AnswerResponse:
    """
    Generate answer for a query.

    Convenience function.

    Args:
        query: User query
        hits: Retrieved chunks
        low_confidence: Whether retrieval confidence is low

    Returns:
        AnswerResponse
    """
    generator = get_generator()
    return generator.generate(query, hits, low_confidence=low_confidence)


def generate_comparison(
    query: str,
    dept_hits: dict[str, list[RetrievalHit]],
) -> AnswerResponse:
    """
    Generate comparison across departments.

    Convenience function.

    Args:
        query: Comparison query
        dept_hits: Department -> hits mapping

    Returns:
        AnswerResponse
    """
    generator = get_generator()
    return generator.generate_comparison(query, dept_hits)


def stream_answer(
    query: str,
    hits: list[RetrievalHit],
) -> Iterator[str]:
    """
    Stream answer generation.

    Convenience function.

    Args:
        query: User query
        hits: Retrieved chunks

    Yields:
        Answer text chunks
    """
    generator = get_generator()
    yield from generator.generate_stream(query, hits)
