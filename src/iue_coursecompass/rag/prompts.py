"""
Prompts Module - RAG prompt templates for grounded generation.
=============================================================

Provides prompt templates that enforce:
- Citation requirements (cite sources by chunk_id)
- Grounding (only claim what's in sources)
- Hallucination prevention (abstain when unsure)
- Structured output for comparisons
"""

from typing import Optional

from iue_coursecompass.shared.logging import get_logger
from iue_coursecompass.shared.schemas import RetrievalHit

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# System Prompts
# ─────────────────────────────────────────────────────────────────────────────


SYSTEM_PROMPT_BASE = """You are an expert academic advisor assistant for Izmir University of Economics (IUE) Faculty of Engineering. Your role is to help students and faculty understand course information, curriculum structures, and academic requirements.

CRITICAL RULES YOU MUST FOLLOW:

1. GROUNDING REQUIREMENT: Base ALL your answers ONLY on the provided source chunks. Do not use external knowledge about courses, curricula, or the university.

2. CITATION REQUIREMENT: When making any factual claim, cite the source using [chunk_id]. Every statement about courses must have a citation.

3. HONESTY ABOUT UNCERTAINTY: If the provided sources do not contain information to answer the question:
   - Say "Based on the indexed sources, I could not find information about..."
   - Do NOT make up or guess information
   - Do NOT claim a course exists unless it appears in the sources

4. TRAP QUESTION HANDLING: If asked about something that doesn't exist in the sources:
   - Answer "No" or "Not found in the indexed sources"
   - Do NOT affirm the existence of non-existent courses or topics

5. FORMAT: Provide clear, well-structured answers. Use bullet points for lists. Always include course codes when mentioning courses.

Remember: It's better to say "I don't have information about that" than to provide potentially incorrect information."""


SYSTEM_PROMPT_COMPARISON = """You are an expert academic advisor assistant for Izmir University of Economics (IUE) Faculty of Engineering, specializing in cross-department curriculum analysis.

CRITICAL RULES:

1. GROUNDING: Base answers ONLY on provided source chunks. Each department's information comes from separate sources.

2. CITATIONS: Cite sources using [chunk_id] for every factual claim.

3. COMPARISON STRUCTURE: When comparing departments:
   - Provide a brief overview first
   - Then give department-by-department breakdown
   - Highlight similarities and differences
   - Use course codes with department prefixes

4. BALANCED ANALYSIS: Give fair treatment to each department. If information is missing for a department, explicitly state this.

5. NO FABRICATION: If you lack information about a department for comparison, say so. Don't invent courses or details.

Output Format for Comparisons:
- Start with a summary paragraph
- Then use headers for each department
- End with a comparison table if appropriate"""


SYSTEM_PROMPT_QUANTITATIVE = """You are an expert academic advisor assistant for Izmir University of Economics (IUE) Faculty of Engineering, specializing in curriculum statistics and course counting.

CRITICAL RULES:

1. ACCURACY: For counting questions, provide exact numbers based on the structured data provided.

2. SHOW YOUR WORK: List the specific courses/items being counted.

3. CITATIONS: Reference the source data for your counts.

4. VERIFICATION: Double-check your counts before responding.

5. SCOPE CLARITY: Clearly state what is being counted and any filters applied (department, year, type, etc.)."""


# ─────────────────────────────────────────────────────────────────────────────
# User Prompt Templates
# ─────────────────────────────────────────────────────────────────────────────


USER_PROMPT_TEMPLATE = """Based on the following source information about IUE Engineering courses, please answer the question.

=== SOURCE CHUNKS ===
{context}
=== END SOURCES ===

Question: {query}

Instructions:
- Answer based ONLY on the sources above
- Cite sources using [chunk_id] format
- If information is not in the sources, say so clearly
- Include course codes when mentioning specific courses"""


USER_PROMPT_COMPARISON_TEMPLATE = """Compare the following aspects across departments based on the source information provided.

=== SOURCES BY DEPARTMENT ===
{context}
=== END SOURCES ===

Comparison Question: {query}

Instructions:
- Analyze each department separately first
- Then provide comparative insights
- Cite sources using [chunk_id]
- Create a summary comparison
- If a department lacks relevant information, state this explicitly"""


USER_PROMPT_QUANTITATIVE_TEMPLATE = """Based on the following course data, answer the quantitative question.

=== COURSE DATA ===
{context}
=== END DATA ===

Question: {query}

Instructions:
- Provide the exact count/total requested
- List the specific items being counted
- Show the calculation if relevant
- Cite the data sources"""


# ─────────────────────────────────────────────────────────────────────────────
# Context Formatting
# ─────────────────────────────────────────────────────────────────────────────


def format_chunk_for_context(hit: RetrievalHit, include_score: bool = False) -> str:
    """Format a single chunk for inclusion in prompt context."""
    parts = [
        f"[{hit.chunk_id}]",
        f"Course: {hit.course_code} - {hit.course_title}",
        f"Department: {hit.department.upper()}",
    ]

    if hit.metadata.get("section_name"):
        parts.append(f"Section: {hit.metadata['section_name']}")

    if include_score:
        parts.append(f"Relevance: {hit.score:.3f}")

    parts.append(f"Content: {hit.text}")

    return "\n".join(parts)


def format_chunks_for_context(
    hits: list[RetrievalHit],
    include_scores: bool = False,
    max_chunks: Optional[int] = None,
) -> str:
    """Format multiple chunks for inclusion in prompt context."""
    if max_chunks:
        hits = hits[:max_chunks]

    formatted = []
    for i, hit in enumerate(hits, 1):
        chunk_text = format_chunk_for_context(hit, include_score=include_scores)
        formatted.append(f"--- Source {i} ---\n{chunk_text}")

    return "\n\n".join(formatted)


def format_chunks_by_department(
    dept_hits: dict[str, list[RetrievalHit]],
    include_scores: bool = False,
) -> str:
    """Format chunks grouped by department for comparison prompts."""
    sections = []

    for dept, hits in dept_hits.items():
        dept_name = dept.upper()
        if not hits:
            sections.append(f"=== {dept_name} ===\nNo relevant sources found for this department.")
            continue

        chunks_text = format_chunks_for_context(hits, include_scores=include_scores)
        sections.append(f"=== {dept_name} ===\n{chunks_text}")

    return "\n\n".join(sections)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Builder
# ─────────────────────────────────────────────────────────────────────────────


class PromptBuilder:
    """
    Builds prompts for RAG generation.

    Handles:
    - Standard single-query prompts
    - Comparison prompts (multi-department)
    - Quantitative prompts
    - Context formatting and truncation

    Example:
        >>> builder = PromptBuilder()
        >>> system, user = builder.build_prompt(query, hits)
    """

    def __init__(
        self,
        max_context_chunks: int = 15,
        include_scores: bool = False,
    ):
        """
        Initialize the prompt builder.

        Args:
            max_context_chunks: Maximum chunks to include in context
            include_scores: Whether to include relevance scores in context
        """
        self.max_context_chunks = max_context_chunks
        self.include_scores = include_scores

    def build_prompt(
        self,
        query: str,
        hits: list[RetrievalHit],
    ) -> tuple[str, str]:
        """
        Build a standard RAG prompt.

        Args:
            query: User query
            hits: Retrieved chunks

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        context = format_chunks_for_context(
            hits,
            include_scores=self.include_scores,
            max_chunks=self.max_context_chunks,
        )

        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=context,
            query=query,
        )

        return SYSTEM_PROMPT_BASE, user_prompt

    def build_comparison_prompt(
        self,
        query: str,
        dept_hits: dict[str, list[RetrievalHit]],
    ) -> tuple[str, str]:
        """
        Build a comparison prompt for multi-department queries.

        Args:
            query: Comparison query
            dept_hits: Dictionary of department -> hits

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        context = format_chunks_by_department(
            dept_hits,
            include_scores=self.include_scores,
        )

        user_prompt = USER_PROMPT_COMPARISON_TEMPLATE.format(
            context=context,
            query=query,
        )

        return SYSTEM_PROMPT_COMPARISON, user_prompt

    def build_quantitative_prompt(
        self,
        query: str,
        data_context: str,
    ) -> tuple[str, str]:
        """
        Build a prompt for quantitative queries.

        Args:
            query: Quantitative query
            data_context: Formatted data for counting

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        user_prompt = USER_PROMPT_QUANTITATIVE_TEMPLATE.format(
            context=data_context,
            query=query,
        )

        return SYSTEM_PROMPT_QUANTITATIVE, user_prompt

    def build_trap_aware_prompt(
        self,
        query: str,
        hits: list[RetrievalHit],
        low_confidence: bool = False,
    ) -> tuple[str, str]:
        """
        Build a prompt with extra trap question awareness.

        Used when retrieval confidence is low.

        Args:
            query: User query
            hits: Retrieved chunks (may be empty or low relevance)
            low_confidence: Whether retrieval confidence is low

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = SYSTEM_PROMPT_BASE

        if low_confidence:
            system_prompt += """

IMPORTANT: The retrieval system found limited relevant information for this query. This may be a trap question about something that doesn't exist. Be extra careful:
- If the sources don't clearly support the existence of something, say it was not found
- Do not assume courses or topics exist without evidence
- Prefer "Not found in indexed sources" over speculation"""

        context = format_chunks_for_context(
            hits,
            include_scores=True,  # Show scores to help judge relevance
            max_chunks=self.max_context_chunks,
        )

        if not hits:
            context = "No relevant sources were found for this query."

        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=context,
            query=query,
        )

        return system_prompt, user_prompt


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────


def build_rag_prompt(
    query: str,
    hits: list[RetrievalHit],
    max_chunks: int = 15,
) -> tuple[str, str]:
    """
    Build a standard RAG prompt.

    Convenience function.

    Args:
        query: User query
        hits: Retrieved chunks
        max_chunks: Maximum context chunks

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    builder = PromptBuilder(max_context_chunks=max_chunks)
    return builder.build_prompt(query, hits)


def build_comparison_prompt(
    query: str,
    dept_hits: dict[str, list[RetrievalHit]],
) -> tuple[str, str]:
    """
    Build a comparison prompt.

    Convenience function.

    Args:
        query: Comparison query
        dept_hits: Dictionary of department -> hits

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    builder = PromptBuilder()
    return builder.build_comparison_prompt(query, dept_hits)
