"""
RAG Module - Retrieval-Augmented Generation pipeline.
=====================================================

This module implements the complete RAG workflow:

- retriever: Query embedding and chunk retrieval from vector store
- prompts: System and user prompt templates for grounded generation
- generator: LLM-based answer generation with Gemini
- grounding: Citation extraction and hallucination prevention
- quantitative: Counting and ECTS queries using structured data

RAG Flow:
    Query → Retriever → Relevant Chunks → Prompt Builder → LLM → Grounded Answer
"""

from iue_coursecompass.rag.retriever import Retriever, retrieve
from iue_coursecompass.rag.prompts import (
    PromptBuilder,
    build_rag_prompt,
    build_comparison_prompt,
)
from iue_coursecompass.rag.generator import Generator, generate_answer
from iue_coursecompass.rag.grounding import (
    GroundingChecker,
    GroundingResult,
    check_grounding,
    is_grounded,
)
from iue_coursecompass.rag.quantitative import (
    QuantitativeHandler,
    QuantitativeQueryType,
    is_quantitative_query,
    detect_quantitative_query,
    count_courses,
    sum_ects,
)

__all__ = [
    # Retriever
    "Retriever",
    "retrieve",
    # Prompts
    "PromptBuilder",
    "build_rag_prompt",
    "build_comparison_prompt",
    # Generator
    "Generator",
    "generate_answer",
    # Grounding
    "GroundingChecker",
    "GroundingResult",
    "check_grounding",
    "is_grounded",
    # Quantitative
    "QuantitativeHandler",
    "QuantitativeQueryType",
    "is_quantitative_query",
    "detect_quantitative_query",
    "count_courses",
    "sum_ects",
]
