"""
Runner Module - Evaluation execution harness.
=============================================

Runs the full evaluation pipeline:
1. Load questions from question bank
2. Run retrieval for each question
3. Generate answers
4. Compute metrics
5. Generate reports
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from iue_coursecompass.shared.config import get_settings
from iue_coursecompass.shared.logging import get_logger
from iue_coursecompass.shared.schemas import RetrievalHit
from iue_coursecompass.shared.utils import save_json

from iue_coursecompass.evaluation.questions import (
    Question,
    QuestionBank,
    QuestionType,
)
from iue_coursecompass.evaluation.metrics import (
    RetrievalMetrics,
    AnswerMetrics,
    EvaluationMetrics,
    RetrievalResult,
    AnswerResult,
    aggregate_retrieval_metrics,
    aggregate_answer_metrics,
)
from iue_coursecompass.rag.grounding import check_grounding

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result Classes
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class QuestionResult:
    """Result for a single question."""

    question_id: str
    question_text: str
    question_type: QuestionType

    # Retrieval results
    retrieved_ids: list[str] = field(default_factory=list)
    retrieved_scores: list[float] = field(default_factory=list)
    retrieval_time_ms: float = 0.0

    # Answer results
    answer: str = ""
    generation_time_ms: float = 0.0
    grounding_score: float = 0.0
    is_grounded: bool = False

    # Expected vs actual
    expected_chunks: list[str] = field(default_factory=list)
    expected_codes: list[str] = field(default_factory=list)
    is_trap: bool = False
    trap_handled_correctly: bool = False

    # Errors
    error: Optional[str] = None


@dataclass
class EvaluationResult:
    """Complete evaluation result."""

    # Metadata
    timestamp: str
    num_questions: int
    duration_seconds: float

    # Metrics
    retrieval_metrics: RetrievalMetrics
    answer_metrics: AnswerMetrics

    # Detailed results
    question_results: list[QuestionResult] = field(default_factory=list)

    # Configuration
    config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "metadata": {
                "timestamp": self.timestamp,
                "num_questions": self.num_questions,
                "duration_seconds": round(self.duration_seconds, 2),
            },
            "metrics": {
                "retrieval": self.retrieval_metrics.to_dict(),
                "answer": self.answer_metrics.to_dict(),
            },
            "question_results": [
                {
                    "question_id": r.question_id,
                    "question_text": r.question_text,
                    "question_type": r.question_type.value,
                    "retrieval_time_ms": round(r.retrieval_time_ms, 2),
                    "generation_time_ms": round(r.generation_time_ms, 2),
                    "grounding_score": round(r.grounding_score, 3),
                    "is_grounded": r.is_grounded,
                    "is_trap": r.is_trap,
                    "trap_handled_correctly": r.trap_handled_correctly,
                    "error": r.error,
                }
                for r in self.question_results
            ],
            "config": self.config,
        }

    def save(self, path: str | Path):
        """Save results to JSON file."""
        save_json(path, self.to_dict())
        logger.info(f"Saved evaluation results to {path}")

    def summary(self) -> str:
        """Generate summary report."""
        metrics = EvaluationMetrics(
            retrieval=self.retrieval_metrics,
            answer=self.answer_metrics,
        )

        lines = [
            "=" * 50,
            "EVALUATION REPORT",
            "=" * 50,
            f"Timestamp: {self.timestamp}",
            f"Questions: {self.num_questions}",
            f"Duration: {self.duration_seconds:.1f}s",
            "",
            metrics.summary(),
            "",
            "=" * 50,
        ]

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Runner
# ─────────────────────────────────────────────────────────────────────────────


class EvaluationRunner:
    """
    Runs evaluation pipeline on a question bank.

    Pipeline:
    1. For each question, run retrieval
    2. Generate answer from retrieved chunks
    3. Check grounding and citations
    4. Compute metrics
    5. Generate report

    Example:
        >>> runner = EvaluationRunner()
        >>> questions = QuestionBank.from_file("questions.json")
        >>> result = runner.evaluate(questions)
        >>> print(result.summary())
    """

    def __init__(
        self,
        retriever=None,
        generator=None,
        top_k: int = 10,
        skip_generation: bool = False,
    ):
        """
        Initialize evaluation runner.

        Args:
            retriever: Retriever instance (lazy-loaded if None)
            generator: Generator instance (lazy-loaded if None)
            top_k: Number of chunks to retrieve
            skip_generation: If True, only evaluate retrieval
        """
        self._retriever = retriever
        self._generator = generator
        self.top_k = top_k
        self.skip_generation = skip_generation

    @property
    def retriever(self):
        """Lazy-load retriever."""
        if self._retriever is None:
            from iue_coursecompass.rag.retriever import get_retriever
            self._retriever = get_retriever()
        return self._retriever

    @property
    def generator(self):
        """Lazy-load generator."""
        if self._generator is None and not self.skip_generation:
            from iue_coursecompass.rag.generator import get_generator
            self._generator = get_generator()
        return self._generator

    def evaluate(
        self,
        questions: QuestionBank | list[Question],
        progress_callback=None,
    ) -> EvaluationResult:
        """
        Run evaluation on questions.

        Args:
            questions: QuestionBank or list of Questions
            progress_callback: Optional callback(current, total, question_id)

        Returns:
            EvaluationResult
        """
        if isinstance(questions, QuestionBank):
            question_list = questions.questions
        else:
            question_list = questions

        logger.info(f"Starting evaluation on {len(question_list)} questions")
        start_time = time.time()

        question_results = []
        retrieval_results = []
        answer_results = []

        for i, question in enumerate(question_list):
            if progress_callback:
                progress_callback(i + 1, len(question_list), question.id)

            try:
                result = self._evaluate_question(question)
                question_results.append(result)

                # Build retrieval result for metrics
                retrieval_results.append(
                    RetrievalResult(
                        query_id=question.id,
                        retrieved_ids=result.retrieved_ids,
                        relevant_ids=set(question.expected_chunks),
                        scores=result.retrieved_scores,
                    )
                )

                # Build answer result for metrics (if generation was done)
                if result.answer:
                    # Convert retrieved IDs to RetrievalHit stubs for grounding
                    sources = [
                        RetrievalHit(
                            chunk_id=cid,
                            text="",
                            score=0.0,
                            course_code="",
                            course_title="",
                            department="",
                        )
                        for cid in result.retrieved_ids
                    ]

                    answer_results.append(
                        AnswerResult(
                            query_id=question.id,
                            answer=result.answer,
                            sources=sources,
                            is_trap=question.is_trap,
                            expected_codes=question.expected_course_codes,
                        )
                    )

            except Exception as e:
                logger.error(f"Error evaluating question {question.id}: {e}")
                question_results.append(
                    QuestionResult(
                        question_id=question.id,
                        question_text=question.question,
                        question_type=question.question_type,
                        error=str(e),
                    )
                )

        # Aggregate metrics
        retrieval_metrics = aggregate_retrieval_metrics(retrieval_results)
        answer_metrics = aggregate_answer_metrics(answer_results)

        duration = time.time() - start_time

        result = EvaluationResult(
            timestamp=datetime.now().isoformat(),
            num_questions=len(question_list),
            duration_seconds=duration,
            retrieval_metrics=retrieval_metrics,
            answer_metrics=answer_metrics,
            question_results=question_results,
            config={
                "top_k": self.top_k,
                "skip_generation": self.skip_generation,
            },
        )

        logger.info(f"Evaluation complete in {duration:.1f}s")
        logger.info(f"MRR: {retrieval_metrics.mrr:.3f}, Grounding: {answer_metrics.grounding_rate:.3f}")

        return result

    def _evaluate_question(self, question: Question) -> QuestionResult:
        """Evaluate a single question."""
        result = QuestionResult(
            question_id=question.id,
            question_text=question.question,
            question_type=question.question_type,
            expected_chunks=question.expected_chunks,
            expected_codes=question.expected_course_codes,
            is_trap=question.is_trap,
        )

        # Run retrieval
        retrieval_start = time.time()
        # Use target_departments for comparison questions, otherwise single department
        if question.question_type == QuestionType.COMPARISON and question.target_departments:
            departments = question.target_departments
        elif question.target_department:
            departments = [question.target_department]
        else:
            departments = None
        hits = self.retriever.retrieve(
            query=question.question,
            top_k=self.top_k,
            departments=departments,
        )
        result.retrieval_time_ms = (time.time() - retrieval_start) * 1000

        result.retrieved_ids = [h.chunk_id for h in hits]
        result.retrieved_scores = [h.score for h in hits]

        # Generate answer (if not skipped)
        if not self.skip_generation and self.generator:
            generation_start = time.time()

            # Determine if low confidence (for trap awareness)
            low_confidence = (
                len(hits) == 0 or
                (hits and hits[0].score < 0.3)
            )

            response = self.generator.generate(
                query=question.question,
                hits=hits,
                low_confidence=low_confidence,
            )
            result.generation_time_ms = (time.time() - generation_start) * 1000
            result.answer = response.answer

            # Check grounding
            grounding_result = check_grounding(response.answer, hits)
            result.grounding_score = grounding_result.grounding_score
            result.is_grounded = grounding_result.is_grounded

            # Check trap handling
            if question.is_trap:
                result.trap_handled_correctly = self._check_trap_response(response.answer)

        return result

    def _check_trap_response(self, answer: str) -> bool:
        """Check if a trap question was handled correctly."""
        answer_lower = answer.lower()

        # Phrases that indicate correct trap handling
        not_found_phrases = [
            "not found",
            "no information",
            "could not find",
            "don't have",
            "doesn't exist",
            "does not exist",
            "not in the sources",
            "no data",
            "unable to find",
            "not mentioned",
            "cannot find",
        ]

        return any(phrase in answer_lower for phrase in not_found_phrases)

    def evaluate_retrieval_only(
        self,
        questions: QuestionBank | list[Question],
    ) -> RetrievalMetrics:
        """
        Evaluate retrieval only (no generation).

        Args:
            questions: Questions to evaluate

        Returns:
            RetrievalMetrics
        """
        self.skip_generation = True
        result = self.evaluate(questions)
        return result.retrieval_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────


def run_evaluation(
    questions: QuestionBank | list[Question] | str | Path,
    output_path: Optional[str | Path] = None,
    skip_generation: bool = False,
    top_k: int = 10,
) -> EvaluationResult:
    """
    Run evaluation on questions.

    Args:
        questions: QuestionBank, list of Questions, or path to question file
        output_path: Optional path to save results
        skip_generation: If True, only evaluate retrieval
        top_k: Number of chunks to retrieve

    Returns:
        EvaluationResult
    """
    # Load questions if path provided
    if isinstance(questions, (str, Path)):
        questions = QuestionBank.from_file(questions)

    runner = EvaluationRunner(
        top_k=top_k,
        skip_generation=skip_generation,
    )

    result = runner.evaluate(questions)

    if output_path:
        result.save(output_path)

    return result


def quick_eval(
    questions: list[str],
    expected_chunks: Optional[list[list[str]]] = None,
) -> RetrievalMetrics:
    """
    Quick retrieval evaluation with just query strings.

    Args:
        questions: List of query strings
        expected_chunks: Optional list of expected chunk IDs per query

    Returns:
        RetrievalMetrics
    """
    from iue_coursecompass.evaluation.questions import create_question

    # Create Question objects
    question_objs = []
    for i, q in enumerate(questions):
        expected = expected_chunks[i] if expected_chunks and i < len(expected_chunks) else []
        question_objs.append(
            create_question(
                question=q,
                question_id=f"quick_{i}",
                expected_course_codes=[],
            )
        )
        question_objs[-1].expected_chunks = expected

    runner = EvaluationRunner(skip_generation=True)
    result = runner.evaluate(question_objs)

    return result.retrieval_metrics


def evaluate_single(
    question: str,
    expected_answer: Optional[str] = None,
    expected_codes: Optional[list[str]] = None,
) -> QuestionResult:
    """
    Evaluate a single question.

    Args:
        question: Question text
        expected_answer: Expected answer (for reference)
        expected_codes: Expected course codes

    Returns:
        QuestionResult
    """
    from iue_coursecompass.evaluation.questions import create_question

    q = create_question(
        question=question,
        expected_answer=expected_answer,
        expected_course_codes=expected_codes or [],
    )

    runner = EvaluationRunner()
    result = runner._evaluate_question(q)

    return result
