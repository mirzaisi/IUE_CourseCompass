"""
Metrics Module - RAG evaluation metrics.
========================================

Provides metrics for evaluating RAG system quality:

Retrieval Metrics:
- MRR (Mean Reciprocal Rank)
- Recall@K
- Precision@K
- Hit Rate

Answer Metrics:
- Grounding Rate
- Citation Accuracy
- Trap Question Accuracy
- Answer Completeness
"""

from dataclasses import dataclass, field
from typing import Optional

from iue_coursecompass.shared.logging import get_logger
from iue_coursecompass.shared.schemas import RetrievalHit
from iue_coursecompass.rag.grounding import check_grounding, GroundingResult

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval Metrics
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RetrievalMetrics:
    """Container for retrieval quality metrics."""

    mrr: float = 0.0  # Mean Reciprocal Rank
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    hit_rate: float = 0.0  # % of queries with at least one relevant result
    avg_score: float = 0.0  # Average similarity score
    num_queries: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "mrr": round(self.mrr, 4),
            "recall@1": round(self.recall_at_1, 4),
            "recall@3": round(self.recall_at_3, 4),
            "recall@5": round(self.recall_at_5, 4),
            "recall@10": round(self.recall_at_10, 4),
            "precision@1": round(self.precision_at_1, 4),
            "precision@3": round(self.precision_at_3, 4),
            "precision@5": round(self.precision_at_5, 4),
            "hit_rate": round(self.hit_rate, 4),
            "avg_score": round(self.avg_score, 4),
            "num_queries": self.num_queries,
        }

    def __str__(self) -> str:
        return (
            f"RetrievalMetrics(MRR={self.mrr:.3f}, "
            f"R@5={self.recall_at_5:.3f}, "
            f"HitRate={self.hit_rate:.3f})"
        )


@dataclass
class AnswerMetrics:
    """Container for answer quality metrics."""

    grounding_rate: float = 0.0  # % of answers that are grounded
    avg_grounding_score: float = 0.0  # Average grounding score
    citation_accuracy: float = 0.0  # % of citations that are valid
    trap_accuracy: float = 0.0  # % of trap questions correctly rejected
    hallucination_rate: float = 0.0  # % of trap questions that hallucinated (1 - trap_accuracy)
    answer_completeness: float = 0.0  # % of expected info included
    num_answers: int = 0
    num_trap_questions: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "grounding_rate": round(self.grounding_rate, 4),
            "avg_grounding_score": round(self.avg_grounding_score, 4),
            "citation_accuracy": round(self.citation_accuracy, 4),
            "trap_accuracy": round(self.trap_accuracy, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "answer_completeness": round(self.answer_completeness, 4),
            "num_answers": self.num_answers,
            "num_trap_questions": self.num_trap_questions,
        }

    def __str__(self) -> str:
        return (
            f"AnswerMetrics(Grounding={self.grounding_rate:.3f}, "
            f"TrapAcc={self.trap_accuracy:.3f})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval Metric Calculations
# ─────────────────────────────────────────────────────────────────────────────


def calculate_mrr(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """
    Calculate Mean Reciprocal Rank for a single query.

    MRR = 1 / rank of first relevant result

    Args:
        retrieved_ids: List of retrieved chunk IDs in order
        relevant_ids: Set of relevant/expected chunk IDs

    Returns:
        Reciprocal rank (0 if no relevant results)
    """
    for i, chunk_id in enumerate(retrieved_ids):
        if chunk_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def calculate_recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Calculate Recall@K for a single query.

    Recall@K = |retrieved@K ∩ relevant| / |relevant|

    Args:
        retrieved_ids: List of retrieved chunk IDs in order
        relevant_ids: Set of relevant/expected chunk IDs
        k: Number of top results to consider

    Returns:
        Recall at K (0 if no relevant IDs)
    """
    if not relevant_ids:
        return 0.0

    top_k = set(retrieved_ids[:k])
    hits = len(top_k & relevant_ids)
    return hits / len(relevant_ids)


def calculate_precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    Calculate Precision@K for a single query.

    Precision@K = |retrieved@K ∩ relevant| / K

    Args:
        retrieved_ids: List of retrieved chunk IDs in order
        relevant_ids: Set of relevant/expected chunk IDs
        k: Number of top results to consider

    Returns:
        Precision at K
    """
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0

    hits = len(set(top_k) & relevant_ids)
    return hits / len(top_k)


def calculate_hit_rate(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """
    Calculate hit rate (whether any relevant result was retrieved).

    Args:
        retrieved_ids: List of retrieved chunk IDs
        relevant_ids: Set of relevant/expected chunk IDs

    Returns:
        1.0 if any relevant result found, 0.0 otherwise
    """
    return 1.0 if set(retrieved_ids) & relevant_ids else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval Metrics Aggregation
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RetrievalResult:
    """Single retrieval result for evaluation."""

    query_id: str
    retrieved_ids: list[str]
    relevant_ids: set[str]
    scores: list[float] = field(default_factory=list)


def aggregate_retrieval_metrics(
    results: list[RetrievalResult],
) -> RetrievalMetrics:
    """
    Aggregate retrieval metrics across multiple queries.

    Args:
        results: List of retrieval results

    Returns:
        Aggregated RetrievalMetrics
    """
    if not results:
        return RetrievalMetrics()

    mrr_scores = []
    recall_1_scores = []
    recall_3_scores = []
    recall_5_scores = []
    recall_10_scores = []
    precision_1_scores = []
    precision_3_scores = []
    precision_5_scores = []
    hit_rates = []
    all_scores = []

    for result in results:
        retrieved = result.retrieved_ids
        relevant = result.relevant_ids

        mrr_scores.append(calculate_mrr(retrieved, relevant))
        recall_1_scores.append(calculate_recall_at_k(retrieved, relevant, 1))
        recall_3_scores.append(calculate_recall_at_k(retrieved, relevant, 3))
        recall_5_scores.append(calculate_recall_at_k(retrieved, relevant, 5))
        recall_10_scores.append(calculate_recall_at_k(retrieved, relevant, 10))
        precision_1_scores.append(calculate_precision_at_k(retrieved, relevant, 1))
        precision_3_scores.append(calculate_precision_at_k(retrieved, relevant, 3))
        precision_5_scores.append(calculate_precision_at_k(retrieved, relevant, 5))
        hit_rates.append(calculate_hit_rate(retrieved, relevant))

        if result.scores:
            all_scores.extend(result.scores)

    n = len(results)

    return RetrievalMetrics(
        mrr=sum(mrr_scores) / n,
        recall_at_1=sum(recall_1_scores) / n,
        recall_at_3=sum(recall_3_scores) / n,
        recall_at_5=sum(recall_5_scores) / n,
        recall_at_10=sum(recall_10_scores) / n,
        precision_at_1=sum(precision_1_scores) / n,
        precision_at_3=sum(precision_3_scores) / n,
        precision_at_5=sum(precision_5_scores) / n,
        hit_rate=sum(hit_rates) / n,
        avg_score=sum(all_scores) / len(all_scores) if all_scores else 0.0,
        num_queries=n,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Answer Metric Calculations
# ─────────────────────────────────────────────────────────────────────────────


def calculate_grounding_rate(
    grounding_results: list[GroundingResult],
) -> float:
    """
    Calculate the rate of grounded answers.

    Args:
        grounding_results: List of grounding check results

    Returns:
        Fraction of answers that are grounded
    """
    if not grounding_results:
        return 0.0

    grounded = sum(1 for r in grounding_results if r.is_grounded)
    return grounded / len(grounding_results)


def calculate_avg_grounding_score(
    grounding_results: list[GroundingResult],
) -> float:
    """
    Calculate average grounding score.

    Args:
        grounding_results: List of grounding check results

    Returns:
        Average grounding score
    """
    if not grounding_results:
        return 0.0

    return sum(r.grounding_score for r in grounding_results) / len(grounding_results)


def calculate_citation_accuracy(
    grounding_results: list[GroundingResult],
) -> float:
    """
    Calculate citation accuracy across answers.

    Args:
        grounding_results: List of grounding check results

    Returns:
        Fraction of valid citations
    """
    total_citations = 0
    valid_citations = 0

    for result in grounding_results:
        total_citations += result.total_citations
        valid_citations += result.valid_citations

    if total_citations == 0:
        return 0.0

    return valid_citations / total_citations


def calculate_trap_accuracy(
    trap_results: list[tuple[bool, bool]],  # (is_trap, correctly_handled)
) -> float:
    """
    Calculate trap question handling accuracy.

    Args:
        trap_results: List of (is_trap, correctly_handled) tuples

    Returns:
        Fraction of trap questions correctly handled
    """
    if not trap_results:
        return 0.0

    correct = sum(1 for _, handled in trap_results if handled)
    return correct / len(trap_results)


def check_trap_handling(
    answer: str,
    is_trap: bool,
) -> bool:
    """
    Check if a trap question was handled correctly.

    For trap questions: Answer should indicate "not found" or similar.
    For non-trap questions: Answer should provide information.

    Args:
        answer: Generated answer
        is_trap: Whether the question is a trap

    Returns:
        True if handled correctly
    """
    answer_lower = answer.lower()

    # Phrases indicating "not found"
    not_found_phrases = [
        "not found",
        "no information",
        "could not find",
        "don't have information",
        "doesn't exist",
        "does not exist",
        "not in the sources",
        "no data",
        "unable to find",
        "not mentioned",
        "cannot find",
        "no record",
    ]

    indicates_not_found = any(phrase in answer_lower for phrase in not_found_phrases)

    if is_trap:
        # Trap question should result in "not found" response
        return indicates_not_found
    else:
        # Non-trap question should NOT result in "not found" (usually)
        # But this is a weak check - the answer might legitimately not find something
        return True  # Give benefit of doubt for non-trap questions


def calculate_answer_completeness(
    answer: str,
    expected_codes: list[str],
) -> float:
    """
    Calculate answer completeness based on expected course codes.

    Args:
        answer: Generated answer
        expected_codes: Expected course codes that should appear

    Returns:
        Fraction of expected codes found in answer
    """
    if not expected_codes:
        return 1.0  # No expectations = complete

    answer_upper = answer.upper()
    found = sum(1 for code in expected_codes if code.upper() in answer_upper)

    return found / len(expected_codes)


# ─────────────────────────────────────────────────────────────────────────────
# Answer Metrics Aggregation
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class AnswerResult:
    """Single answer result for evaluation."""

    query_id: str
    answer: str
    sources: list[RetrievalHit]
    is_trap: bool = False
    expected_codes: list[str] = field(default_factory=list)
    grounding_result: Optional[GroundingResult] = None


def aggregate_answer_metrics(
    results: list[AnswerResult],
) -> AnswerMetrics:
    """
    Aggregate answer metrics across multiple answers.

    Args:
        results: List of answer results

    Returns:
        Aggregated AnswerMetrics
    """
    if not results:
        return AnswerMetrics()

    # Compute grounding for each if not already done
    grounding_results = []
    for result in results:
        if result.grounding_result:
            grounding_results.append(result.grounding_result)
        else:
            gr = check_grounding(result.answer, result.sources)
            grounding_results.append(gr)

    # Trap question handling
    trap_results = []
    num_trap = 0
    for result in results:
        if result.is_trap:
            num_trap += 1
            handled = check_trap_handling(result.answer, is_trap=True)
            trap_results.append((True, handled))

    # Completeness
    completeness_scores = []
    for result in results:
        if result.expected_codes:
            score = calculate_answer_completeness(result.answer, result.expected_codes)
            completeness_scores.append(score)

    return AnswerMetrics(
        grounding_rate=calculate_grounding_rate(grounding_results),
        avg_grounding_score=calculate_avg_grounding_score(grounding_results),
        citation_accuracy=calculate_citation_accuracy(grounding_results),
        trap_accuracy=calculate_trap_accuracy(trap_results) if trap_results else 1.0,
        hallucination_rate=(1.0 - calculate_trap_accuracy(trap_results)) if trap_results else 0.0,
        answer_completeness=(
            sum(completeness_scores) / len(completeness_scores)
            if completeness_scores else 1.0
        ),
        num_answers=len(results),
        num_trap_questions=num_trap,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Combined Metrics
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EvaluationMetrics:
    """Combined retrieval and answer metrics."""

    retrieval: RetrievalMetrics
    answer: AnswerMetrics

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "retrieval": self.retrieval.to_dict(),
            "answer": self.answer.to_dict(),
        }

    def summary(self) -> str:
        """Generate a summary string."""
        lines = [
            "=== Evaluation Metrics ===",
            "",
            "Retrieval:",
            f"  MRR: {self.retrieval.mrr:.3f}",
            f"  Recall@5: {self.retrieval.recall_at_5:.3f}",
            f"  Hit Rate: {self.retrieval.hit_rate:.3f}",
            f"  Queries: {self.retrieval.num_queries}",
            "",
            "Answer Quality:",
            f"  Grounding Rate: {self.answer.grounding_rate:.3f}",
            f"  Avg Grounding Score: {self.answer.avg_grounding_score:.3f}",
            f"  Citation Accuracy: {self.answer.citation_accuracy:.3f}",
            f"  Trap Accuracy: {self.answer.trap_accuracy:.3f}",
            f"  Hallucination Rate: {self.answer.hallucination_rate:.3f}",
            f"  Answers: {self.answer.num_answers}",
        ]
        return "\n".join(lines)
