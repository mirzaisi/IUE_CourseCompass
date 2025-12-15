"""
Evaluation Module - RAG quality evaluation harness.
====================================================

Provides tools for evaluating RAG system quality:
- Question bank loading and management
- Retrieval quality metrics (MRR, Recall@K)
- Answer quality metrics (grounding, accuracy)
- Evaluation runner and reporting

Components:
- questions: Question bank management
- metrics: Retrieval and answer quality metrics
- runner: Evaluation execution harness

Example:
    >>> from iue_coursecompass.evaluation import EvaluationRunner, load_questions
    >>> questions = load_questions("data/questions.json")
    >>> runner = EvaluationRunner()
    >>> results = runner.evaluate(questions)
    >>> print(f"MRR: {results.mrr:.3f}")
"""

from iue_coursecompass.evaluation.questions import (
    QuestionBank,
    load_questions,
    save_questions,
    create_question,
)
from iue_coursecompass.evaluation.metrics import (
    RetrievalMetrics,
    AnswerMetrics,
    calculate_mrr,
    calculate_recall_at_k,
    calculate_grounding_rate,
)
from iue_coursecompass.evaluation.runner import (
    EvaluationRunner,
    EvaluationResult,
    run_evaluation,
)

__all__ = [
    # Questions
    "QuestionBank",
    "load_questions",
    "save_questions",
    "create_question",
    # Metrics
    "RetrievalMetrics",
    "AnswerMetrics",
    "calculate_mrr",
    "calculate_recall_at_k",
    "calculate_grounding_rate",
    # Runner
    "EvaluationRunner",
    "EvaluationResult",
    "run_evaluation",
]
