"""
Tests for Evaluation Module.
============================

Tests for:
- Questions: Question bank management
- Metrics: Retrieval and answer metrics
- Runner: Evaluation execution
"""

import pytest
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Question Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestQuestions:
    """Tests for question bank management."""

    def test_create_question(self):
        """Test question creation."""
        from iue_coursecompass.evaluation.questions import (
            create_question,
            QuestionType,
            QuestionDifficulty,
        )

        question = create_question(
            question="What is SE 301?",
            question_type=QuestionType.FACTUAL,
            difficulty=QuestionDifficulty.EASY,
            target_department="se",
        )

        assert question.id is not None
        assert question.question == "What is SE 301?"
        assert question.question_type == QuestionType.FACTUAL
        assert question.target_department == "se"

    def test_create_trap_question(self):
        """Test trap question creation."""
        from iue_coursecompass.evaluation.questions import create_question, QuestionType

        question = create_question(
            question="What is SE 999?",
            question_type=QuestionType.TRAP,
            is_trap=True,
            trap_topic="SE 999 (non-existent)",
        )

        assert question.is_trap is True
        assert question.trap_topic == "SE 999 (non-existent)"

    def test_question_bank_creation(self, sample_questions):
        """Test QuestionBank creation."""
        from iue_coursecompass.evaluation.questions import QuestionBank

        bank = QuestionBank(sample_questions)

        assert len(bank) == len(sample_questions)

    def test_question_bank_filter_by_type(self, sample_questions):
        """Test filtering questions by type."""
        from iue_coursecompass.evaluation.questions import (
            QuestionBank,
            QuestionType,
        )

        bank = QuestionBank(sample_questions)
        factual = bank.filter(question_type=QuestionType.FACTUAL)

        assert all(q.question_type == QuestionType.FACTUAL for q in factual)

    def test_question_bank_filter_by_department(self, sample_questions):
        """Test filtering questions by department."""
        from iue_coursecompass.evaluation.questions import QuestionBank

        bank = QuestionBank(sample_questions)
        se_questions = bank.filter(department="se")

        assert all(q.target_department == "se" for q in se_questions)

    def test_question_bank_get_trap_questions(self, sample_questions):
        """Test getting trap questions."""
        from iue_coursecompass.evaluation.questions import QuestionBank

        bank = QuestionBank(sample_questions)
        traps = bank.get_trap_questions()

        assert all(q.is_trap for q in traps)

    def test_question_bank_sample(self, sample_questions):
        """Test sampling questions."""
        from iue_coursecompass.evaluation.questions import QuestionBank

        bank = QuestionBank(sample_questions)
        sample = bank.sample(n=2, seed=42)

        assert len(sample) == 2

    def test_question_bank_stats(self, sample_questions):
        """Test question bank statistics."""
        from iue_coursecompass.evaluation.questions import QuestionBank

        bank = QuestionBank(sample_questions)
        stats = bank.stats()

        assert "total" in stats
        assert "by_type" in stats
        assert "by_difficulty" in stats
        assert stats["total"] == len(sample_questions)

    def test_question_bank_save_and_load(self, sample_questions, temp_dir: Path):
        """Test saving and loading question bank."""
        from iue_coursecompass.evaluation.questions import (
            QuestionBank,
            load_questions,
            save_questions,
        )

        bank = QuestionBank(sample_questions)

        # Save
        file_path = temp_dir / "questions.json"
        save_questions(bank, file_path)

        # Load
        loaded = load_questions(file_path)

        assert len(loaded) == len(bank)
        assert loaded.questions[0].id == bank.questions[0].id

    def test_create_sample_questions(self):
        """Test sample question creation."""
        from iue_coursecompass.evaluation.questions import create_sample_questions

        bank = create_sample_questions()

        assert len(bank) > 0
        # Should have variety of types
        types = set(q.question_type for q in bank)
        assert len(types) >= 2


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_calculate_mrr_first_position(self):
        """Test MRR when relevant doc is first."""
        from iue_coursecompass.evaluation.metrics import calculate_mrr

        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc1"}

        mrr = calculate_mrr(retrieved, relevant)

        assert mrr == 1.0

    def test_calculate_mrr_second_position(self):
        """Test MRR when relevant doc is second."""
        from iue_coursecompass.evaluation.metrics import calculate_mrr

        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc2"}

        mrr = calculate_mrr(retrieved, relevant)

        assert mrr == 0.5

    def test_calculate_mrr_not_found(self):
        """Test MRR when relevant doc is not retrieved."""
        from iue_coursecompass.evaluation.metrics import calculate_mrr

        retrieved = ["doc1", "doc2", "doc3"]
        relevant = {"doc4"}

        mrr = calculate_mrr(retrieved, relevant)

        assert mrr == 0.0

    def test_calculate_recall_at_k(self):
        """Test Recall@K calculation."""
        from iue_coursecompass.evaluation.metrics import calculate_recall_at_k

        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc3", "doc6"}

        recall_3 = calculate_recall_at_k(retrieved, relevant, k=3)
        recall_5 = calculate_recall_at_k(retrieved, relevant, k=5)

        # At k=3: found doc1, doc3 = 2/3
        assert recall_3 == pytest.approx(2/3, 0.01)
        # At k=5: same = 2/3
        assert recall_5 == pytest.approx(2/3, 0.01)

    def test_calculate_precision_at_k(self):
        """Test Precision@K calculation."""
        from iue_coursecompass.evaluation.metrics import calculate_precision_at_k

        retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = {"doc1", "doc3"}

        precision_3 = calculate_precision_at_k(retrieved, relevant, k=3)
        precision_5 = calculate_precision_at_k(retrieved, relevant, k=5)

        # At k=3: found doc1, doc3 = 2/3
        assert precision_3 == pytest.approx(2/3, 0.01)
        # At k=5: found doc1, doc3 = 2/5
        assert precision_5 == pytest.approx(2/5, 0.01)

    def test_calculate_hit_rate(self):
        """Test hit rate calculation."""
        from iue_coursecompass.evaluation.metrics import calculate_hit_rate

        # Hit
        assert calculate_hit_rate(["doc1", "doc2"], {"doc1"}) == 1.0
        # No hit
        assert calculate_hit_rate(["doc1", "doc2"], {"doc3"}) == 0.0

    def test_calculate_grounding_rate(self):
        """Test grounding rate calculation."""
        from iue_coursecompass.evaluation.metrics import calculate_grounding_rate
        from iue_coursecompass.rag.grounding import GroundingResult

        results = [
            GroundingResult(
                is_grounded=True, grounding_score=0.9,
                total_citations=2, valid_citations=2,
                invalid_citations=[], claims_checked=1, claims_verified=1,
                unverified_claims=[],
            ),
            GroundingResult(
                is_grounded=False, grounding_score=0.3,
                total_citations=1, valid_citations=0,
                invalid_citations=["bad"], claims_checked=1, claims_verified=0,
                unverified_claims=["claim"],
            ),
        ]

        rate = calculate_grounding_rate(results)

        assert rate == 0.5  # 1 grounded out of 2

    def test_check_trap_handling(self):
        """Test trap question handling check."""
        from iue_coursecompass.evaluation.metrics import check_trap_handling

        # Correct trap handling (says not found)
        assert check_trap_handling("I could not find information about SE 999.", is_trap=True) is True

        # Incorrect trap handling (gives answer)
        assert check_trap_handling("SE 999 is about quantum computing.", is_trap=True) is False

    def test_retrieval_metrics_aggregation(self):
        """Test aggregating retrieval metrics."""
        from iue_coursecompass.evaluation.metrics import (
            RetrievalResult,
            aggregate_retrieval_metrics,
        )

        results = [
            RetrievalResult(
                query_id="q1",
                retrieved_ids=["doc1", "doc2", "doc3"],
                relevant_ids={"doc1"},
                scores=[0.9, 0.8, 0.7],
            ),
            RetrievalResult(
                query_id="q2",
                retrieved_ids=["doc4", "doc5", "doc6"],
                relevant_ids={"doc5"},
                scores=[0.85, 0.75, 0.65],
            ),
        ]

        metrics = aggregate_retrieval_metrics(results)

        assert metrics.num_queries == 2
        assert 0 <= metrics.mrr <= 1
        assert 0 <= metrics.hit_rate <= 1

    def test_answer_metrics_aggregation(self, sample_retrieval_hits):
        """Test aggregating answer metrics."""
        from iue_coursecompass.evaluation.metrics import (
            AnswerResult,
            aggregate_answer_metrics,
        )

        results = [
            AnswerResult(
                query_id="q1",
                answer="SE 301 is about software engineering [se_301_chunk_001].",
                sources=sample_retrieval_hits,
                is_trap=False,
                expected_codes=["SE 301"],
            ),
            AnswerResult(
                query_id="q2",
                answer="I could not find information about SE 999.",
                sources=[],
                is_trap=True,
                expected_codes=[],
            ),
        ]

        metrics = aggregate_answer_metrics(results)

        assert metrics.num_answers == 2
        assert metrics.num_trap_questions == 1

    def test_evaluation_metrics_summary(self):
        """Test EvaluationMetrics summary generation."""
        from iue_coursecompass.evaluation.metrics import (
            RetrievalMetrics,
            AnswerMetrics,
            EvaluationMetrics,
        )

        retrieval = RetrievalMetrics(mrr=0.8, recall_at_5=0.9, hit_rate=0.95)
        answer = AnswerMetrics(grounding_rate=0.85, trap_accuracy=1.0)

        combined = EvaluationMetrics(retrieval=retrieval, answer=answer)
        summary = combined.summary()

        assert "MRR" in summary
        assert "0.8" in summary or "0.80" in summary
        assert "Grounding" in summary


# ─────────────────────────────────────────────────────────────────────────────
# Runner Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestRunner:
    """Tests for evaluation runner."""

    def test_evaluation_result_structure(self):
        """Test EvaluationResult structure."""
        from iue_coursecompass.evaluation.runner import EvaluationResult
        from iue_coursecompass.evaluation.metrics import (
            RetrievalMetrics,
            AnswerMetrics,
        )

        result = EvaluationResult(
            timestamp="2024-01-01T00:00:00",
            num_questions=10,
            duration_seconds=5.0,
            retrieval_metrics=RetrievalMetrics(),
            answer_metrics=AnswerMetrics(),
        )

        assert result.num_questions == 10
        assert result.duration_seconds == 5.0

    def test_evaluation_result_to_dict(self):
        """Test EvaluationResult serialization."""
        from iue_coursecompass.evaluation.runner import EvaluationResult
        from iue_coursecompass.evaluation.metrics import (
            RetrievalMetrics,
            AnswerMetrics,
        )

        result = EvaluationResult(
            timestamp="2024-01-01T00:00:00",
            num_questions=10,
            duration_seconds=5.0,
            retrieval_metrics=RetrievalMetrics(mrr=0.8),
            answer_metrics=AnswerMetrics(grounding_rate=0.9),
        )

        data = result.to_dict()

        assert "metadata" in data
        assert "metrics" in data
        assert data["metrics"]["retrieval"]["mrr"] == 0.8

    def test_evaluation_result_save(self, temp_dir: Path):
        """Test saving evaluation results."""
        from iue_coursecompass.evaluation.runner import EvaluationResult
        from iue_coursecompass.evaluation.metrics import (
            RetrievalMetrics,
            AnswerMetrics,
        )
        import json

        result = EvaluationResult(
            timestamp="2024-01-01T00:00:00",
            num_questions=10,
            duration_seconds=5.0,
            retrieval_metrics=RetrievalMetrics(),
            answer_metrics=AnswerMetrics(),
        )

        file_path = temp_dir / "results.json"
        result.save(file_path)

        assert file_path.exists()

        with open(file_path, encoding="utf-8") as f:
            loaded = json.load(f)
            assert loaded["metadata"]["num_questions"] == 10

    def test_question_result_structure(self):
        """Test QuestionResult structure."""
        from iue_coursecompass.evaluation.runner import QuestionResult
        from iue_coursecompass.evaluation.questions import QuestionType

        result = QuestionResult(
            question_id="q1",
            question_text="What is SE 301?",
            question_type=QuestionType.FACTUAL,
            retrieved_ids=["chunk1", "chunk2"],
            answer="SE 301 is about software engineering.",
            grounding_score=0.9,
        )

        assert result.question_id == "q1"
        assert len(result.retrieved_ids) == 2
        assert result.grounding_score == 0.9

    def test_runner_initialization(self):
        """Test EvaluationRunner initialization."""
        from iue_coursecompass.evaluation.runner import EvaluationRunner

        runner = EvaluationRunner(
            top_k=10,
            skip_generation=True,
        )

        assert runner.top_k == 10
        assert runner.skip_generation is True

    def test_runner_check_trap_response(self):
        """Test trap response checking."""
        from iue_coursecompass.evaluation.runner import EvaluationRunner

        runner = EvaluationRunner()

        # Should be correct - says not found
        assert runner._check_trap_response("I could not find any information about SE 999.") is True

        # Should be incorrect - gives answer
        assert runner._check_trap_response("SE 999 covers advanced topics.") is False


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEvaluationIntegration:
    """Integration tests for evaluation pipeline."""

    def test_question_to_metrics_flow(self, sample_questions):
        """Test flow from questions to metrics."""
        from iue_coursecompass.evaluation.questions import QuestionBank
        from iue_coursecompass.evaluation.metrics import (
            RetrievalResult,
            aggregate_retrieval_metrics,
        )

        bank = QuestionBank(sample_questions)

        # Simulate retrieval results
        retrieval_results = []
        for q in bank.questions:
            retrieval_results.append(
                RetrievalResult(
                    query_id=q.id,
                    retrieved_ids=["chunk1", "chunk2", "chunk3"],
                    relevant_ids=set(q.expected_chunks) if q.expected_chunks else set(),
                    scores=[0.9, 0.8, 0.7],
                )
            )

        # Aggregate
        metrics = aggregate_retrieval_metrics(retrieval_results)

        assert metrics.num_queries == len(sample_questions)

    def test_full_evaluation_pipeline_mocked(self, sample_questions, temp_dir: Path):
        """Test full evaluation pipeline with mocks."""
        from unittest.mock import MagicMock, patch
        from iue_coursecompass.evaluation.runner import EvaluationRunner
        from iue_coursecompass.shared.schemas import RetrievalHit

        # Mock retriever
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = [
            RetrievalHit(
                chunk_id="chunk1",
                text="SE 301 is about software engineering.",
                score=0.9,
                course_code="SE 301",
                course_title="Software Engineering",
                department="se",
            )
        ]

        runner = EvaluationRunner(skip_generation=True)
        runner._retriever = mock_retriever

        # Run evaluation
        result = runner.evaluate(sample_questions[:2])

        assert result.num_questions == 2
        assert result.retrieval_metrics is not None
        assert len(result.question_results) == 2
