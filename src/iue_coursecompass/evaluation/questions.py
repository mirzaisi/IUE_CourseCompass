"""
Questions Module - Question bank management for evaluation.
===========================================================

Manages test questions for RAG evaluation:
- Loading/saving question banks (JSON/JSONL/YAML)
- Question categorization (factual, counting, comparison, trap)
- Expected answer and source tracking
- Question generation helpers
"""

import json
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

from iue_coursecompass.shared.logging import get_logger
from iue_coursecompass.shared.utils import load_json, save_json, load_jsonl, save_jsonl

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# YAML Field Mapping
# ─────────────────────────────────────────────────────────────────────────────

# Maps YAML category/mode values to QuestionType enum values
YAML_CATEGORY_MAP: dict[str, str] = {
    "A": "factual",        # Single-Department
    "B": "factual",        # Topic-Based (still factual retrieval)
    "C": "comparison",     # Cross-Department Comparison
    "D": "counting",       # Quantitative / Counting
    "E": "trap",           # Hallucination / Trap
}

YAML_MODE_MAP: dict[str, str] = {
    "single": "factual",
    "topic": "factual",
    "compare": "comparison",
    "quant": "counting",
    "quant_compare": "counting",
    "trap": "trap",
}


def _map_yaml_question(item: dict[str, Any]) -> dict[str, Any]:
    """
    Map YAML question format to Question model format.
    
    Handles field name differences between questions_v1.yaml and Question model.
    """
    mapped = {}
    
    # Direct mappings
    mapped["id"] = item.get("id", "")
    mapped["question"] = item.get("query", item.get("question", ""))
    mapped["notes"] = item.get("notes")
    mapped["expected_course_codes"] = item.get("expected_course_codes", [])
    
    # Map category/mode to question_type
    category = item.get("category", "")
    mode = item.get("mode", "")
    
    if mode in YAML_MODE_MAP:
        mapped["question_type"] = YAML_MODE_MAP[mode]
    elif category in YAML_CATEGORY_MAP:
        mapped["question_type"] = YAML_CATEGORY_MAP[category]
    else:
        mapped["question_type"] = "factual"
    
    # Map department_scope to target_department
    dept_scope = item.get("department_scope", [])
    if dept_scope and dept_scope != ["ALL"]:
        # Take first department if multiple (for filtering purposes)
        mapped["target_department"] = dept_scope[0] if isinstance(dept_scope, list) else dept_scope
    
    # Map expected_negative or trap mode to is_trap
    mapped["is_trap"] = item.get("expected_negative", False) or mode == "trap" or category == "E"
    
    # Infer difficulty from category/mode
    if category in ["A", "E"] or mode in ["single", "trap"]:
        mapped["difficulty"] = "easy"
    elif category in ["C", "D"] or mode in ["compare", "quant_compare"]:
        mapped["difficulty"] = "hard"
    else:
        mapped["difficulty"] = "medium"
    
    # Store original YAML fields as tags for reference
    tags = []
    if category:
        tags.append(f"category:{category}")
    if mode:
        tags.append(f"mode:{mode}")
    answer_type = item.get("expected_answer_type")
    if answer_type:
        tags.append(f"answer_type:{answer_type}")
    mapped["tags"] = tags
    
    return mapped


# ─────────────────────────────────────────────────────────────────────────────
# Question Types
# ─────────────────────────────────────────────────────────────────────────────


class QuestionType(str, Enum):
    """Types of evaluation questions."""

    FACTUAL = "factual"  # Simple fact lookup
    COUNTING = "counting"  # How many courses...
    COMPARISON = "comparison"  # Compare X and Y
    TRAP = "trap"  # Question about non-existent topic
    MULTI_HOP = "multi_hop"  # Requires combining multiple sources
    TEMPORAL = "temporal"  # Year-specific questions


class QuestionDifficulty(str, Enum):
    """Question difficulty levels."""

    EASY = "easy"  # Direct answer in single chunk
    MEDIUM = "medium"  # May require some inference
    HARD = "hard"  # Requires multiple sources or reasoning


# ─────────────────────────────────────────────────────────────────────────────
# Question Model
# ─────────────────────────────────────────────────────────────────────────────


class Question(BaseModel):
    """A single evaluation question."""

    id: str = Field(..., description="Unique question identifier")
    question: str = Field(..., description="The question text")
    question_type: QuestionType = Field(
        default=QuestionType.FACTUAL,
        description="Type of question",
    )
    difficulty: QuestionDifficulty = Field(
        default=QuestionDifficulty.MEDIUM,
        description="Difficulty level",
    )

    # Expected answer information
    expected_answer: Optional[str] = Field(
        default=None,
        description="Expected answer text (for manual evaluation)",
    )
    expected_course_codes: list[str] = Field(
        default_factory=list,
        description="Course codes that should appear in answer",
    )
    expected_chunks: list[str] = Field(
        default_factory=list,
        description="Chunk IDs that should be retrieved",
    )

    # For trap questions
    is_trap: bool = Field(
        default=False,
        description="Whether this is a trap question (should return 'not found')",
    )
    trap_topic: Optional[str] = Field(
        default=None,
        description="The non-existent topic being asked about",
    )

    # Filtering/targeting
    target_department: Optional[str] = Field(
        default=None,
        description="Department this question targets",
    )
    target_year: Optional[int] = Field(
        default=None,
        description="Academic year this question targets",
    )

    # Metadata
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorization",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Notes about the question",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Question Bank
# ─────────────────────────────────────────────────────────────────────────────


class QuestionBank:
    """
    Manages a collection of evaluation questions.

    Supports:
    - Loading from JSON/JSONL/YAML files
    - Automatic field mapping for questions_v1.yaml format
    - Filtering by type, difficulty, department
    - Random sampling for evaluation
    - Adding/removing questions

    Example:
        >>> bank = QuestionBank.from_file("questions.json")
        >>> bank = QuestionBank.from_file("questions_v1.yaml")  # Also works!
        >>> factual = bank.filter(question_type=QuestionType.FACTUAL)
        >>> sample = bank.sample(n=10)
    """

    def __init__(self, questions: Optional[list[Question]] = None):
        """
        Initialize question bank.

        Args:
            questions: Initial list of questions
        """
        self.questions: list[Question] = questions or []
        self._index: dict[str, Question] = {}
        self._rebuild_index()

    def _rebuild_index(self):
        """Rebuild question ID index."""
        self._index = {q.id: q for q in self.questions}

    @classmethod
    def from_file(cls, path: str | Path) -> "QuestionBank":
        """
        Load question bank from file.

        Args:
            path: Path to JSON, JSONL, or YAML file

        Returns:
            QuestionBank instance
        """
        path = Path(path)
        is_yaml = path.suffix in (".yaml", ".yml")

        if path.suffix == ".jsonl":
            data = load_jsonl(path)
        elif is_yaml:
            with open(path, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)
            # YAML format has questions under "questions" key
            data = yaml_data.get("questions", [])
            # Map YAML fields to Question model fields
            data = [_map_yaml_question(item) for item in data]
        else:
            data = load_json(path)
            # Handle both list format and {"questions": [...]} format
            if isinstance(data, dict) and "questions" in data:
                data = data["questions"]

        questions = [Question(**item) for item in data]
        logger.info(f"Loaded {len(questions)} questions from {path}")

        return cls(questions=questions)

    def save(self, path: str | Path, format: str = "json"):
        """
        Save question bank to file.

        Args:
            path: Output path
            format: "json" or "jsonl"
        """
        path = Path(path)
        data = [q.model_dump() for q in self.questions]

        if format == "jsonl":
            save_jsonl(path, data)
        else:
            save_json(path, {"questions": data, "count": len(data)})

        logger.info(f"Saved {len(self.questions)} questions to {path}")

    def add(self, question: Question):
        """Add a question to the bank."""
        if question.id in self._index:
            logger.warning(f"Question {question.id} already exists, replacing")
        self.questions.append(question)
        self._index[question.id] = question

    def remove(self, question_id: str) -> bool:
        """Remove a question by ID."""
        if question_id in self._index:
            question = self._index.pop(question_id)
            self.questions.remove(question)
            return True
        return False

    def get(self, question_id: str) -> Optional[Question]:
        """Get a question by ID."""
        return self._index.get(question_id)

    def filter(
        self,
        question_type: Optional[QuestionType] = None,
        difficulty: Optional[QuestionDifficulty] = None,
        department: Optional[str] = None,
        is_trap: Optional[bool] = None,
        tags: Optional[list[str]] = None,
    ) -> list[Question]:
        """
        Filter questions by criteria.

        Args:
            question_type: Filter by type
            difficulty: Filter by difficulty
            department: Filter by target department
            is_trap: Filter by trap status
            tags: Filter by tags (any match)

        Returns:
            List of matching questions
        """
        result = self.questions

        if question_type is not None:
            result = [q for q in result if q.question_type == question_type]

        if difficulty is not None:
            result = [q for q in result if q.difficulty == difficulty]

        if department is not None:
            dept_lower = department.lower()
            result = [
                q for q in result
                if q.target_department and q.target_department.lower() == dept_lower
            ]

        if is_trap is not None:
            result = [q for q in result if q.is_trap == is_trap]

        if tags:
            tag_set = set(t.lower() for t in tags)
            result = [
                q for q in result
                if any(t.lower() in tag_set for t in q.tags)
            ]

        return result

    def sample(
        self,
        n: int,
        stratify_by: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> list[Question]:
        """
        Sample questions from the bank.

        Args:
            n: Number of questions to sample
            stratify_by: "type" or "difficulty" for stratified sampling
            seed: Random seed

        Returns:
            Sampled questions
        """
        import random

        if seed is not None:
            random.seed(seed)

        if stratify_by == "type":
            # Sample proportionally from each type
            by_type: dict[QuestionType, list[Question]] = {}
            for q in self.questions:
                if q.question_type not in by_type:
                    by_type[q.question_type] = []
                by_type[q.question_type].append(q)

            result = []
            per_type = max(1, n // len(by_type))
            for qtype, questions in by_type.items():
                sample_n = min(per_type, len(questions))
                result.extend(random.sample(questions, sample_n))

            # Fill remaining slots
            remaining = n - len(result)
            if remaining > 0:
                available = [q for q in self.questions if q not in result]
                if available:
                    result.extend(random.sample(available, min(remaining, len(available))))

            return result[:n]

        elif stratify_by == "difficulty":
            # Sample proportionally from each difficulty
            by_diff: dict[QuestionDifficulty, list[Question]] = {}
            for q in self.questions:
                if q.difficulty not in by_diff:
                    by_diff[q.difficulty] = []
                by_diff[q.difficulty].append(q)

            result = []
            per_diff = max(1, n // len(by_diff))
            for diff, questions in by_diff.items():
                sample_n = min(per_diff, len(questions))
                result.extend(random.sample(questions, sample_n))

            return result[:n]

        else:
            # Simple random sampling
            return random.sample(self.questions, min(n, len(self.questions)))

    def get_trap_questions(self) -> list[Question]:
        """Get all trap questions."""
        return [q for q in self.questions if q.is_trap]

    def get_by_department(self, department: str) -> list[Question]:
        """Get questions targeting a specific department."""
        return self.filter(department=department)

    def stats(self) -> dict:
        """Get statistics about the question bank."""
        by_type = {}
        by_difficulty = {}
        by_department = {}

        for q in self.questions:
            # By type
            qtype = q.question_type.value
            by_type[qtype] = by_type.get(qtype, 0) + 1

            # By difficulty
            diff = q.difficulty.value
            by_difficulty[diff] = by_difficulty.get(diff, 0) + 1

            # By department
            dept = q.target_department or "general"
            by_department[dept] = by_department.get(dept, 0) + 1

        return {
            "total": len(self.questions),
            "by_type": by_type,
            "by_difficulty": by_difficulty,
            "by_department": by_department,
            "trap_questions": len([q for q in self.questions if q.is_trap]),
        }

    def __len__(self) -> int:
        return len(self.questions)

    def __iter__(self):
        return iter(self.questions)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────


def load_questions(path: str | Path) -> QuestionBank:
    """
    Load questions from file.

    Args:
        path: Path to question file

    Returns:
        QuestionBank
    """
    return QuestionBank.from_file(path)


def save_questions(
    questions: QuestionBank | list[Question],
    path: str | Path,
    format: str = "json",
):
    """
    Save questions to file.

    Args:
        questions: QuestionBank or list of Questions
        path: Output path
        format: "json" or "jsonl"
    """
    if isinstance(questions, list):
        bank = QuestionBank(questions)
    else:
        bank = questions

    bank.save(path, format=format)


def create_question(
    question: str,
    question_type: QuestionType = QuestionType.FACTUAL,
    difficulty: QuestionDifficulty = QuestionDifficulty.MEDIUM,
    expected_answer: Optional[str] = None,
    expected_course_codes: Optional[list[str]] = None,
    is_trap: bool = False,
    trap_topic: Optional[str] = None,
    target_department: Optional[str] = None,
    tags: Optional[list[str]] = None,
    question_id: Optional[str] = None,
) -> Question:
    """
    Create a new question.

    Args:
        question: Question text
        question_type: Type of question
        difficulty: Difficulty level
        expected_answer: Expected answer text
        expected_course_codes: Expected course codes in answer
        is_trap: Whether this is a trap question
        trap_topic: Non-existent topic for trap questions
        target_department: Target department
        tags: Question tags
        question_id: Custom ID (auto-generated if not provided)

    Returns:
        Question instance
    """
    import hashlib

    # Generate ID if not provided
    if question_id is None:
        hash_input = question.encode()
        question_id = f"q_{hashlib.md5(hash_input).hexdigest()[:8]}"

    return Question(
        id=question_id,
        question=question,
        question_type=question_type,
        difficulty=difficulty,
        expected_answer=expected_answer,
        expected_course_codes=expected_course_codes or [],
        is_trap=is_trap,
        trap_topic=trap_topic,
        target_department=target_department,
        tags=tags or [],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sample Question Templates
# ─────────────────────────────────────────────────────────────────────────────


def create_sample_questions() -> QuestionBank:
    """
    Create a sample question bank for testing.

    Returns:
        QuestionBank with sample questions
    """
    questions = [
        # Factual questions
        create_question(
            question="What are the prerequisites for SE 301?",
            question_type=QuestionType.FACTUAL,
            difficulty=QuestionDifficulty.EASY,
            target_department="se",
            tags=["prerequisites", "se"],
        ),
        create_question(
            question="How many ECTS credits is CE 100?",
            question_type=QuestionType.FACTUAL,
            difficulty=QuestionDifficulty.EASY,
            target_department="ce",
            tags=["ects", "ce"],
        ),

        # Counting questions
        create_question(
            question="How many required courses are in the SE curriculum?",
            question_type=QuestionType.COUNTING,
            difficulty=QuestionDifficulty.MEDIUM,
            target_department="se",
            tags=["counting", "curriculum"],
        ),
        create_question(
            question="What is the total ECTS for the first year of EEE?",
            question_type=QuestionType.COUNTING,
            difficulty=QuestionDifficulty.MEDIUM,
            target_department="eee",
            tags=["ects", "counting"],
        ),

        # Comparison questions
        create_question(
            question="Compare the programming courses between SE and CE departments",
            question_type=QuestionType.COMPARISON,
            difficulty=QuestionDifficulty.HARD,
            tags=["comparison", "programming"],
        ),

        # Trap questions
        create_question(
            question="What is the syllabus for SE 999?",
            question_type=QuestionType.TRAP,
            difficulty=QuestionDifficulty.EASY,
            is_trap=True,
            trap_topic="SE 999 (non-existent course)",
            target_department="se",
            tags=["trap"],
        ),
        create_question(
            question="Does IUE offer a Quantum Computing course?",
            question_type=QuestionType.TRAP,
            difficulty=QuestionDifficulty.MEDIUM,
            is_trap=True,
            trap_topic="Quantum Computing course",
            tags=["trap"],
        ),

        # Multi-hop questions
        create_question(
            question="What courses should I take before I can enroll in SE 401?",
            question_type=QuestionType.MULTI_HOP,
            difficulty=QuestionDifficulty.HARD,
            target_department="se",
            tags=["prerequisites", "multi-hop"],
        ),
    ]

    return QuestionBank(questions)
