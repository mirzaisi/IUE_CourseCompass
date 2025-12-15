"""
Pytest Configuration and Fixtures.
===================================

Shared fixtures for all test modules:
- Sample data fixtures
- Mock objects
- Temporary directories
- Configuration overrides
"""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest

# Set test environment before importing app modules
os.environ["COURSECOMPASS_ENV"] = "test"


# ─────────────────────────────────────────────────────────────────────────────
# Path Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def config_path(project_root: Path) -> Path:
    """Get the config file path."""
    return project_root / "config" / "settings.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Sample Data Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_course_data() -> dict:
    """Sample course data for testing."""
    return {
        "course_code": "SE 301",
        "course_title": "Software Engineering",
        "department": "se",
        "credits": 3,
        "ects": 6,
        "semester": 5,
        "year": 2024,
        "description": "Introduction to software engineering principles and practices.",
        "prerequisites": ["SE 201", "CE 100"],
        "learning_outcomes": [
            "Understand software development lifecycle",
            "Apply design patterns",
        ],
    }


@pytest.fixture
def sample_course_record(sample_course_data: dict):
    """Sample CourseRecord instance."""
    from iue_coursecompass.shared.schemas import CourseRecord
    return CourseRecord(**sample_course_data)


@pytest.fixture
def sample_chunk_data(sample_course_data: dict) -> dict:
    """Sample chunk data for testing."""
    return {
        "chunk_id": "se_301_chunk_001",
        "course_code": sample_course_data["course_code"],
        "course_title": sample_course_data["course_title"],
        "department": sample_course_data["department"],
        "text": "Software engineering is the application of engineering principles to software development.",
        "section_name": "description",
        "chunk_index": 0,
        "metadata": {
            "year": 2024,
            "semester": 5,
        },
    }


@pytest.fixture
def sample_chunk_record(sample_chunk_data: dict):
    """Sample ChunkRecord instance."""
    from iue_coursecompass.shared.schemas import ChunkRecord
    return ChunkRecord(**sample_chunk_data)


@pytest.fixture
def sample_html() -> str:
    """Sample HTML for parser testing."""
    return """
    <!DOCTYPE html>
    <html>
    <head><title>SE 301 - Software Engineering</title></head>
    <body>
        <h1>SE 301 - Software Engineering</h1>
        <div class="course-info">
            <p><strong>Credits:</strong> 3</p>
            <p><strong>ECTS:</strong> 6</p>
        </div>
        <div class="description">
            <h2>Course Description</h2>
            <p>Introduction to software engineering principles and practices.
            This course covers the software development lifecycle, requirements
            engineering, design patterns, and testing methodologies.</p>
        </div>
        <div class="prerequisites">
            <h2>Prerequisites</h2>
            <ul>
                <li>SE 201 - Introduction to Programming</li>
                <li>CE 100 - Computer Science Fundamentals</li>
            </ul>
        </div>
    </body>
    </html>
    """


@pytest.fixture
def sample_dirty_text() -> str:
    """Sample dirty text for cleaner testing."""
    return """
    Course   Description:

    This is    a   test     course.

    <p>With some HTML tags</p>

    And    multiple


    blank lines.

    """


@pytest.fixture
def sample_retrieval_hits():
    """Sample RetrievalHit instances for testing."""
    from iue_coursecompass.shared.schemas import RetrievalHit

    return [
        RetrievalHit(
            chunk_id="se_301_chunk_001",
            text="Software engineering principles and practices.",
            score=0.95,
            course_code="SE 301",
            course_title="Software Engineering",
            department="se",
            metadata={"section": "description"},
        ),
        RetrievalHit(
            chunk_id="se_302_chunk_001",
            text="Advanced software design patterns.",
            score=0.85,
            course_code="SE 302",
            course_title="Software Design",
            department="se",
            metadata={"section": "description"},
        ),
        RetrievalHit(
            chunk_id="ce_301_chunk_001",
            text="Computer architecture and organization.",
            score=0.75,
            course_code="CE 301",
            course_title="Computer Architecture",
            department="ce",
            metadata={"section": "description"},
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Question Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_questions():
    """Sample questions for evaluation testing."""
    from iue_coursecompass.evaluation.questions import (
        Question,
        QuestionType,
        QuestionDifficulty,
    )

    return [
        Question(
            id="q1",
            question="What are the prerequisites for SE 301?",
            question_type=QuestionType.FACTUAL,
            difficulty=QuestionDifficulty.EASY,
            expected_course_codes=["SE 201", "CE 100"],
            target_department="se",
        ),
        Question(
            id="q2",
            question="How many ECTS is SE 301?",
            question_type=QuestionType.COUNTING,
            difficulty=QuestionDifficulty.EASY,
            expected_answer="6",
            target_department="se",
        ),
        Question(
            id="q3",
            question="What is SE 999?",
            question_type=QuestionType.TRAP,
            difficulty=QuestionDifficulty.EASY,
            is_trap=True,
            trap_topic="SE 999 (non-existent)",
            target_department="se",
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Mock Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_embeddings():
    """Mock embedding vectors for testing."""
    import random
    random.seed(42)

    def generate_embedding(dim: int = 384) -> list[float]:
        return [random.random() for _ in range(dim)]

    return generate_embedding


@pytest.fixture
def mock_settings() -> dict:
    """Mock settings for testing."""
    return {
        "departments": {
            "se": {"name": "Software Engineering"},
            "ce": {"name": "Computer Engineering"},
        },
        "scraping": {
            "rate_limit": 0.1,
            "timeout": 5,
            "years": [2024],
        },
        "chunking": {
            "chunk_size": 500,
            "chunk_overlap": 50,
        },
        "embeddings": {
            "provider": "sbert",
            "model": "all-MiniLM-L6-v2",
            "dimension": 384,
        },
        "retrieval": {
            "top_k": 5,
            "score_threshold": 0.3,
        },
        "generation": {
            "model": "gemini-1.5-flash",
            "temperature": 0.3,
        },
        "paths": {
            "data_dir": "data",
            "cache_dir": "data/cache",
            "index_dir": "data/index",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pytest Configuration
# ─────────────────────────────────────────────────────────────────────────────


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_api: marks tests that require API keys"
    )


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset global singletons between tests."""
    # Reset any cached instances
    import iue_coursecompass.rag.retriever as retriever_module
    import iue_coursecompass.rag.generator as generator_module
    import iue_coursecompass.rag.grounding as grounding_module

    retriever_module._retriever = None
    generator_module._generator = None
    grounding_module._checker = None

    yield

    # Cleanup after test
    retriever_module._retriever = None
    generator_module._generator = None
    grounding_module._checker = None
