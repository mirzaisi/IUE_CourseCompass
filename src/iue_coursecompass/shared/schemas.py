"""
Schemas Module - Pydantic data models for the application.
==========================================================

Defines all data contracts used across the application:
- Course and chunk data models
- Retrieval and response models
- Embedding manifest for versioning
- Evaluation models
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, computed_field


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────


class CourseType(str, Enum):
    """Course type classification."""

    MANDATORY = "mandatory"
    ELECTIVE = "elective"
    TECHNICAL_ELECTIVE = "technical_elective"
    NON_TECHNICAL_ELECTIVE = "non_technical_elective"
    UNKNOWN = "unknown"


class Department(str, Enum):
    """Supported departments."""

    SE = "se"
    CE = "ce"
    EEE = "eee"
    IE = "ie"


class EmbeddingProvider(str, Enum):
    """Embedding provider options."""

    SBERT = "sbert"
    GEMINI = "gemini"


class QueryMode(str, Enum):
    """Query mode for RAG."""

    ANSWER = "answer"
    COMPARE = "compare"


class QuestionCategory(str, Enum):
    """Evaluation question categories."""

    SINGLE_DEPARTMENT = "single_department"
    TOPIC_BASED = "topic_based"
    CROSS_DEPARTMENT = "cross_department"
    QUANTITATIVE = "quantitative"
    TRAP = "trap"


# ─────────────────────────────────────────────────────────────────────────────
# Course Data Models
# ─────────────────────────────────────────────────────────────────────────────


class CourseRecord(BaseModel):
    """
    Represents a single course with all scraped information.

    This is the canonical representation of course data after scraping
    and cleaning. Used for storage in courses.json.
    """

    # Identity
    course_code: str = Field(..., description="Course code (e.g., 'SE 301')")
    course_title: str = Field(..., description="Course title")
    department: str = Field(..., description="Department ID (se, ce, eee, ie)")
    year_range: str = Field(..., description="Academic year range (e.g., '2023-2024')")

    # Classification
    course_type: CourseType = Field(
        default=CourseType.UNKNOWN, description="Mandatory/elective classification"
    )
    semester: Optional[int] = Field(
        default=None, description="Semester number (1-8 for mandatory courses)"
    )

    # Credits
    ects: Optional[float] = Field(default=None, description="ECTS credit value")
    local_credits: Optional[float] = Field(default=None, description="Local credit value")

    # Content
    objectives: Optional[str] = Field(default=None, description="Course objectives")
    description: Optional[str] = Field(default=None, description="Detailed course description")
    prerequisites: Optional[str] = Field(default=None, description="Prerequisites text")
    weekly_topics: Optional[list[str]] = Field(
        default=None, description="List of weekly topics"
    )
    learning_outcomes: Optional[list[str]] = Field(
        default=None, description="Learning outcomes"
    )
    assessment_methods: Optional[str] = Field(default=None, description="Assessment methods")

    # Metadata
    source_url: str = Field(..., description="Source URL of the course page")
    scraped_at: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp of scraping"
    )

    @computed_field
    @property
    def course_id(self) -> str:
        """Generate stable course ID."""
        # Normalize course code: remove spaces, lowercase
        code_normalized = self.course_code.replace(" ", "_").lower()
        return f"{self.department}_{code_normalized}_{self.year_range}"

    def get_full_text(self) -> str:
        """Get concatenated text content for embedding."""
        parts = [
            f"Course: {self.course_code} - {self.course_title}",
            f"Department: {self.department.upper()}",
        ]

        if self.objectives:
            parts.append(f"Objectives: {self.objectives}")

        if self.description:
            parts.append(f"Description: {self.description}")

        if self.prerequisites:
            parts.append(f"Prerequisites: {self.prerequisites}")

        if self.weekly_topics:
            topics_text = " | ".join(self.weekly_topics)
            parts.append(f"Weekly Topics: {topics_text}")

        if self.learning_outcomes:
            outcomes_text = " | ".join(self.learning_outcomes)
            parts.append(f"Learning Outcomes: {outcomes_text}")

        return "\n\n".join(parts)

    model_config = {"use_enum_values": True}


# ─────────────────────────────────────────────────────────────────────────────
# Chunk Data Models
# ─────────────────────────────────────────────────────────────────────────────


class ChunkRecord(BaseModel):
    """
    Represents a text chunk with metadata for vector storage.

    Chunks are created from course content and stored in the vector database.
    Each chunk carries rich metadata for filtering and citation.
    """

    # Identity
    chunk_id: str = Field(..., description="Unique chunk identifier")
    course_id: str = Field(..., description="Parent course ID")

    # Content
    text: str = Field(..., description="Chunk text content")
    section_name: str = Field(..., description="Source section (objectives, description, etc.)")
    chunk_index: int = Field(..., description="Chunk index within section")

    # Metadata (denormalized from course for fast filtering)
    course_code: str = Field(..., description="Course code")
    course_title: str = Field(..., description="Course title")
    department: str = Field(..., description="Department ID")
    year_range: str = Field(..., description="Academic year range")
    course_type: str = Field(default="unknown", description="Course type")
    semester: Optional[int] = Field(default=None, description="Semester number")
    ects: Optional[float] = Field(default=None, description="ECTS credits")
    source_url: str = Field(..., description="Source URL")

    # Hash for deduplication
    text_hash: str = Field(..., description="SHA256 hash of text content")

    @classmethod
    def create(
        cls,
        course: CourseRecord,
        text: str,
        section_name: str,
        chunk_index: int,
        text_hash: str,
    ) -> "ChunkRecord":
        """Create a chunk from course and content."""
        chunk_id = f"{course.course_id}_{section_name}_{chunk_index}_{text_hash[:8]}"

        return cls(
            chunk_id=chunk_id,
            course_id=course.course_id,
            text=text,
            section_name=section_name,
            chunk_index=chunk_index,
            course_code=course.course_code,
            course_title=course.course_title,
            department=course.department,
            year_range=course.year_range,
            course_type=course.course_type.value if isinstance(course.course_type, CourseType) else course.course_type,
            semester=course.semester,
            ects=course.ects,
            source_url=course.source_url,
            text_hash=text_hash,
        )

    def to_metadata_dict(self) -> dict[str, Any]:
        """Convert to metadata dict for vector store."""
        return {
            "chunk_id": self.chunk_id,
            "course_id": self.course_id,
            "course_code": self.course_code,
            "course_title": self.course_title,
            "department": self.department,
            "year_range": self.year_range,
            "course_type": self.course_type,
            "semester": self.semester if self.semester else -1,  # ChromaDB doesn't like None
            "ects": self.ects if self.ects else 0.0,
            "section_name": self.section_name,
            "source_url": self.source_url,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval Models
# ─────────────────────────────────────────────────────────────────────────────


class RetrievalHit(BaseModel):
    """
    A retrieved chunk with similarity score.

    Returned from vector store queries.
    """

    chunk_id: str = Field(..., description="Chunk identifier")
    text: str = Field(..., description="Chunk text content")
    score: float = Field(..., description="Similarity score (0-1)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")

    @property
    def course_code(self) -> str:
        """Get course code from metadata."""
        return self.metadata.get("course_code", "")

    @property
    def course_title(self) -> str:
        """Get course title from metadata."""
        return self.metadata.get("course_title", "")

    @property
    def department(self) -> str:
        """Get department from metadata."""
        return self.metadata.get("department", "")

    @property
    def source_url(self) -> str:
        """Get source URL from metadata."""
        return self.metadata.get("source_url", "")


# ─────────────────────────────────────────────────────────────────────────────
# RAG Response Models
# ─────────────────────────────────────────────────────────────────────────────


class Citation(BaseModel):
    """A citation to a source chunk."""

    chunk_id: str = Field(..., description="Referenced chunk ID")
    course_code: str = Field(..., description="Course code")
    course_title: str = Field(default="", description="Course title")
    source_url: str = Field(default="", description="Source URL")
    text_snippet: str = Field(default="", description="Relevant text snippet")


class DepartmentBreakdown(BaseModel):
    """Department-specific information for comparisons."""

    department: str = Field(..., description="Department ID")
    department_name: str = Field(..., description="Department full name")
    summary: str = Field(..., description="Summary for this department")
    courses_mentioned: list[str] = Field(
        default_factory=list, description="Course codes mentioned"
    )
    citations: list[Citation] = Field(default_factory=list, description="Citations used")


class AnswerResponse(BaseModel):
    """
    Complete RAG response with answer and citations.

    This is the main response model returned to users.
    """

    # Core response
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer text")
    mode: QueryMode = Field(default=QueryMode.ANSWER, description="Query mode used")

    # Citations
    citations: list[Citation] = Field(default_factory=list, description="Source citations")

    # For comparison mode
    department_breakdown: Optional[list[DepartmentBreakdown]] = Field(
        default=None, description="Per-department breakdown for comparisons"
    )

    # Metadata
    retrieval_count: int = Field(default=0, description="Number of chunks retrieved")
    max_similarity: float = Field(default=0.0, description="Highest similarity score")
    min_similarity: float = Field(default=0.0, description="Lowest similarity score")
    is_grounded: bool = Field(default=True, description="Whether answer is fully grounded")
    confidence: float = Field(default=0.0, description="Overall confidence score")

    # For quantitative queries
    is_quantitative: bool = Field(default=False, description="Whether this was a counting query")
    quantitative_result: Optional[dict[str, Any]] = Field(
        default=None, description="Structured quantitative result"
    )

    model_config = {"use_enum_values": True}


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Manifest
# ─────────────────────────────────────────────────────────────────────────────


class EmbeddingManifest(BaseModel):
    """
    Manifest recording embedding index metadata.

    Used to track which model was used, dimensions, and version.
    """

    # Version info
    dataset_version: str = Field(..., description="SHA256 hash of chunks JSONL")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )

    # Embedding info
    embedding_provider: str = Field(..., description="Provider (sbert, gemini)")
    embedding_model: str = Field(..., description="Model name")
    embedding_dimensions: int = Field(..., description="Vector dimensions")

    # Statistics
    total_chunks: int = Field(default=0, description="Number of chunks indexed")
    total_courses: int = Field(default=0, description="Number of courses")
    departments: list[str] = Field(default_factory=list, description="Departments included")

    # Paths
    index_path: str = Field(default="", description="Path to index directory")
    chunks_file: str = Field(default="", description="Source chunks file")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Models
# ─────────────────────────────────────────────────────────────────────────────


class ExpectedResult(BaseModel):
    """Expected result for evaluation."""

    expected_answer: Optional[str] = Field(
        default=None, description="Expected answer pattern"
    )
    expected_courses: list[str] = Field(
        default_factory=list, description="Expected course codes in retrieval"
    )
    expected_negative: bool = Field(
        default=False, description="Whether expected answer is negative/not found"
    )


class QuestionItem(BaseModel):
    """A single evaluation question."""

    id: str = Field(..., description="Question ID")
    question: str = Field(..., description="Question text")
    category: QuestionCategory = Field(..., description="Question category")
    departments: list[str] = Field(
        default_factory=list, description="Target departments (empty = all)"
    )
    expected: ExpectedResult = Field(
        default_factory=ExpectedResult, description="Expected results"
    )

    model_config = {"use_enum_values": True}


class EvaluationResult(BaseModel):
    """Result of evaluating a single question."""

    question_id: str = Field(..., description="Question ID")
    question: str = Field(..., description="Question text")
    category: str = Field(..., description="Question category")

    # Retrieval results
    retrieved_chunk_ids: list[str] = Field(
        default_factory=list, description="Retrieved chunk IDs"
    )
    retrieved_course_codes: list[str] = Field(
        default_factory=list, description="Retrieved course codes"
    )

    # RAG results
    answer: str = Field(default="", description="Generated answer")
    citations: list[str] = Field(default_factory=list, description="Citation chunk IDs")

    # Metrics
    retrieval_accuracy: Optional[float] = Field(
        default=None, description="Retrieval accuracy (0-1)"
    )
    groundedness: Optional[float] = Field(
        default=None, description="Groundedness score (0-1)"
    )
    hallucination_detected: bool = Field(
        default=False, description="Whether hallucination was detected"
    )
    is_correct_negative: Optional[bool] = Field(
        default=None, description="For trap questions: correctly answered negative"
    )

    # Metadata
    max_similarity: float = Field(default=0.0, description="Max retrieval similarity")
    processing_time_ms: float = Field(default=0.0, description="Processing time in ms")


class EvaluationSummary(BaseModel):
    """Summary of evaluation batch run."""

    total_questions: int = Field(default=0, description="Total questions evaluated")
    by_category: dict[str, int] = Field(
        default_factory=dict, description="Questions per category"
    )

    # Aggregate metrics
    avg_retrieval_accuracy: float = Field(default=0.0, description="Average retrieval accuracy")
    avg_groundedness: float = Field(default=0.0, description="Average groundedness")
    hallucination_rate: float = Field(default=0.0, description="Overall hallucination rate")
    trap_question_accuracy: float = Field(
        default=0.0, description="Accuracy on trap questions"
    )

    # Timing
    total_time_seconds: float = Field(default=0.0, description="Total evaluation time")
    avg_time_per_question_ms: float = Field(default=0.0, description="Average time per question")
