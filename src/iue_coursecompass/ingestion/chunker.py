"""
Chunker Module - Semantic text chunking for vector storage.
==========================================================

Splits course content into appropriately sized chunks for embedding:
- Respects semantic boundaries (sections, paragraphs, sentences)
- Configurable chunk size and overlap
- Preserves rich metadata for filtering and citation
- Generates stable, deterministic chunk IDs
"""

import re
from dataclasses import dataclass
from typing import Iterator, Optional

from iue_coursecompass.shared.config import get_settings
from iue_coursecompass.shared.logging import get_logger
from iue_coursecompass.shared.schemas import ChunkRecord, CourseRecord
from iue_coursecompass.shared.utils import compute_hash

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Chunking Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ChunkerConfig:
    """Configuration for text chunking."""

    # Chunk size limits (in characters)
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100

    # Semantic boundary preferences
    prefer_paragraph_breaks: bool = True
    prefer_sentence_breaks: bool = True

    # Section handling
    chunk_sections_separately: bool = True
    include_section_header: bool = True

    # Metadata
    include_course_context: bool = True  # Add course code/title to each chunk


# ─────────────────────────────────────────────────────────────────────────────
# Text Splitter
# ─────────────────────────────────────────────────────────────────────────────


class TextSplitter:
    """
    Splits text into chunks respecting semantic boundaries.

    Splitting priority:
    1. Paragraph breaks (double newline)
    2. Sentence boundaries (. ! ?)
    3. Clause boundaries (, ; :)
    4. Word boundaries (space)
    5. Character boundaries (last resort)
    """

    # Sentence ending pattern
    SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
    
    # Paragraph break pattern
    PARAGRAPH_BREAK = re.compile(r"\n\s*\n")

    def __init__(self, config: ChunkerConfig):
        """Initialize the splitter with configuration."""
        self.config = config

    def split(self, text: str) -> list[str]:
        """
        Split text into chunks.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        if not text or len(text) <= self.config.chunk_size:
            return [text] if text else []

        chunks = []
        
        # First, try splitting by paragraphs
        if self.config.prefer_paragraph_breaks:
            paragraphs = self.PARAGRAPH_BREAK.split(text)
            chunks = self._merge_small_chunks(paragraphs)
        else:
            chunks = [text]

        # Further split any chunks that are too large
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= self.config.chunk_size:
                final_chunks.append(chunk)
            else:
                # Split by sentences
                sub_chunks = self._split_large_chunk(chunk)
                final_chunks.extend(sub_chunks)

        # Add overlap between chunks
        if self.config.chunk_overlap > 0 and len(final_chunks) > 1:
            final_chunks = self._add_overlap(final_chunks)

        # Filter out too-small chunks
        final_chunks = [
            c for c in final_chunks
            if len(c.strip()) >= self.config.min_chunk_size
        ]

        return final_chunks

    def _merge_small_chunks(self, chunks: list[str]) -> list[str]:
        """Merge consecutive small chunks to approach target size."""
        if not chunks:
            return []

        merged = []
        current = ""

        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue

            # If adding this chunk would exceed limit, save current and start new
            if current and len(current) + len(chunk) + 2 > self.config.chunk_size:
                merged.append(current)
                current = chunk
            else:
                # Add to current chunk
                if current:
                    current += "\n\n" + chunk
                else:
                    current = chunk

        # Don't forget the last chunk
        if current:
            merged.append(current)

        return merged

    def _split_large_chunk(self, text: str) -> list[str]:
        """Split a large chunk by sentences or words."""
        if len(text) <= self.config.chunk_size:
            return [text]

        chunks = []
        
        # Try sentence splitting first
        if self.config.prefer_sentence_breaks:
            sentences = self.SENTENCE_END.split(text)
            chunks = self._merge_small_chunks(sentences)
            
            # Check if any chunk is still too large
            final = []
            for chunk in chunks:
                if len(chunk) <= self.config.chunk_size:
                    final.append(chunk)
                else:
                    # Fall back to word splitting
                    final.extend(self._split_by_words(chunk))
            return final
        else:
            return self._split_by_words(text)

    def _split_by_words(self, text: str) -> list[str]:
        """Split text by words, respecting chunk size."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            word_len = len(word) + 1  # +1 for space

            if current_length + word_len > self.config.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += word_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """Add overlap between consecutive chunks."""
        if len(chunks) <= 1:
            return chunks

        overlapped = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]

            # Get overlap from end of previous chunk
            overlap_text = self._get_overlap_text(prev_chunk)
            
            # Prepend overlap to current chunk
            if overlap_text and not current_chunk.startswith(overlap_text):
                overlapped.append(overlap_text + " " + current_chunk)
            else:
                overlapped.append(current_chunk)

        return overlapped

    def _get_overlap_text(self, text: str) -> str:
        """Get text for overlap from the end of a chunk."""
        if len(text) <= self.config.chunk_overlap:
            return text

        # Try to break at a sentence or word boundary
        overlap_region = text[-self.config.chunk_overlap:]
        
        # Find a good break point (sentence end or word boundary)
        sentence_match = re.search(r"[.!?]\s+", overlap_region)
        if sentence_match:
            return overlap_region[sentence_match.end():]
        
        # Fall back to word boundary
        space_idx = overlap_region.find(" ")
        if space_idx > 0:
            return overlap_region[space_idx + 1:]

        return overlap_region


# ─────────────────────────────────────────────────────────────────────────────
# Chunker Class
# ─────────────────────────────────────────────────────────────────────────────


class Chunker:
    """
    Chunks course content into appropriately sized pieces for embedding.

    Each chunk:
    - Has a stable, deterministic ID
    - Carries full metadata for filtering
    - Respects semantic boundaries where possible

    Example:
        >>> chunker = Chunker()
        >>> chunks = chunker.chunk_course(course_record)
        >>> for chunk in chunks:
        ...     print(chunk.chunk_id, len(chunk.text))
    """

    # Section names that map to CourseRecord fields
    SECTION_FIELDS = {
        "objectives": "objectives",
        "description": "description",
        "prerequisites": "prerequisites",
        "weekly_topics": "weekly_topics",
        "learning_outcomes": "learning_outcomes",
        "assessment": "assessment_methods",
    }

    def __init__(self, config: Optional[ChunkerConfig] = None):
        """
        Initialize the chunker.

        Args:
            config: Custom configuration (loads from settings if None)
        """
        if config is None:
            settings = get_settings()
            config = ChunkerConfig(
                chunk_size=settings.chunking.chunk_size,
                chunk_overlap=settings.chunking.chunk_overlap,
                min_chunk_size=settings.chunking.min_chunk_size,
            )

        self.config = config
        self.splitter = TextSplitter(config)

    def chunk_course(self, course: CourseRecord) -> list[ChunkRecord]:
        """
        Chunk a single course into ChunkRecords.

        Args:
            course: CourseRecord to chunk

        Returns:
            List of ChunkRecords
        """
        chunks: list[ChunkRecord] = []

        # Process each section
        for section_name, field_name in self.SECTION_FIELDS.items():
            section_chunks = self._chunk_section(course, section_name, field_name)
            chunks.extend(section_chunks)

        # If no content was chunked, create a minimal chunk from basic info
        if not chunks:
            fallback_chunk = self._create_fallback_chunk(course)
            if fallback_chunk:
                chunks.append(fallback_chunk)

        logger.debug(f"Created {len(chunks)} chunks for course {course.course_code}")
        return chunks

    def _chunk_section(
        self,
        course: CourseRecord,
        section_name: str,
        field_name: str,
    ) -> list[ChunkRecord]:
        """Chunk a specific section of the course."""
        # Get field value
        value = getattr(course, field_name, None)
        if not value:
            return []

        # Convert list to text if needed
        if isinstance(value, list):
            text = self._list_to_text(value, section_name)
        else:
            text = str(value)

        if not text or len(text.strip()) < self.config.min_chunk_size:
            return []

        # Add section header if configured
        if self.config.include_section_header:
            header = self._get_section_header(section_name)
            text = f"{header}\n\n{text}"

        # Add course context if configured
        if self.config.include_course_context:
            context = f"Course: {course.course_code} - {course.course_title}\n\n"
            text = context + text

        # Split text into chunks
        text_chunks = self.splitter.split(text)

        # Create ChunkRecords
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk_text = chunk_text.strip()
            if len(chunk_text) < self.config.min_chunk_size:
                continue

            text_hash = compute_hash(chunk_text)
            chunk = ChunkRecord.create(
                course=course,
                text=chunk_text,
                section_name=section_name,
                chunk_index=i,
                text_hash=text_hash,
            )
            chunks.append(chunk)

        return chunks

    def _list_to_text(self, items: list[str], section_name: str) -> str:
        """Convert a list of items to formatted text."""
        if not items:
            return ""

        # Format based on section type
        if section_name == "weekly_topics":
            # Number the weeks
            lines = [f"Week {i+1}: {item}" for i, item in enumerate(items)]
            return "\n".join(lines)
        elif section_name == "learning_outcomes":
            # Bullet points
            lines = [f"• {item}" for item in items]
            return "\n".join(lines)
        else:
            # Default: newline separated
            return "\n".join(items)

    def _get_section_header(self, section_name: str) -> str:
        """Get a human-readable header for a section."""
        headers = {
            "objectives": "Course Objectives",
            "description": "Course Description",
            "prerequisites": "Prerequisites",
            "weekly_topics": "Weekly Topics",
            "learning_outcomes": "Learning Outcomes",
            "assessment": "Assessment Methods",
        }
        return headers.get(section_name, section_name.replace("_", " ").title())

    def _create_fallback_chunk(self, course: CourseRecord) -> Optional[ChunkRecord]:
        """Create a minimal chunk when no content sections exist."""
        # Build text from basic course info
        parts = [
            f"Course Code: {course.course_code}",
            f"Course Title: {course.course_title}",
            f"Department: {course.department.upper()}",
        ]

        if course.semester:
            parts.append(f"Semester: {course.semester}")

        if course.ects:
            parts.append(f"ECTS Credits: {course.ects}")

        if course.course_type:
            parts.append(f"Type: {course.course_type}")

        text = "\n".join(parts)

        if len(text) < self.config.min_chunk_size:
            return None

        text_hash = compute_hash(text)
        return ChunkRecord.create(
            course=course,
            text=text,
            section_name="basic_info",
            chunk_index=0,
            text_hash=text_hash,
        )

    def chunk_courses(self, courses: list[CourseRecord]) -> Iterator[ChunkRecord]:
        """
        Chunk multiple courses.

        Args:
            courses: List of CourseRecords

        Yields:
            ChunkRecords for all courses
        """
        total_chunks = 0

        for course in courses:
            chunks = self.chunk_course(course)
            total_chunks += len(chunks)
            yield from chunks

        logger.info(f"Created {total_chunks} chunks from {len(courses)} courses")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────


def chunk_course(
    course: CourseRecord,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[ChunkRecord]:
    """
    Chunk a single course into ChunkRecords.

    Args:
        course: CourseRecord to chunk
        chunk_size: Optional custom chunk size
        chunk_overlap: Optional custom overlap

    Returns:
        List of ChunkRecords
    """
    config = None
    if chunk_size is not None or chunk_overlap is not None:
        settings = get_settings()
        config = ChunkerConfig(
            chunk_size=chunk_size or settings.chunking.chunk_size,
            chunk_overlap=chunk_overlap or settings.chunking.chunk_overlap,
        )

    chunker = Chunker(config=config)
    return chunker.chunk_course(course)


def chunk_courses(
    courses: list[CourseRecord],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[ChunkRecord]:
    """
    Chunk multiple courses into ChunkRecords.

    Args:
        courses: List of CourseRecords
        chunk_size: Optional custom chunk size
        chunk_overlap: Optional custom overlap

    Returns:
        List of all ChunkRecords
    """
    config = None
    if chunk_size is not None or chunk_overlap is not None:
        settings = get_settings()
        config = ChunkerConfig(
            chunk_size=chunk_size or settings.chunking.chunk_size,
            chunk_overlap=chunk_overlap or settings.chunking.chunk_overlap,
        )

    chunker = Chunker(config=config)
    return list(chunker.chunk_courses(courses))
