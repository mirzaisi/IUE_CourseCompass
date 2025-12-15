"""
Cleaner Module - Text cleaning and normalization.
=================================================

Cleans and normalizes text content extracted from HTML:
- Remove HTML artifacts and entities
- Fix encoding issues
- Normalize whitespace
- Remove irrelevant content (headers, footers, navigation)
- Standardize formatting
"""

import html
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

from iue_coursecompass.shared.logging import get_logger
from iue_coursecompass.shared.schemas import CourseRecord

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Cleaning Configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CleanerConfig:
    """Configuration for text cleaning operations."""

    # Whitespace handling
    normalize_whitespace: bool = True
    collapse_newlines: bool = True
    max_consecutive_newlines: int = 2
    strip_lines: bool = True

    # Unicode handling
    normalize_unicode: bool = True
    unicode_form: str = "NFKC"  # NFC, NFKC, NFD, NFKD
    remove_control_chars: bool = True

    # HTML cleanup
    decode_html_entities: bool = True
    remove_html_tags: bool = True
    remove_html_comments: bool = True

    # Content filtering
    remove_urls: bool = False  # Keep URLs for source tracking
    remove_emails: bool = True
    remove_phone_numbers: bool = True

    # Patterns to remove (regex)
    remove_patterns: list[str] = field(default_factory=lambda: [
        r"©\s*\d{4}.*?(?:All rights reserved|Tüm hakları saklıdır)\.?",
        r"Last updated:.*?(?:\n|$)",
        r"Page \d+ of \d+",
        r"Print this page",
        r"Back to top",
        r"Cookie policy.*?(?:\n|$)",
    ])

    # Minimum text length (shorter texts are considered empty)
    min_text_length: int = 10


# ─────────────────────────────────────────────────────────────────────────────
# Text Cleaner Class
# ─────────────────────────────────────────────────────────────────────────────


class TextCleaner:
    """
    Text cleaner with configurable cleaning operations.

    Applies a series of cleaning transformations to text content
    to prepare it for chunking and embedding.

    Example:
        >>> cleaner = TextCleaner()
        >>> clean = cleaner.clean("Hello &amp; World!   ")
        >>> print(clean)  # "Hello & World!"
    """

    def __init__(self, config: Optional[CleanerConfig] = None):
        """
        Initialize the cleaner.

        Args:
            config: Custom configuration (uses defaults if None)
        """
        self.config = config or CleanerConfig()
        
        # Pre-compile regex patterns for efficiency
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            for pattern in self.config.remove_patterns
        ]

        # HTML tag pattern
        self._html_tag_pattern = re.compile(r"<[^>]+>")
        
        # HTML comment pattern
        self._html_comment_pattern = re.compile(r"<!--.*?-->", re.DOTALL)

        # URL pattern
        self._url_pattern = re.compile(
            r"https?://[^\s<>\"{}|\\^`\[\]]+"
            r"|www\.[^\s<>\"{}|\\^`\[\]]+"
        )

        # Email pattern
        self._email_pattern = re.compile(
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        )

        # Phone pattern (various formats)
        self._phone_pattern = re.compile(
            r"(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
            r"|\+\d{10,15}"
        )

        # Whitespace patterns
        self._multiple_spaces = re.compile(r"[ \t]+")
        self._multiple_newlines = re.compile(r"\n{3,}")
        self._trailing_whitespace = re.compile(r"[ \t]+$", re.MULTILINE)
        self._leading_whitespace = re.compile(r"^[ \t]+", re.MULTILINE)

    def clean(self, text: Optional[str]) -> str:
        """
        Clean a text string.

        Args:
            text: Text to clean (can be None)

        Returns:
            Cleaned text (empty string if input is None or too short)
        """
        if not text:
            return ""

        result = text

        # HTML cleanup
        if self.config.remove_html_comments:
            result = self._remove_html_comments(result)

        if self.config.decode_html_entities:
            result = self._decode_html_entities(result)

        if self.config.remove_html_tags:
            result = self._remove_html_tags(result)

        # Unicode normalization
        if self.config.normalize_unicode:
            result = self._normalize_unicode(result)

        if self.config.remove_control_chars:
            result = self._remove_control_chars(result)

        # Content filtering
        if self.config.remove_urls:
            result = self._remove_urls(result)

        if self.config.remove_emails:
            result = self._remove_emails(result)

        if self.config.remove_phone_numbers:
            result = self._remove_phone_numbers(result)

        # Remove configured patterns
        result = self._remove_patterns(result)

        # Whitespace normalization
        if self.config.normalize_whitespace:
            result = self._normalize_whitespace(result)

        if self.config.collapse_newlines:
            result = self._collapse_newlines(result)

        if self.config.strip_lines:
            result = self._strip_lines(result)

        # Final strip
        result = result.strip()

        # Check minimum length
        if len(result) < self.config.min_text_length:
            return ""

        return result

    def clean_list(self, items: Optional[list[str]]) -> Optional[list[str]]:
        """
        Clean a list of text items.

        Args:
            items: List of strings to clean

        Returns:
            List of cleaned strings (empty items removed), or None if input is None
        """
        if items is None:
            return None

        cleaned = []
        for item in items:
            clean_item = self.clean(item)
            if clean_item:
                cleaned.append(clean_item)

        return cleaned if cleaned else None

    def _decode_html_entities(self, text: str) -> str:
        """Decode HTML entities like &amp; &lt; etc."""
        # First pass: named entities
        result = html.unescape(text)
        # Second pass for any remaining numeric entities
        result = re.sub(
            r"&#(\d+);",
            lambda m: chr(int(m.group(1))),
            result,
        )
        result = re.sub(
            r"&#x([0-9a-fA-F]+);",
            lambda m: chr(int(m.group(1), 16)),
            result,
        )
        return result

    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML tags, preserving text content."""
        return self._html_tag_pattern.sub(" ", text)

    def _remove_html_comments(self, text: str) -> str:
        """Remove HTML comments."""
        return self._html_comment_pattern.sub("", text)

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters."""
        return unicodedata.normalize(self.config.unicode_form, text)

    def _remove_control_chars(self, text: str) -> str:
        """Remove control characters except common whitespace."""
        # Keep: tab (9), newline (10), carriage return (13), space (32)
        return "".join(
            char for char in text
            if char in "\t\n\r " or (ord(char) >= 32 and unicodedata.category(char) != "Cc")
        )

    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        return self._url_pattern.sub("", text)

    def _remove_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        return self._email_pattern.sub("", text)

    def _remove_phone_numbers(self, text: str) -> str:
        """Remove phone numbers from text."""
        return self._phone_pattern.sub("", text)

    def _remove_patterns(self, text: str) -> str:
        """Remove configured regex patterns."""
        result = text
        for pattern in self._compiled_patterns:
            result = pattern.sub("", result)
        return result

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace (multiple spaces to single)."""
        # Replace tabs with spaces
        result = text.replace("\t", " ")
        # Collapse multiple spaces
        result = self._multiple_spaces.sub(" ", result)
        return result

    def _collapse_newlines(self, text: str) -> str:
        """Collapse multiple consecutive newlines."""
        max_newlines = "\n" * self.config.max_consecutive_newlines
        result = self._multiple_newlines.sub(max_newlines, text)
        return result

    def _strip_lines(self, text: str) -> str:
        """Strip leading/trailing whitespace from each line."""
        result = self._trailing_whitespace.sub("", text)
        result = self._leading_whitespace.sub("", result)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Course Record Cleaning
# ─────────────────────────────────────────────────────────────────────────────


def clean_course_record(
    course: CourseRecord,
    cleaner: Optional[TextCleaner] = None,
) -> CourseRecord:
    """
    Clean all text fields in a CourseRecord.

    Creates a new CourseRecord with cleaned text fields.
    Non-text fields are preserved as-is.

    Args:
        course: CourseRecord to clean
        cleaner: Optional TextCleaner instance (creates default if None)

    Returns:
        New CourseRecord with cleaned text fields
    """
    if cleaner is None:
        cleaner = TextCleaner()

    # Clean text fields
    cleaned_data = {
        "course_code": course.course_code.strip().upper(),
        "course_title": cleaner.clean(course.course_title) or course.course_title,
        "department": course.department.lower().strip(),
        "year_range": course.year_range.strip(),
        "course_type": course.course_type,
        "semester": course.semester,
        "ects": course.ects,
        "local_credits": course.local_credits,
        "objectives": cleaner.clean(course.objectives),
        "description": cleaner.clean(course.description),
        "prerequisites": cleaner.clean(course.prerequisites),
        "weekly_topics": cleaner.clean_list(course.weekly_topics),
        "learning_outcomes": cleaner.clean_list(course.learning_outcomes),
        "assessment_methods": cleaner.clean(course.assessment_methods),
        "source_url": course.source_url,
        "scraped_at": course.scraped_at,
    }

    # Handle empty strings -> None for optional fields
    for field_name in ["objectives", "description", "prerequisites", "assessment_methods"]:
        if cleaned_data[field_name] == "":
            cleaned_data[field_name] = None

    return CourseRecord(**cleaned_data)


def clean_courses(
    courses: list[CourseRecord],
    cleaner: Optional[TextCleaner] = None,
) -> list[CourseRecord]:
    """
    Clean a list of CourseRecords.

    Args:
        courses: List of CourseRecords to clean
        cleaner: Optional TextCleaner instance

    Returns:
        List of cleaned CourseRecords
    """
    if cleaner is None:
        cleaner = TextCleaner()

    cleaned = []
    for course in courses:
        try:
            cleaned_course = clean_course_record(course, cleaner)
            cleaned.append(cleaned_course)
        except Exception as e:
            logger.warning(f"Failed to clean course {course.course_code}: {e}")
            # Keep original if cleaning fails
            cleaned.append(course)

    logger.info(f"Cleaned {len(cleaned)} courses")
    return cleaned


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────


# Global cleaner instance (lazy initialization)
_default_cleaner: Optional[TextCleaner] = None


def get_cleaner() -> TextCleaner:
    """Get the default cleaner instance."""
    global _default_cleaner
    if _default_cleaner is None:
        _default_cleaner = TextCleaner()
    return _default_cleaner


def clean_text(text: Optional[str]) -> str:
    """
    Clean a text string using the default cleaner.

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    return get_cleaner().clean(text)


def clean_html(html_content: str) -> str:
    """
    Clean HTML content, extracting and cleaning text.

    This is a more aggressive cleaning for raw HTML that
    removes all tags and cleans the resulting text.

    Args:
        html_content: Raw HTML string

    Returns:
        Cleaned text content
    """
    cleaner = TextCleaner(
        CleanerConfig(
            remove_html_tags=True,
            remove_html_comments=True,
            decode_html_entities=True,
        )
    )
    return cleaner.clean(html_content)
