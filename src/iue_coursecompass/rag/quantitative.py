"""
Quantitative Module - Counting and calculation queries.
========================================================

Handles quantitative questions about courses and curricula:
- Course counting with filters
- ECTS credit summation
- Semester/year breakdowns
- Prerequisite chain analysis
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from iue_coursecompass.shared.logging import get_logger
from iue_coursecompass.shared.schemas import RetrievalHit

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Query Types
# ─────────────────────────────────────────────────────────────────────────────


class QuantitativeQueryType(str, Enum):
    """Types of quantitative queries."""

    COUNT = "count"  # How many courses...
    SUM_ECTS = "sum_ects"  # Total ECTS...
    LIST = "list"  # List all courses...
    COMPARE_COUNT = "compare_count"  # Which department has more...
    BREAKDOWN = "breakdown"  # Breakdown by semester/type
    NOT_QUANTITATIVE = "not_quantitative"


@dataclass
class CourseInfo:
    """Extracted course information for quantitative analysis."""

    code: str
    title: str = ""
    ects: Optional[float] = None
    credits: Optional[float] = None
    semester: Optional[int] = None
    year: Optional[int] = None
    department: str = ""
    course_type: str = ""  # required, elective, technical, etc.
    chunk_id: str = ""

    @property
    def code_prefix(self) -> str:
        """Get course code prefix (e.g., SE from SE 301)."""
        parts = self.code.split()
        return parts[0] if parts else ""

    @property
    def code_number(self) -> int:
        """Get course code number (e.g., 301 from SE 301)."""
        parts = self.code.split()
        if len(parts) >= 2:
            try:
                return int(parts[1])
            except ValueError:
                pass
        return 0


@dataclass
class QuantitativeResult:
    """Result of a quantitative query."""

    query_type: QuantitativeQueryType
    value: int | float | dict  # Count, sum, or breakdown dict
    courses: list[CourseInfo] = field(default_factory=list)
    formatted_data: str = ""  # For LLM context
    breakdown: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Patterns for Detection
# ─────────────────────────────────────────────────────────────────────────────


COUNT_PATTERNS = [
    r"how many (courses?|classes?|subjects?)",
    r"number of (courses?|classes?|subjects?)",
    r"count (the )?(courses?|classes?)",
    r"total (courses?|classes?)",
]

ECTS_PATTERNS = [
    r"(total|sum|how many) ects",
    r"ects (total|credits?|points?)",
    r"credit hours?",
]

LIST_PATTERNS = [
    r"list (all )?(the )?(courses?|classes?)",
    r"what (courses?|classes?) (are|does)",
    r"show (me )?(all )?(the )?(courses?|classes?)",
]

COMPARE_PATTERNS = [
    r"which (department|program) has (more|fewer|most|least)",
    r"compare .* (number|count|total)",
    r"difference (in|between) .* (courses?|ects)",
]

BREAKDOWN_PATTERNS = [
    r"breakdown (by|of)",
    r"(courses?|ects) (per|by|each) (semester|year)",
    r"distribution of",
]

COURSE_TYPE_KEYWORDS = {
    "required": ["required", "mandatory", "compulsory", "must take"],
    "elective": ["elective", "optional", "choice"],
    "technical": ["technical", "technical elective", "te"],
    "free": ["free", "free elective", "fe"],
    "service": ["service", "service course"],
}

SEMESTER_KEYWORDS = {
    1: ["first semester", "semester 1", "1st semester", "fall year 1"],
    2: ["second semester", "semester 2", "2nd semester", "spring year 1"],
    3: ["third semester", "semester 3", "3rd semester", "fall year 2"],
    4: ["fourth semester", "semester 4", "4th semester", "spring year 2"],
    5: ["fifth semester", "semester 5", "5th semester", "fall year 3"],
    6: ["sixth semester", "semester 6", "6th semester", "spring year 3"],
    7: ["seventh semester", "semester 7", "7th semester", "fall year 4"],
    8: ["eighth semester", "semester 8", "8th semester", "spring year 4"],
}


# ─────────────────────────────────────────────────────────────────────────────
# Query Detection
# ─────────────────────────────────────────────────────────────────────────────


def detect_quantitative_query(query: str) -> QuantitativeQueryType:
    """
    Detect if a query is quantitative and what type.

    Args:
        query: User query

    Returns:
        QuantitativeQueryType
    """
    query_lower = query.lower()

    # Check patterns in order of specificity
    for pattern in COMPARE_PATTERNS:
        if re.search(pattern, query_lower):
            return QuantitativeQueryType.COMPARE_COUNT

    for pattern in BREAKDOWN_PATTERNS:
        if re.search(pattern, query_lower):
            return QuantitativeQueryType.BREAKDOWN

    for pattern in ECTS_PATTERNS:
        if re.search(pattern, query_lower):
            return QuantitativeQueryType.SUM_ECTS

    for pattern in COUNT_PATTERNS:
        if re.search(pattern, query_lower):
            return QuantitativeQueryType.COUNT

    for pattern in LIST_PATTERNS:
        if re.search(pattern, query_lower):
            return QuantitativeQueryType.LIST

    return QuantitativeQueryType.NOT_QUANTITATIVE


def extract_query_filters(query: str) -> dict:
    """
    Extract filters from a quantitative query.

    Args:
        query: User query

    Returns:
        Dictionary of filters (department, semester, type, etc.)
    """
    filters = {}
    query_lower = query.lower()

    # Department filter
    dept_patterns = {
        "se": r"\b(software engineering|se)\b",
        "ce": r"\b(computer engineering|ce)\b",
        "eee": r"\b(electrical|electronics|eee|ee)\b",
        "ie": r"\b(industrial engineering|ie)\b",
    }

    for dept, pattern in dept_patterns.items():
        if re.search(pattern, query_lower):
            filters["department"] = dept
            break

    # Semester filter
    semester_match = re.search(r"semester\s*(\d+)", query_lower)
    if semester_match:
        filters["semester"] = int(semester_match.group(1))
    else:
        for sem, keywords in SEMESTER_KEYWORDS.items():
            for kw in keywords:
                if kw in query_lower:
                    filters["semester"] = sem
                    break

    # Year filter
    year_match = re.search(r"year\s*(\d+)", query_lower)
    if year_match:
        filters["year"] = int(year_match.group(1))

    # Course type filter
    for ctype, keywords in COURSE_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                filters["course_type"] = ctype
                break

    # Code prefix filter
    code_match = re.search(r"\b([A-Z]{2,4})\s+courses?\b", query, re.IGNORECASE)
    if code_match:
        filters["code_prefix"] = code_match.group(1).upper()

    return filters


# ─────────────────────────────────────────────────────────────────────────────
# Course Extraction
# ─────────────────────────────────────────────────────────────────────────────


# Course code pattern
COURSE_CODE_PATTERN = re.compile(r"\b([A-Z]{2,4})\s*(\d{3})\b")

# ECTS pattern
ECTS_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:ECTS|ects)", re.IGNORECASE)

# Credit pattern
CREDIT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:credits?|cr\.?)", re.IGNORECASE)


def extract_courses_from_chunks(
    hits: list[RetrievalHit],
) -> list[CourseInfo]:
    """
    Extract structured course information from chunks.

    Args:
        hits: Retrieved chunks

    Returns:
        List of CourseInfo objects
    """
    courses = []
    seen_codes = set()

    for hit in hits:
        text = hit.text

        # Find course codes in this chunk
        code_matches = COURSE_CODE_PATTERN.findall(text)

        for prefix, number in code_matches:
            code = f"{prefix} {number}"

            # Avoid duplicates
            if code in seen_codes:
                continue
            seen_codes.add(code)

            # Extract ECTS if present
            ects = None
            ects_match = ECTS_PATTERN.search(text)
            if ects_match:
                try:
                    ects = float(ects_match.group(1))
                except ValueError:
                    pass

            # Extract credits if present
            credits = None
            credit_match = CREDIT_PATTERN.search(text)
            if credit_match:
                try:
                    credits = float(credit_match.group(1))
                except ValueError:
                    pass

            # Determine course type from context
            course_type = ""
            text_lower = text.lower()
            for ctype, keywords in COURSE_TYPE_KEYWORDS.items():
                for kw in keywords:
                    if kw in text_lower:
                        course_type = ctype
                        break
                if course_type:
                    break

            # Determine semester from metadata or text
            semester = None
            if hit.metadata.get("semester"):
                try:
                    semester = int(hit.metadata["semester"])
                except (ValueError, TypeError):
                    pass

            courses.append(
                CourseInfo(
                    code=code,
                    title=hit.course_title,
                    ects=ects,
                    credits=credits,
                    semester=semester,
                    department=hit.department,
                    course_type=course_type,
                    chunk_id=hit.chunk_id,
                )
            )

    return courses


def filter_courses(
    courses: list[CourseInfo],
    filters: dict,
) -> list[CourseInfo]:
    """
    Filter courses based on criteria.

    Args:
        courses: List of courses
        filters: Filter criteria

    Returns:
        Filtered course list
    """
    result = courses

    if "department" in filters:
        dept = filters["department"].lower()
        result = [c for c in result if c.department.lower() == dept]

    if "semester" in filters:
        sem = filters["semester"]
        result = [c for c in result if c.semester == sem]

    if "year" in filters:
        year = filters["year"]
        # Year 1 = semesters 1,2; Year 2 = semesters 3,4; etc.
        sem_range = [(year - 1) * 2 + 1, (year - 1) * 2 + 2]
        result = [c for c in result if c.semester in sem_range]

    if "course_type" in filters:
        ctype = filters["course_type"].lower()
        result = [c for c in result if c.course_type.lower() == ctype]

    if "code_prefix" in filters:
        prefix = filters["code_prefix"].upper()
        result = [c for c in result if c.code_prefix == prefix]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Quantitative Handler
# ─────────────────────────────────────────────────────────────────────────────


class QuantitativeHandler:
    """
    Handles quantitative queries about courses.

    Performs counting, summing, and breakdown calculations
    based on structured data extracted from chunks.

    Example:
        >>> handler = QuantitativeHandler()
        >>> result = handler.handle(query, hits)
        >>> print(f"Count: {result.value}")
    """

    def handle(
        self,
        query: str,
        hits: list[RetrievalHit],
    ) -> QuantitativeResult:
        """
        Handle a quantitative query.

        Args:
            query: User query
            hits: Retrieved chunks

        Returns:
            QuantitativeResult
        """
        query_type = detect_quantitative_query(query)
        filters = extract_query_filters(query)

        logger.info(f"Quantitative query type: {query_type}, filters: {filters}")

        # Extract and filter courses
        courses = extract_courses_from_chunks(hits)
        filtered = filter_courses(courses, filters)

        # Handle by type
        if query_type == QuantitativeQueryType.COUNT:
            return self._count(filtered, filters)

        elif query_type == QuantitativeQueryType.SUM_ECTS:
            return self._sum_ects(filtered, filters)

        elif query_type == QuantitativeQueryType.LIST:
            return self._list(filtered, filters)

        elif query_type == QuantitativeQueryType.BREAKDOWN:
            return self._breakdown(filtered, filters)

        elif query_type == QuantitativeQueryType.COMPARE_COUNT:
            # For comparison, don't pre-filter by department
            all_courses = extract_courses_from_chunks(hits)
            return self._compare(all_courses, filters)

        else:
            # Not a quantitative query - return empty result
            return QuantitativeResult(
                query_type=query_type,
                value=0,
                courses=[],
                formatted_data="",
            )

    def _count(
        self,
        courses: list[CourseInfo],
        filters: dict,
    ) -> QuantitativeResult:
        """Count courses matching filters."""
        count = len(courses)

        formatted = self._format_course_list(courses, f"Found {count} courses")

        return QuantitativeResult(
            query_type=QuantitativeQueryType.COUNT,
            value=count,
            courses=courses,
            formatted_data=formatted,
        )

    def _sum_ects(
        self,
        courses: list[CourseInfo],
        filters: dict,
    ) -> QuantitativeResult:
        """Sum ECTS credits."""
        total = sum(c.ects or 0 for c in courses)

        lines = [f"Total ECTS: {total}"]
        lines.append(f"Courses included ({len(courses)}):")
        for c in courses:
            ects_str = f"{c.ects} ECTS" if c.ects else "ECTS unknown"
            lines.append(f"  - {c.code}: {ects_str} [{c.chunk_id}]")

        return QuantitativeResult(
            query_type=QuantitativeQueryType.SUM_ECTS,
            value=total,
            courses=courses,
            formatted_data="\n".join(lines),
        )

    def _list(
        self,
        courses: list[CourseInfo],
        filters: dict,
    ) -> QuantitativeResult:
        """List all matching courses."""
        formatted = self._format_course_list(courses, f"Listing {len(courses)} courses")

        return QuantitativeResult(
            query_type=QuantitativeQueryType.LIST,
            value=len(courses),
            courses=courses,
            formatted_data=formatted,
        )

    def _breakdown(
        self,
        courses: list[CourseInfo],
        filters: dict,
    ) -> QuantitativeResult:
        """Break down courses by semester or type."""
        # Group by semester
        by_semester: dict[int, list[CourseInfo]] = {}
        for c in courses:
            sem = c.semester or 0
            if sem not in by_semester:
                by_semester[sem] = []
            by_semester[sem].append(c)

        lines = ["Course breakdown by semester:"]
        for sem in sorted(by_semester.keys()):
            sem_courses = by_semester[sem]
            sem_label = f"Semester {sem}" if sem > 0 else "Unknown semester"
            lines.append(f"\n{sem_label} ({len(sem_courses)} courses):")
            for c in sem_courses:
                lines.append(f"  - {c.code} [{c.chunk_id}]")

        breakdown_dict = {k: len(v) for k, v in by_semester.items()}

        return QuantitativeResult(
            query_type=QuantitativeQueryType.BREAKDOWN,
            value=breakdown_dict,
            courses=courses,
            formatted_data="\n".join(lines),
            breakdown=breakdown_dict,
        )

    def _compare(
        self,
        courses: list[CourseInfo],
        filters: dict,
    ) -> QuantitativeResult:
        """Compare counts across departments."""
        by_dept: dict[str, list[CourseInfo]] = {}
        for c in courses:
            dept = c.department.upper() or "UNKNOWN"
            if dept not in by_dept:
                by_dept[dept] = []
            by_dept[dept].append(c)

        lines = ["Course comparison by department:"]
        for dept in sorted(by_dept.keys()):
            dept_courses = by_dept[dept]
            lines.append(f"\n{dept}: {len(dept_courses)} courses")
            for c in dept_courses[:5]:  # Show first 5
                lines.append(f"  - {c.code}")
            if len(dept_courses) > 5:
                lines.append(f"  ... and {len(dept_courses) - 5} more")

        comparison_dict = {k: len(v) for k, v in by_dept.items()}

        return QuantitativeResult(
            query_type=QuantitativeQueryType.COMPARE_COUNT,
            value=comparison_dict,
            courses=courses,
            formatted_data="\n".join(lines),
            breakdown=comparison_dict,
        )

    def _format_course_list(
        self,
        courses: list[CourseInfo],
        header: str,
    ) -> str:
        """Format a list of courses for LLM context."""
        lines = [header]

        for c in courses:
            parts = [f"  - {c.code}"]
            if c.ects:
                parts.append(f"({c.ects} ECTS)")
            if c.semester:
                parts.append(f"Sem {c.semester}")
            if c.course_type:
                parts.append(f"[{c.course_type}]")
            parts.append(f"[{c.chunk_id}]")
            lines.append(" ".join(parts))

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────


_handler: Optional[QuantitativeHandler] = None


def get_quantitative_handler() -> QuantitativeHandler:
    """Get or create global quantitative handler."""
    global _handler
    if _handler is None:
        _handler = QuantitativeHandler()
    return _handler


def count_courses(
    hits: list[RetrievalHit],
    filters: Optional[dict] = None,
) -> int:
    """
    Count courses in retrieved chunks.

    Args:
        hits: Retrieved chunks
        filters: Optional filter criteria

    Returns:
        Course count
    """
    courses = extract_courses_from_chunks(hits)
    if filters:
        courses = filter_courses(courses, filters)
    return len(courses)


def sum_ects(
    hits: list[RetrievalHit],
    filters: Optional[dict] = None,
) -> float:
    """
    Sum ECTS credits from retrieved chunks.

    Args:
        hits: Retrieved chunks
        filters: Optional filter criteria

    Returns:
        Total ECTS
    """
    courses = extract_courses_from_chunks(hits)
    if filters:
        courses = filter_courses(courses, filters)
    return sum(c.ects or 0 for c in courses)


def is_quantitative_query(query: str) -> bool:
    """
    Check if a query is quantitative.

    Args:
        query: User query

    Returns:
        True if quantitative
    """
    qtype = detect_quantitative_query(query)
    return qtype != QuantitativeQueryType.NOT_QUANTITATIVE
