"""
Curriculum Lookup Module - Direct course data queries.
======================================================

Provides deterministic lookups for curriculum-structure queries
that cannot reliably be answered through semantic search alone.

Used for:
- "What are the mandatory courses in semester X for department Y?"
- Counting courses by criteria (ECTS, department, etc.)
- Any query requiring exhaustive listing of courses matching criteria
"""

import json
import re
from pathlib import Path
from typing import Optional

from iue_coursecompass.shared.config import get_settings
from iue_coursecompass.shared.logging import get_logger
from iue_coursecompass.shared.schemas import RetrievalHit

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Course Data Cache
# ─────────────────────────────────────────────────────────────────────────────

_courses_cache: Optional[list[dict]] = None


def _load_courses() -> list[dict]:
    """Load courses from processed JSONL file."""
    global _courses_cache
    if _courses_cache is not None:
        return _courses_cache

    settings = get_settings()
    # Use paths.courses_file from settings
    courses_path = Path(settings.paths.courses_file)
    
    # Handle relative path by prepending project root
    if not courses_path.is_absolute():
        from iue_coursecompass.shared.config import _find_project_root
        courses_path = _find_project_root() / courses_path
    
    if not courses_path.exists():
        logger.warning(f"Courses file not found: {courses_path}")
        return []

    courses = []
    with open(courses_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                courses.append(json.loads(line))
    
    _courses_cache = courses
    logger.info(f"Loaded {len(courses)} courses from {courses_path}")
    return courses


def clear_cache():
    """Clear the courses cache."""
    global _courses_cache
    _courses_cache = None


# ─────────────────────────────────────────────────────────────────────────────
# Query Pattern Detection
# ─────────────────────────────────────────────────────────────────────────────

# Patterns for mandatory courses by semester queries
# Use named groups for easier extraction
MANDATORY_SEMESTER_PATTERNS = [
    # "mandatory courses in Semester X for Department"
    r"mandatory\s+courses?\s+(?:in\s+)?semester\s+(?P<sem1>\d+)\s+(?:for\s+)?(?P<dept1>(?:software|computer|electrical\s+and\s+electronics|industrial)(?:\s+engineering)?)",
    # "mandatory courses for Department in Semester X"
    r"mandatory\s+courses?\s+(?:for\s+)?(?P<dept2>(?:software|computer|electrical\s+and\s+electronics|industrial)(?:\s+engineering)?)\s+(?:in\s+)?semester\s+(?P<sem2>\d+)",
    # "Department semester X mandatory"
    r"(?P<dept3>(?:software|computer|electrical\s+and\s+electronics|industrial)(?:\s+engineering)?)\s+semester\s+(?P<sem3>\d+)\s+mandatory",
    # "semester X mandatory courses for Department"
    r"semester\s+(?P<sem4>\d+)\s+mandatory\s+courses?\s+(?:for\s+)?(?P<dept4>(?:software|computer|electrical\s+and\s+electronics|industrial)(?:\s+engineering)?)",
    # Short department codes
    r"mandatory\s+courses?\s+(?:in\s+)?semester\s+(?P<sem5>\d+)\s+(?:for\s+)?(?P<dept5>se|ce|eee|ie)(?:\s|$)",
    r"(?P<dept6>se|ce|eee|ie)\s+semester\s+(?P<sem6>\d+)\s+mandatory",
]

# Department name mappings
DEPARTMENT_ALIASES = {
    "software engineering": "se",
    "software": "se",
    "se": "se",
    "computer engineering": "ce",
    "computer": "ce",
    "ce": "ce",
    "electrical and electronics engineering": "eee",
    "electrical and electronics": "eee",
    "electrical engineering": "eee",
    "eee": "eee",
    "industrial engineering": "ie",
    "industrial": "ie",
    "ie": "ie",
}


def detect_mandatory_semester_query(query: str) -> Optional[tuple[int, str]]:
    """
    Detect if query is asking for mandatory courses in a specific semester.
    
    Args:
        query: User query string
        
    Returns:
        Tuple of (semester_number, department_id) if detected, None otherwise
    """
    query_lower = query.lower()
    
    for pattern in MANDATORY_SEMESTER_PATTERNS:
        match = re.search(pattern, query_lower)
        if match:
            # Use named groups to extract semester and department
            groupdict = match.groupdict()
            
            # Find the semester value (check all possible named groups)
            semester = None
            dept_name = None
            for i in range(1, 7):
                sem_key = f"sem{i}"
                dept_key = f"dept{i}"
                if groupdict.get(sem_key):
                    semester = int(groupdict[sem_key])
                if groupdict.get(dept_key):
                    dept_name = groupdict[dept_key]
            
            if semester and dept_name:
                # Normalize department name
                dept_id = DEPARTMENT_ALIASES.get(dept_name.strip(), None)
                if dept_id and 1 <= semester <= 8:
                    return (semester, dept_id)
    
    return None


def detect_counting_query(query: str) -> Optional[dict]:
    """
    Detect if query is a counting query with specific filters.
    
    Returns:
        Dictionary with filter criteria if detected, None otherwise
    """
    query_lower = query.lower()
    
    result = {
        "query_type": None,
        "filters": {}
    }
    
    # Course code prefix counts
    code_prefix_match = re.search(
        r"how\s+many\s+(?:unique\s+)?courses?\s+(?:have\s+)?(?:a\s+)?(?:course\s+)?code\s+starting\s+with\s+['\"]?(\w+)['\"]?",
        query_lower
    )
    if code_prefix_match:
        result["query_type"] = "count_by_prefix"
        result["filters"]["code_prefix"] = code_prefix_match.group(1).upper()
        return result
    
    # ECTS count
    ects_match = re.search(
        r"how\s+many\s+courses?\s+(?:have\s+)?(?:exactly\s+)?(\d+)\s+ects",
        query_lower
    )
    if ects_match:
        result["query_type"] = "count_by_ects"
        result["filters"]["ects"] = int(ects_match.group(1))
        return result
    
    # Total courses by department
    total_dept_match = re.search(
        r"how\s+many\s+(?:total\s+)?courses?\s+(?:are\s+)?(?:offered\s+)?(?:in\s+)?(?:the\s+)?(\w+(?:\s+engineering)?)\s+department",
        query_lower
    )
    if total_dept_match:
        dept_name = total_dept_match.group(1).strip()
        dept_id = DEPARTMENT_ALIASES.get(dept_name, None)
        if dept_id:
            result["query_type"] = "count_by_department"
            result["filters"]["department"] = dept_id
            return result
    
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Course Lookups
# ─────────────────────────────────────────────────────────────────────────────

def get_mandatory_courses_by_semester(
    semester: int,
    department: str,
) -> list[dict]:
    """
    Get all mandatory courses for a specific semester and department.
    
    Args:
        semester: Semester number (1-8)
        department: Department ID (se, ce, eee, ie)
        
    Returns:
        List of course dictionaries
    """
    courses = _load_courses()
    
    mandatory_courses = [
        c for c in courses
        if c.get("department") == department
        and c.get("semester") == semester
        and c.get("course_type") == "mandatory"
    ]
    
    # Sort by course code for consistent ordering
    mandatory_courses.sort(key=lambda c: c.get("course_code", ""))
    
    logger.info(
        f"Found {len(mandatory_courses)} mandatory courses for {department} semester {semester}"
    )
    
    return mandatory_courses


def count_courses_by_prefix(prefix: str) -> tuple[int, list[dict]]:
    """
    Count unique courses with a given course code prefix.
    
    Args:
        prefix: Course code prefix (e.g., "SE", "CE")
        
    Returns:
        Tuple of (count, list of unique courses)
    """
    courses = _load_courses()
    prefix_upper = prefix.upper()
    
    # Get unique courses by code (same course appears in multiple departments)
    seen_codes = set()
    unique_courses = []
    
    for c in courses:
        code = c.get("course_code", "")
        if code.upper().startswith(prefix_upper) and code not in seen_codes:
            seen_codes.add(code)
            unique_courses.append(c)
    
    unique_courses.sort(key=lambda c: c.get("course_code", ""))
    return len(unique_courses), unique_courses


def count_courses_by_ects(ects: int) -> tuple[int, list[dict]]:
    """
    Count courses with a specific ECTS value.
    
    Args:
        ects: ECTS credit value
        
    Returns:
        Tuple of (count, list of unique courses)
    """
    courses = _load_courses()
    
    # Get unique courses by code
    seen_codes = set()
    matching_courses = []
    
    for c in courses:
        code = c.get("course_code", "")
        course_ects = c.get("ects")
        if course_ects == ects and code not in seen_codes:
            seen_codes.add(code)
            matching_courses.append(c)
    
    matching_courses.sort(key=lambda c: c.get("course_code", ""))
    return len(matching_courses), matching_courses


def count_courses_by_department(department: str) -> tuple[int, dict]:
    """
    Count total courses in a department.
    
    Args:
        department: Department ID
        
    Returns:
        Tuple of (total_count, breakdown_dict)
    """
    courses = _load_courses()
    
    dept_courses = [c for c in courses if c.get("department") == department]
    
    mandatory = [c for c in dept_courses if c.get("course_type") == "mandatory"]
    elective = [c for c in dept_courses if c.get("course_type") == "elective"]
    
    return len(dept_courses), {
        "total": len(dept_courses),
        "mandatory": len(mandatory),
        "elective": len(elective),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Convert to RetrievalHit Format
# ─────────────────────────────────────────────────────────────────────────────

def courses_to_retrieval_hits(courses: list[dict]) -> list[RetrievalHit]:
    """
    Convert course dictionaries to RetrievalHit format for generator.
    
    This allows curriculum lookup results to be used with the standard
    generation pipeline.
    """
    hits = []
    
    for c in courses:
        # Build a text representation of the course
        text_parts = []
        
        course_code = c.get("course_code", "")
        course_title = c.get("course_title", "")
        department = c.get("department", "")
        
        if course_code and course_title:
            text_parts.append(f"{course_code} - {course_title}")
        
        if c.get("ects"):
            text_parts.append(f"ECTS: {c['ects']}")
            
        if c.get("semester"):
            text_parts.append(f"Semester: {c['semester']}")
            
        if c.get("course_type"):
            text_parts.append(f"Type: {c['course_type']}")
            
        if c.get("objectives"):
            text_parts.append(f"Objectives: {c['objectives'][:200]}")
            
        if c.get("prerequisites"):
            text_parts.append(f"Prerequisites: {c['prerequisites']}")
        
        # Generate a pseudo chunk_id
        dept = department or "unknown"
        code = course_code.replace(" ", "_").lower() if course_code else "unknown"
        year = c.get("year_range", "2020-2024").replace("-", "_")
        chunk_id = f"{dept}_{code}_{year}_course_info_0"
        
        # RetrievalHit uses properties that read from metadata
        hit = RetrievalHit(
            chunk_id=chunk_id,
            text="\n".join(text_parts),
            score=1.0,  # Perfect score for direct lookups
            metadata={
                "course_code": course_code,
                "course_title": course_title,
                "department": department,
                "source_url": c.get("source_url", ""),
                "semester": c.get("semester"),
                "course_type": c.get("course_type"),
                "ects": c.get("ects"),
            }
        )
        hits.append(hit)
    
    return hits


def format_mandatory_courses_context(
    courses: list[dict],
    semester: int,
    department: str,
) -> str:
    """
    Format mandatory courses for LLM context.
    
    Returns a structured text that the LLM can use to generate an answer.
    """
    dept_names = {
        "se": "Software Engineering",
        "ce": "Computer Engineering",
        "eee": "Electrical and Electronics Engineering",
        "ie": "Industrial Engineering",
    }
    
    dept_name = dept_names.get(department, department.upper())
    
    lines = [
        f"MANDATORY COURSES FOR {dept_name.upper()} - SEMESTER {semester}",
        f"Total: {len(courses)} courses",
        "",
    ]
    
    for c in courses:
        code = c.get("course_code", "Unknown")
        title = c.get("course_title", "Unknown")
        ects = c.get("ects", "N/A")
        lines.append(f"- {code}: {title} ({ects} ECTS)")
    
    return "\n".join(lines)


def format_counting_context(
    query_type: str,
    count: int,
    details: Optional[dict] = None,
    courses: Optional[list[dict]] = None,
) -> str:
    """
    Format counting result for LLM context.
    """
    lines = []
    
    if query_type == "count_by_prefix":
        prefix = details.get("prefix", "")
        lines.append(f"COURSES WITH CODE STARTING WITH '{prefix}'")
        lines.append(f"Total unique courses: {count}")
        if courses:
            lines.append("")
            for c in courses[:20]:  # Show first 20
                lines.append(f"- {c.get('course_code')}: {c.get('course_title')}")
            if len(courses) > 20:
                lines.append(f"... and {len(courses) - 20} more")
                
    elif query_type == "count_by_ects":
        ects = details.get("ects", 0)
        lines.append(f"COURSES WITH {ects} ECTS CREDITS")
        lines.append(f"Total unique courses: {count}")
        if courses:
            lines.append("")
            for c in courses[:20]:
                lines.append(f"- {c.get('course_code')}: {c.get('course_title')}")
            if len(courses) > 20:
                lines.append(f"... and {len(courses) - 20} more")
                
    elif query_type == "count_by_department":
        lines.append(f"COURSES IN DEPARTMENT")
        lines.append(f"Total: {details.get('total', count)}")
        lines.append(f"Mandatory: {details.get('mandatory', 'N/A')}")
        lines.append(f"Elective: {details.get('elective', 'N/A')}")
    
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main Handler
# ─────────────────────────────────────────────────────────────────────────────

class CurriculumLookup:
    """
    Handles curriculum-structure queries that need deterministic lookups.
    """
    
    def __init__(self):
        # Pre-load courses on initialization
        _load_courses()
    
    def can_handle(self, query: str) -> bool:
        """Check if this query should be handled by curriculum lookup."""
        return (
            detect_mandatory_semester_query(query) is not None
            or detect_counting_query(query) is not None
        )
    
    def handle(self, query: str) -> Optional[tuple[list[RetrievalHit], str]]:
        """
        Handle a curriculum-structure query.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (retrieval_hits, formatted_context) or None if not handled
        """
        # Check for mandatory semester query
        semester_query = detect_mandatory_semester_query(query)
        if semester_query:
            semester, department = semester_query
            courses = get_mandatory_courses_by_semester(semester, department)
            hits = courses_to_retrieval_hits(courses)
            context = format_mandatory_courses_context(courses, semester, department)
            logger.info(
                f"Curriculum lookup: mandatory courses for {department} semester {semester} "
                f"-> {len(courses)} courses"
            )
            return hits, context
        
        # Check for counting query
        counting_query = detect_counting_query(query)
        if counting_query:
            query_type = counting_query["query_type"]
            filters = counting_query["filters"]
            
            if query_type == "count_by_prefix":
                count, courses = count_courses_by_prefix(filters["code_prefix"])
                context = format_counting_context(
                    query_type, count, 
                    {"prefix": filters["code_prefix"]},
                    courses
                )
                hits = courses_to_retrieval_hits(courses[:20])  # Limit hits
                logger.info(f"Curriculum lookup: count by prefix {filters['code_prefix']} -> {count}")
                return hits, context
                
            elif query_type == "count_by_ects":
                count, courses = count_courses_by_ects(filters["ects"])
                context = format_counting_context(
                    query_type, count,
                    {"ects": filters["ects"]},
                    courses
                )
                hits = courses_to_retrieval_hits(courses[:20])
                logger.info(f"Curriculum lookup: count by ects {filters['ects']} -> {count}")
                return hits, context
                
            elif query_type == "count_by_department":
                total, breakdown = count_courses_by_department(filters["department"])
                context = format_counting_context(
                    query_type, total, breakdown
                )
                courses = [c for c in _load_courses() if c.get("department") == filters["department"]]
                hits = courses_to_retrieval_hits(courses[:20])
                logger.info(f"Curriculum lookup: count by dept {filters['department']} -> {total}")
                return hits, context
        
        return None


# Global instance
_curriculum_lookup: Optional[CurriculumLookup] = None


def get_curriculum_lookup() -> CurriculumLookup:
    """Get or create curriculum lookup handler."""
    global _curriculum_lookup
    if _curriculum_lookup is None:
        _curriculum_lookup = CurriculumLookup()
    return _curriculum_lookup


def is_curriculum_query(query: str) -> bool:
    """Check if query should use curriculum lookup."""
    return get_curriculum_lookup().can_handle(query)


def handle_curriculum_query(query: str) -> Optional[tuple[list[RetrievalHit], str]]:
    """Handle a curriculum query if applicable."""
    return get_curriculum_lookup().handle(query)
