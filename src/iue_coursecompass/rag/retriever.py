"""
Retriever Module - Semantic chunk retrieval from vector store.
=============================================================

Handles query embedding and retrieval of relevant chunks:
- Single-department queries with filtering
- Cross-department comparison queries
- Configurable top-k and similarity thresholds
- Query analysis for automatic filter extraction
"""

import re
from typing import Any, Optional

from iue_coursecompass.indexing.vector_store import VectorStore, create_vector_store
from iue_coursecompass.shared.config import get_settings
from iue_coursecompass.shared.logging import get_logger
from iue_coursecompass.shared.schemas import RetrievalHit

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Query Analysis
# ─────────────────────────────────────────────────────────────────────────────

# Mapping of ordinal words to semester numbers
ORDINAL_TO_SEMESTER = {
    "first": 1, "1st": 1, "one": 1,
    "second": 2, "2nd": 2, "two": 2,
    "third": 3, "3rd": 3, "three": 3,
    "fourth": 4, "4th": 4, "four": 4,
    "fifth": 5, "5th": 5, "five": 5,
    "sixth": 6, "6th": 6, "six": 6,
    "seventh": 7, "7th": 7, "seven": 7,
    "eighth": 8, "8th": 8, "eight": 8,
}


def extract_semester_from_query(query: str) -> Optional[int]:
    """
    Extract semester number from a natural language query.
    
    Detects patterns like:
    - "first semester", "1st semester"
    - "semester 1", "semester one"
    - "in the 3rd semester"
    
    Args:
        query: The user's question
        
    Returns:
        Semester number (1-8) if found, None otherwise
    """
    query_lower = query.lower()
    
    # Pattern 1: ordinal + semester (e.g., "first semester", "1st semester")
    for word, num in ORDINAL_TO_SEMESTER.items():
        if re.search(rf'\b{word}\s+semester\b', query_lower):
            logger.debug(f"Extracted semester {num} from pattern '{word} semester'")
            return num
    
    # Pattern 2: semester + number (e.g., "semester 1", "semester one")
    match = re.search(r'\bsemester\s+(\d+|one|two|three|four|five|six|seven|eight)\b', query_lower)
    if match:
        val = match.group(1)
        if val.isdigit():
            num = int(val)
            if 1 <= num <= 8:
                logger.debug(f"Extracted semester {num} from 'semester {val}'")
                return num
        else:
            num = ORDINAL_TO_SEMESTER.get(val)
            if num:
                logger.debug(f"Extracted semester {num} from 'semester {val}'")
                return num
    
    # Pattern 3: number + semester (e.g., "3rd semester", "4 semester")
    match = re.search(r'\b(\d+)(?:st|nd|rd|th)?\s+semester\b', query_lower)
    if match:
        num = int(match.group(1))
        if 1 <= num <= 8:
            logger.debug(f"Extracted semester {num} from '{match.group(0)}'")
            return num
    
    return None


class Retriever:
    """
    Retrieves relevant chunks from the vector store.

    Supports:
    - Single-department queries
    - Multi-department queries
    - Cross-department comparisons (k per department)
    - Configurable filters and thresholds

    Example:
        >>> retriever = Retriever()
        >>> hits = retriever.retrieve("machine learning courses")
        >>> for hit in hits:
        ...     print(f"{hit.course_code}: {hit.score:.3f}")
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ):
        """
        Initialize the retriever.

        Args:
            vector_store: VectorStore instance (creates default if None)
            top_k: Default number of results to return
            similarity_threshold: Minimum similarity score
        """
        settings = get_settings()

        self._vector_store = vector_store
        self._top_k = top_k or settings.get_effective_top_k()
        self._similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.get_effective_threshold()
        )

        # Comparison settings
        self._comparison_top_k_per_dept = settings.retrieval.comparison_top_k_per_dept
        self._comparison_max_total = settings.retrieval.comparison_max_total

        logger.debug(
            f"Retriever initialized: top_k={self._top_k}, "
            f"threshold={self._similarity_threshold}"
        )

    @property
    def vector_store(self) -> VectorStore:
        """Lazy load vector store."""
        if self._vector_store is None:
            self._vector_store = create_vector_store()
        return self._vector_store

    def retrieve(
        self,
        query: str,
        departments: Optional[list[str]] = None,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        year_range: Optional[str] = None,
        course_type: Optional[str] = None,
        semester: Optional[int] = None,
    ) -> list[RetrievalHit]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Query text
            departments: Filter by departments (None = all)
            top_k: Number of results (uses default if None)
            similarity_threshold: Minimum score (uses default if None)
            year_range: Filter by year range
            course_type: Filter by course type
            semester: Filter by semester number (1-8)

        Returns:
            List of RetrievalHit objects sorted by score
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []

        k = top_k or self._top_k
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self._similarity_threshold
        )

        # Build filters
        filters = self._build_filters(departments, year_range, course_type, semester)

        # Query vector store
        hits = self.vector_store.query(
            query_text=query,
            top_k=k,
            filters=filters,
            min_score=threshold,
        )

        logger.info(
            f"Retrieved {len(hits)} chunks for query: '{query[:50]}...' "
            f"(k={k}, threshold={threshold})"
        )

        return hits

    def retrieve_for_comparison(
        self,
        query: str,
        departments: list[str],
        top_k_per_dept: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> dict[str, list[RetrievalHit]]:
        """
        Retrieve chunks separately for each department (for comparisons).

        Args:
            query: Query text
            departments: List of department IDs to compare
            top_k_per_dept: Results per department
            similarity_threshold: Minimum score

        Returns:
            Dictionary mapping department ID to list of hits
        """
        if not query or not query.strip():
            return {}

        k_per_dept = top_k_per_dept or self._comparison_top_k_per_dept
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self._similarity_threshold
        )

        results = self.vector_store.query_by_department(
            query_text=query,
            departments=departments,
            top_k_per_dept=k_per_dept,
            min_score=threshold,
        )

        total_hits = sum(len(hits) for hits in results.values())
        logger.info(
            f"Retrieved {total_hits} chunks across {len(departments)} departments "
            f"for comparison query"
        )

        return results

    def retrieve_all_for_comparison(
        self,
        query: str,
        departments: list[str],
        top_k_per_dept: Optional[int] = None,
        max_total: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> list[RetrievalHit]:
        """
        Retrieve chunks for comparison, merged into single list.

        Ensures balanced representation from each department.

        Args:
            query: Query text
            departments: Departments to compare
            top_k_per_dept: Results per department
            max_total: Maximum total results
            similarity_threshold: Minimum score

        Returns:
            Merged list of hits, balanced across departments
        """
        dept_results = self.retrieve_for_comparison(
            query=query,
            departments=departments,
            top_k_per_dept=top_k_per_dept,
            similarity_threshold=similarity_threshold,
        )

        # Merge results, maintaining department balance
        max_total = max_total or self._comparison_max_total
        merged = self._merge_department_results(dept_results, max_total)

        return merged

    def _build_filters(
        self,
        departments: Optional[list[str]],
        year_range: Optional[str],
        course_type: Optional[str],
        semester: Optional[int] = None,
    ) -> Optional[dict[str, Any]]:
        """Build filter dictionary for vector store query."""
        filters: dict[str, Any] = {}

        if departments:
            if len(departments) == 1:
                filters["department"] = departments[0].lower()
            else:
                filters["department"] = [d.lower() for d in departments]

        if year_range:
            filters["year_range"] = year_range

        if course_type:
            filters["course_type"] = course_type

        if semester is not None:
            filters["semester"] = semester

        return filters if filters else None

    def _merge_department_results(
        self,
        dept_results: dict[str, list[RetrievalHit]],
        max_total: int,
    ) -> list[RetrievalHit]:
        """
        Merge department results with balanced representation.

        Uses round-robin selection to ensure each department is represented.
        """
        if not dept_results:
            return []

        # Round-robin merge
        merged: list[RetrievalHit] = []
        indices = {dept: 0 for dept in dept_results}
        depts = list(dept_results.keys())

        while len(merged) < max_total:
            added_any = False

            for dept in depts:
                if indices[dept] < len(dept_results[dept]):
                    merged.append(dept_results[dept][indices[dept]])
                    indices[dept] += 1
                    added_any = True

                    if len(merged) >= max_total:
                        break

            if not added_any:
                break  # All departments exhausted

        # Sort by score
        merged.sort(key=lambda h: h.score, reverse=True)

        return merged

    def get_retrieval_stats(self, hits: list[RetrievalHit]) -> dict[str, Any]:
        """
        Get statistics about retrieval results.

        Args:
            hits: List of retrieval hits

        Returns:
            Statistics dictionary
        """
        if not hits:
            return {
                "count": 0,
                "max_score": 0.0,
                "min_score": 0.0,
                "avg_score": 0.0,
                "departments": [],
                "courses": [],
            }

        scores = [h.score for h in hits]
        departments = list(set(h.department for h in hits))
        courses = list(set(h.course_code for h in hits))

        return {
            "count": len(hits),
            "max_score": max(scores),
            "min_score": min(scores),
            "avg_score": sum(scores) / len(scores),
            "departments": departments,
            "courses": courses,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────


# Global retriever instance
_retriever: Optional[Retriever] = None


def get_retriever() -> Retriever:
    """Get the global retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def retrieve(
    query: str,
    departments: Optional[list[str]] = None,
    top_k: Optional[int] = None,
    similarity_threshold: Optional[float] = None,
) -> list[RetrievalHit]:
    """
    Retrieve relevant chunks for a query.

    Convenience function using the global retriever.

    Args:
        query: Query text
        departments: Filter by departments
        top_k: Number of results
        similarity_threshold: Minimum score

    Returns:
        List of RetrievalHit objects
    """
    return get_retriever().retrieve(
        query=query,
        departments=departments,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )


def retrieve_for_comparison(
    query: str,
    departments: list[str],
    top_k_per_dept: Optional[int] = None,
) -> dict[str, list[RetrievalHit]]:
    """
    Retrieve chunks for cross-department comparison.

    Convenience function using the global retriever.
    """
    return get_retriever().retrieve_for_comparison(
        query=query,
        departments=departments,
        top_k_per_dept=top_k_per_dept,
    )
