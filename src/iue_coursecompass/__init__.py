"""
IUE CourseCompass - RAG System for Course Information Retrieval
===============================================================

A multi-department Retrieval-Augmented Generation system that scrapes, cleans,
chunks, embeds, and semantically indexes IUE Faculty of Engineering course and
curriculum pages (2020–2024) for:

- Software Engineering (SE)
- Computer Engineering (CE)
- Electrical & Electronics Engineering (EEE)
- Industrial Engineering (IE)

The system supports department-specific queries and cross-department comparisons
using a RAG workflow: embed query → retrieve relevant chunks → generate grounded
response with citations.
"""

__version__ = "0.1.0"
__author__ = "IUE CourseCompass Team"
__license__ = "MIT"

# Public API - lazy imports to avoid circular dependencies and speed up startup
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    # Main modules (imported on demand)
    "shared",
    "ingestion",
    "indexing",
    "rag",
    "evaluation",
    "app",
    "cli",
]
