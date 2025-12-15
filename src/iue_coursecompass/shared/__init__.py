"""
Shared Module - Common utilities, configuration, schemas, and logging.
======================================================================

This module provides foundational components used across all other modules:

- config: Configuration loading and management
- logging: Structured logging setup
- schemas: Pydantic data models
- utils: Utility functions (hashing, file I/O, etc.)
"""

from iue_coursecompass.shared.config import get_settings, Settings
from iue_coursecompass.shared.logging import get_logger, setup_logging
from iue_coursecompass.shared.schemas import (
    CourseRecord,
    ChunkRecord,
    RetrievalHit,
    AnswerResponse,
    EmbeddingManifest,
    DepartmentConfig,
)
from iue_coursecompass.shared.utils import (
    compute_hash,
    compute_file_hash,
    generate_course_id,
    generate_chunk_id,
    ensure_directory,
    load_json,
    save_json,
    load_jsonl,
    save_jsonl,
)

__all__ = [
    # Config
    "get_settings",
    "Settings",
    # Logging
    "get_logger",
    "setup_logging",
    # Schemas
    "CourseRecord",
    "ChunkRecord",
    "RetrievalHit",
    "AnswerResponse",
    "EmbeddingManifest",
    "DepartmentConfig",
    # Utils
    "compute_hash",
    "compute_file_hash",
    "generate_course_id",
    "generate_chunk_id",
    "ensure_directory",
    "load_json",
    "save_json",
    "load_jsonl",
    "save_jsonl",
]
