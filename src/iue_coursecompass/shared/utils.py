"""
Utilities Module - Common helper functions.
==========================================

Provides utility functions for:
- Hashing (SHA256 for deterministic versioning)
- ID generation (stable, reproducible IDs)
- File I/O (JSON, JSONL)
- Directory management
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterator, TypeVar

from pydantic import BaseModel

from iue_coursecompass.shared.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


# ─────────────────────────────────────────────────────────────────────────────
# Hashing Functions
# ─────────────────────────────────────────────────────────────────────────────


def compute_hash(text: str, algorithm: str = "sha256") -> str:
    """
    Compute hash of text content.

    Args:
        text: Text to hash
        algorithm: Hash algorithm (default: sha256)

    Returns:
        Hexadecimal hash string

    Example:
        >>> compute_hash("hello world")
        'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'
    """
    hasher = hashlib.new(algorithm)
    hasher.update(text.encode("utf-8"))
    return hasher.hexdigest()


def compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """
    Compute hash of file contents.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm (default: sha256)

    Returns:
        Hexadecimal hash string

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    hasher = hashlib.new(algorithm)
    
    with open(file_path, "rb") as f:
        # Read in chunks for large files
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    
    return hasher.hexdigest()


def compute_dataset_version(chunks_file: Path) -> str:
    """
    Compute dataset version hash from chunks JSONL file.

    The hash is computed over the canonicalized (sorted) chunks
    to ensure deterministic versioning.

    Args:
        chunks_file: Path to chunks.jsonl

    Returns:
        SHA256 hash string (first 16 chars for brevity)
    """
    if not chunks_file.exists():
        return "empty"

    # Read and sort chunks by chunk_id for deterministic ordering
    chunks = list(load_jsonl(chunks_file))
    chunks.sort(key=lambda x: x.get("chunk_id", ""))

    # Compute hash over canonicalized JSON
    canonical = json.dumps(chunks, sort_keys=True, ensure_ascii=False)
    full_hash = compute_hash(canonical)

    return full_hash[:16]


# ─────────────────────────────────────────────────────────────────────────────
# ID Generation
# ─────────────────────────────────────────────────────────────────────────────


def normalize_code(code: str) -> str:
    """
    Normalize a course code for use in IDs.

    Args:
        code: Course code (e.g., "SE 301", "COMP101")

    Returns:
        Normalized code (lowercase, spaces replaced with underscores)

    Example:
        >>> normalize_code("SE 301")
        'se_301'
    """
    # Remove extra whitespace, replace spaces with underscores, lowercase
    normalized = re.sub(r"\s+", "_", code.strip().lower())
    # Remove any non-alphanumeric characters except underscores
    normalized = re.sub(r"[^a-z0-9_]", "", normalized)
    return normalized


def generate_course_id(department: str, course_code: str, year_range: str) -> str:
    """
    Generate stable course ID.

    Format: {department}_{normalized_code}_{year_range}

    Args:
        department: Department ID (se, ce, eee, ie)
        course_code: Course code
        year_range: Academic year range (e.g., "2023-2024")

    Returns:
        Stable course ID

    Example:
        >>> generate_course_id("se", "SE 301", "2023-2024")
        'se_se_301_2023-2024'
    """
    dept_normalized = department.lower().strip()
    code_normalized = normalize_code(course_code)
    year_normalized = year_range.strip()

    return f"{dept_normalized}_{code_normalized}_{year_normalized}"


def generate_chunk_id(
    course_id: str,
    section_name: str,
    chunk_index: int,
    text_hash: str,
) -> str:
    """
    Generate stable chunk ID.

    Format: {course_id}_{section}_{index}_{hash_prefix}

    Args:
        course_id: Parent course ID
        section_name: Section name (objectives, description, etc.)
        chunk_index: Index of chunk within section
        text_hash: Hash of chunk text content

    Returns:
        Stable chunk ID

    Example:
        >>> generate_chunk_id("se_se_301_2023-2024", "description", 0, "abc123...")
        'se_se_301_2023-2024_description_0_abc123'
    """
    section_normalized = section_name.lower().replace(" ", "_")
    hash_prefix = text_hash[:8]

    return f"{course_id}_{section_normalized}_{chunk_index}_{hash_prefix}"


# ─────────────────────────────────────────────────────────────────────────────
# Directory Management
# ─────────────────────────────────────────────────────────────────────────────


def ensure_directory(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        The path (for chaining)

    Example:
        >>> ensure_directory(Path("data/processed"))
        PosixPath('data/processed')
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_directory(file_path: Path) -> Path:
    """
    Ensure the parent directory of a file exists.

    Args:
        file_path: File path

    Returns:
        The file path (for chaining)
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


# ─────────────────────────────────────────────────────────────────────────────
# JSON File I/O
# ─────────────────────────────────────────────────────────────────────────────


def load_json(file_path: Path) -> Any:
    """
    Load data from a JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(file_path: Path, data: Any, indent: int = 2) -> None:
    """
    Save data to a JSON file.

    Args:
        file_path: Path to JSON file
        data: Data to save (must be JSON serializable)
        indent: Indentation level (default: 2)
    """
    file_path = Path(file_path)
    ensure_parent_directory(file_path)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)

    logger.debug(f"Saved JSON to {file_path}")


# ─────────────────────────────────────────────────────────────────────────────
# JSONL File I/O (for large datasets)
# ─────────────────────────────────────────────────────────────────────────────


def load_jsonl(file_path: Path) -> Iterator[dict[str, Any]]:
    """
    Load data from a JSONL (JSON Lines) file.

    Yields one record at a time for memory efficiency.

    Args:
        file_path: Path to JSONL file

    Yields:
        Parsed JSON objects

    Example:
        >>> for chunk in load_jsonl(Path("chunks.jsonl")):
        ...     print(chunk["chunk_id"])
    """
    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON at line {line_num} in {file_path}: {e}")
                continue


def load_jsonl_as_list(file_path: Path) -> list[dict[str, Any]]:
    """
    Load entire JSONL file into a list.

    Args:
        file_path: Path to JSONL file

    Returns:
        List of parsed JSON objects
    """
    return list(load_jsonl(file_path))


def save_jsonl(file_path: Path, items: list[dict[str, Any]] | Iterator[dict[str, Any]]) -> int:
    """
    Save data to a JSONL (JSON Lines) file.

    Args:
        file_path: Path to JSONL file
        items: Iterable of dictionaries to save

    Returns:
        Number of items saved
    """
    file_path = Path(file_path)
    ensure_parent_directory(file_path)

    count = 0
    with open(file_path, "w", encoding="utf-8") as f:
        for item in items:
            line = json.dumps(item, ensure_ascii=False, default=str)
            f.write(line + "\n")
            count += 1

    logger.debug(f"Saved {count} items to JSONL at {file_path}")
    return count


def append_jsonl(file_path: Path, item: dict[str, Any]) -> None:
    """
    Append a single item to a JSONL file.

    Args:
        file_path: Path to JSONL file
        item: Dictionary to append
    """
    file_path = Path(file_path)
    ensure_parent_directory(file_path)

    with open(file_path, "a", encoding="utf-8") as f:
        line = json.dumps(item, ensure_ascii=False, default=str)
        f.write(line + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Model I/O Helpers
# ─────────────────────────────────────────────────────────────────────────────


def load_models_from_json(file_path: Path, model_class: type[T]) -> list[T]:
    """
    Load a list of Pydantic models from a JSON file.

    Args:
        file_path: Path to JSON file containing a list
        model_class: Pydantic model class

    Returns:
        List of model instances
    """
    data = load_json(file_path)
    if not isinstance(data, list):
        data = [data]
    return [model_class.model_validate(item) for item in data]


def load_models_from_jsonl(file_path: Path, model_class: type[T]) -> Iterator[T]:
    """
    Load Pydantic models from a JSONL file.

    Args:
        file_path: Path to JSONL file
        model_class: Pydantic model class

    Yields:
        Model instances
    """
    for item in load_jsonl(file_path):
        yield model_class.model_validate(item)


def save_models_to_json(file_path: Path, models: list[BaseModel], indent: int = 2) -> None:
    """
    Save a list of Pydantic models to a JSON file.

    Args:
        file_path: Path to JSON file
        models: List of Pydantic model instances
        indent: Indentation level
    """
    data = [model.model_dump() for model in models]
    save_json(file_path, data, indent=indent)


def save_models_to_jsonl(file_path: Path, models: list[BaseModel] | Iterator[BaseModel]) -> int:
    """
    Save Pydantic models to a JSONL file.

    Args:
        file_path: Path to JSONL file
        models: Iterable of Pydantic model instances

    Returns:
        Number of models saved
    """
    items = (model.model_dump() for model in models)
    return save_jsonl(file_path, items)


# ─────────────────────────────────────────────────────────────────────────────
# Text Utilities
# ─────────────────────────────────────────────────────────────────────────────


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def clean_whitespace(text: str) -> str:
    """
    Clean excessive whitespace from text.

    Args:
        text: Text to clean

    Returns:
        Text with normalized whitespace
    """
    # Replace multiple spaces/tabs with single space
    text = re.sub(r"[ \t]+", " ", text)
    # Replace multiple newlines with double newline (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace
    return text.strip()
