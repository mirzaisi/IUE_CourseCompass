"""
Configuration Module - Load and validate application settings.
==============================================================

Loads configuration from:
1. config/settings.yaml (defaults)
2. Environment variables from .env file
3. Environment variables from system

Environment variables override YAML defaults.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env file early
load_dotenv()

# Find project root (where pyproject.toml is located)
def _find_project_root() -> Path:
    """Find the project root directory by looking for pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback to current working directory
    return Path.cwd()


PROJECT_ROOT = _find_project_root()
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_CONFIG_FILE = CONFIG_DIR / "settings.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Nested Configuration Models
# ─────────────────────────────────────────────────────────────────────────────


class DepartmentConfig(BaseModel):
    """Configuration for a single department."""

    id: str
    name: str
    full_name: str
    year_ranges: list[str] = Field(default_factory=list)
    base_url: str = ""


class ScrapingConfig(BaseModel):
    """Scraping-related settings."""

    rate_limit: float = 1.0
    timeout: int = 30
    max_retries: int = 3
    retry_min_wait: int = 1
    retry_max_wait: int = 10
    user_agent: str = "IUE-CourseCompass/0.1.0"
    cache_enabled: bool = True
    cache_expiry_days: int = 30


class ChunkingConfig(BaseModel):
    """Text chunking settings."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    section_names: list[str] = Field(
        default_factory=lambda: [
            "objectives",
            "description",
            "prerequisites",
            "weekly_topics",
            "assessment",
            "learning_outcomes",
        ]
    )


class SBERTConfig(BaseModel):
    """SBERT embeddings settings."""

    model_name: str = "all-MiniLM-L6-v2"
    dimensions: int = 384
    device: str = "auto"
    batch_size: int = 32


class GeminiEmbeddingConfig(BaseModel):
    """Gemini embeddings settings."""

    model_name: str = "text-embedding-004"
    dimensions: int = 768
    batch_size: int = 100
    task_type: str = "RETRIEVAL_DOCUMENT"


class EmbeddingsConfig(BaseModel):
    """Embeddings provider settings."""

    provider: str = "sbert"
    sbert: SBERTConfig = Field(default_factory=SBERTConfig)
    gemini: GeminiEmbeddingConfig = Field(default_factory=GeminiEmbeddingConfig)


class RetrievalConfig(BaseModel):
    """Retrieval settings."""

    top_k: int = 10
    similarity_threshold: float = 0.3
    comparison_top_k_per_dept: int = 5
    comparison_max_total: int = 20
    rerank_enabled: bool = False


class GenerationConfig(BaseModel):
    """LLM generation settings."""

    model_name: str = "gemini-2.0-flash-exp"
    temperature: float = 0.2
    max_output_tokens: int = 2048
    top_p: float = 0.95
    top_k: int = 40
    safety_settings: str = "default"


class GroundingConfig(BaseModel):
    """Grounding and hallucination prevention settings."""

    require_citations: bool = True
    abstain_threshold: float = 0.25
    max_uncited_sentences: int = 2
    strict_grounding_keywords: list[str] = Field(
        default_factory=lambda: ["does", "is there", "exist", "offered", "have"]
    )


class QuantitativeConfig(BaseModel):
    """Quantitative query detection settings."""

    detection_keywords: list[str] = Field(
        default_factory=lambda: [
            "how many",
            "count",
            "total",
            "number of",
            "sum",
            "ects",
            "credits",
            "highest",
            "lowest",
            "average",
            "most",
            "least",
            "percentage",
            "compare",
        ]
    )


class PathsConfig(BaseModel):
    """Data paths configuration."""

    data_dir: str = "data"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    courses_file: str = "data/processed/courses.json"
    chunks_file: str = "data/processed/chunks.jsonl"
    index_dir: str = "data/index"
    manifests_dir: str = "data/manifests"
    questions_dir: str = "evaluation/question_sets"

    def resolve(self, base_path: Path) -> "ResolvedPaths":
        """Resolve paths relative to a base path."""
        return ResolvedPaths(
            data_dir=base_path / self.data_dir,
            raw_dir=base_path / self.raw_dir,
            processed_dir=base_path / self.processed_dir,
            courses_file=base_path / self.courses_file,
            chunks_file=base_path / self.chunks_file,
            index_dir=base_path / self.index_dir,
            manifests_dir=base_path / self.manifests_dir,
            questions_dir=base_path / self.questions_dir,
        )


class ResolvedPaths(BaseModel):
    """Resolved absolute paths."""

    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    courses_file: Path
    chunks_file: Path
    index_dir: Path
    manifests_dir: Path
    questions_dir: Path

    model_config = {"arbitrary_types_allowed": True}


class LoggingConfig(BaseModel):
    """Logging settings."""

    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    rich_console: bool = True
    file: str = ""


class EvaluationConfig(BaseModel):
    """Evaluation harness settings."""

    default_question_set: str = "default_questions.yaml"
    output_dir: str = "evaluation/results"
    metrics: list[str] = Field(
        default_factory=lambda: ["retrieval_accuracy", "groundedness", "hallucination_rate"]
    )
    llm_judge_enabled: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Main Settings Class
# ─────────────────────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    """
    Main application settings.

    Loads from:
    1. config/settings.yaml (defaults)
    2. Environment variables

    Environment variables override YAML settings.
    """

    model_config = SettingsConfigDict(
        env_prefix="",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys (from environment only)
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")

    # Top-level environment overrides
    embedding_provider: Optional[str] = Field(default=None, validation_alias="EMBEDDING_PROVIDER")
    gemini_model: Optional[str] = Field(default=None, validation_alias="GEMINI_MODEL")
    sbert_model: Optional[str] = Field(default=None, validation_alias="SBERT_MODEL")
    top_k: Optional[int] = Field(default=None, validation_alias="TOP_K")
    similarity_threshold: Optional[float] = Field(
        default=None, validation_alias="SIMILARITY_THRESHOLD"
    )
    log_level: Optional[str] = Field(default=None, validation_alias="LOG_LEVEL")

    # Nested configurations (from YAML)
    departments: list[DepartmentConfig] = Field(default_factory=list)
    scraping: ScrapingConfig = Field(default_factory=ScrapingConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    grounding: GroundingConfig = Field(default_factory=GroundingConfig)
    quantitative: QuantitativeConfig = Field(default_factory=QuantitativeConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    # Computed properties
    _project_root: Path = PROJECT_ROOT
    _resolved_paths: Optional[ResolvedPaths] = None

    @field_validator("gemini_api_key", mode="before")
    @classmethod
    def validate_api_key(cls, v: Any) -> str:
        """Allow empty API key but warn if using Gemini features."""
        if v is None:
            return ""
        return str(v)

    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return self._project_root

    @property
    def resolved_paths(self) -> ResolvedPaths:
        """Get resolved absolute paths."""
        if self._resolved_paths is None:
            self._resolved_paths = self.paths.resolve(self._project_root)
        return self._resolved_paths

    def get_department(self, dept_id: str) -> Optional[DepartmentConfig]:
        """Get department configuration by ID."""
        for dept in self.departments:
            if dept.id.lower() == dept_id.lower():
                return dept
        return None

    def get_department_ids(self) -> list[str]:
        """Get list of all department IDs."""
        return [d.id for d in self.departments]

    def get_effective_embedding_provider(self) -> str:
        """Get the effective embedding provider (env override or config)."""
        if self.embedding_provider:
            return self.embedding_provider.lower()
        return self.embeddings.provider.lower()

    def get_effective_top_k(self) -> int:
        """Get the effective top-k value (env override or config)."""
        if self.top_k is not None:
            return self.top_k
        return self.retrieval.top_k

    def get_effective_threshold(self) -> float:
        """Get the effective similarity threshold (env override or config)."""
        if self.similarity_threshold is not None:
            return self.similarity_threshold
        return self.retrieval.similarity_threshold

    def get_effective_log_level(self) -> str:
        """Get the effective log level (env override or config)."""
        if self.log_level:
            return self.log_level.upper()
        return self.logging.level.upper()


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load configuration from YAML file."""
    if not config_path.exists():
        return {}

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data if data else {}


def _create_settings(config_path: Optional[Path] = None) -> Settings:
    """Create settings instance by merging YAML defaults with environment."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_FILE

    # Load YAML defaults
    yaml_config = _load_yaml_config(config_path)

    # Create settings with YAML as defaults, env vars will override
    return Settings(**yaml_config)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get the singleton settings instance.

    Returns:
        Settings instance with merged configuration

    Example:
        >>> settings = get_settings()
        >>> print(settings.retrieval.top_k)
        10
    """
    return _create_settings()


def reload_settings() -> Settings:
    """
    Force reload of settings (clears cache).

    Returns:
        Fresh Settings instance
    """
    get_settings.cache_clear()
    return get_settings()
