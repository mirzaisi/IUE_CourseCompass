"""
Manifest Module - Embedding index versioning and metadata.
==========================================================

Tracks embedding index metadata for:
- Deterministic builds (dataset version hash)
- Embedding model tracking
- Index version management
- Rebuild detection
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from iue_coursecompass.shared.config import get_settings
from iue_coursecompass.shared.logging import get_logger
from iue_coursecompass.shared.schemas import EmbeddingManifest
from iue_coursecompass.shared.utils import (
    compute_dataset_version,
    ensure_directory,
    load_json,
    save_json,
)

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Manifest Manager
# ─────────────────────────────────────────────────────────────────────────────


class ManifestManager:
    """
    Manages embedding index manifests.

    Manifests track:
    - Dataset version (hash of chunks file)
    - Embedding provider and model used
    - Creation timestamp
    - Statistics (chunk/course counts)

    This enables:
    - Detecting when rebuild is needed
    - Tracking multiple index versions
    - Reproducible builds

    Example:
        >>> manager = ManifestManager()
        >>> manifest = manager.create_manifest(provider, chunks)
        >>> print(manifest.dataset_version)
    """

    def __init__(self, manifests_dir: Optional[Path] = None):
        """
        Initialize the manifest manager.

        Args:
            manifests_dir: Directory for manifest files
        """
        settings = get_settings()
        self.manifests_dir = manifests_dir or settings.resolved_paths.manifests_dir
        ensure_directory(self.manifests_dir)

        logger.debug(f"Manifest manager initialized: {self.manifests_dir}")

    def _get_manifest_path(self, dataset_version: str) -> Path:
        """Get the file path for a manifest."""
        return self.manifests_dir / f"{dataset_version}.json"

    def create_manifest(
        self,
        dataset_version: str,
        embedding_provider: str,
        embedding_model: str,
        embedding_dimensions: int,
        total_chunks: int,
        total_courses: int,
        departments: list[str],
        index_path: str = "",
        chunks_file: str = "",
    ) -> EmbeddingManifest:
        """
        Create and save a new manifest.

        Args:
            dataset_version: Hash of the chunks file
            embedding_provider: Provider name (sbert, gemini)
            embedding_model: Model name
            embedding_dimensions: Vector dimensions
            total_chunks: Number of chunks indexed
            total_courses: Number of courses
            departments: List of department IDs
            index_path: Path to index directory
            chunks_file: Path to source chunks file

        Returns:
            Created EmbeddingManifest
        """
        manifest = EmbeddingManifest(
            dataset_version=dataset_version,
            created_at=datetime.utcnow(),
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            total_chunks=total_chunks,
            total_courses=total_courses,
            departments=departments,
            index_path=index_path,
            chunks_file=chunks_file,
        )

        # Save manifest
        manifest_path = self._get_manifest_path(dataset_version)
        save_json(manifest_path, manifest.model_dump())

        logger.info(f"Created manifest: {dataset_version} ({total_chunks} chunks)")
        return manifest

    def load_manifest(self, dataset_version: str) -> Optional[EmbeddingManifest]:
        """
        Load a manifest by dataset version.

        Args:
            dataset_version: Dataset version hash

        Returns:
            EmbeddingManifest or None if not found
        """
        manifest_path = self._get_manifest_path(dataset_version)

        if not manifest_path.exists():
            logger.debug(f"Manifest not found: {dataset_version}")
            return None

        try:
            data = load_json(manifest_path)
            manifest = EmbeddingManifest.model_validate(data)
            logger.debug(f"Loaded manifest: {dataset_version}")
            return manifest
        except Exception as e:
            logger.warning(f"Failed to load manifest {dataset_version}: {e}")
            return None

    def get_latest_manifest(self) -> Optional[EmbeddingManifest]:
        """
        Get the most recently created manifest.

        Returns:
            Most recent EmbeddingManifest or None if no manifests exist
        """
        manifests = self.list_manifests()
        if not manifests:
            return None

        # Sort by creation time (newest first)
        manifests.sort(key=lambda m: m.created_at, reverse=True)
        return manifests[0]

    def list_manifests(self) -> list[EmbeddingManifest]:
        """
        List all available manifests.

        Returns:
            List of EmbeddingManifest objects
        """
        manifests = []

        for manifest_file in self.manifests_dir.glob("*.json"):
            try:
                data = load_json(manifest_file)
                manifest = EmbeddingManifest.model_validate(data)
                manifests.append(manifest)
            except Exception as e:
                logger.warning(f"Failed to load manifest {manifest_file}: {e}")
                continue

        return manifests

    def delete_manifest(self, dataset_version: str) -> bool:
        """
        Delete a manifest.

        Args:
            dataset_version: Dataset version to delete

        Returns:
            True if deleted, False if not found
        """
        manifest_path = self._get_manifest_path(dataset_version)

        if manifest_path.exists():
            manifest_path.unlink()
            logger.info(f"Deleted manifest: {dataset_version}")
            return True

        return False

    def needs_rebuild(
        self,
        chunks_file: Path,
        embedding_provider: str,
        embedding_model: str,
    ) -> tuple[bool, str, Optional[EmbeddingManifest]]:
        """
        Check if the index needs to be rebuilt.

        Rebuild is needed if:
        - Chunks file has changed (different hash)
        - Embedding provider/model has changed
        - No manifest exists

        Args:
            chunks_file: Path to chunks JSONL file
            embedding_provider: Current embedding provider
            embedding_model: Current embedding model

        Returns:
            Tuple of (needs_rebuild, reason, existing_manifest)
        """
        # Compute current dataset version
        if not chunks_file.exists():
            return True, "Chunks file does not exist", None

        current_version = compute_dataset_version(chunks_file)

        # Check for existing manifest
        manifest = self.load_manifest(current_version)

        if manifest is None:
            # Check if any manifest exists
            latest = self.get_latest_manifest()
            if latest is None:
                return True, "No existing index", None
            else:
                return True, f"Dataset changed (was {latest.dataset_version})", latest

        # Check if provider/model changed
        if manifest.embedding_provider != embedding_provider:
            return (
                True,
                f"Provider changed ({manifest.embedding_provider} -> {embedding_provider})",
                manifest,
            )

        if manifest.embedding_model != embedding_model:
            return (
                True,
                f"Model changed ({manifest.embedding_model} -> {embedding_model})",
                manifest,
            )

        # No rebuild needed
        return False, "Index is up to date", manifest

    def get_index_path(self, dataset_version: str) -> Path:
        """
        Get the index directory path for a dataset version.

        Args:
            dataset_version: Dataset version hash

        Returns:
            Path to index directory
        """
        settings = get_settings()
        return settings.resolved_paths.index_dir / dataset_version


# ─────────────────────────────────────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────────────────────────────────────


# Global manager instance
_manager: Optional[ManifestManager] = None


def get_manifest_manager() -> ManifestManager:
    """Get the global manifest manager instance."""
    global _manager
    if _manager is None:
        _manager = ManifestManager()
    return _manager


def create_manifest(
    dataset_version: str,
    embedding_provider: str,
    embedding_model: str,
    embedding_dimensions: int,
    total_chunks: int,
    total_courses: int,
    departments: list[str],
    index_path: str = "",
    chunks_file: str = "",
) -> EmbeddingManifest:
    """
    Create and save a new manifest.

    Convenience function using the global manager.
    """
    return get_manifest_manager().create_manifest(
        dataset_version=dataset_version,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        total_chunks=total_chunks,
        total_courses=total_courses,
        departments=departments,
        index_path=index_path,
        chunks_file=chunks_file,
    )


def load_manifest(dataset_version: str) -> Optional[EmbeddingManifest]:
    """
    Load a manifest by dataset version.

    Convenience function using the global manager.
    """
    return get_manifest_manager().load_manifest(dataset_version)


def get_latest_manifest() -> Optional[EmbeddingManifest]:
    """
    Get the most recent manifest.

    Convenience function using the global manager.
    """
    return get_manifest_manager().get_latest_manifest()


def check_rebuild_needed(
    chunks_file: Optional[Path] = None,
    embedding_provider: Optional[str] = None,
    embedding_model: Optional[str] = None,
) -> tuple[bool, str, Optional[EmbeddingManifest]]:
    """
    Check if index rebuild is needed.

    Args:
        chunks_file: Path to chunks file (uses default if None)
        embedding_provider: Provider name (uses config if None)
        embedding_model: Model name (uses config if None)

    Returns:
        Tuple of (needs_rebuild, reason, existing_manifest)
    """
    settings = get_settings()

    if chunks_file is None:
        chunks_file = settings.resolved_paths.chunks_file

    if embedding_provider is None:
        embedding_provider = settings.get_effective_embedding_provider()

    if embedding_model is None:
        if embedding_provider == "sbert":
            embedding_model = settings.embeddings.sbert.model_name
        else:
            embedding_model = settings.embeddings.gemini.model_name

    return get_manifest_manager().needs_rebuild(
        chunks_file=chunks_file,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
    )
