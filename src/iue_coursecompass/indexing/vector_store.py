"""
Vector Store Module - ChromaDB wrapper for vector storage and retrieval.
========================================================================

Provides a high-level interface to ChromaDB for:
- Persistent vector storage
- Metadata filtering (department, year, course type)
- Semantic similarity search
- Batch operations
"""

from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from iue_coursecompass.indexing.embeddings_base import EmbeddingProvider, get_embedding_provider
from iue_coursecompass.shared.config import get_settings
from iue_coursecompass.shared.logging import get_logger
from iue_coursecompass.shared.schemas import ChunkRecord, RetrievalHit

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Vector Store Class
# ─────────────────────────────────────────────────────────────────────────────


class VectorStore:
    """
    ChromaDB wrapper for vector storage and retrieval.

    Features:
    - Persistent local storage
    - Metadata filtering (department, year_range, course_type, etc.)
    - Semantic similarity search with configurable top-k
    - Batch add/delete operations
    - Automatic embedding generation via provider

    Example:
        >>> store = VectorStore(collection_name="courses_v1")
        >>> store.add_chunks(chunks)
        >>> hits = store.query("machine learning", top_k=5)
        >>> for hit in hits:
        ...     print(hit.score, hit.text[:50])
    """

    def __init__(
        self,
        collection_name: str,
        persist_directory: Optional[Path] = None,
        embedding_provider: Optional[EmbeddingProvider | str] = None,
        embedding_function: Optional[Any] = None,
    ):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for persistent storage
            embedding_provider: EmbeddingProvider instance or provider name string (uses default if None)
            embedding_function: Optional custom embedding function for ChromaDB
        """
        settings = get_settings()

        self.collection_name = collection_name
        self.persist_directory = persist_directory or settings.resolved_paths.index_dir

        # Initialize embedding provider
        if embedding_provider is None:
            self._embedding_provider = get_embedding_provider()
        elif isinstance(embedding_provider, str):
            # Convert string to provider instance
            self._embedding_provider = get_embedding_provider(embedding_provider)
        else:
            self._embedding_provider = embedding_provider

        # Ensure persist directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with persistence
        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Create or get collection
        # Note: We handle embeddings ourselves, so we don't use ChromaDB's embedding function
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )

        logger.info(
            f"Vector store initialized: collection={collection_name}, "
            f"persist_dir={self.persist_directory}, "
            f"existing_count={self._collection.count()}"
        )

    @property
    def count(self) -> int:
        """Get the number of items in the collection."""
        return self._collection.count()

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        """Get the embedding provider."""
        return self._embedding_provider

    def add_chunks(
        self,
        chunks: list[ChunkRecord],
        show_progress: bool = True,
    ) -> int:
        """
        Add chunks to the vector store.

        Args:
            chunks: List of ChunkRecords to add
            show_progress: Whether to show progress during embedding

        Returns:
            Number of chunks added
        """
        if not chunks:
            return 0

        # Extract data for ChromaDB
        ids = [chunk.chunk_id for chunk in chunks]
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.to_metadata_dict() for chunk in chunks]

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self._embedding_provider.embed_batch(texts, show_progress=show_progress)

        # Add to collection in batches (ChromaDB has limits)
        batch_size = 500
        added = 0

        for i in range(0, len(chunks), batch_size):
            batch_end = min(i + batch_size, len(chunks))

            self._collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end],
                documents=texts[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )
            added += batch_end - i

            logger.debug(f"Added batch {i//batch_size + 1}: {added}/{len(chunks)} chunks")

        logger.info(f"Added {added} chunks to vector store")
        return added

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        filters: Optional[dict[str, Any]] = None,
        min_score: Optional[float] = None,
    ) -> list[RetrievalHit]:
        """
        Query the vector store for similar chunks.

        Args:
            query_text: Query text to search for
            top_k: Maximum number of results to return
            filters: Optional metadata filters (e.g., {"department": "se"})
            min_score: Optional minimum similarity score threshold

        Returns:
            List of RetrievalHit objects sorted by score (highest first)
        """
        if not query_text or not query_text.strip():
            return []

        # Generate query embedding
        query_embedding = self._embedding_provider.embed_query(query_text)

        # Build where clause for filters
        where_clause = self._build_where_clause(filters) if filters else None

        # Query ChromaDB
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        # Convert to RetrievalHits
        hits = self._results_to_hits(results)

        # Filter by minimum score if specified
        if min_score is not None:
            hits = [hit for hit in hits if hit.score >= min_score]

        return hits

    def query_by_department(
        self,
        query_text: str,
        departments: list[str],
        top_k_per_dept: int = 5,
        min_score: Optional[float] = None,
    ) -> dict[str, list[RetrievalHit]]:
        """
        Query separately for each department.

        Useful for cross-department comparisons.

        Args:
            query_text: Query text
            departments: List of department IDs
            top_k_per_dept: Results per department
            min_score: Minimum similarity score

        Returns:
            Dictionary mapping department ID to list of hits
        """
        results = {}

        for dept in departments:
            hits = self.query(
                query_text=query_text,
                top_k=top_k_per_dept,
                filters={"department": dept},
                min_score=min_score,
            )
            results[dept] = hits

        return results

    def get_by_ids(self, chunk_ids: list[str]) -> list[RetrievalHit]:
        """
        Get chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs to retrieve

        Returns:
            List of RetrievalHit objects (score will be 1.0)
        """
        if not chunk_ids:
            return []

        results = self._collection.get(
            ids=chunk_ids,
            include=["documents", "metadatas"],
        )

        hits = []
        for i, chunk_id in enumerate(results["ids"]):
            hits.append(
                RetrievalHit(
                    chunk_id=chunk_id,
                    text=results["documents"][i] if results["documents"] else "",
                    score=1.0,  # Direct retrieval, no similarity score
                    metadata=results["metadatas"][i] if results["metadatas"] else {},
                )
            )

        return hits

    def delete_chunks(self, chunk_ids: list[str]) -> int:
        """
        Delete chunks by their IDs.

        Args:
            chunk_ids: List of chunk IDs to delete

        Returns:
            Number of chunks deleted
        """
        if not chunk_ids:
            return 0

        self._collection.delete(ids=chunk_ids)
        logger.info(f"Deleted {len(chunk_ids)} chunks from vector store")
        return len(chunk_ids)

    def clear(self) -> None:
        """Clear all chunks from the collection."""
        # Delete and recreate collection
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Cleared collection: {self.collection_name}")

    def _build_where_clause(self, filters: dict[str, Any]) -> dict[str, Any]:
        """Build ChromaDB where clause from filters."""
        conditions = []

        for key, value in filters.items():
            if value is None:
                continue

            if isinstance(value, list):
                # Multiple values: use $in
                if value:  # Only if non-empty
                    conditions.append({key: {"$in": value}})
            else:
                # Single value: direct equality
                conditions.append({key: value})

        if not conditions:
            return {}
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}

    def _results_to_hits(self, results: dict) -> list[RetrievalHit]:
        """Convert ChromaDB results to RetrievalHit objects."""
        hits = []

        if not results or not results.get("ids") or not results["ids"][0]:
            return hits

        ids = results["ids"][0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i, chunk_id in enumerate(ids):
            # ChromaDB returns distances, convert to similarity score
            # For cosine distance: similarity = 1 - distance
            distance = distances[i] if distances else 0.0
            score = 1.0 - distance

            hits.append(
                RetrievalHit(
                    chunk_id=chunk_id,
                    text=documents[i] if documents else "",
                    score=score,
                    metadata=metadatas[i] if metadatas else {},
                )
            )

        # Sort by score descending
        hits.sort(key=lambda h: h.score, reverse=True)

        return hits

    def get_all_metadata(self, limit: int = 10000) -> list[dict[str, Any]]:
        """
        Get metadata for all chunks (useful for quantitative queries).

        Args:
            limit: Maximum number of records to return

        Returns:
            List of metadata dictionaries
        """
        results = self._collection.get(
            limit=limit,
            include=["metadatas"],
        )

        return results.get("metadatas", [])

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the vector store."""
        metadata_list = self.get_all_metadata()

        # Count by department
        dept_counts: dict[str, int] = {}
        course_codes: set[str] = set()

        for meta in metadata_list:
            dept = meta.get("department", "unknown")
            dept_counts[dept] = dept_counts.get(dept, 0) + 1
            course_codes.add(meta.get("course_code", ""))

        return {
            "collection_name": self.collection_name,
            "total_chunks": self.count,
            "total_courses": len(course_codes),
            "chunks_by_department": dept_counts,
            "persist_directory": str(self.persist_directory),
            "embedding_provider": self._embedding_provider.provider_name,
            "embedding_model": self._embedding_provider.model_name,
            "embedding_dimensions": self._embedding_provider.dimensions,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory Function
# ─────────────────────────────────────────────────────────────────────────────


def create_vector_store(
    collection_name: Optional[str] = None,
    dataset_version: Optional[str] = None,
    persist_directory: Optional[Path] = None,
    embedding_provider: Optional[EmbeddingProvider] = None,
) -> VectorStore:
    """
    Create a vector store instance.

    Args:
        collection_name: Collection name (auto-generated if None)
        dataset_version: Dataset version for collection naming
        persist_directory: Storage directory
        embedding_provider: Embedding provider to use

    Returns:
        VectorStore instance
    """
    settings = get_settings()

    # Generate collection name if not provided
    if collection_name is None:
        provider = embedding_provider or get_embedding_provider()
        provider_name = provider.provider_name
        version_suffix = f"_{dataset_version}" if dataset_version else ""
        collection_name = f"courses_{provider_name}{version_suffix}"

    # Use versioned subdirectory if dataset_version provided
    if persist_directory is None:
        base_dir = settings.resolved_paths.index_dir
        if dataset_version:
            persist_directory = base_dir / dataset_version
        else:
            persist_directory = base_dir

    return VectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_provider=embedding_provider,
    )
