"""
Indexing Module - Embeddings and vector storage.
================================================

This module handles embedding generation and vector database operations:

- embeddings_base: Abstract interface for embedding providers
- embeddings_sbert: SBERT (sentence-transformers) local embeddings
- embeddings_gemini: Gemini API embeddings
- vector_store: ChromaDB wrapper for vector storage and retrieval
- manifest: Embedding index versioning and metadata

Provider abstraction allows switching between SBERT and Gemini
without changing retrieval logic.
"""

from iue_coursecompass.indexing.embeddings_base import (
    EmbeddingProvider,
    get_embedding_provider,
)
from iue_coursecompass.indexing.embeddings_sbert import SBERTEmbeddingProvider
from iue_coursecompass.indexing.embeddings_gemini import GeminiEmbeddingProvider
from iue_coursecompass.indexing.vector_store import VectorStore, create_vector_store
from iue_coursecompass.indexing.manifest import (
    ManifestManager,
    create_manifest,
    load_manifest,
    get_latest_manifest,
)

__all__ = [
    # Base
    "EmbeddingProvider",
    "get_embedding_provider",
    # Providers
    "SBERTEmbeddingProvider",
    "GeminiEmbeddingProvider",
    # Vector Store
    "VectorStore",
    "create_vector_store",
    # Manifest
    "ManifestManager",
    "create_manifest",
    "load_manifest",
    "get_latest_manifest",
]
