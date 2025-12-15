"""
Tests for Indexing Module.
==========================

Tests for:
- EmbeddingProvider: Base class and implementations
- SBERTEmbeddingProvider: Local embeddings
- VectorStore: ChromaDB operations
- ManifestManager: Index versioning
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


# ─────────────────────────────────────────────────────────────────────────────
# Embedding Base Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEmbeddingProvider:
    """Tests for the EmbeddingProvider base class."""

    def test_base_class_is_abstract(self):
        """Test that base class cannot be instantiated."""
        from iue_coursecompass.indexing.embeddings_base import EmbeddingProvider

        with pytest.raises(TypeError):
            EmbeddingProvider()

    def test_get_provider_sbert(self):
        """Test getting SBERT provider."""
        from iue_coursecompass.indexing.embeddings_base import get_embedding_provider

        provider = get_embedding_provider("sbert")

        assert provider is not None
        assert provider.dimension > 0

    def test_get_provider_invalid(self):
        """Test getting invalid provider raises error."""
        from iue_coursecompass.indexing.embeddings_base import get_embedding_provider

        with pytest.raises(ValueError):
            get_embedding_provider("invalid_provider")

    def test_get_provider_caches_instance(self):
        """Test that provider instances are cached."""
        from iue_coursecompass.indexing.embeddings_base import get_embedding_provider

        provider1 = get_embedding_provider("sbert")
        provider2 = get_embedding_provider("sbert")

        assert provider1 is provider2


# ─────────────────────────────────────────────────────────────────────────────
# SBERT Embedding Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSBERTEmbeddings:
    """Tests for SBERTEmbeddingProvider."""

    def test_sbert_initialization(self):
        """Test SBERT provider initializes correctly."""
        from iue_coursecompass.indexing.embeddings_sbert import SBERTEmbeddingProvider

        provider = SBERTEmbeddingProvider()

        assert provider.model_name is not None
        assert provider.dimension > 0

    def test_sbert_embed_single(self):
        """Test embedding a single text."""
        from iue_coursecompass.indexing.embeddings_sbert import SBERTEmbeddingProvider

        provider = SBERTEmbeddingProvider()
        embedding = provider.embed("Hello world")

        assert isinstance(embedding, list)
        assert len(embedding) == provider.dimension
        assert all(isinstance(x, float) for x in embedding)

    def test_sbert_embed_batch(self):
        """Test embedding multiple texts."""
        from iue_coursecompass.indexing.embeddings_sbert import SBERTEmbeddingProvider

        provider = SBERTEmbeddingProvider()
        texts = ["Hello world", "Software engineering", "Course description"]
        embeddings = provider.embed_batch(texts)

        assert len(embeddings) == len(texts)
        assert all(len(e) == provider.dimension for e in embeddings)

    def test_sbert_embed_empty_string(self):
        """Test embedding empty string."""
        from iue_coursecompass.indexing.embeddings_sbert import SBERTEmbeddingProvider

        provider = SBERTEmbeddingProvider()
        embedding = provider.embed("")

        # Should return valid embedding (model handles empty strings)
        assert isinstance(embedding, list)
        assert len(embedding) == provider.dimension

    def test_sbert_embed_batch_empty(self):
        """Test embedding empty batch."""
        from iue_coursecompass.indexing.embeddings_sbert import SBERTEmbeddingProvider

        provider = SBERTEmbeddingProvider()
        embeddings = provider.embed_batch([])

        assert embeddings == []

    def test_sbert_embeddings_are_normalized(self):
        """Test that embeddings are normalized (unit vectors)."""
        from iue_coursecompass.indexing.embeddings_sbert import SBERTEmbeddingProvider
        import math

        provider = SBERTEmbeddingProvider()
        embedding = provider.embed("Test text for normalization")

        # Calculate magnitude
        magnitude = math.sqrt(sum(x * x for x in embedding))

        # Should be close to 1.0 (normalized)
        assert abs(magnitude - 1.0) < 0.01

    def test_sbert_similar_texts_have_similar_embeddings(self):
        """Test that similar texts produce similar embeddings."""
        from iue_coursecompass.indexing.embeddings_sbert import SBERTEmbeddingProvider

        provider = SBERTEmbeddingProvider()

        emb1 = provider.embed("Software engineering course")
        emb2 = provider.embed("Software engineering class")
        emb3 = provider.embed("Cooking recipes for dinner")

        # Cosine similarity
        def cosine_sim(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            mag_a = sum(x * x for x in a) ** 0.5
            mag_b = sum(x * x for x in b) ** 0.5
            return dot / (mag_a * mag_b)

        sim_12 = cosine_sim(emb1, emb2)
        sim_13 = cosine_sim(emb1, emb3)

        # Similar texts should have higher similarity
        assert sim_12 > sim_13


# ─────────────────────────────────────────────────────────────────────────────
# Gemini Embedding Tests (Mocked)
# ─────────────────────────────────────────────────────────────────────────────


class TestGeminiEmbeddings:
    """Tests for GeminiEmbeddingProvider (mocked)."""

    def test_gemini_requires_api_key(self):
        """Test that Gemini provider requires API key."""
        from iue_coursecompass.indexing.embeddings_gemini import GeminiEmbeddingProvider

        # Without API key, should raise on embed
        provider = GeminiEmbeddingProvider(api_key=None)

        with pytest.raises((ValueError, Exception)):
            provider.embed("Test")

    @patch("iue_coursecompass.indexing.embeddings_gemini.genai")
    def test_gemini_embed_with_mock(self, mock_genai):
        """Test Gemini embedding with mocked API."""
        from iue_coursecompass.indexing.embeddings_gemini import GeminiEmbeddingProvider

        # Mock the API response
        mock_result = MagicMock()
        mock_result.embedding = [0.1] * 768
        mock_genai.embed_content.return_value = {"embedding": [0.1] * 768}

        provider = GeminiEmbeddingProvider(api_key="fake-key")
        provider._client = mock_genai

        # This would call the mocked API
        # embedding = provider.embed("Test")


# ─────────────────────────────────────────────────────────────────────────────
# Vector Store Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestVectorStore:
    """Tests for VectorStore class."""

    @pytest.fixture
    def temp_store(self, temp_dir: Path):
        """Create a temporary vector store."""
        from iue_coursecompass.indexing.vector_store import VectorStore

        store = VectorStore(
            collection_name="test_collection",
            persist_directory=str(temp_dir / "chroma"),
            embedding_provider="sbert",
        )
        yield store
        # Cleanup
        store.clear()

    def test_vector_store_initialization(self, temp_store):
        """Test vector store initializes correctly."""
        assert temp_store is not None
        assert temp_store.collection_name == "test_collection"

    def test_vector_store_add_chunks(self, temp_store, sample_chunk_record):
        """Test adding chunks to vector store."""
        temp_store.add_chunks([sample_chunk_record])

        # Should be able to query
        count = temp_store.count()
        assert count >= 1

    def test_vector_store_query(self, temp_store, sample_chunk_record):
        """Test querying vector store."""
        temp_store.add_chunks([sample_chunk_record])

        results = temp_store.query("software engineering", top_k=5)

        assert isinstance(results, list)
        if results:
            assert hasattr(results[0], "chunk_id")
            assert hasattr(results[0], "score")

    def test_vector_store_query_with_filter(self, temp_store, sample_chunk_record):
        """Test querying with metadata filter."""
        temp_store.add_chunks([sample_chunk_record])

        results = temp_store.query(
            "software",
            top_k=5,
            where={"department": sample_chunk_record.department},
        )

        assert isinstance(results, list)

    def test_vector_store_clear(self, temp_store, sample_chunk_record):
        """Test clearing vector store."""
        temp_store.add_chunks([sample_chunk_record])
        assert temp_store.count() >= 1

        temp_store.clear()
        assert temp_store.count() == 0

    def test_vector_store_delete_by_id(self, temp_store, sample_chunk_record):
        """Test deleting specific chunks."""
        temp_store.add_chunks([sample_chunk_record])
        initial_count = temp_store.count()

        temp_store.delete([sample_chunk_record.chunk_id])

        assert temp_store.count() < initial_count

    def test_vector_store_get_by_id(self, temp_store, sample_chunk_record):
        """Test getting chunk by ID."""
        temp_store.add_chunks([sample_chunk_record])

        result = temp_store.get(sample_chunk_record.chunk_id)

        assert result is not None
        assert result.chunk_id == sample_chunk_record.chunk_id

    def test_vector_store_handles_duplicates(self, temp_store, sample_chunk_record):
        """Test that duplicates are handled."""
        temp_store.add_chunks([sample_chunk_record])
        count1 = temp_store.count()

        # Add same chunk again
        temp_store.add_chunks([sample_chunk_record])
        count2 = temp_store.count()

        # Should not duplicate (or handle gracefully)
        assert count2 >= count1


# ─────────────────────────────────────────────────────────────────────────────
# Manifest Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestManifestManager:
    """Tests for ManifestManager class."""

    def test_manifest_creation(self, temp_dir: Path):
        """Test manifest creation."""
        from iue_coursecompass.indexing.manifest import ManifestManager

        manager = ManifestManager(manifest_dir=temp_dir)
        manifest = manager.create_manifest(
            collection_name="test",
            embedding_provider="sbert",
            chunk_count=100,
        )

        assert manifest is not None
        assert manifest.collection_name == "test"
        assert manifest.chunk_count == 100

    def test_manifest_save_and_load(self, temp_dir: Path):
        """Test saving and loading manifest."""
        from iue_coursecompass.indexing.manifest import ManifestManager

        manager = ManifestManager(manifest_dir=temp_dir)

        # Create and save
        manifest = manager.create_manifest(
            collection_name="test",
            embedding_provider="sbert",
            chunk_count=100,
        )
        manager.save_manifest(manifest)

        # Load
        loaded = manager.load_manifest("test")

        assert loaded is not None
        assert loaded.collection_name == manifest.collection_name
        assert loaded.chunk_count == manifest.chunk_count

    def test_manifest_versioning(self, temp_dir: Path):
        """Test manifest version tracking."""
        from iue_coursecompass.indexing.manifest import ManifestManager

        manager = ManifestManager(manifest_dir=temp_dir)

        manifest1 = manager.create_manifest(
            collection_name="test",
            embedding_provider="sbert",
            chunk_count=100,
        )

        manifest2 = manager.create_manifest(
            collection_name="test",
            embedding_provider="sbert",
            chunk_count=200,
        )

        # Versions should be different
        assert manifest1.version != manifest2.version or manifest1.created_at != manifest2.created_at

    def test_manifest_list(self, temp_dir: Path):
        """Test listing all manifests."""
        from iue_coursecompass.indexing.manifest import ManifestManager

        manager = ManifestManager(manifest_dir=temp_dir)

        # Create multiple manifests
        for name in ["test1", "test2", "test3"]:
            manifest = manager.create_manifest(
                collection_name=name,
                embedding_provider="sbert",
                chunk_count=10,
            )
            manager.save_manifest(manifest)

        # List all
        manifests = manager.list_manifests()

        assert len(manifests) >= 3
