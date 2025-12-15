"""
Embeddings Base Module - Abstract interface for embedding providers.
===================================================================

Defines the abstract base class for embedding providers, enabling
provider-agnostic embedding operations. Switching between SBERT
and Gemini doesn't require changes to retrieval logic.
"""

from abc import ABC, abstractmethod
from typing import Optional

from iue_coursecompass.shared.config import get_settings
from iue_coursecompass.shared.logging import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Base Class
# ─────────────────────────────────────────────────────────────────────────────


class EmbeddingProvider(ABC):
    """
    Abstract base class for embedding providers.

    Implementations must provide:
    - embed_text(): Embed a single text string
    - embed_batch(): Embed multiple texts efficiently

    Properties:
    - model_name: Name of the embedding model
    - dimensions: Embedding vector dimensions
    - provider_name: Provider identifier (sbert, gemini)
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name identifier."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name being used."""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Get the embedding vector dimensions."""
        pass

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        pass

    @abstractmethod
    def embed_batch(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> list[list[float]]:
        """
        Embed multiple texts efficiently.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar

        Returns:
            List of embedding vectors
        """
        pass

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a query text.

        Some providers may use different embeddings for queries vs documents.
        Default implementation just calls embed_text().

        Args:
            query: Query text to embed

        Returns:
            Embedding vector
        """
        return self.embed_text(query)

    def is_available(self) -> bool:
        """
        Check if the provider is available and properly configured.

        Returns:
            True if provider can be used
        """
        try:
            # Try a simple embedding to verify
            test_embedding = self.embed_text("test")
            return len(test_embedding) == self.dimensions
        except Exception as e:
            logger.warning(f"Provider {self.provider_name} not available: {e}")
            return False

    def get_info(self) -> dict:
        """
        Get provider information.

        Returns:
            Dictionary with provider details
        """
        return {
            "provider": self.provider_name,
            "model": self.model_name,
            "dimensions": self.dimensions,
            "available": self.is_available(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Provider Factory
# ─────────────────────────────────────────────────────────────────────────────


_provider_cache: dict[str, EmbeddingProvider] = {}


def get_embedding_provider(
    provider_name: Optional[str] = None,
    use_cache: bool = True,
) -> EmbeddingProvider:
    """
    Get an embedding provider instance.

    Factory function that returns the appropriate provider based on
    configuration or explicit name.

    Args:
        provider_name: Provider name ("sbert" or "gemini"). If None, uses config.
        use_cache: Whether to cache and reuse provider instances

    Returns:
        EmbeddingProvider instance

    Raises:
        ValueError: If provider name is invalid
        RuntimeError: If provider cannot be initialized

    Example:
        >>> provider = get_embedding_provider()  # Uses config default
        >>> provider = get_embedding_provider("sbert")  # Explicit SBERT
        >>> embeddings = provider.embed_batch(["text1", "text2"])
    """
    # Get provider name from config if not specified
    if provider_name is None:
        settings = get_settings()
        provider_name = settings.get_effective_embedding_provider()

    provider_name = provider_name.lower().strip()

    # Check cache
    if use_cache and provider_name in _provider_cache:
        return _provider_cache[provider_name]

    # Create provider
    provider: EmbeddingProvider

    if provider_name == "sbert":
        from iue_coursecompass.indexing.embeddings_sbert import SBERTEmbeddingProvider
        provider = SBERTEmbeddingProvider()

    elif provider_name == "gemini":
        from iue_coursecompass.indexing.embeddings_gemini import GeminiEmbeddingProvider
        provider = GeminiEmbeddingProvider()

    else:
        raise ValueError(
            f"Unknown embedding provider: {provider_name}. "
            f"Valid options: sbert, gemini"
        )

    # Cache if requested
    if use_cache:
        _provider_cache[provider_name] = provider

    logger.info(
        f"Initialized embedding provider: {provider.provider_name} "
        f"(model={provider.model_name}, dims={provider.dimensions})"
    )

    return provider


def clear_provider_cache() -> None:
    """Clear the provider cache."""
    _provider_cache.clear()


def list_available_providers() -> list[dict]:
    """
    List all available embedding providers.

    Returns:
        List of provider info dictionaries
    """
    providers = []

    # Try SBERT
    try:
        from iue_coursecompass.indexing.embeddings_sbert import SBERTEmbeddingProvider
        sbert = SBERTEmbeddingProvider()
        providers.append(sbert.get_info())
    except Exception as e:
        providers.append({
            "provider": "sbert",
            "available": False,
            "error": str(e),
        })

    # Try Gemini
    try:
        from iue_coursecompass.indexing.embeddings_gemini import GeminiEmbeddingProvider
        gemini = GeminiEmbeddingProvider()
        providers.append(gemini.get_info())
    except Exception as e:
        providers.append({
            "provider": "gemini",
            "available": False,
            "error": str(e),
        })

    return providers
