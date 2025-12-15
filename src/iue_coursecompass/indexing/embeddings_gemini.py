"""
Gemini Embeddings Module - Google GenAI embeddings API.
=======================================================

Provides high-quality embeddings using Google's Gemini API.
Requires a GEMINI_API_KEY from Google AI Studio.

Available models:
- text-embedding-004: Latest model, 768 dimensions (recommended)
- embedding-001: Legacy model
"""

import time
from typing import Optional

from tqdm import tqdm

from iue_coursecompass.indexing.embeddings_base import EmbeddingProvider
from iue_coursecompass.shared.config import get_settings
from iue_coursecompass.shared.logging import get_logger

logger = get_logger(__name__)


# Model dimension mapping
GEMINI_MODEL_DIMENSIONS = {
    "text-embedding-004": 768,
    "embedding-001": 768,
}

# Task types for embeddings
TASK_TYPES = {
    "RETRIEVAL_DOCUMENT": "RETRIEVAL_DOCUMENT",
    "RETRIEVAL_QUERY": "RETRIEVAL_QUERY",
    "SEMANTIC_SIMILARITY": "SEMANTIC_SIMILARITY",
    "CLASSIFICATION": "CLASSIFICATION",
    "CLUSTERING": "CLUSTERING",
}


class GeminiEmbeddingProvider(EmbeddingProvider):
    """
    Gemini embedding provider using Google GenAI SDK.

    Features:
    - High-quality embeddings from Google
    - Task-specific embeddings (document vs query)
    - Batch processing with rate limiting

    Requires:
    - GEMINI_API_KEY environment variable

    Example:
        >>> provider = GeminiEmbeddingProvider()
        >>> embedding = provider.embed_text("Hello world")
        >>> print(len(embedding))  # 768
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: Optional[int] = None,
        task_type: Optional[str] = None,
    ):
        """
        Initialize the Gemini provider.

        Args:
            model_name: Embedding model name
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            batch_size: Batch size for API calls
            task_type: Task type for embeddings
        """
        settings = get_settings()
        gemini_config = settings.embeddings.gemini

        self._model_name = model_name or gemini_config.model_name
        self._api_key = api_key or settings.gemini_api_key
        self._batch_size = batch_size or gemini_config.batch_size
        self._task_type = task_type or gemini_config.task_type

        # Get dimensions from mapping
        self._dimensions = GEMINI_MODEL_DIMENSIONS.get(
            self._model_name,
            gemini_config.dimensions,
        )

        # Validate API key
        if not self._api_key:
            raise ValueError(
                "Gemini API key is required. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize client lazily
        self._client = None

        logger.debug(
            f"Gemini provider configured: model={self._model_name}, "
            f"batch_size={self._batch_size}, task_type={self._task_type}"
        )

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "gemini"

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        return self._dimensions

    @property
    def client(self):
        """Lazy load and return the Gemini client."""
        if self._client is None:
            self._initialize_client()
        return self._client

    def _initialize_client(self) -> None:
        """Initialize the Google GenAI client."""
        try:
            from google import genai

            self._client = genai.Client(api_key=self._api_key)
            logger.info(f"Gemini client initialized for model: {self._model_name}")

        except ImportError:
            raise RuntimeError(
                "google-genai is required for Gemini embeddings. "
                "Install with: pip install google-genai"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini client: {e}")

    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            return [0.0] * self._dimensions

        try:
            result = self.client.models.embed_content(
                model=self._model_name,
                contents=text,
                config={
                    "task_type": self._task_type,
                },
            )

            # Extract embedding from response
            embedding = result.embeddings[0].values
            return list(embedding)

        except Exception as e:
            logger.error(f"Gemini embedding failed: {e}")
            raise RuntimeError(f"Failed to generate embedding: {e}")

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
        if not texts:
            return []

        embeddings = []
        
        # Process in batches to respect API limits
        iterator = range(0, len(texts), self._batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding (Gemini)")

        for i in iterator:
            batch = texts[i : i + self._batch_size]
            batch_embeddings = self._embed_batch_internal(batch)
            embeddings.extend(batch_embeddings)

            # Rate limiting - small delay between batches
            if i + self._batch_size < len(texts):
                time.sleep(0.1)

        return embeddings

    def _embed_batch_internal(self, texts: list[str]) -> list[list[float]]:
        """Embed a single batch of texts."""
        # Filter empty texts and track indices
        non_empty_indices = []
        non_empty_texts = []

        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(text)

        # Initialize result with zero vectors
        result = [[0.0] * self._dimensions for _ in range(len(texts))]

        if not non_empty_texts:
            return result

        try:
            # Embed all non-empty texts at once
            response = self.client.models.embed_content(
                model=self._model_name,
                contents=non_empty_texts,
                config={
                    "task_type": self._task_type,
                },
            )

            # Extract embeddings
            for idx, (orig_idx, embedding) in enumerate(
                zip(non_empty_indices, response.embeddings)
            ):
                result[orig_idx] = list(embedding.values)

        except Exception as e:
            logger.error(f"Gemini batch embedding failed: {e}")
            # Fall back to individual embedding
            for orig_idx, text in zip(non_empty_indices, non_empty_texts):
                try:
                    result[orig_idx] = self.embed_text(text)
                except Exception:
                    # Keep zero vector on failure
                    pass

        return result

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a query text using RETRIEVAL_QUERY task type.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector
        """
        if not query or not query.strip():
            return [0.0] * self._dimensions

        try:
            result = self.client.models.embed_content(
                model=self._model_name,
                contents=query,
                config={
                    "task_type": "RETRIEVAL_QUERY",
                },
            )

            embedding = result.embeddings[0].values
            return list(embedding)

        except Exception as e:
            logger.error(f"Gemini query embedding failed: {e}")
            # Fall back to regular embedding
            return self.embed_text(query)

    def embed_with_progress(
        self,
        texts: list[str],
        desc: str = "Embedding (Gemini)",
    ) -> list[list[float]]:
        """
        Embed texts with a progress bar.

        Args:
            texts: List of texts to embed
            desc: Progress bar description

        Returns:
            List of embedding vectors
        """
        return self.embed_batch(texts, show_progress=True)

    def is_available(self) -> bool:
        """Check if Gemini is available and properly configured."""
        if not self._api_key:
            return False

        try:
            # Try a simple embedding
            _ = self.embed_text("test")
            return True
        except Exception as e:
            logger.warning(f"Gemini provider not available: {e}")
            return False

    def get_info(self) -> dict:
        """Get provider information."""
        info = super().get_info()

        # Add Gemini-specific info
        info["batch_size"] = self._batch_size
        info["task_type"] = self._task_type
        info["api_key_set"] = bool(self._api_key)

        return info
