"""
SBERT Embeddings Module - Local embeddings via sentence-transformers.
====================================================================

Provides free, local embeddings using pre-trained SBERT models.
No API key required - runs entirely on local hardware.

Recommended models:
- all-MiniLM-L6-v2: Fast, 384 dimensions (default)
- all-mpnet-base-v2: Better quality, 768 dimensions
- multi-qa-MiniLM-L6-cos-v1: Optimized for Q&A
"""

from typing import Optional

from tqdm import tqdm

from iue_coursecompass.indexing.embeddings_base import EmbeddingProvider
from iue_coursecompass.shared.config import get_settings
from iue_coursecompass.shared.logging import get_logger

logger = get_logger(__name__)


# Model dimension mapping for common models
MODEL_DIMENSIONS = {
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "all-mpnet-base-v2": 768,
    "multi-qa-MiniLM-L6-cos-v1": 384,
    "multi-qa-mpnet-base-cos-v1": 768,
    "paraphrase-MiniLM-L6-v2": 384,
    "paraphrase-mpnet-base-v2": 768,
}


class SBERTEmbeddingProvider(EmbeddingProvider):
    """
    SBERT embedding provider using sentence-transformers.

    Features:
    - Free, local embeddings (no API key needed)
    - Multiple model options
    - Automatic device selection (CPU/GPU)
    - Efficient batch processing

    Example:
        >>> provider = SBERTEmbeddingProvider()
        >>> embedding = provider.embed_text("Hello world")
        >>> print(len(embedding))  # 384 for default model
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
    ):
        """
        Initialize the SBERT provider.

        Args:
            model_name: Model name from Hugging Face (default from config)
            device: Device to use ("cpu", "cuda", "auto")
            batch_size: Batch size for encoding
        """
        settings = get_settings()
        sbert_config = settings.embeddings.sbert

        self._model_name = model_name or sbert_config.model_name
        self._device = device or sbert_config.device
        self._batch_size = batch_size or sbert_config.batch_size

        # Get dimensions from mapping or config
        self._dimensions = MODEL_DIMENSIONS.get(
            self._model_name,
            sbert_config.dimensions,
        )

        # Lazy load the model
        self._model = None

        logger.debug(
            f"SBERT provider configured: model={self._model_name}, "
            f"device={self._device}, batch_size={self._batch_size}"
        )

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "sbert"

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        return self._dimensions

    @property
    def model(self):
        """Lazy load and return the sentence transformer model."""
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading SBERT model: {self._model_name}")

            # Determine device
            device = self._device
            if device == "auto":
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"

            self._model = SentenceTransformer(self._model_name, device=device)

            # Update dimensions from actual model
            self._dimensions = self._model.get_sentence_embedding_dimension()

            logger.info(
                f"SBERT model loaded: {self._model_name} "
                f"(dims={self._dimensions}, device={device})"
            )

        except ImportError:
            raise RuntimeError(
                "sentence-transformers is required for SBERT embeddings. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load SBERT model {self._model_name}: {e}")

    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return [0.0] * self._dimensions

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embedding.tolist()

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

        # Handle empty texts
        non_empty_indices = []
        non_empty_texts = []

        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(text)

        # Encode non-empty texts
        if non_empty_texts:
            embeddings = self.model.encode(
                non_empty_texts,
                batch_size=self._batch_size,
                convert_to_numpy=True,
                show_progress_bar=show_progress,
            )

            # Convert to list of lists
            embeddings_list = [emb.tolist() for emb in embeddings]
        else:
            embeddings_list = []

        # Build result with zero vectors for empty texts
        result = [[0.0] * self._dimensions for _ in range(len(texts))]

        for i, emb in zip(non_empty_indices, embeddings_list):
            result[i] = emb

        return result

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a query text.

        For SBERT, query embedding is the same as document embedding.

        Args:
            query: Query text to embed

        Returns:
            Embedding vector
        """
        return self.embed_text(query)

    def embed_with_progress(
        self,
        texts: list[str],
        desc: str = "Embedding",
    ) -> list[list[float]]:
        """
        Embed texts with a progress bar.

        Args:
            texts: List of texts to embed
            desc: Progress bar description

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Process in batches with progress bar
        embeddings = []

        for i in tqdm(range(0, len(texts), self._batch_size), desc=desc):
            batch = texts[i : i + self._batch_size]
            batch_embeddings = self.embed_batch(batch, show_progress=False)
            embeddings.extend(batch_embeddings)

        return embeddings

    def is_available(self) -> bool:
        """Check if SBERT is available."""
        try:
            from sentence_transformers import SentenceTransformer
            # Try loading a minimal test
            _ = SentenceTransformer(self._model_name)
            return True
        except Exception:
            return False

    def get_info(self) -> dict:
        """Get provider information."""
        info = super().get_info()

        # Add SBERT-specific info
        info["batch_size"] = self._batch_size
        info["device"] = self._device

        if self._model is not None:
            info["device_actual"] = str(self._model.device)

        return info
