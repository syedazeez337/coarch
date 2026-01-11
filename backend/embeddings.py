"""Code embedding service using CodeBERT with production-ready features."""

from typing import Dict, List, Optional
import numpy as np

from transformers import AutoTokenizer, AutoModel
import torch

from .logging_config import get_logger

logger = get_logger(__name__)


class CodeEmbedder:
    """Code embedding using CodeBERT with proper error handling."""

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        """Initialize the CodeBERT model.

        Args:
            model_name: HuggingFace model name
            device: "cuda", "cpu", or None for auto-detect
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length

        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
            logger.info("CUDA available, using GPU for embeddings")
        else:
            self.device = "cpu"
            logger.info("Using CPU for embeddings")

        self._init_model()

    def _init_model(self):
        """Initialize the model and tokenizer."""
        try:
            logger.info(f"Loading tokenizer: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            logger.info(f"Loading model: {self.model_name}")
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

            logger.info(
                f"Model loaded on {self.device}, "
                f"hidden_size={self.model.config.hidden_size}"
            )
        except Exception as e:
            logger.exception(f"Failed to load model: {e}")
            raise

    def embed(
        self,
        code_snippets: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Generate embeddings for code snippets.

        Args:
            code_snippets: List of code strings
            batch_size: Batch size for processing
            show_progress: Show progress bar

        Returns:
            numpy array of shape (len(code_snippets), dim)
        """
        if not code_snippets:
            logger.warning("Empty code snippets provided to embed()")
            return np.array([])

        total = len(code_snippets)
        logger.info(f"Generating embeddings for {total} snippets")

        embeddings: List[np.ndarray] = []

        for i in range(0, total, batch_size):
            batch = code_snippets[i : i + batch_size]

            try:
                batch_embeddings = self._embed_batch(batch)
                embeddings.append(batch_embeddings)
            except Exception as e:
                logger.error(f"Failed to embed batch starting at {i}: {e}")
                raise

        result = np.vstack(embeddings)
        logger.info(f"Generated embeddings with shape: {result.shape}")

        return result

    def _embed_batch(self, batch: List[str]) -> np.ndarray:
        """Embed a single batch."""
        inputs = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return batch_embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a search query.

        Args:
            query: Query string

        Returns:
            numpy array of shape (dim,)
        """
        if not query or not query.strip():
            logger.warning("Empty query provided to embed_query()")
            return np.zeros(self.get_dimension(), dtype=np.float32)

        embedding = self.embed([query.strip()])
        return embedding[0]

    def embed_file(self, file_path: str) -> np.ndarray:
        """Generate embedding for a file.

        Args:
            file_path: Path to file

        Returns:
            numpy array of shape (dim,)
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                code = f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return np.zeros(self.get_dimension(), dtype=np.float32)

        return self.embed([code])[0]

    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self.model.config.hidden_size

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage information."""
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()

        result: Dict[str, float] = {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
        }

        if torch.cuda.is_available():
            result["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            result["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024

        return result

    def __repr__(self) -> str:
        return (
            f"CodeEmbedder(model={self.model_name}, "
            f"device={self.device}, dim={self.get_dimension()})"
        )
