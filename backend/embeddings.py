"""Code embedding service using CodeBERT."""

import os
from typing import List, Optional
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch


class CodeEmbedder:
    """Code embedding using CodeBERT."""

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        device: Optional[str] = None,
        max_length: int = 512
    ):
        """Initialize the CodeBERT model.

        Args:
            model_name: HuggingFace model name
            device: "cuda", "cpu", or None for auto-detect
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def embed(self, code_snippets: List[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for code snippets.

        Args:
            code_snippets: List of code strings
            batch_size: Batch size for processing

        Returns:
            numpy array of shape (len(code_snippets), dim)
        """
        embeddings = []

        for i in range(0, len(code_snippets), batch_size):
            batch = code_snippets[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            embeddings.append(batch_embeddings)

        return np.vstack(embeddings)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for a search query.

        Args:
            query: Query string

        Returns:
            numpy array of shape (dim,)
        """
        return self.embed([query])[0]

    def embed_file(self, file_path: str) -> np.ndarray:
        """Generate embedding for a file.

        Args:
            file_path: Path to file

        Returns:
            numpy array of shape (dim,)
        """
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()

        return self.embed([code])[0]

    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self.model.config.hidden_size

    def __repr__(self):
        return f"CodeEmbedder(model={self.model_name}, device={self.device})"
