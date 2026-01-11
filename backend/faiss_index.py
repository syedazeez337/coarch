"""FAISS vector index management."""

import os
import faiss
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SearchResult:
    """A search result from the index."""
    id: int
    score: float
    file_path: str
    start_line: int
    end_line: int
    code: str
    language: str


class FaissIndex:
    """FAISS-based vector index for code embeddings."""

    def __init__(
        self,
        dim: int = 768,
        metric: str = "inner_product",
        index_path: Optional[str] = None
    ):
        """Initialize the FAISS index.

        Args:
            dim: Embedding dimension
            metric: Distance metric ("inner_product", "l2", "hamming")
            index_path: Path to load/save index
        """
        self.dim = dim
        self.metric = metric
        self.index_path = index_path

        if metric == "inner_product":
            self.index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        elif metric == "l2":
            self.index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_L2)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        self.id_to_metadata = {}  # id -> metadata dict
        self.next_id = 0

        if index_path and os.path.exists(index_path):
            self.load(index_path)

    def add(self, embeddings: np.ndarray, metadata: List[dict]) -> List[int]:
        """Add embeddings to the index.

        Args:
            embeddings: numpy array of shape (n, dim)
            metadata: List of metadata dicts for each embedding

        Returns:
            List of assigned IDs
        """
        ids = list(range(self.next_id, self.next_id + len(embeddings)))

        # Normalize for inner product (cosine similarity)
        if self.metric == "inner_product":
            faiss.normalize_L2(embeddings)

        self.index.add(embeddings.astype(np.float32))

        for id_, meta in zip(ids, metadata):
            self.id_to_metadata[id_] = meta

        self.next_id += len(embeddings)
        return ids

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        filter_ids: Optional[List[int]] = None
    ) -> List[SearchResult]:
        """Search for similar embeddings.

        Args:
            query: Query embedding (dim,)
            k: Number of results
            filter_ids: Optional list of IDs to restrict search to

        Returns:
            List of SearchResult objects
        """
        if self.metric == "inner_product":
            faiss.normalize_L2(query.reshape(1, -1))

        scores, ids = self.index.search(query.reshape(1, -1).astype(np.float32), k)

        results = []
        for score, id_ in zip(scores[0], ids[0]):
            if id_ < 0:  # FAISS returns -1 for padded results
                break
            if filter_ids and id_ not in filter_ids:
                continue

            meta = self.id_to_metadata.get(id_, {})
            results.append(SearchResult(
                id=id_,
                score=float(score),
                file_path=meta.get("file_path", ""),
                start_line=meta.get("start_line", 0),
                end_line=meta.get("end_line", 0),
                code=meta.get("code", ""),
                language=meta.get("language", "")
            ))

        return results

    def save(self, path: Optional[str] = None) -> str:
        """Save the index to disk.

        Args:
            path: Optional path, uses index_path if not provided

        Returns:
            Path to saved index
        """
        save_path = path or self.index_path
        if not save_path:
            raise ValueError("No index path specified")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, f"{save_path}.faiss")

        # Save metadata
        import json
        with open(f"{save_path}.meta", "w") as f:
            json.dump({
                "id_to_metadata": self.id_to_metadata,
                "next_id": self.next_id,
                "dim": self.dim,
                "metric": self.metric
            }, f)

        return save_path

    def load(self, path: str):
        """Load the index from disk.

        Args:
            path: Path to the index (without extension)
        """
        self.index = faiss.read_index(f"{path}.faiss")

        import json
        with open(f"{path}.meta", "r") as f:
            data = json.load(f)
            self.id_to_metadata = data["id_to_metadata"]
            self.next_id = data["next_id"]
            self.dim = data["dim"]
            self.metric = data["metric"]

    def count(self) -> int:
        """Return the number of embeddings in the index."""
        return self.index.ntotal

    def clear(self):
        """Clear the index."""
        self.index = faiss.IndexHNSWFlat(self.dim, 32,
            faiss.METRIC_INNER_PRODUCT if self.metric == "inner_product" else faiss.METRIC_L2)
        self.id_to_metadata = {}
        self.next_id = 0
