"""LRU cache for code embeddings to avoid recomputation."""

import hashlib
import os
import pickle
import threading
from pathlib import Path
from typing import Optional
import numpy as np

from .logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingCache:
    """Disk-backed LRU cache for embeddings."""

    def __init__(self, cache_dir: str = "~/.coarch/embedding_cache", max_size_mb: int = 500, model_name: str = "default"):
        self.cache_dir = Path(os.path.expanduser(cache_dir))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.index_file = self.cache_dir / "index.pkl"
        self.model_name = model_name
        self._lock = threading.Lock()
        self._load_index()

    def _load_index(self):
        """Load cache index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "rb") as f:
                    self.index = pickle.load(f)
            except Exception:
                self.index = {}
        else:
            self.index = {}

    def _save_index(self):
        """Save cache index to disk."""
        with open(self.index_file, "wb") as f:
            pickle.dump(self.index, f)

    def _hash_code(self, code: str) -> str:
        """Generate hash for code snippet."""
        key_input = f"{self.model_name}:{code}"
        return hashlib.sha256(key_input.encode()).hexdigest()[:32]

    def get(self, code: str) -> Optional[np.ndarray]:
        """Get cached embedding if exists."""
        key = self._hash_code(code)
        with self._lock:
            if key in self.index:
                cache_file = self.cache_dir / f"{key}.npy"
                if cache_file.exists():
                    try:
                        return np.load(cache_file)
                    except Exception:
                        del self.index[key]
                        self._save_index()
        return None

    def put(self, code: str, embedding: np.ndarray):
        """Cache an embedding."""
        key = self._hash_code(code)
        cache_file = self.cache_dir / f"{key}.npy"
        np.save(cache_file, embedding)
        with self._lock:
            self.index[key] = cache_file.stat().st_size
            self._save_index()
            self._evict_if_needed()

    def _evict_if_needed(self):
        """Evict old entries if cache exceeds max size.

        Note: This method assumes the caller holds self._lock.
        """
        total_size = sum(self.index.values())
        if total_size > self.max_size_bytes:
            # Remove oldest 20% of entries
            to_remove = list(self.index.keys())[:len(self.index) // 5]
            for key in to_remove:
                cache_file = self.cache_dir / f"{key}.npy"
                cache_file.unlink(missing_ok=True)
                del self.index[key]
            self._save_index()

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "entries": len(self.index),
            "size_mb": sum(self.index.values()) / 1024 / 1024,
            "max_size_mb": self.max_size_bytes / 1024 / 1024,
        }
