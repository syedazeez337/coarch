"""FAISS vector index management with production-ready features."""

import os
import json
import shutil
import tempfile
import threading
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import faiss
import numpy as np

from .logging_config import get_logger

try:
    from .progress_tracker import (
        track_progress,
        create_faiss_progress_bar,
        should_show_progress,
        ProgressCallback,
        ETAEstimator,
    )
except ImportError:
    # Fallback if progress tracker is not available
    def track_progress(operation_name, total_items=None, unit="items"):
        class DummyProgress:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def update(self, n=1):
                pass
            def set_description(self, desc):
                pass
            def set_postfix(self, **kwargs):
                pass
        return DummyProgress()

    def create_faiss_progress_bar(total_vectors):
        class DummyBar:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def update(self, n=1):
                pass
            def set_description(self, desc):
                pass
            def set_postfix(self, **kwargs):
                pass
        return DummyBar()

    def should_show_progress(total_items=None):
        return total_items is not None and total_items > 100

    class ProgressCallback:
        def __init__(self, progress_bar, operation_name="Processing"):
            self.progress_bar = progress_bar
            self.operation_name = operation_name
        def __call__(self, chunks_processed, total_chunks=None):
            pass
        def set_description(self, description):
            if hasattr(self.progress_bar, 'set_description'):
                self.progress_bar.set_description(description)
        def set_postfix(self, **kwargs):
            if hasattr(self.progress_bar, 'set_postfix'):
                self.progress_bar.set_postfix(kwargs)
        def close(self):
            if hasattr(self.progress_bar, 'close'):
                self.progress_bar.close()

    class ETAEstimator:
        def __init__(self, total_items):
            self.total_items = total_items
        def update(self, processed_items):
            pass
        def get_eta(self):
            return None
        def get_eta_string(self):
            return "N/A"
        def get_rate_string(self):
            return "0 items/s"

logger = get_logger(__name__)

DEFAULT_DIM = 768
DEFAULT_M = 32
DEFAULT_EF_CONSTRUCTION = 200


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
    """FAISS-based vector index for code embeddings with thread safety."""

    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        metric: str = "inner_product",
        index_path: Optional[str] = None,
        use_ivf: bool = False,
        nlist: int = 100,
    ):
        """Initialize the FAISS index."""
        self.dim = dim
        self.metric = metric
        self.index_path = index_path
        self._lock_dict: Dict[str, threading.Lock] = {}
        self._is_trained = True

        if metric == "inner_product":
            metric_type = faiss.METRIC_INNER_PRODUCT
        elif metric == "l2":
            metric_type = faiss.METRIC_L2
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        if use_ivf and dim <= 1024:
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, metric_type)
            self._is_trained = False
            logger.info(f"Using IVF index with {nlist} partitions")
        else:
            self.index = faiss.IndexHNSWFlat(dim, DEFAULT_M, metric_type)
            logger.info("Using HNSW index")

        self.id_to_metadata: Dict[int, Dict] = {}
        self.next_id = 0

        if index_path and os.path.exists(f"{index_path}.faiss"):
            self.load(index_path)

    def _get_lock(self, key: str) -> threading.Lock:
        """Get a lock for the given key."""
        if key not in self._lock_dict:
            self._lock_dict[key] = threading.Lock()
        return self._lock_dict[key]

    def add(self, embeddings: np.ndarray, metadata: List[Dict]) -> List[int]:
        """Add embeddings to the index."""
        with self._get_lock("add"):
            if len(embeddings) == 0:
                logger.warning("Empty embeddings provided to add()")
                return []

            ids = list(range(self.next_id, self.next_id + len(embeddings)))

            embeddings_copy = embeddings.copy()

            if self.metric == "inner_product":
                faiss.normalize_L2(embeddings_copy)

            # Add progress tracking for large batches
            total_to_add = len(embeddings)
            show_progress = should_show_progress(total_to_add)
            
            if show_progress:
                with track_progress("Adding to FAISS index", total_to_add, "vectors") as progress:
                    if progress:
                        progress.set_description("Adding vectors to index")
                    
                    try:
                        self.index.add(embeddings_copy.astype(np.float32))
                        if progress:
                            progress.update(total_to_add)
                    except Exception as e:
                        logger.exception(f"Failed to add embeddings: {e}")
                        raise
            else:
                try:
                    self.index.add(embeddings_copy.astype(np.float32))
                except Exception as e:
                    logger.exception(f"Failed to add embeddings: {e}")
                    raise

            for id_, meta in zip(ids, metadata):
                self.id_to_metadata[id_] = meta

            self.next_id += len(embeddings)

            logger.info(f"Added {len(ids)} embeddings, total: {self.count()}")

            return ids

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        filter_ids: Optional[List[int]] = None,
        nprobe: int = 10,
    ) -> List[SearchResult]:
        """Search for similar embeddings."""
        with self._get_lock("search"):
            if self.count() == 0:
                logger.warning("Search on empty index")
                return []

            if self.metric == "inner_product":
                query_copy = query.copy().reshape(1, -1)
                faiss.normalize_L2(query_copy)
            else:
                query_copy = query.reshape(1, -1).astype(np.float32)

            if hasattr(self.index, "nprobe"):
                self.index.nprobe = nprobe

            try:
                scores, ids = self.index.search(query_copy, k)
            except Exception as e:
                logger.exception(f"Search failed: {e}")
                raise

            results: List[SearchResult] = []

            for score, id_ in zip(scores[0], ids[0]):
                if id_ < 0:
                    break

                if filter_ids and id_ not in filter_ids:
                    continue

                meta = self.id_to_metadata.get(id_, {})

                results.append(
                    SearchResult(
                        id=id_,
                        score=float(score),
                        file_path=meta.get("file_path", ""),
                        start_line=meta.get("start_line", 0),
                        end_line=meta.get("end_line", 0),
                        code=meta.get("code", ""),
                        language=meta.get("language", ""),
                    )
                )

            return results

    def train(self, embeddings: np.ndarray) -> None:
        """Train the index (required for IVF)."""
        with self._get_lock("train"):
            if not hasattr(self.index, "train"):
                logger.debug("Index does not require training")
                return

            logger.info(f"Training index on {len(embeddings)} embeddings")

            # Add progress tracking for training
            show_progress = should_show_progress(len(embeddings))
            
            if show_progress:
                with track_progress("Training FAISS index", len(embeddings), "samples") as progress:
                    if progress:
                        progress.set_description("Training index")
                    
                    try:
                        self.index.train(embeddings.astype(np.float32))
                        if progress:
                            progress.update(len(embeddings))
                        
                        self._is_trained = True
                        logger.info("Index training complete")
                    except Exception as e:
                        logger.exception(f"Training failed: {e}")
                        raise
            else:
                try:
                    self.index.train(embeddings.astype(np.float32))
                    self._is_trained = True
                    logger.info("Index training complete")
                except Exception as e:
                    logger.exception(f"Training failed: {e}")
                    raise

    def save(self, path: Optional[str] = None) -> str:
        """Save the index to disk atomically.

        Uses temporary files and atomic rename to prevent corruption
        if the process crashes during save.
        """
        with self._get_lock("save"):
            save_path = path or self.index_path

            if not save_path:
                raise ValueError("No index path specified")

            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

            temp_dir = tempfile.mkdtemp(prefix="coarch_save_")

            try:
                temp_faiss_path = os.path.join(temp_dir, "index.faiss")
                temp_meta_path = os.path.join(temp_dir, "index.meta")

                faiss.write_index(self.index, temp_faiss_path)

                metadata = {
                    "id_to_metadata": self.id_to_metadata,
                    "next_id": self.next_id,
                    "dim": self.dim,
                    "metric": self.metric,
                    "_is_trained": self._is_trained,
                    "saved_at": json.dumps({"timestamp": __import__("time").time()}),
                }

                with open(temp_meta_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                final_faiss_path = f"{save_path}.faiss"
                final_meta_path = f"{save_path}.meta"

                shutil.move(temp_faiss_path, final_faiss_path)
                shutil.move(temp_meta_path, final_meta_path)

                logger.info(f"Index saved atomically to {save_path}")

                return save_path

            except Exception as e:
                logger.exception(f"Failed to save index: {e}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise

    def load(self, path: str) -> None:
        """Load the index from disk."""
        with self._get_lock("load"):
            faiss_path = f"{path}.faiss"
            meta_path = f"{path}.meta"

            if not os.path.exists(faiss_path):
                logger.warning(f"FAISS index not found at {faiss_path}")
                return

            try:
                self.index = faiss.read_index(faiss_path)
            except Exception as e:
                logger.exception(f"Failed to load FAISS index: {e}")
                raise

            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        data = json.load(f)
                    # Convert string keys back to integers (JSON converts int keys to strings)
                    raw_metadata = data.get("id_to_metadata", {})
                    self.id_to_metadata = {int(k): v for k, v in raw_metadata.items()}
                    self.next_id = data.get("next_id", 0)
                    self.dim = data.get("dim", self.dim)
                    self.metric = data.get("metric", self.metric)
                    self._is_trained = data.get("_is_trained", True)
                except Exception as e:
                    logger.exception(f"Failed to load metadata: {e}")
                    raise
            else:
                logger.warning(f"Metadata file not found at {meta_path}")
                self.id_to_metadata = {}
                self.next_id = 0
                self._is_trained = True

            logger.info(f"Index loaded from {path}")

    def count(self) -> int:
        """Return the number of embeddings in the index."""
        try:
            return self.index.ntotal
        except Exception:
            return 0

    def clear(self) -> None:
        """Clear the index."""
        with self._get_lock("clear"):
            if self.metric == "inner_product":
                metric_type = faiss.METRIC_INNER_PRODUCT
            else:
                metric_type = faiss.METRIC_L2

            self.index = faiss.IndexHNSWFlat(self.dim, DEFAULT_M, metric_type)
            self.id_to_metadata = {}
            self.next_id = 0
            self._is_trained = True

            logger.info("Index cleared")

    def get_status(self) -> Dict[str, Any]:
        """Get index status information."""
        return {
            "count": self.count(),
            "dim": self.dim,
            "metric": self.metric,
            "is_trained": self._is_trained,
            "index_type": type(self.index).__name__,
            "metadata_entries": len(self.id_to_metadata),
        }

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self._lock_dict.clear()
        except Exception:
            pass
