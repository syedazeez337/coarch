"""Code embedding service using CodeBERT with production-ready features."""

import os
from typing import Any, Dict, List, Optional
import numpy as np
import gc

from transformers import AutoTokenizer, AutoModel
import torch

from .logging_config import get_logger
from .embedding_cache import EmbeddingCache
from .memory_manager import get_memory_manager, MemoryAwareProgressTracker

try:
    from .progress_tracker import (
        track_progress,
        create_embedding_progress_bar,
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

    def create_embedding_progress_bar(total_chunks):
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
        return total_items is not None and total_items > 50

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

DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


class CodeEmbedder:
    """Code embedding using CodeBERT with proper error handling."""

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        device: Optional[str] = None,
        max_length: int = 512,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    ):
        """Initialize the CodeBERT model.

        Args:
            model_name: HuggingFace model name
            device: "cuda", "cpu", or None for auto-detect
            max_length: Maximum sequence length
            max_file_size: Maximum file size in bytes (default 10MB)
        """
        self.model_name = model_name
        self.max_length = max_length
        self.max_file_size = max_file_size

        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
            logger.info("CUDA available, using GPU for embeddings")
        else:
            self.device = "cpu"
            logger.info("Using CPU for embeddings")

        self._init_model()
        self.cache = EmbeddingCache(model_name=self.model_name)

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
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Generate embeddings for code snippets with caching.

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

        # Check cache for each snippet
        cached_embeddings: List[Optional[np.ndarray]] = []
        uncached_indices: List[int] = []
        uncached_snippets: List[str] = []

        # Use progress tracking for cache checking if needed
        cache_check_progress = None
        if should_show_progress(total):
            try:
                cache_check_progress = track_progress("Checking cache", total, "snippets")
                cache_check_progress.__enter__()
            except Exception as e:
                logger.debug(f"Progress tracking failed: {e}")
                cache_check_progress = None

        for i, snippet in enumerate(code_snippets):
            cached = self.cache.get(snippet)
            cached_embeddings.append(cached)
            if cached is None:
                uncached_indices.append(i)
                uncached_snippets.append(snippet)
            
            if cache_check_progress and hasattr(cache_check_progress, 'update'):
                try:
                    cache_check_progress.update(1)
                except Exception as e:
                    logger.debug(f"Progress update failed: {e}")
                    cache_check_progress = None

        if cache_check_progress and hasattr(cache_check_progress, '__exit__'):
            try:
                cache_check_progress.__exit__(None, None, None)
            except Exception as e:
                logger.debug(f"Progress exit failed: {e}")

        cache_hits = total - len(uncached_snippets)
        logger.info(f"Cache hits: {cache_hits}/{total} ({100*cache_hits/total:.1f}%)")

        # Compute embeddings for cache misses
        if uncached_snippets:
            new_embeddings = self._embed_uncached(uncached_snippets, batch_size)

            # Store new embeddings in cache and update cached_embeddings list
            for idx, (snippet, embedding) in enumerate(
                zip(uncached_snippets, new_embeddings)
            ):
                self.cache.put(snippet, embedding)
                cached_embeddings[uncached_indices[idx]] = embedding

        # Build final result array
        result = np.vstack([e for e in cached_embeddings if e is not None])
        logger.info(f"Generated embeddings with shape: {result.shape}")

        return result

    def _embed_uncached(
        self,
        code_snippets: List[str],
        batch_size: int = 64,
        memory_aware: bool = True,
    ) -> np.ndarray:
        """Generate embeddings for code snippets without caching.

        Args:
            code_snippets: List of code strings
            batch_size: Batch size for processing
            memory_aware: Enable memory-aware processing

        Returns:
            numpy array of shape (len(code_snippets), dim)
        """
        if not code_snippets:
            return np.array([])

        total = len(code_snippets)
        embeddings: List[np.ndarray] = []

        # Create ETA estimator for progress tracking
        eta_estimator = ETAEstimator(total) if should_show_progress(total) else None
        
        # Initialize memory management if enabled
        memory_manager = None
        progress_tracker = None
        if memory_aware:
            memory_manager = get_memory_manager()
            progress_tracker = MemoryAwareProgressTracker(
                "Generating embeddings", total, memory_manager
            )

        batch_num = 0
        i = 0
        
        while i < total:
            # Get optimal batch size if memory-aware
            if memory_aware:
                current_batch_size = memory_manager.get_optimal_batch_size(batch_size)
                if batch_num > 0:
                    current_batch_size = memory_manager.adjust_batch_size(batch_num, True)
            else:
                current_batch_size = batch_size
            
            batch = code_snippets[i : i + current_batch_size]
            batch_num += 1

            try:
                batch_embeddings = self._embed_batch(batch)
                embeddings.append(batch_embeddings)
                
                # Update progress
                if progress_tracker:
                    progress_tracker.update(len(batch))
                elif eta_estimator:
                    processed = min(i + current_batch_size, total)
                    eta_estimator.update(processed)
                
                # Memory cleanup between batches
                if memory_aware:
                    # Force garbage collection periodically
                    if batch_num % 3 == 0:
                        memory_manager.force_garbage_collection()
                    
                    # Clear GPU cache if using CUDA
                    if torch.cuda.is_available():
                        memory_manager.cleanup_gpu_memory()
                    
                    # Log memory usage for large batches
                    if batch_num % 10 == 0:
                        memory_manager.log_memory_usage(f"Batch {batch_num}")
                
            except Exception as e:
                logger.error(f"Failed to embed batch starting at {i}: {e}")
                raise
            
            i += current_batch_size

        # Final cleanup
        if progress_tracker:
            progress_tracker.finalize()
        elif memory_aware:
            memory_manager.log_memory_usage("Embedding generation complete")
            memory_manager.full_cleanup(batch_num)

        return np.vstack(embeddings)

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

    def embed_file(self, file_path: str) -> Optional[np.ndarray]:
        """Generate embedding for a file.

        Args:
            file_path: Path to file

        Returns:
            numpy array of shape (dim,), or None if file was skipped
        """
        try:
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                logger.warning(
                    f"Skipping file exceeding size limit: {file_path} "
                    f"({file_size / 1024 / 1024:.2f}MB > {self.max_file_size / 1024 / 1024:.2f}MB)"
                )
                return None

            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                code = f.read()
        except OSError as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return None

        if not code.strip():
            logger.debug(f"Skipping empty file: {file_path}")
            return None

        return self.embed([code])[0]

    def embed_files(
        self, file_paths: List[str], batch_size: int = 64
    ) -> Dict[str, Any]:
        """Generate embeddings for multiple files with size limit enforcement.

        Args:
            file_paths: List of file paths to embed
            batch_size: Batch size for processing

        Returns:
            Dict with embeddings, skipped files info, and stats
        """
        valid_contents: List[str] = []
        valid_paths: List[str] = []
        skipped_size: List[str] = []
        skipped_empty: List[str] = []
        skipped_error: List[str] = []

        for file_path in file_paths:
            try:
                file_size = os.path.getsize(file_path)
                if file_size > self.max_file_size:
                    logger.warning(
                        f"Skipping oversized file: {file_path} "
                        f"({file_size / 1024 / 1024:.2f}MB)"
                    )
                    skipped_size.append(file_path)
                    continue

                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    code = f.read()

                if not code.strip():
                    skipped_empty.append(file_path)
                    continue

                valid_contents.append(code)
                valid_paths.append(file_path)

            except OSError as e:
                logger.error(f"Failed to read {file_path}: {e}")
                skipped_error.append(file_path)

        embeddings = None
        if valid_contents:
            embeddings = self.embed(valid_contents, batch_size=batch_size)

        total_skipped = len(skipped_size) + len(skipped_empty) + len(skipped_error)
        if total_skipped > 0:
            logger.info(
                f"File embedding stats: {len(valid_paths)} processed, "
                f"{len(skipped_size)} oversized, {len(skipped_empty)} empty, "
                f"{len(skipped_error)} errors"
            )

        return {
            "embeddings": embeddings,
            "file_paths": valid_paths,
            "skipped_size_limit": skipped_size,
            "skipped_empty": skipped_empty,
            "skipped_error": skipped_error,
            "stats": {
                "processed": len(valid_paths),
                "skipped_size": len(skipped_size),
                "skipped_empty": len(skipped_empty),
                "skipped_error": len(skipped_error),
            },
        }

    def check_file_size(self, file_path: str) -> bool:
        """Check if a file is within the size limit.

        Args:
            file_path: Path to the file

        Returns:
            True if file is within limit, False otherwise
        """
        try:
            return os.path.getsize(file_path) <= self.max_file_size
        except OSError:
            return False

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
