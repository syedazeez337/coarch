"""Progress tracking utilities for long-running operations."""

import time
from typing import Optional, Callable, Dict, Any
from contextlib import contextmanager
from tqdm import tqdm
import threading


class ProgressTracker:
    """Manages progress tracking for long operations with ETA estimates."""

    def __init__(self, min_duration_seconds: float = 5.0):
        """Initialize progress tracker.

        Args:
            min_duration_seconds: Only show progress bars for operations longer than this
        """
        self.min_duration_seconds = min_duration_seconds
        self._start_time: Optional[float] = None
        self._show_progress = False
        self._lock = threading.Lock()

    def should_show_progress(self, total_items: Optional[int] = None) -> bool:
        """Determine if progress should be shown based on estimated duration."""
        if total_items is None:
            # Always show progress if we can't estimate
            return True

        # Estimate duration based on total items (rough heuristics)
        estimated_duration = self._estimate_duration(total_items)
        return estimated_duration >= self.min_duration_seconds

    def _estimate_duration(self, total_items: int, operation_type: str = 'embedding_generation') -> float:
        """Estimate duration based on operation type and item count."""
        # These are rough estimates based on typical operation times
        estimates = {
            'file_indexing': 0.001,  # 1ms per file
            'embedding_generation': 0.05,  # 50ms per chunk
            'faiss_operations': 0.001,  # 1ms per vector
            'database_operations': 0.0001,  # 0.1ms per chunk
        }

        # Get estimate for operation type, fallback to embedding_generation
        estimate_per_item = estimates.get(operation_type, estimates['embedding_generation'])
        return total_items * estimate_per_item

    def start_timer(self):
        """Start the operation timer."""
        with self._lock:
            self._start_time = time.time()

    def get_elapsed_time(self) -> float:
        """Get elapsed time since timer started."""
        with self._lock:
            if self._start_time is None:
                return 0.0
            return time.time() - self._start_time

    @contextmanager
    def track_operation(self, operation_name: str, total_items: Optional[int] = None):
        """Context manager to track operation progress."""
        self.start_timer()
        show_progress = self.should_show_progress(total_items)
        
        if show_progress:
            progress_bar = self.create_progress_bar(operation_name, total_items)
            progress_bar.__enter__()
            try:
                yield progress_bar
            finally:
                progress_bar.__exit__(None, None, None)
        else:
            # Fast operation, just yield a no-op progress handler
            class NoOpProgress:
                def update(self, n=1): pass
                def set_description(self, desc): pass
                def set_postfix(self, postfix): pass
                def close(self): pass
            
            yield NoOpProgress()

    def create_progress_bar(
        self, 
        operation_name: str, 
        total: Optional[int] = None,
        unit: str = "items",
        **kwargs
    ) -> tqdm:
        """Create a formatted progress bar."""
        desc = f"{operation_name}"
        
        bar_kwargs = {
            'desc': desc,
            'unit': unit,
            'dynamic_ncols': True,
            'leave': False,
        }
        
        if total is not None:
            bar_kwargs['total'] = total
        
        bar_kwargs.update(kwargs)
        
        return tqdm(**bar_kwargs)

    def create_file_progress_bar(self, total_files: int) -> tqdm:
        """Create a progress bar for file operations."""
        return self.create_progress_bar(
            "Processing files",
            total=total_files,
            unit="files",
            ncols=80
        )

    def create_embedding_progress_bar(self, total_chunks: int) -> tqdm:
        """Create a progress bar for embedding operations."""
        return self.create_progress_bar(
            "Generating embeddings",
            total=total_chunks,
            unit="chunks",
            ncols=80
        )

    def create_faiss_progress_bar(self, total_vectors: int) -> tqdm:
        """Create a progress bar for FAISS operations."""
        return self.create_progress_bar(
            "Building index",
            total=total_vectors,
            unit="vectors",
            ncols=80
        )

    def create_deletion_progress_bar(self, total_chunks: int) -> tqdm:
        """Create a progress bar for deletion operations."""
        return self.create_progress_bar(
            "Deleting repository",
            total=total_chunks,
            unit="chunks",
            ncols=80
        )


class ProgressCallback:
    """Callable progress callback for chunked operations."""

    def __init__(self, progress_bar: tqdm, operation_name: str = "Processing"):
        """Initialize progress callback.

        Args:
            progress_bar: tqdm progress bar instance
            operation_name: Name of the operation for logging
        """
        self.progress_bar = progress_bar
        self.operation_name = operation_name
        self._last_update_time = time.time()

    def __call__(self, chunks_processed: int, total_chunks: Optional[int] = None):
        """Update progress for chunked operations."""
        current_time = time.time()
        
        # Throttle updates to avoid overwhelming the display
        if current_time - self._last_update_time >= 0.1:  # Update every 100ms
            if total_chunks is not None:
                self.progress_bar.total = total_chunks
            
            self.progress_bar.update(chunks_processed)
            self._last_update_time = current_time

    def set_description(self, description: str):
        """Set the progress bar description."""
        self.progress_bar.set_description(description)

    def set_postfix(self, **kwargs):
        """Set postfix information on the progress bar."""
        self.progress_bar.set_postfix(**kwargs)

    def close(self):
        """Close the progress bar."""
        self.progress_bar.close()


class ETAEstimator:
    """Estimates time remaining for operations."""

    def __init__(self, total_items: int):
        """Initialize ETA estimator.

        Args:
            total_items: Total number of items to process
        """
        self.total_items = total_items
        self.start_time = time.time()
        self.processed_items = 0
        self._lock = threading.Lock()

    def update(self, processed_items: int):
        """Update processed items count."""
        with self._lock:
            self.processed_items = processed_items

    def get_eta(self) -> Optional[float]:
        """Get estimated time remaining in seconds."""
        with self._lock:
            if self.processed_items == 0:
                return None

            elapsed = time.time() - self.start_time
            if elapsed <= 0:
                return None

            rate = self.processed_items / elapsed
            remaining_items = self.total_items - self.processed_items

            if rate <= 0:
                return None

            return remaining_items / rate

    def get_eta_string(self) -> str:
        """Get formatted ETA string."""
        eta_seconds = self.get_eta()
        if eta_seconds is None:
            return "Calculating..."
        
        if eta_seconds < 60:
            return f"{eta_seconds:.0f}s"
        elif eta_seconds < 3600:
            minutes = eta_seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = eta_seconds / 3600
            return f"{hours:.1f}h"

    def get_rate_string(self) -> str:
        """Get processing rate string."""
        with self._lock:
            elapsed = time.time() - self.start_time
            if self.processed_items == 0 or elapsed <= 0:
                return "0 items/s"
            
            rate = self.processed_items / elapsed
            if rate >= 1:
                return f"{rate:.1f} items/s"
            else:
                return f"{1/rate:.1f} s/item"


# Global progress tracker instance
_global_tracker = ProgressTracker()


def get_progress_tracker() -> ProgressTracker:
    """Get the global progress tracker instance."""
    return _global_tracker


def track_progress(
    operation_name: str, 
    total_items: Optional[int] = None,
    unit: str = "items",
    **kwargs
):
    """Decorator/context manager for tracking progress."""
    tracker = get_progress_tracker()
    return tracker.track_operation(operation_name, total_items)


def create_progress_bar(
    operation_name: str,
    total: Optional[int] = None,
    unit: str = "items",
    **kwargs
) -> tqdm:
    """Create a formatted progress bar."""
    tracker = get_progress_tracker()
    return tracker.create_progress_bar(operation_name, total, unit, **kwargs)


def should_show_progress(total_items: Optional[int] = None) -> bool:
    """Check if progress should be shown for given item count."""
    tracker = get_progress_tracker()
    return tracker.should_show_progress(total_items)


# Convenience functions for common operations
def create_file_progress_bar(total_files: int) -> tqdm:
    """Create a progress bar for file operations."""
    tracker = get_progress_tracker()
    return tracker.create_file_progress_bar(total_files)


def create_embedding_progress_bar(total_chunks: int) -> tqdm:
    """Create a progress bar for embedding operations."""
    tracker = get_progress_tracker()
    return tracker.create_embedding_progress_bar(total_chunks)


def create_faiss_progress_bar(total_vectors: int) -> tqdm:
    """Create a progress bar for FAISS operations."""
    tracker = get_progress_tracker()
    return tracker.create_faiss_progress_bar(total_vectors)


def create_deletion_progress_bar(total_chunks: int) -> tqdm:
    """Create a progress bar for deletion operations."""
    tracker = get_progress_tracker()
    return tracker.create_deletion_progress_bar(total_chunks)