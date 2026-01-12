"""Memory management utilities for large-scale indexing operations."""

import gc
import os
import psutil
import time
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class MemoryPressure(Enum):
    """Memory pressure levels."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    rss_mb: float
    vms_mb: float
    percent: float
    available_mb: Optional[float] = None
    gpu_allocated_mb: Optional[float] = None
    gpu_reserved_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None


class MemoryManager:
    """Manages memory usage for large-scale operations."""

    def __init__(self, 
                 memory_limit_mb: Optional[int] = None,
                 gpu_memory_fraction: float = 0.8,
                 checkpoint_frequency: int = 5,
                 enable_adaptive_batching: bool = True):
        """Initialize memory manager.
        
        Args:
            memory_limit_mb: Memory limit in MB (None for auto-detect)
            gpu_memory_fraction: Fraction of GPU memory to use (0.0-1.0)
            checkpoint_frequency: Save checkpoint every N batches
            enable_adaptive_batching: Enable dynamic batch size adjustment
        """
        self.memory_limit_mb = memory_limit_mb
        self.gpu_memory_fraction = gpu_memory_fraction
        self.checkpoint_frequency = checkpoint_frequency
        self.enable_adaptive_batching = enable_adaptive_batching
        
        # Dynamic batch sizing
        self.base_batch_size = 64
        self.min_batch_size = 8
        self.max_batch_size = 256
        self.current_batch_size = self.base_batch_size
        self.batch_size_history = []
        
        # Memory tracking
        self.baseline_memory = None
        self.peak_memory = 0
        self.memory_warnings = 0
        
        # Initialize baseline
        self._set_baseline()
        
        logger.info(f"MemoryManager initialized: limit={memory_limit_mb}MB, "
                   f"adaptive_batching={enable_adaptive_batching}")

    def _set_baseline(self):
        """Set baseline memory usage."""
        stats = self.get_memory_stats()
        self.baseline_memory = stats.rss_mb
        self.peak_memory = stats.rss_mb
        logger.debug(f"Memory baseline set: {self.baseline_memory:.1f}MB")

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Get system memory info
        system_memory = psutil.virtual_memory()
        
        stats = MemoryStats(
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            percent=system_memory.percent,
            available_mb=system_memory.available / 1024 / 1024
        )
        
        # Add GPU memory info if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                stats.gpu_allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
                stats.gpu_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
                
                # Try to get GPU utilization (may not be available on all systems)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    stats.gpu_utilization = util.gpu
                except (ImportError, Exception):
                    stats.gpu_utilization = None
                    
            except Exception as e:
                logger.debug(f"Could not get GPU memory stats: {e}")
        
        # Update peak memory
        if stats.rss_mb > self.peak_memory:
            self.peak_memory = stats.rss_mb
            
        return stats

    def get_memory_pressure(self) -> MemoryPressure:
        """Determine current memory pressure level."""
        stats = self.get_memory_stats()
        
        if self.memory_limit_mb:
            # Use memory limit if specified
            usage_percent = (stats.rss_mb / self.memory_limit_mb) * 100
            if usage_percent > 90:
                return MemoryPressure.CRITICAL
            elif usage_percent > 75:
                return MemoryPressure.HIGH
            elif usage_percent > 60:
                return MemoryPressure.MODERATE
            else:
                return MemoryPressure.LOW
        else:
            # Use system memory percentage
            if stats.percent > 90:
                return MemoryPressure.CRITICAL
            elif stats.percent > 75:
                return MemoryPressure.HIGH
            elif stats.percent > 60:
                return MemoryPressure.MODERATE
            else:
                return MemoryPressure.LOW

    def should_adjust_batch_size(self, batch_num: int) -> bool:
        """Check if batch size should be adjusted."""
        if not self.enable_adaptive_batching:
            return False
            
        # Adjust every few batches or when memory pressure is high
        pressure = self.get_memory_pressure()
        return (batch_num % 3 == 0) or (pressure in [MemoryPressure.HIGH, MemoryPressure.CRITICAL])

    def adjust_batch_size(self, batch_num: int, last_batch_success: bool = True) -> int:
        """Dynamically adjust batch size based on memory pressure."""
        if not self.enable_adaptive_batching:
            return self.base_batch_size
            
        pressure = self.get_memory_pressure()
        
        if pressure == MemoryPressure.CRITICAL:
            # Reduce batch size significantly
            self.current_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
        elif pressure == MemoryPressure.HIGH:
            # Reduce batch size moderately
            self.current_batch_size = max(self.min_batch_size, int(self.current_batch_size * 0.75))
        elif pressure == MemoryPressure.MODERATE:
            # Slightly reduce if we had issues
            if not last_batch_success:
                self.current_batch_size = max(self.min_batch_size, int(self.current_batch_size * 0.9))
        elif pressure == MemoryPressure.LOW:
            # Gradually increase batch size if we're doing well
            if len(self.batch_size_history) > 5:
                recent_success_rate = sum(self.batch_size_history[-5:]) / 5
                if recent_success_rate > 0.95:  # 95% success rate
                    self.current_batch_size = min(self.max_batch_size, int(self.current_batch_size * 1.1))
        
        self.batch_size_history.append(1 if last_batch_success else 0)
        if len(self.batch_size_history) > 20:
            self.batch_size_history.pop(0)
            
        logger.debug(f"Batch {batch_num}: pressure={pressure.value}, "
                    f"batch_size={self.current_batch_size}")
        
        return self.current_batch_size

    def force_garbage_collection(self) -> MemoryStats:
        """Force garbage collection and return memory stats."""
        collected = gc.collect()
        logger.debug(f"Garbage collection freed {collected} objects")
        
        # Get memory stats after GC
        return self.get_memory_stats()

    def cleanup_gpu_memory(self) -> Optional[MemoryStats]:
        """Clean up GPU memory if using CUDA."""
        if not (TORCH_AVAILABLE and torch.cuda.is_available()):
            return None
            
        try:
            torch.cuda.empty_cache()
            logger.debug("GPU cache cleared")
            return self.get_memory_stats()
        except Exception as e:
            logger.warning(f"Failed to clear GPU cache: {e}")
            return None

    def full_cleanup(self, batch_num: int) -> Tuple[MemoryStats, MemoryStats]:
        """Perform full memory cleanup."""
        logger.debug(f"Performing full cleanup after batch {batch_num}")
        
        # Clean up Python objects
        before_gc = self.get_memory_stats()
        after_gc = self.force_garbage_collection()
        
        # Clean up GPU memory
        after_gpu = self.cleanup_gpu_memory()
        
        # Log memory usage
        self.log_memory_usage(f"After cleanup (batch {batch_num})", after_gc)
        
        return before_gc, after_gc

    def log_memory_usage(self, context: str, stats: Optional[MemoryStats] = None):
        """Log current memory usage."""
        if stats is None:
            stats = self.get_memory_stats()
            
        pressure = self.get_memory_pressure()
        
        message = (f"{context}: RSS={stats.rss_mb:.1f}MB, "
                  f"VMS={stats.vms_mb:.1f}MB, "
                  f"System={stats.percent:.1f}%, "
                  f"Pressure={pressure.value}")
        
        if stats.gpu_allocated_mb is not None:
            message += f", GPU={stats.gpu_allocated_mb:.1f}MB"
            
        if pressure in [MemoryPressure.HIGH, MemoryPressure.CRITICAL]:
            logger.warning(message)
            self.memory_warnings += 1
        else:
            logger.debug(message)

    def should_checkpoint(self, batch_num: int) -> bool:
        """Check if we should save a checkpoint."""
        return batch_num > 0 and batch_num % self.checkpoint_frequency == 0

    def get_optimal_batch_size(self, default_size: int = 64) -> int:
        """Get optimal batch size considering memory constraints."""
        if not self.enable_adaptive_batching:
            return default_size
            
        pressure = self.get_memory_pressure()
        
        # Start with the current adjusted size
        optimal_size = self.current_batch_size
        
        # Apply memory-based constraints
        if self.memory_limit_mb:
            stats = self.get_memory_stats()
            memory_usage_ratio = stats.rss_mb / self.memory_limit_mb
            
            if memory_usage_ratio > 0.8:
                optimal_size = min(optimal_size, 16)
            elif memory_usage_ratio > 0.6:
                optimal_size = min(optimal_size, 32)
        
        # Ensure within bounds
        optimal_size = max(self.min_batch_size, min(self.max_batch_size, optimal_size))
        
        logger.debug(f"Optimal batch size: {optimal_size} (pressure={pressure.value})")
        return optimal_size

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory summary."""
        stats = self.get_memory_stats()
        pressure = self.get_memory_pressure()
        
        return {
            "current_mb": stats.rss_mb,
            "peak_mb": self.peak_memory,
            "baseline_mb": self.baseline_memory,
            "system_percent": stats.percent,
            "available_mb": stats.available_mb,
            "gpu_allocated_mb": stats.gpu_allocated_mb,
            "gpu_reserved_mb": stats.gpu_reserved_mb,
            "pressure": pressure.value,
            "current_batch_size": self.current_batch_size,
            "memory_warnings": self.memory_warnings,
            "memory_limit_mb": self.memory_limit_mb,
        }


# Global memory manager instance
_global_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get the global memory manager instance."""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = MemoryManager()
    return _global_memory_manager


def set_memory_manager(manager: MemoryManager):
    """Set the global memory manager instance."""
    global _global_memory_manager
    _global_memory_manager = manager


class MemoryAwareProgressTracker:
    """Progress tracker with memory-aware features."""
    
    def __init__(self, operation_name: str, total_items: int, memory_manager: Optional[MemoryManager] = None):
        self.operation_name = operation_name
        self.total_items = total_items
        self.memory_manager = memory_manager or get_memory_manager()
        self.processed_items = 0
        self.batch_count = 0
        self.last_memory_log = time.time()
        self.start_time = time.time()
        
        logger.info(f"Starting {operation_name} for {total_items} items")
        self.memory_manager.log_memory_usage(f"Start {operation_name}")
    
    def update(self, batch_size: int):
        """Update progress and check memory."""
        self.processed_items += batch_size
        self.batch_count += 1
        
        # Log memory usage periodically
        current_time = time.time()
        if current_time - self.last_memory_log > 30:  # Every 30 seconds
            self.memory_manager.log_memory_usage(f"{self.operation_name} progress")
            self.last_memory_log = current_time
        
        # Check if we should cleanup
        if self.memory_manager.should_checkpoint(self.batch_count):
            logger.info(f"{self.operation_name}: {self.processed_items}/{self.total_items} items "
                       f"({self.batch_count} batches) - saving checkpoint")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        elapsed_time = time.time() - self.start_time
        rate = self.processed_items / elapsed_time if elapsed_time > 0 else 0
        
        status = {
            "processed": self.processed_items,
            "total": self.total_items,
            "batches": self.batch_count,
            "rate": rate,
            "elapsed_seconds": elapsed_time,
        }
        
        if self.memory_manager:
            status["memory"] = self.memory_manager.get_memory_summary()
            
        return status
    
    def finalize(self):
        """Finalize progress tracking."""
        total_time = time.time() - self.start_time
        logger.info(f"Completed {self.operation_name}: {self.processed_items} items "
                   f"in {total_time:.1f}s ({self.batch_count} batches)")
        self.memory_manager.log_memory_usage(f"End {self.operation_name}")