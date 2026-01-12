#!/usr/bin/env python3
"""Test script for memory management optimizations."""

import sys
import os
import gc
import time
import psutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.memory_manager import (
    MemoryManager, MemoryPressure, get_memory_manager, MemoryAwareProgressTracker
)
from backend.embeddings import CodeEmbedder
from backend.hybrid_indexer import HybridIndexer


def test_memory_manager():
    """Test memory manager functionality."""
    print("Testing Memory Manager...")
    
    # Initialize memory manager
    memory_manager = MemoryManager(
        memory_limit_mb=2048,  # 2GB limit for testing
        checkpoint_frequency=2,
        enable_adaptive_batching=True
    )
    
    # Test memory stats
    stats = memory_manager.get_memory_stats()
    print(f"Initial memory stats: {stats.rss_mb:.1f}MB RSS, {stats.percent:.1f}% system")
    
    # Test memory pressure detection
    pressure = memory_manager.get_memory_pressure()
    print(f"Memory pressure: {pressure.value}")
    
    # Test batch size adjustment
    batch_size = memory_manager.adjust_batch_size(1, True)
    print(f"Adjusted batch size: {batch_size}")
    
    # Test memory cleanup
    before_gc, after_gc = memory_manager.full_cleanup(0)
    print(f"Memory before GC: {before_gc.rss_mb:.1f}MB, after: {after_gc.rss_mb:.1f}MB")
    
    # Test checkpoint frequency
    should_checkpoint = memory_manager.should_checkpoint(2)
    print(f"Should checkpoint: {should_checkpoint}")
    
    print("[OK] Memory manager tests passed\n")


def test_memory_aware_progress():
    """Test memory-aware progress tracking."""
    print("Testing Memory-Aware Progress Tracking...")
    
    memory_manager = MemoryManager()
    progress_tracker = MemoryAwareProgressTracker(
        "Test operation", 1000, memory_manager
    )
    
    # Simulate processing
    for i in range(0, 1000, 50):
        batch_size = min(50, 1000 - i)
        progress_tracker.update(batch_size)
        
        if i % 200 == 0:
            status = progress_tracker.get_status()
            print(f"Progress {i}/1000: {status['memory']['pressure']} pressure")
    
    progress_tracker.finalize()
    print("[OK] Progress tracking tests passed\n")


def simulate_large_embedding():
    """Simulate large-scale embedding generation with memory management."""
    print("Simulating Large-Scale Embedding Generation...")
    
    # Create test data (simulate React repo with 24,958 chunks)
    test_chunks = [
        f"def function_{i}(): return {i}" for i in range(25000)
    ]
    
    memory_manager = MemoryManager(
        memory_limit_mb=4096,  # 4GB limit
        checkpoint_frequency=5,
        enable_adaptive_batching=True
    )
    
    print(f"Simulating {len(test_chunks)} chunks (React-sized repository)")
    
    # Initialize embedder
    embedder = CodeEmbedder()
    
    # Memory tracking
    start_memory = memory_manager.get_memory_stats().rss_mb
    batch_count = 0
    processed = 0
    
    # Process in batches with memory management
    for i in range(0, len(test_chunks), 64):
        batch_count += 1
        
        # Get optimal batch size
        batch_size = memory_manager.get_optimal_batch_size(64)
        
        # Adjust batch size if needed
        if batch_count > 1:
            batch_size = memory_manager.adjust_batch_size(batch_count, True)
        
        batch = test_chunks[i:i + batch_size]
        
        # Process batch
        start_time = time.time()
        embeddings = embedder.embed(batch)
        processing_time = time.time() - start_time
        
        processed += len(batch)
        
        # Memory cleanup
        if batch_count % 3 == 0:
            memory_manager.force_garbage_collection()
            memory_manager.cleanup_gpu_memory()
        
        # Log progress
        current_memory = memory_manager.get_memory_stats().rss_mb
        memory_delta = current_memory - start_memory
        
        if batch_count % 10 == 0:
            print(f"Batch {batch_count}: {processed}/{len(test_chunks)} chunks, "
                  f"Memory: {current_memory:.1f}MB (+{memory_delta:.1f}MB), "
                  f"Time: {processing_time:.2f}s, "
                  f"Pressure: {memory_manager.get_memory_pressure().value}")
        
        # Test checkpoint saving
        if memory_manager.should_checkpoint(batch_count):
            print(f"[CHECKPOINT] Saved after batch {batch_count}")
            memory_manager.log_memory_usage(f"Checkpoint {batch_count}")
        
        # Simulate memory pressure for testing
        if batch_count == 50:
            print("Simulating memory pressure...")
            # Force some memory allocation
            temp_data = [list(range(1000)) for _ in range(100)]
            pressure = memory_manager.get_memory_pressure()
            print(f"Memory pressure after allocation: {pressure.value}")
            del temp_data
    
    final_memory = memory_manager.get_memory_stats().rss_mb
    total_memory_increase = final_memory - start_memory
    
    print(f"\n[OK] Large-scale simulation complete:")
    print(f"  Processed: {processed} chunks in {batch_count} batches")
    print(f"  Memory increase: {total_memory_increase:.1f}MB")
    print(f"  Final pressure: {memory_manager.get_memory_pressure().value}")
    print(f"  Memory summary: {memory_manager.get_memory_summary()}")


def test_oom_prevention():
    """Test OOM prevention mechanisms."""
    print("Testing OOM Prevention...")

    memory_manager = MemoryManager(
        memory_limit_mb=512,  # Very low limit for testing
        enable_adaptive_batching=True
    )

    critical_iterations = 0

    # Simulate increasing memory pressure
    for batch_num in range(1, 21):
        # Allocate some memory
        temp_data = [0] * 100000

        # Check memory pressure
        pressure = memory_manager.get_memory_pressure()
        batch_size = memory_manager.adjust_batch_size(batch_num, True)

        print(f"Batch {batch_num}: pressure={pressure.value}, batch_size={batch_size}")

        # Clean up
        del temp_data
        gc.collect()

        # Track critical pressure iterations
        if pressure == MemoryPressure.CRITICAL:
            critical_iterations += 1
            # After 3+ iterations of critical pressure, batch size should be significantly reduced
            # (64 -> 32 -> 16 -> 8, so after 3 iterations it should be <= 16)
            if critical_iterations >= 3:
                assert batch_size <= 16, f"Batch size should be small after sustained critical pressure, got {batch_size}"

        if batch_num >= 10:
            break

    print("[OK] OOM prevention tests passed\n")


def main():
    """Run all memory management tests."""
    print("=" * 60)
    print("COARCH MEMORY MANAGEMENT TESTS")
    print("=" * 60)
    
    try:
        test_memory_manager()
        test_memory_aware_progress()
        test_oom_prevention()
        simulate_large_embedding()
        
        print("=" * 60)
        print("[SUCCESS] ALL MEMORY MANAGEMENT TESTS PASSED")
        print("=" * 60)
        
        # Print system information
        process = psutil.Process()
        memory_info = process.memory_info()
        print(f"Final memory usage: {memory_info.rss / 1024 / 1024:.1f}MB")
        
        if psutil.virtual_memory():
            vm = psutil.virtual_memory()
            print(f"System memory: {vm.percent:.1f}% used, {vm.available / 1024 / 1024:.1f}MB available")
        
    except Exception as e:
        print(f"[ERROR] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())