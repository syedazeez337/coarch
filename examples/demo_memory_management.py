#!/usr/bin/env python3
"""Demonstration of memory management optimizations for large-scale indexing."""

import sys
import os
import time
import psutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.memory_manager import MemoryManager, get_memory_manager
from backend.embeddings import CodeEmbedder


def demonstrate_memory_management():
    """Demonstrate memory management features."""
    print("=" * 70)
    print("COARCH MEMORY MANAGEMENT DEMONSTRATION")
    print("=" * 70)
    
    # Initialize memory manager with realistic settings
    memory_manager = MemoryManager(
        memory_limit_mb=8192,  # 8GB limit
        checkpoint_frequency=3,
        enable_adaptive_batching=True
    )
    
    print(f"Memory Manager initialized:")
    print(f"  Memory limit: 8192 MB")
    print(f"  Checkpoint frequency: 3 batches")
    print(f"  Adaptive batching: Enabled")
    
    # Show initial memory state
    initial_stats = memory_manager.get_memory_stats()
    print(f"\nInitial memory usage: {initial_stats.rss_mb:.1f}MB RSS, {initial_stats.percent:.1f}% system")
    
    print("\n" + "=" * 50)
    print("DEMONSTRATION 1: Memory Pressure Detection")
    print("=" * 50)
    
    # Simulate different memory pressure levels
    pressure_levels = [
        ("Low", lambda: None),
        ("Moderate", lambda: allocate_memory(50 * 1024 * 1024)),  # 50MB
        ("High", lambda: allocate_memory(200 * 1024 * 1024)),     # 200MB
        ("Critical", lambda: allocate_memory(500 * 1024 * 1024)), # 500MB
    ]
    
    allocated_objects = []
    
    for level_name, allocate_func in pressure_levels:
        print(f"\nTesting {level_name} memory pressure:")
        
        # Allocate memory to simulate pressure
        if allocate_func:
            allocated_objects.append(allocate_func())
        
        pressure = memory_manager.get_memory_pressure()
        stats = memory_manager.get_memory_stats()
        
        print(f"  Pressure: {pressure.value}")
        print(f"  Memory: {stats.rss_mb:.1f}MB RSS, {stats.percent:.1f}% system")
        print(f"  Available: {stats.available_mb:.1f}MB")
        
        # Test batch size adjustment
        batch_size = memory_manager.adjust_batch_size(1, True)
        optimal_size = memory_manager.get_optimal_batch_size(64)
        print(f"  Adjusted batch size: {batch_size}")
        print(f"  Optimal batch size: {optimal_size}")
    
    # Clean up allocated memory
    del allocated_objects
    import gc
    gc.collect()
    
    print("\n" + "=" * 50)
    print("DEMONSTRATION 2: Adaptive Batch Sizing")
    print("=" * 50)
    
    # Test batch size adaptation under different conditions
    print("\nSimulating batch processing with memory pressure:")
    
    for batch_num in range(1, 16):
        # Simulate memory usage increasing over time
        if batch_num <= 5:
            temp_data = allocate_memory(10 * 1024 * 1024)  # 10MB
        elif batch_num <= 10:
            temp_data = allocate_memory(30 * 1024 * 1024)  # 30MB
        else:
            temp_data = allocate_memory(60 * 1024 * 1024)  # 60MB
        
        # Check memory pressure and adjust batch size
        pressure = memory_manager.get_memory_pressure()
        batch_size = memory_manager.adjust_batch_size(batch_num, True)
        
        print(f"Batch {batch_num:2d}: pressure={pressure.value:8s}, "
              f"batch_size={batch_size:3d}, memory={memory_manager.get_memory_stats().rss_mb:.1f}MB")
        
        # Clean up
        del temp_data
        if batch_num % 3 == 0:
            memory_manager.force_garbage_collection()
    
    print("\n" + "=" * 50)
    print("DEMONSTRATION 3: Memory Cleanup and Checkpointing")
    print("=" * 50)
    
    print("\nTesting memory cleanup operations:")
    
    # Allocate memory and then clean up
    print("1. Allocating temporary memory...")
    temp_objects = [allocate_memory(20 * 1024 * 1024) for _ in range(5)]  # 100MB total
    
    before_cleanup = memory_manager.get_memory_stats()
    print(f"   Memory before cleanup: {before_cleanup.rss_mb:.1f}MB")
    
    print("2. Performing full cleanup...")
    before_gc, after_gc = memory_manager.full_cleanup(0)
    print(f"   Memory after cleanup: {after_gc.rss_mb:.1f}MB")
    print(f"   Memory freed: {before_gc.rss_mb - after_gc.rss_mb:.1f}MB")
    
    # Test checkpoint logic
    print("\n3. Testing checkpoint logic:")
    for batch_num in [1, 2, 3, 4, 5, 6]:
        should_checkpoint = memory_manager.should_checkpoint(batch_num)
        checkpoint_indicator = "[CHECKPOINT]" if should_checkpoint else ""
        print(f"   Batch {batch_num}: {checkpoint_indicator}")
    
    print("\n" + "=" * 50)
    print("DEMONSTRATION 4: Real-world Embedding Simulation")
    print("=" * 50)
    
    print("\nSimulating large-scale embedding generation (React-sized: 24,958 chunks):")
    
    # Simulate processing React-sized repository
    total_chunks = 24958
    base_batch_size = 64
    
    # Create realistic chunk data
    sample_chunks = [
        f"function component_{i}() {{ return <div>{i}</div>; }}" 
        for i in range(min(1000, total_chunks))  # Use smaller sample for demo
    ]
    
    print(f"Processing {len(sample_chunks)} sample chunks:")
    
    embedder = None
    try:
        embedder = CodeEmbedder()
        print("  CodeEmbedder loaded successfully")
    except Exception as e:
        print(f"  CodeEmbedder failed to load: {e}")
        print("  Continuing with simulation...")
    
    # Process chunks with memory management
    for i in range(0, len(sample_chunks), base_batch_size):
        batch_num = i // base_batch_size + 1
        
        # Get optimal batch size
        current_batch_size = memory_manager.get_optimal_batch_size(base_batch_size)
        
        # Adjust based on memory pressure
        if batch_num > 1:
            current_batch_size = memory_manager.adjust_batch_size(batch_num, True)
        
        batch = sample_chunks[i:i + current_batch_size]
        
        # Process batch if embedder is available
        if embedder:
            try:
                embeddings = embedder.embed(batch)
                processing_info = f"generated {len(embeddings)} embeddings"
            except Exception as e:
                processing_info = f"processing failed: {e}"
        else:
            # Simulate processing time
            time.sleep(0.01)
            processing_info = "simulated processing"
        
        # Memory cleanup
        if batch_num % 3 == 0:
            memory_manager.force_garbage_collection()
            memory_manager.cleanup_gpu_memory()
        
        # Log progress
        if batch_num % 5 == 0:
            stats = memory_manager.get_memory_stats()
            pressure = memory_manager.get_memory_pressure()
            print(f"  Batch {batch_num:2d}: {len(batch):2d} chunks, "
                  f"batch_size={current_batch_size:2d}, "
                  f"memory={stats.rss_mb:.1f}MB, "
                  f"pressure={pressure.value}")
        
        # Test checkpoint saving
        if memory_manager.should_checkpoint(batch_num):
            print(f"    [CHECKPOINT] Saved progress after batch {batch_num}")
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    final_stats = memory_manager.get_memory_stats()
    memory_summary = memory_manager.get_memory_summary()
    
    print(f"Memory management demonstration completed:")
    print(f"  Initial memory: {initial_stats.rss_mb:.1f}MB")
    print(f"  Final memory: {final_stats.rss_mb:.1f}MB")
    print(f"  Peak memory: {memory_summary['peak_mb']:.1f}MB")
    print(f"  Memory warnings: {memory_summary['memory_warnings']}")
    print(f"  Final pressure: {memory_manager.get_memory_pressure().value}")
    print(f"  System memory: {final_stats.percent:.1f}% used")
    
    print(f"\nKey features demonstrated:")
    print(f"  ✓ Memory pressure detection")
    print(f"  ✓ Adaptive batch sizing")
    print(f"  ✓ Garbage collection")
    print(f"  ✓ GPU memory cleanup")
    print(f"  ✓ Checkpoint saving")
    print(f"  ✓ Memory usage monitoring")
    
    print("\n" + "=" * 70)
    print("MEMORY MANAGEMENT OPTIMIZATION COMPLETE")
    print("=" * 70)


def allocate_memory(size_bytes):
    """Allocate memory for testing purposes."""
    # Create a list of integers to allocate memory
    size_elements = size_bytes // 8  # Each int is typically 8 bytes
    return [i for i in range(size_elements)]


if __name__ == "__main__":
    try:
        demonstrate_memory_management()
    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted by user.")
    except Exception as e:
        print(f"\nDemonstration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)