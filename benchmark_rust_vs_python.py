"""Coarch Rust vs Python Benchmark

Compare performance of Rust-based indexing vs Python-based indexing.
"""

import sys
import os
import time
import tempfile

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("Coarch Rust vs Python Benchmark")
print("=" * 70)

REPO_PATH = "C:/Users/Azeez/Myproject/faiss"
REPO_NAME = "FAISS"

# Check if Rust module is available
rust_available = False
try:
    from backend.coarch_rust import RustIndexer

    rust_available = True
    print("\n✓ Rust module loaded successfully")
except ImportError as e:
    print(f"\n✗ Rust module not available: {e}")
    print("  Building: cd rust-indexer && cargo build --release")

# Python indexer
from backend.hybrid_indexer import HybridIndexer
from backend.ast_analyzer import TreeSitterAnalyzer

print(f"\nRepository: {REPO_PATH}")
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# ===== RUST BENCHMARK =====
rust_stats = None
if rust_available:
    print("\n" + "=" * 70)
    print("RUST INDEXER (Parallel + Rayon)")
    print("=" * 70)

    rust_start = time.time()

    temp_dir = tempfile.mkdtemp()
    rust_indexer = RustIndexer(
        index_path=os.path.join(temp_dir, "rust_index"),
        db_path=os.path.join(temp_dir, "rust.db"),
    )

    rust_stats_py = rust_indexer.index_directory(REPO_PATH, REPO_NAME)
    rust_indexing_time = time.time() - rust_start

    # Convert to dict
    rust_chunk_count = rust_indexer.get_chunk_count()

    print(f"\n   Indexing time: {rust_indexing_time:.3f}s")
    print(f"   Files indexed: {rust_stats_py.get('files_indexed', 'N/A')}")
    print(f"   Chunks created: {rust_chunk_count}")
    print(
        f"   Throughput: {rust_stats_py.get('files_indexed', 1) / rust_indexing_time:.1f} files/s"
    )

    # Test search
    search_start = time.time()
    for query in ["authentication function", "database connection", "search algorithm"]:
        results = rust_indexer.search(query, 5)
    rust_search_time = (time.time() - search_start) / 3

    print(f"\n   Average search time: {rust_search_time*1000:.2f}ms")
    print(f"   Search results sample:")
    for r in rust_indexer.search("indexing", 3)[:1]:
        print(f"      - {r['file']} (score: {r['score']:.3f})")

# ===== PYTHON BENCHMARK =====
print("\n" + "=" * 70)
print("PYTHON INDEXER (Tree-sitter)")
print("=" * 70)

python_start = time.time()

temp_dir = tempfile.mkdtemp()
python_indexer = HybridIndexer(db_path=os.path.join(temp_dir, "python.db"))
analyzer = TreeSitterAnalyzer()

python_stats = python_indexer.index_repository(REPO_PATH, REPO_NAME)
python_indexing_time = time.time() - python_start

python_chunks = python_indexer.get_chunks_for_embedding()
python_chunk_count = len(python_chunks)

print(f"\n   Indexing time: {python_indexing_time:.3f}s")
print(f"   Files indexed: {python_stats['files_indexed']}")
print(f"   Chunks created: {python_chunk_count}")
print(
    f"   Throughput: {python_stats['files_indexed'] / python_indexing_time:.1f} files/s"
)

# ===== COMPARISON =====
print("\n" + "=" * 70)
print("PERFORMANCE COMPARISON")
print("=" * 70)

if rust_available and python_stats:
    speedup = python_indexing_time / rust_indexing_time if rust_indexing_time > 0 else 0

    print(f"\n{'Metric':<30} {'Python':>15} {'Rust':>15} {'Speedup':>15}")
    print("-" * 75)
    print(
        f"{'Files indexed':<30} {python_stats['files_indexed']:>15} {rust_stats_py.get('files_indexed', 'N/A'):>15} {'-':>15}"
    )
    print(
        f"{'Chunks created':<30} {python_chunk_count:>15} {rust_chunk_count:>15} {'-':>15}"
    )
    print(
        f"{'Indexing time (s)':<30} {python_indexing_time:>15.3f} {rust_indexing_time:>15.3f} {speedup:>14.1f}x"
    )
    print(
        f"{'Files/second':<30} {python_stats['files_indexed']/python_indexing_time:>15.1f} {rust_stats_py.get('files_indexed', 1)/rust_indexing_time:>15.1f} {speedup:>14.1f}x"
    )

    print(f"\n{'='*70}")
    if speedup > 1:
        print(f"✓ Rust is {speedup:.1f}x FASTER than Python!")
    elif speedup < 1:
        print(f"✓ Python is {1/speedup:.1f}x FASTER than Rust (unexpected!)")
    else:
        print("- Python and Rust have similar performance")
    print(f"{'='*70}")

print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)
if rust_available:
    print(
        f"""
• Rust indexing: {rust_indexing_time*1000:.0f}ms for {rust_stats_py.get('files_indexed', 'N/A')} files
• Python indexing: {python_indexing_time*1000:.0f}ms for {python_stats['files_indexed']} files
• Speedup: {speedup:.1f}x with Rust

• Rust advantages:
  - Parallel file scanning with Rayon
  - Zero-copy operations where possible
  - Type-safe memory management
  - No GIL (Global Interpreter Lock)

• Python advantages:
  - Tree-sitter integration for accurate AST parsing
  - Easier debugging and development
  - Rich ecosystem of NLP libraries
"""
    )
print("=" * 70)
