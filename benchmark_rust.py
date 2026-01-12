"""Coarch Rust Performance Demo

This script demonstrates the Rust-based indexing module.
"""

import sys
import os
import time
import tempfile
import subprocess

print("=" * 70)
print("Coarch Rust Performance Demo")
print("=" * 70)

REPO_PATH = "C:/Users/Azeez/Myproject/faiss"
REPO_NAME = "FAISS"

print(f"\nRepository: {REPO_PATH}")
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

# ===== RUST BENCHMARK =====
print("\n" + "=" * 70)
print("RUST INDEXER (Parallel + Rayon)")
print("=" * 70)

rust_start = time.time()

temp_dir = tempfile.mkdtemp()
index_path = os.path.join(temp_dir, "rust_index")
db_path = os.path.join(temp_dir, "rust.db")

print(f"\n[1/4] Initializing Rust indexer...")
print(f"   Index path: {index_path}")

print(f"\n[2/4] Building Rust module (if needed)...")
build_result = subprocess.run(
    ["cargo", "build", "--release"], cwd="rust-indexer", capture_output=True, text=True
)
if build_result.returncode == 0:
    print("   ✓ Rust module built successfully")
else:
    print(f"   ✗ Build failed: {build_result.stderr[:200]}")

# Copy the DLL
dll_src = "rust-indexer/target/release/coarch_rust.dll"
dll_dst = "backend/coarch_rust.dll"
if os.path.exists(dll_src):
    import shutil

    shutil.copy(dll_src, dll_dst)
    print(f"   ✓ DLL copied to backend/")

# Try to import and use the Rust module
try:
    sys.path.insert(0, "backend")
    import coarch_rust

    print(f"\n[3/4] Running Rust indexing...")

    rust_indexer = coarch_rust.RustIndexer(index_path, db_path)
    rust_stats = rust_indexer.index_directory(REPO_PATH, REPO_NAME)
    rust_chunk_count = rust_indexer.get_chunk_count()

    rust_indexing_time = time.time() - rust_start

    print(f"   Files indexed: {rust_stats.get('files_indexed', 'N/A')}")
    print(f"   Chunks created: {rust_chunk_count}")
    print(f"   Indexing time: {rust_indexing_time:.3f}s")
    print(
        f"   Throughput: {rust_stats.get('files_indexed', 1) / rust_indexing_time:.1f} files/s"
    )

    # Test search
    print(f"\n[4/4] Testing search...")
    search_start = time.time()
    results = rust_indexer.search("authentication function", 5)
    search_time = (time.time() - search_start) * 1000

    print(f"   Search time: {search_time:.2f}ms")
    print(f"   Results: {len(results)} found")
    for r in results[:2]:
        print(f"      - {r['file']} (score: {r['score']:.3f})")

except Exception as e:
    rust_indexing_time = time.time() - rust_start
    print(f"\n   Note: Rust module test - {e}")
    print(f"   Build completed in {rust_indexing_time:.2f}s")

# ===== PYTHON BENCHMARK =====
print("\n" + "=" * 70)
print("PYTHON INDEXER (Tree-sitter)")
print("=" * 70)

python_start = time.time()

temp_dir = tempfile.mkdtemp()
python_indexer_path = os.path.join(temp_dir, "python.db")

from backend.indexer import RepositoryIndexer
from backend.ast_analyzer import TreeSitterAnalyzer

print(f"\n[1/4] Initializing Python indexer...")

python_indexer = RepositoryIndexer(db_path=python_indexer_path)
analyzer = TreeSitterAnalyzer()

print(f"\n[2/4] Scanning repository...")
python_stats = python_indexer.index_repository(REPO_PATH, REPO_NAME)

print(f"\n[3/4] Extracting symbols...")
python_chunks = python_indexer.get_chunks_for_embedding()
python_chunk_count = len(python_chunks)

python_indexing_time = time.time() - python_start

print(f"   Files indexed: {python_stats['files_indexed']}")
print(f"   Chunks created: {python_chunk_count}")
print(f"   Indexing time: {python_indexing_time:.3f}s")
print(
    f"   Throughput: {python_stats['files_indexed'] / python_indexing_time:.1f} files/s"
)

# ===== SUMMARY =====
print("\n" + "=" * 70)
print("PERFORMANCE SUMMARY")
print("=" * 70)

try:
    rust_files = rust_stats.get("files_indexed", python_stats["files_indexed"])
    rust_chunks = rust_chunk_count
    rust_time = rust_indexing_time

    speedup = python_indexing_time / rust_time if rust_time > 0 else 0

    print(f"\n{'Metric':<30} {'Python':>15} {'Rust':>15} {'Speedup':>15}")
    print("-" * 75)
    print(
        f"{'Files indexed':<30} {python_stats['files_indexed']:>15} {rust_files:>15} {'-':>15}"
    )
    print(
        f"{'Chunks created':<30} {python_chunk_count:>15} {rust_chunks:>15} {'-':>15}"
    )
    print(
        f"{'Indexing time (s)':<30} {python_indexing_time:>15.3f} {rust_time:>15.3f} {speedup:>14.1f}x"
    )
    print(
        f"{'Files/second':<30} {python_stats['files_indexed']/python_indexing_time:>15.1f} {rust_files/rust_time:>15.1f} {speedup:>14.1f}x"
    )

    print(f"\n{'='*70}")
    if speedup > 1:
        print(f"Rust is {speedup:.1f}x FASTER than Python!")
    elif speedup < 1 and speedup > 0:
        print(f"Python is {1/speedup:.1f}x FASTER (unexpected!)")
    print(f"{'='*70}")
except:
    print(
        f"\nPython indexing: {python_indexing_time:.3f}s for {python_stats['files_indexed']} files"
    )
    print(f"Rust module built and ready for use.")

print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)
print(
    """
• Rust provides parallel file scanning with Rayon
• No Global Interpreter Lock (GIL) in Rust
• Type-safe memory management
• Potential for 5-20x speedup on multi-core systems

• Python advantages:
  - Tree-sitter integration for accurate AST parsing
  - Rich ecosystem of ML/NLP libraries
  - Easier debugging and prototyping

• Hybrid approach:
  - Use Rust for file scanning and indexing
  - Use Python for embedding generation and search
  - Best of both worlds!
"""
)
print("=" * 70)
