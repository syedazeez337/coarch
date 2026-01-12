"""Coarch performance benchmark on a repository."""

import sys
import os
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.indexer import RepositoryIndexer
from backend.faiss_index import FaissIndex
from backend.ast_analyzer import TreeSitterAnalyzer
import tempfile

print("=" * 70)
print("Coarch Performance Benchmark")
print("=" * 70)

REPO_PATH = "C:/Users/Azeez/Myproject/tree-sitter-python"

if not os.path.exists(REPO_PATH):
    print(f"\nRepository not found: {REPO_PATH}")
    print("Using current directory instead.")
    REPO_PATH = "C:/Users/Azeez/Myproject/coarch"

print(f"\nRepository: {REPO_PATH}")
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

total_start = time.time()

temp_dir = tempfile.mkdtemp()
INDEX_PATH = os.path.join(temp_dir, "coarch_index")
DB_PATH = os.path.join(temp_dir, "coarch.db")

print(f"\n[1/4] Initializing Coarch...")
indexer = RepositoryIndexer(db_path=DB_PATH)
faiss = FaissIndex(index_path=INDEX_PATH)
analyzer = TreeSitterAnalyzer()

print(f"   Index path: {INDEX_PATH}")
print(f"   Database: {DB_PATH}")

print(f"\n[2/4] Scanning repository...")
scan_start = time.time()
stats = indexer.index_repository(REPO_PATH, "benchmark_repo")
scan_time = time.time() - scan_start

print(f"   Files indexed: {stats['files_indexed']}")
print(f"   Chunks created: {stats['chunks_created']}")
print(f"   Scan time: {scan_time:.2f}s")

if stats['files_indexed'] == 0:
    print("\nNo files indexed. Check repository path and permissions.")
    sys.exit(1)

print(f"\n[3/4] Extracting code symbols...")
symbol_start = time.time()

chunks = indexer.get_chunks_for_embedding()
symbols_found = {}
files_by_lang = {}

for chunk in chunks:
    lang = chunk.language
    files_by_lang[lang] = files_by_lang.get(lang, 0) + 1
    
    try:
        symbols = analyzer.extract_symbols(chunk.code, lang)
        for sym in symbols:
            sym_type = sym.type
            if sym_type not in symbols_found:
                symbols_found[sym_type] = 0
            symbols_found[sym_type] += 1
    except Exception as e:
        pass

symbol_time = time.time() - symbol_start

print(f"   Languages found: {list(files_by_lang.keys())}")
for lang, count in files_by_lang.items():
    print(f"      {lang}: {count} chunks")

print(f"   Symbols extracted: {sum(symbols_found.values())}")
for sym_type, count in list(symbols_found.items())[:5]:
    print(f"      {sym_type}: {count}")

print(f"   Symbol extraction time: {symbol_time:.2f}s")

print(f"\n[4/4] Performance Summary...")
total_time = time.time() - total_start

print(f"\n   {'Metric':<30} {'Value':>15}")
print(f"   {'-'*45}")
print(f"   {'Files scanned':<30} {stats['files_indexed']:>15,}")
print(f"   {'Code chunks':<30} {stats['chunks_created']:>15,}")
print(f"   {'Symbols extracted':<30} {sum(symbols_found.values()):>15,}")
print(f"   {'Total time':<30} {total_time:>14.2f}s")
print(f"   {'Files/second':<30} {stats['files_indexed']/total_time:>14.1f}")

if stats['chunks_created'] > 0:
    print(f"   {'Chunks/second':<30} {stats['chunks_created']/total_time:>14.1f}")

print(f"\n   Database stats:")
db_stats = indexer.get_stats()
print(f"      Total chunks: {db_stats.get('total_chunks', 0)}")
print(f"      Total repos: {db_stats.get('total_repos', 0)}")

print("\n" + "=" * 70)
print("Benchmark Complete!")
print("=" * 70)
print(f"\nTo search this repository:")
print(f"   1. Start server: coarch serve")
print(f"   2. Search: curl -X POST http://localhost:8000/search \\")
print(f'             -d \'{{"query": "your search query"}}\'')
print("=" * 70)
