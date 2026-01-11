"""Index the FAISS repository itself."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.indexer import RepositoryIndexer

print("=" * 60)
print("Indexing the Coarch repository...")
print("=" * 60)

repo_path = os.path.dirname(os.path.abspath(__file__))
print(f"\nRepository: {repo_path}")

indexer = RepositoryIndexer(db_path="coarch_index.db")

print("\nIndexing files...")
stats = indexer.index_repository(repo_path, "coarch")
print(f"   Files indexed: {stats['files_indexed']}")
print(f"   Chunks created: {stats['chunks_created']}")

db_stats = indexer.get_stats()
print(f"\nTotal chunks in database: {db_stats['total_chunks']}")
print("By language:")
for lang, count in db_stats["by_language"].items():
    print(f"   - {lang}: {count}")

chunks = indexer.get_chunks_for_embedding()
print(f"\nChunks ready for embedding: {len(chunks)}")

print("\nSample chunks:")
for chunk in chunks[:5]:
    print(
        f"   [{chunk.language}] {chunk.file_path}:{chunk.start_line}-{chunk.end_line}"
    )
    print(f"   Symbols: {chunk.symbols}")
    print()

print("=" * 60)
print("Repository indexed successfully!")
print("=" * 60)
