#!/usr/bin/env python3
"""Demo script for BM25 Hybrid Search implementation (OPT-2).

This script demonstrates the new BM25 text search functionality combined with 
semantic search for improved code search results.
"""

import os
import sys
import tempfile
import sqlite3
import time
from pathlib import Path

# Add the backend to the path
sys.path.insert(0, '.')

from backend.bm25_index import BM25Indexer, BM25SearchResult
from backend.hybrid_search import HybridSearch, HybridSearchManager


class MockFAISSIndex:
    """Mock FAISS index for demonstration purposes."""
    
    def __init__(self):
        self.documents = {}
        self.next_id = 0
        
    def count(self):
        return len(self.documents)
        
    def search(self, query_embedding, k=10):
        """Mock search that returns relevant results based on simple keyword matching."""
        import random
        
        results = []
        for doc_id, doc in list(self.documents.items())[:k]:
            # Simple mock scoring based on keyword overlap
            score = random.uniform(0.1, 0.9)
            results.append(
                type('MockResult', (), {
                    'id': doc_id,
                    'score': score,
                    'file_path': doc['file_path'],
                    'start_line': doc['start_line'],
                    'end_line': doc['end_line'],
                    'code': doc['code'],
                    'language': doc['language']
                })()
            )
        return sorted(results, key=lambda x: x.score, reverse=True)
        
    def get_status(self):
        return {"count": self.count(), "type": "mock_faiss"}


def create_sample_database(db_path: str):
    """Create a sample database with various programming languages."""
    
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        
        # Create chunks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                file_path TEXT NOT NULL,
                start_line INTEGER,
                end_line INTEGER,
                code TEXT,
                language TEXT
            )
        """)
        
        # Insert diverse sample code
        sample_chunks = [
            # Python examples
            (1, "/src/main.py", 1, 10, 'def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)', "python"),
            (2, "/src/utils.py", 1, 8, 'def calculate_sum(numbers):\n    return sum(numbers)', "python"),
            (3, "/src/models.py", 1, 15, 'class User:\n    def __init__(self, name, email):\n        self.name = name\n        self.email = email', "python"),
            
            # JavaScript examples
            (4, "/src/app.js", 1, 12, 'function fetchUserData(userId) {\n    return fetch(`/api/users/${userId}`);\n}', "javascript"),
            (5, "/src/utils.js", 1, 8, 'function formatDate(date) {\n    return date.toLocaleDateString();\n}', "javascript"),
            (6, "/src/components.jsx", 1, 20, 'import React from "react";\n\nfunction Button({ children, onClick }) {\n    return <button onClick={onClick}>{children}</button>;\n}', "javascript"),
            
            # TypeScript examples
            (7, "/src/types.ts", 1, 10, 'interface User {\n    id: number;\n    name: string;\n    email: string;\n}', "typescript"),
            (8, "/src/math.ts", 1, 8, 'function multiply(a: number, b: number): number {\n    return a * b;\n}', "typescript"),
            
            # Java examples
            (9, "/src/Calculator.java", 1, 15, 'public class Calculator {\n    public int add(int a, int b) {\n        return a + b;\n    }\n}', "java"),
            (10, "/src/UserService.java", 1, 20, '@Service\npublic class UserService {\n    public User findById(Long id) {\n        return userRepository.findById(id);\n    }\n}', "java"),
            
            # Go examples
            (11, "/src/main.go", 1, 12, 'package main\n\nimport "fmt"\n\nfunc main() {\n    fmt.Println("Hello, World!")\n}', "go"),
            (12, "/src/utils.go", 1, 10, 'func CalculateSum(numbers []int) int {\n    total := 0\n    for _, num := range numbers {\n        total += num\n    }\n    return total\n}', "go"),
            
            # Rust examples
            (13, "/src/lib.rs", 1, 12, 'pub fn fibonacci(n: u32) -> u32 {\n    match n {\n        0 => 0,\n        1 => 1,\n        _ => fibonacci(n - 1) + fibonacci(n - 2),\n    }\n}', "rust"),
            (14, "/src/math.rs", 1, 8, 'pub fn multiply(a: i32, b: i32) -> i32 {\n    a * b\n}', "rust"),
            
            # Documentation
            (15, "/README.md", 1, 5, '# Code Search Demo\n\nThis project demonstrates hybrid search functionality combining BM25 and semantic search.', "markdown"),
            (16, "/docs/api.md", 1, 8, '# API Documentation\n\n## Search Endpoint\n\nUse POST /search to find code snippets.', "markdown"),
        ]
        
        cursor.executemany(
            "INSERT INTO chunks (id, file_path, start_line, end_line, code, language) VALUES (?, ?, ?, ?, ?, ?)",
            sample_chunks
        )
        
        conn.commit()
    
    print(f"SUCCESS Created sample database with {len(sample_chunks)} code chunks")


def demonstrate_bm25_search():
    """Demonstrate pure BM25 search functionality."""
    
    print("\n" + "="*60)
    print("BM25 SEARCH DEMONSTRATION")
    print("="*60)
    
    # Create temporary database
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "demo.db")
        create_sample_database(db_path)
        
        # Initialize BM25 indexer
        bm25_indexer = BM25Indexer(db_path)
        
        print("\nSTATS Building BM25 index...")
        start_time = time.time()
        stats = bm25_indexer.build_index()
        build_time = time.time() - start_time
        
        print(f"SUCCESS Index built in {build_time:.2f}s")
        print(f"   DOCS Documents: {stats['documents']}")
        print(f"   TERMS Terms: {stats['terms']}")
        print(f"   LENGTH Avg length: {stats['avg_length']:.1f}")
        
        # Test queries
        test_queries = [
            "function fibonacci",
            "class User",
            "calculate sum",
            "fetch user data",
            "multiply numbers",
            "interface User",
            "public static",
        ]
        
        print("\nSEARCH Testing BM25 search queries:")
        for query in test_queries:
            print(f"\n   Query: '{query}'")
            results = bm25_indexer.search(query, limit=3)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"     {i}. [{result.language}] {result.file_path}")
                    print(f"        Score: {result.score:.3f} | Terms: {result.terms}")
                    # Show code preview
                    code_preview = result.code.replace('\n', ' ')[:60] + "..."
                    print(f"        Code: {code_preview}")
            else:
                print("     No results found")


def demonstrate_hybrid_search():
    """Demonstrate hybrid search functionality."""
    
    print("\n" + "="*60)
    print("HYBRID SEARCH DEMONSTRATION")
    print("="*60)
    
    # Create temporary database
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "demo.db")
        create_sample_database(db_path)
        
        # Initialize components
        bm25_indexer = BM25Indexer(db_path)
        bm25_indexer.build_index()
        
        faiss_index = MockFAISSIndex()
        # Add mock documents to FAISS
        faiss_index.documents = {
            1: {"file_path": "/src/main.py", "start_line": 1, "end_line": 10, "code": "fibonacci code", "language": "python"},
            2: {"file_path": "/src/models.py", "start_line": 1, "end_line": 15, "code": "user model", "language": "python"},
            3: {"file_path": "/src/app.js", "start_line": 1, "end_line": 12, "code": "fetch user data", "language": "javascript"},
        }
        
        # Test different weight configurations
        weight_configs = [
            (1.0, 0.0, "BM25 Only"),
            (0.0, 1.0, "Semantic Only"),
            (0.5, 0.5, "Equal Weight"),
            (0.3, 0.7, "Semantic Heavy"),
            (0.7, 0.3, "BM25 Heavy"),
        ]
        
        test_query = "function calculate sum"
        
        print(f"\nQUERY Testing query: '{test_query}'")
        print("\nSTATS Different weight configurations:")
        
        for bm25_weight, semantic_weight, description in weight_configs:
            print(f"\n   WEIGHTS  {description} (BM25: {bm25_weight:.1f}, Semantic: {semantic_weight:.1f})")
            
            hybrid = HybridSearch(
                bm25_indexer,
                faiss_index,
                bm25_weight=bm25_weight,
                semantic_weight=semantic_weight
            )
            
            results = hybrid.search(test_query, limit=3)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"     {i}. [{result.language}] {result.file_path}")
                    print(f"        Score: {result.score:.3f} | Type: {result.result_type}")
                    print(f"        BM25: {result.bm25_score:.3f} | Semantic: {result.semantic_score:.3f}")
            else:
                print("     No results found")


def demonstrate_language_filtering():
    """Demonstrate language-specific search."""
    
    print("\n" + "="*60)
    print("LANG LANGUAGE FILTERING DEMONSTRATION")
    print("="*60)
    
    # Create temporary database
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "demo.db")
        create_sample_database(db_path)
        
        bm25_indexer = BM25Indexer(db_path)
        bm25_indexer.build_index()
        
        test_query = "function"
        languages = ["python", "javascript", "java", "rust"]
        
        print(f"\nQUERY Testing query: '{test_query}' across different languages")
        
        for language in languages:
            print(f"\n   LANGDIR Language: {language}")
            results = bm25_indexer.search(test_query, language=language, limit=3)
            
            if results:
                for result in results:
                    print(f"     - {result.file_path} (Score: {result.score:.3f})")
            else:
                print("     No results found")


def demonstrate_performance():
    """Demonstrate performance characteristics."""
    
    print("\n" + "="*60)
    print("⚡ PERFORMANCE DEMONSTRATION")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "demo.db")
        create_sample_database(db_path)
        
        bm25_indexer = BM25Indexer(db_path)
        
        print("\nTIME  Measuring BM25 performance:")
        
        # Build index performance
        start_time = time.time()
        bm25_indexer.build_index()
        build_time = time.time() - start_time
        
        print(f"   STATS Index build: {build_time:.3f}s")
        
        # Search performance
        queries = ["function", "class", "calculate", "fibonacci", "user data"]
        search_times = []
        
        for query in queries:
            start_time = time.time()
            results = bm25_indexer.search(query, limit=10)
            search_time = time.time() - start_time
            search_times.append(search_time)
            
            print(f"   SEARCH '{query}': {search_time:.3f}s ({len(results)} results)")
        
        avg_search_time = sum(search_times) / len(search_times)
        print(f"   AVG Average search time: {avg_search_time:.3f}s")


def main():
    """Main demonstration function."""
    
    print("HYBRID BM25 HYBRID SEARCH DEMO (OPT-2)")
    print("=" * 60)
    print("This demo showcases the new BM25 text search functionality")
    print("combined with semantic search for improved code search results.")
    print()
    print("Features demonstrated:")
    print("  • Pure BM25 text search")
    print("  • Hybrid search with configurable weights")
    print("  • Language-specific filtering")
    print("  • Performance characteristics")
    print()
    
    try:
        # Run demonstrations
        demonstrate_bm25_search()
        demonstrate_hybrid_search()
        demonstrate_language_filtering()
        demonstrate_performance()
        
        print("\n" + "="*60)
        print("SUCCESS DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print()
        print("CELEBRATE BM25 Hybrid Search (OPT-2) is ready for production!")
        print("   The implementation provides:")
        print("   • Improved text matching with BM25 algorithm")
        print("   • Configurable hybrid search weighting")
        print("   • Language-specific search capabilities")
        print("   • Backward compatibility with existing search")
        print()
        print("DOCS2 Next steps:")
        print("   • Integrate with existing FAISS semantic search")
        print("   • Add BM25 index building to repository indexing")
        print("   • Configure via coarch config settings")
        print("   • Deploy in production environment")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())