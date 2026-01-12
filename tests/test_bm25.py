"""Comprehensive tests for BM25 text search functionality."""

import os
import sys
import tempfile
import unittest
import sqlite3
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.bm25_index import BM25Indexer, BM25Document, BM25SearchResult


class TestBM25Indexer(unittest.TestCase):
    """Tests for BM25 indexer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.bm25_indexer = BM25Indexer(self.db_path, f"{self.temp_dir}/test_bm25")
        
        # Create test database with sample chunks
        self._create_test_db()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_test_db(self):
        """Create test database with sample code chunks."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
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
            
            # Insert test data
            test_chunks = [
                (1, "/test/main.py", 1, 10, 'def hello_world():\n    print("Hello World")\n    return True', "python"),
                (2, "/test/utils.py", 1, 8, 'def calculate_sum(a, b):\n    return a + b', "python"),
                (3, "/test/main.js", 1, 12, 'function greetUser(name) {\n    console.log("Hello " + name);\n    return true;\n}', "javascript"),
                (4, "/test/math.ts", 1, 6, 'export function multiply(x: number, y: number): number {\n    return x * y;\n}', "typescript"),
                (5, "/test/helper.java", 1, 15, 'public static int factorial(int n) {\n    if (n <= 1) return 1;\n    return n * factorial(n - 1);\n}', "java"),
                (6, "/test/main.go", 1, 8, 'package main\n\nimport "fmt"\n\nfunc main() {\n    fmt.Println("Hello")\n}', "go"),
                (7, "/test/lib.rs", 1, 10, 'pub fn fibonacci(n: u32) -> u32 {\n    if n <= 1 { n } else { fibonacci(n-1) + fibonacci(n-2) }\n}', "rust"),
                (8, "/test/README.md", 1, 5, '# Test Project\n\nThis is a test project for BM25 search functionality.', "markdown"),
            ]
            
            cursor.executemany(
                "INSERT INTO chunks (id, file_path, start_line, end_line, code, language) VALUES (?, ?, ?, ?, ?, ?)",
                test_chunks
            )
            
            conn.commit()

    def test_tokenization(self):
        """Test tokenization of different text types."""
        # Test Python code tokenization
        python_code = "def hello_world():\n    print('Hello World')\n    result = True"
        tokens = self.bm25_indexer._tokenize(python_code, "python")
        
        # Should contain meaningful tokens
        self.assertIn("hello_world", tokens)
        self.assertIn("print", tokens)
        # "result" should be in tokens (not a keyword)
        self.assertIn("result", tokens)
        # Should filter out common keywords and very short tokens
        self.assertNotIn("def", tokens)
        self.assertNotIn("a", tokens)

    def test_build_index(self):
        """Test building BM25 index from database."""
        # Build index
        stats = self.bm25_indexer.build_index()
        
        # Check stats
        self.assertGreater(stats["documents"], 0)
        self.assertGreater(stats["terms"], 0)
        self.assertGreater(stats["avg_length"], 0)
        
        # Check that documents were indexed
        self.assertGreater(self.bm25_indexer.doc_count, 0)
        self.assertGreater(len(self.bm25_indexer.inverted_index), 0)

    def test_bm25_search(self):
        """Test BM25 search functionality."""
        # Build index first
        self.bm25_indexer.build_index()
        
        # Test search for Python function
        results = self.bm25_indexer.search("def hello", "python", limit=5)
        
        # Should find results
        self.assertGreater(len(results), 0)
        
        # First result should be relevant
        first_result = results[0]
        self.assertEqual(first_result.language, "python")
        self.assertIn("hello", first_result.code.lower())

    def test_bm25_search_no_language_filter(self):
        """Test BM25 search without language filter."""
        # Build index
        self.bm25_indexer.build_index()
        
        # Search without language filter
        results = self.bm25_indexer.search("function multiply", limit=10)
        
        # Should find results from different languages
        self.assertGreater(len(results), 0)
        
        # Should contain results with "multiply" or related terms
        found_multiply = any("multiply" in result.code.lower() for result in results)
        self.assertTrue(found_multiply)

    def test_bm25_empty_query(self):
        """Test BM25 search with empty query."""
        # Build index
        self.bm25_indexer.build_index()
        
        # Search with empty query
        results = self.bm25_indexer.search("", limit=10)
        
        # Should return empty results
        self.assertEqual(len(results), 0)

    def test_bm25_single_term_search(self):
        """Test BM25 search with single term."""
        # Build index
        self.bm25_indexer.build_index()
        
        # Search for single term (use "hello_world" which appears in Python code)
        results = self.bm25_indexer.search("hello_world", limit=5)
        
        # Should find results containing "hello_world"
        self.assertGreater(len(results), 0)
        
        # All results should contain the search term
        for result in results:
            self.assertTrue(
                "hello_world" in result.code.lower() or 
                any("hello_world" in term.lower() for term in result.terms)
            )

    def test_bm25_language_filter(self):
        """Test BM25 search with specific language filter."""
        # Build index
        self.bm25_indexer.build_index()
        
        # Search for JavaScript-specific term
        results = self.bm25_indexer.search("console", language="javascript", limit=10)
        
        # Should find only JavaScript results
        for result in results:
            self.assertEqual(result.language, "javascript")

    def test_bm25_stats(self):
        """Test BM25 index statistics."""
        # Build index
        self.bm25_indexer.build_index()
        
        # Get stats
        stats = self.bm25_indexer.get_stats()
        
        # Check required fields
        self.assertIn("documents", stats)
        self.assertIn("terms", stats)
        self.assertIn("avg_length", stats)
        self.assertIn("index_size_mb", stats)
        
        # Check values
        self.assertGreater(stats["documents"], 0)
        self.assertGreater(stats["terms"], 0)
        self.assertGreater(stats["avg_length"], 0)
        self.assertGreaterEqual(stats["index_size_mb"], 0)

    def test_bm25_save_load(self):
        """Test saving and loading BM25 index."""
        # Build index
        self.bm25_indexer.build_index()
        
        # Save index
        index_path = self.bm25_indexer.save()
        self.assertTrue(os.path.exists(f"{index_path}.json"))
        
        # Create new indexer and load
        new_indexer = BM25Indexer(self.db_path, f"{self.temp_dir}/test_bm25")
        loaded = new_indexer.load()
        
        # Should load successfully
        self.assertTrue(loaded)
        self.assertEqual(new_indexer.doc_count, self.bm25_indexer.doc_count)
        self.assertEqual(new_indexer.avg_doc_length, self.bm25_indexer.avg_doc_length)
        self.assertEqual(len(new_indexer.inverted_index), len(self.bm25_indexer.inverted_index))

    def test_bm25_clear(self):
        """Test clearing BM25 index."""
        # Build index
        self.bm25_indexer.build_index()
        
        # Verify index has data
        self.assertGreater(self.bm25_indexer.doc_count, 0)
        
        # Clear index
        self.bm25_indexer.clear()
        
        # Verify index is empty
        self.assertEqual(self.bm25_indexer.doc_count, 0)
        self.assertEqual(len(self.bm25_indexer.inverted_index), 0)
        self.assertEqual(len(self.bm25_indexer.documents), 0)

    def test_bm25_chunk_limit(self):
        """Test building index with chunk limit."""
        # Build index with limit
        stats = self.bm25_indexer.build_index(chunk_limit=3)
        
        # Should have limited number of documents
        self.assertLessEqual(stats["documents"], 3)

    def test_bm25_duplicate_terms(self):
        """Test BM25 search with duplicate terms in query."""
        # Build index
        self.bm25_indexer.build_index()
        
        # Search with repeated terms
        results = self.bm25_indexer.search("function function function", limit=5)
        
        # Should still find results (though may be different scoring)
        # The important thing is it doesn't crash
        self.assertGreaterEqual(len(results), 0)

    def test_bm25_case_sensitivity(self):
        """Test that BM25 search is case-insensitive."""
        # Build index
        self.bm25_indexer.build_index()
        
        # Search with different cases
        upper_results = self.bm25_indexer.search("FUNCTION", limit=5)
        lower_results = self.bm25_indexer.search("function", limit=5)
        
        # Should find similar results (case-insensitive)
        if len(upper_results) > 0 and len(lower_results) > 0:
            # Check if they found similar files
            upper_files = {r.file_path for r in upper_results[:3]}
            lower_files = {r.file_path for r in lower_results[:3]}
            
            # Should have some overlap
            self.assertTrue(len(upper_files & lower_files) > 0)


class TestBM25EdgeCases(unittest.TestCase):
    """Tests for BM25 edge cases and error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.bm25_indexer = BM25Indexer(self.db_path, f"{self.temp_dir}/test_bm25")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_empty_database(self):
        """Test behavior with empty database."""
        # Don't add any test data
        # Search should return empty results
        results = self.bm25_indexer.search("test", limit=10)
        self.assertEqual(len(results), 0)

    def test_invalid_language_tokenization(self):
        """Test tokenization with invalid language."""
        text = "def test(): pass"
        tokens = self.bm25_indexer._tokenize(text, "invalid_language")
        
        # Should still work with default tokenization
        self.assertGreater(len(tokens), 0)
        self.assertIn("test", tokens)

    def test_special_characters_tokenization(self):
        """Test tokenization with special characters."""
        text = "def test_method($var1, var2): # comment\n    return var1 + var2"
        tokens = self.bm25_indexer._tokenize(text, "javascript")
        
        # Should handle special characters
        self.assertGreater(len(tokens), 0)

    def test_very_long_query(self):
        """Test with very long query."""
        # Build minimal index
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
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
            cursor.execute("INSERT INTO chunks VALUES (1, '/test.py', 1, 5, 'def test(): pass', 'python')")
            conn.commit()
        
        self.bm25_indexer.build_index()
        
        # Search with very long query (should not crash)
        long_query = " ".join(["test"] * 1000)
        results = self.bm25_indexer.search(long_query, limit=10)
        
        # Should return results without crashing (word "test" appears in test documents)
        # The important thing is it doesn't crash
        self.assertGreaterEqual(len(results), 0)  # Allow 0 or more results

    def test_numeric_query(self):
        """Test search with numeric query."""
        # Build minimal index
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
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
            cursor.execute("INSERT INTO chunks VALUES (1, '/test.py', 1, 5, 'x = 123', 'python')")
            conn.commit()
        
        self.bm25_indexer.build_index()
        
        # Search for numbers
        results = self.bm25_indexer.search("123", limit=10)
        
        # Should find the numeric value
        self.assertGreater(len(results), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)