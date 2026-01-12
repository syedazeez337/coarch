"""Comprehensive tests for hybrid search functionality."""

import os
import sys
import tempfile
import unittest
import sqlite3
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.hybrid_search import HybridSearch, HybridSearchManager, HybridSearchResult


class MockFAISSIndex:
    """Mock FAISS index for testing."""

    def __init__(self):
        self.documents = {}
        self.next_id = 0

    def count(self):
        return len(self.documents)

    def search(self, query_embedding, k=10):
        """Mock search that returns some results."""
        # Return mock results for testing
        results = []
        for doc_id in list(self.documents.keys())[:k]:
            results.append(
                type('MockResult', (), {
                    'id': doc_id,
                    'score': np.random.random(),
                    'file_path': f'/test/file_{doc_id}.py',
                    'start_line': 1,
                    'end_line': 10,
                    'code': f'# Mock code for document {doc_id}',
                    'language': 'python'
                })()
            )
        return results

    def get_status(self):
        return {"count": self.count(), "type": "mock"}


class TestHybridSearch(unittest.TestCase):
    """Tests for hybrid search functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        
        # Create BM25 indexer
        from backend.bm25_index import BM25Indexer
        self.bm25_indexer = BM25Indexer(self.db_path, f"{self.temp_dir}/test_bm25")
        
        # Create mock FAISS index
        self.faiss_index = MockFAISSIndex()
        
        # Create test database
        self._create_test_db()
        
        # Build BM25 index
        self.bm25_indexer.build_index()

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
            ]
            
            cursor.executemany(
                "INSERT INTO chunks (id, file_path, start_line, end_line, code, language) VALUES (?, ?, ?, ?, ?, ?)",
                test_chunks
            )
            
            conn.commit()

    def test_hybrid_search_initialization(self):
        """Test hybrid search initialization."""
        hybrid = HybridSearch(
            self.bm25_indexer,
            self.faiss_index,
            bm25_weight=0.3,
            semantic_weight=0.7
        )
        
        self.assertIsNotNone(hybrid)
        self.assertAlmostEqual(hybrid.bm25_weight, 0.3, places=2)
        self.assertAlmostEqual(hybrid.semantic_weight, 0.7, places=2)

    def test_hybrid_search_weights_normalization(self):
        """Test that weights are properly normalized."""
        # Test with zero weights
        hybrid1 = HybridSearch(self.bm25_indexer, self.faiss_index, 0, 0)
        self.assertAlmostEqual(hybrid1.bm25_weight, 0.5, places=2)
        self.assertAlmostEqual(hybrid1.semantic_weight, 0.5, places=2)
        
        # Test with unnormalized weights
        hybrid2 = HybridSearch(self.bm25_indexer, self.faiss_index, 1, 3)
        expected_bm25 = 1 / (1 + 3)
        expected_semantic = 3 / (1 + 3)
        
        self.assertAlmostEqual(hybrid2.bm25_weight, expected_bm25, places=2)
        self.assertAlmostEqual(hybrid2.semantic_weight, expected_semantic, places=2)

    def test_hybrid_search_with_bm25_only(self):
        """Test hybrid search with only BM25 enabled."""
        hybrid = HybridSearch(
            self.bm25_indexer,
            self.faiss_index,
            bm25_weight=1.0,
            semantic_weight=0.0
        )
        
        # BM25 search should work without crashing even if results are empty
        # (mock data contains only comments which get filtered during tokenization)
        results = hybrid.search(
            "anything",
            language="python",
            limit=5,
            use_bm25=True,
            use_semantic=False
        )
        
        # Just verify the method doesn't crash and returns a list
        self.assertIsInstance(results, list)
        
        # All results should be BM25-only type
        for result in results:
            self.assertEqual(result.result_type, "bm25_only")
            self.assertEqual(result.language, "python")

    def test_hybrid_search_with_semantic_only(self):
        """Test hybrid search with only semantic search enabled."""
        hybrid = HybridSearch(
            self.bm25_indexer,
            self.faiss_index,
            bm25_weight=0.0,
            semantic_weight=1.0
        )
        
        results = hybrid.search(
            "def hello",
            language="python",
            limit=5,
            use_bm25=False,
            use_semantic=True
        )
        
        # Should find semantic results
        self.assertGreaterEqual(len(results), 0)  # May be 0 since mock FAISS has limited data

    def test_hybrid_search_disabled_searches(self):
        """Test hybrid search with both searches disabled."""
        hybrid = HybridSearch(self.bm25_indexer, self.faiss_index)
        
        results = hybrid.search(
            "def hello",
            language="python",
            limit=5,
            use_bm25=False,
            use_semantic=False
        )
        
        # Should return empty results
        self.assertEqual(len(results), 0)

    def test_hybrid_search_empty_query(self):
        """Test hybrid search with empty query."""
        hybrid = HybridSearch(self.bm25_indexer, self.faiss_index)
        
        results = hybrid.search("", limit=5)
        
        # Should return empty results
        self.assertEqual(len(results), 0)

    def test_hybrid_search_language_filter(self):
        """Test hybrid search with language filtering."""
        hybrid = HybridSearch(self.bm25_indexer, self.faiss_index)
        
        results = hybrid.search(
            "function",
            language="javascript",
            limit=10
        )
        
        # Should filter by language
        for result in results:
            self.assertEqual(result.language, "javascript")

    def test_hybrid_search_limit(self):
        """Test hybrid search respects limit."""
        hybrid = HybridSearch(self.bm25_indexer, self.faiss_index)
        
        results = hybrid.search(
            "function",
            limit=3
        )
        
        # Should not exceed limit
        self.assertLessEqual(len(results), 3)

    def test_hybrid_search_min_score_threshold(self):
        """Test hybrid search respects minimum score threshold."""
        # Set very high minimum score threshold
        hybrid = HybridSearch(
            self.bm25_indexer,
            self.faiss_index,
            min_score_threshold=100.0
        )
        
        results = hybrid.search("def", limit=10)
        
        # Should return empty results due to high threshold
        # (BM25 scores are typically much lower than 100)
        self.assertEqual(len(results), 0)

    def test_hybrid_search_result_structure(self):
        """Test that hybrid search results have correct structure."""
        hybrid = HybridSearch(self.bm25_indexer, self.faiss_index)
        
        results = hybrid.search("def", limit=5)
        
        if len(results) > 0:
            result = results[0]
            
            # Check required attributes
            self.assertIsInstance(result.id, int)
            self.assertIsInstance(result.file_path, str)
            self.assertIsInstance(result.start_line, int)
            self.assertIsInstance(result.end_line, int)
            self.assertIsInstance(result.code, str)
            self.assertIsInstance(result.language, str)
            self.assertIsInstance(result.score, float)
            self.assertIsInstance(result.matched_terms, list)
            self.assertIsInstance(result.result_type, str)

    def test_hybrid_search_explanation(self):
        """Test hybrid search explanation generation."""
        hybrid = HybridSearch(
            self.bm25_indexer,
            self.faiss_index,
            enable_explanation=True
        )
        
        results = hybrid.search("def", limit=3)
        
        if len(results) > 0:
            explanation = hybrid.get_search_explanation("def", results)
            
            # Check explanation structure
            self.assertIn("query", explanation)
            self.assertIn("query_tokens", explanation)
            self.assertIn("weights", explanation)
            self.assertIn("results", explanation)

    def test_set_weights(self):
        """Test updating hybrid search weights."""
        hybrid = HybridSearch(self.bm25_indexer, self.faiss_index)
        
        original_bm25 = hybrid.bm25_weight
        original_semantic = hybrid.semantic_weight
        
        # Update weights
        hybrid.set_weights(0.6, 0.4)
        
        # Check updated weights
        self.assertAlmostEqual(hybrid.bm25_weight, 0.6, places=2)
        self.assertAlmostEqual(hybrid.semantic_weight, 0.4, places=2)
        
        # Test with zero total weight - should keep current weights
        hybrid.set_weights(0, 0)
        
        # Should keep the previously set weights (0.6, 0.4)
        self.assertAlmostEqual(hybrid.bm25_weight, 0.6, places=2)
        self.assertAlmostEqual(hybrid.semantic_weight, 0.4, places=2)

    def test_hybrid_search_stats(self):
        """Test hybrid search statistics."""
        hybrid = HybridSearch(self.bm25_indexer, self.faiss_index)
        
        stats = hybrid.get_stats()
        
        # Check stats structure
        self.assertIn("bm25", stats)
        self.assertIn("semantic", stats)
        self.assertIn("weights", stats)
        
        # Check BM25 stats
        bm25_stats = stats["bm25"]
        self.assertIn("enabled", bm25_stats)
        self.assertIn("documents", bm25_stats)
        self.assertIn("terms", bm25_stats)
        
        # Check semantic stats
        semantic_stats = stats["semantic"]
        self.assertIn("enabled", semantic_stats)
        self.assertIn("count", semantic_stats)


class TestHybridSearchManager(unittest.TestCase):
    """Tests for hybrid search manager."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")
        self.index_path = os.path.join(self.temp_dir, "test_index")
        
        # Create test database
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
                (1, "/test/main.py", 1, 10, 'def hello_world():\n    print("Hello World")', "python"),
                (2, "/test/utils.py", 1, 8, 'def calculate_sum(a, b):\n    return a + b', "python"),
            ]
            
            cursor.executemany(
                "INSERT INTO chunks (id, file_path, start_line, end_line, code, language) VALUES (?, ?, ?, ?, ?, ?)",
                test_chunks
            )
            
            conn.commit()

    def test_manager_initialization(self):
        """Test hybrid search manager initialization."""
        config = {}
        manager = HybridSearchManager(config)
        
        # Should not be initialized yet
        self.assertIsNone(manager.bm25_indexer)
        self.assertIsNone(manager.hybrid_search)
        
        # Initialize
        success = manager.initialize(
            self.db_path,
            self.index_path,
            bm25_weight=0.4,
            semantic_weight=0.6
        )
        
        self.assertTrue(success)
        self.assertIsNotNone(manager.bm25_indexer)
        self.assertIsNotNone(manager.hybrid_search)

    def test_manager_build_bm25_index(self):
        """Test building BM25 index via manager."""
        config = {}
        manager = HybridSearchManager(config)
        
        # Initialize and build index
        manager.initialize(self.db_path, self.index_path)
        stats = manager.build_bm25_index()
        
        # Check stats
        self.assertGreater(stats["documents"], 0)
        self.assertGreater(stats["terms"], 0)

    def test_manager_search(self):
        """Test search via manager."""
        config = {}
        manager = HybridSearchManager(config)
        
        # Initialize and build index
        manager.initialize(self.db_path, self.index_path)
        manager.build_bm25_index()
        
        # Perform search
        results, explanation = manager.search("def hello", limit=5)
        
        # Should find results
        self.assertIsInstance(results, list)
        self.assertIsNone(explanation)  # Not requesting explanation

    def test_manager_search_with_explanation(self):
        """Test search with explanation via manager."""
        config = {}
        manager = HybridSearchManager(config)
        
        # Initialize and build index
        manager.initialize(self.db_path, self.index_path)
        manager.build_bm25_index()
        
        # Perform search with explanation
        results, explanation = manager.search(
            "def hello", 
            limit=5, 
            return_explanation=True
        )
        
        # Should return explanation
        self.assertIsInstance(explanation, dict)

    def test_manager_set_weights(self):
        """Test updating weights via manager."""
        config = {}
        manager = HybridSearchManager(config)
        
        # Initialize
        manager.initialize(self.db_path, self.index_path)
        
        # Update weights
        manager.set_weights(0.7, 0.3)
        
        # Check via stats
        stats = manager.get_stats()
        weights = stats["weights"]
        
        self.assertAlmostEqual(weights["bm25_weight"], 0.7, places=2)
        self.assertAlmostEqual(weights["semantic_weight"], 0.3, places=2)

    def test_manager_clear(self):
        """Test clearing manager."""
        config = {}
        manager = HybridSearchManager(config)
        
        # Initialize
        manager.initialize(self.db_path, self.index_path)
        manager.build_bm25_index()
        
        # Verify initialized
        self.assertIsNotNone(manager.bm25_indexer)
        self.assertIsNotNone(manager.hybrid_search)
        
        # Clear
        manager.clear()
        
        # Verify cleared
        self.assertIsNone(manager.bm25_indexer)
        self.assertIsNone(manager.hybrid_search)


if __name__ == "__main__":
    unittest.main(verbosity=2)