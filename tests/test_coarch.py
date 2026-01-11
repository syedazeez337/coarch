"""Comprehensive unit tests for Coarch."""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestConfig(unittest.TestCase):
    """Tests for configuration management."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, "test_config.json")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_default_config(self):
        """Test default configuration values."""
        from backend.config import CoarchConfig

        config = CoarchConfig()

        self.assertEqual(config.index_path, "~/.coarch/index")
        self.assertEqual(config.db_path, "~/.coarch/coarch.db")
        self.assertEqual(config.model_name, "microsoft/codebert-base")
        self.assertEqual(config.max_sequence_length, 512)
        self.assertEqual(config.batch_size, 32)
        self.assertTrue(config.use_quantization)
        self.assertFalse(config.use_gpu)
        self.assertEqual(config.server_port, 8000)
        self.assertEqual(config.max_results, 20)

    def test_save_and_load_config(self):
        """Test saving and loading config."""
        from backend.config import CoarchConfig

        config = CoarchConfig(
            index_path="/custom/index",
            db_path="/custom/db",
            batch_size=64,
            server_port=9000,
        )

        config.save(self.config_path)

        loaded = CoarchConfig.load(self.config_path)

        self.assertEqual(loaded.index_path, "/custom/index")
        self.assertEqual(loaded.db_path, "/custom/db")
        self.assertEqual(loaded.batch_size, 64)
        self.assertEqual(loaded.server_port, 9000)

    def test_add_remove_indexed_repo(self):
        """Test adding and removing indexed repos."""
        from backend.config import CoarchConfig

        config = CoarchConfig()
        config.save(self.config_path)

        config.add_indexed_repo("/path/to/repo1", "repo1")
        config.add_indexed_repo("/path/to/repo2", "repo2")

        self.assertEqual(len(config.indexed_repos), 2)

        config.remove_indexed_repo("/path/to/repo1")
        self.assertEqual(len(config.indexed_repos), 1)
        self.assertEqual(config.indexed_repos[0]["name"], "repo2")

    def test_get_expanded_paths(self):
        """Test path expansion."""
        from backend.config import CoarchConfig

        config = CoarchConfig(
            index_path="~/coarch_index",
            db_path="~/coarch.db"
        )

        paths = config.get_expanded_paths()

        self.assertTrue(paths["index_path"].endswith("coarch_index"))
        self.assertTrue(paths["db_path"].endswith("coarch.db"))


class TestIndexer(unittest.TestCase):
    """Tests for repository indexer."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test.db")

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_language_detection(self):
        """Test language detection from file extensions."""
        from backend.indexer import RepositoryIndexer

        indexer = RepositoryIndexer(db_path=self.db_path)

        self.assertEqual(indexer.get_language("test.py"), "python")
        self.assertEqual(indexer.get_language("test.js"), "javascript")
        self.assertEqual(indexer.get_language("test.ts"), "typescript")
        self.assertEqual(indexer.get_language("test.java"), "java")
        self.assertEqual(indexer.get_language("test.cpp"), "cpp")
        self.assertEqual(indexer.get_language("test.go"), "go")
        self.assertEqual(indexer.get_language("test.rs"), "rust")
        self.assertEqual(indexer.get_language("test.rb"), "ruby")
        self.assertEqual(indexer.get_language("test.php"), "php")
        self.assertEqual(indexer.get_language("test.swift"), "swift")
        self.assertEqual(indexer.get_language("test.kt"), "kotlin")
        self.assertEqual(indexer.get_language("test.md"), "markdown")
        self.assertEqual(indexer.get_language("test.json"), "json")
        self.assertEqual(indexer.get_language("test.yaml"), "yaml")
        self.assertEqual(indexer.get_language("test.css"), "css")
        self.assertEqual(indexer.get_language("test.sql"), "sql")
        self.assertEqual(indexer.get_language("test.sh"), "bash")
        self.assertEqual(indexer.get_language("test.unknown"), None)

    def test_ignore_patterns(self):
        """Test ignore pattern matching."""
        from backend.indexer import RepositoryIndexer

        indexer = RepositoryIndexer(db_path=self.db_path)

        self.assertTrue(indexer.should_ignore(Path(".git/config")))
        self.assertTrue(indexer.should_ignore(Path("node_modules/package.json")))
        self.assertTrue(indexer.should_ignore(Path("__pycache__/cache.py")))
        self.assertTrue(indexer.should_ignore(Path(".venv/env.py")))
        self.assertTrue(indexer.should_ignore(Path("build/output.jar")))
        self.assertTrue(indexer.should_ignore(Path("dist/app")))

        self.assertFalse(indexer.should_ignore(Path("src/main.py")))
        self.assertFalse(indexer.should_ignore(Path("lib/utils.js")))

    def test_extract_code_chunks(self):
        """Test code chunk extraction."""
        from backend.indexer import RepositoryIndexer

        indexer = RepositoryIndexer(db_path=self.db_path)

        code = '''
def hello():
    print("Hello")

class World:
    def greet(self):
        return "Hello"

def world():
    pass
'''

        chunks = indexer.extract_code_chunks("test.py", code)

        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertEqual(chunk.language, "python")
            self.assertIn("def", chunk.code.lower())
            self.assertGreater(chunk.start_line, 0)


class TestASTAnalyzer(unittest.TestCase):
    """Tests for AST analyzer."""

    def test_python_symbols(self):
        """Test Python symbol extraction."""
        from backend.ast_analyzer import TreeSitterAnalyzer

        analyzer = TreeSitterAnalyzer()

        code = '''
def hello_world(name):
    """Greet someone."""
    return f"Hello, {name}!"

class Greeter:
    def __init__(self, greeting="Hello"):
        self.greeting = greeting

    def greet(self, name):
        return f"{self.greeting}, {name}!"
'''

        symbols = analyzer.extract_symbols(code, "python")

        symbol_names = [s.name for s in symbols]
        symbol_types = {s.type for s in symbols}

        self.assertIn("hello_world", symbol_names)
        self.assertIn("Greeter", symbol_names)
        self.assertIn("__init__", symbol_names)
        self.assertIn("greet", symbol_names)

    def test_javascript_symbols(self):
        """Test JavaScript symbol extraction."""
        from backend.ast_analyzer import TreeSitterAnalyzer

        analyzer = TreeSitterAnalyzer()

        code = '''
import React from "react";
import { useState } from "react";

function Counter({ initial = 0 }) {
    const [count, setCount] = useState(initial);
    return count;
}

class Button {
    constructor(label) {
        this.label = label;
    }

    click() {
        return "Clicked";
    }
}
'''

        symbols = analyzer.extract_symbols(code, "javascript")

        symbol_names = [s.name for s in symbols]
        symbol_types = {s.type for s in symbols}

        self.assertIn("Counter", symbol_names)
        self.assertIn("Button", symbol_names)
        self.assertIn("click", symbol_names)

    def test_go_symbols(self):
        """Test Go symbol extraction."""
        from backend.ast_analyzer import TreeSitterAnalyzer

        analyzer = TreeSitterAnalyzer()

        code = '''
package main

import "fmt"

func hello(name string) string {
    return fmt.Sprintf("Hello, %s", name)
}

type Person struct {
    Name string
    Age  int
}

func (p *Person) Greet() string {
    return fmt.Sprintf("Hi, I'm %s", p.Name)
}
'''

        symbols = analyzer.extract_symbols(code, "go")

        symbol_names = [s.name for s in symbols]

        self.assertIn("hello", symbol_names)
        self.assertIn("Person", symbol_names)
        self.assertIn("Greet", symbol_names)

    def test_complexity_calculation(self):
        """Test cyclomatic complexity calculation."""
        from backend.ast_analyzer import TreeSitterAnalyzer

        analyzer = TreeSitterAnalyzer()

        simple_code = "def foo(): pass"
        complex_code = '''
def foo(x):
    if x > 0:
        if x > 10:
            while True:
                if x % 2 == 0:
                    return x
    return 0
'''

        simple_complexity = analyzer._calculate_complexity(simple_code)
        complex_complexity = analyzer._calculate_complexity(complex_code)

        self.assertGreater(complex_complexity, simple_complexity)


class TestFAISSIndex(unittest.TestCase):
    """Tests for FAISS index."""

    def setUp(self):
        """Set up test fixtures."""
        import faiss
        import numpy as np

        self.dim = 768
        self.index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)

        self.test_vectors = np.random.random((100, self.dim)).astype(np.float32)
        faiss.normalize_L2(self.test_vectors)
        self.index.add(self.test_vectors)

    def test_add_vectors(self):
        """Test adding vectors to index."""
        import numpy as np
        import faiss

        dim = 64
        index = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)

        vectors = np.random.random((50, dim)).astype(np.float32)
        faiss.normalize_L2(vectors)
        index.add(vectors)

        self.assertEqual(index.ntotal, 50)

    def test_search_vectors(self):
        """Test searching vectors."""
        import numpy as np
        import faiss

        dim = 64
        index = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)

        vectors = np.random.random((100, dim)).astype(np.float32)
        faiss.normalize_L2(vectors)
        index.add(vectors)

        query = np.random.random((1, dim)).astype(np.float32)
        faiss.normalize_L2(query)

        scores, ids = index.search(query, 10)

        self.assertEqual(len(scores), 1)
        self.assertEqual(len(ids[0]), 10)
        self.assertTrue(all(id_ >= -1 for id_ in ids[0]))

    def test_search_with_filter(self):
        """Test searching with ID filtering."""
        import numpy as np
        import faiss

        dim = 64
        index = faiss.IndexHNSWFlat(dim, 16, faiss.METRIC_INNER_PRODUCT)

        vectors = np.random.random((100, dim)).astype(np.float32)
        faiss.normalize_L2(vectors)
        index.add(vectors)

        query = np.random.random((1, dim)).astype(np.float32)
        faiss.normalize_L2(query)

        scores, ids = index.search(query, 50)

        self.assertEqual(len(ids[0]), 50)


class TestFileWatcher(unittest.TestCase):
    """Tests for file watcher."""

    def test_file_event_creation(self):
        """Test FileEvent creation."""
        from backend.file_watcher import FileEvent

        event = FileEvent(
            event_type="modified",
            file_path="/path/to/file.py"
        )

        self.assertEqual(event.event_type, "modified")
        self.assertEqual(event.file_path, "/path/to/file.py")
        self.assertIsInstance(event.timestamp, float)

    def test_file_watcher_initialization(self):
        """Test FileWatcher initialization."""
        from backend.file_watcher import FileWatcher

        watcher = FileWatcher(
            paths={"/tmp", "/var"},
            ignore_patterns={".git", "node_modules"},
            debounce_ms=100
        )

        self.assertEqual(watcher.debounce_ms, 100)
        self.assertEqual(len(watcher.paths), 2)


class TestCodeChunk(unittest.TestCase):
    """Tests for CodeChunk dataclass."""

    def test_code_chunk_creation(self):
        """Test CodeChunk creation."""
        from backend.indexer import CodeChunk

        chunk = CodeChunk(
            file_path="/src/main.py",
            start_line=10,
            end_line=25,
            code="def hello(): pass",
            language="python",
            symbols=["hello"],
            ast_hash="abc123"
        )

        self.assertEqual(chunk.file_path, "/src/main.py")
        self.assertEqual(chunk.start_line, 10)
        self.assertEqual(chunk.end_line, 25)
        self.assertEqual(chunk.language, "python")
        self.assertIn("hello", chunk.symbols)


class TestStructuralInfo(unittest.TestCase):
    """Tests for StructuralInfo."""

    def test_structural_info_creation(self):
        """Test StructuralInfo creation."""
        from backend.ast_analyzer import StructuralInfo

        info = StructuralInfo(
            file_path="/test.py",
            symbols=[],
            imports=["os", "sys"],
            function_calls=["open", "read"],
            ast_hash="hash123",
            complexity=5
        )

        self.assertEqual(info.file_path, "/test.py")
        self.assertEqual(len(info.imports), 2)
        self.assertEqual(len(info.function_calls), 2)
        self.assertEqual(info.complexity, 5)


class TestSearchResult(unittest.TestCase):
    """Tests for SearchResult."""

    def test_search_result_creation(self):
        """Test SearchResult creation."""
        from backend.faiss_index import SearchResult

        result = SearchResult(
            id=0,
            score=0.95,
            file_path="/src/main.py",
            start_line=10,
            end_line=20,
            code="def hello(): pass",
            language="python"
        )

        self.assertEqual(result.id, 0)
        self.assertAlmostEqual(result.score, 0.95, places=2)
        self.assertEqual(result.file_path, "/src/main.py")
        self.assertEqual(result.language, "python")


if __name__ == "__main__":
    unittest.main(verbosity=2)
