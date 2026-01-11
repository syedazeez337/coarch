"""Quick test script for Coarch core functionality."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Coarch - Code Search Engine Test")
print("=" * 60)

print("\n[1/5] Testing FAISS index...")
try:
    import numpy as np
    import faiss

    dim = 768
    index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)

    vectors = np.random.random((100, dim)).astype(np.float32)
    faiss.normalize_L2(vectors)
    index.add(vectors)

    print(f"   [OK] Added {index.ntotal} vectors to index")

    query = np.random.random((1, dim)).astype(np.float32)
    faiss.normalize_L2(query)
    scores, ids = index.search(query, 5)
    print(f"   [OK] Search returned {len(ids[0])} results")
    print(f"   [OK] Best score: {scores[0][0]:.4f}")
except Exception as e:
    print(f"   [FAIL] Error: {e}")

print("\n[2/5] Testing indexer (SQLite)...")
try:
    from backend.indexer import RepositoryIndexer

    indexer = RepositoryIndexer(db_path=":memory:")
    print("   [OK] SQLite indexer created")

    py_file = "test.py"
    js_file = "test.js"
    md_file = "README.md"

    assert indexer.get_language(py_file) == "python"
    assert indexer.get_language(js_file) == "javascript"
    from pathlib import Path

    assert indexer.get_language(md_file) == "markdown"
    print("   [OK] Language detection works")

    assert indexer.should_ignore(Path(".git/config"))
    assert indexer.should_ignore(Path("node_modules/package.json"))
    assert indexer.should_ignore(Path("__pycache__/cache.py"))
    print("   [OK] Ignore patterns work")

except Exception as e:
    print(f"   [FAIL] Error: {e}")

print("\n[3/5] Testing AST analyzer...")
try:
    from backend.ast_analyzer import TreeSitterAnalyzer

    analyzer = TreeSitterAnalyzer()

    python_code = """
def hello_world(name):
    print(f"Hello, {name}!")
    return True

class Greeter:
    def __init__(self, greeting="Hello"):
        self.greeting = greeting

    def greet(self, name):
        return f"{self.greeting}, {name}!"
"""

    symbols = analyzer.extract_symbols(python_code, "python")
    print(f"   [OK] Found {len(symbols)} symbols in Python code")
    for sym in symbols[:5]:
        print(f"      - {sym.type}: {sym.name}")

    js_code = """
import React from "react";
import { useState } from "react";

function Counter({ initial = 0 }) {
    const [count, setCount] = useState(initial);
    return <button>{count}</button>;
}
"""

    js_symbols = analyzer.extract_symbols(js_code, "javascript")
    print(f"   [OK] Found {len(js_symbols)} symbols in JavaScript code")
    for sym in js_symbols[:5]:
        print(f"      - {sym.type}: {sym.name}")

except Exception as e:
    print(f"   [FAIL] Error: {e}")

print("\n[4/5] Testing structural analysis...")
try:
    from backend.ast_analyzer import StructuralInfo

    info = StructuralInfo(
        file_path="/test/file.py",
        symbols=[],
        imports=["os", "sys"],
        function_calls=["open", "read", "write"],
        ast_hash="abc123",
        complexity=5,
    )
    print(f"   [OK] StructuralInfo created")
    print(f"   [OK] Imports: {info.imports}")
    print(f"   [OK] Function calls: {info.function_calls}")
    print(f"   [OK] Complexity: {info.complexity}")

except Exception as e:
    print(f"   [FAIL] Error: {e}")

print("\n[5/5] Testing file watcher...")
try:
    from backend.file_watcher import FileWatcher, FileEvent

    event = FileEvent(event_type="modified", file_path="/test/file.py")
    print(f"   [OK] FileEvent created: {event.event_type} - {event.file_path}")

    watcher = FileWatcher({"/tmp"}, debounce_ms=100)
    print(f"   [OK] FileWatcher initialized")

except Exception as e:
    print(f"   [FAIL] Error: {e}")

print("\n" + "=" * 60)
print("Core tests completed!")
print("=" * 60)
