#!/usr/bin/env python3
"""Performance benchmarks for CLI commands.

This module provides performance regression tests with pass/fail thresholds.
Run with: python tests/perf_benchmarks.py
"""

import os
import sys
import time
import tempfile
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['COARCH_TEST_MODE'] = '1'
os.environ['COARCH_INDEX_PATH'] = 'test_coarch_index'
os.environ['COARCH_DB_PATH'] = 'test_coarch.db'

from click.testing import CliRunner


class MockHybridIndexer:
    """Mock HybridIndexer for performance testing."""

    def __init__(self, db_path: str = "test.db"):
        self.db_path = db_path
        self.stats = {
            'total_chunks': 150,
            'total_repos': 3,
            'by_language': {'python': 80, 'javascript': 45, 'go': 25}
        }
        self._indexed_repos = {}

    def index_repository(self, path: str, name: str = None):
        """Mock index repository."""
        repo_id = len(self._indexed_repos) + 1
        self._indexed_repos[repo_id] = {
            'path': path,
            'name': name or os.path.basename(path)
        }
        return {
            'files_indexed': 45,
            'chunks_created': 150,
            'repo_id': repo_id
        }

    def get_chunks_for_embedding(self):
        """Mock get chunks for embedding."""
        if not hasattr(self, '_chunks_yielded'):
            self._chunks_yielded = True
            return [
                {
                    'id': i,
                    'file_path': f'/test/file{i}.py',
                    'code': f'def function_{i}(): pass',
                    'language': 'python'
                }
                for i in range(10)
            ]
        return []

    def update_chunk_embedding(self, chunk_id: int, embedding_id: int):
        pass

    def get_stats(self):
        return self.stats

    def delete_repo(self, repo_id: int):
        if repo_id in self._indexed_repos:
            del self._indexed_repos[repo_id]
            return 42
        return 0


class MockCodeEmbedder:
    """Mock CodeEmbedder for performance testing."""

    def __init__(self):
        self.dimension = 768

    def embed(self, texts):
        import numpy as np
        return np.array([np.random.random(self.dimension).astype(np.float32) for _ in texts], dtype=np.float32)

    def embed_query(self, query):
        import numpy as np
        return np.random.random(self.dimension).astype(np.float32)

    def get_dimension(self):
        return self.dimension


class MockFaissIndex:
    """Mock FaissIndex for performance testing."""

    def __init__(self, dim: int = 768, index_path: str = "test_index"):
        self.dim = dim
        self.index_path = index_path
        self._vectors = 150
        self._results = [
            Mock(id=0, score=0.95, file_path="/test/main.py", start_line=10, end_line=20,
                 code="def hello(): pass", language="python"),
            Mock(id=1, score=0.87, file_path="/test/utils.py", start_line=5, end_line=15,
                 code="def world(): pass", language="python")
        ]

    def add(self, embeddings, metadata):
        self._vectors += len(embeddings) if hasattr(embeddings, '__len__') else 1

    def save(self):
        pass

    def search(self, query_embedding, k: int = 10):
        return self._results[:k]

    def count(self):
        return self._vectors


def create_test_repo(path: str, num_files: int = 5) -> str:
    """Create a test repository."""
    repo_path = os.path.join(path, f"test_repo_{int(time.time())}")
    os.makedirs(repo_path, exist_ok=True)

    for i in range(1, num_files + 1):
        content = f'''
"""Test file {i}."""

def function_{i}():
    result = {i} * 2
    return result

class Class{i}:
    def __init__(self):
        self.value = {i}

    def get_value(self):
        return self.value
'''
        with open(os.path.join(repo_path, f"file_{i}.py"), 'w') as f:
            f.write(content)

    return repo_path


class PerformanceBenchmark:
    """Performance benchmark runner with pass/fail thresholds."""

    def __init__(self):
        self.results = []
        self.thresholds = {
            'index_small': 5.0,
            'index_medium': 10.0,
            'index_large': 30.0,
            'search': 2.0,
            'status': 1.0,
            'config': 1.0,
            'help': 1.0,
            'version': 0.5,
        }

    def benchmark_index_small(self, runner, test_repo):
        """Benchmark indexing a small repository."""
        from cli.main import main

        start = time.time()
        with patch('backend.hybrid_indexer.HybridIndexer', MockHybridIndexer), \
             patch('backend.embeddings.CodeEmbedder', MockCodeEmbedder), \
             patch('backend.faiss_index.FaissIndex', MockFaissIndex):
            result = runner.invoke(main, ['index', test_repo])
        elapsed = time.time() - start

        passed = elapsed <= self.thresholds['index_small'] and result.exit_code == 0
        self.results.append({
            'name': 'index_small',
            'elapsed': elapsed,
            'threshold': self.thresholds['index_small'],
            'passed': passed,
            'output': result.output[:100] if result.output else ''
        })

        return passed, elapsed

    def benchmark_index_medium(self, runner, test_repo):
        """Benchmark indexing a medium repository."""
        from cli.main import main

        start = time.time()
        with patch('backend.hybrid_indexer.HybridIndexer', MockHybridIndexer), \
             patch('backend.embeddings.CodeEmbedder', MockCodeEmbedder), \
             patch('backend.faiss_index.FaissIndex', MockFaissIndex):
            result = runner.invoke(main, ['index', test_repo])
        elapsed = time.time() - start

        passed = elapsed <= self.thresholds['index_medium'] and result.exit_code == 0
        self.results.append({
            'name': 'index_medium',
            'elapsed': elapsed,
            'threshold': self.thresholds['index_medium'],
            'passed': passed,
            'output': result.output[:100] if result.output else ''
        })

        return passed, elapsed

    def benchmark_search(self, runner):
        """Benchmark search command."""
        from cli.main import main

        start = time.time()
        with patch('backend.hybrid_indexer.HybridIndexer', MockHybridIndexer), \
             patch('backend.embeddings.CodeEmbedder', MockCodeEmbedder), \
             patch('backend.faiss_index.FaissIndex', MockFaissIndex):
            result = runner.invoke(main, ['search', 'test function', '--limit', '10'])
        elapsed = time.time() - start

        passed = elapsed <= self.thresholds['search'] and result.exit_code == 0
        self.results.append({
            'name': 'search',
            'elapsed': elapsed,
            'threshold': self.thresholds['search'],
            'passed': passed,
            'output': result.output[:100] if result.output else ''
        })

        return passed, elapsed

    def benchmark_status(self, runner):
        """Benchmark status command."""
        from cli.main import main

        start = time.time()
        with patch('backend.hybrid_indexer.HybridIndexer', MockHybridIndexer):
            result = runner.invoke(main, ['status'])
        elapsed = time.time() - start

        passed = elapsed <= self.thresholds['status'] and result.exit_code == 0
        self.results.append({
            'name': 'status',
            'elapsed': elapsed,
            'threshold': self.thresholds['status'],
            'passed': passed,
            'output': result.output[:100] if result.output else ''
        })

        return passed, elapsed

    def benchmark_config(self, runner):
        """Benchmark config command."""
        from cli.main import main

        start = time.time()
        result = runner.invoke(main, ['config'])
        elapsed = time.time() - start

        passed = elapsed <= self.thresholds['config'] and result.exit_code == 0
        self.results.append({
            'name': 'config',
            'elapsed': elapsed,
            'threshold': self.thresholds['config'],
            'passed': passed,
            'output': result.output[:100] if result.output else ''
        })

        return passed, elapsed

    def benchmark_help(self, runner):
        """Benchmark help command."""
        from cli.main import main

        start = time.time()
        result = runner.invoke(main, ['--help'])
        elapsed = time.time() - start

        passed = elapsed <= self.thresholds['help'] and result.exit_code == 0
        self.results.append({
            'name': 'help',
            'elapsed': elapsed,
            'threshold': self.thresholds['help'],
            'passed': passed,
            'output': result.output[:100] if result.output else ''
        })

        return passed, elapsed

    def benchmark_version(self, runner):
        """Benchmark version command."""
        from cli.main import main

        start = time.time()
        result = runner.invoke(main, ['--version'])
        elapsed = time.time() - start

        passed = elapsed <= self.thresholds['version'] and result.exit_code == 0
        self.results.append({
            'name': 'version',
            'elapsed': elapsed,
            'threshold': self.thresholds['version'],
            'passed': passed,
            'output': result.output[:100] if result.output else ''
        })

        return passed, elapsed

    def run_all(self):
        """Run all benchmarks."""
        print("=" * 70)
        print("COARCH PERFORMANCE BENCHMARKS")
        print("=" * 70)
        print()

        runner = CliRunner()
        temp_dir = tempfile.mkdtemp(prefix='coarch_perf_')

        try:
            small_repo = create_test_repo(temp_dir, 10)
            medium_repo = create_test_repo(temp_dir, 20)

            benchmarks = [
                ('Version', lambda: self.benchmark_version(runner)),
                ('Help', lambda: self.benchmark_help(runner)),
                ('Config', lambda: self.benchmark_config(runner)),
                ('Status', lambda: self.benchmark_status(runner)),
                ('Search', lambda: self.benchmark_search(runner)),
                ('Index (small, 10 files)', lambda: self.benchmark_index_small(runner, small_repo)),
                ('Index (medium, 20 files)', lambda: self.benchmark_index_medium(runner, medium_repo)),
            ]

            for name, benchmark_fn in benchmarks:
                try:
                    passed, elapsed = benchmark_fn()
                    status = "PASS" if passed else "FAIL"
                    print(f"[{status}] {name}: {elapsed:.3f}s (threshold: {self.thresholds.get(name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '').replace(',', ''), 1.0):.1f}s)")
                except Exception as e:
                    print(f"[ERROR] {name}: {e}")
                    self.results.append({'name': name, 'passed': False, 'error': str(e)})
                print()

        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        self.print_summary()

    def print_summary(self):
        """Print benchmark summary."""
        print("=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)

        passed = sum(1 for r in self.results if r.get('passed', False))
        failed = len(self.results) - passed

        print(f"Total benchmarks: {len(self.results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print()

        if failed > 0:
            print("Failed benchmarks:")
            for r in self.results:
                if not r.get('passed', True):
                    print(f"  - {r['name']}: {r.get('elapsed', 'N/A')}s (threshold: {r.get('threshold', 'N/A')}s)")
        print()

        if passed == len(self.results):
            print("All benchmarks PASSED!")
        else:
            print("Some benchmarks FAILED - regression detected!")

        print("=" * 70)


class TestPerformanceRegression(unittest.TestCase):
    """Unit tests for performance regression detection."""

    def setUp(self):
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp(prefix='coarch_perf_test_')

    def tearDown(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_index_performance_threshold(self):
        """Test that indexing meets performance threshold."""
        benchmark = PerformanceBenchmark()
        test_repo = create_test_repo(self.temp_dir, 10)

        passed, elapsed = benchmark.benchmark_index_small(self.runner, test_repo)

        self.assertLessEqual(elapsed, 5.0,
            f"Indexing took {elapsed}s, exceeds 5s threshold")
        self.assertTrue(passed, "Indexing should complete successfully")

    def test_search_performance_threshold(self):
        """Test that search meets performance threshold."""
        benchmark = PerformanceBenchmark()

        passed, elapsed = benchmark.benchmark_search(self.runner)

        self.assertLessEqual(elapsed, 2.0,
            f"Search took {elapsed}s, exceeds 2s threshold")
        self.assertTrue(passed, "Search should complete successfully")

    def test_status_performance_threshold(self):
        """Test that status meets performance threshold."""
        benchmark = PerformanceBenchmark()

        passed, elapsed = benchmark.benchmark_status(self.runner)

        self.assertLessEqual(elapsed, 1.0,
            f"Status took {elapsed}s, exceeds 1s threshold")
        self.assertTrue(passed, "Status should complete successfully")

    def test_version_performance_threshold(self):
        """Test that version meets performance threshold."""
        benchmark = PerformanceBenchmark()

        passed, elapsed = benchmark.benchmark_version(self.runner)

        self.assertLessEqual(elapsed, 0.5,
            f"Version took {elapsed}s, exceeds 0.5s threshold")
        self.assertTrue(passed, "Version should complete successfully")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--unittest':
        unittest.main(argv=[''])
    else:
        benchmark = PerformanceBenchmark()
        benchmark.run_all()
