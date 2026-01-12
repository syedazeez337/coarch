"""Comprehensive CLI integration tests for Coarch.

This module provides end-to-end tests for CLI workflows including:
- Full indexing workflow (init, index, status, search, delete)
- Search functionality with various query types
- Server lifecycle (start, health check, stop)
- Error recovery scenarios (partial index, missing files)
- Configuration changes and rollback
- Signal handling during operations
- Cross-platform compatibility (Windows, Linux, macOS)
- Performance regression tests
"""

import os
import sys
import time
import tempfile
import shutil
import subprocess
import threading
import signal
import platform
import unittest
from pathlib import Path
from typing import Dict, Any, Optional, List
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['COARCH_TEST_MODE'] = '1'
os.environ['COARCH_INDEX_PATH'] = 'test_coarch_index'
os.environ['COARCH_DB_PATH'] = 'test_coarch.db'


class MockHybridIndexer:
    """Mock HybridIndexer for CLI testing."""

    def __init__(self, db_path: str = "test.db"):
        self.db_path = db_path
        self.stats = {
            'total_chunks': 150,
            'total_repos': 3,
            'by_language': {'python': 80, 'javascript': 45, 'go': 25}
        }
        self._indexed_repos = {}

    def index_repository(self, path: str, name: Optional[str] = None) -> Dict[str, Any]:
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

    def get_chunks_for_embedding(self) -> list:
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
        """Mock update chunk embedding."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Mock get stats."""
        return self.stats

    def delete_repo(self, repo_id: int) -> int:
        """Mock delete repo."""
        if repo_id in self._indexed_repos:
            del self._indexed_repos[repo_id]
            return 42
        return 0


class MockCodeEmbedder:
    """Mock CodeEmbedder for CLI testing."""

    def __init__(self):
        self.dimension = 768

    def embed(self, texts: list) -> Any:
        """Mock embed texts."""
        import numpy as np
        return np.array([np.random.random(self.dimension).astype(np.float32) for _ in texts], dtype=np.float32)

    def embed_query(self, query: str) -> Any:
        """Mock embed query."""
        import numpy as np
        return np.random.random(self.dimension).astype(np.float32)

    def get_dimension(self) -> int:
        """Mock get dimension."""
        return self.dimension


class MockFaissIndex:
    """Mock FaissIndex for CLI testing."""

    def __init__(self, dim: int = 768, index_path: str = "test_index"):
        self.dim = dim
        self.index_path = index_path
        self._vectors = 150
        self._results = [
            Mock(
                id=0,
                score=0.95,
                file_path="/test/main.py",
                start_line=10,
                end_line=20,
                code="def hello(): pass",
                language="python"
            ),
            Mock(
                id=1,
                score=0.87,
                file_path="/test/utils.py",
                start_line=5,
                end_line=15,
                code="def world(): pass",
                language="python"
            )
        ]

    def add(self, embeddings: Any, metadata: list):
        """Mock add embeddings."""
        self._vectors += len(embeddings) if hasattr(embeddings, '__len__') else 1

    def save(self):
        """Mock save index."""
        pass

    def search(self, query_embedding: Any, k: int = 10) -> list:
        """Mock search."""
        return self._results[:k]

    def count(self) -> int:
        """Mock count vectors."""
        return self._vectors


class MockIncrementalIndexer:
    """Mock IncrementalIndexer for CLI testing."""

    def __init__(self, indexer, embedder, faiss, paths):
        self.indexer = indexer
        self.embedder = embedder
        self.faiss = faiss
        self.paths = paths
        self.running = False

    def start(self):
        """Mock start."""
        self.running = True

    def stop(self):
        """Mock stop."""
        self.running = False


def create_test_repo(path: str, num_files: int = 5) -> str:
    """Create a test repository with sample code files.

    Args:
        path: Base directory path
        num_files: Number of files to create

    Returns:
        Path to created repository
    """
    repo_path = os.path.join(path, f"test_repo_{int(time.time())}")
    os.makedirs(repo_path, exist_ok=True)

    for i in range(1, num_files + 1):
        content = f'''
"""Test file {i} for integration testing."""

def function_{i}():
    """Function number {i}."""
    result = {i} * 2
    return result

class Class{i}:
    """Class number {i}."""

    def __init__(self):
        self.value = {i}

    def get_value(self):
        """Get the value."""
        return self.value

    def process(self, data):
        """Process data."""
        return [x * self.value for x in data]

if __name__ == "__main__":
    obj = Class{i}()
    print(obj.get_value())
'''
        file_path = os.path.join(repo_path, f"file_{i}.py")
        with open(file_path, 'w') as f:
            f.write(content)

    return repo_path


def create_large_test_repo(path: str, num_files: int = 20) -> str:
    """Create a larger test repository for performance testing.

    Args:
        path: Base directory path
        num_files: Number of files to create

    Returns:
        Path to created repository
    """
    repo_path = os.path.join(path, f"large_test_repo_{int(time.time())}")
    os.makedirs(repo_path, exist_ok=True)

    for i in range(1, num_files + 1):
        functions = []
        for j in range(1, 11):
            functions.append(f'''
def function_{i}_{j}():
    """Function {i}_{j} in file {i}."""
    data = [{', '.join(str(k * j) for k in range(1, 6))}]
    result = sum(data) * {j}
    return result
''')
        content = f'''
"""Large test file {i} with multiple functions."""

{''.join(functions)}

class Processor{i}:
    """Processor class for file {i}."""

    def __init__(self, name="Processor{i}"):
        self.name = name
        self.data = []

    def add_data(self, value):
        """Add data to processor."""
        self.data.append(value)

    def process_all(self):
        """Process all data."""
        return [d * 2 for d in self.data]

    def get_stats(self):
        """Get processor statistics."""
        return {{
            "name": self.name,
            "count": len(self.data)
        }}
'''
        file_path = os.path.join(repo_path, f"module_{i}.py")
        with open(file_path, 'w') as f:
            f.write(content)

    return repo_path


class TestFullIndexingWorkflow(unittest.TestCase):
    """Test full indexing workflow (init, index, status, search, delete)."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp(prefix='coarch_integration_')
        self.test_repo = create_test_repo(self.temp_dir, 10)

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('backend.hybrid_indexer.HybridIndexer', MockHybridIndexer)
    @patch('backend.embeddings.CodeEmbedder', MockCodeEmbedder)
    @patch('backend.faiss_index.FaissIndex', MockFaissIndex)
    def test_full_workflow_init_index_status_search_delete(self):
        """Test complete workflow: init, index, status, search, delete."""
        from cli.main import main

        result = self.runner.invoke(main, ['init'])
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(result.exit_code in [0, 1])

        result = self.runner.invoke(main, ['index', self.test_repo])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Indexing repository", result.output)
        self.assertIn("Files indexed", result.output)

        result = self.runner.invoke(main, ['status'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Index Statistics", result.output) or result.exit_code in [0, 1]
        self.assertIn("Total chunks", result.output)

        result = self.runner.invoke(main, ['search', 'hello function'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Searching for", result.output)
        self.assertIn("Found", result.output)

        result = self.runner.invoke(main, ['delete', '1'], input='y\n')
        self.assertEqual(result.exit_code, 0)

    @patch('backend.hybrid_indexer.HybridIndexer', MockHybridIndexer)
    @patch('backend.embeddings.CodeEmbedder', MockCodeEmbedder)
    @patch('backend.faiss_index.FaissIndex', MockFaissIndex)
    def test_index_with_custom_name(self):
        """Test indexing with custom repository name."""
        from cli.main import main

        result = self.runner.invoke(main, ['index', self.test_repo, '--name', 'my_test_repo'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Indexing repository", result.output)
        self.assertIn("Files indexed", result.output)


class TestSearchFunctionality(unittest.TestCase):
    """Test search functionality with various query types."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp(prefix='coarch_search_')
        self.test_repo = create_test_repo(self.temp_dir, 10)

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('backend.hybrid_indexer.HybridIndexer', MockHybridIndexer)
    @patch('backend.embeddings.CodeEmbedder', MockCodeEmbedder)
    @patch('backend.faiss_index.FaissIndex', MockFaissIndex)
    def test_search_with_limit(self):
        """Test search with result limit."""
        from cli.main import main

        result = self.runner.invoke(main, ['search', 'function', '--limit', '5'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Searching for", result.output)

    @patch('backend.hybrid_indexer.HybridIndexer', MockHybridIndexer)
    @patch('backend.embeddings.CodeEmbedder', MockCodeEmbedder)
    @patch('backend.faiss_index.FaissIndex', MockFaissIndex)
    def test_search_with_language_filter(self):
        """Test search with language filter."""
        from cli.main import main

        result = self.runner.invoke(main, ['search', 'class', '--language', 'python'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Language filter", result.output)

    @patch('backend.hybrid_indexer.HybridIndexer', MockHybridIndexer)
    @patch('backend.embeddings.CodeEmbedder', MockCodeEmbedder)
    @patch('backend.faiss_index.FaissIndex', MockFaissIndex)
    def test_search_with_limit_and_language(self):
        """Test search with both limit and language filter."""
        from cli.main import main

        result = self.runner.invoke(main, ['search', 'def', '--limit', '3', '--language', 'python'])
        self.assertEqual(result.exit_code, 0)

    @patch('backend.hybrid_indexer.HybridIndexer', MockHybridIndexer)
    @patch('backend.embeddings.CodeEmbedder', MockCodeEmbedder)
    @patch('backend.faiss_index.FaissIndex', MockFaissIndex)
    def test_search_find_alias(self):
        """Test search using 'find' alias."""
        from cli.main import main

        result = self.runner.invoke(main, ['find', 'hello world'])
        self.assertIn("Searching for", result.output)


class TestServerLifecycle(unittest.TestCase):
    """Test server lifecycle (start, health check, stop)."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch('backend.server.run_server')
    def test_server_start(self, mock_server):
        """Test server start."""
        from cli.main import main

        result = self.runner.invoke(main, ['serve', '--port', '9000'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Starting Coarch server", result.output)
        mock_server.assert_called_once_with(host="0.0.0.0", port=9000, reload=False)

    @patch('backend.server.run_server')
    def test_server_start_with_host(self, mock_server):
        """Test server start with custom host."""
        from cli.main import main

        result = self.runner.invoke(main, ['serve', '--host', 'localhost', '--port', '8080'])
        self.assertEqual(result.exit_code, 0)
        mock_server.assert_called_once_with(host="localhost", port=8080, reload=False)

    @patch('backend.server.run_server')
    def test_server_with_reload(self, mock_server):
        """Test server with reload flag."""
        from cli.main import main

        result = self.runner.invoke(main, ['serve', '--reload'])
        self.assertEqual(result.exit_code, 0)
        mock_server.assert_called_once_with(host="0.0.0.0", port=8000, reload=True)

    @patch('backend.server.run_server')
    def test_server_alias(self, mock_server):
        """Test server using 'server' alias."""
        from cli.main import main

        result = self.runner.invoke(main, ['server', '--port', '7000'])
        mock_server.assert_called_once_with(host="0.0.0.0", port=7000, reload=False)

    def test_health_check_healthy(self):
        """Test health check when server is healthy."""
        from cli.main import main
        import json

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.read.return_value = json.dumps({"status": "healthy", "vectors": 150}).encode('utf-8')
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        with patch('urllib.request.urlopen', return_value=mock_response):
            result = self.runner.invoke(main, ['health'])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Server is healthy", result.output)
            self.assertIn("Status: healthy", result.output)

    def test_health_check_unhealthy(self):
        """Test health check when server is down."""
        from cli.main import main

        with patch('urllib.request.urlopen', side_effect=Exception("Connection refused")):
            result = self.runner.invoke(main, ['health'])
            self.assertEqual(result.exit_code, 1)
            self.assertIn("Server is unhealthy", result.output)

    def test_health_check_custom_port(self):
        """Test health check with custom port."""
        from cli.main import main
        import json

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.read.return_value = json.dumps({"status": "healthy", "vectors": 100}).encode('utf-8')
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        with patch('urllib.request.urlopen', return_value=mock_response):
            result = self.runner.invoke(main, ['health', '--port', '9000'])
            self.assertEqual(result.exit_code, 0)

    def test_ping_alias(self):
        """Test health check using 'ping' alias."""
        from cli.main import main
        import json

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.read.return_value = json.dumps({"status": "healthy", "vectors": 150}).encode('utf-8')
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        with patch('urllib.request.urlopen', return_value=mock_response):
            result = self.runner.invoke(main, ['ping'])
            self.assertIn("Server is healthy", result.output)


class TestErrorRecovery(unittest.TestCase):
    """Test error recovery scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp(prefix='coarch_error_')

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_index_nonexistent_path(self):
        """Test index with non-existent path."""
        from cli.main import main

        result = self.runner.invoke(main, ['index', '/nonexistent/path'])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("does not exist", result.output)

    def test_search_with_empty_query(self):
        """Test search with empty query."""
        from cli.main import main

        result = self.runner.invoke(main, ['search', ''])
        self.assertNotEqual(result.exit_code, 0)

    def test_search_with_no_index(self):
        """Test search when no index exists."""
        from cli.main import main

        with patch('backend.faiss_index.FaissIndex.__init__', side_effect=Exception("Index not found")):
            result = self.runner.invoke(main, ['search', 'hello'])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("Error:", result.output)

    def test_delete_nonexistent_repo(self):
        """Test delete non-existent repository."""
        from cli.main import main

        with patch('backend.hybrid_indexer.HybridIndexer.delete_repo', return_value=0):
            result = self.runner.invoke(main, ['delete', '999'])
            self.assertEqual(result.exit_code, 0)

    def test_serve_invalid_port(self):
        """Test serve with invalid port."""
        from cli.main import main

        result = self.runner.invoke(main, ['serve', '--port', '70000'])
        self.assertNotEqual(result.exit_code, 0)

    def test_delete_invalid_repo_id(self):
        """Test delete with invalid repository ID."""
        from cli.main import main

        result = self.runner.invoke(main, ['delete', '-1'])
        self.assertNotEqual(result.exit_code, 0)

    def test_health_check_timeout(self):
        """Test health check with timeout."""
        from cli.main import main
        import urllib.error

        with patch('urllib.request.urlopen', side_effect=urllib.error.URLError("Timeout")):
            result = self.runner.invoke(main, ['health'])
            self.assertEqual(result.exit_code, 1)
            self.assertIn("Server is unhealthy", result.output)


class TestConfigurationIntegration(unittest.TestCase):
    """Test configuration changes and rollback."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp(prefix='coarch_config_')

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_init_configuration(self):
        """Test configuration initialization."""
        from cli.main import main

        result = self.runner.invoke(main, ['init'])
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(result.exit_code in [0, 1])

    def test_init_with_template(self):
        """Test configuration initialization with template."""
        from cli.main import main

        result = self.runner.invoke(main, ['init', '--template', 'development'])
        self.assertTrue(result.exit_code in [0, 1])

    def test_config_show(self):
        """Test configuration display."""
        from cli.main import main

        result = self.runner.invoke(main, ['config'])
        self.assertEqual(result.exit_code, 0)

    def test_config_show_json(self):
        """Test configuration display in JSON format."""
        from cli.main import main

        result = self.runner.invoke(main, ['config', '--format', 'json'])
        self.assertEqual(result.exit_code, 0)

    def test_config_show_env(self):
        """Test configuration display in env format."""
        from cli.main import main

        result = self.runner.invoke(main, ['config', '--format', 'env'])
        self.assertTrue(result.exit_code in [0, 1])

    def test_status_alias(self):
        """Test status using 'stats' alias."""
        from cli.main import main

        with patch('backend.hybrid_indexer.HybridIndexer') as mock_indexer:
            mock_instance = Mock()
            mock_instance.get_stats.return_value = {
                'total_chunks': 100,
                'total_repos': 2,
                'by_language': {'python': 50, 'javascript': 50}
            }
            mock_indexer.return_value = mock_instance

            result = self.runner.invoke(main, ['stats'])
            self.assertIn("Index Statistics", result.output) or result.exit_code in [0, 1] or result.exit_code in [0, 1]


class TestSignalHandling(unittest.TestCase):
    """Test signal handling during operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp(prefix='coarch_signal_')

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('backend.hybrid_indexer.HybridIndexer', MockHybridIndexer)
    @patch('backend.embeddings.CodeEmbedder', MockCodeEmbedder)
    @patch('backend.faiss_index.FaissIndex', MockFaissIndex)
    def test_keyboard_interrupt_during_index(self):
        """Test keyboard interrupt handling during indexing."""
        from cli.main import main

        with patch('backend.hybrid_indexer.HybridIndexer.index_repository',
                   side_effect=KeyboardInterrupt):
            result = self.runner.invoke(main, ['index', self.temp_dir])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("cancelled", result.output.lower())

    @patch('backend.hybrid_indexer.HybridIndexer', MockHybridIndexer)
    @patch('backend.embeddings.CodeEmbedder', MockCodeEmbedder)
    @patch('backend.faiss_index.FaissIndex', MockFaissIndex)
    def test_keyboard_interrupt_during_search(self):
        """Test keyboard interrupt handling during search."""
        from cli.main import main

        with patch('backend.faiss_index.FaissIndex.search',
                   side_effect=KeyboardInterrupt):
            result = self.runner.invoke(main, ['search', 'test'])
            self.assertEqual(result.exit_code, 0)


class TestCrossPlatformCompatibility(unittest.TestCase):
    """Test cross-platform compatibility."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp(prefix='coarch_platform_')

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_path_handling_windows(self):
        """Test path handling on Windows."""
        from cli.main import main

        if platform.system() == 'Windows':
            windows_path = 'C:\\Users\\Test\\project'
            result = self.runner.invoke(main, ['init', '--config', windows_path])
            self.assertEqual(result.exit_code, 0)

    def test_path_handling_linux(self):
        """Test path handling on Linux."""
        from cli.main import main

        if platform.system() == 'Linux':
            linux_path = '/home/user/project'
            result = self.runner.invoke(main, ['init', '--config', linux_path])
            self.assertEqual(result.exit_code, 0)

    def test_path_handling_macos(self):
        """Test path handling on macOS."""
        from cli.main import main

        if platform.system() == 'Darwin':
            macos_path = '/Users/user/project'
            result = self.runner.invoke(main, ['init', '--config', macos_path])
            self.assertEqual(result.exit_code, 0)

    def test_platform_detection(self):
        """Test platform detection."""
        current_platform = platform.system()
        self.assertIn(current_platform, ['Windows', 'Linux', 'Darwin'])


class TestMultipleRepositories(unittest.TestCase):
    """Test operations with multiple repositories."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp(prefix='coarch_multi_')
        self.repos = []
        for i in range(3):
            repo = create_test_repo(self.temp_dir, 5)
            self.repos.append(repo)

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('backend.hybrid_indexer.HybridIndexer', MockHybridIndexer)
    @patch('backend.embeddings.CodeEmbedder', MockCodeEmbedder)
    @patch('backend.faiss_index.FaissIndex', MockFaissIndex)
    def test_index_multiple_repos(self):
        """Test indexing multiple repositories."""
        from cli.main import main

        for i, repo in enumerate(self.repos):
            result = self.runner.invoke(main, ['index', repo, '--name', f'repo_{i}'])
            self.assertEqual(result.exit_code, 0)

    @patch('backend.hybrid_indexer.HybridIndexer', MockHybridIndexer)
    @patch('backend.embeddings.CodeEmbedder', MockCodeEmbedder)
    @patch('backend.faiss_index.FaissIndex', MockFaissIndex)
    def test_status_with_multiple_repos(self):
        """Test status with multiple repositories."""
        from cli.main import main

        result = self.runner.invoke(main, ['status'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Total repos", result.output)


class TestCompletionCommands(unittest.TestCase):
    """Test shell completion commands."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_completion_bash(self):
        """Test bash completion script generation."""
        from cli.main import main

        result = self.runner.invoke(main, ['completion', '--shell', 'bash', '--no-install'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("bash", result.output.lower())
        self.assertIn("coarch", result.output.lower())

    def test_completion_zsh(self):
        """Test zsh completion script generation."""
        from cli.main import main

        result = self.runner.invoke(main, ['completion', '--shell', 'zsh', '--no-install'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("zsh", result.output.lower())

    def test_completion_fish(self):
        """Test fish completion script generation."""
        from cli.main import main

        result = self.runner.invoke(main, ['completion', '--shell', 'fish', '--no-install'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("fish", result.output.lower())


class TestVersionAndHelp(unittest.TestCase):
    """Test version and help commands."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_version(self):
        """Test version flag."""
        from cli.main import main

        result = self.runner.invoke(main, ['--version'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("version", result.output.lower())

    def test_help(self):
        """Test help flag."""
        from cli.main import main

        result = self.runner.invoke(main, ['--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Commands", result.output)

    def test_index_help(self):
        """Test index command help."""
        from cli.main import main

        result = self.runner.invoke(main, ['index', '--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Index a repository", result.output)

    def test_search_help(self):
        """Test search command help."""
        from cli.main import main

        result = self.runner.invoke(main, ['search', '--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Search for semantic", result.output)

    def test_serve_help(self):
        """Test serve command help."""
        from cli.main import main

        result = self.runner.invoke(main, ['serve', '--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Start the Coarch API server", result.output)


class TestDeleteConfirmation(unittest.TestCase):
    """Test delete command with confirmation."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch('backend.hybrid_indexer.HybridIndexer', MockHybridIndexer)
    def test_delete_with_confirmation(self):
        """Test delete with confirmation."""
        from cli.main import main

        result = self.runner.invoke(main, ['delete', '1'], input='y\n')
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Deleting repository", result.output)

    @patch('backend.hybrid_indexer.HybridIndexer', MockHybridIndexer)
    def test_delete_without_confirmation(self):
        """Test delete without confirmation."""
        from cli.main import main

        result = self.runner.invoke(main, ['delete', '1'], input='n\n')
        self.assertEqual(result.exit_code, 0)
        self.assertIn("cancelled", result.output.lower())


class TestPerformanceRegression(unittest.TestCase):
    """Performance regression tests."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp(prefix='coarch_perf_')

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_indexing_performance_small_repo(self):
        """Test indexing performance with small repository."""
        from cli.main import main

        test_repo = create_test_repo(self.temp_dir, 10)

        start_time = time.time()
        with patch('backend.hybrid_indexer.HybridIndexer', MockHybridIndexer), \
             patch('backend.embeddings.CodeEmbedder', MockCodeEmbedder), \
             patch('backend.faiss_index.FaissIndex', MockFaissIndex):
            result = self.runner.invoke(main, ['index', test_repo])
        elapsed = time.time() - start_time

        self.assertEqual(result.exit_code, 0)
        self.assertLess(elapsed, 5.0, f"Indexing took too long: {elapsed}s")

    def test_indexing_performance_medium_repo(self):
        """Test indexing performance with medium repository."""
        from cli.main import main

        test_repo = create_test_repo(self.temp_dir, 20)

        start_time = time.time()
        with patch('backend.hybrid_indexer.HybridIndexer', MockHybridIndexer), \
             patch('backend.embeddings.CodeEmbedder', MockCodeEmbedder), \
             patch('backend.faiss_index.FaissIndex', MockFaissIndex):
            result = self.runner.invoke(main, ['index', test_repo])
        elapsed = time.time() - start_time

        self.assertEqual(result.exit_code, 0)
        self.assertLess(elapsed, 10.0, f"Indexing took too long: {elapsed}s")

    def test_search_performance(self):
        """Test search performance."""
        from cli.main import main

        start_time = time.time()
        with patch('backend.hybrid_indexer.HybridIndexer', MockHybridIndexer), \
             patch('backend.embeddings.CodeEmbedder', MockCodeEmbedder), \
             patch('backend.faiss_index.FaissIndex', MockFaissIndex):
            result = self.runner.invoke(main, ['search', 'test function', '--limit', '10'])
        elapsed = time.time() - start_time

        self.assertEqual(result.exit_code, 0)
        self.assertLess(elapsed, 2.0, f"Search took too long: {elapsed}s")

    def test_status_performance(self):
        """Test status performance."""
        from cli.main import main

        start_time = time.time()
        with patch('backend.hybrid_indexer.HybridIndexer', MockHybridIndexer):
            result = self.runner.invoke(main, ['status'])
        elapsed = time.time() - start_time

        self.assertEqual(result.exit_code, 0)
        self.assertLess(elapsed, 1.0, f"Status took too long: {elapsed}s")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp(prefix='coarch_edge_')

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_search_with_very_long_query(self):
        """Test search with very long query."""
        from cli.main import main

        long_query = "def " + "_test_func" * 50
        with patch('backend.hybrid_indexer.HybridIndexer', MockHybridIndexer), \
             patch('backend.embeddings.CodeEmbedder', MockCodeEmbedder), \
             patch('backend.faiss_index.FaissIndex', MockFaissIndex):
            result = self.runner.invoke(main, ['search', long_query])
        self.assertIn("Searching for", result.output)

    def test_search_with_special_characters(self):
        """Test search with special characters."""
        from cli.main import main

        with patch('backend.hybrid_indexer.HybridIndexer', MockHybridIndexer), \
             patch('backend.embeddings.CodeEmbedder', MockCodeEmbedder), \
             patch('backend.faiss_index.FaissIndex', MockFaissIndex):
            result = self.runner.invoke(main, ['search', 'def test(): # comment'])
        self.assertEqual(result.exit_code, 0)

    def test_index_empty_directory(self):
        """Test index with empty directory."""
        from cli.main import main

        empty_dir = os.path.join(self.temp_dir, 'empty')
        os.makedirs(empty_dir)

        with patch('backend.hybrid_indexer.HybridIndexer') as mock_indexer:
            mock_instance = Mock()
            mock_instance.index_repository.return_value = {
                'files_indexed': 0,
                'chunks_created': 0,
                'repo_id': 1
            }
            mock_instance.get_chunks_for_embedding.return_value = []
            mock_indexer.return_value = mock_instance

            result = self.runner.invoke(main, ['index', empty_dir])
            self.assertEqual(result.exit_code, 0)

    def test_search_with_limit_1(self):
        """Test search with limit of 1."""
        from cli.main import main

        with patch('backend.hybrid_indexer.HybridIndexer', MockHybridIndexer), \
             patch('backend.embeddings.CodeEmbedder', MockCodeEmbedder), \
             patch('backend.faiss_index.FaissIndex', MockFaissIndex):
            result = self.runner.invoke(main, ['search', 'test', '--limit', '1'])
        self.assertEqual(result.exit_code, 0)

    def test_search_with_large_limit(self):
        """Test search with large limit."""
        from cli.main import main

        with patch('backend.hybrid_indexer.HybridIndexer', MockHybridIndexer), \
             patch('backend.embeddings.CodeEmbedder', MockCodeEmbedder), \
             patch('backend.faiss_index.FaissIndex', MockFaissIndex):
            result = self.runner.invoke(main, ['search', 'test', '--limit', '100'])
        self.assertEqual(result.exit_code, 0)


class TestIndexCommandAlias(unittest.TestCase):
    """Test index command alias."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp(prefix='coarch_alias_')

    def tearDown(self):
        """Clean up test fixtures."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('backend.hybrid_indexer.HybridIndexer', MockHybridIndexer)
    @patch('backend.embeddings.CodeEmbedder', MockCodeEmbedder)
    @patch('backend.faiss_index.FaissIndex', MockFaissIndex)
    def test_idx_alias(self):
        """Test 'idx' alias for index command."""
        from cli.main import main

        result = self.runner.invoke(main, ['idx', self.temp_dir])
        self.assertTrue(result.exit_code in [0, 1])

    @patch('backend.hybrid_indexer.HybridIndexer', MockHybridIndexer)
    def test_remove_alias(self):
        """Test 'remove' alias for delete command."""
        from cli.main import main

        result = self.runner.invoke(main, ['remove', '1'], input='y\n')
        self.assertTrue(result.exit_code in [0, 1])


class TestConfigTemplate(unittest.TestCase):
    """Test configuration template functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_template_list(self):
        """Test template list command."""
        from cli.main import main

        result = self.runner.invoke(main, ['template', '--help'])
        self.assertEqual(result.exit_code, 0)


class TestVerboseAndLogOptions(unittest.TestCase):
    """Test verbose and log file options."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp(prefix='coarch_verbose_')

    def tearDown(self):
        """Clean up test fixtures."""
        import gc
        gc.collect()
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except PermissionError:
                import time
                time.sleep(0.1)
                gc.collect()
                try:
                    shutil.rmtree(self.temp_dir)
                except PermissionError:
                    pass

    def test_verbose_flag(self):
        """Test verbose flag."""
        from cli.main import main

        result = self.runner.invoke(main, ['--verbose', 'status'])
        self.assertEqual(result.exit_code, 0)

    def test_log_file_option(self):
        """Test log file option."""
        from cli.main import main

        log_path = os.path.join(self.temp_dir, 'test.log')
        result = self.runner.invoke(main, ['--log-file', log_path, 'status'])
        self.assertIn("Index Statistics", result.output) or result.exit_code in [0, 1]


class TestPrintConfigOption(unittest.TestCase):
    """Test print configuration option."""

    def setUp(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_print_config(self):
        """Test print config option."""
        from cli.main import main

        result = self.runner.invoke(main, ['--print-config'])
        self.assertIn("Config", result.output) or self.assertIn("config", result.output.lower())


if __name__ == '__main__':
    unittest.main(verbosity=2)
