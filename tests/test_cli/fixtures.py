"""CLI test utilities and fixtures for Coarch testing."""

import os
import sys
import tempfile
import shutil
import sqlite3
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, Optional, Union
from unittest.mock import Mock, MagicMock, patch
from click.testing import CliRunner

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import CLI main
try:
    from cli.main import main
except ImportError:
    main = None


class MockHybridIndexer:
    """Mock HybridIndexer for CLI testing."""
    
    def __init__(self, db_path: str = "test.db"):
        self.db_path = db_path
        self.stats = {
            'total_chunks': 150,
            'total_repos': 3,
            'by_language': {'python': 80, 'javascript': 45, 'go': 25}
        }
        
    def index_repository(self, path: str, name: Optional[str] = None) -> Dict[str, Any]:
        """Mock index repository."""
        return {
            'files_indexed': 45,
            'chunks_created': 150,
            'repo_id': 1
        }
        
    def get_chunks_for_embedding(self) -> list:
        """Mock get chunks for embedding."""
        # Return mock chunks, then empty list to end iteration
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
        return 42


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


class MockServer:
    """Mock server for CLI testing."""
    
    def __init__(self):
        self.running = False
        
    def run_server(self, host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
        """Mock run server."""
        self.running = True
        self.host = host
        self.port = port
        self.reload = reload


class CLITestFixtures:
    """Fixture class for CLI testing."""
    
    def __init__(self):
        self.temp_dir = None
        self.original_env = {}
        
    def setup(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp(prefix='coarch_cli_test_')
        
        # Store original environment
        for key in ['COARCH_INDEX_PATH', 'COARCH_DB_PATH', 'COARCH_LOG_JSON']:
            self.original_env[key] = os.environ.get(key)
            
        # Set test environment
        os.environ['COARCH_INDEX_PATH'] = os.path.join(self.temp_dir, 'index')
        os.environ['COARCH_DB_PATH'] = os.path.join(self.temp_dir, 'coarch.db')
        os.environ['COARCH_LOG_JSON'] = 'false'
        
        # Create test index directory
        os.makedirs(os.environ['COARCH_INDEX_PATH'], exist_ok=True)
        
    def teardown(self):
        """Clean up test fixtures."""
        # Restore original environment
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
                
        # Clean up temp directory
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            
    def create_test_repo(self, repo_name: str = "test_repo") -> str:
        """Create a test repository with sample code."""
        if self.temp_dir is None:
            raise RuntimeError("Test fixtures not initialized. Call setup() first.")
        
        repo_path = os.path.join(self.temp_dir, repo_name) if self.temp_dir else repo_name
        os.makedirs(repo_path, exist_ok=True)
        
        # Create sample Python files
        files = {
            'main.py': '''
def hello():
    """Print hello message."""
    print("Hello, World!")

class Calculator:
    def add(self, a, b):
        return a + b
        
    def multiply(self, a, b):
        return a * b

if __name__ == "__main__":
    hello()
''',
            'utils.py': '''
import os
import sys

def read_file(path):
    """Read file content."""
    with open(path, 'r') as f:
        return f.read()

def write_file(path, content):
    """Write content to file."""
    with open(path, 'w') as f:
        f.write(content)
''',
            'config.py': '''
DEBUG = True
VERSION = "1.0.0"

class Config:
    def __init__(self):
        self.debug = DEBUG
        self.version = VERSION
        
    def get_env(self):
        return os.environ.get('APP_ENV', 'development')
'''
        }
        
        for filename, content in files.items():
            file_path = os.path.join(repo_path, filename)
            with open(file_path, 'w') as f:
                f.write(content)
                
        return repo_path


def create_cli_runner() -> CliRunner:
    """Create a CliRunner for testing CLI commands."""
    return CliRunner()


def mock_backend_modules():
    """Context manager to mock backend modules."""
    return patch.multiple(
        'backend.hybrid_indexer',
        HybridIndexer=MockHybridIndexer
    ), patch.multiple(
        'backend.embeddings',
        CodeEmbedder=MockCodeEmbedder
    ), patch.multiple(
        'backend.faiss_index',
        FaissIndex=MockFaissIndex
    ), patch.multiple(
        'backend.file_watcher',
        IncrementalIndexer=MockIncrementalIndexer
    )


def mock_server_module():
    """Context manager to mock server module."""
    return patch('cli.main.run_server', return_value=None)


def create_mock_response(status_code: int = 200, data: Optional[Dict[str, Any]] = None) -> Any:
    """Create a mock HTTP response."""
    mock_response = Mock()
    mock_response.status_code = status_code
    response_data = data if data is not None else {"status": "ok"}
    mock_response.read.return_value = lambda: str(response_data).encode('utf-8')
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=None)
    return mock_response


# Global fixture instance
fixtures = CLITestFixtures()


def pytest_configure(config):
    """Configure pytest."""
    # Setup fixtures
    fixtures.setup()


def pytest_unconfigure(config):
    """Unconfigure pytest."""
    # Teardown fixtures
    fixtures.teardown()