"""Pytest configuration for Coarch CLI tests."""

import pytest
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def pytest_configure(config):
    """Configure pytest."""
    # Add CLI test markers
    config.addinivalue_line(
        "markers", "cli: mark test as CLI-related"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add cli marker to all CLI tests
        if "test_cli" in str(item.fspath):
            item.add_marker(pytest.mark.cli)
            
        # Add integration marker to integration tests
        if "integration" in item.name.lower():
            item.add_marker(pytest.mark.integration)
            
        # Add slow marker to tests that might be slow
        if any(keyword in item.name.lower() for keyword in ["server", "serve", "watch"]):
            item.add_marker(pytest.mark.slow)


# Fixtures for CLI testing
@pytest.fixture
def cli_runner():
    """Create a Click test runner."""
    from click.testing import CliRunner
    return CliRunner()


@pytest.fixture
def test_repo(tmp_path):
    """Create a test repository with sample code."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    
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
        file_path = repo_path / filename
        file_path.write_text(content)
        
    return str(repo_path)


@pytest.fixture
def mock_backend(monkeypatch):
    """Mock backend modules for testing."""
    class MockIndexer:
        def index_repository(self, path, name=None):
            return {'files_indexed': 10, 'chunks_created': 50}
        def get_chunks_for_embedding(self):
            return []
        def update_chunk_embedding(self, chunk_id, embedding_id):
            pass
        def get_stats(self):
            return {'total_chunks': 150, 'total_repos': 3, 'by_language': {'python': 80}}
        def delete_repo(self, repo_id):
            return 42

    class MockEmbedder:
        def get_dimension(self):
            return 768
        def embed(self, texts):
            import numpy as np
            return np.random.random((len(texts), 768)).astype(np.float32)
        def embed_query(self, query):
            import numpy as np
            return np.random.random(768).astype(np.float32)

    class MockFaiss:
        def __init__(self, dim=768, index_path="test"):
            pass
        def add(self, embeddings, metadata):
            pass
        def save(self):
            pass
        def search(self, query_embedding, k=10):
            from unittest.mock import Mock
            return [Mock(id=0, score=0.95, file_path="/test/main.py", start_line=10, end_line=20, code="def hello(): pass", language="python")]
        def count(self):
            return 150
    
    # Apply mocks
    monkeypatch.setattr('backend.hybrid_indexer.HybridIndexer', MockIndexer)
    monkeypatch.setattr('backend.embeddings.CodeEmbedder', MockEmbedder)
    monkeypatch.setattr('backend.faiss_index.FaissIndex', MockFaiss)
    
    return {
        'indexer': MockIndexer(),
        'embedder': MockEmbedder(),
        'faiss': MockFaiss()
    }


# Custom test runner for running only CLI tests
def run_cli_tests():
    """Run only CLI tests."""
    import subprocess
    import sys
    
    result = subprocess.run([
        sys.executable, '-m', 'pytest', 
        'tests/test_cli/', 
        '-v',
        '--tb=short'
    ], cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    return result.returncode


if __name__ == '__main__':
    # Allow running this file directly to run CLI tests
    sys.exit(run_cli_tests())