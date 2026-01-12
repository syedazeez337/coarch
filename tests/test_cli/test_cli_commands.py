"""Simplified command functionality tests for Coarch CLI."""

import os
import sys
import unittest
import tempfile
import shutil
from unittest.mock import Mock, patch
from click.testing import CliRunner

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def create_cli_runner():
    from click.testing import CliRunner
    return CliRunner()

def create_test_repo():
    """Create a test repository with sample code."""
    temp_dir = tempfile.mkdtemp(prefix='coarch_cli_test_')
    
    # Create sample Python file
    file_path = os.path.join(temp_dir, 'test.py')
    with open(file_path, 'w') as f:
        f.write('def hello(): pass\nclass Test: pass')
        
    return temp_dir

# Mock classes
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
        return [Mock(id=0, score=0.95, file_path="/test/main.py", start_line=10, end_line=20, code="def hello(): pass", language="python")]
    def count(self):
        return 150


class TestBasicCommands(unittest.TestCase):
    """Test basic command functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()
        self.test_repo = create_test_repo()
        
    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_repo and os.path.exists(self.test_repo):
            shutil.rmtree(self.test_repo)
            
    def test_version_option(self):
        """Test --version option."""
        from cli.main import main
        
        result = self.runner.invoke(main, ['--version'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("1.0.0", result.output)
        
    def test_help_option(self):
        """Test --help option."""
        from cli.main import main
        
        result = self.runner.invoke(main, ['--help'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Coarch - Local-first code search engine", result.output)
        
    def test_index_nonexistent_path(self):
        """Test indexing non-existent path."""
        from cli.main import main
        
        result = self.runner.invoke(main, ['index', '/nonexistent/path'])
        # Click returns exit code 2 for usage errors (missing file)
        self.assertEqual(result.exit_code, 2)
        self.assertIn("does not exist", result.output)
        
    @patch('backend.hybrid_indexer.HybridIndexer', MockIndexer)
    @patch('backend.embeddings.CodeEmbedder', MockEmbedder)
    @patch('backend.faiss_index.FaissIndex', MockFaiss)
    def test_index_success(self):
        """Test successful indexing."""
        from cli.main import main
        
        result = self.runner.invoke(main, ['index', self.test_repo])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Indexing repository", result.output)
        
    @patch('backend.embeddings.CodeEmbedder', MockEmbedder)
    @patch('backend.faiss_index.FaissIndex', MockFaiss)
    def test_search_success(self):
        """Test successful search."""
        from cli.main import main
        
        result = self.runner.invoke(main, ['search', 'hello'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Searching for", result.output)
        
    @patch('backend.hybrid_indexer.HybridIndexer', MockIndexer)
    def test_delete_success(self):
        """Test successful deletion."""
        from cli.main import main
        
        result = self.runner.invoke(main, ['delete', '1'], input='y\n')
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Deleting repository", result.output)
        
    @patch('backend.hybrid_indexer.HybridIndexer', MockIndexer)
    @patch('backend.faiss_index.FaissIndex', MockFaiss)
    def test_status_success(self):
        """Test successful status."""
        from cli.main import main
        
        result = self.runner.invoke(main, ['status'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Index Statistics", result.output)
        
    @patch('backend.server.run_server')
    def test_serve_success(self, mock_server):
        """Test successful serve."""
        from cli.main import main
        
        result = self.runner.invoke(main, ['serve'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Starting Coarch server", result.output)
        
    def test_health_success(self):
        """Test successful health check."""
        from cli.main import main
        import urllib.request
        
        # Mock response with proper JSON
        mock_response = Mock()
        mock_response.status_code = 200
        json_data = b'{"status": "healthy"}'
        mock_response.read.return_value = json_data
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)
        
        with patch.object(urllib.request, 'urlopen', return_value=mock_response):
            result = self.runner.invoke(main, ['health'])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Server is healthy", result.output)
            
    @patch('backend.config.init_config')
    def test_init_success(self, mock_init):
        """Test successful init."""
        from cli.main import main
        
        # Mock config
        mock_config = Mock()
        mock_config.get_default_config_path.return_value = "/test/config.json"
        mock_config.index_path = "/test/index"
        mock_config.db_path = "/test/db"
        mock_init.return_value = mock_config
        
        result = self.runner.invoke(main, ['init'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Configuration initialized successfully", result.output)


if __name__ == '__main__':
    unittest.main()