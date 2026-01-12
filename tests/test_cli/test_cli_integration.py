"""Integration tests for Coarch CLI workflows."""

import os
import sys
import unittest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Enable test mode for CLI error handling
os.environ['COARCH_TEST_MODE'] = '1'

# Simple mock classes for integration testing
class MockIndexer:
    def index_repository(self, path, name=None):
        return {'files_indexed': 10, 'chunks_created': 50, 'repo_id': 1}
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

def create_cli_runner():
    from click.testing import CliRunner
    return CliRunner()

def create_test_repo():
    """Create a test repository with sample code."""
    temp_dir = tempfile.mkdtemp(prefix='coarch_integration_test_')
    
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
        file_path = os.path.join(temp_dir, filename)
        with open(file_path, 'w') as f:
            f.write(content)
            
    return temp_dir


class TestFullWorkflow(unittest.TestCase):
    """Test full CLI workflow integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()
        self.test_repo = create_test_repo()
        
    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_repo and os.path.exists(self.test_repo):
            shutil.rmtree(self.test_repo)
            
    @patch('backend.hybrid_indexer.HybridIndexer', MockIndexer)
    @patch('backend.embeddings.CodeEmbedder', MockEmbedder)
    @patch('backend.faiss_index.FaissIndex', MockFaiss)
    def test_full_index_and_search_workflow(self):
        """Test full index and search workflow."""
        from cli.main import main
        
        # Step 1: Index the repository
        result = self.runner.invoke(main, ['index', self.test_repo])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Indexing repository", result.output)
        self.assertIn("Files indexed", result.output)
        
        # Step 2: Search for code
        result = self.runner.invoke(main, ['search', 'hello function'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Searching for", result.output)
        self.assertIn("Found", result.output)
        
        # Step 3: Check status
        result = self.runner.invoke(main, ['status'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Index Statistics", result.output)
        self.assertIn("Total chunks", result.output)
        
    @patch('backend.hybrid_indexer.HybridIndexer', MockIndexer)
    @patch('backend.embeddings.CodeEmbedder', MockEmbedder)
    @patch('backend.faiss_index.FaissIndex', MockFaiss)
    def test_multiple_repositories_workflow(self):
        """Test multiple repositories workflow."""
        from cli.main import main
        
        # Create second test repo
        test_repo2 = create_test_repo()
        
        try:
            # Index first repository
            result = self.runner.invoke(main, ['index', self.test_repo, '--name', 'repo1'])
            self.assertEqual(result.exit_code, 0)
            
            # Index second repository
            result = self.runner.invoke(main, ['index', test_repo2, '--name', 'repo2'])
            self.assertEqual(result.exit_code, 0)
            
            # Check status with multiple repos
            result = self.runner.invoke(main, ['status'])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Total repos", result.output)
            
        finally:
            if test_repo2 and os.path.exists(test_repo2):
                shutil.rmtree(test_repo2)
                
    @patch('backend.hybrid_indexer.HybridIndexer', MockIndexer)
    def test_repository_deletion_workflow(self):
        """Test repository deletion workflow."""
        from cli.main import main
        
        # Delete a repository
        result = self.runner.invoke(main, ['delete', '1'], input='y\n')
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Deleting repository 1", result.output)
        self.assertIn("Deleted", result.output)


class TestServerIntegration(unittest.TestCase):
    """Test server integration with CLI."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()
        
    def tearDown(self):
        """Clean up test fixtures."""
        pass
        
    @patch('backend.server.run_server')
    def test_server_lifecycle_integration(self, mock_server):
        """Test server lifecycle integration."""
        from cli.main import main
        
        # Start server
        result = self.runner.invoke(main, ['serve', '--port', '9000'])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Starting Coarch server", result.output)
        mock_server.assert_called_once_with(host="0.0.0.0", port=9000, reload=False)
        
    @patch('backend.server.run_server')
    def test_server_with_reload_integration(self, mock_server):
        """Test server with reload option."""
        from cli.main import main
        
        result = self.runner.invoke(main, ['serve', '--reload'])
        self.assertEqual(result.exit_code, 0)
        mock_server.assert_called_once_with(host="0.0.0.0", port=8000, reload=True)
        
    def test_health_check_integration(self):
        """Test health check integration."""
        from cli.main import main
        
        # Mock urllib.request.urlopen
        mock_response = Mock()
        mock_response.status_code = 200
        import json
        mock_response.read.return_value = json.dumps({"status": "healthy", "vectors": 150}).encode('utf-8')
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)
        
        with patch('urllib.request.urlopen', return_value=mock_response):
            result = self.runner.invoke(main, ['health'])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Server is healthy", result.output)
            self.assertIn("Status: healthy", result.output)


class TestErrorRecovery(unittest.TestCase):
    """Test error recovery scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()
        self.test_repo = create_test_repo()
        
    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_repo and os.path.exists(self.test_repo):
            shutil.rmtree(self.test_repo)
            
    def test_search_with_no_index(self):
        """Test search when no index exists."""
        from cli.main import main
        
        # Mock FAISS to raise exception (no index)
        with patch('backend.faiss_index.FaissIndex.__init__', side_effect=Exception("Index not found")):
            result = self.runner.invoke(main, ['search', 'hello'])
            self.assertEqual(result.exit_code, 1)
            self.assertIn("Error:", result.output)
            
    def test_index_nonexistent_path_recovery(self):
        """Test recovery from non-existent path error."""
        from cli.main import main
        
        result = self.runner.invoke(main, ['index', '/nonexistent/path'])
        # Click returns exit code 2 for usage errors
        self.assertEqual(result.exit_code, 2)
        self.assertIn("does not exist", result.output)
        
    def test_health_check_server_down(self):
        """Test health check when server is down."""
        from cli.main import main
        
        with patch('urllib.request.urlopen', side_effect=Exception("Connection refused")):
            result = self.runner.invoke(main, ['health'])
            self.assertEqual(result.exit_code, 1)
            self.assertIn("Server is unhealthy", result.output)


class TestConfigurationIntegration(unittest.TestCase):
    """Test configuration integration scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()
        
    def tearDown(self):
        """Clean up test fixtures."""
        pass
        
    def test_init_configuration(self):
        """Test configuration initialization."""
        from cli.main import main
        from backend.config import CoarchConfig
        
        # Mock init_config
        mock_config = Mock()
        mock_config.get_default_config_path.return_value = "/test/config.json"
        mock_config.index_path = "/test/index"
        mock_config.db_path = "/test/db"
        
        with patch('backend.config.init_config', return_value=mock_config):
            result = self.runner.invoke(main, ['init'])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Configuration initialized successfully", result.output)


if __name__ == '__main__':
    unittest.main()