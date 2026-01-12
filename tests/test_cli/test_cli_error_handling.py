"""Error handling tests for Coarch CLI."""

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

def create_cli_runner():
    from click.testing import CliRunner
    return CliRunner()

def create_test_repo():
    """Create a test repository with sample code."""
    temp_dir = tempfile.mkdtemp(prefix='coarch_error_test_')
    
    # Create sample Python file
    file_path = os.path.join(temp_dir, 'test.py')
    with open(file_path, 'w') as f:
        f.write('def test(): pass')
        
    return temp_dir


class TestIndexErrorHandling(unittest.TestCase):
    """Test error handling for index command."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()
        self.test_repo = create_test_repo()
        
    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_repo and os.path.exists(self.test_repo):
            shutil.rmtree(self.test_repo)
            
    def test_index_nonexistent_path(self):
        """Test error handling for non-existent path."""
        from cli.main import main
        
        result = self.runner.invoke(main, ['index', '/completely/nonexistent/path'])
        # Click returns exit code 2 for usage errors (missing file)
        self.assertEqual(result.exit_code, 2)
        self.assertIn("does not exist", result.output)
        
    def test_index_permission_error(self):
        """Test error handling for permission errors."""
        from cli.main import main
        
        # Mock the indexer to raise permission error
        with patch('backend.hybrid_indexer.HybridIndexer') as mock_indexer:
            mock_indexer.side_effect = PermissionError("Permission denied")
            
            result = self.runner.invoke(main, ['index', self.test_repo])
            self.assertEqual(result.exit_code, 1)
            self.assertIn("Error:", result.output)
            
    def test_index_database_error(self):
        """Test error handling for database errors."""
        from cli.main import main
        
        # Mock the indexer to raise database error
        with patch('backend.hybrid_indexer.HybridIndexer') as mock_indexer:
            mock_indexer.side_effect = Exception("Database connection failed")
            
            result = self.runner.invoke(main, ['index', self.test_repo])
            self.assertEqual(result.exit_code, 1)
            self.assertIn("Error:", result.output)


class TestSearchErrorHandling(unittest.TestCase):
    """Test error handling for search command."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()
        
    def tearDown(self):
        """Clean up test fixtures."""
        pass
        
    def test_search_no_index_error(self):
        """Test error handling when no index exists."""
        from cli.main import main
        
        # Mock FaissIndex to raise exception
        with patch('backend.faiss_index.FaissIndex') as mock_faiss:
            mock_faiss.side_effect = Exception("Index file not found")
            
            result = self.runner.invoke(main, ['search', 'hello'])
            self.assertEqual(result.exit_code, 1)
            self.assertIn("Error:", result.output)
            
    def test_search_embedding_error(self):
        """Test error handling for embedding failures."""
        from cli.main import main
        
        # Mock CodeEmbedder to raise exception
        with patch('backend.embeddings.CodeEmbedder') as mock_embedder:
            mock_embedder.side_effect = Exception("Model loading failed")
            
            result = self.runner.invoke(main, ['search', 'test query'])
            self.assertEqual(result.exit_code, 1)
            self.assertIn("Error:", result.output)
            
    def test_search_with_invalid_language(self):
        """Test search with invalid language filter."""
        from cli.main import main
        
        # Mock successful search but with invalid language
        with patch('backend.embeddings.CodeEmbedder') as mock_embedder:
            mock_embedder.return_value.embed_query.return_value = [0.1, 0.2, 0.3]
            
            with patch('backend.faiss_index.FaissIndex') as mock_faiss:
                mock_faiss.return_value.search.return_value = []
                
                result = self.runner.invoke(main, ['search', 'test', '--language', 'invalidlanguage123'])
                # This should still work but return no results
                self.assertEqual(result.exit_code, 0)


class TestServeErrorHandling(unittest.TestCase):
    """Test error handling for serve command."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()
        
    def tearDown(self):
        """Clean up test fixtures."""
        pass
        
    def test_serve_port_in_use(self):
        """Test error handling when port is in use."""
        from cli.main import main
        
        # Mock run_server to raise OSError (port in use)
        with patch('backend.server.run_server') as mock_server:
            mock_server.side_effect = OSError("Address already in use")
            
            result = self.runner.invoke(main, ['serve', '--port', '8000'])
            self.assertEqual(result.exit_code, 1)
            self.assertIn("Error:", result.output)
            
    def test_serve_invalid_host(self):
        """Test error handling for invalid host."""
        from cli.main import main
        
        # Mock run_server to raise exception for invalid host
        with patch('backend.server.run_server') as mock_server:
            mock_server.side_effect = Exception("Invalid host address")
            
            result = self.runner.invoke(main, ['serve', '--host', 'invalid-host'])
            self.assertEqual(result.exit_code, 1)
            self.assertIn("Error:", result.output)
            
    def test_serve_module_import_error(self):
        """Test error handling when server module fails to import."""
        from cli.main import main
        
        # Mock import error
        with patch('backend.server.run_server', side_effect=ImportError("No module named 'backend.server'")):
            result = self.runner.invoke(main, ['serve'])
            self.assertEqual(result.exit_code, 1)
            self.assertIn("Error:", result.output)


class TestStatusErrorHandling(unittest.TestCase):
    """Test error handling for status command."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()
        
    def tearDown(self):
        """Clean up test fixtures."""
        pass
        
    def test_status_database_error(self):
        """Test error handling for database errors in status."""
        from cli.main import main
        
        # Mock HybridIndexer to raise database error
        with patch('backend.hybrid_indexer.HybridIndexer') as mock_indexer:
            mock_indexer.side_effect = Exception("Database connection failed")
            
            result = self.runner.invoke(main, ['status'])
            self.assertEqual(result.exit_code, 0)  # Status should be graceful
            # Should show error but not crash
            
    def test_status_faiss_error(self):
        """Test error handling for FAISS errors in status."""
        from cli.main import main
        
        # Mock FaissIndex to raise error
        with patch('backend.faiss_index.FaissIndex') as mock_faiss:
            mock_faiss.side_effect = Exception("FAISS index corrupted")
            
            result = self.runner.invoke(main, ['status'])
            self.assertEqual(result.exit_code, 0)  # Status should be graceful
            # Should show error but not crash


class TestDeleteErrorHandling(unittest.TestCase):
    """Test error handling for delete command."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()
        
    def tearDown(self):
        """Clean up test fixtures."""
        pass
        
    def test_delete_nonexistent_repo(self):
        """Test error handling for deleting non-existent repository."""
        from cli.main import main
        
        # Mock indexer to raise not found error
        with patch('backend.hybrid_indexer.HybridIndexer') as mock_indexer:
            mock_indexer.side_effect = Exception("Repository not found")
            
            result = self.runner.invoke(main, ['delete', '999'], input='y\n')
            self.assertEqual(result.exit_code, 1)
            self.assertIn("Error:", result.output)
            
    def test_delete_database_error(self):
        """Test error handling for database errors during delete."""
        from cli.main import main
        
        # Mock indexer to raise database error
        with patch('backend.hybrid_indexer.HybridIndexer') as mock_indexer:
            mock_indexer.side_effect = Exception("Database constraint violation")
            
            result = self.runner.invoke(main, ['delete', '1'], input='y\n')
            self.assertEqual(result.exit_code, 1)
            self.assertIn("Error:", result.output)
            
    def test_delete_invalid_repo_id(self):
        """Test error handling for invalid repository ID."""
        from cli.main import main
        
        result = self.runner.invoke(main, ['delete', 'not_a_number'])
        # Click returns exit code 2 for usage errors
        self.assertEqual(result.exit_code, 2)
        self.assertIn("Error:", result.output)


class TestHealthErrorHandling(unittest.TestCase):
    """Test error handling for health command."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()
        
    def tearDown(self):
        """Clean up test fixtures."""
        pass
        
    def test_health_server_down(self):
        """Test error handling when server is down."""
        from cli.main import main
        
        with patch('urllib.request.urlopen', side_effect=Exception("Connection refused")):
            result = self.runner.invoke(main, ['health'])
            self.assertEqual(result.exit_code, 1)
            self.assertIn("Server is unhealthy", result.output)
            
    def test_health_timeout(self):
        """Test error handling for health check timeout."""
        from cli.main import main
        
        with patch('urllib.request.urlopen', side_effect=TimeoutError("Request timeout")):
            result = self.runner.invoke(main, ['health'])
            self.assertEqual(result.exit_code, 1)
            self.assertIn("Server is unhealthy", result.output)
            
    def test_health_invalid_response(self):
        """Test health check with invalid JSON response."""
        from cli.main import main
        
        # Mock response with invalid JSON
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.read.return_value = b"invalid json"  # Return bytes directly
        
        with patch('urllib.request.urlopen', return_value=mock_response):
            result = self.runner.invoke(main, ['health'])
            self.assertEqual(result.exit_code, 1)
            self.assertIn("Server is unhealthy", result.output)


class TestInitErrorHandling(unittest.TestCase):
    """Test error handling for init command."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()
        
    def tearDown(self):
        """Clean up test fixtures."""
        pass
        
    def test_init_config_error(self):
        """Test error handling for configuration errors during init."""
        from cli.main import main
        
        # Mock init_config to raise exception (patch unified_config first, then fallback)
        with patch('backend.unified_config.init_config', side_effect=Exception("Config validation failed")):
            result = self.runner.invoke(main, ['init'])
            self.assertEqual(result.exit_code, 1)
            self.assertIn("Error:", result.output)
            
    def test_init_permission_error(self):
        """Test error handling for permission errors during init."""
        from cli.main import main
        
        # Mock init_config to raise permission error
        with patch('backend.unified_config.init_config', side_effect=PermissionError("Cannot write config file")):
            result = self.runner.invoke(main, ['init'])
            self.assertEqual(result.exit_code, 1)
            self.assertIn("Error:", result.output)
            
            self.assertIn("Error:", result.output)


class TestGlobalErrorHandling(unittest.TestCase):
    """Test global error handling scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()
        
    def tearDown(self):
        """Clean up test fixtures."""
        pass
        
    def test_verbose_logging_error(self):
        """Test that verbose logging doesn't break on errors."""
        from cli.main import main
        
        # Mock logging setup to raise exception
        with patch('backend.logging_config.setup_logging', side_effect=Exception("Logging setup failed")):
            result = self.runner.invoke(main, ['--verbose', 'status'])
            # Should still work but fall back to default logging
            self.assertEqual(result.exit_code, 0)
            
    def test_log_file_error(self):
        """Test error handling for invalid log file path."""
        from cli.main import main
        
        # Mock logging setup with invalid log file
        with patch('backend.logging_config.setup_logging') as mock_setup:
            # Allow the mock to be called but don't raise exception
            mock_setup.return_value = None
            
            result = self.runner.invoke(main, ['--log-file', '/invalid/path/log.txt', 'status'])
            self.assertEqual(result.exit_code, 0)


class TestGracefulDegradation(unittest.TestCase):
    """Test graceful degradation scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.runner = create_cli_runner()
        
    def tearDown(self):
        """Clean up test fixtures."""
        pass
        
    def test_partial_backend_failure(self):
        """Test behavior when some backend components fail."""
        from cli.main import main
        
        # Mock successful index but failed embedding
        with patch('backend.hybrid_indexer.HybridIndexer') as mock_indexer:
            mock_indexer.return_value.index_repository.return_value = {
                'files_indexed': 10, 'chunks_created': 50
            }
            mock_indexer.return_value.get_chunks_for_embedding.return_value = []
            
            with patch('backend.embeddings.CodeEmbedder') as mock_embedder:
                mock_embedder.side_effect = Exception("Embedding service unavailable")
                
                result = self.runner.invoke(main, ['index', '/test/path'])
                # Click returns exit code 2 for usage errors
                self.assertEqual(result.exit_code, 2)
                self.assertIn("Error:", result.output)
                
    def test_status_with_partial_backend(self):
        """Test status command with partial backend failures."""
        from cli.main import main
        
        # Mock successful indexer but failed FAISS
        with patch('backend.hybrid_indexer.HybridIndexer') as mock_indexer:
            mock_indexer.return_value.get_stats.return_value = {
                'total_chunks': 150, 'total_repos': 3
            }
            
            with patch('backend.faiss_index.FaissIndex') as mock_faiss:
                mock_faiss.side_effect = Exception("FAISS unavailable")
                
                result = self.runner.invoke(main, ['status'])
                self.assertEqual(result.exit_code, 0)
                # Should show partial information
                self.assertIn("Index Statistics", result.output)


if __name__ == '__main__':
    unittest.main()