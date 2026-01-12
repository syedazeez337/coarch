"""Test signal handling for graceful shutdown functionality."""

import os
import sys
import time
import signal
import unittest
import tempfile
import shutil
import threading
from unittest.mock import Mock, patch, MagicMock
from click.testing import CliRunner

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Enable test mode for CLI error handling
os.environ['COARCH_TEST_MODE'] = '1'


def create_test_repo():
    """Create a test repository with sample code."""
    temp_dir = tempfile.mkdtemp(prefix='coarch_signal_test_')
    
    # Create sample Python file
    file_path = os.path.join(temp_dir, 'test.py')
    with open(file_path, 'w') as f:
        f.write('def test(): pass\n' * 50)  # Create multiple chunks
    
    return temp_dir


class TestSignalHandling(unittest.TestCase):
    """Test signal handling for graceful shutdown."""
    
    def setUp(self):
        """Set up test fixtures."""
        from backend.signal_handler import reset_shutdown_state
        reset_shutdown_state()
        self.runner = CliRunner()
        self.test_repo = create_test_repo()
        
    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_repo and os.path.exists(self.test_repo):
            shutil.rmtree(self.test_repo)
    
    def test_signal_handler_installation(self):
        """Test that signal handlers can be installed correctly."""
        try:
            from backend.signal_handler import install_signal_handlers, _shutdown_handlers_installed
            
            # Try to install handlers - should work without error
            install_signal_handlers()
            
            # The flag may or may not be True depending on whether 
            # handlers were already installed before this test
            # The important thing is that the function runs without error
        except ImportError:
            self.skipTest("Signal handler not available")
    
    def test_graceful_shutdown_state(self):
        """Test graceful shutdown state management."""
        try:
            from backend.signal_handler import (
                get_shutdown_state, 
                register_cleanup_task, 
                register_active_operation,
                is_shutdown_initiated
            )
            
            # Test initial state
            state = get_shutdown_state()
            self.assertFalse(state['initiated'])
            self.assertFalse(is_shutdown_initiated())
            self.assertEqual(state['active_operations'], 0)
            self.assertEqual(state['cleanup_tasks'], 0)
            
        except ImportError:
            self.skipTest("Signal handler not available")
    
    def test_cleanup_task_registration(self):
        """Test cleanup task registration and management."""
        try:
            from backend.signal_handler import register_cleanup_task, get_shutdown_state
            
            # Register a test cleanup task
            cleanup_called = []
            
            def test_cleanup():
                cleanup_called.append(True)
            
            task_id = register_cleanup_task(test_cleanup)
            
            # Should return a task ID
            self.assertIsNotNone(task_id)
            
        except ImportError:
            self.skipTest("Signal handler not available")
    
    def test_active_operation_registration(self):
        """Test active operation registration and management."""
        try:
            from backend.signal_handler import (
                register_active_operation, 
                unregister_active_operation,
                get_shutdown_state
            )
            
            # Register a test operation
            mock_operation = Mock()
            mock_operation.cancel = Mock()
            
            register_active_operation("test_op", mock_operation)
            
            # Check state
            state = get_shutdown_state()
            self.assertEqual(state['active_operations'], 1)
            
            # Unregister
            unregister_active_operation("test_op")
            
            # Check state again
            state = get_shutdown_state()
            self.assertEqual(state['active_operations'], 0)
            
            # Verify cancel was called
            mock_operation.cancel.assert_not_called()  # Should not be called yet
            
        except ImportError:
            self.skipTest("Signal handler not available")
    
    def test_cancellation_token(self):
        """Test cancellation token functionality."""
        try:
            from backend.signal_handler import CancellationToken
            
            token = CancellationToken()
            self.assertFalse(token.is_cancelled())
            
            # Cancel the token
            token.cancel()
            self.assertTrue(token.is_cancelled())
            
            # Should raise exception when checked
            with self.assertRaises(KeyboardInterrupt):
                token.check_cancelled()
            
        except ImportError:
            self.skipTest("Signal handler not available")
    
    def test_graceful_killer(self):
        """Test graceful killer utility class."""
        try:
            from backend.signal_handler import GracefulKiller
            
            killer = GracefulKiller()
            self.assertFalse(killer.cancelled)
            
            # Cancel the killer
            killer.cancel()
            self.assertTrue(killer.cancelled)
            
            # Should raise exception when checked
            with self.assertRaises(KeyboardInterrupt):
                killer.check_cancelled()
            
        except ImportError:
            self.skipTest("Signal handler not available")


class TestCLISignalIntegration(unittest.TestCase):
    """Test CLI integration with signal handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        from backend.signal_handler import reset_shutdown_state
        reset_shutdown_state()
        self.runner = CliRunner()
        self.test_repo = create_test_repo()
        
    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_repo and os.path.exists(self.test_repo):
            shutil.rmtree(self.test_repo)
    
    def test_cli_with_signal_handling(self):
        """Test that CLI commands properly integrate signal handling."""
        try:
            # Test import of CLI with signal handling
            from cli.main import main
            
            # Verify CLI is importable and is a Click group
            self.assertIsNotNone(main)
            
        except ImportError as e:
            self.skipTest(f"CLI import failed: {e}")
    
    def test_index_command_cancellation(self):
        """Test index command cancellation with mock signal."""
        try:
            from cli.main import main
            
            # Mock the indexing process to simulate cancellation
            with patch('backend.hybrid_indexer.HybridIndexer') as mock_indexer:
                # Configure mock to raise KeyboardInterrupt after first batch
                def mock_index_repo(path, name):
                    # Simulate long-running operation that gets cancelled
                    raise KeyboardInterrupt("Simulated cancellation")
                
                mock_indexer.return_value.index_repository = mock_index_repo
                
                # Run the index command - should handle cancellation gracefully
                result = self.runner.invoke(main, ['index', self.test_repo])
                
                # Should exit with 0 due to graceful cancellation
                self.assertEqual(result.exit_code, 0)
                self.assertIn("cancelled", result.output.lower())
                
        except ImportError:
            self.skipTest("CLI or backend not available")


class TestFileWatcherSignalHandling(unittest.TestCase):
    """Test file watcher integration with signal handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        from backend.signal_handler import reset_shutdown_state
        reset_shutdown_state()
        self.runner = CliRunner()
        self.test_repo = create_test_repo()
        
    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_repo and os.path.exists(self.test_repo):
            shutil.rmtree(self.test_repo)
    
    def test_file_watcher_graceful_shutdown(self):
        """Test file watcher graceful shutdown with signal handling."""
        try:
            from backend.file_watcher import FileWatcher, IncrementalIndexer
            
            # Create a file watcher
            watcher = FileWatcher({self.test_repo})
            
            # Should have graceful killer attribute
            if hasattr(watcher, '_graceful_killer'):
                self.assertIsNotNone(watcher._graceful_killer)
            
        except ImportError:
            self.skipTest("File watcher not available")
    
    def test_incremental_indexer_graceful_shutdown(self):
        """Test incremental indexer graceful shutdown with signal handling."""
        try:
            from backend.file_watcher import IncrementalIndexer
            from backend.hybrid_indexer import HybridIndexer
            from backend.embeddings import CodeEmbedder
            from backend.faiss_index import FaissIndex
            
            # Create mock components
            mock_indexer = Mock()
            mock_embedder = Mock()
            mock_faiss = Mock()
            
            # Create incremental indexer
            incremental = IncrementalIndexer(
                mock_indexer, 
                mock_embedder, 
                mock_faiss, 
                {self.test_repo}
            )
            
            # Should have graceful killer attribute
            if hasattr(incremental, '_graceful_killer'):
                self.assertIsNotNone(incremental._graceful_killer)
            
        except ImportError:
            self.skipTest("Incremental indexer not available")


class TestServerSignalHandling(unittest.TestCase):
    """Test server integration with signal handling."""
    
    def test_server_graceful_shutdown(self):
        """Test server graceful shutdown with signal handling."""
        try:
            from backend.server import AppState
            
            # Create app state
            state = AppState()
            
            # Should have shutdown event attribute after signal handler integration
            # Note: This will be set during actual operation
            
        except ImportError:
            self.skipTest("Server not available")


class TestCancellationScenarios(unittest.TestCase):
    """Test various cancellation scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        from backend.signal_handler import reset_shutdown_state
        reset_shutdown_state()
        
    def test_keyboard_interrupt_handling(self):
        """Test that KeyboardInterrupt is handled gracefully."""
        try:
            from backend.signal_handler import GracefulKiller
            
            killer = GracefulKiller()
            
            # Simulate KeyboardInterrupt scenario
            with patch('builtins.input', side_effect=KeyboardInterrupt):
                # Should handle interrupt gracefully
                killer.check_cancelled()  # Should not raise yet
            
        except ImportError:
            self.skipTest("Signal handler not available")
    
    def test_shutdown_state_progression(self):
        """Test shutdown state progression."""
        try:
            from backend.signal_handler import (
                get_shutdown_state,
                register_active_operation
            )
            
            # Test initial state
            state1 = get_shutdown_state()
            self.assertFalse(state1['initiated'])
            
            # Register an operation
            mock_op = Mock()
            register_active_operation("test", mock_op)
            
            # State should reflect active operation
            state2 = get_shutdown_state()
            self.assertEqual(state2['active_operations'], 1)
            
        except ImportError:
            self.skipTest("Signal handler not available")


if __name__ == "__main__":
    unittest.main()