"""Test progress tracking functionality."""

import time
import pytest
from unittest.mock import Mock, patch

from backend.progress_tracker import (
    ProgressTracker,
    ProgressCallback,
    ETAEstimator,
    track_progress,
    create_file_progress_bar,
    create_embedding_progress_bar,
    create_faiss_progress_bar,
    create_deletion_progress_bar,
    should_show_progress,
)


class TestProgressTracker:
    """Test the ProgressTracker class."""

    def test_init(self):
        """Test progress tracker initialization."""
        tracker = ProgressTracker(min_duration_seconds=3.0)
        assert tracker.min_duration_seconds == 3.0

    def test_should_show_progress_small_operation(self):
        """Test that small operations don't show progress."""
        tracker = ProgressTracker(min_duration_seconds=5.0)
        
        # Small operations should not show progress
        assert not tracker.should_show_progress(10)
        assert not tracker.should_show_progress(1)
        assert not tracker.should_show_progress(0)

    def test_should_show_progress_large_operation(self):
        """Test that large operations show progress."""
        tracker = ProgressTracker(min_duration_seconds=5.0)
        
        # Large operations should show progress
        assert tracker.should_show_progress(1000)
        assert tracker.should_show_progress(100)

    def test_should_show_progress_none_total(self):
        """Test behavior when total is None."""
        tracker = ProgressTracker(min_duration_seconds=5.0)
        
        # When total is None, always show progress
        assert tracker.should_show_progress(None)

    def test_estimate_duration(self):
        """Test duration estimation."""
        tracker = ProgressTracker()
        
        # Test different operation types
        file_time = tracker._estimate_duration(1000, 'file_indexing')  # Files
        embedding_time = tracker._estimate_duration(100, 'embedding_generation')  # Embeddings
        faiss_time = tracker._estimate_duration(1000, 'faiss_operations')  # FAISS
        
        # Embedding operations should take longer
        assert embedding_time > file_time
        assert embedding_time > faiss_time

    def test_timer_functionality(self):
        """Test timer start and elapsed time."""
        tracker = ProgressTracker()
        
        # Timer should not be started yet
        assert tracker.get_elapsed_time() == 0.0
        
        # Start timer
        tracker.start_timer()
        
        # Should have some elapsed time after a short delay
        time.sleep(0.01)
        elapsed = tracker.get_elapsed_time()
        assert elapsed >= 0.0
        assert elapsed < 1.0  # Should be less than 1 second

    def test_track_operation_context_manager(self):
        """Test operation tracking context manager."""
        tracker = ProgressTracker()
        
        # Test with large operation (should show progress)
        with tracker.track_operation("test", total_items=1000) as progress:
            assert progress is not None
        
        # Test with small operation (may not show progress)
        with tracker.track_operation("test", total_items=1) as progress:
            # May return None or dummy progress
            pass

    def test_create_progress_bar(self):
        """Test progress bar creation."""
        tracker = ProgressTracker()
        
        # Create file progress bar
        bar = tracker.create_file_progress_bar(100)
        assert bar is not None
        
        # Create embedding progress bar
        bar = tracker.create_embedding_progress_bar(50)
        assert bar is not None
        
        # Create FAISS progress bar
        bar = tracker.create_faiss_progress_bar(200)
        assert bar is not None
        
        # Create deletion progress bar
        bar = tracker.create_deletion_progress_bar(75)
        assert bar is not None


class TestProgressCallback:
    """Test the ProgressCallback class."""

    def test_init(self):
        """Test callback initialization."""
        mock_bar = Mock()
        callback = ProgressCallback(mock_bar, "test operation")
        
        assert callback.progress_bar == mock_bar
        assert callback.operation_name == "test operation"

    def test_call(self):
        """Test callback invocation."""
        mock_bar = Mock()
        callback = ProgressCallback(mock_bar)
        
        # Should not raise any errors
        callback(10, 100)
        callback(50)
        callback(0, 0)

    def test_set_description(self):
        """Test description setting."""
        mock_bar = Mock()
        callback = ProgressCallback(mock_bar)
        
        callback.set_description("new description")
        mock_bar.set_description.assert_called_once_with("new description")

    def test_set_postfix(self):
        """Test postfix setting."""
        mock_bar = Mock()
        callback = ProgressCallback(mock_bar)
        
        callback.set_postfix(status="processing", rate="100 items/s")
        mock_bar.set_postfix.assert_called_once_with(
            status="processing", rate="100 items/s"
        )

    def test_close(self):
        """Test callback close."""
        mock_bar = Mock()
        callback = ProgressCallback(mock_bar)
        
        callback.close()
        mock_bar.close.assert_called_once()


class TestETAEstimator:
    """Test the ETAEstimator class."""

    def test_init(self):
        """Test estimator initialization."""
        estimator = ETAEstimator(100)
        assert estimator.total_items == 100
        assert estimator.processed_items == 0

    def test_update(self):
        """Test processed items update."""
        estimator = ETAEstimator(100)
        
        estimator.update(25)
        assert estimator.processed_items == 25
        
        estimator.update(50)
        assert estimator.processed_items == 50

    def test_get_eta_no_processed(self):
        """Test ETA when no items processed."""
        estimator = ETAEstimator(100)
        
        assert estimator.get_eta() is None

    def test_get_eta_string(self):
        """Test formatted ETA string."""
        estimator = ETAEstimator(100)
        
        # When no ETA available (0 items processed)
        assert estimator.get_eta_string() == "Calculating..."

    def test_get_rate_string(self):
        """Test processing rate string."""
        estimator = ETAEstimator(100)
        
        # Should return rate string even with no processed items
        rate_str = estimator.get_rate_string()
        assert "items/s" in rate_str


class TestConvenienceFunctions:
    """Test convenience functions."""

    @patch('backend.progress_tracker._global_tracker')
    def test_track_progress_function(self, mock_tracker):
        """Test the track_progress convenience function."""
        mock_progress = Mock()
        mock_tracker.track_operation.return_value = mock_progress
        
        result = track_progress("test", total_items=100, unit="items")
        
        mock_tracker.track_operation.assert_called_once_with("test", 100)
        assert result == mock_progress

    @patch('backend.progress_tracker._global_tracker')
    def test_create_file_progress_bar(self, mock_tracker):
        """Test create_file_progress_bar function."""
        mock_bar = Mock()
        mock_tracker.create_file_progress_bar.return_value = mock_bar
        
        result = create_file_progress_bar(50)
        
        mock_tracker.create_file_progress_bar.assert_called_once_with(50)
        assert result == mock_bar

    @patch('backend.progress_tracker._global_tracker')
    def test_create_embedding_progress_bar(self, mock_tracker):
        """Test create_embedding_progress_bar function."""
        mock_bar = Mock()
        mock_tracker.create_embedding_progress_bar.return_value = mock_bar
        
        result = create_embedding_progress_bar(100)
        
        mock_tracker.create_embedding_progress_bar.assert_called_once_with(100)
        assert result == mock_bar

    @patch('backend.progress_tracker._global_tracker')
    def test_create_faiss_progress_bar(self, mock_tracker):
        """Test create_faiss_progress_bar function."""
        mock_bar = Mock()
        mock_tracker.create_faiss_progress_bar.return_value = mock_bar
        
        result = create_faiss_progress_bar(200)
        
        mock_tracker.create_faiss_progress_bar.assert_called_once_with(200)
        assert result == mock_bar

    @patch('backend.progress_tracker._global_tracker')
    def test_create_deletion_progress_bar(self, mock_tracker):
        """Test create_deletion_progress_bar function."""
        mock_bar = Mock()
        mock_tracker.create_deletion_progress_bar.return_value = mock_bar
        
        result = create_deletion_progress_bar(75)
        
        mock_tracker.create_deletion_progress_bar.assert_called_once_with(75)
        assert result == mock_bar

    @patch('backend.progress_tracker._global_tracker')
    def test_should_show_progress_function(self, mock_tracker):
        """Test should_show_progress function."""
        mock_tracker.should_show_progress.return_value = True
        
        result = should_show_progress(100)
        
        mock_tracker.should_show_progress.assert_called_once_with(100)
        assert result is True


class TestProgressBarIntegration:
    """Test integration with actual tqdm progress bars."""

    def test_tqdm_progress_bar_creation(self):
        """Test that progress bars can be created with tqdm."""
        tracker = ProgressTracker()
        
        # This should work without errors when tqdm is available
        try:
            from tqdm import tqdm
            bar = tracker.create_progress_bar("test", total=100)
            assert bar.total == 100
            bar.close()
        except ImportError:
            pytest.skip("tqdm not available")

    def test_progress_bar_with_realistic_timing(self):
        """Test progress bar behavior with realistic operation timing."""
        tracker = ProgressTracker(min_duration_seconds=0.1)  # Short threshold for testing
        
        # Create a realistic operation simulation
        items = list(range(50))
        
        try:
            from tqdm import tqdm
            
            with tracker.track_operation("test operation", total_items=50) as progress:
                if progress:
                    for i, item in enumerate(items):
                        # Simulate work
                        time.sleep(0.001)
                        progress.update(1)
                        progress.set_description(f"Processing item {i+1}/50")
        except ImportError:
            pytest.skip("tqdm not available")


if __name__ == "__main__":
    pytest.main([__file__])