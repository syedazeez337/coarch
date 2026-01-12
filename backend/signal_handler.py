"""Signal handling for graceful shutdown with cleanup capabilities."""

import os
import signal
import sys
import time
import threading
import atexit
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager

from .logging_config import get_logger

logger = get_logger(__name__)


class ShutdownReason(Enum):
    """Reasons for shutdown."""
    USER_INTERRUPT = "user_interrupt"  # Ctrl+C
    SIGTERM = "sigterm"  # System termination signal
    CLEAN_EXIT = "clean_exit"  # Normal exit
    ERROR_EXIT = "error_exit"  # Unhandled error


@dataclass
class ShutdownState:
    """Global shutdown state and tracking."""
    reason: ShutdownReason = ShutdownReason.CLEAN_EXIT
    initiated: bool = False
    completed: bool = False
    start_time: float = field(default_factory=time.time)
    cleanup_tasks: List[Callable] = field(default_factory=list)
    active_operations: Dict[str, Any] = field(default_factory=dict)
    force_kill_scheduled: bool = False
    shutdown_timeout: float = 10.0  # seconds
    lock: threading.Lock = field(default_factory=threading.Lock)


# Global shutdown state
_shutdown_state = ShutdownState()
_shutdown_handlers_installed = False


def reset_shutdown_state() -> None:
    """Reset shutdown state for testing purposes."""
    global _shutdown_state, _shutdown_handlers_installed
    _shutdown_state = ShutdownState()
    _shutdown_handlers_installed = False


def install_signal_handlers() -> None:
    """Install signal handlers for graceful shutdown."""
    global _shutdown_handlers_installed
    
    if _shutdown_handlers_installed:
        return
    
    def signal_handler(signum, frame):
        """Handle shutdown signals."""
        with _shutdown_state.lock:
            if _shutdown_state.initiated:
                logger.warning("Shutdown already in progress, forcing exit")
                os._exit(1)
            
            if signum == signal.SIGINT:
                _shutdown_state.reason = ShutdownReason.USER_INTERRUPT
                logger.info("Received SIGINT (Ctrl+C), initiating graceful shutdown...")
            elif signum == signal.SIGTERM:
                _shutdown_state.reason = ShutdownReason.SIGTERM
                logger.info("Received SIGTERM, initiating graceful shutdown...")
            else:
                logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
            
            _shutdown_state.initiated = True
        
        # Schedule force kill if shutdown takes too long
        def schedule_force_kill():
            time.sleep(_shutdown_state.shutdown_timeout)
            with _shutdown_state.lock:
                if not _shutdown_state.completed and not _shutdown_state.force_kill_scheduled:
                    _shutdown_state.force_kill_scheduled = True
                    logger.error(f"Shutdown timeout ({_shutdown_state.shutdown_timeout}s), forcing exit")
                    os._exit(1)
        
        threading.Thread(target=schedule_force_kill, daemon=True).start()
        
        # Initiate graceful shutdown
        initiate_graceful_shutdown()
    
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        _shutdown_handlers_installed = True
        logger.debug("Signal handlers installed for graceful shutdown")
    except Exception as e:
        logger.warning(f"Could not install signal handlers: {e}")


def initiate_graceful_shutdown() -> None:
    """Initiate graceful shutdown process."""
    with _shutdown_state.lock:
        if _shutdown_state.completed:
            return
        
        start_time = time.time()
        logger.info(f"Starting graceful shutdown ({_shutdown_state.reason.value})")
    
    try:
        # Cancel all active operations
        cancel_active_operations()
        
        # Run cleanup tasks
        run_cleanup_tasks()
        
        # Mark shutdown as completed
        with _shutdown_state.lock:
            _shutdown_state.completed = True
            elapsed = time.time() - start_time
            logger.info(f"Graceful shutdown completed in {elapsed:.2f}s")
    
    except Exception as e:
        logger.error(f"Error during graceful shutdown: {e}")
        with _shutdown_state.lock:
            _shutdown_state.completed = True
    
    # Exit cleanly
    sys.exit(0)


def cancel_active_operations() -> None:
    """Cancel all active operations."""
    with _shutdown_state.lock:
        operations = list(_shutdown_state.active_operations.items())
    
    if operations:
        logger.info(f"Cancelling {len(operations)} active operations...")
        
        for op_id, operation in operations:
            try:
                if hasattr(operation, 'cancel'):
                    operation.cancel()
                elif hasattr(operation, 'stop'):
                    operation.stop()
                elif hasattr(operation, 'shutdown'):
                    operation.shutdown()
                logger.debug(f"Cancelled operation: {op_id}")
            except Exception as e:
                logger.warning(f"Error cancelling operation {op_id}: {e}")


def run_cleanup_tasks() -> None:
    """Run all registered cleanup tasks."""
    with _shutdown_state.lock:
        tasks = list(_shutdown_state.cleanup_tasks)
    
    if tasks:
        logger.info(f"Running {len(tasks)} cleanup tasks...")
        
        for i, task in enumerate(tasks):
            try:
                logger.debug(f"Running cleanup task {i+1}/{len(tasks)}")
                task()
            except Exception as e:
                logger.warning(f"Error in cleanup task {i+1}: {e}")


def register_cleanup_task(task: Callable) -> str:
    """Register a cleanup task to run during shutdown.
    
    Args:
        task: Callable that will be invoked during shutdown
        
    Returns:
        Task ID for later unregistration
    """
    import uuid
    
    task_id = str(uuid.uuid4())[:8]
    
    with _shutdown_state.lock:
        _shutdown_state.cleanup_tasks.append(task)
    
    logger.debug(f"Registered cleanup task: {task_id}")
    return task_id


def unregister_cleanup_task(task_id: str) -> bool:
    """Unregister a cleanup task.
    
    Args:
        task_id: ID returned by register_cleanup_task
        
    Returns:
        True if task was found and removed
    """
    # Note: In a full implementation, we'd track task IDs
    # For now, this is a placeholder
    logger.debug(f"Unregistering cleanup task: {task_id}")
    return True


def register_active_operation(op_id: str, operation: Any) -> None:
    """Register an active operation for cancellation during shutdown.
    
    Args:
        op_id: Unique identifier for the operation
        operation: Operation object with cancel/stop/shutdown method
    """
    with _shutdown_state.lock:
        _shutdown_state.active_operations[op_id] = operation
    
    logger.debug(f"Registered active operation: {op_id}")


def unregister_active_operation(op_id: str) -> None:
    """Unregister an active operation.
    
    Args:
        op_id: ID of operation to unregister
    """
    with _shutdown_state.lock:
        _shutdown_state.active_operations.pop(op_id, None)
    
    logger.debug(f"Unregistered active operation: {op_id}")


def is_shutdown_initiated() -> bool:
    """Check if graceful shutdown has been initiated."""
    with _shutdown_state.lock:
        return _shutdown_state.initiated


def should_cancel_operation() -> bool:
    """Check if current operation should be cancelled due to shutdown.
    
    Returns:
        True if operation should be cancelled
    """
    return is_shutdown_initiated()


@contextmanager
def cancellable_operation(op_id: str, operation: Any):
    """Context manager for cancellable operations.
    
    Args:
        op_id: Unique identifier for the operation
        operation: Operation object that supports cancellation
    """
    register_active_operation(op_id, operation)
    try:
        yield operation
    finally:
        unregister_active_operation(op_id)


class CancellationToken:
    """Token for checking cancellation status."""
    
    def __init__(self):
        self._cancelled = False
        self._lock = threading.Lock()
    
    def cancel(self):
        """Cancel the operation."""
        with self._lock:
            self._cancelled = True
    
    def is_cancelled(self) -> bool:
        """Check if operation has been cancelled."""
        with self._lock:
            return self._cancelled
    
    def check_cancelled(self):
        """Raise exception if operation has been cancelled."""
        if self.is_cancelled():
            raise KeyboardInterrupt("Operation cancelled")


def check_shutdown_cancelled() -> bool:
    """Check if shutdown has been initiated and operation should be cancelled.
    
    Returns:
        True if operation should be cancelled
    """
    return should_cancel_operation()


class GracefulKiller:
    """Utility class for operations that want to be cancellable."""
    
    def __init__(self):
        self.cancelled = False
    
    def cancel(self):
        """Cancel this operation."""
        self.cancelled = True
    
    def check_cancelled(self):
        """Check if this operation has been cancelled or shutdown initiated.
        
        Raises:
            KeyboardInterrupt: If cancelled or shutdown initiated
        """
        if self.cancelled or check_shutdown_cancelled():
            raise KeyboardInterrupt("Operation cancelled")


def setup_atexit_handlers():
    """Set up atexit handlers for cleanup."""
    def cleanup_on_exit():
        """Cleanup function called on normal exit."""
        logger.debug("Performing exit cleanup")
        # Run cleanup tasks but don't block
        try:
            run_cleanup_tasks()
        except Exception as e:
            logger.error(f"Error in exit cleanup: {e}")
    
    atexit.register(cleanup_on_exit)


# Initialize signal handlers and atexit handlers
install_signal_handlers()
setup_atexit_handlers()


def get_shutdown_state() -> Dict[str, Any]:
    """Get current shutdown state for monitoring."""
    with _shutdown_state.lock:
        return {
            "reason": _shutdown_state.reason.value,
            "initiated": _shutdown_state.initiated,
            "completed": _shutdown_state.completed,
            "elapsed_seconds": time.time() - _shutdown_state.start_time,
            "active_operations": len(_shutdown_state.active_operations),
            "cleanup_tasks": len(_shutdown_state.cleanup_tasks),
            "shutdown_timeout": _shutdown_state.shutdown_timeout,
        }