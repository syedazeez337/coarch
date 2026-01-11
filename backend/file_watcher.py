"""Real-time file watching for incremental indexing."""

import os
import time
import hashlib
from pathlib import Path
from typing import Set, Callable, Optional, Dict, List, Any
from dataclasses import dataclass, field
import threading
from queue import Queue, Empty

from .logging_config import get_logger

logger = get_logger(__name__)

CHUNK_SIZE = 50
OVERLAP = 10


@dataclass
class FileEvent:
    """A file system event."""

    event_type: str
    file_path: str
    timestamp: float = field(default_factory=time.time)
    file_hash: Optional[str] = None


class FileWatcher:
    """Watch a directory for file changes with proper error handling."""

    DEFAULT_IGNORE_PATTERNS = {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        "*.pyc",
        "*.pyo",
        "*.so",
        "*.o",
        "*.a",
        "*.lib",
        "build",
        "dist",
        ".next",
        "out",
        "target",
        ".idea",
        ".vscode",
        "*.egg-info",
        "*.whl",
        "*.tar.gz",
    }

    def __init__(
        self,
        paths: Set[str],
        ignore_patterns: Optional[Set[str]] = None,
        debounce_ms: int = 100,
        poll_interval: float = 0.5,
    ):
        """Initialize the file watcher.

        Args:
            paths: Paths to watch
            ignore_patterns: Patterns to ignore
            debounce_ms: Debounce time in milliseconds
            poll_interval: Poll interval in seconds
        """
        self.paths = {os.path.abspath(p) for p in paths if os.path.exists(p)}
        self.ignore_patterns = ignore_patterns or self.DEFAULT_IGNORE_PATTERNS.copy()
        self.debounce_ms = debounce_ms
        self.poll_interval = poll_interval
        self._callbacks: List[Callable[[FileEvent], None]] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._event_queue: Queue = Queue()
        self._last_events: Dict[str, float] = {}
        self._file_hashes: Dict[str, str] = {}
        self._lock = threading.Lock()

    def start(self):
        """Start watching for changes."""
        if self._running:
            logger.warning("File watcher already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()
        logger.info(f"Started file watcher for {len(self.paths)} paths")

    def stop(self):
        """Stop watching for changes."""
        if not self._running:
            return

        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

        logger.info("File watcher stopped")

    def add_callback(self, callback: Callable[[FileEvent], None]):
        """Add a callback for file events."""
        self._callbacks.append(callback)

    def _compute_file_hash(self, file_path: str) -> Optional[str]:
        """Compute hash of file contents."""
        try:
            with open(file_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.debug(f"Failed to hash file {file_path}: {e}")
            return None

    def _watch_loop(self):
        """Main watching loop."""
        while self._running:
            try:
                events = self._scan_changes()
                for event in events:
                    self._event_queue.put(event)

                self._process_queue()

                time.sleep(self.poll_interval)
            except Exception as e:
                logger.exception(f"Error in watch loop: {e}")
                time.sleep(1.0)

    def _process_queue(self):
        """Process events from the queue."""
        while True:
            try:
                event = self._event_queue.get_nowait()
            except Empty:
                break

            now = time.time()
            key = f"{event.event_type}:{event.file_path}"

            with self._lock:
                last_time = self._last_events.get(key, 0)
                if (now - last_time) < (self.debounce_ms / 1000):
                    continue
                self._last_events[key] = now

            for callback in self._callbacks:
                try:
                    callback(event)
                except Exception as e:
                    logger.exception(f"Callback error: {e}")

    def _scan_changes(self) -> List[FileEvent]:
        """Scan for file changes (cross-platform)."""
        events: List[FileEvent] = []

        for path in self.paths:
            if not os.path.exists(path):
                continue

            try:
                for root, dirs, files in os.walk(path, followlinks=False):
                    dirs[:] = [
                        d
                        for d in dirs
                        if not self._should_ignore(os.path.join(root, d))
                    ]

                    for filename in files:
                        if self._should_ignore(filename):
                            continue

                        file_path = os.path.join(root, filename)

                        if self._should_ignore(file_path):
                            continue

                        current_hash = self._compute_file_hash(file_path)

                        if current_hash is None:
                            continue

                        old_hash = self._file_hashes.get(file_path)

                        if old_hash is None:
                            event_type = "created"
                        elif old_hash != current_hash:
                            event_type = "modified"
                        else:
                            continue

                        self._file_hashes[file_path] = current_hash

                        events.append(
                            FileEvent(
                                event_type=event_type,
                                file_path=file_path,
                                file_hash=current_hash,
                            )
                        )
            except Exception as e:
                logger.error(f"Error scanning {path}: {e}")

        return events

    def _should_ignore(self, path: str) -> bool:
        """Check if a path should be ignored."""
        path_obj = Path(path)
        name = path_obj.name

        if name.startswith("."):
            return True

        for pattern in self.ignore_patterns:
            if pattern.startswith("*"):
                if name.endswith(pattern[1:]):
                    return True
            elif pattern in path:
                return True

        return False


class IncrementalIndexer:
    """Incrementally index files as they change with proper error handling."""

    def __init__(
        self,
        indexer,
        embedder,
        faiss_index,
        watch_paths: Set[str],
        debounce_ms: int = 500,
    ):
        """Initialize the incremental indexer.

        Args:
            indexer: RepositoryIndexer instance
            embedder: CodeEmbedder instance
            faiss_index: FaissIndex instance
            watch_paths: Paths to watch for changes
            debounce_ms: Debounce time for file changes
        """
        self.indexer = indexer
        self.embedder = embedder
        self.faiss_index = faiss_index
        self.watcher = FileWatcher(watch_paths, debounce_ms=debounce_ms)
        self.pending_files: Set[str] = set()
        self._lock = threading.Lock()
        self._running = False

    def start(self):
        """Start incremental indexing."""
        self._running = True

        def on_change(event: FileEvent):
            if self._running:
                with self._lock:
                    self.pending_files.add(event.file_path)
                logger.debug(f"File changed: {event.file_path}")

        self.watcher.add_callback(on_change)
        self.watcher.start()

        threading.Thread(
            target=self._process_loop, daemon=True, name="IncrementalIndexer"
        ).start()

        logger.info("Incremental indexer started")

    def stop(self):
        """Stop incremental indexing."""
        self._running = False
        self.watcher.stop()
        logger.info("Incremental indexer stopped")

    def _process_loop(self):
        """Process pending file changes."""
        while self._running:
            try:
                with self._lock:
                    files = list(self.pending_files)
                    self.pending_files.clear()

                if files:
                    self._process_files(files)

                time.sleep(0.5)
            except Exception as e:
                logger.exception(f"Error in process loop: {e}")
                time.sleep(1.0)

    def _process_files(self, file_paths: List[str]):
        """Process a batch of changed files."""
        for file_path in file_paths:
            try:
                self._reindex_file(file_path)
            except Exception as e:
                logger.error(f"Failed to reindex {file_path}: {e}")

    def _reindex_file(self, file_path: str):
        """Reindex a single file."""
        language = self.indexer.get_language(file_path)
        if not language:
            return

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                code = f.read()
        except Exception as e:
            logger.debug(f"Failed to read file {file_path}: {e}")
            return

        chunks = self.indexer.extract_code_chunks(file_path, code)

        if not chunks:
            return

        logger.debug(f"Reindexing {file_path} ({len(chunks)} chunks)")

        try:
            if self.embedder and self.faiss_index:
                code_texts = [chunk.code for chunk in chunks]
                embeddings = self.embedder.embed(code_texts)

                metadata = [
                    {
                        "file_path": chunk.file_path,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "code": chunk.code,
                        "language": chunk.language,
                    }
                    for chunk in chunks
                ]

                self.faiss_index.add(embeddings, metadata)

                logger.info(f"Reindexed {file_path}: {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Failed to add embeddings for {file_path}: {e}")

    def get_pending_count(self) -> int:
        """Get the number of pending files."""
        with self._lock:
            return len(self.pending_files)

    def get_status(self) -> Dict[str, Any]:
        """Get the status of the incremental indexer."""
        return {
            "running": self._running,
            "pending_files": self.get_pending_count(),
            "watch_paths": list(self.watcher.paths),
        }
